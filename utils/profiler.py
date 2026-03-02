"""
profiler_utils.py

Purpose
-------
This module provides two small, *optional* performance/diagnostics helpers:

1) torch_profiler_ctx(...)
   - A context manager that wraps `torch.profiler.profile` and emits a trace
     (Chrome/TensorBoard-compatible) **only when explicitly enabled**.
   - This keeps the main simulation/training code clean, while still making it
     easy to turn profiling on/off without modifying code.

2) nvidia_smi_summary()
   - A tiny utility that tries to call `nvidia-smi` (if available) and returns
     a one-line GPU utilization / memory / power summary.
   - Useful for logging periodic health checks in long-running experiments.

Design Principles
-----------------
- Opt-in behavior: profiling is OFF by default to avoid slowing experiments.
- Zero dependency on nvidia-smi: if it does not exist, we degrade gracefully.
- Minimal overhead when disabled: return quickly and do not import profiler.
"""

from __future__ import annotations  # Postpones evaluation of type hints (PEP 563 / PEP 649 era).
                                    # Practical benefit: faster imports, fewer circular import issues,
                                    # and you can use forward references in annotations cleanly.

from contextlib import contextmanager  # Lets us write a generator-based context manager using @contextmanager.
from typing import Optional            # Optional[T] means "T or None" in type hints.

import os, shutil, subprocess          # os: environment variables + filesystem
                                      # shutil: locating executables (shutil.which)
                                      # subprocess: running external commands safely

import torch                           # PyTorch: needed for torch.cuda.is_available and torch.profiler usage


_LEGACY_PROFILER_ENV_WARNED = False

def profiler_enabled() -> bool:
    """
    Decide whether profiling is enabled.

    Why an environment variable?
    ----------------------------
    In ML/Simulation projects, it's common to run the same code in many contexts:
    local dev, CI, remote machines, SLURM jobs, Docker containers, etc.

    If you tie profiling to code changes, you risk:
    - accidentally committing profiling instrumentation
    - changing runtime behavior between runs
    - forgetting to turn it off (profiling adds overhead)

    Using an env var makes profiling a runtime switch:
        FWS_TORCH_PROFILER=1 python main.py

    Backward compatibility:
    - Legacy profiler toggle via FWS_PROFILE=1 is still supported temporarily.
    - This collides with config.py profile-name semantics (FWS_PROFILE=debug/train_fast/...),
      so a deprecation warning is emitted when legacy boolean-style values are used.

    Accepted values:
    - "1", "true", "True" => enabled
    - everything else     => disabled

    Returns
    -------
    bool
        True if profiling should be activated; otherwise False.
    """
    global _LEGACY_PROFILER_ENV_WARNED

    truthy = {"1", "true", "yes", "y", "on", "t"}
    falsey = {"0", "false", "no", "n", "off", "f"}

    # Preferred, non-colliding env key.
    raw_new = os.getenv("FWS_TORCH_PROFILER", None)
    if raw_new is not None:
        return raw_new.strip().lower() in truthy

    # Backward-compatible fallback: only treat FWS_PROFILE as profiler toggle if it
    # looks like a boolean-ish value; otherwise leave it to config profile semantics.
    raw_legacy = os.getenv("FWS_PROFILE", None)
    if raw_legacy is None:
        return False

    legacy_norm = raw_legacy.strip().lower()
    if legacy_norm in truthy:
        if not _LEGACY_PROFILER_ENV_WARNED:
            print(
                "[profiler][WARN] Legacy profiler toggle via FWS_PROFILE is deprecated "
                "and collides with config profile selection. Use FWS_TORCH_PROFILER=1 instead."
            )
            _LEGACY_PROFILER_ENV_WARNED = True
        return True
    if legacy_norm in falsey:
        return False

    return False


@contextmanager
def torch_profiler_ctx(activity_cuda: bool = True, out_dir: str = "prof"):
    """
    Context manager: runs a minimal torch.profiler session when enabled.

    Usage Pattern
    -------------
    The typical call site looks like:

        with torch_profiler_ctx(activity_cuda=True, out_dir="prof") as prof:
            # ... your training/simulation loop ...
            # If prof is not None, you may call prof.step() each iteration.

    Important behavior:
    -------------------
    - If profiling is disabled, this yields None and returns immediately.
      That means calling code does not need to branch:
        `prof` is either a profiler object or None.

    Output
    ------
    When enabled, it will produce trace files via:
        torch.profiler.tensorboard_trace_handler(out_dir)

    These traces can be viewed with:
        tensorboard --logdir=prof

    Parameters
    ----------
    activity_cuda : bool
        Whether to include CUDA profiling activity (only if CUDA is available).
        If CUDA is not available, it silently falls back to CPU-only activities.

    out_dir : str
        Folder where profiler traces will be written.

    Notes on profiling schedule (performance + correctness)
    -------------------------------------------------------
    PyTorch profiler can be expensive. A schedule allows "sampling windows"
    that reduce overhead and avoid capturing startup noise.

    schedule(wait=2, warmup=2, active=6, repeat=1) means:
    - wait  : ignore first 2 steps (no recording)
    - warmup: next 2 steps (prepare/collect but typically not saved as final)
    - active: next 6 steps (recorded)
    - repeat: do this pattern once

    This is a practical choice for many loops:
    you skip initial jitter (memory allocation, kernel caching, graph warmup),
    then capture a representative steady-state segment.

    Returns (via yield)
    -------------------
    - When disabled: yields None
    - When enabled : yields a torch.profiler.profile object
    """
    # Fast path: if profiling is disabled, behave like a no-op context manager.
    # `yield None` allows callers to keep a uniform `with ... as prof:` structure.
    if not profiler_enabled():
        yield None
        return

    # We wrap profiling in a try/finally to ensure we always exit cleanly.
    # Even if exceptions occur in the profiled block, the context manager is exited.
    try:
        # Import torch.profiler lazily so that:
        # - we do not pay import costs when profiling is off
        # - environments without profiler support fail only when profiling is enabled
        from torch.profiler import profile, ProfilerActivity

        # We always include CPU profiling.
        acts = [ProfilerActivity.CPU]

        # If the caller wants CUDA activity AND CUDA is actually available,
        # we include CUDA profiling. Otherwise, we keep it CPU-only.
        # This avoids crashes on CPU-only machines.
        if activity_cuda and torch.cuda.is_available():
            acts.append(ProfilerActivity.CUDA)

        # Ensure output directory exists.
        # exist_ok=True prevents errors if the directory is already present.
        os.makedirs(out_dir, exist_ok=True)

        # Enter the profiler context.
        # The `with profile(...) as prof:` produces a profiler object that can be stepped.
        with profile(
            activities=acts,

            # Scheduling reduces overhead and focuses on a window of interest.
            schedule=torch.profiler.schedule(wait=2, warmup=2, active=6, repeat=1),

            # Called when a trace is ready (end of an "active" window).
            # tensorboard_trace_handler writes the trace files into out_dir.
            on_trace_ready=torch.profiler.tensorboard_trace_handler(out_dir),

            # record_shapes=False:
            # - If True, records tensor shapes for ops, increasing overhead and trace size.
            # - False is faster; enable if you specifically need shape-level diagnosis.
            record_shapes=False,

            # with_stack=False:
            # - If True, attempts to collect Python stack traces for ops (very expensive).
            # - Keep False for minimal overhead.
            with_stack=False,

            # with_flops=False:
            # - If True, estimates FLOPs for some ops; adds overhead.
            # - Keep False for minimal profiling footprint.
            with_flops=False,
        ) as prof:
            # Yield profiler to the caller. Caller may call prof.step() per iteration.
            yield prof

    finally:
        # `finally: pass` is intentionally empty.
        # Explanation:
        # - The `with profile(...)` block handles flushing/closing resources.
        # - Having a finally block keeps the structure ready for future cleanup hooks
        #   (e.g., logging "profiling finished", extra file management, etc.)
        pass


def nvidia_smi_summary() -> Optional[str]:
    """
    Query a short GPU status line via `nvidia-smi`, if available.

    What this does
    --------------
    - Checks if the `nvidia-smi` executable exists in PATH.
    - If it exists, runs a very short query returning:
        utilization.gpu, memory.used, memory.total, power.draw
      using CSV format without headers/units.
    - Returns only the first line (GPU 0) if multiple GPUs are present.

    Why it returns Optional[str]
    ----------------------------
    This function can fail for legitimate reasons:
    - system has no NVIDIA GPU
    - NVIDIA drivers not installed
    - `nvidia-smi` not in PATH
    - permission issues / runtime error
    - command timed out

    In all such cases, returning None is a clean, non-crashing behavior.
    It lets calling code do:

        line = nvidia_smi_summary()
        if line is not None:
            print("GPU:", line)

    Returns
    -------
    Optional[str]
        - A single CSV line (string) if successful, else None.

    Example returned string (no units, depends on GPU/driver):
        "12, 4329, 6144, 27.80"
    """
    # Locate `nvidia-smi` on the system PATH.
    # shutil.which returns the absolute path to the executable if found, else None.
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None

    try:
        # subprocess.check_output runs the command and returns stdout as bytes.
        # Key safety/performance choices:
        # - stderr redirected to DEVNULL: avoid noisy logs if nvidia-smi complains
        # - timeout=1.0: prevents hanging (important for long-running loops/logging)
        out = subprocess.check_output(
            [
                exe,
                "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            timeout=1.0,
        ).decode("utf-8", errors="ignore").strip()

        # Multiple GPUs may produce multiple lines, one per GPU.
        # We return the first GPU line to keep the output single-line and log-friendly.
        return out.splitlines()[0] if out else None

    except Exception:
        # Any failure returns None: do not crash training/simulation due to diagnostics.
        return None