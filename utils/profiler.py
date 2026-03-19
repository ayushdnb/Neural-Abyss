"""Optional profiling and GPU-status helpers."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Optional

import os
import shutil
import subprocess

import torch


_TRUE_VALUES = {"1", "true", "yes", "y", "on", "t"}
_FALSE_VALUES = {"0", "false", "no", "n", "off", "f"}
_LEGACY_PROFILER_ENV_WARNED = False


def profiler_enabled() -> bool:
    """Return whether torch profiling is enabled."""
    global _LEGACY_PROFILER_ENV_WARNED

    raw = os.getenv("FWS_TORCH_PROFILER")
    if raw is not None:
        return raw.strip().lower() in _TRUE_VALUES

    legacy = os.getenv("FWS_PROFILE")
    if legacy is None:
        return False

    legacy_norm = legacy.strip().lower()
    if legacy_norm in _TRUE_VALUES:
        if not _LEGACY_PROFILER_ENV_WARNED:
            print(
                "[profiler][WARN] Legacy profiler toggle via FWS_PROFILE is deprecated. "
                "Use FWS_TORCH_PROFILER=1 instead."
            )
            _LEGACY_PROFILER_ENV_WARNED = True
        return True
    if legacy_norm in _FALSE_VALUES:
        return False
    return False


@contextmanager
def torch_profiler_ctx(
    activity_cuda: bool = True,
    out_dir: str = "prof",
) -> Iterator[Optional[object]]:
    """Yield a torch profiler when profiling is enabled, else ``None``."""
    if not profiler_enabled():
        yield None
        return

    from torch.profiler import ProfilerActivity, profile

    activities = [ProfilerActivity.CPU]
    if activity_cuda and torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    os.makedirs(out_dir, exist_ok=True)

    with profile(
        activities=activities,
        schedule=torch.profiler.schedule(wait=2, warmup=2, active=6, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(out_dir),
        record_shapes=False,
        with_stack=False,
        with_flops=False,
    ) as profiler:
        yield profiler


def nvidia_smi_summary() -> Optional[str]:
    """Return a one-line ``nvidia-smi`` summary for GPU 0 when available."""
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return None

    try:
        output = subprocess.check_output(
            [
                executable,
                "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            timeout=1.0,
        ).decode("utf-8", errors="ignore").strip()
    except Exception:
        return None

    return output.splitlines()[0] if output else None
