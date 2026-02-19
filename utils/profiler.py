from __future__ import annotations
from contextlib import contextmanager
from typing import Optional
import os, shutil, subprocess

import torch

def profiler_enabled() -> bool:
    # opt-in via env var or config flag you can pass down
    return os.getenv("FWS_PROFILE", "0") in {"1", "true", "True"}

@contextmanager
def torch_profiler_ctx(activity_cuda: bool = True, out_dir: str = "prof"):
    """
    Minimal torch.profiler wrapper; creates a chrome trace in out_dir if enabled.
    """
    if not profiler_enabled():
        yield None
        return
    try:
        from torch.profiler import profile, ProfilerActivity
        acts = [ProfilerActivity.CPU]
        if activity_cuda and torch.cuda.is_available():
            acts.append(ProfilerActivity.CUDA)
        os.makedirs(out_dir, exist_ok=True)
        with profile(
            activities=acts,
            schedule=torch.profiler.schedule(wait=2, warmup=2, active=6, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(out_dir),
            record_shapes=False,
            with_stack=False,
            with_flops=False,
        ) as prof:
            yield prof
    finally:
        pass

def nvidia_smi_summary() -> Optional[str]:
    """
    Returns a one-line nvidia-smi util/mem summary if nvidia-smi exists; else None.
    """
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    try:
        out = subprocess.check_output(
            [exe, "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
            timeout=1.0,
        ).decode("utf-8", errors="ignore").strip()
        # Possibly multiple GPUs: return first line
        return out.splitlines()[0] if out else None
    except Exception:
        return None
