from __future__ import annotations
from typing import List, Tuple, Optional, Any
import torch
import torch.nn as nn
import config

try:
    from torch.func import functional_call, vmap, stack_module_state
except Exception:
    functional_call = None
    vmap = None
    stack_module_state = None


class _DistWrap:
    """Lightweight container to mimic a torch.distributions object with logits."""
    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits


_VMAP_WARNED: bool = False

def _is_torchscript_module(m: nn.Module) -> bool:
    return isinstance(m, torch.jit.ScriptModule) or isinstance(m, torch.jit.RecursiveScriptModule)

def _maybe_debug(msg: str) -> None:
    if bool(getattr(config, "VMAP_DEBUG", False)):
        print(msg)

def _maybe_warn_once(msg: str) -> None:
    global _VMAP_WARNED
    if _VMAP_WARNED:
        return
    _VMAP_WARNED = True
    _maybe_debug(msg)


@torch.no_grad()
def _ensemble_forward_loop(models: List[nn.Module], obs: torch.Tensor) -> Tuple[_DistWrap, torch.Tensor]:
    """
    Original safe loop implementation (kept as canonical fallback).
    """
    device = obs.device
    K = int(obs.size(0)) if obs.dim() > 0 else 0
    if K == 0:
        return _DistWrap(logits=torch.empty((0, 0), device=device)), torch.empty((0,), device=device)

    logits_out: List[torch.Tensor] = []
    values_out: List[torch.Tensor] = []
    for i, model in enumerate(models):
        o = obs[i].unsqueeze(0)  # (1,F)
        out = model(o)
        if not (isinstance(out, tuple) and len(out) == 2):
            raise RuntimeError("Brain.forward must return (logits_or_dist, value)")
        head, val = out
        logits = head.logits if hasattr(head, "logits") else head  # (1,A)
        v = val.view(-1)  # (1,)
        logits_out.append(logits)
        values_out.append(v)

    logits_cat = torch.cat(logits_out, dim=0)  # (K,A)
    values_cat = torch.cat(values_out, dim=0)  # (K,)
    if values_cat.dim() == 0:
        values_cat = values_cat.unsqueeze(0)
    # Invariants
    if logits_cat.dim() != 2 or logits_cat.size(0) != K:
        raise RuntimeError(f"ensemble_forward loop: logits shape invalid: got {tuple(logits_cat.shape)}, K={K}")
    if values_cat.dim() != 1 or values_cat.size(0) != K:
        raise RuntimeError(f"ensemble_forward loop: values shape invalid: got {tuple(values_cat.shape)}, K={K}")
    return _DistWrap(logits=logits_cat), values_cat


@torch.no_grad()
def _ensemble_forward_vmap(models: List[nn.Module], obs: torch.Tensor) -> Tuple[_DistWrap, torch.Tensor]:
    """
    vmap-based inference across *independent* parameter sets.
    - NO parameter sharing (each agent has its own weights).
    - NO optimizer sharing (inference-only path).
    """
    if functional_call is None or vmap is None or stack_module_state is None:
        raise RuntimeError("torch.func is not available in this PyTorch build")

    K = int(obs.size(0))
    if K == 0:
        return _DistWrap(logits=torch.empty((0, 0), device=obs.device)), torch.empty((0,), device=obs.device)

    # For safety, avoid TorchScript modules
    if any(_is_torchscript_module(m) for m in models):
        raise RuntimeError("TorchScript module in bucket (vmap disabled)")

    # stack_module_state expects identical module structure across the list.
    base = models[0]
    # Create batched params/buffers where leading dim is K
    params_batched, buffers_batched = stack_module_state(models)

    # Ensure obs has a batch dimension aligned with K
    x = obs
    if x.dim() != 2 or x.size(0) != K:
        raise RuntimeError(f"vmap: obs must be (K,F). got {tuple(x.shape)} expected K={K}")

    # Define per-model forward: takes (params_i, buffers_i, x_i) and returns (logits_i, value_i)
    def _f(params_i: Any, buffers_i: Any, x_i: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # model expects (1,F)
        out = functional_call(base, (params_i, buffers_i), (x_i.unsqueeze(0),))
        if not (isinstance(out, tuple) and len(out) == 2):
            raise RuntimeError("Brain.forward must return (logits_or_dist, value)")
        head, val = out
        logits = head.logits if hasattr(head, "logits") else head  # (1,A)
        logits = logits.squeeze(0)  # (A,)
        # Return scalar value (so vmap stacks to (K,))
        v = val.view(-1)[0]
        return logits, v

    logits_KA, values_K = vmap(_f, in_dims=(0, 0, 0))(params_batched, buffers_batched, x)

    # Invariants
    if logits_KA.dim() != 2 or logits_KA.size(0) != K:
        raise RuntimeError(f"vmap: logits shape invalid: got {tuple(logits_KA.shape)}, K={K}")
    if values_K.dim() != 1 or values_K.size(0) != K:
        raise RuntimeError(f"vmap: values shape invalid: got {tuple(values_K.shape)}, K={K}")

    return _DistWrap(logits=logits_KA), values_K


@torch.no_grad()
def ensemble_forward(models: List[nn.Module], obs: torch.Tensor) -> Tuple[_DistWrap, torch.Tensor]:
    """
    Fuses per-agent models for a bucket into one batched tensor of outputs.
    Args:
      models: list of nn.Module, length K
      obs:    (K, F) observation batch aligned with models ordering
    Returns:
      - dist-like object with .logits -> (K, A)
      - values tensor -> (K,)    (NEVER 0-dim)
    Contract:
      Each model.forward(x: (1,F)) -> (logits: (1,A)) or (dist_with_logits, value)
    """
    K = int(obs.size(0)) if obs.dim() > 0 else 0
    if K == 0:
        return _DistWrap(logits=torch.empty((0, 0), device=obs.device)), torch.empty((0,), device=obs.device)

    use_vmap = bool(getattr(config, "USE_VMAP", False))
    min_bucket = int(getattr(config, "VMAP_MIN_BUCKET", 8))

    # vmap path is inference-only; this function is already @no_grad
    if use_vmap and K >= min_bucket:
        # Quick structural checks before attempting vmap:
        # - requires torch.func
        # - avoid TorchScript modules
        # - bucket should be homogeneous architecture (already intended by build_buckets)
        if functional_call is None or vmap is None or stack_module_state is None:
            _maybe_warn_once("[vmap] torch.func not available; falling back to loop")
        elif any(_is_torchscript_module(m) for m in models):
            _maybe_debug("[vmap] TorchScript module detected; falling back to loop")
        else:
            try:
                return _ensemble_forward_vmap(models, obs)
            except Exception as e:
                _maybe_debug(f"[vmap] vmap path failed ({type(e).__name__}: {e}); falling back to loop")

    return _ensemble_forward_loop(models, obs)