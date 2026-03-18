"""Grouped forward execution for heterogeneous agent models."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Tuple, Optional, Any
# ^ Standard typing imports:
#   - List[T]: list of T
#   - Tuple[A, B]: a 2-item tuple
#   - Optional[T]: either T or None
#   - Any: intentionally unconstrained type

import torch
import torch.nn as nn
import config
# ^ Local project configuration module. This code reads feature flags from config, such as:
#   - USE_VMAP: enable/disable vmap path
#   - VMAP_MIN_BUCKET: minimum bucket size for vmap
#   - VMAP_DEBUG: print debug messages

# Optional imports from torch.func (PyTorch functional transforms).
# These APIs may not exist in older or minimal PyTorch builds, so we import them defensively.
# If the import fails, we set them to None and fall back to a safe Python loop.
try:
    from torch.func import functional_call, vmap, stack_module_state
except Exception:
    functional_call = None
    vmap = None
    stack_module_state = None


class _DistWrap:
    """
    Lightweight container to mimic a torch.distributions object with logits.

    Why this exists:
    - Some RL policy code expects a "distribution-like" object that exposes `.logits`.
    - This wrapper provides a minimal, compatible interface: only `.logits` is stored.
    - It avoids depending on a specific distribution class (e.g., Categorical).
    """
    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits


# Global state used to ensure we only emit the "warn once" message a single time.
_VMAP_WARNED: bool = False


# Cache for stacked torch.func module state used by the vmap inference path.
# Safety model:
# - Cache reuse is keyed by the ordered model identities and an explicit
#   fingerprint of every parameter/buffer tensor currently owned by those models.
# - The fingerprint includes each tensor object identity plus its in-place
#   mutation version counter, so optimizer.step() / load_state_dict() / buffer
#   updates naturally invalidate the cache without changing external code paths.
# - Brain replacement naturally misses because registry/respawn paths replace the
#   nn.Module object stored for a slot.
# - Entries retain strong references to the models they were built from so Python
#   cannot recycle an object id into a false cache hit while the entry is alive.
# - Size is bounded to avoid unbounded memory growth if bucket compositions churn.


@dataclass
class _StackedStateCacheEntry:
    models: Tuple[nn.Module, ...]
    fingerprints: Tuple[Tuple[Any, ...], ...]
    params_batched: Any
    buffers_batched: Any


_VMAP_STACK_CACHE: "OrderedDict[Tuple[Any, ...], _StackedStateCacheEntry]" = OrderedDict()


def _stack_cache_max_entries() -> int:
    """Return the configured max number of cached stacked-state entries.

    A value <= 0 disables the cache entirely.
    """
    try:
        return int(getattr(config, "VMAP_STACK_CACHE_MAX", 128))
    except Exception:
        return 128


def clear_stacked_vmap_cache() -> None:
    """Clear all cached stacked vmap parameter/buffer entries."""
    _VMAP_STACK_CACHE.clear()


def _evict_old_stack_cache_entries(max_entries: int) -> None:
    """Bound the number of cached bucket entries to avoid memory growth."""
    while len(_VMAP_STACK_CACHE) > max_entries:
        _VMAP_STACK_CACHE.popitem(last=False)


def _tensor_state_token(t: torch.Tensor) -> Tuple[Any, ...]:
    """Return a lightweight identity+freshness token for one tensor.

    The private tensor version counter increments on in-place mutations, which is
    exactly what optimizer.step() and state_dict restoration do to parameters and
    buffers in this codebase.
    """
    return (
        id(t),
        int(getattr(t, "_version", -1)),
        t.device.type,
        t.device.index,
        str(t.dtype),
        tuple(t.shape),
    )


def _model_state_fingerprint(model: nn.Module) -> Tuple[Any, ...]:
    """Build a deterministic fingerprint of one model's current tensor state."""
    parts: List[Tuple[Any, ...]] = []
    for p in model.parameters():
        parts.append(_tensor_state_token(p))
    for b in model.buffers():
        parts.append(_tensor_state_token(b))
    return tuple(parts)


def _make_stack_cache_key(models: List[nn.Module], device: torch.device) -> Tuple[Tuple[Any, ...], Tuple[Tuple[Any, ...], ...]]:
    """Build a cache key from ordered model identities, tensor freshness, and device."""
    fingerprints = tuple(_model_state_fingerprint(m) for m in models)
    key = (device.type, device.index, tuple((id(m), fp) for m, fp in zip(models, fingerprints)))
    return key, fingerprints


def _stacked_module_state_cached(models: List[nn.Module], device: torch.device) -> Tuple[Any, Any]:
    """Return stacked module state, reusing a safe cache entry when possible.

    Reuse is allowed only when:
    - the ordered model list is the exact same list of nn.Module objects, and
    - every parameter/buffer freshness token in every model matches.
    """
    max_entries = _stack_cache_max_entries()
    if max_entries <= 0:
        return stack_module_state(models)

    key, fingerprints = _make_stack_cache_key(models, device)
    entry = _VMAP_STACK_CACHE.get(key)
    if entry is not None:
        same_models = (
            len(entry.models) == len(models)
            and entry.fingerprints == fingerprints
            and all(cached is current for cached, current in zip(entry.models, models))
        )
        if same_models:
            _VMAP_STACK_CACHE.move_to_end(key)
            return entry.params_batched, entry.buffers_batched

        # Defensive cleanup: if identities/fingerprints mismatched despite a key
        # hit (for example after manual tampering), drop the stale entry.
        _VMAP_STACK_CACHE.pop(key, None)

    params_batched, buffers_batched = stack_module_state(models)
    _VMAP_STACK_CACHE[key] = _StackedStateCacheEntry(
        models=tuple(models),
        fingerprints=fingerprints,
        params_batched=params_batched,
        buffers_batched=buffers_batched,
    )
    _evict_old_stack_cache_entries(max_entries)
    return params_batched, buffers_batched

def _is_torchscript_module(m: nn.Module) -> bool:
    """
    Returns True if the given module is a TorchScript-compiled module.

    TorchScript modules (torch.jit.ScriptModule / RecursiveScriptModule) can have constraints
    that make certain torch.func operations unsafe or unsupported.
    """
    return isinstance(m, torch.jit.ScriptModule) or isinstance(m, torch.jit.RecursiveScriptModule)

def _maybe_debug(msg: str) -> None:
    """
    Debug-print helper gated by config.VMAP_DEBUG.

    This keeps the main control flow clean while still allowing opt-in diagnostics.
    """
    if bool(getattr(config, "VMAP_DEBUG", False)):
        print(msg)

def _maybe_warn_once(msg: str) -> None:
    """
    "Warn once" helper.

    This is used for vmap fallback messaging so logs do not get spammed every step.
    """
    global _VMAP_WARNED
    if _VMAP_WARNED:
        return
    _VMAP_WARNED = True
    _maybe_debug(msg)


@torch.no_grad()
def _ensemble_forward_loop(models: List[nn.Module], obs: torch.Tensor) -> Tuple[_DistWrap, torch.Tensor]:
    """
    Original safe loop implementation (kept as canonical fallback).

    Purpose:
    - We have K independent models (one per agent).
    - Each model receives its aligned observation obs[i].
    - We run forward passes one-by-one (Python loop), then concatenate outputs.

    Inputs:
    - models: list of nn.Module, length K
    - obs:    tensor shaped (K, F), where:
              K = number of agents in the bucket
              F = feature dimension of a single observation

    Outputs:
    - (_DistWrap with logits): logits shaped (K, A)
    - values tensor:           values shaped (K,)

    Notes:
    - This path is robust and does not require torch.func.
    - It is often slower than vectorized approaches, but it is a reliable baseline.
    """
    device = obs.device

    # Determine bucket size K in a safe way:
    # - If obs has at least one dimension, K = obs.size(0)
    # - If obs is unexpected (dim == 0), treat as empty.
    K = int(obs.size(0)) if obs.dim() > 0 else 0

    # Handle empty bucket:
    # Return empty logits and empty values on the same device.
    if K == 0:
        return _DistWrap(logits=torch.empty((0, 0), device=device)), torch.empty((0,), device=device)

    # Run the first model to determine output shapes/dtypes, then write all
    # subsequent outputs directly into preallocated tensors.
    first_out = models[0](obs[0].unsqueeze(0))
    if not (isinstance(first_out, tuple) and len(first_out) == 2):
        raise RuntimeError("Brain.forward must return (logits_or_dist, value)")

    first_head, first_val = first_out
    first_logits = first_head.logits if hasattr(first_head, "logits") else first_head  # (1,A)
    first_v = first_val.view(-1)  # (1,)

    if first_logits.dim() != 2 or first_logits.size(0) != 1:
        raise RuntimeError(f"ensemble_forward loop: per-model logits shape invalid: got {tuple(first_logits.shape)}")
    if first_v.numel() != 1:
        raise RuntimeError(f"ensemble_forward loop: per-model value shape invalid: got {tuple(first_v.shape)}")

    logits_cat = torch.empty(
        (K, int(first_logits.size(1))),
        device=first_logits.device,
        dtype=first_logits.dtype,
    )
    values_cat = torch.empty((K,), device=first_v.device, dtype=first_v.dtype)

    logits_cat[0].copy_(first_logits.squeeze(0))
    values_cat[0] = first_v[0]

    # Iterate over remaining models/agents in the bucket.
    for i, model in enumerate(models[1:], start=1):
        out = model(obs[i].unsqueeze(0))
        if not (isinstance(out, tuple) and len(out) == 2):
            raise RuntimeError("Brain.forward must return (logits_or_dist, value)")

        head, val = out
        logits = head.logits if hasattr(head, "logits") else head  # (1,A)
        v = val.view(-1)  # (1,)

        if logits.dim() != 2 or logits.size(0) != 1:
            raise RuntimeError(f"ensemble_forward loop: per-model logits shape invalid: got {tuple(logits.shape)}")
        if v.numel() != 1:
            raise RuntimeError(f"ensemble_forward loop: per-model value shape invalid: got {tuple(v.shape)}")

        logits_cat[i].copy_(logits.squeeze(0))
        values_cat[i] = v[0]

    # Ensure values never become 0-dim (scalar tensor).
    # This is defensive: some edge cases can cause shape collapsing.
    if values_cat.dim() == 0:
        values_cat = values_cat.unsqueeze(0)

    # Invariants (hard checks):
    # 1) logits must be rank-2 with leading dimension K.
    # 2) values must be rank-1 with length K.
    if logits_cat.dim() != 2 or logits_cat.size(0) != K:
        raise RuntimeError(f"ensemble_forward loop: logits shape invalid: got {tuple(logits_cat.shape)}, K={K}")
    if values_cat.dim() != 1 or values_cat.size(0) != K:
        raise RuntimeError(f"ensemble_forward loop: values shape invalid: got {tuple(values_cat.shape)}, K={K}")

    return _DistWrap(logits=logits_cat), values_cat


@torch.no_grad()
def _ensemble_forward_vmap(models: List[nn.Module], obs: torch.Tensor) -> Tuple[_DistWrap, torch.Tensor]:
    """
    vmap-based inference across *independent* parameter sets.

    Key idea:
    - We want to evaluate K models "in parallel" without writing a Python loop.
    - PyTorch's torch.func can treat a module as a pure function:
        f(params, buffers, x) -> y
    - If we "stack" K sets of params/buffers, we can vmap over them:
        vmap(f)(params_batched, buffers_batched, x_batched)

    Important constraints explicitly stated in the docstring:
    - NO parameter sharing: each agent has its own weights.
    - NO optimizer sharing: this path is inference-only (and function is @no_grad).

    Inputs:
    - models: list of nn.Module, length K, same architecture
    - obs:    tensor (K, F)

    Outputs:
    - logits (K, A)
    - values (K,)
    """
    # If torch.func features are not available, we cannot run this path.
    if functional_call is None or vmap is None or stack_module_state is None:
        raise RuntimeError("torch.func is not available in this PyTorch build")

    K = int(obs.size(0))

    # Empty bucket handling.
    if K == 0:
        return _DistWrap(logits=torch.empty((0, 0), device=obs.device)), torch.empty((0,), device=obs.device)

    # Safety: TorchScript modules may be incompatible with torch.func transforms.
    if any(_is_torchscript_module(m) for m in models):
        raise RuntimeError("TorchScript module in bucket (vmap disabled)")

    # stack_module_state requires identical module structure across the list.
    # We use the first model as the "base" definition for functional_call.
    base = models[0]

    # Stack parameters and buffers from each model:
    # - params_batched: pytree-like structure where each tensor has leading dim K
    # - buffers_batched: same idea for buffers (e.g., running stats in BatchNorm)
    # The expensive stack_module_state(models) reconstruction is cached as long as
    # the ordered model list and each model's parameter/buffer freshness tokens
    # remain unchanged.
    params_batched, buffers_batched = _stacked_module_state_cached(models, obs.device)

    # Ensure obs matches (K, F) exactly.
    x = obs
    if x.dim() != 2 or x.size(0) != K:
        raise RuntimeError(f"vmap: obs must be (K,F). got {tuple(x.shape)} expected K={K}")

    # Define a per-model forward function for vmap.
    # Inputs (per single model i):
    # - params_i: parameters of model i
    # - buffers_i: buffers of model i
    # - x_i: observation for model i (shape (F,))
    # Output:
    # - logits for model i, shape (A,)
    # - value for model i, scalar
    def _f(params_i: Any, buffers_i: Any, x_i: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # The base model expects input shape (batch, F), so add batch dimension: (1, F)
        out = functional_call(base, (params_i, buffers_i), (x_i.unsqueeze(0),))

        # Validate contract: must return (head, val)
        if not (isinstance(out, tuple) and len(out) == 2):
            raise RuntimeError("Brain.forward must return (logits_or_dist, value)")

        head, val = out

        # Support either a distribution-like head with `.logits` or direct logits tensor.
        logits = head.logits if hasattr(head, "logits") else head  # (1,A)

        # Remove batch dimension so per-item logits are (A,)
        logits = logits.squeeze(0)  # (A,)

        # Return scalar value so that vmap stacks values into shape (K,)
        v = val.view(-1)[0]
        return logits, v

    # Vectorize _f over leading dimension of (params, buffers, x).
    # in_dims=(0,0,0) means:
    # - take params_batched[i]
    # - take buffers_batched[i]
    # - take x[i]
    # and apply _f for each i in 0..K-1, stacking results.
    logits_KA, values_K = vmap(_f, in_dims=(0, 0, 0))(params_batched, buffers_batched, x)

    # Invariants: enforce output shapes.
    if logits_KA.dim() != 2 or logits_KA.size(0) != K:
        raise RuntimeError(f"vmap: logits shape invalid: got {tuple(logits_KA.shape)}, K={K}")
    if values_K.dim() != 1 or values_K.size(0) != K:
        raise RuntimeError(f"vmap: values shape invalid: got {tuple(values_K.shape)}, K={K}")

    return _DistWrap(logits=logits_KA), values_K


@torch.no_grad()
def ensemble_forward(models: List[nn.Module], obs: torch.Tensor) -> Tuple[_DistWrap, torch.Tensor]:
    """
    Fuses per-agent models for a bucket into one batched tensor of outputs.

    This function is the public entry point that chooses between:
    1) A safe Python loop (_ensemble_forward_loop), and
    2) A potentially faster torch.func + vmap path (_ensemble_forward_vmap).

    Args:
      models: list of nn.Module, length K
      obs:    (K, F) observation batch aligned with models ordering

    Returns:
      - dist-like object with .logits -> (K, A)
      - values tensor -> (K,)    (NEVER 0-dim)

    Contract (hard requirement for model.forward):
      Each model.forward(x: (1,F)) -> (logits: (1,A)) or (dist_with_logits, value)

    Design intent:
    - "Bucket" means a group of agents/models assumed to share the same architecture.
    - If bucket size is large enough, we attempt vmap for performance.
    - If anything is incompatible or fails, we fall back safely to the loop.
    """
    # Determine K robustly.
    K = int(obs.size(0)) if obs.dim() > 0 else 0

    # Empty bucket returns empty outputs.
    if K == 0:
        return _DistWrap(logits=torch.empty((0, 0), device=obs.device)), torch.empty((0,), device=obs.device)

    # Feature flags / thresholds from config:
    # - USE_VMAP: master switch
    # - VMAP_MIN_BUCKET: minimum K to justify vmap overhead
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
            # Attempt vmap path; if it fails for any reason, fall back to loop.
            # This is a reliability-first design: performance is optional, correctness is mandatory.
            try:
                return _ensemble_forward_vmap(models, obs)
            except Exception as e:
                _maybe_debug(f"[vmap] vmap path failed ({type(e).__name__}: {e}); falling back to loop")

    # Default and fallback: the canonical safe implementation.
    return _ensemble_forward_loop(models, obs)
