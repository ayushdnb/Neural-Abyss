from __future__ import annotations
from typing import Dict, Tuple
import torch
import config

# Cache index tensors per (device, name) to avoid per-step allocations.
_IDX_CACHE: Dict[Tuple[torch.device, str], torch.Tensor] = {}

def _idx(name: str, device: torch.device) -> torch.Tensor:
    """Get cached index tensor for semantic token by name."""
    key = (device, name)
    t = _IDX_CACHE.get(key)
    if t is not None:
        return t
    if name not in config.SEMANTIC_RICH_BASE_INDICES:
        raise KeyError(f"Unknown semantic token name: {name}")
    idx = torch.tensor(config.SEMANTIC_RICH_BASE_INDICES[name], dtype=torch.long, device=device)
    _IDX_CACHE[key] = idx
    return idx

def split_obs_flat(obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Split flat observation into three components:
      rays_flat  : (B, RAYS_FLAT_DIM)
      rich_base  : (B, RICH_BASE_DIM)
      instinct   : (B, INSTINCT_DIM)
    
    Performs strict shape checking and fails loudly on mismatch.
    """
    if obs.dim() != 2:
        raise RuntimeError(f"obs must be rank-2 (B,F). got shape={tuple(obs.shape)}")
    
    B, F = int(obs.shape[0]), int(obs.shape[1])
    if F != int(config.OBS_DIM):
        raise RuntimeError(f"obs_dim mismatch: got F={F}, expected config.OBS_DIM={int(config.OBS_DIM)}")

    rays_dim = int(config.RAYS_FLAT_DIM)
    rich_total = int(config.RICH_TOTAL_DIM)
    if rays_dim + rich_total != F:
        raise RuntimeError(f"layout mismatch: rays_dim({rays_dim}) + rich_total({rich_total}) != F({F})")

    # Split observation into rays and rich tail
    rays_flat = obs[:, :rays_dim]
    rich_tail = obs[:, rays_dim:]

    # Split rich tail into base and instinct
    base_dim = int(config.RICH_BASE_DIM)
    inst_dim = int(config.INSTINCT_DIM)
    if base_dim + inst_dim != int(rich_tail.shape[1]):
        raise RuntimeError(
            f"rich_tail mismatch: got {int(rich_tail.shape[1])}, expected {base_dim}+{inst_dim}"
        )

    rich_base = rich_tail[:, :base_dim]
    instinct = rich_tail[:, base_dim:base_dim + inst_dim]
    return rays_flat, rich_base, instinct

def build_semantic_tokens(
    rich_base: torch.Tensor,
    instinct: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Build semantic token tensors from rich_base and instinct components.
    
    Returns dictionary containing:
      own_context, world_context, zone_context, team_context, combat_context, instinct_context
    
    Each token tensor has shape (B, token_dim) and resides on same device as inputs.
    """
    if rich_base.dim() != 2:
        raise RuntimeError(f"rich_base must be (B,D). got {tuple(rich_base.shape)}")
    if instinct.dim() != 2:
        raise RuntimeError(f"instinct must be (B,4). got {tuple(instinct.shape)}")

    B = int(rich_base.shape[0])
    if int(rich_base.shape[1]) != int(config.RICH_BASE_DIM):
        raise RuntimeError(
            f"rich_base dim mismatch: got {int(rich_base.shape[1])}, expected {int(config.RICH_BASE_DIM)}"
        )
    if int(instinct.shape[0]) != B or int(instinct.shape[1]) != int(config.INSTINCT_DIM):
        raise RuntimeError(
            f"instinct shape mismatch: got {tuple(instinct.shape)}, expected ({B},{int(config.INSTINCT_DIM)})"
        )

    device = rich_base.device
    out: Dict[str, torch.Tensor] = {}

    # Extract each semantic token from rich_base using pre-defined indices
    for name in ("own_context", "world_context", "zone_context", "team_context", "combat_context"):
        idx = _idx(name, device)
        tok = torch.index_select(rich_base, dim=1, index=idx)
        out[name] = tok

    # Add instinct token directly
    out["instinct_context"] = instinct
    return out