from __future__ import annotations
from typing import Optional
import torch
import numpy as np

import config

# Grid channels:
#   0: occupancy (0 empty, 1 wall, 2 red, 3 blue)
#   1: hp        (0..MAX_HP)
#   2: agent_id  (-1 if empty)

def _generate_64_directions() -> torch.Tensor:
    """Generates 64 unique direction vectors, evenly spaced around a circle."""
    angles = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    dx = np.cos(angles)
    dy = np.sin(angles)
    
    # Create the direction vectors as floats, without rounding or making them unique,
    # as this preserves the 64 distinct directions.
    final_dirs = np.stack([dx, dy], axis=1)
        
    return torch.tensor(final_dirs, dtype=torch.float32)

DIRS64 = _generate_64_directions()

# One-hot types per first hit:
# 0 none, 1 wall, 2 red-soldier, 3 red-archer, 4 blue-soldier, 5 blue-archer  -> 6 classes
_TYPE_CLASSES = 6

@torch.no_grad()
def raycast64_firsthit(
    pos_xy: torch.Tensor,               # (N,2) long
    grid: torch.Tensor,                 # (3,H,W)
    unit_map: torch.Tensor,             # (H,W) int32 ∈{-1,1,2}
    max_steps_each: Optional[torch.Tensor] = None,  # (N,) long — per-agent vision
) -> torch.Tensor:
    """
    First-hit ray features with fixed 8 dims per ray for 64 directions.
      [ onehot6(none,wall,red-sold,red-arch,blue-sold,blue-arch), dist_norm, hp_norm ]
    Total = 64 rays * 8 dims = 512 per agent.
    """
    device = grid.device
    dtype = getattr(config, "TORCH_DTYPE", torch.float32)

    pos_xy = pos_xy.to(dtype=torch.long, device=device)
    N = int(pos_xy.size(0))
    H, W = int(grid.size(1)), int(grid.size(2))

    # Global cap & per-agent steps
    R_global = int(getattr(config, "RAYCAST_MAX_STEPS", 10))
    if max_steps_each is None:
        max_steps_each = torch.full((N,), R_global, device=device, dtype=torch.long)
    else:
        max_steps_each = torch.clamp(max_steps_each.to(device=device, dtype=torch.long), 0, R_global)

    # Prepare coordinates for all steps up to global cap
    dirs = DIRS64.to(device).view(1, 64, 2)                  # (1,64,2)
    base = pos_xy.view(N, 1, 1, 2).float()                   # (N,1,1,2)
    steps = torch.arange(1, R_global + 1, device=device, dtype=torch.float32).view(1, 1, R_global, 1)  # (1,1,S,1)
    
    # Calculate ray coordinates using float arithmetic and then cast to long for indexing
    coords_float = base + dirs.view(1, 64, 1, 2) * steps
    coords = coords_float.long() # (N,64,S,2)

    x = coords[..., 0].clamp_(0, W - 1)                      # (N,64,S)
    y = coords[..., 1].clamp_(0, H - 1)                      # (N,64,S)

    step_ids = torch.arange(1, R_global + 1, device=device, dtype=torch.long).view(1, 1, R_global)
    active = step_ids <= max_steps_each.view(N, 1, 1)        # (N,1,S)

    occ = grid[0][y, x]                                      # (N,64,S)
    hp  = grid[1][y, x]                                      # (N,64,S)
    
    is_wall = (occ == 1) & active
    has_agent = (grid[2][y, x] >= 0) & active

    idx_wall = torch.where(is_wall.any(dim=-1), is_wall.to(torch.float32).argmax(dim=-1), -1)
    idx_agent = torch.where(has_agent.any(dim=-1), has_agent.to(torch.float32).argmax(dim=-1), -1)

    first_kind = torch.full((N, 64), 0, dtype=torch.int64, device=device)
    first_idx  = torch.full((N, 64), -1, dtype=torch.long, device=device)

    both_hit = (idx_wall >= 0) & (idx_agent >= 0)
    only_wall = (idx_wall >= 0) & ~both_hit
    only_agent = (idx_agent >= 0) & ~both_hit

    if both_hit.any():
        earlier_is_wall = (idx_wall <= idx_agent)
        first_idx[both_hit] = torch.where(earlier_is_wall, idx_wall, idx_agent)[both_hit]
        first_kind[both_hit & earlier_is_wall] = 1
        first_kind[both_hit & ~earlier_is_wall] = -2 # Temp code for agent

    if only_wall.any():
        first_idx[only_wall] = idx_wall[only_wall]
        first_kind[only_wall] = 1

    if only_agent.any():
        first_idx[only_agent] = idx_agent[only_agent]
        first_kind[only_agent] = -2 # Temp code for agent

    agent_mask = (first_kind == -2)
    if agent_mask.any():
        gather_idx = first_idx.clamp_min(0).unsqueeze(-1)
        gather_y = torch.gather(y, 2, gather_idx).squeeze(-1)
        gather_x = torch.gather(x, 2, gather_idx).squeeze(-1)
        t = grid[0][gather_y, gather_x].to(torch.int32)
        u = unit_map[gather_y, gather_x].to(torch.int32)
        code = torch.zeros_like(t, dtype=torch.int64)
        code[(t == 2) & (u == 1)] = 2
        code[(t == 2) & (u == 2)] = 3
        code[(t == 3) & (u == 1)] = 4
        code[(t == 3) & (u == 2)] = 5
        first_kind[agent_mask] = code[agent_mask]

    den = max_steps_each.clamp_min(1).to(torch.float32).view(N, 1)
    dist_idx = first_idx.to(torch.float32) + 1.0
    valid = (first_idx >= 0).to(torch.float32)
    dist_norm = (dist_idx / den) * valid

    hp_first = torch.gather(hp, 2, first_idx.clamp_min(0).unsqueeze(-1)).squeeze(-1) * valid
    onehot = torch.zeros((N, 64, _TYPE_CLASSES), dtype=dtype, device=device)
    idx_valid = first_kind.clamp(min=0, max=_TYPE_CLASSES - 1)
    onehot.scatter_(2, idx_valid.unsqueeze(-1), 1.0)

    max_hp = float(getattr(config, "MAX_HP", 1.0)) or 1.0
    hp_norm = (hp_first / max_hp).clamp(0.0, 1.0).to(dtype)
    dist_norm = dist_norm.to(dtype)

    feat = torch.cat([onehot, dist_norm.unsqueeze(-1), hp_norm.unsqueeze(-1)], dim=-1)
    return feat.reshape(N, 64 * 8)
