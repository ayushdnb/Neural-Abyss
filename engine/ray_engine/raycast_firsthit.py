from __future__ import annotations
from typing import Optional
import torch

import config

# Grid channels:
#   0: occupancy (0 empty, 1 wall, 2 red, 3 blue)
#   1: hp        (0..MAX_HP)
#   2: agent_id  (-1 if empty)

# 8 directions (dx, dy): N, NE, E, SE, S, SW, W, NW (consistent with move_mask)
DIRS8 = torch.tensor([
    [ 0, -1],
    [ 1, -1],
    [ 1,  0],
    [ 1,  1],
    [ 0,  1],
    [-1,  1],
    [-1,  0],
    [-1, -1],
], dtype=torch.long)

# One-hot types per first hit:
# 0 none, 1 wall, 2 red-soldier, 3 red-archer, 4 blue-soldier, 5 blue-archer  -> 6 classes
_TYPE_CLASSES = 6


@torch.no_grad()
def build_unit_map(agent_data: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """
    Build an HxW int32 map of unit types from registry tensor and grid agent ids.
    -1 where empty/no agent; else 1 (soldier) or 2 (archer).
    """
    H, W = int(grid.size(1)), int(grid.size(2))
    unit_map = torch.full((H, W), -1, dtype=torch.int32, device=grid.device)

    ids = grid[2].to(torch.long)  # (H,W) -1 if empty
    has_agent = ids >= 0
    if not has_agent.any():
        return unit_map

    # Gather unit types by agent id and scatter into map
    # agent_data shape: (N, features); COL_UNIT assumed to be float with values {1.0,2.0}
    from ..agent_registry import COL_UNIT  # local import to avoid circulars
    units_by_id = agent_data[:, COL_UNIT].to(torch.int32)  # (N,)
    picked = torch.where(
        has_agent,
        units_by_id[ids.clamp_min(0)],
        torch.tensor(-1, device=grid.device, dtype=torch.int32),
    )
    unit_map.copy_(picked)
    return unit_map


@torch.no_grad()
def raycast8_firsthit(
    pos_xy: torch.Tensor,               # (N,2) long
    grid: torch.Tensor,                 # (3,H,W)
    unit_map: torch.Tensor,             # (H,W) int32 ∈{-1,1,2}
    max_steps_each: Optional[torch.Tensor] = None,  # (N,) long — per-agent vision; optional
) -> torch.Tensor:
    """
    First-hit ray features with fixed 8 dims per ray:
      [ onehot6(none,wall,red-sold,red-arch,blue-sold,blue-arch), dist_norm, hp_norm ]
    Total = 8 rays * 8 dims = 64 per agent.

    New: accepts per-agent max range via `max_steps_each`. If None, uses
         `config.RAYCAST_MAX_STEPS` (or 10 if missing).
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
    dirs = DIRS8.to(device).view(1, 8, 2)                    # (1,8,2)
    base = pos_xy.view(N, 1, 1, 2)                           # (N,1,1,2)
    steps = torch.arange(1, R_global + 1, device=device, dtype=torch.long).view(1, 1, R_global, 1)  # (1,1,S,1)
    coords = base + dirs.view(1, 8, 1, 2) * steps            # (N,8,S,2)

    x = coords[..., 0].clamp_(0, W - 1)                      # (N,8,S)
    y = coords[..., 1].clamp_(0, H - 1)                      # (N,8,S)

    # Active mask per agent per step (agents cannot see beyond their own max range)
    step_ids = torch.arange(1, R_global + 1, device=device, dtype=torch.long).view(1, 1, R_global)
    active = step_ids <= max_steps_each.view(N, 1, 1)        # (N,1,S)

    occ = grid[0][y, x]                                      # (N,8,S)
    hp  = grid[1][y, x]                                      # (N,8,S)
    uid = unit_map[y, x]                                     # (N,8,S) ∈ {-1,1,2}

    # Determine first hit per (N,8)
    # Case order of precedence: wall, agent, else none
    is_wall = (occ == 1) & active                            # (N,8,S) but active is (N,1,S) -> broadcast
    has_agent = (grid[2][y, x] >= 0) & active

    # First wall step index
    any_wall = is_wall.any(dim=-1)
    idx_wall = torch.where(
        any_wall,
        is_wall.to(torch.float32).argmax(dim=-1),
        torch.full(is_wall.shape[:-1], -1, device=device, dtype=torch.long),
    )  # (N,8)

    # First agent step index
    any_agent = has_agent.any(dim=-1)
    idx_agent = torch.where(
        any_agent,
        has_agent.to(torch.float32).argmax(dim=-1),
        torch.full(has_agent.shape[:-1], -1, device=device, dtype=torch.long),
    )  # (N,8)

    # Resolve which occurs first (prefer smaller non-negative index)
    first_kind = torch.full((N, 8), 0, dtype=torch.int64, device=device)  # 0 none
    first_idx  = torch.full((N, 8), -1, dtype=torch.long, device=device)

    both_hit = (idx_wall >= 0) & (idx_agent >= 0)
    only_wall = (idx_wall >= 0) & (~(idx_agent >= 0))
    only_agent = (~(idx_wall >= 0)) & (idx_agent >= 0)

    if both_hit.any():
        earlier_is_wall = (idx_wall <= idx_agent)
        fi = torch.where(earlier_is_wall, idx_wall, idx_agent)
        first_idx[both_hit] = fi[both_hit]
        first_kind[both_hit & earlier_is_wall] = 1  # wall
        first_kind[both_hit & (~earlier_is_wall)] = -2  # agent (temp)

    if only_wall.any():
        first_idx[only_wall] = idx_wall[only_wall]
        first_kind[only_wall] = 1  # wall

    if only_agent.any():
        first_idx[only_agent] = idx_agent[only_agent]
        first_kind[only_agent] = -2  # agent (temp)

    # Build type codes for agent hits using team+unit
    agent_mask = (first_kind == -2)
    if agent_mask.any():
        gather_y = torch.gather(y, 2, first_idx.clamp_min(0).unsqueeze(-1)).squeeze(-1)  # (N,8)
        gather_x = torch.gather(x, 2, first_idx.clamp_min(0).unsqueeze(-1)).squeeze(-1)  # (N,8)
        t = grid[0][gather_y, gather_x].to(torch.int32)  # (N,8)
        u = unit_map[gather_y, gather_x].to(torch.int32) # (N,8)
        code = torch.full_like(t, 0, dtype=torch.int64)
        # red
        code[(t == 2) & (u == 1)] = 2
        code[(t == 2) & (u == 2)] = 3
        # blue
        code[(t == 3) & (u == 1)] = 4
        code[(t == 3) & (u == 2)] = 5
        first_kind[agent_mask] = code[agent_mask]

    # Distance normalized by each agent's own vision max
    den = max_steps_each.clamp_min(1).to(torch.float32).view(N, 1).expand(N, 8)
    dist_idx = first_idx.to(torch.float32) + 1.0  # steps start at 1
    valid = (first_idx >= 0).to(torch.float32)
    dist_norm = (dist_idx / den) * valid

    # Gather hp at first hit
    hp_first = torch.gather(hp, 2, first_idx.clamp_min(0).unsqueeze(-1)).squeeze(-1)  # (N,8)
    hp_first = hp_first * valid

    # One-hot over 6 classes
    onehot = torch.zeros((N, 8, _TYPE_CLASSES), dtype=dtype, device=device)
    idx_valid = first_kind.clamp(min=0, max=_TYPE_CLASSES - 1)
    onehot.scatter_(2, idx_valid.unsqueeze(-1), 1.0)

    # Normalize hp
    max_hp = float(getattr(config, "MAX_HP", 1.0))
    if max_hp <= 0:  # avoid div by zero
        max_hp = 1.0
    hp_norm = (hp_first / max_hp).clamp(0.0, 1.0).to(dtype)
    dist_norm = dist_norm.to(dtype)

    # Concatenate per ray: onehot(6) + dist_norm + hp_norm -> 8 dims
    feat = torch.cat([onehot, dist_norm.unsqueeze(-1), hp_norm.unsqueeze(-1)], dim=-1)  # (N,8,8)
    return feat.reshape(N, 8 * 8)
