"""Legacy 8-direction first-hit ray features."""

from __future__ import annotations

from typing import Optional

import torch

import config


DIRS8 = torch.tensor(
    [
        [0, -1],
        [1, -1],
        [1, 0],
        [1, 1],
        [0, 1],
        [-1, 1],
        [-1, 0],
        [-1, -1],
    ],
    dtype=torch.long,
)
_TYPE_CLASSES = 6


@torch.inference_mode()
def build_unit_map(agent_data: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """Return an ``(H, W)`` map of unit types for occupied cells."""
    from ..agent_registry import COL_UNIT

    height, width = int(grid.size(1)), int(grid.size(2))
    unit_map = torch.full((height, width), -1, dtype=torch.int32, device=grid.device)
    agent_ids = grid[2].to(torch.long)
    has_agent = agent_ids >= 0
    if not bool(has_agent.any().item()):
        return unit_map

    units_by_id = agent_data[:, COL_UNIT].to(torch.int32)
    unit_map[has_agent] = units_by_id[agent_ids[has_agent]]
    return unit_map


@torch.inference_mode()
def raycast8_firsthit(
    pos_xy: torch.Tensor,
    grid: torch.Tensor,
    unit_map: torch.Tensor,
    max_steps_each: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return ``(N, 64)`` first-hit features for the 8-direction raycaster."""
    device = grid.device
    dtype = getattr(config, "TORCH_DTYPE", torch.float32)

    pos_xy = pos_xy.to(dtype=torch.long, device=device)
    num_agents = int(pos_xy.size(0))
    height, width = int(grid.size(1)), int(grid.size(2))
    max_steps = int(getattr(config, "RAYCAST_MAX_STEPS", 10))

    if max_steps_each is None:
        max_steps_each = torch.full((num_agents,), max_steps, device=device, dtype=torch.long)
    else:
        max_steps_each = torch.clamp(
            max_steps_each.to(device=device, dtype=torch.long),
            min=0,
            max=max_steps,
        )

    dirs = DIRS8.to(device=device).view(1, 8, 1, 2)
    base = pos_xy.view(num_agents, 1, 1, 2)
    steps = torch.arange(1, max_steps + 1, device=device, dtype=torch.long).view(1, 1, max_steps, 1)
    coords = base + dirs * steps

    x = coords[..., 0].clamp_(0, width - 1)
    y = coords[..., 1].clamp_(0, height - 1)

    step_ids = torch.arange(1, max_steps + 1, device=device, dtype=torch.long).view(1, 1, max_steps)
    active = step_ids <= max_steps_each.view(num_agents, 1, 1)

    occupancy = grid[0][y, x]
    hp = grid[1][y, x]

    is_wall = (occupancy == 1.0) & active
    has_agent = (grid[2][y, x] >= 0) & active

    any_wall = is_wall.any(dim=-1)
    any_agent = has_agent.any(dim=-1)

    idx_wall = torch.where(
        any_wall,
        is_wall.to(torch.float32).argmax(dim=-1),
        torch.full(is_wall.shape[:-1], -1, device=device, dtype=torch.long),
    )
    idx_agent = torch.where(
        any_agent,
        has_agent.to(torch.float32).argmax(dim=-1),
        torch.full(has_agent.shape[:-1], -1, device=device, dtype=torch.long),
    )

    first_kind = torch.zeros((num_agents, 8), dtype=torch.int64, device=device)
    first_idx = torch.full((num_agents, 8), -1, dtype=torch.long, device=device)

    both_hit = (idx_wall >= 0) & (idx_agent >= 0)
    only_wall = (idx_wall >= 0) & (idx_agent < 0)
    only_agent = (idx_wall < 0) & (idx_agent >= 0)

    if bool(both_hit.any().item()):
        earlier_is_wall = idx_wall <= idx_agent
        first_idx[both_hit] = torch.where(earlier_is_wall, idx_wall, idx_agent)[both_hit]
        first_kind[both_hit & earlier_is_wall] = 1
        first_kind[both_hit & (~earlier_is_wall)] = -2

    if bool(only_wall.any().item()):
        first_idx[only_wall] = idx_wall[only_wall]
        first_kind[only_wall] = 1

    if bool(only_agent.any().item()):
        first_idx[only_agent] = idx_agent[only_agent]
        first_kind[only_agent] = -2

    agent_mask = first_kind == -2
    if bool(agent_mask.any().item()):
        gather_idx = first_idx.clamp_min(0).unsqueeze(-1)
        gather_y = torch.gather(y, 2, gather_idx).squeeze(-1)
        gather_x = torch.gather(x, 2, gather_idx).squeeze(-1)

        team = grid[0][gather_y, gather_x].to(torch.int32)
        unit = unit_map[gather_y, gather_x].to(torch.int32)

        resolved = torch.zeros_like(team, dtype=torch.int64)
        resolved[(team == 2) & (unit == 1)] = 2
        resolved[(team == 2) & (unit == 2)] = 3
        resolved[(team == 3) & (unit == 1)] = 4
        resolved[(team == 3) & (unit == 2)] = 5
        first_kind[agent_mask] = resolved[agent_mask]

    denom = max_steps_each.clamp_min(1).to(torch.float32).view(num_agents, 1).expand(num_agents, 8)
    valid = (first_idx >= 0).to(torch.float32)
    dist_norm = ((first_idx.to(torch.float32) + 1.0) / denom) * valid

    hp_first = torch.gather(hp, 2, first_idx.clamp_min(0).unsqueeze(-1)).squeeze(-1) * valid

    onehot = torch.zeros((num_agents, 8, _TYPE_CLASSES), dtype=dtype, device=device)
    onehot.scatter_(2, first_kind.clamp(min=0, max=_TYPE_CLASSES - 1).unsqueeze(-1), 1.0)

    max_hp = float(getattr(config, "MAX_HP", 1.0))
    if max_hp <= 0.0:
        max_hp = 1.0

    hp_norm = (hp_first / max_hp).clamp(0.0, 1.0).to(dtype)
    feat = torch.cat([onehot, dist_norm.to(dtype).unsqueeze(-1), hp_norm.unsqueeze(-1)], dim=-1)
    return feat.reshape(num_agents, 8 * 8)
