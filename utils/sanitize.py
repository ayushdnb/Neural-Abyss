"""Periodic runtime validation for grid and agent tensors."""

from __future__ import annotations

import torch

import config
from engine.agent_registry import (
    COL_ALIVE,
    COL_TEAM,
    NUM_COLS,
    TEAM_BLUE_ID,
    TEAM_RED_ID,
)


def assert_finite_tensor(tensor: torch.Tensor, name: str) -> None:
    """Raise when ``tensor`` contains NaN or infinity."""
    finite_mask = torch.isfinite(tensor)
    if bool(finite_mask.all().item()):
        return
    bad = int((~finite_mask).sum().item())
    raise RuntimeError(f"{name} contains {bad} non-finite values")


def assert_grid_ok(grid: torch.Tensor) -> None:
    """Validate the grid tensor shape and occupancy range."""
    if grid.ndim != 3 or int(grid.size(0)) != 3:
        raise RuntimeError(f"grid shape must be (3, H, W), got {tuple(grid.shape)}")

    assert_finite_tensor(grid, "grid")

    occupancy = grid[0]
    if not bool(((occupancy >= 0.0) & (occupancy <= 3.0)).all().item()):
        raise RuntimeError("grid[0] occupancy must stay within [0, 3]")

    expected_device = getattr(config, "TORCH_DEVICE", None)
    if expected_device is not None and grid.device.type != expected_device.type:
        raise RuntimeError(
            f"grid device type mismatch: expected {expected_device.type}, got {grid.device.type}"
        )


def assert_agent_data_ok(agent_data: torch.Tensor) -> None:
    """Validate the registry tensor shape and categorical columns."""
    if agent_data.ndim != 2 or int(agent_data.size(1)) < int(NUM_COLS):
        raise RuntimeError(
            f"agent_data must be (N, >= {int(NUM_COLS)}), got {tuple(agent_data.shape)}"
        )

    assert_finite_tensor(agent_data, "agent_data")

    alive = agent_data[:, COL_ALIVE]
    if not bool(((alive >= 0.0) & (alive <= 1.0)).all().item()):
        raise RuntimeError("agent_data alive flag must stay within [0, 1]")

    team = agent_data[:, COL_TEAM]
    valid_team = (team == 0.0) | (team == TEAM_RED_ID) | (team == TEAM_BLUE_ID)
    if not bool(valid_team.all().item()):
        raise RuntimeError("agent_data team id must be 0.0, TEAM_RED_ID, or TEAM_BLUE_ID")


def runtime_sanity_check(grid: torch.Tensor, agent_data: torch.Tensor) -> None:
    """Run the configured runtime state checks."""
    assert_grid_ok(grid)
    assert_agent_data_ok(agent_data)
