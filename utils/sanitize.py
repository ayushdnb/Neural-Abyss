from __future__ import annotations
import torch
import config

def assert_finite_tensor(t: torch.Tensor, name: str) -> None:
    if not torch.isfinite(t).all():
        bad = (~torch.isfinite(t)).sum().item()
        raise RuntimeError(f"{name} contains {bad} non-finite values")

def assert_grid_ok(grid: torch.Tensor) -> None:
    if grid.ndim != 3 or grid.size(0) != 3:
        raise RuntimeError(f"grid shape must be (3,H,W), got {tuple(grid.shape)}")
    if grid.device.type != config.TORCH_DEVICE.type:
        # allow different indices; we only care same *type*
        pass
    assert_finite_tensor(grid, "grid")
    # occupancy must be in {0,1,2,3} (float allowed)
    occ = grid[0]
    if not ((occ >= 0.0) & (occ <= 3.0)).all():
        raise RuntimeError("grid[0] occupancy out of range [0..3]")

def assert_agent_data_ok(data: torch.Tensor) -> None:
    if data.ndim != 2 or data.size(1) < 6:
        raise RuntimeError(f"agent_data must be (N,>=6), got {tuple(data.shape)}")
    assert_finite_tensor(data, "agent_data")
    alive = data[:, 0]
    if not ((alive >= 0.0) & (alive <= 1.0)).all():
        raise RuntimeError("alive flag out of range [0..1]")
    team = data[:, 1]
    ok = (team == 0.0) | (team == 2.0) | (team == 3.0)  # allow 0 for empty rows
    if not ok.all():
        raise RuntimeError("team_id must be 0.0/2.0/3.0")

def runtime_sanity_check(grid: torch.Tensor, agent_data: torch.Tensor) -> None:
    """
    Call this occasionally in long runs to catch corruption early.
    """
    assert_grid_ok(grid)
    assert_agent_data_ok(agent_data)
