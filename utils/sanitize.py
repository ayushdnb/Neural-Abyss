"""Periodic runtime validation for grid and agent tensors."""

from __future__ import annotations

import torch
import config


def assert_finite_tensor(t: torch.Tensor, name: str) -> None:
    """
    Assert that every element of tensor `t` is finite.

    Why this matters
    ----------------
    In PyTorch (and numerical computing), NaNs/Infs are contagious:
    once you have one NaN in your state, many computations will turn into NaN.
    This is especially destructive in:
    - reinforcement learning (value loss becomes NaN)
    - physics-like simulations (positions/health explode)
    - neural nets (gradients become NaN)

    Implementation details
    ----------------------
    - `torch.isfinite(t)` returns a boolean tensor of the same shape:
      True where element is finite, False where it is NaN or Inf.
    - `.all()` checks if all elements are True.
    - We count the number of "bad" values for a helpful error message.

    Parameters
    ----------
    t:
        Any PyTorch tensor you want to validate.
    name:
        A human-readable label used in error messages.

    Raises
    ------
    RuntimeError:
        If any element is NaN or Inf.
    """
    # torch.isfinite(t): boolean tensor -> True if value is not NaN/Inf
    # If any value is not finite, we raise.
    if not torch.isfinite(t).all():
        # (~mask) flips booleans: True becomes False, False becomes True
        # Here, (~torch.isfinite(t)) is True exactly where values are NaN/Inf.
        bad = (~torch.isfinite(t)).sum().item()  # `.item()` converts 0-d tensor to a Python number.
        raise RuntimeError(f"{name} contains {bad} non-finite values")


def assert_grid_ok(grid: torch.Tensor) -> None:
    """
    Validate the world/grid tensor.

    Expected representation
    -----------------------
    This code assumes `grid` is a 3D tensor with shape:

        (3, H, W)

    where the first dimension is **channels** (like layers in an image).
    A common pattern is:
    - grid[0] = occupancy / terrain / entity ids
    - grid[1] = some feature map (e.g., team control, resources, heatmap)
    - grid[2] = another feature map

    Only the constraints enforced below are guaranteed by this file.
    Everything else is project-specific.

    Checks performed
    ---------------
    1) Shape constraint: must be 3D and have exactly 3 channels.
    2) Device type check (CPU vs CUDA) compared to `config.TORCH_DEVICE.type`.
       - The code intentionally allows different device indices (e.g., cuda:0 vs cuda:1)
         but expects the same device *type*.
    3) All values must be finite (no NaN/Inf).
    4) Occupancy channel range constraint:
       - grid[0] must be in [0..3] (float allowed).
         This implies an encoding like:
           0 = empty
           1/2/3 = different object/team/wall types (exact meaning depends on project)

    Parameters
    ----------
    grid:
        The grid tensor to validate.

    Raises
    ------
    RuntimeError:
        If shape is wrong or occupancy is out of range.
    RuntimeError:
        If any non-finite values exist (delegated to assert_finite_tensor).
    """
    # ---- 1) Shape check ----
    # grid.ndim is number of tensor dimensions.
    # We require exactly 3 dimensions: (C, H, W) where C must be 3.
    if grid.ndim != 3 or grid.size(0) != 3:
        raise RuntimeError(f"grid shape must be (3,H,W), got {tuple(grid.shape)}")

    # ---- 2) Device type check ----
    # In PyTorch, tensors live on a "device":
    # - CPU: grid.device.type == "cpu"
    # - GPU: grid.device.type == "cuda"  (or "mps" on Apple)
    # `config.TORCH_DEVICE` is presumably something like:
    #   torch.device("cuda") or torch.device("cuda:0") or torch.device("cpu")
    # This check is lenient: it does NOT error if the index differs
    # (e.g. cuda:0 vs cuda:1). It only cares that the type matches.
    if grid.device.type != config.TORCH_DEVICE.type:
        # allow different indices; we only care same *type*
        pass

    # ---- 3) Finite-value check ----
    assert_finite_tensor(grid, "grid")

    # ---- 4) Occupancy range check ----
    # grid[0] selects the first channel (occupancy).
    # It may be float for GPU-friendly operations, but values must stay in [0..3].
    occ = grid[0]
    if not ((occ >= 0.0) & (occ <= 3.0)).all():
        # `&` is elementwise AND between boolean tensors.
        # We raise if any cell violates the constraint.
        raise RuntimeError("grid[0] occupancy out of range [0..3]")


def assert_agent_data_ok(data: torch.Tensor) -> None:
    """
    Validate the per-agent data table tensor.

    Expected representation
    -----------------------
    This code assumes `agent_data` is a 2D tensor with shape:

        (N, >= 6)

    Where:
    - N = number of agent rows.
    - Each row is a fixed set of numeric fields for one agent.
    - The first few columns are assumed to have this meaning:
        col 0: alive flag (0.0 or 1.0)
        col 1: team id (allowed values: 0.0, 2.0, 3.0)
      Remaining columns exist (>= 6 total), but this file does not validate them.

    Why allow team == 0.0?
    ----------------------
    Many simulations allocate a fixed-sized tensor for agents.
    Some rows may be unused; team_id == 0.0 can represent an "empty row".

    Checks performed
    ---------------
    1) Shape constraint: must be 2D and have at least 6 columns.
    2) All values must be finite (no NaN/Inf).
    3) alive flag must be in [0..1].
       - This allows float values but expects they stay within boolean-like range.
    4) team id must be one of {0.0, 2.0, 3.0}.

    Parameters
    ----------
    data:
        Agent data tensor to validate.

    Raises
    ------
    RuntimeError:
        If shape is wrong or flags/ids are out of legal range.
    RuntimeError:
        If any non-finite values exist (delegated to assert_finite_tensor).
    """
    # ---- 1) Shape check ----
    # agent_data must be (N, >= 6).
    # The >= 6 suggests there are at least 6 fields per agent.
    if data.ndim != 2 or data.size(1) < 6:
        raise RuntimeError(f"agent_data must be (N,>=6), got {tuple(data.shape)}")

    # ---- 2) Finite-value check ----
    assert_finite_tensor(data, "agent_data")

    # ---- 3) Alive flag range check ----
    # data[:, 0] means: take column 0 across all rows.
    # This is a vector of length N.
    alive = data[:, 0]
    if not ((alive >= 0.0) & (alive <= 1.0)).all():
        raise RuntimeError("alive flag out of range [0..1]")

    # ---- 4) Team id allowed-values check ----
    team = data[:, 1]
    # ok is a boolean vector: ok[i] == True if team[i] is legal.
    ok = (team == 0.0) | (team == 2.0) | (team == 3.0)  # allow 0 for empty rows
    # `|` is elementwise OR between boolean tensors.
    if not ok.all():
        raise RuntimeError("team_id must be 0.0/2.0/3.0")


def runtime_sanity_check(grid: torch.Tensor, agent_data: torch.Tensor) -> None:
    """
    Run a bundle of sanity checks on the simulation runtime state.

    Intended usage
    --------------
    Call this occasionally (not necessarily every tick) in a long run:

        if tick % 1000 == 0:
            runtime_sanity_check(grid, agent_data)

    Why "occasionally"?
    -------------------
    These checks involve scanning tensors to ensure validity.
    On large grids / many agents, this can be expensive to do every frame,
    especially on GPU. But doing it periodically is a strong safety net:
    you catch corruption early, close to the tick that caused it.

    What corruption looks like in practice
    --------------------------------------
    - Random NaNs show up after an unstable update (division by near-zero).
    - Occupancy goes out of range because an indexing bug writes 4 or -1.
    - Alive flags drift due to unintended floating-point operations.
    - Team ids become weird decimals due to mixing float ops with categorical fields.

    Parameters
    ----------
    grid:
        World grid tensor (3, H, W).
    agent_data:
        Agent table tensor (N, >= 6).

    Raises
    ------
    RuntimeError:
        On any violated constraint (shape, finiteness, ranges).
    """
    assert_grid_ok(grid)
    assert_agent_data_ok(agent_data)
