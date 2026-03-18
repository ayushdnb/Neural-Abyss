from __future__ import annotations
# ^ Defers evaluation of type annotations. Benefits:
#   - Forward references in type hints
#   - Reduced runtime overhead for annotation evaluation
#   - Improved compatibility with tooling and older Python versions

import torch
import config

# Action Masking with Optional Line-of-Sight (LOS) Wall Blocking
# This module builds an action mask for each agent in a grid-based simulation.
# "Action mask" meaning:
#   - A boolean tensor mask[N, A] where:
#       mask[n, a] = True  -> action "a" is currently valid for agent "n"
#       mask[n, a] = False -> action "a" is invalid and should be disallowed
# This is typically used in RL with discrete actions to:
#   - prevent illegal actions from being sampled by the policy
#   - reduce wasted exploration on impossible moves
#   - simplify credit assignment by removing nonsensical actions
# The environment uses an occupancy grid:
#   grid: (3, H, W) float
#     - channel 0 (grid[0]) is occupancy / type map.
#       Values used in this file:
#         0.0 = empty
#         1.0 = wall
#         2.0 = team red (or red unit marker)
#         3.0 = team blue (or blue unit marker)
#       (Other channels exist but are not used in this mask builder.)
# "Teams" tensor:
#   teams: (N,) float where team identity is encoded as 2.0 (red) or 3.0 (blue)
# "Units" tensor:
#   unit: (N,) where:
#     1 = soldier
#     2 = archer
# Action layout:
#   A = config.NUM_ACTIONS
#   Two supported layouts are implied by code:
#     - A <= 17: idle + 8 moves + 8 melee attacks (range 1 only)
#     - A  = 41: idle + 8 moves + (8 directions × 4 ranges) ranged attack actions
# Optional LOS rule:
#   If config.ARCHER_LOS_BLOCKS_WALLS is True:
#     - ranged attacks are disabled when a wall exists in any intermediate cell
#       between attacker and target along that direction.


# 8 directions (dx, dy): N, NE, E, SE, S, SW, W, NW
# Coordinate convention implied by usage:
#   - x increases to the right
#   - y increases downward
# Therefore:
#   N  = (0, -1)
#   E  = (1,  0)
#   S  = (0,  1)
#   W  = (-1, 0)
DIRS8 = torch.tensor(
    [
        [ 0, -1],
        [ 1, -1],
        [ 1,  0],
        [ 1,  1],
        [ 0,  1],
        [-1,  1],
        [-1,  0],
        [-1, -1],
    ],
    dtype=torch.long
)


def _los_blocked_by_walls_grid0(
    occ0: torch.Tensor,  # (H, W) occupancy channel 0
    x0: torch.Tensor,    # (N,) attacker x
    y0: torch.Tensor,    # (N,) attacker y
    dxy: torch.Tensor,   # (8, 2) direction vectors (dx, dy)
    RMAX: int,
) -> torch.Tensor:
    """
    Compute line-of-sight blocking due to walls (occupancy value == 1.0) on channel 0.

    Returns:
        blocked: (N, 8, RMAX) boolean tensor where:
          blocked[n, d, r] == True means:
            along direction d, there exists at least one wall in the intermediate cells
            between the attacker and a target at range (r+1).

    Clarifying the "range index":
        - The function returns a dimension of size RMAX corresponding to target ranges:
              r = 1..RMAX (in world-space steps)
          But stored in the tensor at index:
              r_index = r - 1
          Hence blocked[:, :, 0] corresponds to r = 1.

    Intermediate-cell rule:
        - For an attack at range r, intermediate steps are 1..(r-1).
        - If r == 1, there are no intermediate cells, so LOS cannot be blocked by "in-between" walls.

    Practical interpretation:
        - This function is designed for ranged line-of-sight checks along 8-direction rays.
        - It does not check the target cell itself for being a wall; it checks only intermediate cells.
          (Target validity is handled elsewhere by verifying the target contains an enemy.)

    Performance considerations:
        - The function vectorizes wall detection for all agents N, all directions 8, and all intermediate steps.
        - The final step aggregates over intermediate cells using .any(dim=2).
        - The final loop over r=1..RMAX is small (RMAX is 4), so it is inexpensive.
    """
    device = occ0.device
    N = int(x0.numel())
    if N == 0:
        # No agents: return an empty boolean tensor with correct rank.
        return torch.zeros((0, 8, int(RMAX)), dtype=torch.bool, device=device)

    # steps 1..(RMAX-1) are the only possible "in-between" cells
    # Example with RMAX=4:
    #   steps = [1, 2, 3]
    #   For a target at range r=4, intermediate steps are 1..3.
    steps = torch.arange(1, int(RMAX), device=device, dtype=torch.long)  # (RMAX-1,)

    # If RMAX == 1, there are no intermediate steps to check.
    if int(steps.numel()) == 0:
        return torch.zeros((N, 8, int(RMAX)), dtype=torch.bool, device=device)

    # Reshape direction vectors for broadcasting:
    #   dx: (1, 8, 1)
    #   dy: (1, 8, 1)
    dx = dxy[:, 0].view(1, 8, 1)
    dy = dxy[:, 1].view(1, 8, 1)

    # Reshape step values for broadcasting:
    #   sx: (1, 1, RMAX-1)
    sx = steps.view(1, 1, -1)

    # Compute intermediate coordinates for each attacker, direction, and step:
    #   ix, iy: (N, 8, RMAX-1)
    # ix = x0 + dx * step
    # iy = y0 + dy * step
    ix = x0.view(N, 1, 1) + dx * sx
    iy = y0.view(N, 1, 1) + dy * sx

    # Grid bounds.
    H = int(occ0.shape[0])
    W = int(occ0.shape[1])

    # In-bounds mask for each intermediate cell.
    inb = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)

    # Clamp indices for safe indexing.
    # Important: clamping alone is not enough; we also AND with inb so that
    # out-of-bounds locations do not accidentally count as walls.
    ix_c = ix.clamp(0, W - 1)
    iy_c = iy.clamp(0, H - 1)

    # Detect walls on occupancy channel 0.
    # occ0[iy_c, ix_c] uses advanced indexing to gather values at all coordinates.
    # is_wall: (N, 8, RMAX-1)
    is_wall = (occ0[iy_c, ix_c] == 1.0) & inb

    # For each range r=1..RMAX, blocked if any wall exists at steps < r.
    # blocked[:, :, r-1] checks intermediate steps 1..(r-1).
    blocked = torch.zeros((N, 8, int(RMAX)), dtype=torch.bool, device=device)
    for r in range(1, int(RMAX) + 1):
        if r <= 1:
            # Range 1: no intermediate cells exist, so no LOS blockage by "in-between".
            continue

        # active mask selects intermediate steps strictly less than r.
        # steps are [1..RMAX-1], so steps < r implements 1..(r-1).
        active = steps.view(1, 1, -1) < int(r)

        # A direction is blocked at range r if any intermediate step is a wall.
        blocked[:, :, r - 1] = (is_wall & active).any(dim=2)

    return blocked


@torch.no_grad()
def build_mask(
    pos_xy: torch.Tensor,            # (N, 2) long/float (x, y)
    teams: torch.Tensor,             # (N,) float: 2.0=red, 3.0=blue
    grid: torch.Tensor,              # (3, H, W) float; ch0=occ(0,1,2,3)
    unit: torch.Tensor | None = None # (N,) long/float: 1=soldier, 2=archer
) -> torch.Tensor:
    """
    Build an action validity mask for N agents.

    Decorator:
        @torch.no_grad() indicates this function is intended for inference / masking,
        not for gradient-based learning. This avoids unnecessary autograd overhead.

    Args:
        pos_xy:
            (N, 2) positions for each agent: x and y.
            Values may be float or long; they are converted to long for indexing.

        teams:
            (N,) team identifiers (float), where:
              2.0 represents red
              3.0 represents blue
            Converted to long for comparisons.

        grid:
            (3, H, W) grid tensor.
            Only channel 0 is used here:
              grid[0] == 0.0 -> empty
              grid[0] == 1.0 -> wall
              grid[0] == 2.0 or 3.0 -> a unit / team marker

        unit:
            Optional (N,) unit type:
              1 = soldier
              2 = archer
            If None, defaults to archer for all agents (permissive default).

    Returns:
        mask:
            (N, A) boolean mask where A = config.NUM_ACTIONS (default 17 in this code).
            True indicates an action is legal/available.

    Action layouts:
      17-action layout:
        - index 0: idle
        - indices 1..8: moves in 8 directions
        - indices 9..16: melee attacks in 8 directions (range = 1 only)

      41-action layout:
        - index 0: idle
        - indices 1..8: moves in 8 directions
        - indices 9..40: attacks
            arranged as 8 direction blocks, each block has 4 range columns:
              base = 9
              direction d ∈ {0..7}:
                columns [base + d*4 ... base + d*4 + 3] correspond to ranges 1..4

    Unit gating for 41-action layout:
      - soldiers: range 1 only
      - archers: range 1..ARCHER_RANGE (clipped to max 4)
    """
    device = grid.device
    N = int(pos_xy.size(0))
    H, W = int(grid.size(1)), int(grid.size(2))

    # Number of actions (A). Defaults to 17 if config.NUM_ACTIONS is missing.
    A = int(getattr(config, "NUM_ACTIONS", 17))

    # Initialize all actions as invalid. We will set valid ones to True.
    mask = torch.zeros((N, A), dtype=torch.bool, device=device)

    # IDLE action (always valid if action space has at least one action).
    if A >= 1:
        mask[:, 0] = True

    # Early exit if there are no agents or no other actions exist.
    if N == 0 or A <= 1:
        return mask

    # Extract integer positions (x0, y0) for grid indexing.
    # non_blocking=True can help when tensors are on pinned memory and transfers occur.
    x0 = pos_xy[:, 0].to(torch.long, non_blocking=True)  # (N,)
    y0 = pos_xy[:, 1].to(torch.long, non_blocking=True)  # (N,)
    dirs = DIRS8.to(device)  # (8, 2)

    # MOVE actions (columns 1..8)
    # Compute neighbor positions in all 8 directions:
    #   nx, ny: (N, 8)
    nx = x0.view(N, 1) + dirs[:, 0].view(1, 8)
    ny = y0.view(N, 1) + dirs[:, 1].view(1, 8)

    # In-bounds mask for those neighbor cells.
    inb = (nx >= 0) & (nx < W) & (ny >= 0) & (ny < H)

    # Clamp for safe indexing (still need inb to prevent clamped OOB from counting as valid).
    nx_cl = nx.clamp(0, W - 1)
    ny_cl = ny.clamp(0, H - 1)

    # Occupancy at neighbor cells: occ is (N, 8).
    # Note: grid[0][ny_cl, nx_cl] uses advanced indexing.
    occ = grid[0][ny_cl, nx_cl]

    # A move is allowed if:
    #   - target cell is empty (occ == 0.0)
    #   - and the target is in bounds
    free = (occ == 0.0) & inb

    # Determine how many move columns exist in this action space.
    # For A=17, move_cols = min(8, 16) = 8 -> all 8 moves exist.
    move_cols = min(8, max(0, A - 1))
    if move_cols > 0:
        mask[:, 1:1 + move_cols] = free[:, :move_cols]

    # ATTACK actions
    # If action space ends at index 8, there are no attack actions.
    if A <= 9:
        return mask

    # Convert teams to integer codes for comparisons.
    teamv = teams.to(torch.long, non_blocking=True)  # (N,)

    # Legacy 17-action layout: melee only at range 1 per direction
    if A <= 17:
        # For melee at range 1, the potential target is the neighbor cell occupancy already computed:
        #   tgt_team: (N, 8) uses occ from move computation.
        tgt_team = occ

        # Enemy detection rule for melee:
        #   - not empty (!= 0.0)
        #   - not wall (!= 1.0)
        #   - not own team code (!= teamv)
        enemy = (tgt_team != 0.0) & (tgt_team != 1.0) & (tgt_team != teamv.view(N, 1))

        # Attack columns start at index 9.
        # k ensures we do not write beyond action dimension.
        k = min(8, max(0, A - 9))
        if k > 0:
            mask[:, 9:9 + k] = enemy[:, :k]
        return mask

    # 41-action layout: 8 directions × 4 ranges
    RMAX = 4

    # Prepare broadcast shapes for direction and range computations.
    dx = dirs[:, 0].view(1, 8, 1)
    dy = dirs[:, 1].view(1, 8, 1)

    # rvec is the range step vector: [1, 2, 3, 4], reshaped for broadcasting.
    rvec = torch.arange(1, RMAX + 1, device=device, dtype=torch.long).view(1, 1, RMAX)

    # Target coordinates for each agent, direction, and range:
    #   tx, ty: (N, 8, 4)
    tx = x0.view(N, 1, 1) + dx * rvec
    ty = y0.view(N, 1, 1) + dy * rvec

    # In-bounds mask for target cells.
    inb_r = (tx >= 0) & (tx < W) & (ty >= 0) & (ty < H)

    # Clamp for safe indexing.
    txc = tx.clamp(0, W - 1)
    tyc = ty.clamp(0, H - 1)

    # Occupancy at targets: (N, 8, 4)
    tgt_occ = grid[0][tyc, txc]

    # Only allow attacks if the target cell currently contains an enemy:
    #   - not empty (!= 0.0)
    #   - not wall (!= 1.0)
    #   - not own team code
    # Important: The code compares tgt_occ.to(long) to teamv.
    # This implies team values in grid[0] match the team coding (2 or 3).
    enemy_r = (tgt_occ != 0.0) & (tgt_occ != 1.0) & (tgt_occ.to(torch.long) != teamv.view(N, 1, 1))

    # Enforce bounds. (Note: this line appears twice in the original code.)
    enemy_r &= inb_r
    enemy_r &= inb_r

    # Unit gating: determine which ranges are allowed per unit type
    if unit is None:
        # Default permissive behavior: treat everyone as an archer.
        units = torch.full((N,), 2, device=device, dtype=torch.long)
    else:
        units = unit.to(torch.long, non_blocking=True)

    # Archer range from config, clipped to [1, RMAX].
    ar_range = int(getattr(config, "ARCHER_RANGE", 4))
    ar_range = max(1, min(RMAX, ar_range))

    # allow_r[n, r] indicates whether agent n can attack at range r+1.
    # Shape: (N, 4)
    allow_r = torch.zeros((N, RMAX), dtype=torch.bool, device=device)

    # Soldiers: allow only range 1 (index 0).
    allow_r[units == 1, 0] = True

    # Archers: allow ranges 1..ar_range.
    if (units == 2).any():
        allow_r[units == 2, :ar_range] = True

    # Combine enemy existence with range allowance.
    # allow_r.view(N, 1, RMAX) broadcasts across directions (8).
    atk_ok = enemy_r & allow_r.view(N, 1, RMAX)  # (N, 8, 4)

    # Optional LOS wall blocking
    # If enabled, ranged attacks are disabled if there is a wall in any intermediate cell
    # between attacker and target along the attack direction.
    # Important subtlety:
    # - blocked is computed for all agents and directions for ranges 1..4.
    # - For range 1, blocked is always False (no intermediate cells exist).
    if bool(getattr(config, "ARCHER_LOS_BLOCKS_WALLS", False)):
        blocked = _los_blocked_by_walls_grid0(
            occ,                          # Note: occ here is neighbor occupancy (N, 8), not the full (H, W) grid.
            x0,
            y0,
            DIRS8.to(device=device),
            RMAX
        )
        atk_ok = atk_ok & (~blocked)

    # Write attack mask into action columns
    # Attack columns start at base=9, with contiguous blocks of size RMAX per direction:
    #   direction d block:
    #     columns [9 + d*4, 9 + d*4 + 1, 9 + d*4 + 2, 9 + d*4 + 3]
    base = 9
    for d in range(8):
        c0 = base + d * RMAX
        c1 = c0 + RMAX
        if c0 >= A:
            break

        cols = slice(c0, min(c1, A))
        rlim = cols.stop - cols.start  # number of range columns written (<= 4)
        if rlim > 0:
            mask[:, cols] = atk_ok[:, d, :rlim]

    return mask
