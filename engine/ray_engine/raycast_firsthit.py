from __future__ import annotations
# Future annotations
# This import changes how Python evaluates type annotations.
# With this enabled, annotations are stored as strings (forward references) and
# are not immediately evaluated at function definition time.
# Practical benefits:
#   • Prevents import-order issues in complex projects (especially with circular
#     imports between modules).
#   • Allows referencing types that are defined later in the same module.
#   • Slightly reduces runtime overhead during import.
# This affects only annotation behavior; it does not alter tensor computation.
from __future__ import annotations

# Typing imports
# Optional[T] means the value is either of type T or it is None.
# Here, `max_steps_each` may be omitted, in which case a default range is used.
from typing import Optional

# PyTorch import
# torch is the primary tensor computation library used for:
#   • CPU/GPU execution
#   • efficient vectorized operations
#   • tensor indexing and broadcasting
# In this module, torch is used for building an integer unit map and for batched
# raycasting feature extraction.
import torch

# Project configuration import
# The code reads several configuration knobs from `config`, typically including:
#   • TORCH_DTYPE         : floating dtype for returned features (e.g. float32)
#   • RAYCAST_MAX_STEPS   : global maximum ray length in grid steps
#   • MAX_HP              : normalization constant for hit points
# getattr(...) is used defensively so missing fields do not immediately crash.
import config

# Grid representation documentation
# This module assumes a "grid" tensor of shape (3, H, W) with the following
# semantics per channel:
#   channel 0: occupancy / tile type encoding
#       0 = empty
#       1 = wall
#       2 = red team occupancy marker
#       3 = blue team occupancy marker
#   channel 1: hp (hit points), typically in [0, MAX_HP]
#   channel 2: agent_id
#       -1 indicates no agent present
#       >=0 indicates an agent exists at that cell (agent registry index/id)
# Additionally, an HxW `unit_map` is used for unit subtype classification:
#   -1 = empty/no agent
#    1 = soldier
#    2 = archer

# DIRS8: the 8-connected neighborhood directions
# These are the standard 8 compass directions (including diagonals) used in many
# grid-world environments, corresponding to 8-neighborhood connectivity:
#   Index: 0   1    2   3   4   5    6   7
#   Dir:   N  NE    E  SE   S  SW    W  NW
# Coordinate convention used here:
#   The direction vectors are in (dx, dy) format.
#   • dx increases as we move to the right (east).
#   • dy increases as we move downward (south).
# Hence:
#   N  = ( 0, -1)
#   E  = ( 1,  0)
#   S  = ( 0,  1)
#   W  = (-1,  0)
# Diagonals follow accordingly:
#   NE = ( 1, -1), SE = ( 1,  1), SW = (-1,  1), NW = (-1, -1)
# The comment notes this ordering is consistent with `move_mask`, which strongly
# implies other parts of the project assume the same directional indexing.
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

# One-hot class count for ray "first hit" classification
# The ray feature encodes what the ray hits first using a one-hot vector with 6
# possible classes:
#   0 = none          (no hit within max range)
#   1 = wall
#   2 = red-soldier
#   3 = red-archer
#   4 = blue-soldier
#   5 = blue-archer
# The encoding for soldier/archer is derived from `unit_map` values {1,2}, while
# red/blue is derived from occupancy channel values {2,3}.
_TYPE_CLASSES = 6


@torch.no_grad()
def build_unit_map(agent_data: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """
    Construct an (H, W) integer map describing the unit subtype at each grid cell.

    Purpose:
    --------
    In many grid simulations, the occupancy grid alone (e.g. "red agent present")
    is insufficient to fully identify what kind of agent is present. This helper
    function resolves agent subtypes (e.g., soldier vs archer) and produces a
    per-cell map that can be used during raycasting and feature extraction.

    Inputs:
    -------
    agent_data:
        A tensor representing a registry of agent attributes.
        Shape is documented implicitly as (N, features), where N is the number
        of agents. One of these feature columns encodes "unit type".

    grid:
        A tensor of shape (3, H, W), where:
          • grid[2] contains agent ids: -1 for empty, >=0 for agent id.

    Output:
    -------
    unit_map:
        A tensor of shape (H, W), dtype int32, on the same device as `grid`,
        containing:
          • -1 where there is no agent
          •  1 where an agent exists and is of subtype "soldier"
          •  2 where an agent exists and is of subtype "archer"

    Implementation strategy:
    ------------------------
    1) Initialize unit_map to -1 everywhere.
    2) Read the agent_id layer from grid[2].
    3) For each cell where agent_id >= 0, map that id into agent_data to retrieve
       the unit subtype.
    4) Scatter that subtype into the corresponding unit_map cell.

    Design note:
    ------------
    The function uses a local import for COL_UNIT to reduce the risk of circular
    imports. This is a common technique in large codebases where registries and
    feature extraction code may otherwise depend on each other at import time.
    """

    # Extract spatial dimensions from grid = (3, H, W).
    H, W = int(grid.size(1)), int(grid.size(2))

    # Initialize the unit map as "empty/no agent" everywhere.
    # dtype=int32 is chosen likely for compactness and fast comparison.
    # device=grid.device ensures no CPU↔GPU synchronization or transfers.
    unit_map = torch.full((H, W), -1, dtype=torch.int32, device=grid.device)

    # ids is the agent id layer. Convention:
    #   -1 = empty
    #   >=0 = index/id into agent_data registry
    ids = grid[2].to(torch.long)  # (H,W) -1 if empty

    # Boolean mask marking locations that contain an agent.
    has_agent = ids >= 0

    # Fast path: if grid is empty of agents, return the all -1 unit_map.
    # This avoids unnecessary work and prevents indexing operations.
    if not has_agent.any():
        return unit_map

    # Gather unit type by agent id and place it in the spatial map.
    # Local import to avoid circular module dependencies.
    # COL_UNIT is assumed to be the column index in agent_data that stores the
    # unit subtype. The comment indicates it is float values {1.0, 2.0}.
    from ..agent_registry import COL_UNIT  # local import to avoid circulars

    # units_by_id:
    #   A vector of length N (number of agents), where units_by_id[i] is the unit
    #   subtype of agent i.
    # It is cast to int32 to align with the desired unit_map encoding.
    units_by_id = agent_data[:, COL_UNIT].to(torch.int32)  # (N,)

    # picked:
    #   An (H, W) tensor where for each cell:
    #     • if has_agent is True, value = units_by_id[agent_id]
    #     • else, value = -1
    # ids.clamp_min(0) ensures that empty cells (ids=-1) do not attempt to index
    # units_by_id with a negative index. These entries are anyway discarded by
    # torch.where using has_agent as the condition.
    picked = torch.where(
        has_agent,
        units_by_id[ids.clamp_min(0)],
        torch.tensor(-1, device=grid.device, dtype=torch.int32),
    )

    # Copy the picked values into unit_map in-place.
    # copy_ keeps the same tensor object and can be marginally more efficient.
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
    Compute first-hit raycast features using 8 discrete directions (DIRS8).

    High-level concept:
    -------------------
    Each agent is equipped with 8 rays corresponding to the 8 compass directions.
    Along each ray, the algorithm "marches" outward one grid cell at a time up to
    some maximum range. It records the *first* meaningful obstacle encountered,
    either:
      • a wall, or
      • another agent,
    and if neither is found, it records "none".

    Feature schema per ray (8 dimensions total):
    --------------------------------------------
      [ onehot6(none,wall,red-sold,red-arch,blue-sold,blue-arch),
        dist_norm,
        hp_norm ]

    where:
      • onehot6: encodes the category of the first hit.
      • dist_norm: distance to first hit normalized by that agent’s own vision
        range (max_steps_each).
      • hp_norm: hit points at the hit cell, normalized by MAX_HP.

    Output shape:
    -------------
      • 8 rays × 8 dims = 64 features per agent
      • Return tensor has shape (N, 64)

    Per-agent range:
    ----------------
    The function supports per-agent maximum ray range (vision distance) via
    `max_steps_each`. This is useful in environments where agents may have
    different senses, debuffs, line-of-sight constraints, or unit-type vision.

    Performance characteristics:
    ----------------------------
    The implementation is fully vectorized:
      • It constructs coordinates for all agents, all rays, all steps at once.
      • It uses tensor indexing and reductions to find first-hit indices.
      • It avoids Python loops, which is critical for speed at scale.
    """

    # Device and dtype policy
    # Ensure computation takes place on the same device as grid to avoid
    # costly device transfers and to enable GPU acceleration.
    device = grid.device

    # Feature tensors (one-hot and normalized scalars) are produced using a
    # configurable floating dtype. This can be used to reduce memory bandwidth
    # (e.g. float16/bfloat16) or preserve numeric stability (float32).
    dtype = getattr(config, "TORCH_DTYPE", torch.float32)

    # Input normalization: positions, sizes
    # pos_xy: agent coordinates, expected as integer indices (x, y).
    # Force onto device and long dtype for safe grid indexing.
    pos_xy = pos_xy.to(dtype=torch.long, device=device)

    # N = number of agents.
    N = int(pos_xy.size(0))

    # H, W = grid height and width.
    H, W = int(grid.size(1)), int(grid.size(2))

    # Configure ray length: global cap and per-agent effective max
    # R_global is a hard maximum number of steps for any ray.
    # It bounds computation and memory cost.
    R_global = int(getattr(config, "RAYCAST_MAX_STEPS", 10))

    # If per-agent max is not provided, all agents use the global cap.
    # If provided, clamp to [0, R_global] for correctness and safety.
    if max_steps_each is None:
        max_steps_each = torch.full((N,), R_global, device=device, dtype=torch.long)
    else:
        max_steps_each = torch.clamp(
            max_steps_each.to(device=device, dtype=torch.long),
            0,
            R_global
        )

    # Construct coordinates for all agents × rays × steps
    # dirs: (1, 8, 2) direction vectors.
    # Broadcasting allows adding these to all agents in one operation.
    dirs = DIRS8.to(device).view(1, 8, 2)                    # (1,8,2)

    # base: (N, 1, 1, 2) base positions.
    # This will broadcast across rays and steps.
    base = pos_xy.view(N, 1, 1, 2)                           # (N,1,1,2)

    # steps: (1, 1, S, 1) step magnitudes 1..R_global inclusive.
    # dtype long because DIRS8 and base are integer, so coords remain integer.
    steps = torch.arange(
        1, R_global + 1,
        device=device,
        dtype=torch.long
    ).view(1, 1, R_global, 1)  # (1,1,S,1)

    # coords: (N, 8, S, 2)
    # coords[i, r, s] = pos_xy[i] + DIRS8[r] * (s+1)
    # (s+1 because steps start at 1).
    coords = base + dirs.view(1, 8, 1, 2) * steps            # (N,8,S,2)

    # Split coords into x and y and clamp to grid bounds.
    # clamp_ is in-place, reducing allocations.
    x = coords[..., 0].clamp_(0, W - 1)                      # (N,8,S)
    y = coords[..., 1].clamp_(0, H - 1)                      # (N,8,S)

    # Active mask: disable steps beyond each agent's max vision range
    # step_ids is (1, 1, S) holding 1..S.
    step_ids = torch.arange(
        1, R_global + 1,
        device=device,
        dtype=torch.long
    ).view(1, 1, R_global)

    # active is (N, 1, S) and broadcasts across the 8 rays.
    # active[i, 0, s] is True if s+1 <= max_steps_each[i].
    active = step_ids <= max_steps_each.view(N, 1, 1)        # (N,1,S)

    # Gather grid values along the ray paths
    # occ: occupancy type along rays.
    # hp:  hit points along rays.
    # uid: unit subtype along rays from unit_map.
    # Note: uid is not directly used to locate hits; it is used later for
    # class resolution (soldier vs archer) once an agent hit is confirmed.
    occ = grid[0][y, x]                                      # (N,8,S)
    hp  = grid[1][y, x]                                      # (N,8,S)
    uid = unit_map[y, x]                                     # (N,8,S) ∈ {-1,1,2}

    # Determine hit masks: wall hits and agent hits (within active range)
    # is_wall: True where occupancy indicates wall and the step is within range.
    # Broadcasting note:
    #   active is (N,1,S) and is_wall is (N,8,S); the singleton ray dimension
    #   in active broadcasts to 8 automatically.
    is_wall = (occ == 1) & active                            # (N,8,S)

    # has_agent: True where agent_id channel indicates presence and within range.
    has_agent = (grid[2][y, x] >= 0) & active                # (N,8,S)

    # Compute first-hit indices for wall and agent separately
    # First wall step index for each (N,8).
    any_wall = is_wall.any(dim=-1)                           # (N,8) whether any wall exists along ray

    idx_wall = torch.where(
        any_wall,
        is_wall.to(torch.float32).argmax(dim=-1),            # first True index via argmax on {0,1}
        torch.full(is_wall.shape[:-1], -1, device=device, dtype=torch.long),
    )  # (N,8)

    # First agent step index for each (N,8).
    any_agent = has_agent.any(dim=-1)                        # (N,8) whether any agent exists along ray

    idx_agent = torch.where(
        any_agent,
        has_agent.to(torch.float32).argmax(dim=-1),
        torch.full(has_agent.shape[:-1], -1, device=device, dtype=torch.long),
    )  # (N,8)

    # Resolve which hit occurs first (wall vs agent) per ray
    # first_kind is the semantic class code:
    #   0 = none
    #   1 = wall
    #   -2 = temporary marker for "agent hit" that must later be mapped to 2..5
    first_kind = torch.full((N, 8), 0, dtype=torch.int64, device=device)  # 0 none

    # first_idx is the step index (0-based along the steps axis) of the first hit.
    # -1 indicates no hit.
    first_idx  = torch.full((N, 8), -1, dtype=torch.long, device=device)

    # Determine which rays have both a wall and an agent somewhere in range.
    both_hit = (idx_wall >= 0) & (idx_agent >= 0)

    # Rays with only wall hits and no agent hits.
    only_wall = (idx_wall >= 0) & (~(idx_agent >= 0))

    # Rays with only agent hits and no wall hits.
    only_agent = (~(idx_wall >= 0)) & (idx_agent >= 0)

    # If both are present, choose the earlier hit (smaller index).
    # Ties go to wall because <=.
    if both_hit.any():
        earlier_is_wall = (idx_wall <= idx_agent)
        fi = torch.where(earlier_is_wall, idx_wall, idx_agent)

        first_idx[both_hit] = fi[both_hit]
        first_kind[both_hit & earlier_is_wall] = 1            # wall
        first_kind[both_hit & (~earlier_is_wall)] = -2        # agent (temp)

    # If only wall exists, it must be the first hit.
    if only_wall.any():
        first_idx[only_wall] = idx_wall[only_wall]
        first_kind[only_wall] = 1                              # wall

    # If only agent exists, it must be the first hit.
    if only_agent.any():
        first_idx[only_agent] = idx_agent[only_agent]
        first_kind[only_agent] = -2                            # agent (temp)

    # Resolve agent hits into {2,3,4,5} class codes using team + unit subtype
    agent_mask = (first_kind == -2)

    if agent_mask.any():
        # Gather the (y, x) coordinates at the first-hit step for each (N,8).
        # As before, clamp_min(0) avoids invalid gather indices; entries that
        # were actually invalid are not in agent_mask and thus not used.
        gather_y = torch.gather(
            y,
            2,
            first_idx.clamp_min(0).unsqueeze(-1)
        ).squeeze(-1)  # (N,8)

        gather_x = torch.gather(
            x,
            2,
            first_idx.clamp_min(0).unsqueeze(-1)
        ).squeeze(-1)  # (N,8)

        # t: occupancy marker at hit cell: 2 for red, 3 for blue.
        t = grid[0][gather_y, gather_x].to(torch.int32)       # (N,8)

        # u: unit subtype at hit cell: 1 for soldier, 2 for archer.
        u = unit_map[gather_y, gather_x].to(torch.int32)      # (N,8)

        # code tensor will hold the resolved 2..5 values.
        code = torch.full_like(t, 0, dtype=torch.int64)

        # Red team resolutions:
        code[(t == 2) & (u == 1)] = 2                         # red-soldier
        code[(t == 2) & (u == 2)] = 3                         # red-archer

        # Blue team resolutions:
        code[(t == 3) & (u == 1)] = 4                         # blue-soldier
        code[(t == 3) & (u == 2)] = 5                         # blue-archer

        # Write back resolved codes into first_kind for those rays.
        first_kind[agent_mask] = code[agent_mask]

    # Distance feature: normalize by each agent's own max vision
    # den: (N,8) denominator equals max_steps_each per agent, expanded to match rays.
    # clamp_min(1) avoids division by zero if any agent has max_steps_each=0.
    den = max_steps_each.clamp_min(1).to(torch.float32).view(N, 1).expand(N, 8)

    # dist_idx: convert 0-based step index to 1-based distance in steps.
    dist_idx = first_idx.to(torch.float32) + 1.0

    # valid: 1.0 if hit exists, else 0.0
    valid = (first_idx >= 0).to(torch.float32)

    # dist_norm: normalized distance to first hit, or 0 if no hit.
    dist_norm = (dist_idx / den) * valid

    # HP feature: gather HP at the first-hit location and normalize
    # hp_first: gather HP from hp along step dimension at first_idx.
    hp_first = torch.gather(
        hp,
        2,
        first_idx.clamp_min(0).unsqueeze(-1)
    ).squeeze(-1)  # (N,8)

    # Zero out invalid rays (no hit).
    hp_first = hp_first * valid

    # One-hot encoding for the 6 first-hit classes
    onehot = torch.zeros((N, 8, _TYPE_CLASSES), dtype=dtype, device=device)

    # Ensure indices are within [0, 5] before scattering.
    idx_valid = first_kind.clamp(min=0, max=_TYPE_CLASSES - 1)

    # Scatter 1.0 into the correct class slot.
    onehot.scatter_(2, idx_valid.unsqueeze(-1), 1.0)

    # Normalize HP by MAX_HP with defensive safeguards
    max_hp = float(getattr(config, "MAX_HP", 1.0))
    if max_hp <= 0:  # avoid division by zero or negative normalization
        max_hp = 1.0

    # hp_norm: clamp into [0,1] to avoid out-of-range artifacts.
    hp_norm = (hp_first / max_hp).clamp(0.0, 1.0).to(dtype)

    # dist_norm should match dtype for consistent downstream consumption.
    dist_norm = dist_norm.to(dtype)

    # Assemble final feature tensor:
    #   onehot(6) + dist_norm(1) + hp_norm(1) = 8 dims per ray
    feat = torch.cat(
        [onehot, dist_norm.unsqueeze(-1), hp_norm.unsqueeze(-1)],
        dim=-1
    )  # (N,8,8)

    # Flatten per-agent ray features into a single vector:
    #   (N, 8 * 8) = (N, 64)
    return feat.reshape(N, 8 * 8)
