from __future__ import annotations
# =============================================================================
# This module implements agent spawning strategies and team-aware brain selection.
#
# The code operates within a grid-based simulation where:
#   - The world state is represented by a 3-channel tensor grid[C, H, W].
#   - Agents are tracked in an AgentsRegistry (dense tensor + module lists).
#   - Each spawned agent is assigned:
#       (i)  a position (x, y),
#       (ii) a unit type (soldier vs archer) with corresponding stats,
#       (iii) a "brain" (policy/controller network) chosen by configurable rules.
#
# A central design objective is to keep the spawning logic:
#   - deterministic when desired (via seeds),
#   - configurable (via config module knobs),
#   - efficient (vectorizable where possible, minimal bookkeeping).
#
# IMPORTANT: Per your instruction, the executable code is not modified.
# Only explanatory comments are added.
# =============================================================================

import math
import random
from typing import Optional, Tuple

import torch
import config
from .agent_registry import AgentsRegistry

# Brains
from agent.mlp_brain import create_mlp_brain


def _rect_dims(n: int, max_cols: int, max_rows: int) -> Tuple[int, int, int]:
    """Calculates dimensions for a compact rectangle to place n agents.

    PURPOSE
    -------
    When spawning agents in formations, we typically want to arrange them in a
    compact rectangle (rows × cols) that can fit inside a constrained region.

    Given:
      - n: desired number of placements,
      - max_cols: maximum allowable columns,
      - max_rows: maximum allowable rows,

    we return:
      - cols: chosen number of columns,
      - rows: chosen number of rows,
      - n_eff: effective number of agents that can fit (<= n)

    KEY DESIGN PRINCIPLES
    ---------------------
    1) Compactness:
       A near-square rectangle minimizes perimeter for a given area. A standard
       heuristic is to set cols ≈ sqrt(n), then compute rows accordingly.

    2) Constraint compliance:
       cols <= max_cols
       rows <= max_rows
       cols * rows <= max_cols * max_rows (implicitly)

    3) Safety:
       If n <= 0, return zeros.

    MATHEMATICS
    -----------
    - Start with a target column count near sqrt(n).
      sqrt(n) arises from minimizing (rows + cols) given rows * cols ≈ n.
    - Compute rows = ceil(n / cols) so that rows * cols >= n.
    - Clamp both cols and rows to the provided maxima.

    RETURNS
    -------
    (cols, rows, n_eff)
      n_eff = min(n, cols * rows) ensures we do not claim to place more agents
      than the rectangle can hold.

    NOTE ON INTEGER ARITHMETIC
    --------------------------
    - math.sqrt(n) yields float; int(...) truncates toward zero.
    - math.ceil(...) yields the smallest integer >= argument.
    """
    if n <= 0:
        return 0, 0, 0

    # Choose columns: at most max_cols, at least 1, close to sqrt(n).
    cols = min(max_cols, max(1, int(math.sqrt(n))))

    # Choose rows so that rows * cols can cover n placements.
    rows = min(max_rows, int(math.ceil(n / cols)))

    # Effective number of agents that can actually fit.
    n_eff = min(n, cols * rows)
    return cols, rows, n_eff

# ----------------------------------------------------------------------
# Team brain selection (supports exclusive split and mixed teams)
# ----------------------------------------------------------------------

# Deterministic per-team alternating counter (only used when mix+alternate).
_TEAM_BRAIN_MIX_COUNTER = {True: 0, False: 0}  # True=red, False=blue
# This module-level state enables deterministic "round robin" assignment:
#   - For each team independently:
#       0 -> tron
#       1 -> mirror
#       2 -> tron
#       3 -> mirror
#       ...
# IMPORTANT:
#   Because this is at module scope, it persists across calls to spawn functions.
#   This is intentional when you want globally consistent alternation.


def _make_team_mix_rng(team_is_red: bool):
    """
    Dedicated RNG so brain selection does NOT perturb world spawn RNG.
    If TEAM_BRAIN_MIX_SEED == 0 -> non-deterministic (SystemRandom).

    RATIONALE
    ---------
    In simulation code, coupling random streams can create subtle and unwanted
    correlations. For example, if brain selection consumes random numbers from
    the same RNG as coordinate sampling, changing brain selection strategy could
    change spawn locations, causing non-local behavioral differences.

    This function constructs a dedicated RNG stream for *brain assignment* so that:
      - random draws for brain architecture selection do not change the world RNG,
      - reproducibility is improved: one can vary brain mixing without changing
        spawn location patterns (assuming other RNG usage is separated similarly).

    CONFIGURATION
    -------------
    TEAM_BRAIN_MIX_SEED:
      - 0 => SystemRandom() (OS entropy; not reproducible)
      - non-zero => random.Random(seed + salt) (deterministic)

    SALTING
    -------
    A salt is added per team to ensure the red and blue teams do not share the
    same pseudo-random sequence when using a single base seed. This avoids
    mirrored brain selection patterns across teams.
    """
    seed = int(getattr(config, "TEAM_BRAIN_MIX_SEED", 0))
    if seed == 0:
        return random.SystemRandom()
    # Salt per team so red/blue don't mirror the same sequence
    salt = 101 if team_is_red else 202
    return random.Random(seed + salt)

_TEAM_BRAIN_MIX_RNG = {True: _make_team_mix_rng(True), False: _make_team_mix_rng(False)}
# Two RNGs: one for red, one for blue, keyed by team_is_red boolean.


def _resolve_team_brain_kind(team_is_red: bool) -> str:
    """
    Returns one of the configured MLP brain kinds.

    HIGH-LEVEL POLICY
    -----------------
    This function maps a team identity to a brain architecture name, governed by
    config-driven policies.

    Supported modes (TEAM_BRAIN_ASSIGNMENT_MODE):
      1) "exclusive" / "split" / "team"
         - Fixed architecture per team.

      2) "mix" / "hybrid" / "both"
         - Each team may spawn configured MLP variants according to a strategy:
             a) alternate / roundrobin / rr
             b) random / prob / probabilistic

    STRATEGIES
    ----------
    alternate:
      - deterministic cycling based on per-team counter and configured sequence

    probabilistic:
      - weighted random draw across the configured MLP family

    ERROR HANDLING
    --------------
    If an unknown mode or strategy is configured, a ValueError is raised. This is
    an intentionally "fail fast" policy: misconfiguration should surface early.
    """
    mode = str(getattr(config, "TEAM_BRAIN_ASSIGNMENT_MODE", "exclusive")).strip().lower()
    default_kind = str(getattr(config, "BRAIN_KIND", "whispering_abyss")).strip().lower()

    if mode in ("exclusive", "split", "team"):
        if team_is_red:
            return str(getattr(config, "TEAM_BRAIN_EXCLUSIVE_RED", default_kind)).strip().lower()
        return str(getattr(config, "TEAM_BRAIN_EXCLUSIVE_BLUE", default_kind)).strip().lower()

    if mode in ("mix", "hybrid", "both"):
        strategy = str(getattr(config, "TEAM_BRAIN_MIX_STRATEGY", "alternate")).strip().lower()
        seq = tuple(
            str(x).strip().lower()
            for x in getattr(config, "TEAM_BRAIN_MIX_SEQUENCE", (default_kind,))
            if str(x).strip()
        ) or (default_kind,)

        if strategy in ("alternate", "roundrobin", "rr"):
            i = _TEAM_BRAIN_MIX_COUNTER[team_is_red]
            _TEAM_BRAIN_MIX_COUNTER[team_is_red] = i + 1
            return seq[i % len(seq)]

        if strategy in ("random", "prob", "probabilistic"):
            r = _TEAM_BRAIN_MIX_RNG[team_is_red].random()
            weighted = (
                ("whispering_abyss", max(0.0, float(getattr(config, "TEAM_BRAIN_MIX_P_WHISPERING_ABYSS", 0.0)))),
                ("veil_of_echoes", max(0.0, float(getattr(config, "TEAM_BRAIN_MIX_P_VEIL_OF_ECHOES", 0.0)))),
                ("cathedral_of_ash", max(0.0, float(getattr(config, "TEAM_BRAIN_MIX_P_CATHEDRAL_OF_ASH", 0.0)))),
                ("dreamer_in_black_fog", max(0.0, float(getattr(config, "TEAM_BRAIN_MIX_P_DREAMER_IN_BLACK_FOG", 0.0)))),
                ("obsidian_pulse", max(0.0, float(getattr(config, "TEAM_BRAIN_MIX_P_OBSIDIAN_PULSE", 0.0)))),
            )
            total = sum(w for _, w in weighted)
            if total <= 0.0:
                return default_kind
            r *= total
            acc = 0.0
            for kind, w in weighted:
                acc += w
                if r <= acc:
                    return kind
            return weighted[-1][0]

        raise ValueError(f"Unknown TEAM_BRAIN_MIX_STRATEGY={strategy!r}")

    raise ValueError(f"Unknown TEAM_BRAIN_ASSIGNMENT_MODE={mode!r}")


def _mk_brain(device: torch.device, *, team_is_red: Optional[bool] = None) -> torch.nn.Module:
    """Creates a new brain.

    MODES OF OPERATION
    ------------------
    This function returns one of the configured MLP brain variants for both PPO
    and non-PPO contexts.

    CONFIGURATION LOGIC
    -------------------
    obs_dim = OBS_DIM in config (default 0)
    act_dim = NUM_ACTIONS in config (default 41)

    - If TEAM_BRAIN_ASSIGNMENT is enabled AND team_is_red is provided:
        choose by TEAM_BRAIN_ASSIGNMENT_MODE and strategy via _resolve_team_brain_kind.
    - Else:
        choose config.BRAIN_KIND (default "whispering_abyss").

    TYPE DISCUSSION
    ---------------
    - The return type is torch.nn.Module.
    - The brain is moved to the target device using .to(device) immediately.
      This ensures that later forward passes and parameter storage are consistent
      with the simulation's chosen hardware (CPU/GPU).

    WHY Optional[bool] for team_is_red?
    ----------------------------------
    Some contexts may not have a meaningful team assignment at brain creation time.
    In those cases, the code gracefully falls back to BRAIN_KIND.
    """
    obs_dim = int(getattr(config, "OBS_DIM", 0))
    act_dim = int(getattr(config, "NUM_ACTIONS", 41))

    team_assign = bool(getattr(config, "TEAM_BRAIN_ASSIGNMENT", True))
    if team_assign and team_is_red is not None:
        brain_kind = _resolve_team_brain_kind(bool(team_is_red))
    else:
        brain_kind = str(getattr(config, "BRAIN_KIND", "whispering_abyss")).strip().lower()

    return create_mlp_brain(brain_kind, obs_dim, act_dim).to(device)


def _choose_unit(is_archer_prob: float) -> float:
    # Choose between two unit types according to is_archer_prob.
    #
    # UNIT_ARCHER and UNIT_SOLDIER are read from config.
    # The function returns float(...) to match downstream storage conventions,
    # likely because agent/unit values are stored in float tensors.
    #
    # Bernoulli process:
    #   u ~ Uniform(0,1)
    #   if u < p => archer else soldier
    return float(config.UNIT_ARCHER if random.random() < is_archer_prob else config.UNIT_SOLDIER)


def _unit_stats(unit_val: float) -> Tuple[float, float, int]:
    """Returns (hp, atk, vision_range) for a given unit id.

    INPUT
    -----
    unit_val: float
      Semantically categorical, but stored as float.

    OUTPUT
    ------
    (hp, atk, vision)
      hp: float  - initial health (and used also as hp_max)
      atk: float - attack coefficient / damage parameter
      vision: int - perception range

    CONFIG DEPENDENCIES
    -------------------
    VISION_RANGE_BY_UNIT is expected to be a dict-like mapping unit_id -> range.
    Safe default is {} if not present.

    NOTE ON int(unit_val)
    ---------------------
    unit_val may be stored as 1.0/2.0. Converting to int is robust here.
    """
    vision_map = getattr(config, "VISION_RANGE_BY_UNIT", {})
    if int(unit_val) == int(config.UNIT_ARCHER):
        hp = float(config.ARCHER_HP)
        atk = float(config.ARCHER_ATK)
        vision = int(vision_map.get(config.UNIT_ARCHER, 15))
    else:
        hp = float(config.SOLDIER_HP)
        atk = float(config.SOLDIER_ATK)
        vision = int(vision_map.get(config.UNIT_SOLDIER, 10))
    return hp, atk, vision


def _place_if_free(
    reg: AgentsRegistry,
    grid: torch.Tensor,
    slot: int,
    *,
    team_is_red: bool,
    x: int,
    y: int,
    unit_val: float,
) -> bool:
    """Places an agent if the cell is free and registers it.

    CONTRACTUAL CONSTRAINTS
    -----------------------
    The correctness of spawning depends on consistent assumptions across:
      - grid representation,
      - registry representation,
      - perception / raycasting / movement logic elsewhere in the codebase.

    The comment in the function summarizes these critical contracts:

      - Grid layout in this project:
          channel0 = team_id / empty / wall encoding
          channel1 = hp
          channel2 = slot

      - AgentsRegistry.register requires:
          slot + agent_id + vision_range keyword arguments.

    CELL OCCUPANCY CHECK
    --------------------
    grid[0, y, x] != 0.0 is considered occupied, meaning:
      - walls are non-zero,
      - agents are non-zero (team IDs),
      - any other non-zero sentinel values also block placement.

    This is a very common pattern in grid sims:
      channel0 is effectively a "solid occupancy" code.

    REGISTRATION PROCESS
    --------------------
    If free:
      1) compute stats based on unit type,
      2) allocate unique agent_id,
      3) call reg.register(...) to populate registry and attach brain,
      4) update grid channels to match the newly placed agent.

    RETURNS
    -------
    True if placement succeeded, False otherwise.
    """
    # occupied if channel0 != 0 (either team id or wall encoding)
    if grid[0, y, x] != 0.0:
        return False

    hp, atk, vision = _unit_stats(unit_val)

    # Unique identifier for the agent (distinct from slot index).
    # Slot is a registry index (0..capacity-1), agent_id is a logical UID.
    agent_id = reg.get_next_id()

    # Register the agent into the registry.
    # Note: generation=1 indicates a baseline or initial cohort spawn.
    reg.register(
        slot,
        agent_id=agent_id,
        team_is_red=team_is_red,
        x=x,
        y=y,
        hp=hp,
        atk=atk,
        brain=_mk_brain(reg.device, team_is_red=team_is_red),
        unit=unit_val,
        hp_max=hp,
        vision_range=vision,
        generation=1,
    )

    # Update grid:
    #   - channel0: occupancy/team encoding (2.0 for red, 3.0 for blue)
    #   - channel1: hp
    #   - channel2: registry slot index
    grid[0, y, x] = 2.0 if team_is_red else 3.0
    grid[1, y, x] = hp
    grid[2, y, x] = float(slot)
    return True


def spawn_symmetric(reg: AgentsRegistry, grid: torch.Tensor, per_team: int) -> None:
    """Spawns agents in symmetric rectangular formations on opposite sides.

    GOAL
    ----
    Create two mirrored formations:
      - Red team on the left half of the map
      - Blue team on the right half of the map
    centered vertically, with a margin from borders.

    WHY SYMMETRIC SPAWNING?
    -----------------------
    Symmetric initialization is frequently used in:
      - competitive self-play,
      - fairness-sensitive benchmarks,
      - debugging and visualization,
    because it removes initial positional advantage.

    SPATIAL COMPUTATION
    -------------------
    H, W:
      grid.size(1) and grid.size(2) give height and width.
      (grid is channel-first: [C, H, W])

    margin:
      a fixed buffer to avoid spawning on borders.

    half_w:
      W // 2 divides the map into left and right halves.

    placeable_w:
      width available on one side = half_w - margin.
      (Note: right side also uses margin; blue_x0 accounts for this.)

    placeable_h:
      height available = H - 2*margin.

    CAPACITY LIMITS
    ---------------
    per_team_eff is constrained by:
      - per_team requested,
      - registry capacity // 2 (reserve half slots per team in symmetric mode),
      - spatial capacity placeable_w * placeable_h.
    """
    H, W = grid.size(1), grid.size(2)
    margin = 2
    half_w = W // 2
    placeable_w = half_w - margin
    placeable_h = H - 2 * margin

    per_team_eff = min(per_team, reg.capacity // 2, placeable_w * placeable_h)
    if per_team_eff <= 0:
        return

    ar_ratio = float(getattr(config, "SPAWN_ARCHER_RATIO", 0.4))

    # Red team (left)
    r_cols, r_rows, r_n = _rect_dims(per_team_eff, placeable_w, placeable_h)
    red_x0, red_y0 = margin, (H - r_rows) // 2
    # red_x0 begins at left margin; red_y0 centers vertically.

    # Blue team (right)
    b_cols, b_rows, b_n = _rect_dims(per_team_eff, placeable_w, placeable_h)
    blue_x0, blue_y0 = W - margin - b_cols, (H - b_rows) // 2
    # blue_x0 positions the rectangle so it ends at right margin.
    # blue_y0 centers vertically.

    slot = 0

    # Place Red
    for iy in range(r_rows):
        for ix in range(r_cols):
            # Stop if we have placed all intended red agents or used up capacity.
            if slot >= r_n or slot >= reg.capacity:
                break
            x, y = red_x0 + ix, red_y0 + iy
            unit = _choose_unit(ar_ratio)
            if _place_if_free(reg, grid, slot, team_is_red=True, x=x, y=y, unit_val=unit):
                slot += 1
        if slot >= r_n or slot >= reg.capacity:
            break

    # Place Blue
    blue_start_slot = slot
    for iy in range(b_rows):
        for ix in range(b_cols):
            # For blue, we stop when we have placed b_n additional agents
            # (relative to blue_start_slot) or hit capacity.
            if slot >= blue_start_slot + b_n or slot >= reg.capacity:
                break
            x, y = blue_x0 + ix, blue_y0 + iy
            unit = _choose_unit(ar_ratio)
            if _place_if_free(reg, grid, slot, team_is_red=False, x=x, y=y, unit_val=unit):
                slot += 1
        if slot >= blue_start_slot + b_n or slot >= reg.capacity:
            break


def spawn_uniform_random(reg: AgentsRegistry, grid: torch.Tensor, per_team: int) -> None:
    """Spawns agents for both teams randomly across the entire map.

    GOAL
    ----
    Place agents across the map using uniform random sampling of coordinates.

    ADVANTAGES
    ----------
    - Produces diverse initial conditions for training.
    - Stress-tests collision, movement, perception under varied densities.
    - Removes formation bias.

    CAPACITY + COUNT POLICY
    -----------------------
    total_to_spawn = min(per_team * 2, reg.capacity)
      - spawn at most two teams worth, but never exceed registry capacity.

    red_to_spawn:
      - target count for red (bounded by per_team and total_to_spawn)

    blue_to_spawn:
      - remainder so that total equals total_to_spawn.

    FAILURE HANDLING
    ----------------
    Rejection sampling may fail if the map is too full.
    The code sets:
      max_attempts = total_to_spawn * 50
    and warns if it cannot place all requested agents.

    FAIRNESS AND BIAS CONTROL
    -------------------------
    When both teams still need spawns, the code chooses which team to attempt
    first using probability proportional to remaining need:

        P(spawn_red_first) = red_to_spawn / (red_to_spawn + blue_to_spawn)

    This adaptive bias prevents one team from dominating placements simply
    due to ordering effects.

    NOTE ON SLOT ASSIGNMENT
    -----------------------
    slot increments only when placement succeeds. Thus:
      - slot indicates "number of successfully spawned agents so far".
      - final slot count is used for reporting partial success.
    """
    H, W = grid.size(1), grid.size(2)
    margin = 2
    ar_ratio = float(getattr(config, "SPAWN_ARCHER_RATIO", 0.4))

    total_to_spawn = min(per_team * 2, reg.capacity)
    red_to_spawn = min(per_team, total_to_spawn)
    blue_to_spawn = total_to_spawn - red_to_spawn

    attempts = 0
    max_attempts = total_to_spawn * 50
    slot = 0

    while (red_to_spawn > 0 or blue_to_spawn > 0) and attempts < max_attempts and slot < total_to_spawn:
        # Sample a coordinate uniformly from the interior (excluding margin).
        x = random.randint(margin, W - margin - 1)
        y = random.randint(margin, H - margin - 1)

        # Only consider truly empty cells (channel0 == 0.0).
        if grid[0, y, x] == 0.0:
            team_placed = False

            # Decide which team to try first (bias toward the team that still needs more).
            if red_to_spawn > 0 and blue_to_spawn > 0:
                # Probability proportional to remaining demand.
                spawn_red = (random.random() < (red_to_spawn / (red_to_spawn + blue_to_spawn)))
            else:
                # If only one team remains, choose that team deterministically.
                spawn_red = (red_to_spawn > 0)

            # Attempt placement for the chosen team first.
            if spawn_red and red_to_spawn > 0:
                unit = _choose_unit(ar_ratio)
                if _place_if_free(reg, grid, slot, team_is_red=True, x=x, y=y, unit_val=unit):
                    slot += 1
                    red_to_spawn -= 1
                    team_placed = True
            elif (not spawn_red) and blue_to_spawn > 0:
                unit = _choose_unit(ar_ratio)
                if _place_if_free(reg, grid, slot, team_is_red=False, x=x, y=y, unit_val=unit):
                    slot += 1
                    blue_to_spawn -= 1
                    team_placed = True

            # If first attempt failed (rare), try the other team once.
            #
            # This compensates for any subtle mismatches between:
            #   - our initial "cell free" check (grid[0]==0)
            #   - _place_if_free internal checks
            #
            # It also handles edge cases where registration could fail for reasons
            # other than occupancy (e.g., registry constraints), though none are
            # shown explicitly here.
            if not team_placed:
                if spawn_red and blue_to_spawn > 0:
                    unit = _choose_unit(ar_ratio)
                    if _place_if_free(reg, grid, slot, team_is_red=False, x=x, y=y, unit_val=unit):
                        slot += 1
                        blue_to_spawn -= 1
                elif (not spawn_red) and red_to_spawn > 0:
                    unit = _choose_unit(ar_ratio)
                    if _place_if_free(reg, grid, slot, team_is_red=True, x=x, y=y, unit_val=unit):
                        slot += 1
                        red_to_spawn -= 1

        # Count every attempted coordinate sample, whether or not it was usable.
        attempts += 1

    # If we could not spawn the full requested population, warn loudly.
    # This is preferable to silently under-spawning, which could confuse training.
    if slot < total_to_spawn:
        print(f"[spawn] Warning: Could only spawn {slot}/{total_to_spawn} agents. The map might be too full.")
