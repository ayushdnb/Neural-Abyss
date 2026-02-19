from __future__ import annotations

import math
import random
from typing import Optional, Tuple

import torch
import config
from .agent_registry import AgentsRegistry

# Brains
from agent.transformer_brain import TransformerBrain, scripted_transformer_brain
from agent.tron_brain import TronBrain
from agent.mirror_brain import MirrorBrain


def _rect_dims(n: int, max_cols: int, max_rows: int) -> Tuple[int, int, int]:
    """Calculates dimensions for a compact rectangle to place n agents."""
    if n <= 0:
        return 0, 0, 0
    cols = min(max_cols, max(1, int(math.sqrt(n))))
    rows = min(max_rows, int(math.ceil(n / cols)))
    n_eff = min(n, cols * rows)
    return cols, rows, n_eff

# ----------------------------------------------------------------------
# Team brain selection (supports exclusive split and mixed teams)
# ----------------------------------------------------------------------

# Deterministic per-team alternating counter (only used when mix+alternate).
_TEAM_BRAIN_MIX_COUNTER = {True: 0, False: 0}  # True=red, False=blue

def _make_team_mix_rng(team_is_red: bool):
    """
    Dedicated RNG so brain selection does NOT perturb world spawn RNG.
    If TEAM_BRAIN_MIX_SEED == 0 -> non-deterministic (SystemRandom).
    """
    seed = int(getattr(config, "TEAM_BRAIN_MIX_SEED", 0))
    if seed == 0:
        return random.SystemRandom()
    # Salt per team so red/blue don't mirror the same sequence
    salt = 101 if team_is_red else 202
    return random.Random(seed + salt)

_TEAM_BRAIN_MIX_RNG = {True: _make_team_mix_rng(True), False: _make_team_mix_rng(False)}

def _resolve_team_brain_kind(team_is_red: bool) -> str:
    """
    Returns: "tron" | "mirror" | "transformer"
    Default behavior remains: exclusive split (red=tron, blue=mirror).
    """
    mode = str(getattr(config, "TEAM_BRAIN_ASSIGNMENT_MODE", "exclusive")).strip().lower()

    # Old behavior
    if mode in ("exclusive", "split", "team"):
        return "tron" if team_is_red else "mirror"

    if mode in ("mix", "hybrid", "both"):
        strategy = str(getattr(config, "TEAM_BRAIN_MIX_STRATEGY", "alternate")).strip().lower()

        # Deterministic 50/50: tron, mirror, tron, mirror...
        if strategy in ("alternate", "roundrobin", "rr"):
            i = _TEAM_BRAIN_MIX_COUNTER[team_is_red]
            _TEAM_BRAIN_MIX_COUNTER[team_is_red] = i + 1
            return "tron" if (i % 2 == 0) else "mirror"

        # Probabilistic: P(tron)=TEAM_BRAIN_MIX_P_TRON
        if strategy in ("random", "prob", "probabilistic"):
            p_tron = float(getattr(config, "TEAM_BRAIN_MIX_P_TRON", 0.5))
            p_tron = max(0.0, min(1.0, p_tron))
            r = _TEAM_BRAIN_MIX_RNG[team_is_red].random()
            return "tron" if (r < p_tron) else "mirror"

        raise ValueError(f"Unknown TEAM_BRAIN_MIX_STRATEGY={strategy!r}")

    raise ValueError(f"Unknown TEAM_BRAIN_ASSIGNMENT_MODE={mode!r}")

def _mk_brain(device: torch.device, *, team_is_red: Optional[bool] = None) -> torch.nn.Module:
    """Creates a new brain.

    Non-PPO: scripted transformer (existing behavior).
    PPO:
      - If TEAM_BRAIN_ASSIGNMENT is enabled and team_is_red is known:
          uses TEAM_BRAIN_ASSIGNMENT_MODE:
            * exclusive: red=tron, blue=mirror (old behavior)
            * mix: each team can spawn both (alternate/random)
      - Otherwise: falls back to config.BRAIN_KIND.
    """
    obs_dim = int(getattr(config, "OBS_DIM", 0))
    act_dim = int(getattr(config, "NUM_ACTIONS", 41))

    is_ppo = bool(getattr(config, "PPO_ENABLED", False))
    if not is_ppo:
        return scripted_transformer_brain(obs_dim, act_dim).to(device)

    team_assign = bool(getattr(config, "TEAM_BRAIN_ASSIGNMENT", True))
    if team_assign and team_is_red is not None:
        brain_kind = _resolve_team_brain_kind(bool(team_is_red))
    else:
        brain_kind = str(getattr(config, "BRAIN_KIND", "tron")).strip().lower()

    if brain_kind == "transformer":
        return TransformerBrain(obs_dim, act_dim).to(device)
    if brain_kind == "mirror":
        return MirrorBrain(obs_dim, act_dim).to(device)
    return TronBrain(obs_dim, act_dim).to(device)


def _choose_unit(is_archer_prob: float) -> float:
    return float(config.UNIT_ARCHER if random.random() < is_archer_prob else config.UNIT_SOLDIER)


def _unit_stats(unit_val: float) -> Tuple[float, float, int]:
    """Returns (hp, atk, vision_range) for a given unit id."""
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

    IMPORTANT:
      - Grid layout in this project: channel0=team_id/empty, channel1=hp, channel2=slot
      - AgentsRegistry.register requires slot + agent_id + vision_range keyword.
    """
    # occupied if channel0 != 0 (either team id or wall encoding)
    if grid[0, y, x] != 0.0:
        return False

    hp, atk, vision = _unit_stats(unit_val)

    agent_id = reg.get_next_id()

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

    # Update grid
    grid[0, y, x] = 2.0 if team_is_red else 3.0
    grid[1, y, x] = hp
    grid[2, y, x] = float(slot)
    return True


def spawn_symmetric(reg: AgentsRegistry, grid: torch.Tensor, per_team: int) -> None:
    """Spawns agents in symmetric rectangular formations on opposite sides."""
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

    # Blue team (right)
    b_cols, b_rows, b_n = _rect_dims(per_team_eff, placeable_w, placeable_h)
    blue_x0, blue_y0 = W - margin - b_cols, (H - b_rows) // 2

    slot = 0

    # Place Red
    for iy in range(r_rows):
        for ix in range(r_cols):
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
            if slot >= blue_start_slot + b_n or slot >= reg.capacity:
                break
            x, y = blue_x0 + ix, blue_y0 + iy
            unit = _choose_unit(ar_ratio)
            if _place_if_free(reg, grid, slot, team_is_red=False, x=x, y=y, unit_val=unit):
                slot += 1
        if slot >= blue_start_slot + b_n or slot >= reg.capacity:
            break


def spawn_uniform_random(reg: AgentsRegistry, grid: torch.Tensor, per_team: int) -> None:
    """Spawns agents for both teams randomly across the entire map."""
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
        x = random.randint(margin, W - margin - 1)
        y = random.randint(margin, H - margin - 1)

        if grid[0, y, x] == 0.0:
            team_placed = False

            # Decide which team to try first (bias toward the team that still needs more)
            if red_to_spawn > 0 and blue_to_spawn > 0:
                spawn_red = (random.random() < (red_to_spawn / (red_to_spawn + blue_to_spawn)))
            else:
                spawn_red = (red_to_spawn > 0)

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

            # If first attempt failed (rare), try the other team once
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

        attempts += 1

    if slot < total_to_spawn:
        print(f"[spawn] Warning: Could only spawn {slot}/{total_to_spawn} agents. The map might be too full.")
