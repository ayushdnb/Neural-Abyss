"""Respawn policy and slot reactivation logic."""

from __future__ import annotations

from dataclasses import dataclass, field
# dataclass:
#   - Provides automatic generation of __init__, __repr__, __eq__, etc.
#   - Encourages "configuration-as-data" patterns (clear, explicit fields).
# field(default_factory=...):
#   - Ensures a new default is computed at instantiation time, not at import time.
#   - Crucial when defaults depend on config values that may differ per run.

from typing import Optional, Tuple, List, Dict, Any
# Optional[T] means T or None
# Tuple[A, B] indicates fixed-size tuple with (A, B)
# List[T] and Dict[K, V] for container typing (readability + static checking)

import copy
import math
import random

import torch
import torch.jit
# torch.jit:
#   - Enables TorchScript compilation / scripting.
#   - ScriptModule is the runtime type for scripted modules.
# In this file, TorchScript awareness is used primarily to infer brain type.

import config
# Global configuration object/module.
# This file repeatedly uses getattr(config, "...", default) to:
#   - preserve backward compatibility if config lacks a field,
#   - allow optional knobs,
#   - avoid crashing when config is incomplete.

from .agent_registry import (
    AgentsRegistry,
    COL_ALIVE,
    COL_TEAM,
    COL_X,
    COL_Y,
    COL_HP,
    COL_HP_MAX,
    COL_VISION,
    COL_ATK,
    COL_UNIT,
    COL_AGENT_ID,
    TEAM_RED_ID,
    TEAM_BLUE_ID,
)
# This import supplies:
#   - AgentsRegistry: core state container for agents (tensor storage + brains list).
#   - Column indices (COL_*) that index into reg.agent_data.
#   - TEAM_* constants: numeric IDs used as team labels in tensors & grid occupancy.
# A key design pattern here is "struct-of-arrays":
#   - agent_data is a 2D tensor of shape (N_agents, N_fields)
#   - each field is a column (indexed by COL_* constants)
# This is GPU-friendly and vectorizable.

from agent.mlp_brain import (
    create_mlp_brain,
    normalize_brain_kind,
    brain_kind_from_module,
)
# "Brains" are policy/value networks or decision modules controlling agents.
# The code supports multiple architectures:
#   - TransformerBrain (learnable, not necessarily TorchScript)
#   - scripted_transformer_brain (TorchScript-ready version for non-PPO mode)
#   - TronBrain, MirrorBrain (likely lighter MLP-based / heuristic / special policies)
# The policy selection logic is team-aware when TEAM_BRAIN_ASSIGNMENT is enabled.

_BIRTH_DOCTRINES = frozenset({
    "overall",
    "killer",
    "cp",
    "health",
    "kill_health",
    "health_cp",
    "kill_cp",
    "trinity",
    "highest_spike",
    "personal_points",
    "random_per_birth",
})


# Respawn configuration dataclass (extended with new parameters)
@dataclass
class RespawnCfg:
    """Configuration for agent respawning.

    DESIGN INTENT
    -------------
    This configuration object defines "what respawn should do" rather than
    embedding constants directly into logic. This approach yields:

      1) Reproducibility:
         - One can record or version the configuration for experiments.

      2) Backward compatibility:
         - Legacy fields remain, even if not used by the new controller.

      3) Separation of concerns:
         - Controller logic is parameterized; changes in behavior can be driven
           by config changes rather than code edits.

    COMPATIBILITY STRATEGY
    ----------------------
    For many fields, defaults are pulled from `config` via getattr(..., default).
    This ensures:
      - If a new config knob does not exist, a safe default is used.
      - Older config.py files still run without modification.

    NOTE ON TYPES
    -------------
    Fields are typed with bool / int / float for clarity and tooling.
    At runtime, the implementation frequently casts config values using int(...)
    or float(...), which ensures predictable types even if config values are
    accidentally provided as strings or NumPy scalars.
    """

    # Master switch
    enabled: bool = True
    # If enabled is False, the controller performs no respawn actions.

    # ---------- Legacy simple probabilistic fields ----------
    prob_per_dead_per_tick: float = 0.05
    # Legacy knob: originally would spawn each dead agent with some probability.
    # In the new controller logic, this is explicitly "not used" but is kept so:
    #   - existing configs do not break,
    #   - external callers relying on this field do not fail attribute access.

    spawn_tries: int = 200
    # Maximum attempts to sample a free spawn cell. This is used by the location
    # picker. If too low, spawns may fail frequently in crowded maps.
    # If too high, worst-case CPU time per spawn increases.

    mutation_std: float = 0.02
    # Standard deviation of Gaussian noise added to cloned brain parameters.
    # If parameters θ are mutated in-place, typical perturbation is:
    #     θ ← θ + ε,   where ε ~ Normal(0, σ^2)
    # "std" is σ (sigma). Larger σ increases exploration but may destabilize.

    clone_prob: float = 0.50
    # Probability of cloning an existing alive agent (parent) rather than creating
    # a fresh brain. This implements a simplistic evolutionary mechanism:
    #   - cloning preserves successful policies,
    #   - mutation introduces variation.

    use_team_elite: bool = True
    # Whether to consider only alive agents from the same team as parents.
    # NOTE: In this code, parents are already filtered by same team; this flag is
    # retained but not actively referenced in the selection logic shown.

    reset_optimizer_on_respawn: bool = True
    # In PPO training pipelines, each agent might have an optimizer state.
    # Resetting avoids mixing momentum/Adam statistics across distinct individuals.
    # Here, the code sets reg.optimizers[slot] = None (if optimizers exist),
    # which implies the training loop will recreate optimizers when needed.

    # ---------- New controller fields ----------
    floor_per_team: int = field(default_factory=lambda: int(getattr(config, "RESP_FLOOR_PER_TEAM", 50)))
    # Minimum number of alive agents per team. If alive count falls below this
    # floor, the controller will attempt to respawn until floor is reached (subject
    # to caps and cooldown).

    max_per_tick: int = field(default_factory=lambda: int(getattr(config, "RESP_MAX_PER_TICK", 5)))
    # Hard cap on how many agents per team may be spawned in a single tick.
    # Prevents sudden bursts that:
    #   - cause large compute spikes,
    #   - dramatically change environment density in one step.

    period_ticks: int = field(default_factory=lambda: int(getattr(config, "RESP_PERIOD_TICKS", 500)))
    # The controller has a periodic "budget distribution" mechanism. Every
    # period_ticks, it allocates a number of respawns across teams based on their
    # current alive counts (inverse proportional split).

    period_budget: int = field(default_factory=lambda: int(getattr(config, "RESP_PERIOD_BUDGET", 20)))
    # How many respawns total (across teams) to allocate each time a period elapses.
    # The split across teams is computed by _inverse_split(...).

    cooldown_ticks: int = field(default_factory=lambda: int(getattr(config, "RESP_HYST_COOLDOWN_TICKS", 30)))
    # A hysteresis mechanism: once a team climbs back to the floor, the controller
    # waits cooldown_ticks before applying floor-based spawning again.
    # This reduces oscillations:
    #   - without cooldown, small fluctuations around the floor could trigger
    #     respawn almost every tick.

    wall_margin: int = field(default_factory=lambda: int(getattr(config, "RESP_WALL_MARGIN", 2)))
    # Spawn positions must be at least wall_margin cells away from the borders.
    # This prevents immediate wall collisions or degenerate behavior near edges.

    # Unit type configuration (pulled from config)
    unit_soldier: int = field(default_factory=lambda: int(getattr(config, "UNIT_SOLDIER", 1)))
    unit_archer: int = field(default_factory=lambda: int(getattr(config, "UNIT_ARCHER", 2)))

    spawn_archer_ratio: float = field(default_factory=lambda: float(getattr(config, "SPAWN_ARCHER_RATIO", 0.40)))
    # Probability of spawning an archer rather than a soldier.

    soldier_hp: float = field(default_factory=lambda: float(getattr(config, "SOLDIER_HP", 1.0)))
    soldier_atk: float = field(default_factory=lambda: float(getattr(config, "SOLDIER_ATK", 0.05)))
    archer_hp: float = field(default_factory=lambda: float(getattr(config, "ARCHER_HP", 1.0)))
    archer_atk: float = field(default_factory=lambda: float(getattr(config, "ARCHER_ATK", 0.02)))

    vision_soldier: int = field(default_factory=lambda: int(getattr(config, "VISION_RANGE_BY_UNIT", {}).get(1, 10)))
    vision_archer: int = field(default_factory=lambda: int(getattr(config, "VISION_RANGE_BY_UNIT", {}).get(2, 15)))
    # Vision ranges come from a dict in config (VISION_RANGE_BY_UNIT) if present.
    # If config does not define it, fall back to reasonable defaults.

    # ---------- Optional evolution layer knobs (all backward-compatible) ----------
    rare_mutation_tick_window_enable: bool = field(default_factory=lambda: bool(getattr(config, "RESPAWN_RARE_MUTATION_TICK_WINDOW_ENABLE", False)))
    rare_mutation_tick_window_ticks: int = field(default_factory=lambda: int(getattr(config, "RESPAWN_RARE_MUTATION_TICK_WINDOW_TICKS", 1000)))
    rare_mutation_physical_enable: bool = field(default_factory=lambda: bool(getattr(config, "RESPAWN_RARE_MUTATION_PHYSICAL_ENABLE", False)))
    rare_mutation_physical_drift_std_frac: float = field(default_factory=lambda: float(getattr(config, "RESPAWN_RARE_MUTATION_PHYSICAL_DRIFT_STD_FRAC", 0.03)))
    rare_mutation_physical_drift_clip_frac: float = field(default_factory=lambda: float(getattr(config, "RESPAWN_RARE_MUTATION_PHYSICAL_DRIFT_CLIP_FRAC", 0.10)))
    rare_mutation_inherited_brain_noise_enable: bool = field(default_factory=lambda: bool(getattr(config, "RESPAWN_RARE_MUTATION_INHERITED_BRAIN_NOISE_ENABLE", False)))
    rare_mutation_inherited_brain_noise_std: float = field(default_factory=lambda: float(getattr(config, "RESPAWN_RARE_MUTATION_INHERITED_BRAIN_NOISE_STD", 0.20)))

    parent_selection_mode: str = field(default_factory=lambda: str(getattr(config, "RESPAWN_PARENT_SELECTION_MODE", "topk_weighted")).strip().lower())
    parent_selection_topk_frac: float = field(default_factory=lambda: float(getattr(config, "RESPAWN_PARENT_SELECTION_TOPK_FRAC", 0.25)))
    parent_selection_score_power: float = field(default_factory=lambda: float(getattr(config, "RESPAWN_PARENT_SELECTION_SCORE_POWER", 1.0)))
    require_parent_for_birth: bool = field(default_factory=lambda: bool(getattr(config, "RESPAWN_REQUIRE_PARENT_FOR_BIRTH", True)))
    birth_doctrine_mode: str = field(default_factory=lambda: str(getattr(config, "RESPAWN_BIRTH_DOCTRINE_MODE", "overall")).strip().lower())
    birth_random_doctrine_pool: Tuple[str, ...] = field(default_factory=lambda: tuple(
        str(x).strip().lower()
        for x in getattr(config, "RESPAWN_BIRTH_RANDOM_DOCTRINE_POOL", ("overall",))
        if str(x).strip()
    ))
    birth_topk_size: int = field(default_factory=lambda: int(getattr(config, "RESPAWN_BIRTH_TOPK_SIZE", 0)))
    birth_zero_score_fallback: str = field(default_factory=lambda: str(
        getattr(config, "RESPAWN_BIRTH_ZERO_SCORE_FALLBACK", "uniform_candidates")
    ).strip().lower())
    birth_blend_weight_kill: float = field(default_factory=lambda: float(getattr(config, "RESPAWN_BIRTH_BLEND_WEIGHT_KILL", 1.0)))
    birth_blend_weight_cp: float = field(default_factory=lambda: float(getattr(config, "RESPAWN_BIRTH_BLEND_WEIGHT_CP", 1.0)))
    birth_blend_weight_health: float = field(default_factory=lambda: float(getattr(config, "RESPAWN_BIRTH_BLEND_WEIGHT_HEALTH", 1.0)))
    birth_blend_weight_personal: float = field(default_factory=lambda: float(getattr(config, "RESPAWN_BIRTH_BLEND_WEIGHT_PERSONAL", 1.0)))

    spawn_location_mode: str = field(default_factory=lambda: str(getattr(config, "RESPAWN_SPAWN_LOCATION_MODE", "uniform")))
    spawn_near_parent_radius: int = field(default_factory=lambda: int(getattr(config, "RESPAWN_SPAWN_NEAR_PARENT_RADIUS", max(1, int(getattr(config, "RESPAWN_JITTER_RADIUS", 1))))))
    child_unit_mode: str = field(
        default_factory=lambda: str(
            getattr(config, "RESPAWN_CHILD_UNIT_MODE", "inherit_parent_on_clone")
        ).strip().lower()
    )


# Helper functions for team counts and distribution
@dataclass
class TeamCounts:
    """Simple container for alive agent counts per team."""
    red: int
    blue: int


def _team_counts(reg: AgentsRegistry) -> TeamCounts:
    """Return the number of alive agents for each team.

    METHOD
    ------
    - Reads reg.agent_data (a tensor containing per-agent attributes).
    - Defines "alive" as COL_ALIVE > 0.5, implying COL_ALIVE is stored as float
      (0.0 or 1.0) rather than boolean.
    - Counts alive agents where COL_TEAM equals TEAM_RED_ID or TEAM_BLUE_ID.

    PERFORMANCE
    -----------
    This uses vectorized tensor operations:
      alive = (d[:, COL_ALIVE] > 0.5)
      red   = sum(alive & team_is_red)
    which is significantly faster than Python loops, especially for large N.
    """
    d = reg.agent_data
    alive = (d[:, COL_ALIVE] > 0.5)

    # alive & (team == red) yields a boolean tensor.
    # sum().item() converts tensor -> Python number.
    red = int((alive & (d[:, COL_TEAM] == TEAM_RED_ID)).sum().item())
    blue = int((alive & (d[:, COL_TEAM] == TEAM_BLUE_ID)).sum().item())
    return TeamCounts(red=red, blue=blue)


def _inverse_split(a: int, b: int, budget: int) -> Tuple[int, int]:
    """
    Split a budget inversely proportional to the two numbers.
    Used to give more respawns to the team with fewer alive agents.

    MATHEMATICAL FORMULATION
    ------------------------
    Let a = alive count for team A, b = alive count for team B.
    We define weights inversely proportional to counts:

        w_a = 1/a
        w_b = 1/b

    Total weight:

        S = w_a + w_b = 1/a + 1/b

    Allocate budget to team A as:

        q_a = round( budget * w_a / S )

    And team B gets the remainder:

        q_b = budget - q_a

    PROPERTIES
    ----------
    - If a < b then 1/a > 1/b, so q_a tends to be larger.
    - If a == b then q_a ≈ budget/2.
    - The rounding step ensures q_a is integer.
    - Using "remainder" ensures q_a + q_b == budget exactly.

    SAFETY
    ------
    The code clamps a and b to at least 1 to avoid division by zero.
    """
    a = max(1, a)
    b = max(1, b)
    s = 1.0 / a + 1.0 / b
    qa = int(round(budget * (1.0 / a) / s))
    return qa, budget - qa


def _cap(n: int, cfg: RespawnCfg) -> int:
    """Clamp a desired respawn count to the per-tick maximum.

    This is a small but important control:
      - n may be very large if a team collapses.
      - cfg.max_per_tick bounds the immediate impact per tick.

    Returned value is in [0, cfg.max_per_tick].
    """
    return max(0, min(n, cfg.max_per_tick))


# Team brain selection (supports exclusive split and mixed teams)

_TEAM_BRAIN_MIX_COUNTER: Dict[float, int] = {}
# Per-team counter used for deterministic alternation strategies.
# Because it is module-level, it persists across controller instances.

def _make_team_mix_rng(team_id: float):
    # This function constructs a random number generator (RNG) used when the
    # team brain assignment mode uses probabilistic mixing.
    # If TEAM_BRAIN_MIX_SEED == 0:
    #   - Use SystemRandom(), which draws entropy from the OS.
    #   - This is non-deterministic across runs (not reproducible).
    # Otherwise:
    #   - Use random.Random(seed + salt) for deterministic behavior.
    #   - "salt" ensures red and blue do not share identical RNG streams even if
    #     base seed is the same.
    seed = int(getattr(config, "TEAM_BRAIN_MIX_SEED", 0))
    if seed == 0:
        return random.SystemRandom()
    salt = 101 if team_id == TEAM_RED_ID else 202
    return random.Random(seed + salt)

_TEAM_BRAIN_MIX_RNG: Dict[float, random.Random] = {}
# Dictionary mapping team_id -> RNG instance.


def reset_team_brain_runtime_state() -> None:
    """Rebuild the per-team brain-mix counters and RNG streams from config."""
    _TEAM_BRAIN_MIX_COUNTER.clear()
    _TEAM_BRAIN_MIX_COUNTER.update({
        TEAM_RED_ID: 0,
        TEAM_BLUE_ID: 0,
    })
    _TEAM_BRAIN_MIX_RNG.clear()
    _TEAM_BRAIN_MIX_RNG.update({
        TEAM_RED_ID: _make_team_mix_rng(TEAM_RED_ID),
        TEAM_BLUE_ID: _make_team_mix_rng(TEAM_BLUE_ID),
    })


def export_team_brain_runtime_state() -> Dict[str, Any]:
    """Return checkpoint-safe state for mixed-brain assignment continuity."""
    state: Dict[str, Any] = {
        "counter": {},
        "rng_state": {},
    }
    for team_id, label in ((TEAM_RED_ID, "red"), (TEAM_BLUE_ID, "blue")):
        state["counter"][label] = int(_TEAM_BRAIN_MIX_COUNTER.get(team_id, 0))
        rng = _TEAM_BRAIN_MIX_RNG.get(team_id)
        if rng is None or isinstance(rng, random.SystemRandom):
            state["rng_state"][label] = None
        else:
            state["rng_state"][label] = rng.getstate()
    return state


def set_team_brain_runtime_state(state: Optional[Dict[str, Any]]) -> None:
    """Restore mixed-brain assignment counters/RNG state from a checkpoint."""
    if not state:
        reset_team_brain_runtime_state()
        return

    counters = state.get("counter", {}) if isinstance(state, dict) else {}
    rng_states = state.get("rng_state", {}) if isinstance(state, dict) else {}

    _TEAM_BRAIN_MIX_COUNTER.clear()
    _TEAM_BRAIN_MIX_RNG.clear()
    for team_id, label in ((TEAM_RED_ID, "red"), (TEAM_BLUE_ID, "blue")):
        _TEAM_BRAIN_MIX_COUNTER[team_id] = int(counters.get(label, 0))
        rng_state = rng_states.get(label)
        if rng_state is None:
            _TEAM_BRAIN_MIX_RNG[team_id] = _make_team_mix_rng(team_id)
            continue
        rng = random.Random()
        try:
            rng.setstate(rng_state)
        except Exception:
            rng = _make_team_mix_rng(team_id)
        _TEAM_BRAIN_MIX_RNG[team_id] = rng


reset_team_brain_runtime_state()

def _resolve_team_brain_kind_from_team(team_id: float) -> str:
    # Determine the configured brain kind for a team.
    # TEAM_BRAIN_ASSIGNMENT_MODE controls fixed-per-team versus mixed-per-spawn
    # selection. TEAM_BRAIN_MIX_STRATEGY controls alternate vs weighted-random
    # selection inside mixed mode. Unknown team ids fall back to config.BRAIN_KIND.
    mode = str(getattr(config, "TEAM_BRAIN_ASSIGNMENT_MODE", "exclusive")).strip().lower()
    default_kind = normalize_brain_kind(
        str(getattr(config, "BRAIN_KIND", "throne_of_ashen_dreams")).strip().lower()
    )

    if mode in ("exclusive", "split", "team"):
        if team_id == TEAM_RED_ID:
            return normalize_brain_kind(
                str(getattr(config, "TEAM_BRAIN_EXCLUSIVE_RED", default_kind)).strip().lower()
            )
        if team_id == TEAM_BLUE_ID:
            return normalize_brain_kind(
                str(getattr(config, "TEAM_BRAIN_EXCLUSIVE_BLUE", default_kind)).strip().lower()
            )
        return default_kind

    if mode in ("mix", "hybrid", "both"):
        strategy = str(getattr(config, "TEAM_BRAIN_MIX_STRATEGY", "alternate")).strip().lower()
        seq = tuple(
            normalize_brain_kind(str(x).strip().lower())
            for x in getattr(config, "TEAM_BRAIN_MIX_SEQUENCE", (default_kind,))
            if str(x).strip()
        ) or (default_kind,)

        if strategy in ("alternate", "roundrobin", "rr"):
            i = _TEAM_BRAIN_MIX_COUNTER.get(team_id, 0)
            _TEAM_BRAIN_MIX_COUNTER[team_id] = i + 1
            return seq[i % len(seq)]

        if strategy in ("random", "prob", "probabilistic"):
            rng = _TEAM_BRAIN_MIX_RNG.get(team_id, random.SystemRandom())
            weighted = (
                (
                    "throne_of_ashen_dreams",
                    max(0.0, float(getattr(config, "TEAM_BRAIN_MIX_P_THRONE_OF_ASHEN_DREAMS", 0.0))),
                ),
                (
                    "veil_of_the_hollow_crown",
                    max(0.0, float(getattr(config, "TEAM_BRAIN_MIX_P_VEIL_OF_THE_HOLLOW_CROWN", 0.0))),
                ),
                (
                    "black_grail_of_nightfire",
                    max(0.0, float(getattr(config, "TEAM_BRAIN_MIX_P_BLACK_GRAIL_OF_NIGHTFIRE", 0.0))),
                ),
            )
            total = sum(w for _, w in weighted)
            if total <= 0.0:
                return default_kind
            r = rng.random() * total
            acc = 0.0
            for kind, w in weighted:
                acc += w
                if r <= acc:
                    return kind
            return weighted[-1][0]

        raise ValueError(f"Unknown TEAM_BRAIN_MIX_STRATEGY={strategy!r}")

    raise ValueError(f"Unknown TEAM_BRAIN_ASSIGNMENT_MODE={mode!r}")

def _infer_kind_from_parent(parent: torch.nn.Module) -> Optional[str]:
    return brain_kind_from_module(parent)


def _slot_uid(reg: AgentsRegistry, data: torch.Tensor, slot: int) -> int:
    """Resolve the persistent UID for one registry slot."""
    if hasattr(reg, "agent_uids"):
        return int(reg.agent_uids[int(slot)].item())
    return int(data[int(slot), COL_AGENT_ID].item())


def _non_negative_finite(value: object) -> float:
    """Return a finite non-negative float, or 0.0 for invalid input."""
    try:
        x = float(value)
    except Exception:
        return 0.0
    if not math.isfinite(x):
        return 0.0
    return max(x, 0.0)


def _normalize_non_negative(values: List[float]) -> List[float]:
    """Normalize non-negative values by their positive max; all-zero stays all-zero."""
    clean = [_non_negative_finite(v) for v in values]
    top = max(clean) if clean else 0.0
    if top <= 0.0:
        return [0.0] * len(clean)
    return [v / top for v in clean]


def _resolve_birth_random_pool(cfg: RespawnCfg) -> Tuple[str, ...]:
    """Return a sanitized non-recursive doctrine pool for random_per_birth."""
    raw = tuple(
        str(x).strip().lower()
        for x in getattr(cfg, "birth_random_doctrine_pool", ("overall",))
        if str(x).strip()
    )
    cleaned = tuple(x for x in raw if x in _BIRTH_DOCTRINES and x != "random_per_birth")
    return cleaned or ("overall",)


def _resolve_birth_doctrine(cfg: RespawnCfg) -> str:
    """Resolve the active doctrine for one birth event."""
    doctrine = str(getattr(cfg, "birth_doctrine_mode", "overall")).strip().lower()
    if doctrine == "random_per_birth":
        return str(random.choice(_resolve_birth_random_pool(cfg)))
    if doctrine in _BIRTH_DOCTRINES:
        return doctrine
    return "overall"


def _resolve_parent_topk_size(n: int, cfg: RespawnCfg) -> int:
    """Resolve explicit top-k size first, then fall back to legacy fraction mode."""
    explicit = int(getattr(cfg, "birth_topk_size", 0))
    if explicit > 0:
        return max(1, min(n, explicit))

    frac = float(getattr(cfg, "parent_selection_topk_frac", 0.25))
    frac = min(max(frac, 0.0), 1.0)
    if frac <= 0.0:
        return 1
    return max(1, min(n, int(round(n * frac))))


def _build_doctrine_scores(
    parents: torch.Tensor,
    reg: AgentsRegistry,
    data: torch.Tensor,
    cfg: RespawnCfg,
    *,
    doctrine: str,
    engine: Optional[Any],
) -> List[float]:
    """Compute one doctrine score per candidate parent."""
    parent_slots = [int(x) for x in parents.detach().cpu().tolist()]
    if not parent_slots:
        return []

    kill_store = getattr(engine, "agent_kill_counts", {}) if engine is not None else {}
    cp_store = getattr(engine, "agent_cp_points", {}) if engine is not None else {}
    personal_store = getattr(engine, "agent_scores", {}) if engine is not None else {}

    health_raw: List[float] = []
    kill_raw: List[float] = []
    cp_raw: List[float] = []
    personal_raw: List[float] = []

    for slot in parent_slots:
        hp = _non_negative_finite(data[slot, COL_HP].item())
        hp_max = max(_non_negative_finite(data[slot, COL_HP_MAX].item()), 1e-8)
        health_raw.append(min(max(hp / hp_max, 0.0), 1.0))

        uid = _slot_uid(reg, data, slot)
        kill_raw.append(_non_negative_finite(kill_store.get(uid, 0.0) if hasattr(kill_store, "get") else 0.0))
        cp_raw.append(_non_negative_finite(cp_store.get(uid, 0.0) if hasattr(cp_store, "get") else 0.0))
        personal_raw.append(_non_negative_finite(personal_store.get(uid, 0.0) if hasattr(personal_store, "get") else 0.0))

    kill_norm = _normalize_non_negative(kill_raw)
    cp_norm = _normalize_non_negative(cp_raw)
    personal_norm = _normalize_non_negative(personal_raw)
    health_norm = [min(max(v, 0.0), 1.0) for v in health_raw]

    w_kill = _non_negative_finite(getattr(cfg, "birth_blend_weight_kill", 1.0))
    w_cp = _non_negative_finite(getattr(cfg, "birth_blend_weight_cp", 1.0))
    w_health = _non_negative_finite(getattr(cfg, "birth_blend_weight_health", 1.0))
    w_personal = _non_negative_finite(getattr(cfg, "birth_blend_weight_personal", 1.0))

    if (w_kill + w_cp + w_health + w_personal) <= 0.0:
        w_kill = w_cp = w_health = w_personal = 1.0

    n = len(parent_slots)
    if doctrine == "killer":
        return kill_norm
    if doctrine == "cp":
        return cp_norm
    if doctrine == "health":
        return health_norm
    if doctrine == "personal_points":
        return personal_norm
    if doctrine == "kill_health":
        return [(w_kill * kill_norm[i]) + (w_health * health_norm[i]) for i in range(n)]
    if doctrine == "health_cp":
        return [(w_health * health_norm[i]) + (w_cp * cp_norm[i]) for i in range(n)]
    if doctrine == "kill_cp":
        return [(w_kill * kill_norm[i]) + (w_cp * cp_norm[i]) for i in range(n)]
    if doctrine == "trinity":
        return [
            (w_kill * kill_norm[i]) + (w_cp * cp_norm[i]) + (w_health * health_norm[i])
            for i in range(n)
        ]
    if doctrine == "highest_spike":
        return [max(kill_norm[i], cp_norm[i], health_norm[i]) for i in range(n)]

    return [
        (w_personal * personal_norm[i])
        + (w_kill * kill_norm[i])
        + (w_cp * cp_norm[i])
        + (w_health * health_norm[i])
        for i in range(n)
    ]


# Brain creation and cloning (preserves team-specific assignment)
def _new_brain(device: torch.device, *, team_id: Optional[float] = None) -> torch.nn.Module:
    """Create a brain module for a newly activated slot.

    The architecture depends on PPO mode, team assignment settings, and the
    configured default brain kind.
    """
    obs_dim = int(getattr(config, "OBS_DIM", 0))
    act_dim = int(getattr(config, "NUM_ACTIONS", 41))

    team_assign = bool(getattr(config, "TEAM_BRAIN_ASSIGNMENT", True))
    if team_assign and team_id is not None:
        brain_kind = _resolve_team_brain_kind_from_team(float(team_id))
    else:
        brain_kind = normalize_brain_kind(
            str(getattr(config, "BRAIN_KIND", "throne_of_ashen_dreams")).strip().lower()
        )

    return create_mlp_brain(brain_kind, obs_dim, act_dim).to(device)


def _clone_brain(
    parent: Optional[torch.nn.Module],
    device: torch.device,
    *,
    team_id: Optional[float] = None,
) -> torch.nn.Module:
    """Clone a parent brain.

    OVERVIEW
    --------
    This function produces a "child" brain from a parent brain, with behavior
    depending on whether PPO is enabled.

    Non-PPO:
      - Attempt to deepcopy(parent).
      - If deepcopy fails, return a fresh brain (_new_brain).

    PPO:
      - Determine the target brain_kind using:
          a) team assignment mode (exclusive vs mix)
          b) parent's inferred kind (when in mix mode)
          c) fallback to config.BRAIN_KIND
      - Instantiate a new module of that kind.
      - Load weights only if architecture is compatible.

    WHY INSTANTIATE A NEW MODULE IN PPO MODE?
    ----------------------------------------
    Deepcopy of PyTorch modules is not always reliable or efficient.
    Additionally, optimizer state and internal buffers may behave poorly if
    copied improperly. Creating a clean child module and selectively copying
    state_dict is a robust and explicit approach.

    WHY CONDITIONAL load_state_dict?
    --------------------------------
    state_dict compatibility requires matching parameter names and shapes.
    Loading weights into a different architecture would raise errors or,
    worse, silently misalign if forced (not done here).
    """
    if parent is None:
        return _new_brain(device, team_id=team_id)

    obs_dim = int(getattr(config, "OBS_DIM", 0))
    act_dim = int(getattr(config, "NUM_ACTIONS", 41))

    is_ppo = bool(getattr(config, "PPO_ENABLED", False))
    if not is_ppo:
        try:
            # deepcopy attempts to duplicate the Python object graph.
            # May fail for modules holding non-copyable resources.
            return copy.deepcopy(parent)
        except Exception:
            # Fallback ensures system remains functional.
            return _new_brain(device, team_id=team_id)

    team_assign = bool(getattr(config, "TEAM_BRAIN_ASSIGNMENT", True))
    mode = str(getattr(config, "TEAM_BRAIN_ASSIGNMENT_MODE", "exclusive")).strip().lower()

    parent_kind = _infer_kind_from_parent(parent)

    if team_assign and team_id is not None:
        if mode in ("exclusive", "split", "team"):
            brain_kind = _resolve_team_brain_kind_from_team(float(team_id))
        else:
            brain_kind = parent_kind or _resolve_team_brain_kind_from_team(float(team_id))
    else:
        brain_kind = normalize_brain_kind(
            str(getattr(config, "BRAIN_KIND", "throne_of_ashen_dreams")).strip().lower()
        )

    child = create_mlp_brain(brain_kind, obs_dim, act_dim).to(device)
    if parent_kind == brain_kind:
        child.load_state_dict(parent.state_dict())
    return child


@torch.no_grad()
def _perturb_brain_(brain: torch.nn.Module, std: float) -> None:
    """Add small Gaussian noise to all trainable parameters.

    IMPORTANT DECORATOR: @torch.no_grad()
    -------------------------------------
    This disables gradient tracking within this function, which is desirable because:
      - This is not a learning update; it is a mutation / perturbation operation.
      - Avoids polluting autograd graphs and saves memory.

    MATHEMATICAL MEANING
    --------------------
    For each trainable parameter tensor p (a vector or matrix of weights),
    we perform an in-place update:

        p ← p + N(0, std^2)   elementwise

    Where torch.randn_like(p) produces samples from N(0, 1) for each element.
    Multiplying by std scales to N(0, std^2).

    ENGINEERING NOTES
    -----------------
    - Mutation is applied only if p.requires_grad is True.
      This avoids perturbing frozen layers or buffers.
    - In-place add_ is used for efficiency.
    """
    if std <= 0.0:
        return
    for p in brain.parameters():
        if p.requires_grad:
            p.add_(torch.randn_like(p) * std)


# Spawn cell selection (with wall margin)
def _cell_free(grid: torch.Tensor, x: int, y: int, cfg: RespawnCfg) -> bool:
    """Check if a cell is free for spawning (no wall, no agent).

    GRID CONTRACT (as documented elsewhere in the project)
    -----------------------------------------------------
    grid is shaped (C, H, W) with:
      grid[0, y, x] = occupancy code:
        0.0 empty
        1.0 wall
        2.0 red occupant
        3.0 blue occupant
      grid[2, y, x] = slot id (agent index) or -1.0 if empty

    WALL MARGIN
    -----------
    The function enforces:
      cfg.wall_margin <= x < W - cfg.wall_margin
      cfg.wall_margin <= y < H - cfg.wall_margin

    This ensures spawns do not occur too close to borders.

    WHY .item()?
    ------------
    grid[...] returns a scalar tensor.
    .item() converts it into a Python number for comparisons.
    This is convenient but note:
      - .item() triggers a device-to-host sync if grid is on GPU.
      - For high-performance spawning at scale, one might prefer vectorized
        operations. Here the spawn count per tick is capped, so overhead is likely acceptable.
    """
    H, W = grid.shape[1], grid.shape[2]
    return (cfg.wall_margin <= x < W - cfg.wall_margin and
            cfg.wall_margin <= y < H - cfg.wall_margin and
            grid[0, y, x].item() == 0.0 and      # empty (no wall, no agent)
            grid[2, y, x].item() == -1.0)         # slot id must be -1 for empty


def _pick_uniform(grid: torch.Tensor, cfg: RespawnCfg) -> Tuple[int, int]:
    """Randomly sample a free cell with uniform distribution.

    METHOD
    ------
    Rejection sampling:
      - Sample (x, y) uniformly from allowed interior region.
      - Accept if _cell_free is True.
      - Repeat up to cfg.spawn_tries times.

    EXPECTATION
    -----------
    If free-space ratio is high, expected tries is small.
    If grid is dense (few free cells), tries may be near spawn_tries and fail.

    FAILURE MODE
    ------------
    Returns (-1, -1) if no free cell is found within the try limit.
    Caller must interpret x < 0 as failure (as done in _respawn_some).

    NOTE ON RANDOMNESS
    ------------------
    This uses Python's random module, not torch RNG.
    This means:
      - randomness is not tied to torch seeds unless user also seeds random.
      - distribution is CPU-side.
    """
    H, W = grid.shape[1], grid.shape[2]
    for _ in range(cfg.spawn_tries):
        x = random.randint(cfg.wall_margin, W - cfg.wall_margin - 1)
        y = random.randint(cfg.wall_margin, H - cfg.wall_margin - 1)
        if _cell_free(grid, x, y, cfg):
            return x, y
    return -1, -1


def _pick_near_parent(grid: torch.Tensor, cfg: RespawnCfg, parent_xy: Tuple[int, int]) -> Tuple[int, int]:
    """Try to place child near parent; fall back to failure so caller can fallback safely."""
    px, py = int(parent_xy[0]), int(parent_xy[1])
    r = max(1, int(getattr(cfg, "spawn_near_parent_radius", 1)))

    # Generate local offsets (excluding (0,0)), shuffled for stochasticity.
    offsets: List[Tuple[int, int]] = []
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dx == 0 and dy == 0:
                continue
            offsets.append((dx, dy))
    random.shuffle(offsets)

    tried = 0
    max_tries = max(1, int(cfg.spawn_tries))
    while tried < max_tries:
        for dx, dy in offsets:
            x = px + dx
            y = py + dy
            tried += 1
            if _cell_free(grid, x, y, cfg):
                return x, y
            if tried >= max_tries:
                break

    # Also allow exact parent cell only if it became free (usually not, but harmless fallback).
    if _cell_free(grid, px, py, cfg):
        return px, py
    return -1, -1


def _pick_location(
    grid: torch.Tensor,
    cfg: RespawnCfg,
    parent_xy: Optional[Tuple[int, int]] = None,
) -> Tuple[int, int]:
    """Main entry point for spawn cell selection (uniform by default; near-parent optional)."""
    mode = str(getattr(cfg, "spawn_location_mode", "uniform")).strip().lower()
    if mode in ("near_parent", "parent_near") and parent_xy is not None:
        x, y = _pick_near_parent(grid, cfg, parent_xy)
        if x >= 0:
            return x, y
    return _pick_uniform(grid, cfg)


# Unit type and stats (from config)
def _choose_unit(cfg: RespawnCfg) -> int:
    """Randomly choose a unit type based on spawn_archer_ratio.

    Let r = cfg.spawn_archer_ratio in [0, 1].
    Draw u ~ Uniform(0, 1).
      - If u < r => choose archer
      - Else    => choose soldier

    This yields P(archer) = r, P(soldier) = 1 - r.

    NOTE
    ----
    This uses Python RNG; see earlier note about reproducibility.
    """
    return cfg.unit_archer if random.random() < cfg.spawn_archer_ratio else cfg.unit_soldier


def _choose_unit_for_spawn(
    cfg: RespawnCfg,
    data: torch.Tensor,
    *,
    use_clone: bool,
    parent_slot: Optional[int],
) -> int:
    """
    Resolve child unit type deterministically from configured reproduction semantics.

    Supported modes:
      - "inherit_parent_on_clone" (default):
            clone path inherits parent unit exactly;
            fresh/non-clone path uses random spawn ratio.
      - "random":
            always uses random spawn ratio.
    """
    mode = str(getattr(cfg, "child_unit_mode", "inherit_parent_on_clone")).strip().lower()

    if mode not in ("inherit_parent_on_clone", "inherit_parent", "clone_inherits_unit", "random"):
        raise RuntimeError(f"[respawn] Unsupported child_unit_mode={mode!r}")

    if use_clone and mode in ("inherit_parent_on_clone", "inherit_parent", "clone_inherits_unit"):
        if parent_slot is None:
            raise RuntimeError("[respawn] clone child_unit_mode requires a concrete parent slot")
        if int(parent_slot) < 0 or int(parent_slot) >= int(data.shape[0]):
            raise RuntimeError(f"[respawn] parent slot out of range for unit inheritance: {parent_slot}")

        parent_unit = int(round(float(data[int(parent_slot), COL_UNIT].item())))
        if parent_unit not in (int(cfg.unit_soldier), int(cfg.unit_archer)):
            raise RuntimeError(f"[respawn] unsupported parent unit for inheritance: {parent_unit}")
        return int(parent_unit)

    return int(_choose_unit(cfg))


def _unit_stats(unit_id: int, cfg: RespawnCfg) -> Tuple[float, float, int]:
    """Return (hp, atk, vision_range) for the given unit type.

    This maps unit_id -> base stats.

    The code is intentionally simple:
      - If archer: use archer_hp, archer_atk, vision_archer
      - Otherwise: use soldier_hp, soldier_atk, vision_soldier

    DESIGN NOTE
    -----------
    This function centralizes stat mapping so that future changes (more unit types,
    dynamic stats, scaling, etc.) have a single choke point.
    """
    if unit_id == cfg.unit_archer:
        return cfg.archer_hp, cfg.archer_atk, cfg.vision_archer
    return cfg.soldier_hp, cfg.soldier_atk, cfg.vision_soldier


# Write agent to registry (direct tensor assignment, preserves old behavior)
def _write_agent_to_registry(
    reg: AgentsRegistry,
    slot: int,
    team_id: float,
    x: int,
    y: int,
    unit_id: int,
    hp: float,
    atk: float,
    vision: int,
    brain: torch.nn.Module,
    generation: int = 0,
) -> None:
    """Fill the registry tensors and brain list for a newly spawned agent.

    REGISTRY LAYOUT ASSUMPTION
    --------------------------
    AgentsRegistry is assumed to contain:
      - reg.agent_data: tensor of shape (N, D) storing numeric attributes
      - reg.brains: list-like container mapping slot -> brain module
      - reg.device: torch.device for module placement
      - reg.get_next_id(): method returning unique agent identifier

    This function updates:
      (1) agent_data columns for the given slot
      (2) agent_uids side-tensor if present (preferred storage for unique IDs)
      (3) reg.brains and reg.optimizers if present

    WHY BOTH agent_uids AND COL_AGENT_ID?
    -------------------------------------
    The code indicates a design evolution:
      - agent_data may be float16/float32; storing large integer IDs in float16
        risks overflow, rounding, or inf.
      - So a robust int64 tensor (reg.agent_uids) is used if available.
      - The legacy float column remains for display/debug but is clamped.
    """

    # Mark as alive. The system uses float flags: alive > 0.5.
    reg.agent_data[slot, COL_ALIVE] = 1.0

    # Assign team identifier (likely 2.0 for red and 3.0 for blue, though actual
    # numeric values come from TEAM_* constants).
    reg.agent_data[slot, COL_TEAM] = team_id

    # Store location. x and y are stored as floats (consistent with the tensor dtype).
    reg.agent_data[slot, COL_X] = float(x)
    reg.agent_data[slot, COL_Y] = float(y)

    # Initialize HP and HP_MAX.
    # HP_MAX represents maximum HP capacity; initial spawn sets HP=HP_MAX.
    reg.agent_data[slot, COL_HP] = hp
    reg.agent_data[slot, COL_HP_MAX] = hp

    # Vision stored as float to match tensor dtype (even though semantically int).
    reg.agent_data[slot, COL_VISION] = float(vision)

    # Attack power is stored as float.
    reg.agent_data[slot, COL_ATK] = atk

    # Unit type stored as float (semantically categorical).
    reg.agent_data[slot, COL_UNIT] = float(unit_id)

    # B7: Assign a fresh agent_id for this newly spawned individual.
    # get_next_id() is assumed to be provided by AgentsRegistry.
    new_aid = reg.get_next_id()

    # Store true UID in int64 side-tensor (robust even if agent_data is float16).
    if hasattr(reg, "agent_uids"):
        reg.agent_uids[slot] = int(new_aid)

    # Keep legacy float column as display-only (clamp to avoid inf when dtype is float16).
    # torch.finfo(dtype).max gives the maximum finite representable value for that dtype.
    # For float16, max is ~65504. Large IDs beyond that would become inf without clamping.
    try:
        max_f = torch.finfo(reg.agent_data.dtype).max
        reg.agent_data[slot, COL_AGENT_ID] = float(min(float(new_aid), float(max_f)))
    except Exception:
        # If dtype is not floating or finfo fails, store directly.
        reg.agent_data[slot, COL_AGENT_ID] = float(new_aid)

    # Replace brain and keep architecture metadata synchronized.
    reg.set_brain(slot, brain)
    if hasattr(reg, "optimizers"):
        reg.optimizers[slot] = None
    if hasattr(reg, "generations") and 0 <= int(slot) < len(reg.generations):
        reg.generations[slot] = int(generation)


# Parent selection + anomaly helpers
def _pick_parent_slot(
    parents: torch.Tensor,
    reg: AgentsRegistry,
    data: torch.Tensor,
    cfg: RespawnCfg,
    *,
    engine: Optional[Any] = None,
) -> Tuple[int, Optional[str], float]:
    """Choose a parent slot and report the doctrine and merit score used."""
    n = int(parents.numel())
    if n <= 0:
        raise RuntimeError("_pick_parent_slot called with empty parents")

    mode = str(getattr(cfg, "parent_selection_mode", "random")).strip().lower()
    if mode in ("random", "uniform", "baseline"):
        return int(parents[random.randrange(n)].item()), None, 0.0

    if mode in ("topk_weighted", "stronger_biased", "elite_weighted"):
        doctrine = _resolve_birth_doctrine(cfg)
        scores = _build_doctrine_scores(parents, reg, data, cfg, doctrine=doctrine, engine=engine)
        if len(scores) != n:
            raise RuntimeError(
                f"[respawn] doctrine score count mismatch: got {len(scores)} expected {n} for doctrine={doctrine!r}"
            )

        if not any(float(s) > 0.0 for s in scores):
            fallback = str(getattr(cfg, "birth_zero_score_fallback", "uniform_candidates")).strip().lower()
            if fallback == "abort_birth":
                return -1, doctrine, 0.0
            return int(parents[random.randrange(n)].item()), doctrine, 0.0

        k = _resolve_parent_topk_size(n, cfg)
        order = sorted(range(n), key=lambda i: scores[i], reverse=True)
        cand_idx = order[:k]
        power = max(float(getattr(cfg, "parent_selection_score_power", 1.0)), 0.0)
        weights = [max(float(scores[i]), 1e-12) ** power for i in cand_idx]
        if not any(w > 0.0 for w in weights):
            fallback = str(getattr(cfg, "birth_zero_score_fallback", "uniform_candidates")).strip().lower()
            if fallback == "abort_birth":
                return -1, doctrine, 0.0
            return int(parents[random.randrange(n)].item()), doctrine, 0.0
        chosen_local = random.choices(cand_idx, weights=weights, k=1)[0]
        return int(parents[chosen_local].item()), doctrine, float(scores[chosen_local])

    raise RuntimeError(f"[respawn] unsupported parent_selection_mode={mode!r}")


def _apply_rare_physical_drift(hp0: float, atk0: float, vision0: int, cfg: RespawnCfg) -> Tuple[float, float, int]:
    """Apply small bounded multiplicative drift to physical traits."""
    std = max(float(getattr(cfg, "rare_mutation_physical_drift_std_frac", 0.03)), 0.0)
    clip = max(float(getattr(cfg, "rare_mutation_physical_drift_clip_frac", 0.10)), 0.0)

    def _mult() -> float:
        if std <= 0.0:
            return 1.0
        d = random.gauss(0.0, std)
        if d > clip:
            d = clip
        elif d < -clip:
            d = -clip
        m = 1.0 + d
        return max(0.05, m)

    hp1 = float(hp0) * _mult()
    atk1 = float(atk0) * _mult()
    vis1 = max(1, int(round(float(vision0) * _mult())))
    return hp1, atk1, vis1


# Core respawn function (spawns a given number of agents for a team)
@torch.no_grad()
def _respawn_some(
    reg: AgentsRegistry,
    grid: torch.Tensor,
    team_id: float,
    count: int,
    cfg: RespawnCfg,
    tick: int,
    meta_out: Optional[List[Dict[str, object]]] = None,
    controller: Optional["RespawnController"] = None,
    spawn_reason: str = "respawn",
    engine: Optional[Any] = None,
) -> int:
    """
    Attempt to spawn up to `count` agents of the given team.

    RETURNS
    -------
    int: number of agents actually spawned.
         This may be less than requested due to:
           - insufficient dead slots,
           - no free spawn locations found,
           - early termination on sampling failure.

    SIDE EFFECTS
    ------------
    - Mutates reg.agent_data (revives slots).
    - Mutates reg.brains (replaces brain modules).
    - Mutates reg.optimizers (sets to None when present).
    - Mutates grid to reflect occupancy/hp/slot-id at spawn location.
    - Appends metadata dictionaries into meta_out if provided.

    IMPORTANT: @torch.no_grad()
    ---------------------------
    Respawning is not a differentiable learning step. Disabling gradients:
      - avoids autograd memory overhead,
      - prevents accidental graph creation if tensors are on GPU.
    """

    # Trivial guard: do nothing if count is non-positive.
    if count <= 0:
        return 0

    # Find dead slots (alive == 0).
    data = reg.agent_data
    alive = (data[:, COL_ALIVE] > 0.5)
    dead_slots = (~alive).nonzero(as_tuple=False).squeeze(1)
    # nonzero(...).squeeze(1) yields a 1D tensor of indices.

    if dead_slots.numel() == 0:
        # No available slots to respawn into.
        return 0

    # Gather potential parents (alive agents of the same team).
    # This produces indices into reg.brains / reg.agent_data.
    parents = (alive & (data[:, COL_TEAM] == team_id)).nonzero(as_tuple=False).squeeze(1)
    require_parent_for_birth = bool(getattr(cfg, "require_parent_for_birth", False))

    if require_parent_for_birth and parents.numel() == 0:
        return 0

    spawned = 0

    # Iterate sequentially over dead slots. This is deterministic given dead_slots
    # order, but the spawn locations and clone selection introduce randomness.
    for k in range(min(count, dead_slots.numel())):
        slot = int(dead_slots[k].item())

        # Decide whether the child inherits the parent's exact brain weights.
        use_clone = (parents.numel() > 0) and (random.random() < cfg.clone_prob)

        pj = None
        selected_doctrine: Optional[str] = None
        parent_merit_score = 0.0
        parent_xy: Optional[Tuple[int, int]] = None
        if parents.numel() > 0 and (require_parent_for_birth or use_clone):
            pj, selected_doctrine, parent_merit_score = _pick_parent_slot(
                parents,
                reg,
                data,
                cfg,
                engine=engine,
            )
            if pj < 0:
                continue
            if str(getattr(cfg, "spawn_location_mode", "uniform")).strip().lower() in ("near_parent", "parent_near"):
                try:
                    px = int(round(float(data[pj, COL_X].item())))
                    py = int(round(float(data[pj, COL_Y].item())))
                    parent_xy = (px, py)
                except Exception:
                    parent_xy = None

        # Choose spawn coordinates.
        x, y = _pick_location(grid, cfg, parent_xy=parent_xy)
        if x < 0:
            # If location selection fails, break early.
            # This prevents repeated failures and wasted computation.
            break

        parent_unit_id: Optional[int] = None
        parent_team_id: Optional[int] = None
        parent_generation: Optional[int] = None
        if pj is not None:
            if int(pj) < 0 or int(pj) >= int(data.shape[0]):
                raise RuntimeError(f"[respawn] picked parent slot out of range: {pj}")
            if float(data[pj, COL_ALIVE].item()) <= 0.5:
                raise RuntimeError(f"[respawn] picked parent slot is not alive: {pj}")
            if float(data[pj, COL_TEAM].item()) != float(team_id):
                raise RuntimeError(
                    f"[respawn] parent team mismatch: parent_team={float(data[pj, COL_TEAM].item())} child_team={float(team_id)}"
                )
            parent_unit_id = int(round(float(data[pj, COL_UNIT].item())))
            parent_team_id = int(round(float(data[pj, COL_TEAM].item())))
            if hasattr(reg, "generations") and 0 <= int(pj) < len(reg.generations):
                parent_generation = int(reg.generations[pj])

        # Choose unit type using explicit child-unit semantics.
        unit_id = _choose_unit_for_spawn(
            cfg,
            data,
            use_clone=bool(use_clone),
            parent_slot=(int(pj) if pj is not None else None),
        )
        hp_base, atk_base, vision_base = _unit_stats(unit_id, cfg)
        hp0, atk0, vision0 = hp_base, atk_base, vision_base

        # Rare mutation trigger path:
        # - legacy behavior (every 1000th respawn globally) is preserved unless
        #   the tick-window trigger is explicitly enabled.
        rare_mutation = False
        if bool(getattr(cfg, "rare_mutation_tick_window_enable", False)) and controller is not None:
            rare_mutation = bool(controller._consume_pending_rare_mutation_ticket())
        else:
            if controller is not None:
                controller._legacy_respawn_counter = int(getattr(controller, "_legacy_respawn_counter", 0)) + 1
                legacy_counter = int(controller._legacy_respawn_counter)
            else:
                # Defensive fallback for direct/test invocation without a controller.
                cur = int(getattr(_respawn_some, "_legacy_respawn_counter", 0))
                setattr(_respawn_some, "_legacy_respawn_counter", cur + 1)
                legacy_counter = int(getattr(_respawn_some, "_legacy_respawn_counter", 0))

            if legacy_counter % 1000 == 0:
                rare_mutation = True
                hp0 *= (1.0 + random.uniform(0.5, 2.0))
                atk0 *= (1.0 + random.uniform(0.5, 2.0))
                vision0 = int(vision0 * (1.0 + random.uniform(0.5, 2.0)))

        # New anomaly physical effect (small drift) is independently switchable.
        # If disabled under tick-window mode, the ticket still produces an anomaly birth
        # without stat drift (brain-side effect may still apply on clone path).
        if rare_mutation and bool(getattr(cfg, "rare_mutation_tick_window_enable", False)):
            if bool(getattr(cfg, "rare_mutation_physical_enable", False)):
                hp0, atk0, vision0 = _apply_rare_physical_drift(hp0, atk0, vision0, cfg)
            print(
                f"** Rare Mutation on slot {slot} (team {'red' if team_id==TEAM_RED_ID else 'blue'})! "
                f"HP:{hp0:.2f}, ATK:{atk0:.2f}, VIS:{vision0} **"
            )
        elif rare_mutation:
            # Legacy path still prints after legacy large-scaling mutation.
            print(
                f"** Rare Mutation on slot {slot} (team {'red' if team_id==TEAM_RED_ID else 'blue'})! "
                f"HP:{hp0:.2f}, ATK:{atk0:.2f}, VIS:{vision0} **"
            )

        if use_clone:
            if pj is None:
                raise RuntimeError("[respawn] clone path entered without parent slot")
            # Clone the parent's brain, preserving architecture rules.
            brain = _clone_brain(reg.brains[pj], reg.device, team_id=team_id)

            # Apply standard Gaussian perturbation to weights for diversity.
            _perturb_brain_(brain, cfg.mutation_std)

            # Optional heavy anomaly noise applies only on inherited/cloned path.
            if (
                rare_mutation
                and bool(getattr(cfg, "rare_mutation_inherited_brain_noise_enable", False))
                and float(getattr(cfg, "rare_mutation_inherited_brain_noise_std", 0.0)) > 0.0
            ):
                _perturb_brain_(brain, float(cfg.rare_mutation_inherited_brain_noise_std))
        else:
            # Create a brand-new brain using team-aware assignment rules.
            brain = _new_brain(reg.device, team_id=team_id)

        mutation_delta_hp = float(hp0) - float(hp_base)
        mutation_delta_atk = float(atk0) - float(atk_base)
        mutation_delta_vis = int(vision0) - int(vision_base)
        mutation_flag = bool(
            (use_clone and float(cfg.mutation_std) > 0.0)
            or bool(rare_mutation)
            or abs(mutation_delta_hp) > 0.0
            or abs(mutation_delta_atk) > 0.0
            or int(mutation_delta_vis) != 0
        )
        child_generation = int(parent_generation + 1) if (pj is not None and parent_generation is not None) else 0

        # Commit agent state into registry tensors and brain storage.
        _write_agent_to_registry(
            reg,
            slot,
            team_id,
            x,
            y,
            unit_id,
            hp0,
            atk0,
            vision0,
            brain,
            generation=child_generation,
        )

        # Telemetry hook: capture spawn metadata without affecting simulation.
        # meta_out is an optional list. If provided, append a dict describing spawn.
        # This can be used by logging/telemetry systems to record births,
        # evolutionary lineage, and spawn positions.
        if meta_out is not None:
            # Read true UID from int64 side-tensor if present.
            if hasattr(reg, "agent_uids"):
                child_aid = int(reg.agent_uids[slot].item())
                parent_aid = int(reg.agent_uids[pj].item()) if pj is not None else None
            else:
                # Fallback for older registries (may be unsafe under float16).
                child_aid = int(reg.agent_data[slot, COL_AGENT_ID].item())
                parent_aid = int(reg.agent_data[pj, COL_AGENT_ID].item()) if pj is not None else None

            meta_out.append({
                "tick": int(tick),
                "slot": int(slot),
                "agent_id": int(child_aid),
                "team_id": float(team_id),
                "unit_id": int(unit_id),
                "spawn_reason": str(spawn_reason),
                "spawn_origin": ("clone" if use_clone else ("fresh_parented" if pj is not None else "fresh")),
                "x": int(x),
                "y": int(y),
                "cloned": bool(use_clone),
                "parent_slot": int(pj) if pj is not None else None,
                "parent_agent_id": int(parent_aid) if parent_aid is not None else None,
                "parent_unit_id": int(parent_unit_id) if parent_unit_id is not None else None,
                "parent_team_id": int(parent_team_id) if parent_team_id is not None else None,
                "parent_generation": int(parent_generation) if parent_generation is not None else None,
                "child_generation": int(child_generation),
                "unit_inherited_from_parent": bool(use_clone and parent_unit_id is not None and int(parent_unit_id) == int(unit_id)),
                "birth_doctrine": (str(selected_doctrine) if selected_doctrine is not None else None),
                "birth_merit_score": (float(parent_merit_score) if pj is not None else None),
                "closed_cradle_enforced": bool(require_parent_for_birth),
                "mutation_std": float(cfg.mutation_std),
                "mutation_flag": bool(mutation_flag),
                # Additive observability fields (no behavior impact)
                "rare_mutation": bool(rare_mutation),
                "rare_mutation_trigger_tick_window": bool(getattr(cfg, "rare_mutation_tick_window_enable", False)),
                "rare_mutation_heavy_brain_noise_applied": bool(
                    rare_mutation and use_clone and bool(getattr(cfg, "rare_mutation_inherited_brain_noise_enable", False))
                ),
                "spawn_hp": float(hp0),
                "spawn_atk": float(atk0),
                "spawn_vis": int(vision0),
                "base_hp": float(hp_base),
                "base_atk": float(atk_base),
                "base_vis": int(vision_base),
                "mutation_delta_hp": float(mutation_delta_hp),
                "mutation_delta_atk": float(mutation_delta_atk),
                "mutation_delta_vis": int(mutation_delta_vis),
            })

        # Update grid to reflect the spawned agent.
        # IMPORTANT CONTRACT (as stated in comment):
        #   grid[0]: occupancy code
        #   grid[1]: hp
        #   grid[2]: slot index (-1 for empty)
        # Here occupancy is set to float(team_id), which implies TEAM_RED_ID and
        # TEAM_BLUE_ID are chosen to match the occupancy encoding for teams
        # (commonly 2.0 and 3.0).
        grid[0, y, x] = float(team_id)
        grid[1, y, x] = float(hp0)
        grid[2, y, x] = float(slot)

        spawned += 1

    return spawned


# RespawnController: advanced team-aware respawn logic
class RespawnController:
    """Controller that manages floor-based and periodic respawning.

    RESPONSIBILITIES
    ----------------
    The controller enforces two respawn mechanisms:

      1) Floor-based respawn:
         - If alive count for a team is below cfg.floor_per_team,
           spawn enough agents to move toward the floor, subject to:
             - cfg.max_per_tick cap
             - cfg.cooldown_ticks hysteresis

      2) Periodic respawn:
         - Every cfg.period_ticks ticks,
           distribute cfg.period_budget respawns across teams based on inverse
           proportional split of current alive counts.

    TELEMETRY
    ---------
    last_spawn_meta is filled during each step() call with per-agent metadata,
    enabling external logging without mutating simulation logic further.
    """

    def __init__(self, cfg: RespawnCfg):
        self.cfg = cfg

        # Cooldown boundaries: when floor-based spawns are allowed again.
        self._cooldown_red_until = 0
        self._cooldown_blue_until = 0

        # Tracks last tick when periodic budget was distributed.
        self._last_period_tick = 0

        # Rare-mutation ticket state (tick-window mode; checkpoint-safe via checkpointing.py).
        self._rare_mutation_pending_ticket = 0  # int for easy serialization
        self._rare_mutation_last_window_idx = -1

        # Legacy every-Nth anomaly counter kept controller-scoped so fresh runs/controllers
        # do not inherit module-global state.
        self._legacy_respawn_counter = 0

        # Per-step metadata for all spawns performed in the most recent step.
        self.last_spawn_meta: List[Dict[str, object]] = []

    def _update_rare_mutation_ticket(self, tick: int) -> None:
        """Tick-window anomaly ticket generator (at most one pending ticket, no stacking)."""
        if not bool(getattr(self.cfg, "rare_mutation_tick_window_enable", False)):
            return

        window_ticks = max(1, int(getattr(self.cfg, "rare_mutation_tick_window_ticks", 1000)))
        window_idx = int(tick) // window_ticks

        # Initialize baseline without granting a ticket at tick 0.
        if self._rare_mutation_last_window_idx < 0:
            self._rare_mutation_last_window_idx = window_idx
            return

        if window_idx > self._rare_mutation_last_window_idx:
            if int(self._rare_mutation_pending_ticket) <= 0:
                self._rare_mutation_pending_ticket = 1
            # Advance to the latest observed window (no ticket stacking).
            self._rare_mutation_last_window_idx = window_idx

    def _consume_pending_rare_mutation_ticket(self) -> bool:
        if int(self._rare_mutation_pending_ticket) > 0:
            self._rare_mutation_pending_ticket = 0
            return True
        return False

    def step(
        self,
        tick: int,
        reg: AgentsRegistry,
        grid: torch.Tensor,
        engine: Optional[Any] = None,
    ) -> Tuple[int, int]:
        """
        Execute one step of the respawn logic.

        PARAMETERS
        ----------
        tick: int
          Current simulation tick.
        reg: AgentsRegistry
          Registry containing agent states and brains.
        grid: torch.Tensor
          Grid tensor to update occupancy/hp/slot-id for spawned agents.

        RETURNS
        -------
        (spawned_red, spawned_blue): Tuple[int, int]
          Number of agents spawned for each team during this tick.

        IMPORTANT STATEFUL BEHAVIOR
        ---------------------------
        This method uses internal controller state:
          - cooldown thresholds
          - last periodic tick
          - last_spawn_meta (reset per step)
        """
        # Reset per-step metadata.
        self.last_spawn_meta = []

        if not self.cfg.enabled:
            return 0, 0

        self._update_rare_mutation_ticket(int(tick))

        counts = _team_counts(reg)
        spawned_r = spawned_b = 0

        # 1) FLOOR-BASED RESPAWN WITH COOLDOWN (HYSTERESIS)
        # If counts.red < floor and cooldown passed:
        #   need = floor - counts.red
        #   spawn = min(need, max_per_tick)
        #   if we reach floor after spawning:
        #       set cooldown to tick + cooldown_ticks
        # This hysteresis prevents repeated firing due to small fluctuations.
        if counts.red < self.cfg.floor_per_team and tick >= self._cooldown_red_until:
            need = self.cfg.floor_per_team - counts.red
            spawned = _respawn_some(
                reg, grid, TEAM_RED_ID,
                _cap(need, self.cfg),
                self.cfg,
                tick,
                self.last_spawn_meta,
                controller=self,
                spawn_reason="floor",
                engine=engine,
            )
            spawned_r += spawned
            if counts.red + spawned >= self.cfg.floor_per_team:
                self._cooldown_red_until = tick + self.cfg.cooldown_ticks

        if counts.blue < self.cfg.floor_per_team and tick >= self._cooldown_blue_until:
            need = self.cfg.floor_per_team - counts.blue
            spawned = _respawn_some(
                reg, grid, TEAM_BLUE_ID,
                _cap(need, self.cfg),
                self.cfg,
                tick,
                self.last_spawn_meta,
                controller=self,
                spawn_reason="floor",
                engine=engine,
            )
            spawned_b += spawned
            if counts.blue + spawned >= self.cfg.floor_per_team:
                self._cooldown_blue_until = tick + self.cfg.cooldown_ticks

        # 2) PERIODIC RESPAWN BUDGET (INVERSE SPLIT)
        # Every period_ticks, allocate period_budget respawns.
        # Distribution is based on inverse of alive counts, so smaller team
        # receives a larger share.
        if tick - self._last_period_tick >= self.cfg.period_ticks:
            self._last_period_tick = tick
            total_alive = counts.red + counts.blue
            if total_alive > 0:
                q_r, q_b = _inverse_split(counts.red, counts.blue, self.cfg.period_budget)
                spawned_r += _respawn_some(
                    reg, grid, TEAM_RED_ID,
                    _cap(q_r, self.cfg),
                    self.cfg,
                    tick,
                    self.last_spawn_meta,
                    controller=self,
                    spawn_reason="periodic",
                    engine=engine,
                )
                spawned_b += _respawn_some(
                    reg, grid, TEAM_BLUE_ID,
                    _cap(q_b, self.cfg),
                    self.cfg,
                    tick,
                    self.last_spawn_meta,
                    controller=self,
                    spawn_reason="periodic",
                    engine=engine,
                )

        return spawned_r, spawned_b


# Public API: respawn_tick (backward compatible with old code)
def respawn_tick(reg: AgentsRegistry, grid: torch.Tensor, cfg: RespawnCfg) -> None:
    """
    Perform one tick of respawning.

    BACKWARD COMPATIBILITY NOTE
    ---------------------------
    This function preserves the old signature:
      respawn_tick(reg, grid, cfg)

    However, it constructs a *new* RespawnController each call and invokes step
    with tick=0.

    IMPLICATIONS OF CONSTRUCTING A NEW CONTROLLER EACH CALL
    ------------------------------------------------------
    Because the controller is re-created:
      - cooldown state is reset every call,
      - _last_period_tick is reset every call,
      - periodic and hysteresis behavior will not behave as intended if you call
        respawn_tick repeatedly expecting statefulness.

    In other words, this wrapper is suitable for "one-off" respawn operations
    or compatibility stubs, but not for long-running tick-by-tick control unless
    it is replaced by a persistent controller instance elsewhere in the engine.

    LEGACY FIELDS
    -------------
    prob_per_dead_per_tick is explicitly ignored, as the new controller does not
    implement probabilistic respawn in this path.

    RETURN VALUE
    ------------
    None. step() returns counts, but they are ignored to match the old API.
    """
    controller = RespawnController(cfg)
    controller.step(0, reg, grid, engine=None)  # tick number is not critical for one-off calls
    # Note: step returns counts, but we ignore them to match the old API.
