from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict   
import random
import copy
import random

import torch
import torch.jit

import config
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
from agent.transformer_brain import TransformerBrain, scripted_transformer_brain
from agent.tron_brain import TronBrain
from agent.mirror_brain import MirrorBrain

# ----------------------------------------------------------------------
# Respawn configuration dataclass (extended with new parameters)
# ----------------------------------------------------------------------
@dataclass
class RespawnCfg:
    """Configuration for agent respawning.

    This dataclass combines the original simple probabilistic respawn
    parameters with the new advanced controller parameters. All fields
    have defaults pulled from the global config module for backward
    compatibility and ease of use.
    """
    # Master switch
    enabled: bool = True

    # ---------- Legacy simple probabilistic fields ----------
    prob_per_dead_per_tick: float = 0.05      # Not used in new controller, kept for API stability
    spawn_tries: int = 200                    # Maximum attempts to find a free cell
    mutation_std: float = 0.02                 # Noise std when mutating cloned brains
    clone_prob: float = 0.50                   # Probability to clone an existing agent instead of creating a fresh brain
    use_team_elite: bool = True                 # Whether to consider only alive agents from the same team as parents
    reset_optimizer_on_respawn: bool = True     # Whether to reset optimizers (for PPO)

    # ---------- New controller fields ----------
    floor_per_team: int = field(default_factory=lambda: int(getattr(config, "RESP_FLOOR_PER_TEAM", 50)))
    """Minimum desired number of alive agents per team; if below, respawn tries to fill up."""

    max_per_tick: int = field(default_factory=lambda: int(getattr(config, "RESP_MAX_PER_TICK", 5)))
    """Maximum number of respawns per team per tick."""

    period_ticks: int = field(default_factory=lambda: int(getattr(config, "RESP_PERIOD_TICKS", 500)))
    """Interval (in ticks) at which a periodic respawn budget is distributed."""

    period_budget: int = field(default_factory=lambda: int(getattr(config, "RESP_PERIOD_BUDGET", 20)))
    """Total number of respawns to distribute among teams every period_ticks."""

    cooldown_ticks: int = field(default_factory=lambda: int(getattr(config, "RESP_HYST_COOLDOWN_TICKS", 30)))
    """Cooldown after reaching the floor for a team before another floor‑based respawn is allowed."""

    wall_margin: int = field(default_factory=lambda: int(getattr(config, "RESP_WALL_MARGIN", 2)))
    """Minimum distance from the grid border for spawn positions."""

    # Unit type configuration (pulled from config)
    unit_soldier: int = field(default_factory=lambda: int(getattr(config, "UNIT_SOLDIER", 1)))
    unit_archer: int = field(default_factory=lambda: int(getattr(config, "UNIT_ARCHER", 2)))
    spawn_archer_ratio: float = field(default_factory=lambda: float(getattr(config, "SPAWN_ARCHER_RATIO", 0.40)))
    soldier_hp: float = field(default_factory=lambda: float(getattr(config, "SOLDIER_HP", 1.0)))
    soldier_atk: float = field(default_factory=lambda: float(getattr(config, "SOLDIER_ATK", 0.05)))
    archer_hp: float = field(default_factory=lambda: float(getattr(config, "ARCHER_HP", 1.0)))
    archer_atk: float = field(default_factory=lambda: float(getattr(config, "ARCHER_ATK", 0.02)))
    vision_soldier: int = field(default_factory=lambda: int(getattr(config, "VISION_RANGE_BY_UNIT", {}).get(1, 10)))
    vision_archer: int = field(default_factory=lambda: int(getattr(config, "VISION_RANGE_BY_UNIT", {}).get(2, 15)))

# ----------------------------------------------------------------------
# Global counter for rare mutation events
# ----------------------------------------------------------------------
_respawn_counter = 0


# ----------------------------------------------------------------------
# Helper functions for team counts and distribution
# ----------------------------------------------------------------------
@dataclass
class TeamCounts:
    """Simple container for alive agent counts per team."""
    red: int
    blue: int


def _team_counts(reg: AgentsRegistry) -> TeamCounts:
    """Return the number of alive agents for each team."""
    d = reg.agent_data
    alive = (d[:, COL_ALIVE] > 0.5)
    red = int((alive & (d[:, COL_TEAM] == TEAM_RED_ID)).sum().item())
    blue = int((alive & (d[:, COL_TEAM] == TEAM_BLUE_ID)).sum().item())
    return TeamCounts(red=red, blue=blue)


def _inverse_split(a: int, b: int, budget: int) -> Tuple[int, int]:
    """
    Split a budget inversely proportional to the two numbers.
    Used to give more respawns to the team with fewer alive agents.
    """
    a = max(1, a)
    b = max(1, b)
    s = 1.0 / a + 1.0 / b
    qa = int(round(budget * (1.0 / a) / s))
    return qa, budget - qa


def _cap(n: int, cfg: RespawnCfg) -> int:
    """Clamp a desired respawn count to the per‑tick maximum."""
    return max(0, min(n, cfg.max_per_tick))

# ----------------------------------------------------------------------
# Team brain selection (supports exclusive split and mixed teams)
# ----------------------------------------------------------------------

_TEAM_BRAIN_MIX_COUNTER = {TEAM_RED_ID: 0, TEAM_BLUE_ID: 0}

def _make_team_mix_rng(team_id: float):
    seed = int(getattr(config, "TEAM_BRAIN_MIX_SEED", 0))
    if seed == 0:
        return random.SystemRandom()
    salt = 101 if team_id == TEAM_RED_ID else 202
    return random.Random(seed + salt)

_TEAM_BRAIN_MIX_RNG = {
    TEAM_RED_ID: _make_team_mix_rng(TEAM_RED_ID),
    TEAM_BLUE_ID: _make_team_mix_rng(TEAM_BLUE_ID),
}

def _resolve_team_brain_kind_from_team(team_id: float) -> str:
    mode = str(getattr(config, "TEAM_BRAIN_ASSIGNMENT_MODE", "exclusive")).strip().lower()

    if mode in ("exclusive", "split", "team"):
        if team_id == TEAM_RED_ID:
            return "tron"
        if team_id == TEAM_BLUE_ID:
            return "mirror"
        return str(getattr(config, "BRAIN_KIND", "tron")).strip().lower()

    if mode in ("mix", "hybrid", "both"):
        strategy = str(getattr(config, "TEAM_BRAIN_MIX_STRATEGY", "alternate")).strip().lower()

        if strategy in ("alternate", "roundrobin", "rr"):
            i = _TEAM_BRAIN_MIX_COUNTER.get(team_id, 0)
            _TEAM_BRAIN_MIX_COUNTER[team_id] = i + 1
            return "tron" if (i % 2 == 0) else "mirror"

        if strategy in ("random", "prob", "probabilistic"):
            p_tron = float(getattr(config, "TEAM_BRAIN_MIX_P_TRON", 0.5))
            p_tron = max(0.0, min(1.0, p_tron))
            rng = _TEAM_BRAIN_MIX_RNG.get(team_id, random.SystemRandom())
            return "tron" if (rng.random() < p_tron) else "mirror"

        raise ValueError(f"Unknown TEAM_BRAIN_MIX_STRATEGY={strategy!r}")

    raise ValueError(f"Unknown TEAM_BRAIN_ASSIGNMENT_MODE={mode!r}")

def _infer_kind_from_parent(parent: torch.nn.Module) -> Optional[str]:
    # ScriptModule counts as transformer for our purposes
    if isinstance(parent, torch.jit.ScriptModule):
        return "transformer"
    if isinstance(parent, TronBrain):
        return "tron"
    if isinstance(parent, MirrorBrain):
        return "mirror"
    if isinstance(parent, TransformerBrain):
        return "transformer"
    return None

# ----------------------------------------------------------------------
# Brain creation and cloning (preserves team‑specific assignment)
# ----------------------------------------------------------------------
def _new_brain(device: torch.device, *, team_id: Optional[float] = None) -> torch.nn.Module:
    """Create a new brain module.

    - Non-PPO: scripted transformer (existing behavior).
    - PPO:
        If TEAM_BRAIN_ASSIGNMENT is enabled and team_id is known:
            - exclusive: red=tron, blue=mirror (old)
            - mix: per-team mixed assignment (alternate/random)
        Otherwise: falls back to BRAIN_KIND.
    """
    obs_dim = int(getattr(config, "OBS_DIM", 0))
    act_dim = int(getattr(config, "NUM_ACTIONS", 41))

    is_ppo = bool(getattr(config, "PPO_ENABLED", False))
    if not is_ppo:
        return scripted_transformer_brain(obs_dim, act_dim).to(device)

    team_assign = bool(getattr(config, "TEAM_BRAIN_ASSIGNMENT", True))
    if team_assign and team_id is not None:
        brain_kind = _resolve_team_brain_kind_from_team(float(team_id))
    else:
        brain_kind = str(getattr(config, "BRAIN_KIND", "tron")).strip().lower()

    if brain_kind == "transformer":
        return TransformerBrain(obs_dim, act_dim).to(device)
    if brain_kind == "mirror":
        return MirrorBrain(obs_dim, act_dim).to(device)
    return TronBrain(obs_dim, act_dim).to(device)



def _clone_brain(
    parent: Optional[torch.nn.Module],
    device: torch.device,
    *,
    team_id: Optional[float] = None,
) -> torch.nn.Module:
    """Clone a parent brain.

    - Non-PPO: try deepcopy, else fall back to fresh.
    - PPO:
        * exclusive mode: may force architecture by team (old behavior)
        * mix mode: preserve parent's architecture when possible
    """
    if parent is None:
        return _new_brain(device, team_id=team_id)

    obs_dim = int(getattr(config, "OBS_DIM", 0))
    act_dim = int(getattr(config, "NUM_ACTIONS", 41))

    is_ppo = bool(getattr(config, "PPO_ENABLED", False))
    if not is_ppo:
        try:
            return copy.deepcopy(parent)
        except Exception:
            return _new_brain(device, team_id=team_id)

    team_assign = bool(getattr(config, "TEAM_BRAIN_ASSIGNMENT", True))
    mode = str(getattr(config, "TEAM_BRAIN_ASSIGNMENT_MODE", "exclusive")).strip().lower()

    parent_kind = _infer_kind_from_parent(parent)

    if team_assign and team_id is not None:
        # exclusive: old behavior (force by team)
        if mode in ("exclusive", "split", "team"):
            brain_kind = _resolve_team_brain_kind_from_team(float(team_id))
        else:
            # mix: keep parent architecture if we can
            brain_kind = parent_kind or _resolve_team_brain_kind_from_team(float(team_id))
    else:
        brain_kind = str(getattr(config, "BRAIN_KIND", "tron")).strip().lower()

    # Instantiate + load weights only if compatible
    if brain_kind == "transformer":
        child = TransformerBrain(obs_dim, act_dim).to(device)
        if isinstance(parent, (torch.jit.ScriptModule, TransformerBrain)):
            child.load_state_dict(parent.state_dict())
        return child

    if brain_kind == "mirror":
        child = MirrorBrain(obs_dim, act_dim).to(device)
        if isinstance(parent, MirrorBrain):
            child.load_state_dict(parent.state_dict())
        return child

    # Default: tron
    child = TronBrain(obs_dim, act_dim).to(device)
    if isinstance(parent, TronBrain):
        child.load_state_dict(parent.state_dict())
    return child



@torch.no_grad()
def _perturb_brain_(brain: torch.nn.Module, std: float) -> None:
    """Add small Gaussian noise to all trainable parameters."""
    if std <= 0.0:
        return
    for p in brain.parameters():
        if p.requires_grad:
            p.add_(torch.randn_like(p) * std)


# ----------------------------------------------------------------------
# Spawn cell selection (with wall margin)
# ----------------------------------------------------------------------
def _cell_free(grid: torch.Tensor, x: int, y: int, cfg: RespawnCfg) -> bool:
    """Check if a cell is free for spawning (no wall, no agent)."""
    H, W = grid.shape[1], grid.shape[2]
    return (cfg.wall_margin <= x < W - cfg.wall_margin and
            cfg.wall_margin <= y < H - cfg.wall_margin and
            grid[0, y, x].item() == 0.0 and      # empty (no wall, no agent)
            grid[2, y, x].item() == -1.0)         # slot id must be -1 for empty


def _pick_uniform(grid: torch.Tensor, cfg: RespawnCfg) -> Tuple[int, int]:
    """Randomly sample a free cell with uniform distribution."""
    H, W = grid.shape[1], grid.shape[2]
    for _ in range(cfg.spawn_tries):
        x = random.randint(cfg.wall_margin, W - cfg.wall_margin - 1)
        y = random.randint(cfg.wall_margin, H - cfg.wall_margin - 1)
        if _cell_free(grid, x, y, cfg):
            return x, y
    return -1, -1  


def _pick_location(grid: torch.Tensor, cfg: RespawnCfg) -> Tuple[int, int]:
    """Main entry point for spawn cell selection. Currently uniform."""
    return _pick_uniform(grid, cfg)


# ----------------------------------------------------------------------
# Unit type and stats (from config)
# ----------------------------------------------------------------------
def _choose_unit(cfg: RespawnCfg) -> int:
    """Randomly choose a unit type based on spawn_archer_ratio."""
    return cfg.unit_archer if random.random() < cfg.spawn_archer_ratio else cfg.unit_soldier


def _unit_stats(unit_id: int, cfg: RespawnCfg) -> Tuple[float, float, int]:
    """Return (hp, atk, vision_range) for the given unit type."""
    if unit_id == cfg.unit_archer:
        return cfg.archer_hp, cfg.archer_atk, cfg.vision_archer
    return cfg.soldier_hp, cfg.soldier_atk, cfg.vision_soldier


# ----------------------------------------------------------------------
# Write agent to registry (direct tensor assignment, preserves old behavior)
# ----------------------------------------------------------------------
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
) -> None:
    """Fill the registry tensors and brain list for a newly spawned agent."""
    reg.agent_data[slot, COL_ALIVE] = 1.0
    reg.agent_data[slot, COL_TEAM] = team_id
    reg.agent_data[slot, COL_X] = float(x)
    reg.agent_data[slot, COL_Y] = float(y)
    reg.agent_data[slot, COL_HP] = hp
    reg.agent_data[slot, COL_HP_MAX] = hp
    reg.agent_data[slot, COL_VISION] = float(vision)
    reg.agent_data[slot, COL_ATK] = atk
    reg.agent_data[slot, COL_UNIT] = float(unit_id)

    # B7: Assign a fresh agent_id for this newly spawned individual.
    # get_next_id() is assumed to be provided by AgentsRegistry.
    new_aid = reg.get_next_id()
    # Store true UID in int64 side-tensor (robust even if agent_data is float16).
    if hasattr(reg, "agent_uids"):
        reg.agent_uids[slot] = int(new_aid)
    # Keep legacy float column as display-only (clamp to avoid inf when dtype is float16).
    try:
        max_f = torch.finfo(reg.agent_data.dtype).max
        reg.agent_data[slot, COL_AGENT_ID] = float(min(float(new_aid), float(max_f)))
    except Exception:
        reg.agent_data[slot, COL_AGENT_ID] = float(new_aid)

    # Replace brain and clear optimizer (optimizer will be recreated by PPO if needed)
    reg.brains[slot] = brain
    if hasattr(reg, "optimizers"):
        reg.optimizers[slot] = None


# ----------------------------------------------------------------------
# Core respawn function (spawns a given number of agents for a team)
# ----------------------------------------------------------------------
@torch.no_grad()
def _respawn_some(
    reg: AgentsRegistry,
    grid: torch.Tensor,
    team_id: float,
    count: int,
    cfg: RespawnCfg,
    tick: int,                                 # B5: added tick parameter
    meta_out: Optional[List[Dict[str, object]]] = None,   # B5: added meta_out parameter
) -> int:
    """
    Attempt to spawn up to `count` agents of the given team.

    Returns the number actually spawned (may be less due to lack of free cells).
    """
    global _respawn_counter

    if count <= 0:
        return 0

    # Find dead slots (alive == 0) that belong to the same team (optional, but safe)
    data = reg.agent_data
    alive = (data[:, COL_ALIVE] > 0.5)
    dead_slots = (~alive).nonzero(as_tuple=False).squeeze(1)
    if dead_slots.numel() == 0:
        return 0

    # Gather potential parents (alive agents of the same team)
    parents = (alive & (data[:, COL_TEAM] == team_id)).nonzero(as_tuple=False).squeeze(1)

    spawned = 0
    # We will iterate over dead slots sequentially; stop when we have spawned enough or run out.
    for k in range(min(count, dead_slots.numel())):
        slot = int(dead_slots[k].item())
        x, y = _pick_location(grid, cfg)
        if x < 0:  # no free cell found
            break

        # Choose unit type and base stats
        unit_id = _choose_unit(cfg)
        hp0, atk0, vision0 = _unit_stats(unit_id, cfg)

        # Rare mutation event (every 1000th respawn globally)
        _respawn_counter += 1
        if _respawn_counter % 1000 == 0:
            hp0 *= (1.0 + random.uniform(0.5, 2.0))
            atk0 *= (1.0 + random.uniform(0.5, 2.0))
            vision0 = int(vision0 * (1.0 + random.uniform(0.5, 2.0)))
            print(f"** Rare Mutation on slot {slot} (team {'red' if team_id==TEAM_RED_ID else 'blue'})! "
                  f"HP:{hp0:.2f}, ATK:{atk0:.2f}, VIS:{vision0} **")

        # Decide whether to clone an existing parent or create a fresh brain
        use_clone = (parents.numel() > 0) and (random.random() < cfg.clone_prob)
        pj = None  # B8: define pj here so it's accessible later for metadata
        if use_clone:
            pj = int(parents[random.randrange(parents.numel())].item())
            brain = _clone_brain(reg.brains[pj], reg.device, team_id=team_id)
            _perturb_brain_(brain, cfg.mutation_std)
        else:
            brain = _new_brain(reg.device, team_id=team_id)

        # Write to registry
        _write_agent_to_registry(reg, slot, team_id, x, y, unit_id, hp0, atk0, vision0, brain)

        # B8: Telemetry hook: capture spawn metadata without affecting simulation.
        if meta_out is not None:
            # Read true UID from int64 side-tensor (never overflows/rounds).
            if hasattr(reg, "agent_uids"):
                child_aid = int(reg.agent_uids[slot].item())
                parent_aid = int(reg.agent_uids[pj].item()) if use_clone else None
            else:
                # Fallback for older registries (may be unsafe under float16).
                child_aid = int(reg.agent_data[slot, COL_AGENT_ID].item())
                parent_aid = int(reg.agent_data[pj, COL_AGENT_ID].item()) if use_clone else None
            meta_out.append({
                "tick": int(tick),
                "slot": int(slot),
                "agent_id": int(child_aid),
                "team_id": float(team_id),
                "unit_id": int(unit_id),
                "x": int(x),
                "y": int(y),
                "cloned": bool(use_clone),
                "parent_slot": int(pj) if use_clone else None,
                "parent_agent_id": int(parent_aid) if parent_aid is not None else None,
                "mutation_std": float(cfg.mutation_std),
            })

        # Update grid (MUST MATCH spawn.py + raycast contract):
        #   grid[0]: occupancy (0 empty, 1 wall, 2 red, 3 blue)
        #   grid[1]: hp
        #   grid[2]: slot id (agent index), -1 for empty
        grid[0, y, x] = float(team_id)
        grid[1, y, x] = float(hp0)
        grid[2, y, x] = float(slot)

        spawned += 1

    return spawned


# ----------------------------------------------------------------------
# RespawnController: advanced team‑aware respawn logic
# ----------------------------------------------------------------------
class RespawnController:
    """Controller that manages floor‑based and periodic respawning."""

    def __init__(self, cfg: RespawnCfg):
        self.cfg = cfg
        self._cooldown_red_until = 0
        self._cooldown_blue_until = 0
        self._last_period_tick = 0
        # B3: Filled each step() with metadata for newly spawned agents (for telemetry).
        self.last_spawn_meta: List[Dict[str, object]] = []

    def step(self, tick: int, reg: AgentsRegistry, grid: torch.Tensor) -> Tuple[int, int]:
        """
        Execute one step of the respawn logic.

        Returns:
            Tuple (spawned_red, spawned_blue) with the numbers of agents spawned this tick.
        """
        # B4: reset per-step spawn metadata
        self.last_spawn_meta = []

        if not self.cfg.enabled:
            return 0, 0

        counts = _team_counts(reg)
        spawned_r = spawned_b = 0

        # Floor‑based respawn (with cooldown)
        if counts.red < self.cfg.floor_per_team and tick >= self._cooldown_red_until:
            need = self.cfg.floor_per_team - counts.red
            # B6: pass tick and meta_out
            spawned = _respawn_some(reg, grid, TEAM_RED_ID, _cap(need, self.cfg), self.cfg, tick, self.last_spawn_meta)
            spawned_r += spawned
            if counts.red + spawned >= self.cfg.floor_per_team:
                self._cooldown_red_until = tick + self.cfg.cooldown_ticks

        if counts.blue < self.cfg.floor_per_team and tick >= self._cooldown_blue_until:
            need = self.cfg.floor_per_team - counts.blue
            # B6: pass tick and meta_out
            spawned = _respawn_some(reg, grid, TEAM_BLUE_ID, _cap(need, self.cfg), self.cfg, tick, self.last_spawn_meta)
            spawned_b += spawned
            if counts.blue + spawned >= self.cfg.floor_per_team:
                self._cooldown_blue_until = tick + self.cfg.cooldown_ticks

        # Periodic respawn (inverse proportional to current team sizes)
        if tick - self._last_period_tick >= self.cfg.period_ticks:
            self._last_period_tick = tick
            total_alive = counts.red + counts.blue
            if total_alive > 0:
                q_r, q_b = _inverse_split(counts.red, counts.blue, self.cfg.period_budget)
                # B6: pass tick and meta_out for both calls
                spawned_r += _respawn_some(reg, grid, TEAM_RED_ID, _cap(q_r, self.cfg), self.cfg, tick, self.last_spawn_meta)
                spawned_b += _respawn_some(reg, grid, TEAM_BLUE_ID, _cap(q_b, self.cfg), self.cfg, tick, self.last_spawn_meta)

        return spawned_r, spawned_b


# ----------------------------------------------------------------------
# Public API: respawn_tick (backward compatible with old code)
# ----------------------------------------------------------------------
def respawn_tick(reg: AgentsRegistry, grid: torch.Tensor, cfg: RespawnCfg) -> None:
    """
    Perform one tick of respawning.

    This function maintains the original signature. It creates a temporary
    RespawnController and runs its step. The old probabilistic fields
    (prob_per_dead_per_tick) are ignored; the new controller uses the
    floor/period logic defined in the extended RespawnCfg.
    """
    controller = RespawnController(cfg)
    controller.step(0, reg, grid)  # tick number is not critical for one‑off calls
    # Note: step returns counts, but we ignore them to match old API.