"""Core tick loop for simulation state updates."""

from __future__ import annotations

from dataclasses import dataclass
import collections
import os
from typing import Dict, Optional, List, Tuple, TYPE_CHECKING

# PyTorch is used for tensor math and GPU acceleration.
import torch

# Project imports (simulation config + engine subsystems)
import config
from simulation.stats import SimulationStats
from engine.agent_registry import (
    AgentsRegistry,
    COL_ALIVE, COL_TEAM, COL_X, COL_Y, COL_HP, COL_ATK, COL_UNIT, COL_VISION, COL_HP_MAX, COL_AGENT_ID
)
from engine.ray_engine.raycast_32 import raycast32_firsthit
from engine.game.move_mask import build_mask, DIRS8
from engine.respawn import RespawnController, RespawnCfg
from engine.mapgen import Zones
from engine.catastrophe import CatastropheRuntimeSignal, HealZoneCatastropheController

from agent.ensemble import ensemble_forward


# TYPE_CHECKING is True only for static analysis / IDE type checking.
# This avoids runtime circular imports while still enabling autocomplete.
if TYPE_CHECKING:
    from rl.ppo_runtime import PerAgentPPORuntime

# Optional PPO runtime import:
# The PPO runtime might not exist / might fail to import in some deployments.
# In that case, PPO is disabled at runtime.
try:
    from rl.ppo_runtime import PerAgentPPORuntime as _PerAgentPPORuntimeRT
except Exception:
    _PerAgentPPORuntimeRT = None


# TickMetrics: per-tick counters (telemetry-like numeric metrics)
# A dataclass automatically generates:
#   - __init__
#   - readable __repr__
#   - comparisons (optional)
# Here it is used as a simple “struct” to track what happened in one tick.
# NOTE ABOUT DESIGN:
# - These are cheap scalar aggregates (Python ints/floats) updated once per tick.
# - They are meant for printing/logging/telemetry dashboards, not for training.
# - Keeping them minimal avoids overhead in the hot loop.
@dataclass
class TickMetrics:
    """Holds counters for one simulation tick."""
    alive: int = 0
    moved: int = 0
    attacks: int = 0
    deaths: int = 0

    # Death breakdown (explicit semantics; helps telemetry/forensics)
    deaths_combat: int = 0
    deaths_metabolism: int = 0
    deaths_environmental: int = 0
    deaths_collision: int = 0
    deaths_unknown: int = 0

    tick: int = 0
    cp_red_tick: float = 0.0
    cp_blue_tick: float = 0.0

    # Movement (cheap aggregates; see telemetry/move_summary.csv)
    move_attempted: int = 0
    move_can_move: int = 0
    move_blocked_wall: int = 0
    move_blocked_occupied: int = 0
    move_conflict_lost: int = 0
    move_conflict_tie: int = 0


# TickEngine: the main simulation stepper
# Responsibilities per tick:
#   1) Build observations for alive agents
#   2) Build action masks (legal actions)
#   3) Run policy inference (neural net forward pass)
#   4) Resolve combat (damage, kills)
#   5) Apply deaths (remove dead agents from grid + registry)
#   6) Resolve movement (conflicts, grid updates)
#   7) Environment effects (healing, metabolism, capture points)
#   8) PPO logging (if enabled)
#   9) Respawn dead agents
# IMPORTANT: This implementation emphasizes:
#   - GPU parallelism
#   - deterministic aggregation where duplicate indices occur (e.g., multi-attacker damage)
#   - invariant checks (optional, debug mode)
class TickEngine:
    # CORE THEORY: THE GAME LOOP (Tick Engine)
    # A simulation runs in "Ticks" or frames. Each tick, the engine looks at the
    # world, decides what every entity should do, and then applies the rules of
    # physics/gameplay. This class manages that massive update.
    # Responsibilities:
    #   - Process attacks (damage, kills)
    #   - Apply deaths (combat-first)
    #   - Move agents (with conflict resolution)
    #   - Apply zone healing and capture point scoring
    #   - Record telemetry and PPO data
    #   - Respawn dead agents

    def __init__(
        self,
        registry: AgentsRegistry,
        grid: torch.Tensor,
        stats: SimulationStats,
        zones: Optional[Zones] = None
    ) -> None:
        """
        Initialize the tick engine.

        Args:
            registry:
                Holds all agent data (positions, health, team, etc.).
                In typical designs, this includes a dense tensor:
                    registry.agent_data: shape (capacity, num_features)

            grid:
                3-layer grid for the map. Common interpretation here:
                  grid[0] = occupancy / team encoding / walls (float)
                  grid[1] = HP channel (float)
                  grid[2] = slot id channel (float, but used as int-like)
                Shape generally: (3, H, W)

            stats:
                Global simulation statistics (kills, deaths, scores, tick counter).

            zones:
                Optional zone masks (healing areas, capture points).
        """
        self.registry = registry
        self.grid = grid
        self.stats = stats

        # The device is typically 'cuda' or 'cpu'.
        # Keeping tensors on GPU avoids CPU<->GPU transfer overhead.
        self.device = grid.device

        # Grid dimensions:
        # grid.size(1) = H, grid.size(2) = W for grid shaped (C, H, W)
        self.H, self.W = int(grid.size(1)), int(grid.size(2))

        # Respawn system: restores dead agents back into the environment
        # according to RespawnCfg parameters.
        self.respawner = RespawnController(RespawnCfg())

        # Per-agent cumulative doctrine/personal merit keyed by persistent UID:
        # - agent_scores: accumulated personal reward/merit
        # - agent_kill_counts: lifetime credited kills
        # - agent_cp_points: lifetime objective-control share
        self.agent_scores: Dict[int, float] = collections.defaultdict(float)
        self.agent_kill_counts: Dict[int, float] = collections.defaultdict(float)
        self.agent_cp_points: Dict[int, float] = collections.defaultdict(float)

        # Zones and corresponding cached tensors on the simulation device.
        self.zones: Optional[Zones] = zones
        self._z_heal: Optional[torch.Tensor] = None
        self._z_cp_masks: List[torch.Tensor] = []
        self._zone_runtime_revision: Optional[int] = None
        self._ensure_zone_tensors()

        # Catastrophe runtime owner. It only controls the Zones catastrophe slot
        # and never touches PPO / reward / observation semantics.
        self.catastrophe_controller = HealZoneCatastropheController(self.zones)

        # DIRS8 is a standard 8-neighborhood direction table:
        # N, NE, E, SE, S, SW, W, NW
        # We copy it to device for fast indexing.
        self.DIRS8_dev = DIRS8.to(self.device)

        # Action / observation dimensions from config.
        self._ACTIONS = int(getattr(config, "NUM_ACTIONS", 41))
        self._OBS_DIM = config.OBS_DIM

        # Cache dtypes:
        # - grid dtype (often float32/float16)
        # - registry data dtype (often float32/float16)
        self._grid_dt = self.grid.dtype
        self._data_dt = self.registry.agent_data.dtype
        # Rationale: run_tick() creates 4 fresh GPU tensors (zeros/full, H×W) every tick
        # inside the movement hot path. Preallocating once and using .zero_()/.fill_()
        # in-place eliminates the per-tick allocator overhead.
        _nc = self.H * self.W
        self._move_claim_cnt = torch.zeros(_nc, device=self.device, dtype=torch.int32)
        self._move_max_hp    = torch.zeros(_nc, device=self.device, dtype=self.registry.agent_data.dtype)
        self._move_max_cnt   = torch.zeros(_nc, device=self.device, dtype=torch.int32)
        self._move_win_cnt   = torch.zeros(_nc, device=self.device, dtype=torch.int32)
        # Cached scalar constant used in movement occupancy check.
        self._g_wall = torch.tensor(1.0, device=self.device, dtype=self._grid_dt)

        # Preallocated per-tick reward / observation scratch
        # These tensors are capacity-bounded and reused in-place every tick.
        # This removes several hot torch.zeros/torch.empty allocations from run_tick()
        # and _build_transformer_obs() without changing what the simulation computes.
        self._capacity = int(self.registry.capacity)
        self._reward_individual_total = torch.zeros(self._capacity, device=self.device, dtype=self._data_dt)
        self._reward_kill_individual = torch.zeros(self._capacity, device=self.device, dtype=torch.float32)
        self._reward_damage_dealt_individual = torch.zeros(self._capacity, device=self.device, dtype=torch.float32)
        self._reward_damage_taken_penalty = torch.zeros(self._capacity, device=self.device, dtype=torch.float32)
        self._reward_contested_cp_individual = torch.zeros(self._capacity, device=self.device, dtype=torch.float32)
        self._reward_healing_recovered = torch.zeros(self._capacity, device=self.device, dtype=torch.float32)

        self._obs_on_heal = torch.zeros(self._capacity, device=self.device, dtype=torch.bool)
        self._obs_on_cp = torch.zeros(self._capacity, device=self.device, dtype=torch.bool)
        self._obs_rich_base = torch.empty((self._capacity, 23), device=self.device, dtype=self._data_dt)
        self._obs_rich = torch.empty(
            (self._capacity, int(self._OBS_DIM) - (32 * 8)),
            device=self.device,
            dtype=self._data_dt,
        )

        # Instinct cache / scratch (computed under no_grad)
        # "Instinct" here is a computed feature: local density of allies/enemies
        # around each agent within a radius R. Computing offsets for a discrete
        # circle can be expensive, so we cache offsets by radius.
        # The (N,M) scratch tensors are allocated lazily because M depends on the
        # configured instinct radius. They are sliced to the live alive-count each call.
        self._instinct_cached_r: int = -999999
        self._instinct_offsets: Optional[torch.Tensor] = None
        self._instinct_area: float = 1.0
        self._instinct_scratch_m: int = 0
        self._instinct_xx: Optional[torch.Tensor] = None
        self._instinct_yy: Optional[torch.Tensor] = None
        self._instinct_ally_mask: Optional[torch.Tensor] = None
        self._instinct_enemy_mask: Optional[torch.Tensor] = None
        self._instinct_ally_arch_mask: Optional[torch.Tensor] = None
        self._instinct_ally_sold_mask: Optional[torch.Tensor] = None
        self._instinct_ally_occ: Optional[torch.Tensor] = None
        self._instinct_enemy_occ: Optional[torch.Tensor] = None
        self._instinct_noise = torch.empty((self._capacity,), device=self.device, dtype=torch.float32)
        self._instinct_out = torch.empty((self._capacity, 4), device=self.device, dtype=self._data_dt)

        # THEORY: Constants on GPU
        # Creating small tensors repeatedly can be expensive and can cause
        # device synchronization overhead. So we allocate frequently used
        # constants once on the correct device and dtype.
        self._g0 = torch.tensor(0.0, device=self.device, dtype=self._grid_dt)     # grid “zero”
        self._gneg = torch.tensor(-1.0, device=self.device, dtype=self._grid_dt)  # grid “-1”
        self._d0 = torch.tensor(0.0, device=self.device, dtype=self._data_dt)     # data “zero”

        # PPO integration
        # PPO_ENABLED: feature flag in config.
        # If enabled and runtime class is importable, initialize PPO runtime.
        self._ppo_enabled = bool(getattr(config, "PPO_ENABLED", False))
        self._ppo: Optional["PerAgentPPORuntime"] = None
        if self._ppo_enabled and _PerAgentPPORuntimeRT is not None:
            self._ppo = _PerAgentPPORuntimeRT(
                registry=self.registry,
                device=self.device,
                obs_dim=self._OBS_DIM,
                act_dim=self._ACTIONS,
            )

    # Per-tick scratch reset helpers
    def _reset_tick_reward_buffers(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Reset and return the capacity-sized reward buffers reused by run_tick().

        Safety rule:
        - Every buffer is zeroed at tick start before any reward accumulation.
        - The returned tensors are the canonical per-tick reward stores for this engine.
        """
        self._reward_individual_total.zero_()
        self._reward_kill_individual.zero_()
        self._reward_damage_dealt_individual.zero_()
        self._reward_damage_taken_penalty.zero_()
        self._reward_contested_cp_individual.zero_()
        self._reward_healing_recovered.zero_()
        return (
            self._reward_individual_total,
            self._reward_kill_individual,
            self._reward_damage_dealt_individual,
            self._reward_damage_taken_penalty,
            self._reward_contested_cp_individual,
            self._reward_healing_recovered,
        )

    # PPO: reset state for agents that just respawned
    def _ppo_reset_on_respawn(self, was_dead: torch.Tensor) -> None:
        """
        Reset per-slot PPO state for any slot that was dead before respawn
        and is alive after.

        Why reset?
        - PPO maintains per-agent buffers / running statistics.
        - When an agent respawns, its “episode continuity” may break.
        - Resetting prevents training on invalid transitions.
        """
        if self._ppo is None:
            return

        data = self.registry.agent_data

        # now_alive: boolean mask of alive agents AFTER respawn.
        now_alive = (data[:, COL_ALIVE] > 0.5)

        # spawned_slots: were dead AND now alive.
        spawned_slots = (was_dead & now_alive).nonzero(as_tuple=False).squeeze(1)
        if spawned_slots.numel() == 0:
            return

        self._ppo.reset_agents(spawned_slots)

        # Optional logging.
        if bool(getattr(config, "PPO_RESET_LOG", False)):
            sl = spawned_slots[:16].tolist()
            suffix = "" if spawned_slots.numel() <= 16 else "..."
            print(f"[ppo] reset state for {int(spawned_slots.numel())} respawned slots: {sl}{suffix}")

    # Zones: ensure masks are on the correct device
    def _ensure_zone_tensors(self) -> None:
        """
        Convert zone masks to tensors on the simulation device.

        - heal_mask: boolean grid marking cells that heal agents standing on them
        - cp_masks: list of boolean grids marking capture point regions
        """
        self._z_heal, self._z_cp_masks = None, []
        self._zone_runtime_revision = None
        if self.zones is None:
            return

        try:
            # non_blocking=True can allow overlap of copy with CPU work if pinned memory.
            if getattr(self.zones, "heal_mask", None) is not None:
                self._z_heal = self.zones.heal_mask.to(self.device, non_blocking=True).bool()

            self._z_cp_masks = [
                m.to(self.device, non_blocking=True).bool()
                for m in getattr(self.zones, "cp_masks", [])
            ]
            self._zone_runtime_revision = self._current_zone_runtime_revision()
        except Exception as e:
            # Safety: if zones fail, disable them rather than crash the simulation.
            print(f"[tick] WARN: zone tensor setup failed ({e}); zones disabled.")


    def refresh_zone_runtime_cache(self) -> None:
        """
        Rebuild cached device-side zone tensors after runtime zone-state changes.

        This is the single engine hook used by catastrophe scheduling so the
        effective heal mask consumed by gameplay stays aligned with the Zones
        source of truth.
        """
        self._ensure_zone_tensors()

    def _current_zone_runtime_revision(self) -> Optional[int]:
        if self.zones is None:
            return None
        revision = getattr(self.zones, "runtime_heal_revision", None)
        return None if revision is None else int(revision)

    def _refresh_zone_runtime_cache_if_needed(self) -> None:
        """
        Detect out-of-band heal-zone changes (for example viewer/manual actions)
        and refresh cached device masks before gameplay consumes them.
        """
        current_revision = self._current_zone_runtime_revision()
        if current_revision is None:
            return
        if self._zone_runtime_revision != current_revision:
            self.refresh_zone_runtime_cache()

    def _build_catastrophe_runtime_signal(self, alive_idx: torch.Tensor) -> CatastropheRuntimeSignal:
        """
        Build a small interpretable signal bundle for optional dynamic triggering.
        """
        alive_count = int(alive_idx.numel())
        on_heal_count = 0

        if alive_count > 0 and self._z_heal is not None:
            pos_xy = self.registry.positions_xy(alive_idx)
            on_heal = self._z_heal[pos_xy[:, 1], pos_xy[:, 0]]
            on_heal_count = int(on_heal.to(torch.int64).sum().item())

        return CatastropheRuntimeSignal(
            tick=int(self.stats.tick),
            alive_count=alive_count,
            on_heal_count=on_heal_count,
        )

    # Indexing helper
    @staticmethod
    def _as_long(x: torch.Tensor) -> torch.Tensor:
        """
        Convert tensor to long dtype (used for indexing).

        PyTorch indexing generally requires integer types (torch.long).
        If you attempt to index with float tensors, it will error.
        """
        return x.to(torch.long)

    # Identity helpers (SLOT ID vs PERSISTENT AGENT UID)
    def _slot_ids_to_agent_uids_list(self, slot_idx: torch.Tensor) -> List[int]:
        """
        Convert registry slot indices -> persistent agent ids (UIDs) for telemetry/analysis.

        IMPORTANT:
        - slot index = runtime storage location in registry (reused after death/respawn)
        - agent UID   = persistent identity across lifetime (preferred for telemetry)

        Fallback behavior:
        - If registry.agent_uids does not exist, we fall back to COL_AGENT_ID.
          COL_AGENT_ID is treated as display/compat field and may be float-backed.
        """
        if slot_idx.numel() == 0:
            return []

        if hasattr(self.registry, "agent_uids"):
            return (
                self.registry.agent_uids
                .index_select(0, slot_idx)
                .detach()
                .cpu()
                .to(torch.int64)
                .tolist()
            )

        data = self.registry.agent_data
        return (
            data[slot_idx, COL_AGENT_ID]
            .detach()
            .cpu()
            .to(torch.int64)
            .tolist()
        )

    # Grid HP sync helper (minimize grid/registry desync windows)
    def _sync_grid_hp_for_slots(self, slot_idx: torch.Tensor) -> None:
        """
        Sync grid HP channel (grid[1]) from registry HP for the given slots.

        Why this exists:
        - Combat/environment often mutate registry HP first.
        - We want a single, explicit helper to shrink the "registry updated but grid HP stale"
          window and make code review easier.
        """
        if slot_idx.numel() == 0:
            return
        data = self.registry.agent_data
        gx = self._as_long(data[slot_idx, COL_X])
        gy = self._as_long(data[slot_idx, COL_Y])
        self.grid[1, gy, gx] = data[slot_idx, COL_HP].to(self._grid_dt)

    # Optional telemetry phase context helper (safe no-op if telemetry lacks API)
    def _telemetry_set_phase(self, telemetry, *, tick: int, phase: str) -> None:
        """
        Best-effort phase marker for telemetry event ordering.
        This is additive only; if telemetry doesn't support it, nothing breaks.
        """
        if telemetry is None or not getattr(telemetry, "enabled", False):
            return
        try:
            if hasattr(telemetry, "begin_tick_event_context"):
                telemetry.begin_tick_event_context(int(tick))
            if hasattr(telemetry, "set_event_phase"):
                telemetry.set_event_phase(str(phase), tick=int(tick))
        except Exception:
            # Telemetry should never destabilize sim.
            pass

    # Alive filtering
    def _recompute_alive_idx(self) -> torch.Tensor:
        """
        Returns a 1D tensor of slot indices where the agent is alive.

        Performance rationale:
        - Many tensors are capacity-sized, but only a subset are alive.
        - Filtering early reduces compute cost in later phases.
        """
        return (self.registry.agent_data[:, COL_ALIVE] > 0.5).nonzero(as_tuple=False).squeeze(1)

    # Instinct scratch: lazily sized (capacity, M) reusable buffers
    def _ensure_instinct_scratch(self, min_offsets: int) -> None:
        """
        Ensure the reusable instinct scratch tensors can hold (capacity, min_offsets).

        Why lazy sizing?
        - The instinct offset count M depends on the configured radius.
        - Capacity is fixed, but M can change if config changes between runs/resumes.

        Safety rule:
        - We allocate once per new required width and then always slice [:N, :M].
        - Callers must still fully overwrite or zero the used slices before reading them.
        """
        need_m = max(1, int(min_offsets))
        if self._instinct_xx is not None and self._instinct_scratch_m >= need_m:
            return

        shape_nm = (self._capacity, need_m)
        self._instinct_xx = torch.empty(shape_nm, device=self.device, dtype=torch.long)
        self._instinct_yy = torch.empty(shape_nm, device=self.device, dtype=torch.long)
        self._instinct_ally_mask = torch.empty(shape_nm, device=self.device, dtype=torch.bool)
        self._instinct_enemy_mask = torch.empty(shape_nm, device=self.device, dtype=torch.bool)
        self._instinct_ally_arch_mask = torch.empty(shape_nm, device=self.device, dtype=torch.bool)
        self._instinct_ally_sold_mask = torch.empty(shape_nm, device=self.device, dtype=torch.bool)
        self._instinct_ally_occ = torch.empty((self._capacity,), device=self.device, dtype=self._grid_dt)
        self._instinct_enemy_occ = torch.empty((self._capacity,), device=self.device, dtype=self._grid_dt)
        self._instinct_scratch_m = need_m

    # Instinct offsets: cached discrete circle offsets
    @torch.no_grad()
    def _get_instinct_offsets(self) -> Tuple[torch.Tensor, float]:
        """
        Returns cached integer (dx, dy) offsets inside a discrete circle of radius R cells,
        plus the offset-count area used for density normalization.

        Math note:
        - Circle condition in grid coordinates:
            dx^2 + dy^2 <= R^2
          This is directly from the Pythagorean theorem.
        """
        R = int(getattr(config, "INSTINCT_RADIUS", 6))
        if R < 0:
            R = 0

        # Cache miss if radius changed or offsets not yet built.
        if self._instinct_offsets is None or self._instinct_cached_r != R:
            if R == 0:
                # Only self-cell offset.
                offsets = torch.zeros((1, 2), device=self.device, dtype=torch.long)
            else:
                # Build a square grid [-R..R] x [-R..R]
                r = torch.arange(-R, R + 1, device=self.device, dtype=torch.long)

                # meshgrid creates matrices of dx and dy coordinates.
                # indexing="xy" ensures (x,y) convention.
                dx, dy = torch.meshgrid(r, r, indexing="xy")

                # Circle mask: keep offsets within radius.
                mask = (dx * dx + dy * dy) <= (R * R)

                # Extract the offsets (M,2) where M is number of points in the discrete circle.
                offsets = torch.stack([dx[mask], dy[mask]], dim=1).contiguous()

                # Safety: ensure we never return empty offsets.
                if offsets.numel() == 0:
                    offsets = torch.zeros((1, 2), device=self.device, dtype=torch.long)

            self._instinct_offsets = offsets
            self._instinct_area = float(int(offsets.size(0)))  # used to normalize counts into densities
            self._instinct_cached_r = R

        return self._instinct_offsets, self._instinct_area

    # Instinct feature computation
    @torch.no_grad()
    def _compute_instinct_context(
        self,
        alive_idx: torch.Tensor,
        pos_xy: torch.Tensor,
        unit_map: torch.Tensor,
    ) -> torch.Tensor:
        """
        Instinct token (4 floats) per alive agent:

          1) ally_archer_density
          2) ally_soldier_density
          3) noisy_enemy_density
          4) threat_ratio = enemy_density / (ally_total_density + eps)

        Interpretation:
        - The agent gets a “local summary” of nearby population.
        - This helps learning: the policy can condition on “crowding” and “threat”.

        Density math:
        - We count how many neighbors of each type exist within radius offsets,
          then divide by area (number of offsets) to obtain a density-like value.

        Noise:
        - enemy count is perturbed by Gaussian noise to simulate uncertain perception.
        """
        N = int(alive_idx.numel())
        if N == 0:
            return torch.empty((0, 4), device=self.device, dtype=self._data_dt)

        data = self.registry.agent_data
        offsets, area = self._get_instinct_offsets()
        M = int(offsets.size(0))
        out = self._instinct_out[:N]
        if M <= 0 or area <= 0.0:
            out.zero_()
            return out

        self._ensure_instinct_scratch(M)
        xx = self._instinct_xx[:N, :M]
        yy = self._instinct_yy[:N, :M]
        ally_mask = self._instinct_ally_mask[:N, :M]
        enemy_mask = self._instinct_enemy_mask[:N, :M]
        ally_arch = self._instinct_ally_arch_mask[:N, :M]
        ally_sold = self._instinct_ally_sold_mask[:N, :M]

        # Broadcasting (important tensor technique)
        # x0: (N,1), y0: (N,1)
        # ox: (1,M), oy: (1,M)
        # x0+ox => (N,M), y0+oy => (N,M)
        # This computes all neighborhood coordinates for all agents at once.
        x0 = pos_xy[:, 0].to(torch.long).view(N, 1)
        y0 = pos_xy[:, 1].to(torch.long).view(N, 1)
        ox = offsets[:, 0].view(1, M)
        oy = offsets[:, 1].view(1, M)

        # Clamp to map bounds to avoid indexing outside the grid.
        torch.add(x0, ox, out=xx)
        xx.clamp_(0, self.W - 1)
        torch.add(y0, oy, out=yy)
        yy.clamp_(0, self.H - 1)

        # occ: occupancy/team at each sampled cell
        # uid: unit id at each sampled cell
        occ = self.grid[0][yy, xx]
        uid = unit_map[yy, xx]

        # teams for each alive agent
        teams = data[alive_idx, COL_TEAM]
        team_is_red = (teams == 2.0)

        # Determine ally and enemy occupancy codes for each agent.
        # These values (2.0 and 3.0) are project conventions for team encoding.
        ally_occ = self._instinct_ally_occ[:N]
        enemy_occ = self._instinct_enemy_occ[:N]
        ally_occ.fill_(3.0)
        ally_occ[team_is_red] = 2.0
        enemy_occ.fill_(2.0)
        enemy_occ[team_is_red] = 3.0

        torch.eq(occ, ally_occ.view(N, 1), out=ally_mask)
        torch.eq(occ, enemy_occ.view(N, 1), out=enemy_mask)

        # Unit types:
        # 1 = soldier, 2 = archer (as implied by later checks)
        torch.eq(uid, 2, out=ally_arch)
        torch.logical_and(ally_mask, ally_arch, out=ally_arch)
        torch.eq(uid, 1, out=ally_sold)
        torch.logical_and(ally_mask, ally_sold, out=ally_sold)

        # Count nearby allies/enemies per agent by summing along offset dimension.
        ally_arch_c = ally_arch.sum(dim=1).to(torch.float32)
        ally_sold_c = ally_sold.sum(dim=1).to(torch.float32)
        enemy_c = enemy_mask.sum(dim=1).to(torch.float32)

        # Remove self-count (an agent should not count itself as a neighbor).
        self_unit = data[alive_idx, COL_UNIT]
        ally_arch_c = (ally_arch_c - (self_unit == 2.0).to(torch.float32)).clamp_min(0.0)
        ally_sold_c = (ally_sold_c - (self_unit == 1.0).to(torch.float32)).clamp_min(0.0)

        # Add noise to enemy count (fog-of-war style uncertainty).
        noise = self._instinct_noise[:N]
        torch.randn((N,), out=noise)
        noise.mul_(0.25)
        enemy_c_noisy = (enemy_c + noise).clamp_min(0.0)

        # Normalize by area to convert counts -> densities.
        inv_area = 1.0 / float(area)
        ally_arch_d = ally_arch_c * inv_area
        ally_sold_d = ally_sold_c * inv_area
        enemy_d = enemy_c_noisy * inv_area

        # Avoid division by zero: eps depends on dtype precision.
        eps = 1e-4 if self._data_dt == torch.float16 else 1e-6
        ally_total_d = ally_arch_d + ally_sold_d
        threat = enemy_d / (ally_total_d + eps)

        out[:, 0].copy_(ally_arch_d.to(self._data_dt))
        out[:, 1].copy_(ally_sold_d.to(self._data_dt))
        out[:, 2].copy_(enemy_d.to(self._data_dt))
        out[:, 3].copy_(threat.to(self._data_dt))
        return out

    # Death application: remove dead agents from grid and update stats/metrics
    def _apply_deaths(
        self,
        sel: torch.Tensor,
        metrics: TickMetrics,
        credit_kills: bool = True,
        death_cause: str = "unknown",
        killer_slot_by_victim: Optional[Dict[int, int]] = None,
    ) -> Tuple[int, int]:
        """
        Kill agents indicated by sel (boolean mask or index tensor).
        Updates grid, agent data, and metrics. Returns (red_deaths, blue_deaths).

        Important semantics:
        - "credit_kills" controls whether deaths increase opponent kill counters.
          Example: starvation/metabolism deaths should typically not be credited as enemy kills.
        - "death_cause" is a telemetry/forensics label ("combat", "metabolism", ...).
        """
        data = self.registry.agent_data

        # sel can be either:
        #   - bool mask of shape (capacity,)
        #   - explicit indices
        dead_idx = sel.nonzero(as_tuple=False).squeeze(1) if sel.dtype == torch.bool else sel.view(-1)

        # Early exit: nobody died.
        if dead_idx.numel() == 0:
            return 0, 0

        death_cause = str(death_cause).strip().lower()
        allowed_death_causes = {"combat", "metabolism", "environmental", "collision", "unknown"}
        if death_cause not in allowed_death_causes:
            raise RuntimeError(f"[tick] unsupported death_cause={death_cause!r}")

        # Snapshot metadata BEFORE mutation (positions/teams/units/uids are still readable now).
        dead_slots = dead_idx.detach().cpu().to(torch.int64).tolist()
        dead_rows = data.index_select(0, dead_idx)
        dead_team_list = dead_rows[:, COL_TEAM].detach().cpu().to(torch.int64).tolist()
        dead_unit_list = dead_rows[:, COL_UNIT].detach().cpu().to(torch.int64).tolist()
        dead_ids = self._slot_ids_to_agent_uids_list(dead_idx)

        # Structured killer alignment for telemetry/death forensics. Batch-extract
        # killer metadata once so the remaining bookkeeping loop consumes plain
        # Python values rather than triggering repeated per-element tensor reads.
        killer_slots_list: List[Optional[int]] = []
        killer_ids_list: List[Optional[int]] = []
        killer_team_list: List[Optional[int]] = []

        if killer_slot_by_victim is not None and len(killer_slot_by_victim) > 0:
            raw_killer_slots: List[Optional[int]] = []
            valid_killer_slots: List[int] = []
            capacity = int(data.shape[0])

            for dead_slot in dead_slots:
                ks = killer_slot_by_victim.get(int(dead_slot), None)
                ks = (int(ks) if ks is not None else None)
                if ks is None or ks < 0:
                    raw_killer_slots.append(None)
                    continue
                if ks >= capacity:
                    raise RuntimeError(f"[tick] killer slot out of range: victim_slot={dead_slot} killer_slot={ks}")
                raw_killer_slots.append(ks)
                valid_killer_slots.append(ks)

            killer_team_by_slot: Dict[int, int] = {}
            killer_id_by_slot: Dict[int, int] = {}
            if valid_killer_slots:
                uniq_killer_slots = sorted(set(valid_killer_slots))
                uniq_killer_tensor = torch.tensor(uniq_killer_slots, device=self.device, dtype=torch.long)
                killer_rows = data.index_select(0, uniq_killer_tensor)
                killer_teams_cpu = killer_rows[:, COL_TEAM].detach().cpu().to(torch.int64).tolist()
                if hasattr(self.registry, "agent_uids"):
                    killer_ids_cpu = self.registry.agent_uids.index_select(0, uniq_killer_tensor).detach().cpu().to(torch.int64).tolist()
                else:
                    killer_ids_cpu = killer_rows[:, COL_AGENT_ID].detach().cpu().to(torch.int64).tolist()
                killer_team_by_slot = {slot: team for slot, team in zip(uniq_killer_slots, killer_teams_cpu)}
                killer_id_by_slot = {slot: kid for slot, kid in zip(uniq_killer_slots, killer_ids_cpu)}

            for dead_slot, dead_team_val, ks in zip(dead_slots, dead_team_list, raw_killer_slots):
                if ks is None:
                    killer_slots_list.append(None)
                    killer_ids_list.append(None)
                    killer_team_list.append(None)
                    continue

                if ks == dead_slot:
                    raise RuntimeError(f"[tick] self-kill attribution detected for slot={ks}")

                killer_team_val = killer_team_by_slot.get(ks, None)
                if killer_team_val is None:
                    raise RuntimeError(f"[tick] missing killer metadata for killer_slot={ks}")
                if killer_team_val == dead_team_val:
                    raise RuntimeError(
                        f"[tick] friendly-fire killer attribution detected: victim_slot={dead_slot} killer_slot={ks}"
                    )

                killer_slots_list.append(ks)
                killer_ids_list.append(killer_id_by_slot.get(ks, None))
                killer_team_list.append(killer_team_val)
        else:
            killer_slots_list = [None] * len(dead_slots)
            killer_ids_list = [None] * len(dead_slots)
            killer_team_list = [None] * len(dead_slots)

        # Count deaths by team encoding (read before ALIVE zeroing).
        red_deaths = dead_team_list.count(2)
        blue_deaths = dead_team_list.count(3)

        # Update global stats (semantic counters).
        if red_deaths:
            self.stats.add_death("red", red_deaths)
            if credit_kills:
                self.stats.add_kill("blue", red_deaths)

        if blue_deaths:
            self.stats.add_death("blue", blue_deaths)
            if credit_kills:
                self.stats.add_kill("red", blue_deaths)

        # STATE MUTATION (registry + grid)
        # We apply the world-state mutation before emitting death telemetry so that
        # post-death readers do not observe "death event emitted but agent still alive on grid".
        gx = self._as_long(data[dead_idx, COL_X])
        gy = self._as_long(data[dead_idx, COL_Y])
        gx_list = gx.detach().cpu().to(torch.int64).tolist()
        gy_list = gy.detach().cpu().to(torch.int64).tolist()

        # Root structured death log for ResultsWriter -> dead_agents_log.csv.
        # This legacy CSV keeps a compact stable schema; richer cause/killer
        # metadata is emitted through telemetry when enabled.
        try:
            rec_fn = getattr(self.stats, "record_death_entry", None)
            if callable(rec_fn):
                n = min(len(dead_ids), len(dead_team_list), int(gx.numel()), int(gy.numel()))
                for i in range(n):
                    victim_team = int(dead_team_list[i])
                    if victim_team not in (2, 3):
                        continue

                    killer_team = killer_team_list[i] if i < len(killer_team_list) else None
                    killer_id = killer_ids_list[i] if i < len(killer_ids_list) else None
                    killer_slot = killer_slots_list[i] if i < len(killer_slots_list) else None

                    rec_fn(
                        agent_id=int(dead_ids[i]),
                        team_id_val=float(victim_team),
                        x=int(gx_list[i]),
                        y=int(gy_list[i]),
                        killer_team_id_val=(float(killer_team) if killer_team in (2, 3) else None),
                        killer_agent_id=(int(killer_id) if killer_id is not None else None),
                        killer_slot=(int(killer_slot) if killer_slot is not None else None),
                        death_cause=str(death_cause),
                        notes=(f"credit_kills={1 if credit_kills else 0}"),
                    )
        except Exception:
            # Fail-safe: root death CSV must never break the simulation tick.
            pass

        # Clear occupancy / hp / slot-id in grid first (spatial truth for fast queries).
        self.grid[0][gy, gx], self.grid[1][gy, gx], self.grid[2][gy, gx] = self._g0, self._g0, self._gneg

        # Then mark registry ALIVE=0 (agent remains in slot storage, but dead).
        data[dead_idx, COL_ALIVE] = self._d0

        # TELEMETRY (after state mutation)
        telemetry = getattr(self, "telemetry", None)
        if telemetry is not None and getattr(telemetry, "enabled", False):
            try:
                tick_now = int(self.stats.tick)
                self._telemetry_set_phase(
                    telemetry,
                    tick=tick_now,
                    phase=("death_combat" if death_cause == "combat" else f"death_{death_cause}"),
                )
                telemetry.record_deaths(
                    tick=tick_now,
                    dead_ids=dead_ids,
                    dead_team=dead_team_list,
                    dead_unit=dead_unit_list,
                    dead_slots=dead_slots,
                    notes=f"cause={death_cause}; credit_kills={1 if credit_kills else 0}",
                    death_causes=[str(death_cause)] * len(dead_ids),
                    killer_ids=killer_ids_list,
                    killer_slots=killer_slots_list,
                    killer_teams=killer_team_list,
                )
            except Exception as e:
                try:
                    telemetry._anomaly(f"_apply_deaths telemetry hook failed: {e}")
                except Exception:
                    pass

        metrics.deaths += int(dead_idx.numel())
        if death_cause == "combat":
            metrics.deaths_combat += int(dead_idx.numel())
        elif death_cause == "metabolism":
            metrics.deaths_metabolism += int(dead_idx.numel())
        elif death_cause == "environmental":
            metrics.deaths_environmental += int(dead_idx.numel())
        elif death_cause == "collision":
            metrics.deaths_collision += int(dead_idx.numel())
        else:
            metrics.deaths_unknown += int(dead_idx.numel())
        return red_deaths, blue_deaths

    # Observation builder for transformer/policy
    @torch.no_grad()
    def _build_transformer_obs(self, alive_idx: torch.Tensor, pos_xy: torch.Tensor) -> torch.Tensor:
        """
        Build observation tensor for all alive agents.

        Observation = raycast features (32 directions x 8 values) + rich features.

        Key principle:
        - The policy neural network consumes numeric vectors (tensors).
        - The observation is a fixed-length vector per agent:
              obs.shape == (N_alive, OBS_DIM)

        Raycast part:
        - Simulates line-of-sight “sensing” in 32 directions.
        - Each ray returns 8 values (project convention), so 32*8 = 256.

        Rich part:
        - HP ratio, normalized position, team/unit flags, global stats,
          zone flags, instinct context, etc.
        """
        from engine.ray_engine.raycast_firsthit import build_unit_map  # local import to avoid circular deps

        data = self.registry.agent_data
        N = alive_idx.numel()

        # --- Zone flags for rich features ---
        if self._z_heal is not None:
            on_heal = self._z_heal[pos_xy[:, 1], pos_xy[:, 0]]
        else:
            on_heal = self._obs_on_heal[:N]
            on_heal.zero_()

        on_cp = self._obs_on_cp[:N]
        on_cp.zero_()
        if self._z_cp_masks:
            for cp_mask in self._z_cp_masks:
                # |= accumulates: if any CP mask is True at that cell, on_cp becomes True.
                on_cp |= cp_mask[pos_xy[:, 1], pos_xy[:, 0]]

        expected_ray_dim = 32 * 8
        alive_data = data[alive_idx]

        # unit_map is a grid-shaped tensor encoding which unit type occupies each cell.
        unit_map = build_unit_map(data, self.grid)

        # Raycasting: 32 rays, “first hit” info, bounded by vision range per agent.
        rays = raycast32_firsthit(
            pos_xy, self.grid, unit_map,
            max_steps_each=alive_data[:, COL_VISION].long()
        )
        if rays.shape != (N, expected_ray_dim):
            raise RuntimeError(
                f"[obs] ray tensor shape mismatch: got {tuple(rays.shape)}, "
                f"expected ({N}, {expected_ray_dim})."
            )

        # hp_max clamped to avoid division-by-zero.
        hp_max = alive_data[:, COL_HP_MAX].clamp_min(1.0)

        # Rich features:
        # - normalized HP (hp/hp_max)
        # - normalized positions (x/(W-1), y/(H-1))
        # - one-hot-ish flags for team and unit types
        # - normalized attack and vision
        # - zone flags
        # - normalized global stats
        # Build the exact same 23-column layout, but write directly into a single
        # preallocated tensor to avoid many temporary (N,) vectors each tick.
        rich_base = self._obs_rich_base[:N]
        rich_base[:, 0] = alive_data[:, COL_HP] / hp_max
        rich_base[:, 1] = alive_data[:, COL_X] / (self.W - 1)
        rich_base[:, 2] = alive_data[:, COL_Y] / (self.H - 1)
        rich_base[:, 3] = (alive_data[:, COL_TEAM] == 2.0).to(self._data_dt)
        rich_base[:, 4] = (alive_data[:, COL_TEAM] == 3.0).to(self._data_dt)
        rich_base[:, 5] = (alive_data[:, COL_UNIT] == 1.0).to(self._data_dt)
        rich_base[:, 6] = (alive_data[:, COL_UNIT] == 2.0).to(self._data_dt)
        rich_base[:, 7] = alive_data[:, COL_ATK] / (config.MAX_ATK or 1.0)
        rich_base[:, 8] = alive_data[:, COL_VISION] / (config.RAYCAST_MAX_STEPS or 15.0)
        rich_base[:, 9] = on_heal.to(self._data_dt)
        rich_base[:, 10] = on_cp.to(self._data_dt)
        rich_base[:, 11].fill_(float(self.stats.tick) / 50000.0)
        rich_base[:, 12].fill_(float(self.stats.red.score) / 1000.0)
        rich_base[:, 13].fill_(float(self.stats.blue.score) / 1000.0)
        rich_base[:, 14].fill_(float(self.stats.red.cp_points) / 500.0)
        rich_base[:, 15].fill_(float(self.stats.blue.cp_points) / 500.0)
        rich_base[:, 16].fill_(float(self.stats.red.kills) / 500.0)
        rich_base[:, 17].fill_(float(self.stats.blue.kills) / 500.0)
        rich_base[:, 18].fill_(float(self.stats.red.deaths) / 500.0)
        rich_base[:, 19].fill_(float(self.stats.blue.deaths) / 500.0)
        # Padding slots keep the layout exactly matching the required dimension sizes.
        rich_base[:, 20:].zero_()

        instinct = self._compute_instinct_context(alive_idx=alive_idx, pos_xy=pos_xy, unit_map=unit_map)
        if instinct.shape != (N, 4):
            raise RuntimeError(f"instinct shape {tuple(instinct.shape)} != (N,4)")

        # Concatenate base rich features and instinct features.
        rich = self._obs_rich[:N]
        rich[:, :23].copy_(rich_base)
        rich[:, 23:].copy_(instinct)

        expected_rich_dim = int(self._OBS_DIM) - expected_ray_dim
        if rich.shape != (N, expected_rich_dim):
            raise RuntimeError(
                f"[obs] rich tensor shape mismatch: got {tuple(rich.shape)}, "
                f"expected ({N}, {expected_rich_dim})."
            )

        # Final observation: rays + rich
        obs = torch.cat([rays, rich.to(rays.dtype)], dim=1)
        if obs.shape != (N, int(self._OBS_DIM)):
            raise RuntimeError(
                f"[obs] final obs shape mismatch: got {tuple(obs.shape)}, "
                f"expected ({N}, {int(self._OBS_DIM)})."
            )
        return obs

    # Debug invariants to catch grid<->registry desync
    def _debug_invariants(self, where: str) -> None:
        """
        Optional, gated invariants to catch grid<->registry desync early.
        Only runs if environment variable FWS_DEBUG_INVARIANTS is set to "1" or "true".

        Invariants checked:
        - Alive positions in bounds
        - grid[2] at alive positions equals alive slot ids
        - grid[0] occupancy matches team encoding at alive positions
        - grid[2] contains no duplicate slot ids
        - Alive slot set matches grid[2] present set
        - No “ghost cells” (grid[2]>=0 but occupancy==0)
        """
        if os.getenv("FWS_DEBUG_INVARIANTS", "0") not in {"1", "true", "True"}:
            return

        data = self.registry.agent_data
        H, W = self.grid.shape[-2], self.grid.shape[-1]

        alive = (data[:, COL_ALIVE] > 0.5)
        alive_idx = alive.nonzero(as_tuple=False).squeeze(1)

        if alive_idx.numel() > 0:
            xs = self._as_long(data[alive_idx, COL_X])
            ys = self._as_long(data[alive_idx, COL_Y])
            if not ((xs >= 0).all() and (xs < W).all() and (ys >= 0).all() and (ys < H).all()):
                raise RuntimeError(f"[invariants:{where}] alive position out of bounds")

            g2_at = self._as_long(self.grid[2, ys, xs])
            if not torch.equal(g2_at, alive_idx):
                raise RuntimeError(f"[invariants:{where}] grid[2] slot-id mismatch at alive positions")

            team = data[alive_idx, COL_TEAM].to(self._grid_dt)
            g0_at = self.grid[0, ys, xs]
            if not torch.equal(g0_at, team):
                raise RuntimeError(f"[invariants:{where}] grid[0] occupancy/team mismatch at alive positions")

        ids = self._as_long(self.grid[2]).view(-1)
        present = ids[ids >= 0]
        uniq, counts = present.unique(return_counts=True) if present.numel() > 0 else (present, present)

        if counts.numel() > 0 and not (counts == 1).all():
            bad = uniq[counts != 1][:16].tolist()
            raise RuntimeError(f"[invariants:{where}] duplicate slot ids in grid[2]: {bad}")

        if alive_idx.numel() != uniq.numel():
            raise RuntimeError(f"[invariants:{where}] grid[2] ids != alive slots (alive={alive_idx.numel()} grid={uniq.numel()})")

        if alive_idx.numel() > 0:
            if not torch.equal(alive_idx.sort().values, uniq.sort().values):
                raise RuntimeError(f"[invariants:{where}] grid[2] set != alive slot set")

        ghost = (self.grid[2] >= 0) & (self.grid[0] == 0)
        if ghost.any():
            raise RuntimeError(f"[invariants:{where}] ghost cells: grid[2]>=0 but grid[0]==0")

    # run_tick: main step function
    @torch.no_grad()
    def run_tick(self) -> Dict[str, float]:
        """
        Execute one simulation tick:
          - Process attacks (damage, kills).
          - Apply deaths.
          - Move agents (with conflict resolution).
          - Apply zone healing and capture point scoring.
          - Record telemetry and PPO data.
          - Respawn dead agents.

        Returns:
            A dictionary of metrics for this tick, typically from vars(TickMetrics).
        """
        data = self.registry.agent_data
        telemetry = getattr(self, "telemetry", None)
        tick_now = int(self.stats.tick)

        # Initialize per-tick telemetry ordering context (additive only; safe no-op if unsupported).
        self._telemetry_set_phase(telemetry, tick=tick_now, phase="tick_start")

        metrics = TickMetrics()
        alive_idx = self._recompute_alive_idx()

        # Manual/viewer zone edits can happen between ticks, so detect any
        # out-of-band heal-mask mutation before the scheduler/runtime reads it.
        self._refresh_zone_runtime_cache_if_needed()

        # Catastrophe runtime update happens at the start of the tick so any
        # newly applied or expired suppression affects the same tick's
        # environment phase cleanly.
        catastrophe_signal = self._build_catastrophe_runtime_signal(alive_idx)
        if self.catastrophe_controller.update(
            tick_now=tick_now,
            runtime_signal=catastrophe_signal,
        ):
            self.refresh_zone_runtime_cache()

        # If absolutely everyone is dead, fast-forward time and respawn.
        if alive_idx.numel() == 0:
            # If the previous tick ended a PPO window, there are no surviving slots left
            # to bootstrap here, so the pending boundary can be finalized immediately.
            if self._ppo is not None:
                self._ppo.finalize_pending_window_from_cache()

            self.stats.on_tick_advanced(1)
            metrics.tick = int(self.stats.tick)

            # PPO bookkeeping: flush dead agents if PPO enabled.
            was_dead = (data[:, COL_ALIVE] <= 0.5) if self._ppo is not None else None
            if was_dead is not None:
                dead_slots = was_dead.nonzero(as_tuple=False).squeeze(1)
                if dead_slots.numel() > 0:
                    self._ppo.flush_agents(dead_slots)

            # Respawn.
            self.respawner.step(self.stats.tick, self.registry, self.grid, engine=self)

            # Reset PPO state for respawned slots.
            if was_dead is not None:
                self._ppo_reset_on_respawn(was_dead)

            self._debug_invariants("post_respawn")
            return vars(metrics)

        # 1) GET OBSERVATIONS
        pos_xy = self.registry.positions_xy(alive_idx)
        obs = self._build_transformer_obs(alive_idx, pos_xy)
        if obs.dim() != 2 or int(obs.shape[1]) != int(config.OBS_DIM):
            raise RuntimeError(
                f"[obs] shape mismatch: got {tuple(obs.shape)}, expected (N,{int(config.OBS_DIM)})"
            )

        # 2) GET ACTION MASK (prevents illegal actions)
        # Mask shape convention: (N_alive, NUM_ACTIONS) boolean
        # mask[i, a] == True means action a is legal for agent i.
        mask = build_mask(
            pos_xy,
            data[alive_idx, COL_TEAM],
            self.grid,
            unit=self._as_long(data[alive_idx, COL_UNIT])
        )

        # Actions will be filled in alive_idx order.
        actions = torch.zeros_like(alive_idx, dtype=torch.long)

        # PPO record buffers (only used if PPO enabled)
        rec_agent_ids, rec_obs, rec_logits, rec_values, rec_actions, rec_action_masks, rec_teams = [], [], [], [], [], [], []

        # 3) AI DECISION TIME (bucketed inference)
        # Agents can have different models/brains. We group them into buckets to run
        # ensemble_forward efficiently per model type.
        neg_inf = torch.finfo(torch.float32).min
        for bucket in self.registry.build_buckets(alive_idx):
            # loc are positions of bucket.indices within alive_idx (sorted alignment assumed)
            loc = bucket.locs
            bucket_obs = obs[loc]
            bucket_mask = mask[loc]

            # Forward pass: returns distribution (dist) and value estimate (vals)
            dist, vals = ensemble_forward(bucket.models, bucket_obs)

            # Mask logits: illegal actions get very negative logits (approx -inf)
            logits32 = torch.where(bucket_mask, dist.logits.to(torch.float32), neg_inf)

            # Sample action from categorical distribution
            a = torch.distributions.Categorical(logits=logits32).sample()

            if self._ppo:
                # Store trajectory elements needed for PPO training
                rec_agent_ids.append(bucket.indices)
                rec_obs.append(bucket_obs)
                rec_logits.append(logits32)
                rec_values.append(vals)
                rec_actions.append(a)
                rec_action_masks.append(bucket_mask)
                rec_teams.append(data[bucket.indices, COL_TEAM])

            actions[loc] = a

        # Debug: force actions / validate actions
        force = os.getenv("FWS_DEBUG_FORCE_ACTIONS", "").strip()
        if force:
            parts = [p.strip() for p in force.split(",") if p.strip()]
            for i, p in enumerate(parts):
                if i >= actions.numel():
                    break
                a = int(p)
                if a < 0 or a >= config.NUM_ACTIONS:
                    raise ValueError(
                        f"[tick] forced action out of range: local_idx={i} action={a} (NUM_ACTIONS={config.NUM_ACTIONS})"
                    )
                actions[i] = a

        if force or os.getenv("FWS_DEBUG_VALIDATE_ACTIONS", "0") in ("1", "true", "True"):
            if actions.numel() > 0:
                bad_range = ((actions < 0) | (actions >= config.NUM_ACTIONS)).nonzero(as_tuple=False).squeeze(1)
                if bad_range.numel() != 0:
                    bad_range_list = bad_range[:8].detach().cpu().to(torch.int64).tolist()
                    raise RuntimeError(
                        f"[tick] sampled action out of range (first_bad_local_idx={bad_range_list}; mask/sampling bug)"
                    )
                ar = torch.arange(actions.numel(), device=actions.device)
                bad_masked = (~mask[ar, actions]).nonzero(as_tuple=False).squeeze(1)
                if bad_masked.numel() != 0:
                    bad = bad_masked[:8].detach().cpu().to(torch.int64).tolist()
                    raise RuntimeError(
                        f"[tick] chosen action is masked out (first_bad_local_idx={bad}, use FWS_DEBUG_FORCE_ACTIONS to reproduce)"
                    )

        # Update the persistent PPO value cache from the normal main inference pass.
        # This removes the need for a second post-step observation/inference branch
        # just to supply bootstrap values at PPO window boundaries.
        if self._ppo and rec_agent_ids:
            self._ppo.update_value_cache(
                agent_ids=torch.cat(rec_agent_ids),
                values=torch.cat(rec_values),
            )

            # If the previous tick ended a PPO window without an explicit bootstrap,
            # finalize it now using the cached V(s_t) from this tick's normal forward.
            self._ppo.finalize_pending_window_from_cache()

        metrics.alive = int(alive_idx.numel())

        combat_rd, combat_bd = 0, 0
        meta_rd, meta_bd = 0, 0

        # Individual rewards tensor over ALL slots (capacity-sized).
        # This is later indexed for just the participating agents in PPO logging.
        (
            individual_rewards,
            reward_kill_individual,
            reward_damage_dealt_individual,
            reward_damage_taken_penalty,
            reward_contested_cp_individual,
            reward_healing_recovered,
        ) = self._reset_tick_reward_buffers()

        # Victim slot -> credited killer slot for combat deaths in this tick.
        combat_killer_slot_by_victim: Dict[int, int] = {}

        # 4) COMBAT (combat-first semantics)
        if alive_idx.numel() > 0:
            # Attack actions are encoded as action >= 9 in this project.
            if (is_attack := actions >= 9).any():
                atk_idx, atk_act = alive_idx[is_attack], actions[is_attack]

                # Action decoding:
                # - atk_act in [9..] encodes a direction and a range (1..4)
                # r = ((atk_act - 9) % 4) + 1  => range ∈ {1,2,3,4}
                # dir_idx = (atk_act - 9) // 4 => direction index into DIRS8
                # Math note:
                # A single integer encodes 2 parameters via quotient and remainder.
                # This is a standard encoding pattern:
                #   x = q * base + rem
                #   q = x // base
                #   rem = x % base
                r, dir_idx = ((atk_act - 9) % 4) + 1, (atk_act - 9) // 4

                # Direction vector scaled by range r:
                # dxy shape: (num_attackers, 2)
                dxy = self.DIRS8_dev[dir_idx] * r.unsqueeze(1)

                # Attacker positions:
                ax, ay = pos_xy[is_attack].T

                # Target positions:
                tx = (ax + dxy[:, 0]).clamp(0, self.W - 1)
                ty = (ay + dxy[:, 1]).clamp(0, self.H - 1)

                # Optional LOS wall blocking for archers:
                # Even if an illegal action is forced, prevent damage through walls.
                if bool(getattr(config, "ARCHER_LOS_BLOCKS_WALLS", False)):
                    is_archer = (data[atk_idx, COL_UNIT] == 2.0)
                    if is_archer.any():
                        # Check intermediate cells at steps 1..(r-1) along DIRS8.
                        steps = torch.arange(1, 4, device=self.device, dtype=torch.long).view(1, 3)  # max RMAX-1
                        dx = self.DIRS8_dev[dir_idx, 0].view(-1, 1)
                        dy = self.DIRS8_dev[dir_idx, 1].view(-1, 1)
                        ix = ax.view(-1, 1) + dx * steps
                        iy = ay.view(-1, 1) + dy * steps
                        active = steps < r.view(-1, 1)  # exclude target cell at step r

                        inb = (ix >= 0) & (ix < self.W) & (iy >= 0) & (iy < self.H)
                        ix_c = ix.clamp(0, self.W - 1)
                        iy_c = iy.clamp(0, self.H - 1)
                        is_wall = (self.grid[0][iy_c, ix_c] == 1.0) & inb

                        # Treat out-of-bounds intermediate steps as blocked
                        los_blocked = (((~inb) | is_wall) & active).any(dim=1) & is_archer

                        if os.getenv("FWS_DEBUG_COMBAT", "0") in ("1", "true", "True"):
                            if bool(los_blocked.any().item()):
                                raise RuntimeError("[combat] archer LOS blocked by wall (mask mismatch)")

                        if bool(los_blocked.any().item()):
                            keep = ~los_blocked
                            atk_idx = atk_idx[keep]
                            atk_act = atk_act[keep]
                            r = r[keep]
                            dir_idx = dir_idx[keep]
                            ax = ax[keep]
                            ay = ay[keep]
                            tx = tx[keep]
                            ty = ty[keep]
                            if atk_idx.numel() == 0:
                                # All attacks LOS-blocked
                                pass

                # Debug checks (optional)
                if os.getenv("FWS_DEBUG_COMBAT", "0") in ("1", "true", "True"):
                    victims_dbg = self._as_long(self.grid[2][ty, tx])
                    if victims_dbg.numel() != atk_idx.numel():
                        raise RuntimeError("victims_dbg shape mismatch")

                if os.getenv("FWS_DEBUG_COMBAT", "0") in ("1", "true", "True"):
                    victims_dbg = self._as_long(self.grid[2][ty, tx])
                    if (victims_dbg < 0).any():
                        raise RuntimeError("[combat] attack targeted empty cell (mask mismatch)")
                    if (data[atk_idx, COL_TEAM] == data[victims_dbg, COL_TEAM]).any():
                        raise RuntimeError("[combat] attack targeted friendly cell (mask mismatch)")
                    is_soldier = (data[atk_idx, COL_UNIT] == 1.0)
                    if (is_soldier & (r > 1)).any():
                        raise RuntimeError("[combat] soldier used ranged>1 (mask mismatch)")

                # victims are slot ids in grid[2] at target cells.
                victims = self._as_long(self.grid[2][ty, tx])

                # Only hits where there is an agent (slot id >= 0)
                if (valid_hit := victims >= 0).any():
                    atk_idx, victims = atk_idx[valid_hit], victims[valid_hit]

                    # Only enemy hits (team differs)
                    is_enemy = (data[atk_idx, COL_TEAM] != data[victims, COL_TEAM])
                    victims = victims[is_enemy]
                    atk_idx = atk_idx[is_enemy]

                    if victims.numel() > 0:
                        # DETERMINISTIC FOCUS-FIRE DAMAGE
                        # Duplicate index hazard:
                        #   If multiple attackers hit the same victim, writing
                        #   victim_hp[victim] -= dmg in parallel can race.
                        # Safe approach:
                        #   1) Sort by victim id
                        #   2) Compute total damage per unique victim
                        #   3) Apply hp -= total_damage once per victim
                        dmg = data[atk_idx, COL_ATK]
                        order = victims.argsort()
                        sv = victims[order]
                        sdmg = dmg[order]
                        satk = atk_idx[order]

                        # Counts consecutive duplicates after sorting.
                        uniq_v, counts = torch.unique_consecutive(sv, return_counts=True)

                        # Sum damage per victim using prefix sums.
                        cums = sdmg.cumsum(0)
                        ends = counts.cumsum(0) - 1
                        starts = ends - counts + 1
                        prev = torch.where(
                            starts > 0,
                            cums[starts - 1],
                            torch.zeros_like(starts, dtype=cums.dtype)
                        )
                        dmg_sum = cums[ends] - prev

                        # Apply damage
                        hp_before = data[uniq_v, COL_HP].clone()
                        data[uniq_v, COL_HP] = hp_before - dmg_sum
                        hp_after = data[uniq_v, COL_HP]

                        # We delay kill-event emission until AFTER damage telemetry so replay order is:
                        # damage -> kill_credit -> death (later in _apply_deaths).
                        kill_event_killer_slots = None
                        kill_event_victim_slots = None

                        # Preserve single-attacker kill credit to match the established reward contract.
                        # Why:
                        # - Prevents multi-attacker focus-fire from granting multiple kill credits
                        #   for the same victim in one tick.
                        # How (deterministic):
                        #   1) choose highest same-tick damage contributor for that victim
                        #   2) if tied on damage, choose smallest attacker slot id
                        killed_v = (hp_before > 0) & (hp_after <= 0)
                        if killed_v.any():
                            reward_val = float(config.PPO_REWARD_KILL_INDIVIDUAL)

                            killed_group_idx = killed_v.nonzero(as_tuple=False).squeeze(1)
                            if killed_group_idx.numel() > 0:
                                credited_victims = uniq_v[killed_group_idx]
                                credited_killers = torch.empty(
                                    (killed_group_idx.numel(),),
                                    device=satk.device,
                                    dtype=satk.dtype,
                                )

                                for out_i, g in enumerate(killed_group_idx.tolist()):
                                    s = int(starts[g].item())
                                    e = int(ends[g].item()) + 1  # slice end is exclusive
                                    grp_atk = satk[s:e]
                                    grp_dmg = sdmg[s:e]

                                    max_dmg = grp_dmg.max()
                                    tied_killers = grp_atk[grp_dmg == max_dmg]
                                    credited_killers[out_i] = tied_killers.min()

                                # Deterministic accumulation per killer slot:
                                k_order = credited_killers.argsort()
                                sk = credited_killers[k_order]
                                uniq_k, k_counts = torch.unique_consecutive(sk, return_counts=True)

                                reward_add = (k_counts.to(torch.float32) * reward_val)
                                individual_rewards[uniq_k] += reward_add.to(self._data_dt)
                                reward_kill_individual[uniq_k] += reward_add

                                # Accumulate doctrine/personal merit keyed by persistent agent id.
                                for killer_slot, cnt in zip(uniq_k.tolist(), k_counts.tolist()):
                                    if hasattr(self.registry, "agent_uids"):
                                        uid = int(self.registry.agent_uids[killer_slot].item())
                                    else:
                                        uid = int(data[killer_slot, COL_AGENT_ID].item())
                                    self.agent_kill_counts[uid] += float(cnt)
                                    if self._ppo is None:
                                        self.agent_scores[uid] += reward_val * float(cnt)

                                for victim_slot, killer_slot in zip(credited_victims.tolist(), credited_killers.tolist()):
                                    combat_killer_slot_by_victim[int(victim_slot)] = int(killer_slot)

                                # Defer kill-event emission until AFTER damage telemetry (causal ordering).
                                kill_event_killer_slots = credited_killers
                                kill_event_victim_slots = credited_victims

                        # Individual PPO dense damage shaping (per-agent only; no team reward path here).
                        ppo_dmg_dealt_coef = float(getattr(config, "PPO_REWARD_DMG_DEALT_INDIVIDUAL", 0.0))
                        ppo_dmg_taken_pen = float(getattr(config, "PPO_PENALTY_DMG_TAKEN_INDIVIDUAL", 0.0))

                        # Aggregate per-attacker damage once so it can feed both PPO shaping and telemetry.
                        uniq_a = torch.empty((0,), device=satk.device, dtype=satk.dtype)
                        dmg_a = torch.empty((0,), device=sdmg.device, dtype=sdmg.dtype)
                        need_attacker_sum = (satk.numel() > 0) and (
                            (ppo_dmg_dealt_coef != 0.0)
                            or (telemetry is not None and getattr(telemetry, "enabled", False))
                        )
                        if need_attacker_sum:
                            uniq_a, inv_a = satk.unique(return_inverse=True)
                            dmg_a = torch.zeros((uniq_a.numel(),), device=sdmg.device, dtype=sdmg.dtype)
                            dmg_a.scatter_add_(0, inv_a, sdmg)

                        if ppo_dmg_dealt_coef != 0.0 and uniq_a.numel() > 0:
                            dmg_dealt_add = (dmg_a.to(torch.float32) * float(ppo_dmg_dealt_coef))
                            individual_rewards[uniq_a] += dmg_dealt_add.to(self._data_dt)
                            reward_damage_dealt_individual[uniq_a] += dmg_dealt_add

                        if ppo_dmg_taken_pen != 0.0 and uniq_v.numel() > 0:
                            dmg_taken_pen = (dmg_sum.to(torch.float32) * float(ppo_dmg_taken_pen))
                            individual_rewards[uniq_v] -= dmg_taken_pen.to(self._data_dt)
                            reward_damage_taken_penalty[uniq_v] -= dmg_taken_pen

                        # Sync grid HP immediately after registry HP mutation (shrink desync window).
                        # This reduces the chance that debug hooks / future readers observe
                        # registry HP(t) while grid HP is still stale.
                        self._sync_grid_hp_for_slots(uniq_v)

                        self._telemetry_set_phase(telemetry, tick=tick_now, phase="combat_damage")

                        # TELEMETRY: damage totals + damage events
                        if telemetry is not None and getattr(telemetry, "enabled", False):
                            try:
                                # Victim-sum (unique victims)
                                if hasattr(self.registry, "agent_uids"):
                                    v_ids = self.registry.agent_uids.index_select(0, uniq_v).detach().cpu().tolist()
                                else:
                                    v_ids = [int(data[slot, COL_AGENT_ID].item()) for slot in uniq_v.tolist()]
                                v_team = [int(data[slot, COL_TEAM].item()) for slot in uniq_v.tolist()]
                                v_unit = [int(data[slot, COL_UNIT].item()) for slot in uniq_v.tolist()]
                                dmg_v = [float(x) for x in dmg_sum.detach().cpu().tolist()]
                                hp_b = [float(x) for x in hp_before.detach().cpu().tolist()]
                                hp_a = [float(x) for x in hp_after.detach().cpu().tolist()]
                                telemetry.record_damage_victim_sum(
                                    tick=tick_now,
                                    victim_ids=v_ids,
                                    victim_team=v_team,
                                    victim_unit=v_unit,
                                    damage=dmg_v,
                                    hp_before=hp_b,
                                    hp_after=hp_a,
                                )

                                # Attacker-sum (aggregate per attacker)
                                if hasattr(self.registry, "agent_uids"):
                                    a_ids = self.registry.agent_uids.index_select(0, uniq_a).detach().cpu().tolist()
                                else:
                                    a_ids = [int(data[slot, COL_AGENT_ID].item()) for slot in uniq_a.tolist()]
                                telemetry.record_damage_attacker_sum(
                                    tick=tick_now,
                                    attacker_ids=a_ids,
                                    damage_dealt=[float(x) for x in dmg_a.detach().cpu().tolist()],
                                )

                                # Optional per-hit logging
                                if str(getattr(telemetry, "damage_mode", "victim_sum")).lower() == "per_hit":
                                    if hasattr(self.registry, "agent_uids"):
                                        atk_ids = self.registry.agent_uids.index_select(0, satk).detach().cpu().tolist()
                                        vic_ids = self.registry.agent_uids.index_select(0, sv).detach().cpu().tolist()
                                    else:
                                        atk_ids = [int(data[slot, COL_AGENT_ID].item()) for slot in satk.tolist()]
                                        vic_ids = [int(data[slot, COL_AGENT_ID].item()) for slot in sv.tolist()]
                                    telemetry.record_damage_per_hit(
                                        tick=tick_now,
                                        attacker_ids=atk_ids,
                                        victim_ids=vic_ids,
                                        damage=[float(x) for x in sdmg.detach().cpu().tolist()],
                                    )
                            except Exception as e:
                                try:
                                    telemetry._anomaly(f"tick.damage hook failed: {e}")
                                except Exception:
                                    pass

                        # TELEMETRY: kill events (emitted AFTER damage telemetry for causal replay ordering)
                        if (
                            telemetry is not None
                            and getattr(telemetry, "enabled", False)
                            and kill_event_killer_slots is not None
                            and kill_event_victim_slots is not None
                            and kill_event_killer_slots.numel() > 0
                        ):
                            try:
                                self._telemetry_set_phase(telemetry, tick=tick_now, phase="combat_kill_credit")
                                killer_ids = self._slot_ids_to_agent_uids_list(kill_event_killer_slots)
                                victim_ids = self._slot_ids_to_agent_uids_list(kill_event_victim_slots)
                                telemetry.record_kills(
                                    tick=tick_now,
                                    killer_ids=killer_ids,
                                    victim_ids=victim_ids,
                                )
                            except Exception as e:
                                try:
                                    telemetry._anomaly(f"tick.kill hook failed: {e}")
                                except Exception:
                                    pass

                        metrics.attacks += int(atk_idx.numel())
                        # ===== END NEW DAMAGE =====

        # 5) Apply deaths after combat
        # IMPORTANT SEMANTIC:
        # - Combat damage is applied first.
        # - Combat deaths are resolved here (before movement).
        # Therefore, agents killed in combat do NOT move this tick.
        rD, bD = self._apply_deaths(
            (data[:, COL_ALIVE] > 0.5) & (data[:, COL_HP] <= 0.0),
            metrics,
            credit_kills=True,
            death_cause="combat",
            killer_slot_by_victim=(combat_killer_slot_by_victim if combat_killer_slot_by_victim else None),
        )
        combat_rd += rD
        combat_bd += bD
        self._debug_invariants("post_combat")

        # 6) MOVE HANDLING WITH CONFLICT RESOLUTION (Law 1)
        # Combat-first semantics: agents killed this tick do NOT move.
        alive_after = (data[alive_idx, COL_ALIVE] > 0.5)

        # Movement actions are 1..8 (directional).
        is_move = alive_after & (actions >= 1) & (actions <= 8)

        # Movement aggregates (kept cheap; used for telemetry + debug)
        metrics.move_attempted = 0
        metrics.move_can_move = 0
        metrics.move_blocked_wall = 0
        metrics.move_blocked_occupied = 0
        metrics.move_conflict_lost = 0
        metrics.move_conflict_tie = 0

        if is_move.any():
            # 6.1) Compute intended destinations for ALL attempted moves
            move_idx_all = alive_idx[is_move]
            a_all = actions[is_move]               # chosen action code (1..8)
            dir_idx = a_all - 1                    # 0..7

            x0_all, y0_all = pos_xy[is_move].T
            nx_all = (x0_all + self.DIRS8_dev[dir_idx, 0]).clamp(0, self.W - 1)
            ny_all = (y0_all + self.DIRS8_dev[dir_idx, 1]).clamp(0, self.H - 1)

            metrics.move_attempted = int(move_idx_all.numel())

            # Destination occupancy code (grid channel 0):
            #   0.0 empty | 1.0 wall | 2.0 red | 3.0 blue
            occ_all = self.grid[0][ny_all, nx_all]
            g_wall = self._g_wall

            # Destination must be empty to be eligible for movement
            can_move_all = (occ_all == self._g0)
            metrics.move_can_move = int(move_idx_all[can_move_all].numel())

            # Blocked outcomes (attempted but destination not empty)
            blocked_all = ~can_move_all
            if blocked_all.any():
                occ_blocked = occ_all[blocked_all]
                blocked_wall = (occ_blocked == g_wall)
                # NOTE: anything that's not wall and not empty is treated as "occupied"
                metrics.move_blocked_wall = int(blocked_wall.sum().item())
                metrics.move_blocked_occupied = int((~blocked_wall).sum().item())

            # These masks exist only for the can-move subset; set defaults for telemetry
            winner_mask = torch.empty((0,), device=self.device, dtype=torch.bool)
            tie_mask = torch.empty((0,), device=self.device, dtype=torch.bool)
            lost_mask = torch.empty((0,), device=self.device, dtype=torch.bool)
            can_locs = torch.empty((0,), device=self.device, dtype=torch.long)

            # 6.2) Resolve moves among those whose destination is empty
            if can_move_all.any():
                move_idx = move_idx_all[can_move_all]
                x0, y0, nx, ny = x0_all[can_move_all], y0_all[can_move_all], nx_all[can_move_all], ny_all[can_move_all]
                can_locs = can_move_all.nonzero(as_tuple=False).squeeze(1)

                # Conflict resolution:
                # If multiple agents target same destination:
                #   winner = highest HP
                #   tie = nobody moves
                dest_key = (ny * self.W + nx).to(torch.long)  # unique cell id
                hp = data[move_idx, COL_HP]                   # hp used to decide winners

                # Per-destination claimant count (works even if scatter_reduce_ is unavailable)
                num_cells = self.H * self.W
                claim_cnt = self._move_claim_cnt.zero_()
                claim_cnt.scatter_add_(0, dest_key, torch.ones_like(dest_key, dtype=torch.int32))

                try:
                    max_hp = self._move_max_hp.fill_(torch.finfo(hp.dtype).min)
                    max_hp.scatter_reduce_(0, dest_key, hp, reduce="amax", include_self=True)
                    is_max = (hp == max_hp[dest_key])

                    # Count how many claimants share the max HP (tie detection).
                    max_cnt = self._move_max_cnt.zero_()
                    max_cnt.scatter_add_(0, dest_key, is_max.to(torch.int32))

                    winner_mask = is_max & (max_cnt[dest_key] == 1)
                except Exception:
                    # Fallback path if scatter_reduce_ not available.
                    # Keep behavior consistent with the original: unique max HP wins, ties -> nobody.
                    winner_mask = torch.zeros_like(dest_key, dtype=torch.bool)

                    order = torch.argsort(dest_key)
                    dest_s = dest_key[order]
                    hp_s = hp[order]
                    if dest_s.numel() > 0:
                        starts = torch.cat([
                            torch.zeros(1, device=self.device, dtype=torch.long),
                            (dest_s[1:] != dest_s[:-1]).nonzero(as_tuple=False).squeeze(1) + 1
                        ])
                        ends = torch.cat([starts[1:], torch.tensor([dest_s.numel()], device=self.device, dtype=torch.long)])
                        for s, e in zip(starts.tolist(), ends.tolist()):
                            grp_hp = hp_s[s:e]
                            maxv = grp_hp.max()
                            if (grp_hp == maxv).sum().item() == 1:
                                win_off = (grp_hp == maxv).nonzero(as_tuple=False).squeeze(1).item() + s
                                winner_mask[order[win_off]] = True

                # Classify conflict outcomes (used for aggregates + optional per-agent move events)
                win_cnt = self._move_win_cnt.zero_()
                win_cnt.scatter_add_(0, dest_key, winner_mask.to(torch.int32))

                conflict_mask = (claim_cnt[dest_key] > 1)
                tie_mask = conflict_mask & (win_cnt[dest_key] == 0)                       # tie at destination => nobody moves
                lost_mask = conflict_mask & (win_cnt[dest_key] == 1) & (~winner_mask)     # unique winner exists, others lose

                if tie_mask.any():
                    metrics.move_conflict_tie = int(tie_mask.sum().item())
                if lost_mask.any():
                    metrics.move_conflict_lost = int(lost_mask.sum().item())

                if winner_mask.any():
                    w_move_idx = move_idx[winner_mask]
                    w_x0, w_y0, w_nx, w_ny = x0[winner_mask], y0[winner_mask], nx[winner_mask], ny[winner_mask]

                    # Clear old spots on grid
                    self.grid[0, w_y0, w_x0], self.grid[1, w_y0, w_x0], self.grid[2, w_y0, w_x0] = self._g0, self._g0, self._gneg

                    # Update registry positions
                    data[w_move_idx, COL_X], data[w_move_idx, COL_Y] = w_nx.to(self._data_dt), w_ny.to(self._data_dt)

                    # Write to new spots on grid
                    self.grid[0, w_ny, w_nx] = data[w_move_idx, COL_TEAM].to(self._grid_dt)
                    self.grid[1, w_ny, w_nx] = data[w_move_idx, COL_HP].to(self._grid_dt)
                    self.grid[2, w_ny, w_nx] = w_move_idx.to(self._grid_dt)

                    # Count moved winners
                    metrics.moved = int(w_move_idx.numel())

            # 6.3) Telemetry: movement aggregates + optional per-agent events
            if telemetry is not None and getattr(telemetry, "enabled", False) and bool(getattr(telemetry, "log_moves", False)):
                self._telemetry_set_phase(telemetry, tick=tick_now, phase="move")

                # Aggregates are always cheap and always recorded when log_moves is enabled.
                try:
                    telemetry.record_move_summary(
                        tick=tick_now,
                        attempted=int(metrics.move_attempted),
                        can_move=int(metrics.move_can_move),
                        moved=int(metrics.moved),
                        blocked_wall=int(metrics.move_blocked_wall),
                        blocked_occupied=int(metrics.move_blocked_occupied),
                        conflict_lost=int(metrics.move_conflict_lost),
                        conflict_tie=int(metrics.move_conflict_tie),
                    )
                except Exception:
                    pass

                # Per-agent movement totals (AgentLife snapshot)
                # NOTE: This runs *after* conflict resolution and after winner moves are applied,
                # so outcomes are finalized.
                # Why this exists:
                # - The simulation already records high-level move aggregates (move_summary.csv)
                # - It also optionally records sampled per-move events (events_*.jsonl)
                # - But AgentLife snapshots (agent_life.csv) typically want *per-agent totals*
                #   (e.g., moves attempted/success/blocked counts) so you can analyze behavior
                #   per agent over long runs without reading huge event logs.
                # This hook sends, in one vectorized call, the "attempted move outcome code"
                # for every slot that attempted to move this tick. Telemetry can then accumulate
                # these per slot across ticks.
                out_code_all = None
                try:
                    n_all = int(move_idx_all.numel())
                    if n_all > 0 and hasattr(telemetry, "record_move_totals_by_slot"):
                        # Outcome codes for attempted moves:
                        # 0=success | 1=blocked_wall | 2=blocked_occupied | 3=conflict_lost | 4=conflict_tie
                        # Default is "blocked_occupied" because:
                        # - The most general non-empty case is “occupied”
                        # - We then refine wall-blocks and can-move / conflict outcomes below.
                        out_code_all = torch.full((n_all,), 2, device=self.device, dtype=torch.int16)  # default: blocked_occupied

                        # Blocked outcomes (attempted but destination not empty)
                        # blocked_all tells us "destination not empty"
                        # occ_all == g_wall specializes that into "blocked by wall"
                        out_code_all[blocked_all & (occ_all == g_wall)] = 1
                        # If destination was empty, the move is "can move", which is *provisionally* success.
                        # We will override for ties/losses below if conflicts existed.
                        out_code_all[can_move_all] = 0  # provisional: can-move defaults to success

                        # Override can-move outcomes with conflict results (if any)
                        # IMPORTANT: tie_mask / lost_mask are indexed over the can-move subset.
                        # can_locs maps those subset positions back into the original n_all indexing.
                        if can_locs.numel() > 0:
                            if tie_mask.numel() > 0 and tie_mask.any():
                                out_code_all[can_locs[tie_mask]] = 4
                            if lost_mask.numel() > 0 and lost_mask.any():
                                out_code_all[can_locs[lost_mask]] = 3
                            if winner_mask.numel() > 0:
                                weird = (~winner_mask) & (~tie_mask)
                                if weird.any():
                                    out_code_all[can_locs[weird]] = 3

                        # This is the “per-slot totals” ingestion point:
                        # - slot_ids is the registry slot index (stable identity within capacity)
                        # - outcome_code is the per-attempt result for that slot in THIS tick
                        telemetry.record_move_totals_by_slot(
                            tick=tick_now,
                            slot_ids=move_idx_all,
                            outcome_code=out_code_all,
                        )
                except Exception:
                    pass

                # Per-agent move events are optional and sampling-based.
                try:
                    every = int(getattr(telemetry, "move_events_every", 0))
                    max_ev = int(getattr(telemetry, "move_events_max_per_tick", 0))
                    rate = float(getattr(telemetry, "move_events_sample_rate", 1.0))
                    if every > 0 and max_ev > 0 and (tick_now % every) == 0 and rate > 0.0:
                        # Outcome codes for attempted moves:
                        # 0=success | 1=blocked_wall | 2=blocked_occupied | 3=conflict_lost | 4=conflict_tie
                        n_all = int(move_idx_all.numel())
                        # Reuse the already-computed out_code_all when available to avoid duplicate work.
                        out_code = out_code_all
                        if out_code is None:
                            out_code = torch.full((n_all,), 2, device=self.device, dtype=torch.int16)  # default: blocked_occupied

                            # Blocked outcomes (attempted but destination not empty)
                            out_code[blocked_all & (occ_all == g_wall)] = 1
                            out_code[can_move_all] = 0  # provisional: can-move defaults to success

                            # Override can-move outcomes with conflict results (if any)
                            if can_locs.numel() > 0:
                                if tie_mask.numel() > 0 and tie_mask.any():
                                    out_code[can_locs[tie_mask]] = 4
                                if lost_mask.numel() > 0 and lost_mask.any():
                                    out_code[can_locs[lost_mask]] = 3
                                if winner_mask.numel() > 0:
                                    weird = (~winner_mask) & (~tie_mask)
                                    if weird.any():
                                        out_code[can_locs[weird]] = 3

                        # Deterministic sampling (no RNG; stable across runs):
                        # hash = (slot_id * 1103515245 + tick * 12345) mod 2^32
                        sel = torch.arange(n_all, device=self.device, dtype=torch.long)
                        if rate < 1.0:
                            slot64 = move_idx_all.to(torch.int64)
                            h = (slot64 * 1103515245 + int(tick_now) * 12345) & 0xFFFFFFFF
                            thr = int(rate * 4294967296.0)
                            sel = sel[h < thr]

                        if sel.numel() > 0:
                            sel = sel[:max_ev]  # cap volume (deterministic order)

                            sel_slots = move_idx_all.index_select(0, sel)
                            # Event IDs must be persistent agent UIDs (NOT registry slot ids).
                            event_agent_uids = self._slot_ids_to_agent_uids_list(sel_slots)

                            act_ids = a_all.index_select(0, sel).detach().cpu().to(torch.int64).tolist()
                            fx = x0_all.index_select(0, sel).detach().cpu().to(torch.int64).tolist()
                            fy = y0_all.index_select(0, sel).detach().cpu().to(torch.int64).tolist()
                            tx = nx_all.index_select(0, sel).detach().cpu().to(torch.int64).tolist()
                            ty = ny_all.index_select(0, sel).detach().cpu().to(torch.int64).tolist()
                            oc = out_code.index_select(0, sel).detach().cpu().to(torch.int64).tolist()

                            telemetry.record_move_events(
                                tick=tick_now,
                                agent_ids=event_agent_uids,
                                actions=act_ids,
                                from_x=fx,
                                from_y=fy,
                                to_x=tx,
                                to_y=ty,
                                outcome_code=oc,
                            )
                except Exception:
                    pass

        # ----- END MOVE HANDLING -----
        self._debug_invariants("post_move")

        # 7) ENVIRONMENT (Heal, Metabolism, Objectives)
        # IMPORTANT SEMANTIC:
        # - This phase runs AFTER movement.
        # - Metabolism/starvation deaths therefore happen AFTER movement in the same tick.
        # - These deaths are not enemy kills (credit_kills=False).
        # Recompute alive index because environment effects may kill agents.
        if (alive_idx := self._recompute_alive_idx()).numel() > 0:
            pos_xy = self.registry.positions_xy(alive_idx)

            # Healing zones
            if self._z_heal is not None and (on_heal := self._z_heal[pos_xy[:, 1], pos_xy[:, 0]]).any():
                heal_idx = alive_idx[on_heal]
                hp_before_heal = data[heal_idx, COL_HP].clone()
                data[heal_idx, COL_HP] = (data[heal_idx, COL_HP] + config.HEAL_RATE).clamp_max(data[heal_idx, COL_HP_MAX])
                healed_delta = (data[heal_idx, COL_HP] - hp_before_heal).clamp_min(0.0).to(torch.float32)
                self.grid[1, pos_xy[on_heal, 1], pos_xy[on_heal, 0]] = data[heal_idx, COL_HP].to(self._grid_dt)

                heal_reward_coef = float(getattr(config, "PPO_REWARD_HEALING_RECOVERY", 0.0))
                if heal_reward_coef != 0.0 and int(heal_idx.numel()) > 0:
                    heal_reward = healed_delta * heal_reward_coef
                    individual_rewards.index_add_(0, heal_idx, heal_reward.to(self._data_dt))
                    reward_healing_recovered.index_add_(0, heal_idx, heal_reward)

            # Metabolism / attrition
            if meta_drain := getattr(config, "METABOLISM_ENABLED", True):
                drain = torch.where(
                    data[alive_idx, COL_UNIT] == 1.0,
                    config.META_SOLDIER_HP_PER_TICK,
                    config.META_ARCHER_HP_PER_TICK
                )
                data[alive_idx, COL_HP] -= drain.to(self._data_dt)
                self.grid[1, pos_xy[:, 1], pos_xy[:, 0]] = data[alive_idx, COL_HP].to(self._grid_dt)

                # Starvation deaths
                if (data[alive_idx, COL_HP] <= 0.0).any():
                    rD, bD = self._apply_deaths(
                        alive_idx[data[alive_idx, COL_HP] <= 0.0],
                        metrics,
                        credit_kills=False,
                        death_cause="metabolism",
                    )
                    meta_rd += rD
                    meta_bd += bD

            # Capture points
            if self._z_cp_masks and (alive_idx := self._recompute_alive_idx()).numel() > 0:
                pos_xy, teams_alive = self.registry.positions_xy(alive_idx), data[alive_idx, COL_TEAM]
                for cp_mask in self._z_cp_masks:
                    if (on_cp := cp_mask[pos_xy[:, 1], pos_xy[:, 0]]).any():
                        red_on = (on_cp & (teams_alive == 2.0)).sum().item()
                        blue_on = (on_cp & (teams_alive == 3.0)).sum().item()

                        # King-of-the-hill scoring
                        winner_team_id = None
                        if red_on > blue_on:
                            winner_team_id = 2.0
                            self.stats.add_capture_points("red", config.CP_REWARD_PER_TICK)
                            metrics.cp_red_tick += config.CP_REWARD_PER_TICK
                        elif blue_on > red_on:
                            winner_team_id = 3.0
                            self.stats.add_capture_points("blue", config.CP_REWARD_PER_TICK)
                            metrics.cp_blue_tick += config.CP_REWARD_PER_TICK

                        winners_idx = None
                        if winner_team_id is not None:
                            winners_on_cp = on_cp & (teams_alive == winner_team_id)
                            if winners_on_cp.any():
                                winners_idx = alive_idx[winners_on_cp]
                                share = float(config.CP_REWARD_PER_TICK) / float(max(int(winners_idx.numel()), 1))
                                if hasattr(self.registry, "agent_uids"):
                                    cp_uids = self.registry.agent_uids.index_select(0, winners_idx).detach().cpu().tolist()
                                else:
                                    cp_uids = data.index_select(0, winners_idx)[:, COL_AGENT_ID].detach().cpu().tolist()
                                for uid in cp_uids:
                                    self.agent_cp_points[int(uid)] += float(share)

                        # Individual reward for contested CP
                        if red_on > 0 and blue_on > 0 and winners_idx is not None and winners_idx.numel() > 0:
                            reward_val = float(config.PPO_REWARD_CONTESTED_CP)

                            cp_add = torch.full(
                                (int(winners_idx.numel()),),
                                reward_val,
                                device=self.device,
                                dtype=torch.float32,
                            )

                            individual_rewards.index_add_(
                                0,
                                winners_idx,
                                cp_add.to(self._data_dt),
                            )
                            reward_contested_cp_individual.index_add_(0, winners_idx, cp_add)

        # 8) PPO REINFORCEMENT LEARNING LOGGING
        if self._ppo and rec_agent_ids:
            # IMPORTANT: these are REGISTRY SLOT IDS captured at decision time (not persistent UIDs).
            # PPO runtime is slot-based because it tracks per-slot state in the live registry.
            ppo_slot_ids = torch.cat(rec_agent_ids)

            # Per-agent HP reward each tick (dense shaping)
            current_hp = data[ppo_slot_ids, COL_HP]
            hp_reward_base = (current_hp * config.PPO_REWARD_HP_TICK).to(self._data_dt)
            hp_reward_mode = str(getattr(config, "PPO_HP_REWARD_MODE", "raw")).strip().lower()
            if hp_reward_mode == "threshold_ramp":
                hp_max = data[ppo_slot_ids, COL_HP_MAX].clamp_min(1e-8)
                hp_pct = (current_hp / hp_max).clamp(0.0, 1.0)

                hp_thr = float(getattr(config, "PPO_HP_REWARD_THRESHOLD", 0.60))
                hp_thr = max(0.0, min(1.0, hp_thr))
                if hp_thr >= 1.0:
                    hp_reward = torch.zeros_like(hp_reward_base)
                else:
                    ramp = ((hp_pct - hp_thr) / (1.0 - hp_thr)).clamp(0.0, 1.0)
                    hp_reward = hp_reward_base * ramp.to(self._data_dt)
            else:
                # Legacy/default behavior (backward compatible): linear in current HP.
                hp_reward = hp_reward_base

            rec_team_ids = torch.cat(rec_teams)
            team_is_red = (rec_team_ids == 2.0)

            team_kill_reward = torch.where(
                team_is_red,
                torch.full((int(ppo_slot_ids.numel()),), float(combat_bd * config.TEAM_KILL_REWARD), device=self.device, dtype=torch.float32),
                torch.full((int(ppo_slot_ids.numel()),), float(combat_rd * config.TEAM_KILL_REWARD), device=self.device, dtype=torch.float32),
            )
            team_death_reward = torch.where(
                team_is_red,
                torch.full((int(ppo_slot_ids.numel()),), float((combat_rd + meta_rd) * config.PPO_REWARD_DEATH), device=self.device, dtype=torch.float32),
                torch.full((int(ppo_slot_ids.numel()),), float((combat_bd + meta_bd) * config.PPO_REWARD_DEATH), device=self.device, dtype=torch.float32),
            )
            team_cp_reward = torch.where(
                team_is_red,
                torch.full((int(ppo_slot_ids.numel()),), float(metrics.cp_red_tick), device=self.device, dtype=torch.float32),
                torch.full((int(ppo_slot_ids.numel()),), float(metrics.cp_blue_tick), device=self.device, dtype=torch.float32),
            )

            individual_total_reward = (
                reward_kill_individual[ppo_slot_ids]
                + reward_damage_dealt_individual[ppo_slot_ids]
                + reward_damage_taken_penalty[ppo_slot_ids]
                + reward_contested_cp_individual[ppo_slot_ids]
                + reward_healing_recovered[ppo_slot_ids]
            )
            team_total_reward = team_kill_reward + team_death_reward + team_cp_reward
            hp_reward_f32 = hp_reward.to(torch.float32)

            # Final reward: individual + team + hp shaping
            final_rewards = (individual_total_reward + team_total_reward + hp_reward_f32).to(self._data_dt)

            if ppo_slot_ids.numel() > 0:
                if hasattr(self.registry, "agent_uids"):
                    merit_uids = self.registry.agent_uids.index_select(0, ppo_slot_ids).detach().cpu().tolist()
                else:
                    merit_uids = data.index_select(0, ppo_slot_ids)[:, COL_AGENT_ID].detach().cpu().tolist()
                merit_rewards = final_rewards.detach().float().cpu().tolist()
                for uid, reward_value in zip(merit_uids, merit_rewards):
                    self.agent_scores[int(uid)] += float(reward_value)

            if telemetry is not None and getattr(telemetry, "enabled", False):
                try:
                    self._telemetry_set_phase(telemetry, tick=int(self.stats.tick), phase="ppo_reward_components")
                    telemetry.record_ppo_reward_components_by_slot(
                        tick=int(self.stats.tick),
                        slot_ids=ppo_slot_ids,
                        reward_total=final_rewards.to(torch.float32),
                        reward_hp=hp_reward_f32,
                        reward_kill_individual=reward_kill_individual[ppo_slot_ids],
                        reward_damage_dealt_individual=reward_damage_dealt_individual[ppo_slot_ids],
                        reward_damage_taken_penalty=reward_damage_taken_penalty[ppo_slot_ids],
                        reward_contested_cp_individual=reward_contested_cp_individual[ppo_slot_ids],
                        reward_team_kill=team_kill_reward,
                        reward_team_death=team_death_reward,
                        reward_team_cp=team_cp_reward,
                        reward_healing_recovered=reward_healing_recovered[ppo_slot_ids],
                    )
                except Exception as e:
                    try:
                        telemetry._anomaly(f"tick.ppo_reward_components hook failed: {e}")
                    except Exception:
                        pass

            # PPO window bootstrap now comes from the persistent slot-local value cache.
            # On the final step of a window, record_step() stages the boundary batch.
            # The next tick's *normal* main inference pass updates the cache with V(s_{t+1})
            # for the surviving slots, and finalize_pending_window_from_cache() consumes it.
            # This removes the old duplicate post-step:
            #   positions_xy -> _build_transformer_obs -> build_buckets -> ensemble_forward
            # bootstrap-only pass from the hot path.
            with torch.enable_grad():
                self._ppo.record_step(
                    agent_ids=ppo_slot_ids,
                    obs=torch.cat(rec_obs),
                    logits=torch.cat(rec_logits),
                    values=torch.cat(rec_values),
                    actions=torch.cat(rec_actions),
                    rewards=final_rewards,
                    done=(data[ppo_slot_ids, COL_ALIVE] <= 0.5),
                    action_masks=torch.cat(rec_action_masks),
                    bootstrap_values=None,
                )

        # 9) Advance tick, flush PPO dead, respawn
        self.stats.on_tick_advanced(1)
        metrics.tick = int(self.stats.tick)

        was_dead = (data[:, COL_ALIVE] <= 0.5) if self._ppo is not None else None

        # Flush dead agents from PPO before respawn
        if was_dead is not None:
            dead_slots = was_dead.nonzero(as_tuple=False).squeeze(1)
            if dead_slots.numel() > 0:
                self._ppo.flush_agents(dead_slots)

        # Respawn dead units
        self.respawner.step(self.stats.tick, self.registry, self.grid, engine=self)

        # TELEMETRY: births + lineage edges
        if telemetry is not None and getattr(telemetry, "enabled", False):
            try:
                # Birth events happen on the advanced tick (post self.stats.on_tick_advanced(1)).
                self._telemetry_set_phase(telemetry, tick=int(self.stats.tick), phase="respawn_birth")
                meta = getattr(self.respawner, "last_spawn_meta", None) or []
                telemetry.ingest_spawn_meta(meta)
            except Exception as e:
                try:
                    telemetry._anomaly(f"respawn meta ingest failed: {e}")
                except Exception:
                    pass

        if was_dead is not None:
            self._ppo_reset_on_respawn(was_dead)

        self._debug_invariants("post_respawn")

        # Periodic telemetry flush/validation
        if telemetry is not None and getattr(telemetry, "enabled", False):
            try:
                if hasattr(telemetry, "clear_event_context"):
                    telemetry.clear_event_context()
                telemetry.on_tick_end(metrics.tick)
            except Exception:
                pass

        # Return metrics as a dict
        return vars(metrics)
