# Infinite_War_Simulation/engine/tick.py
# =============================================================================
# Tick Engine (Combat-First, Vectorized, Multi-Agent RL Simulation)
# =============================================================================
#
# This file implements the **core simulation loop** (“tick loop”) for a large,
# grid-based, multi-agent environment. The environment is designed to run
# **thousands of agents in parallel** using **PyTorch tensor operations**,
# typically on a **GPU (CUDA)**.
#
# ---------------------------------------------------------------------------
# HIGH-LEVEL CONCEPTS (READ THIS FIRST)
# ---------------------------------------------------------------------------
#
# 1) "Tick" / "Frame" / "Step"
#    A simulation advances in discrete time steps called ticks. Each tick:
#       - Agents observe the world (Observation construction)
#       - Agents choose actions (Policy / Neural network forward pass)
#       - The environment applies physics/game rules (combat, movement, zones)
#       - Rewards/telemetry are recorded (for RL training and analytics)
#       - Time advances by 1
#
# 2) Vectorization (Why PyTorch?)
#    The key performance goal is to avoid Python loops over agents.
#    Instead, we represent agent properties as tensors:
#       - positions: (N,)
#       - health:    (N,)
#       - actions:   (N,)
#    and compute updates in parallel on GPU.
#
# 3) World State Exists in Two Places (CRITICAL INVARIANT)
#    - Agent registry: per-agent “truth” (positions, hp, team, unit, alive flag)
#    - Grid tensor: a spatial map used for fast queries (occupancy, hp channel, slot id)
#
#    These must remain consistent. Any mismatch can cause subtle bugs:
#       - “ghost agents” on the grid but dead in registry
#       - slot-id mismatches
#       - collisions / combat targeting errors
#
# 4) Combat-First Semantics
#    In THIS engine: combat resolves BEFORE movement.
#    Consequence:
#       - An agent killed in combat does NOT get to move in the same tick.
#    This changes dynamics compared to movement-first engines.
#
# 5) PPO Integration (Reinforcement Learning)
#    PPO (Proximal Policy Optimization) requires collecting trajectories:
#       - obs_t, action_t, logits_t, value_t, reward_t, done_t, bootstrap_value_t+1
#    This engine records those per tick (if PPO is enabled).
#
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass
import collections
import os
from typing import Dict, Optional, List, Tuple, TYPE_CHECKING

# -------------------------------------------------------------------------
# PyTorch is used for tensor math and GPU acceleration.
# -------------------------------------------------------------------------
import torch

# -------------------------------------------------------------------------
# Project imports (simulation config + engine subsystems)
# -------------------------------------------------------------------------
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

from agent.ensemble import ensemble_forward
from agent.transformer_brain import TransformerBrain  # (Imported, may be used elsewhere / kept as-is)

# -------------------------------------------------------------------------
# TYPE_CHECKING is True only for static analysis / IDE type checking.
# This avoids runtime circular imports while still enabling autocomplete.
# -------------------------------------------------------------------------
if TYPE_CHECKING:
    from rl.ppo_runtime import PerAgentPPORuntime

# -------------------------------------------------------------------------
# Optional PPO runtime import:
# The PPO runtime might not exist / might fail to import in some deployments.
# In that case, PPO is disabled at runtime.
# -------------------------------------------------------------------------
try:
    from rl.ppo_runtime import PerAgentPPORuntime as _PerAgentPPORuntimeRT
except Exception:
    _PerAgentPPORuntimeRT = None


# =============================================================================
# TickMetrics: per-tick counters (telemetry-like numeric metrics)
# =============================================================================
#
# A dataclass automatically generates:
#   - __init__
#   - readable __repr__
#   - comparisons (optional)
#
# Here it is used as a simple “struct” to track what happened in one tick.
#
# NOTE ABOUT DESIGN:
# - These are cheap scalar aggregates (Python ints/floats) updated once per tick.
# - They are meant for printing/logging/telemetry dashboards, not for training.
# - Keeping them minimal avoids overhead in the hot loop.
# =============================================================================
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


# =============================================================================
# TickEngine: the main simulation stepper
# =============================================================================
#
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
#
# IMPORTANT: This implementation emphasizes:
#   - GPU parallelism
#   - deterministic aggregation where duplicate indices occur (e.g., multi-attacker damage)
#   - invariant checks (optional, debug mode)
# =============================================================================
class TickEngine:
    # =========================================================================
    # CORE THEORY: THE GAME LOOP (Tick Engine)
    # A simulation runs in "Ticks" or frames. Each tick, the engine looks at the
    # world, decides what every entity should do, and then applies the rules of
    # physics/gameplay. This class manages that massive update.
    #
    # Responsibilities:
    #   - Process attacks (damage, kills)
    #   - Apply deaths (combat-first)
    #   - Move agents (with conflict resolution)
    #   - Apply zone healing and capture point scoring
    #   - Record telemetry and PPO data
    #   - Respawn dead agents
    # =========================================================================

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

        # agent_scores: reward tally keyed by persistent agent identifier (not slot).
        # defaultdict(float) ensures missing keys default to 0.0.
        self.agent_scores: Dict[int, float] = collections.defaultdict(float)

        # Zones and corresponding cached tensors on the simulation device.
        self.zones: Optional[Zones] = zones
        self._z_heal: Optional[torch.Tensor] = None
        self._z_cp_masks: List[torch.Tensor] = []
        self._ensure_zone_tensors()

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

        # ================================================================
        # Instinct cache (computed under no_grad)
        # ================================================================
        #
        # "Instinct" here is a computed feature: local density of allies/enemies
        # around each agent within a radius R. Computing offsets for a discrete
        # circle can be expensive, so we cache offsets by radius.
        #
        self._instinct_cached_r: int = -999999
        self._instinct_offsets: Optional[torch.Tensor] = None
        self._instinct_area: float = 1.0

        # ================================================================
        # THEORY: Constants on GPU
        # ================================================================
        #
        # Creating small tensors repeatedly can be expensive and can cause
        # device synchronization overhead. So we allocate frequently used
        # constants once on the correct device and dtype.
        #
        self._g0 = torch.tensor(0.0, device=self.device, dtype=self._grid_dt)     # grid “zero”
        self._gneg = torch.tensor(-1.0, device=self.device, dtype=self._grid_dt)  # grid “-1”
        self._d0 = torch.tensor(0.0, device=self.device, dtype=self._data_dt)     # data “zero”

        # ================================================================
        # PPO integration
        # ================================================================
        #
        # PPO_ENABLED: feature flag in config.
        # If enabled and runtime class is importable, initialize PPO runtime.
        #
        self._ppo_enabled = bool(getattr(config, "PPO_ENABLED", False))
        self._ppo: Optional["PerAgentPPORuntime"] = None
        if self._ppo_enabled and _PerAgentPPORuntimeRT is not None:
            self._ppo = _PerAgentPPORuntimeRT(
                registry=self.registry,
                device=self.device,
                obs_dim=self._OBS_DIM,
                act_dim=self._ACTIONS,
            )

    # -------------------------------------------------------------------------
    # PPO: reset state for agents that just respawned
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # Zones: ensure masks are on the correct device
    # -------------------------------------------------------------------------
    def _ensure_zone_tensors(self) -> None:
        """
        Convert zone masks to tensors on the simulation device.

        - heal_mask: boolean grid marking cells that heal agents standing on them
        - cp_masks: list of boolean grids marking capture point regions
        """
        self._z_heal, self._z_cp_masks = None, []
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
        except Exception as e:
            # Safety: if zones fail, disable them rather than crash the simulation.
            print(f"[tick] WARN: zone tensor setup failed ({e}); zones disabled.")

    # -------------------------------------------------------------------------
    # Indexing helper
    # -------------------------------------------------------------------------
    @staticmethod
    def _as_long(x: torch.Tensor) -> torch.Tensor:
        """
        Convert tensor to long dtype (used for indexing).

        PyTorch indexing generally requires integer types (torch.long).
        If you attempt to index with float tensors, it will error.
        """
        return x.to(torch.long)

    # -------------------------------------------------------------------------
    # Identity helpers (SLOT ID vs PERSISTENT AGENT UID)
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # Grid HP sync helper (minimize grid/registry desync windows)
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # Optional telemetry phase context helper (safe no-op if telemetry lacks API)
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # Alive filtering
    # -------------------------------------------------------------------------
    def _recompute_alive_idx(self) -> torch.Tensor:
        """
        Returns a 1D tensor of slot indices where the agent is alive.

        Performance rationale:
        - Many tensors are capacity-sized, but only a subset are alive.
        - Filtering early reduces compute cost in later phases.
        """
        return (self.registry.agent_data[:, COL_ALIVE] > 0.5).nonzero(as_tuple=False).squeeze(1)

    # -------------------------------------------------------------------------
    # Instinct offsets: cached discrete circle offsets
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # Instinct feature computation
    # -------------------------------------------------------------------------
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
        if M <= 0 or area <= 0.0:
            return torch.zeros((N, 4), device=self.device, dtype=self._data_dt)

        # ---------------------------------------------------------------------
        # Broadcasting (important tensor technique)
        # ---------------------------------------------------------------------
        # x0: (N,1), y0: (N,1)
        # ox: (1,M), oy: (1,M)
        # x0+ox => (N,M), y0+oy => (N,M)
        #
        # This computes all neighborhood coordinates for all agents at once.
        # ---------------------------------------------------------------------
        x0 = pos_xy[:, 0].to(torch.long).view(N, 1)
        y0 = pos_xy[:, 1].to(torch.long).view(N, 1)
        ox = offsets[:, 0].view(1, M)
        oy = offsets[:, 1].view(1, M)

        # Clamp to map bounds to avoid indexing outside the grid.
        xx = (x0 + ox).clamp(0, self.W - 1)
        yy = (y0 + oy).clamp(0, self.H - 1)

        # occ: occupancy/team at each sampled cell
        # uid: unit id at each sampled cell
        occ = self.grid[0][yy, xx]
        uid = unit_map[yy, xx]

        # teams for each alive agent
        teams = data[alive_idx, COL_TEAM]
        team_is_red = (teams == 2.0)

        # Determine ally and enemy occupancy codes for each agent.
        # These values (2.0 and 3.0) are project conventions for team encoding.
        ally_occ = torch.where(team_is_red, occ.new_full((N,), 2.0), occ.new_full((N,), 3.0)).view(N, 1)
        enemy_occ = torch.where(team_is_red, occ.new_full((N,), 3.0), occ.new_full((N,), 2.0)).view(N, 1)

        ally_mask = (occ == ally_occ)
        enemy_mask = (occ == enemy_occ)

        # Unit types:
        # 1 = soldier, 2 = archer (as implied by later checks)
        ally_arch = ally_mask & (uid == 2)
        ally_sold = ally_mask & (uid == 1)

        # Count nearby allies/enemies per agent by summing along offset dimension.
        ally_arch_c = ally_arch.sum(dim=1).to(torch.float32)
        ally_sold_c = ally_sold.sum(dim=1).to(torch.float32)
        enemy_c = enemy_mask.sum(dim=1).to(torch.float32)

        # Remove self-count (an agent should not count itself as a neighbor).
        self_unit = data[alive_idx, COL_UNIT]
        ally_arch_c = (ally_arch_c - (self_unit == 2.0).to(torch.float32)).clamp_min(0.0)
        ally_sold_c = (ally_sold_c - (self_unit == 1.0).to(torch.float32)).clamp_min(0.0)

        # Add noise to enemy count (fog-of-war style uncertainty).
        noise = torch.randn((N,), device=self.device, dtype=torch.float32) * 0.25
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

        out = torch.stack([ally_arch_d, ally_sold_d, enemy_d, threat], dim=1)
        return out.to(dtype=self._data_dt)

    # -------------------------------------------------------------------------
    # Death application: remove dead agents from grid and update stats/metrics
    # -------------------------------------------------------------------------
    def _apply_deaths(
        self,
        sel: torch.Tensor,
        metrics: TickMetrics,
        credit_kills: bool = True,
        death_cause: str = "unknown",
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

        # Snapshot metadata BEFORE mutation (positions/teams/units/uids are still readable now).
        dead_slots = [int(s) for s in dead_idx.tolist()]
        dead_ids = self._slot_ids_to_agent_uids_list(dead_idx)
        dead_team_list = [int(data[slot, COL_TEAM].item()) for slot in dead_slots]
        dead_unit_list = [int(data[slot, COL_UNIT].item()) for slot in dead_slots]

        # Count deaths by team encoding (read before ALIVE zeroing).
        dead_team = data[dead_idx, COL_TEAM]
        red_deaths = int((dead_team == 2.0).sum().item())
        blue_deaths = int((dead_team == 3.0).sum().item())

        # Update global stats (semantic counters).
        if red_deaths:
            self.stats.add_death("red", red_deaths)
            if credit_kills:
                self.stats.add_kill("blue", red_deaths)

        if blue_deaths:
            self.stats.add_death("blue", blue_deaths)
            if credit_kills:
                self.stats.add_kill("red", blue_deaths)

        # -----------------------------------------------------------------
        # STATE MUTATION (registry + grid)
        # -----------------------------------------------------------------
        # We apply the world-state mutation before emitting death telemetry so that
        # post-death readers do not observe "death event emitted but agent still alive on grid".
        gx = self._as_long(data[dead_idx, COL_X])
        gy = self._as_long(data[dead_idx, COL_Y])
                # Back-compat structured death log for ResultsWriter -> dead_agents_log.csv.
        # Legacy schema requires a killer_team field, so we only emit entries for
        # combat deaths where kill credit is enabled (team-level attribution is known).
        if death_cause == "combat" and credit_kills:
            try:
                rec_fn = getattr(self.stats, "record_death_entry", None)
                if callable(rec_fn):
                    n = min(len(dead_ids), len(dead_team_list), int(gx.numel()), int(gy.numel()))
                    for i in range(n):
                        victim_team = int(dead_team_list[i])
                        if victim_team not in (2, 3):
                            continue  # defensive: unknown team encoding

                        # Team-level attribution only (legacy CSV stores killer_team, not killer_id)
                        killer_team = 3 if victim_team == 2 else 2

                        rec_fn(
                            agent_id=int(dead_ids[i]),
                            team_id_val=float(victim_team),
                            x=int(gx[i].item()),
                            y=int(gy[i].item()),
                            killer_team_id_val=float(killer_team),
                        )
            except Exception:
                # Fail-safe: legacy death CSV must never break the simulation tick.
                pass
        # Clear occupancy / hp / slot-id in grid first (spatial truth for fast queries).
        self.grid[0][gy, gx], self.grid[1][gy, gx], self.grid[2][gy, gx] = self._g0, self._g0, self._gneg

        # Then mark registry ALIVE=0 (agent remains in slot storage, but dead).
        data[dead_idx, COL_ALIVE] = self._d0

        # -----------------------------------------------------------------
        # TELEMETRY (after state mutation)
        # -----------------------------------------------------------------
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
                )
            except Exception as e:
                try:
                    telemetry._anomaly(f"_apply_deaths telemetry hook failed: {e}")
                except Exception:
                    pass

        metrics.deaths += int(dead_idx.numel())
        return red_deaths, blue_deaths

    # -------------------------------------------------------------------------
    # Observation builder for transformer/policy
    # -------------------------------------------------------------------------
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
            on_heal = torch.zeros(N, device=self.device, dtype=torch.bool)

        on_cp = torch.zeros(N, device=self.device, dtype=torch.bool)
        if self._z_cp_masks:
            for cp_mask in self._z_cp_masks:
                # |= accumulates: if any CP mask is True at that cell, on_cp becomes True.
                on_cp |= cp_mask[pos_xy[:, 1], pos_xy[:, 0]]

        def _norm_const(v: float, scale: float) -> torch.Tensor:
            """
            Normalize a scalar constant into a per-agent vector.

            Neural networks train more stably when inputs are within reasonable ranges,
            often near [-1,1] or [0,1]. For large numbers (like tick=50,000),
            raw input can destabilize gradients and learning dynamics.

            So we normalize by a fixed scale factor.
            """
            s = scale if scale > 0 else 1.0
            return torch.full((N,), v / s, dtype=self._data_dt, device=self.device)

        expected_ray_dim = 32 * 8

        # unit_map is a grid-shaped tensor encoding which unit type occupies each cell.
        unit_map = build_unit_map(data, self.grid)

        # Raycasting: 32 rays, “first hit” info, bounded by vision range per agent.
        rays = raycast32_firsthit(
            pos_xy, self.grid, unit_map,
            max_steps_each=data[alive_idx, COL_VISION].long()
        )
        if rays.shape != (N, expected_ray_dim):
            raise RuntimeError(
                f"[obs] ray tensor shape mismatch: got {tuple(rays.shape)}, "
                f"expected ({N}, {expected_ray_dim})."
            )

        # hp_max clamped to avoid division-by-zero.
        hp_max = data[alive_idx, COL_HP_MAX].clamp_min(1.0)

        # Rich features:
        # - normalized HP (hp/hp_max)
        # - normalized positions (x/(W-1), y/(H-1))
        # - one-hot-ish flags for team and unit types
        # - normalized attack and vision
        # - zone flags
        # - normalized global stats
        #
        # NOTE: Several torch.zeros paddings are used to maintain exact OBS_DIM layout.
        rich_base = torch.stack([
            data[alive_idx, COL_HP] / hp_max,
            data[alive_idx, COL_X] / (self.W - 1),
            data[alive_idx, COL_Y] / (self.H - 1),
            (data[alive_idx, COL_TEAM] == 2.0),
            (data[alive_idx, COL_TEAM] == 3.0),
            (data[alive_idx, COL_UNIT] == 1.0),
            (data[alive_idx, COL_UNIT] == 2.0),
            data[alive_idx, COL_ATK] / (config.MAX_ATK or 1.0),
            data[alive_idx, COL_VISION] / (config.RAYCAST_MAX_STEPS or 15.0),
            on_heal.to(self._data_dt),
            on_cp.to(self._data_dt),
            _norm_const(float(self.stats.tick), 50000.0),
            _norm_const(self.stats.red.score, 1000.0), _norm_const(self.stats.blue.score, 1000.0),
            _norm_const(self.stats.red.cp_points, 500.0), _norm_const(self.stats.blue.cp_points, 500.0),
            _norm_const(self.stats.red.kills, 500.0), _norm_const(self.stats.blue.kills, 500.0),
            _norm_const(self.stats.red.deaths, 500.0), _norm_const(self.stats.blue.deaths, 500.0),
            # Padding slots keep the layout exactly matching the required dimension sizes.
            torch.zeros(N, device=self.device, dtype=self._data_dt),
            torch.zeros(N, device=self.device, dtype=self._data_dt),
            torch.zeros(N, device=self.device, dtype=self._data_dt),
        ], dim=1).to(dtype=self._data_dt)

        instinct = self._compute_instinct_context(alive_idx=alive_idx, pos_xy=pos_xy, unit_map=unit_map)
        if instinct.shape != (N, 4):
            raise RuntimeError(f"instinct shape {tuple(instinct.shape)} != (N,4)")

        # Concatenate base rich features and instinct features.
        rich = torch.cat([rich_base, instinct], dim=1)

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

    # -------------------------------------------------------------------------
    # Debug invariants to catch grid<->registry desync
    # -------------------------------------------------------------------------
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

    # =============================================================================
    # run_tick: main step function
    # =============================================================================
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

        # If absolutely everyone is dead, fast-forward time and respawn.
        if alive_idx.numel() == 0:
            self.stats.on_tick_advanced(1)
            metrics.tick = int(self.stats.tick)

            # PPO bookkeeping: flush dead agents if PPO enabled.
            was_dead = (data[:, COL_ALIVE] <= 0.5) if self._ppo is not None else None
            if was_dead is not None:
                dead_slots = was_dead.nonzero(as_tuple=False).squeeze(1)
                if dead_slots.numel() > 0:
                    self._ppo.flush_agents(dead_slots)

            # Respawn.
            self.respawner.step(self.stats.tick, self.registry, self.grid)

            # Reset PPO state for respawned slots.
            if was_dead is not None:
                self._ppo_reset_on_respawn(was_dead)

            self._debug_invariants("post_respawn")
            return vars(metrics)

        # ---------------------------------------------------------------------
        # 1) GET OBSERVATIONS
        # ---------------------------------------------------------------------
        pos_xy = self.registry.positions_xy(alive_idx)
        obs = self._build_transformer_obs(alive_idx, pos_xy)
        if obs.dim() != 2 or int(obs.shape[1]) != int(config.OBS_DIM):
            raise RuntimeError(
                f"[obs] shape mismatch: got {tuple(obs.shape)}, expected (N,{int(config.OBS_DIM)})"
            )

        # ---------------------------------------------------------------------
        # 2) GET ACTION MASK (prevents illegal actions)
        # ---------------------------------------------------------------------
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

        # ---------------------------------------------------------------------
        # 3) AI DECISION TIME (bucketed inference)
        # ---------------------------------------------------------------------
        # Agents can have different models/brains. We group them into buckets to run
        # ensemble_forward efficiently per model type.
        for bucket in self.registry.build_buckets(alive_idx):
            # loc are positions of bucket.indices within alive_idx (sorted alignment assumed)
            loc = torch.searchsorted(alive_idx, bucket.indices)

            # Forward pass: returns distribution (dist) and value estimate (vals)
            dist, vals = ensemble_forward(bucket.models, obs[loc])

            # Mask logits: illegal actions get very negative logits (approx -inf)
            logits32 = torch.where(mask[loc], dist.logits, torch.finfo(torch.float32).min).to(torch.float32)

            # Sample action from categorical distribution
            a = torch.distributions.Categorical(logits=logits32).sample()

            if self._ppo:
                # Store trajectory elements needed for PPO training
                rec_agent_ids.append(bucket.indices)
                rec_obs.append(obs[loc])
                rec_logits.append(logits32)
                rec_values.append(vals)
                rec_actions.append(a)
                rec_action_masks.append(mask[loc])
                rec_teams.append(data[bucket.indices, COL_TEAM])

            actions[loc] = a

        # ---------------------------------------------------------------------
        # Debug: force actions / validate actions
        # ---------------------------------------------------------------------
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
                if (actions < 0).any() or (actions >= config.NUM_ACTIONS).any():
                    raise RuntimeError("[tick] sampled action out of range (mask/sampling bug)")
                ar = torch.arange(actions.numel(), device=actions.device)
                ok = mask[ar, actions]
                if not bool(ok.all().item()):
                    bad = ar[~ok][:8].detach().cpu().tolist()
                    raise RuntimeError(
                        f"[tick] chosen action is masked out (first_bad_local_idx={bad}, use FWS_DEBUG_FORCE_ACTIONS to reproduce)"
                    )

        metrics.alive = int(alive_idx.numel())

        combat_rd, combat_bd = 0, 0
        meta_rd, meta_bd = 0, 0

        # Individual rewards tensor over ALL slots (capacity-sized).
        # This is later indexed for just the participating agents in PPO logging.
        individual_rewards = torch.zeros(self.registry.capacity, device=self.device, dtype=self._data_dt)

        # ---------------------------------------------------------------------
        # 4) COMBAT (combat-first semantics)
        # ---------------------------------------------------------------------
        if alive_idx.numel() > 0:
            # Attack actions are encoded as action >= 9 in this project.
            if (is_attack := actions >= 9).any():
                atk_idx, atk_act = alive_idx[is_attack], actions[is_attack]

                # Action decoding:
                # - atk_act in [9..] encodes a direction and a range (1..4)
                # r = ((atk_act - 9) % 4) + 1  => range ∈ {1,2,3,4}
                # dir_idx = (atk_act - 9) // 4 => direction index into DIRS8
                #
                # Math note:
                # A single integer encodes 2 parameters via quotient and remainder.
                # This is a standard encoding pattern:
                #   x = q * base + rem
                #   q = x // base
                #   rem = x % base
                #
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
                        # =====================================================
                        # DETERMINISTIC FOCUS-FIRE DAMAGE
                        # =====================================================
                        #
                        # Duplicate index hazard:
                        #   If multiple attackers hit the same victim, writing
                        #   victim_hp[victim] -= dmg in parallel can race.
                        #
                        # Safe approach:
                        #   1) Sort by victim id
                        #   2) Compute total damage per unique victim
                        #   3) Apply hp -= total_damage once per victim
                        #
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

                        # PATCH: reward/credit exactly one attacker per kill (not all contributors).
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

                                individual_rewards[uniq_k] += (k_counts.to(self._data_dt) * reward_val)

                                # agent_scores keyed by persistent agent id
                                for killer_slot, cnt in zip(uniq_k.tolist(), k_counts.tolist()):
                                    if hasattr(self.registry, "agent_uids"):
                                        uid = int(self.registry.agent_uids[killer_slot].item())
                                    else:
                                        uid = int(data[killer_slot, COL_AGENT_ID].item())
                                    self.agent_scores[uid] += reward_val * float(cnt)

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
                            individual_rewards[uniq_a] += (dmg_a.to(self._data_dt) * float(ppo_dmg_dealt_coef))

                        if ppo_dmg_taken_pen != 0.0 and uniq_v.numel() > 0:
                            individual_rewards[uniq_v] -= (dmg_sum.to(self._data_dt) * float(ppo_dmg_taken_pen))

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

        # ---------------------------------------------------------------------
        # 5) Apply deaths after combat
        # ---------------------------------------------------------------------
        # IMPORTANT SEMANTIC:
        # - Combat damage is applied first.
        # - Combat deaths are resolved here (before movement).
        # Therefore, agents killed in combat do NOT move this tick.
        rD, bD = self._apply_deaths(
            (data[:, COL_ALIVE] > 0.5) & (data[:, COL_HP] <= 0.0),
            metrics,
            credit_kills=True,
            death_cause="combat",
        )
        metrics.deaths_combat += (rD + bD)
        combat_rd += rD
        combat_bd += bD
        self._debug_invariants("post_combat")

        # ---------------------------------------------------------------------
        # 6) MOVE HANDLING WITH CONFLICT RESOLUTION (Law 1)
        # ---------------------------------------------------------------------
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
            # -----------------------------------------------------------------
            # 6.1) Compute intended destinations for ALL attempted moves
            # -----------------------------------------------------------------
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
            g_wall = (self._g0 + 1.0)

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

            # -----------------------------------------------------------------
            # 6.2) Resolve moves among those whose destination is empty
            # -----------------------------------------------------------------
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
                num_cells = int(self.H * self.W)
                claim_cnt = torch.zeros((num_cells,), device=self.device, dtype=torch.int32)
                claim_cnt.scatter_add_(0, dest_key, torch.ones_like(dest_key, dtype=torch.int32))

                try:
                    # scatter_reduce_ (amax) finds per-cell maximum HP among claimants.
                    max_hp = torch.full((num_cells,), torch.finfo(hp.dtype).min, device=self.device, dtype=hp.dtype)
                    max_hp.scatter_reduce_(0, dest_key, hp, reduce="amax", include_self=True)
                    is_max = (hp == max_hp[dest_key])

                    # Count how many claimants share the max HP (tie detection).
                    max_cnt = torch.zeros((num_cells,), device=self.device, dtype=torch.int32)
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
                win_cnt = torch.zeros((num_cells,), device=self.device, dtype=torch.int32)
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

            # -----------------------------------------------------------------
            # 6.3) Telemetry: movement aggregates + optional per-agent events
            # -----------------------------------------------------------------
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

                # -------------------------------------------------------------
                # Per-agent movement totals (AgentLife snapshot)
                # -------------------------------------------------------------
                # NOTE: This runs *after* conflict resolution and after winner moves are applied,
                # so outcomes are finalized.
                #
                # Why this exists:
                # - The simulation already records high-level move aggregates (move_summary.csv)
                # - It also optionally records sampled per-move events (events_*.jsonl)
                # - But AgentLife snapshots (agent_life.csv) typically want *per-agent totals*
                #   (e.g., moves attempted/success/blocked counts) so you can analyze behavior
                #   per agent over long runs without reading huge event logs.
                #
                # This hook sends, in one vectorized call, the "attempted move outcome code"
                # for every slot that attempted to move this tick. Telemetry can then accumulate
                # these per slot across ticks.
                out_code_all = None
                try:
                    n_all = int(move_idx_all.numel())
                    if n_all > 0 and hasattr(telemetry, "record_move_totals_by_slot"):
                        # Outcome codes for attempted moves:
                        # 0=success | 1=blocked_wall | 2=blocked_occupied | 3=conflict_lost | 4=conflict_tie
                        #
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

        # ---------------------------------------------------------------------
        # 7) ENVIRONMENT (Heal, Metabolism, Objectives)
        # ---------------------------------------------------------------------
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
                data[heal_idx, COL_HP] = (data[heal_idx, COL_HP] + config.HEAL_RATE).clamp_max(data[heal_idx, COL_HP_MAX])
                self.grid[1, pos_xy[on_heal, 1], pos_xy[on_heal, 0]] = data[heal_idx, COL_HP].to(self._grid_dt)

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
                    metrics.deaths_metabolism += (rD + bD)
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
                        if red_on > blue_on:
                            self.stats.add_capture_points("red", config.CP_REWARD_PER_TICK)
                            metrics.cp_red_tick += config.CP_REWARD_PER_TICK
                        elif blue_on > red_on:
                            self.stats.add_capture_points("blue", config.CP_REWARD_PER_TICK)
                            metrics.cp_blue_tick += config.CP_REWARD_PER_TICK

                        # Individual reward for contested CP
                        if red_on > 0 and blue_on > 0:
                            winners_on_cp = None
                            if red_on > blue_on:
                                winners_on_cp = on_cp & (teams_alive == 2.0)
                            elif blue_on > red_on:
                                winners_on_cp = on_cp & (teams_alive == 3.0)
                            if winners_on_cp is not None and winners_on_cp.any():
                                winners_idx = alive_idx[winners_on_cp]
                                reward_val = config.PPO_REWARD_CONTESTED_CP
                                individual_rewards.index_add_(
                                    0,
                                    winners_idx,
                                    torch.full_like(winners_idx, reward_val, dtype=self._data_dt),
                                )

        # ---------------------------------------------------------------------
        # 8) PPO REINFORCEMENT LEARNING LOGGING
        # ---------------------------------------------------------------------
        if self._ppo and rec_agent_ids:
            # IMPORTANT: these are REGISTRY SLOT IDS captured at decision time (not persistent UIDs).
            # PPO runtime is slot-based because it tracks per-slot state in the live registry.
            ppo_slot_ids = torch.cat(rec_agent_ids)

            # Team-level rewards:
            # - TEAM_KILL_REWARD for kills
            # - PPO_REWARD_DEATH for deaths (may be negative)
            # - capture point reward from metrics.cp_*_tick
            team_r_rew = (combat_bd * config.TEAM_KILL_REWARD) + ((combat_rd + meta_rd) * config.PPO_REWARD_DEATH) + metrics.cp_red_tick
            team_b_rew = (combat_rd * config.TEAM_KILL_REWARD) + ((combat_bd + meta_bd) * config.PPO_REWARD_DEATH) + metrics.cp_blue_tick

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

            # Final reward: individual + team + hp shaping
            final_rewards = individual_rewards[ppo_slot_ids] + torch.where(torch.cat(rec_teams) == 2.0, team_r_rew, team_b_rew) + hp_reward

            # Bootstrapping values for next state if training next step
            #
            # NOTE (PPO/GAE semantics):
            # bootstrap_values here are V(s_{t+1}) for surviving slots, so they are computed
            # from POST-resolution observations (after combat/move/env in this tick).
            # This is expected behavior for standard actor-critic bootstrapping.
            bootstrap_values = None
            if self._ppo.will_train_next_step():
                alive_mask = (data[ppo_slot_ids, COL_ALIVE] > 0.5)
                bootstrap_values = torch.zeros((ppo_slot_ids.numel(),), device=ppo_slot_ids.device, dtype=torch.float32)
                if alive_mask.any():
                    alive_pos = alive_mask.nonzero(as_tuple=False).squeeze(1)
                    post_step_slot_ids = ppo_slot_ids[alive_pos]

                    order = torch.argsort(post_step_slot_ids)
                    post_step_slot_ids = post_step_slot_ids[order]
                    alive_pos = alive_pos[order]

                    pos_xy_post = self.registry.positions_xy(post_step_slot_ids)
                    obs_post = self._build_transformer_obs(post_step_slot_ids, pos_xy_post)

                    for bucket in self.registry.build_buckets(post_step_slot_ids):
                        loc = torch.searchsorted(post_step_slot_ids, bucket.indices)
                        _, vals = ensemble_forward(bucket.models, obs_post[loc])
                        bootstrap_values[alive_pos[loc]] = vals.to(torch.float32)

            # PPO needs gradients to update the policy/value networks.
            # The rest of the tick uses no_grad for speed.
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
                    bootstrap_values=bootstrap_values,
                )

        # ---------------------------------------------------------------------
        # 9) Advance tick, flush PPO dead, respawn
        # ---------------------------------------------------------------------
        self.stats.on_tick_advanced(1)
        metrics.tick = int(self.stats.tick)

        was_dead = (data[:, COL_ALIVE] <= 0.5) if self._ppo is not None else None

        # Flush dead agents from PPO before respawn
        if was_dead is not None:
            dead_slots = was_dead.nonzero(as_tuple=False).squeeze(1)
            if dead_slots.numel() > 0:
                self._ppo.flush_agents(dead_slots)

        # Respawn dead units
        self.respawner.step(self.stats.tick, self.registry, self.grid)

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
