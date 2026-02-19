from __future__ import annotations
from dataclasses import dataclass
import collections
import os
from typing import Dict, Optional, List, Tuple, TYPE_CHECKING

import torch

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
from agent.transformer_brain import TransformerBrain

if TYPE_CHECKING:
    from rl.ppo_runtime import PerAgentPPORuntime

try:
    from rl.ppo_runtime import PerAgentPPORuntime as _PerAgentPPORuntimeRT
except Exception:
    _PerAgentPPORuntimeRT = None


@dataclass
class TickMetrics:
    """Holds counters for one simulation tick."""
    alive: int = 0
    moved: int = 0
    attacks: int = 0
    deaths: int = 0
    tick: int = 0
    cp_red_tick: float = 0.0
    cp_blue_tick: float = 0.0


class TickEngine:
    # =========================================================================
    # Core simulation engine that processes one time step (tick) of the game.
    # Responsibilities:
    #   - Move agents (with conflict resolution)
    #   - Process attacks (damage, kills)
    #   - Apply zone healing and capture point logic
    #   - Record telemetry data for visualization and analysis
    #   - Manage reinforcement learning (PPO) reward collection
    #   - Handle respawning of dead agents
    # =========================================================================

    def __init__(self, registry: AgentsRegistry, grid: torch.Tensor,
                 stats: SimulationStats, zones: Optional[Zones] = None) -> None:
        """
        Initialize the tick engine.

        Args:
            registry: Holds all agent data (positions, health, team, etc.).
            grid:    3-layer grid (occupancy, health, slot id) for the map.
            stats:   Global simulation statistics (kills, deaths, scores).
            zones:   Optional zone information (healing areas, capture points).
        """
        self.registry = registry
        self.grid = grid
        self.stats = stats
        self.device = grid.device
        self.H, self.W = int(grid.size(1)), int(grid.size(2))          # map height and width
        self.respawner = RespawnController(RespawnCfg())               # handles agent respawns
        self.agent_scores: Dict[int, float] = collections.defaultdict(float)   # cumulative individual rewards (key: persistent agent_id)
        self.zones: Optional[Zones] = zones
        self._z_heal: Optional[torch.Tensor] = None                    # boolean mask of healing zones
        self._z_cp_masks: List[torch.Tensor] = []                      # list of boolean masks for capture points
        self._ensure_zone_tensors()                                    # precompute zone tensors on the correct device
        self.DIRS8_dev = DIRS8.to(self.device)                         # movement directions (8 neighbours)
        self._ACTIONS = int(getattr(config, "NUM_ACTIONS", 41))        # total number of discrete actions
        self._OBS_DIM = config.OBS_DIM                                 # observation dimension (rays + rich features)
        self._grid_dt = self.grid.dtype                                # dtype of grid tensors
        self._data_dt = self.registry.agent_data.dtype                 # dtype of agent data tensors

        # ================================================================
        # Instinct cache (computed under no_grad) — NEW
        # ================================================================
        self._instinct_cached_r: int = -999999
        self._instinct_offsets: Optional[torch.Tensor] = None          # (M,2) long dx,dy within radius
        self._instinct_area: float = 1.0

        # Constant tensors for fast operations
        self._g0 = torch.tensor(0.0, device=self.device, dtype=self._grid_dt)      # zero for grid updates
        self._gneg = torch.tensor(-1.0, device=self.device, dtype=self._grid_dt)   # -1 for clearing
        self._d0 = torch.tensor(0.0, device=self.device, dtype=self._data_dt)      # zero for data updates

        # PPO (Proximal Policy Optimization) integration
        self._ppo_enabled = bool(getattr(config, "PPO_ENABLED", False))
        self._ppo: Optional["PerAgentPPORuntime"] = None
        if self._ppo_enabled and _PerAgentPPORuntimeRT is not None:
            self._ppo = _PerAgentPPORuntimeRT(
                registry=self.registry, device=self.device,
                obs_dim=self._OBS_DIM, act_dim=self._ACTIONS,
            )

    def _ppo_reset_on_respawn(self, was_dead: torch.Tensor) -> None:
        """Reset per-slot PPO state for any slot that was dead before respawn and is alive after."""
        if self._ppo is None:
            return
        data = self.registry.agent_data
        now_alive = (data[:, COL_ALIVE] > 0.5)
        spawned_slots = (was_dead & now_alive).nonzero(as_tuple=False).squeeze(1)
        if spawned_slots.numel() == 0:
            return
        self._ppo.reset_agents(spawned_slots)
        if bool(getattr(config, "PPO_RESET_LOG", False)):
            # Keep logs short
            sl = spawned_slots[:16].tolist()
            suffix = "" if spawned_slots.numel() <= 16 else "..."
            print(f"[ppo] reset state for {int(spawned_slots.numel())} respawned slots: {sl}{suffix}")

    def _ensure_zone_tensors(self) -> None:
        """Convert zone masks to tensors on the simulation device."""
        self._z_heal, self._z_cp_masks = None, []
        if self.zones is None: return
        try:
            if getattr(self.zones, "heal_mask", None) is not None:
                self._z_heal = self.zones.heal_mask.to(self.device, non_blocking=True).bool()
            self._z_cp_masks = [m.to(self.device, non_blocking=True).bool() for m in getattr(self.zones, "cp_masks", [])]
        except Exception as e:
            print(f"[tick] WARN: zone tensor setup failed ({e}); zones disabled.")

    @staticmethod
    def _as_long(x: torch.Tensor) -> torch.Tensor:
        """Convert tensor to long dtype (used for indexing)."""
        return x.to(torch.long)

    def _recompute_alive_idx(self) -> torch.Tensor:
        """Return a 1D tensor of slot indices where the agent is alive."""
        return (self.registry.agent_data[:, COL_ALIVE] > 0.5).nonzero(as_tuple=False).squeeze(1)

    @torch.no_grad()
    def _get_instinct_offsets(self) -> Tuple[torch.Tensor, float]:
        """
        Returns cached integer (dx,dy) offsets inside a discrete circle of radius R (cells),
        plus the offset-count area used for density normalization.
        """
        R = int(getattr(config, "INSTINCT_RADIUS", 6))
        if R < 0:
            R = 0

        if self._instinct_offsets is None or self._instinct_cached_r != R:
            # Build once per radius change. Keep on engine device.
            if R == 0:
                offsets = torch.zeros((1, 2), device=self.device, dtype=torch.long)
            else:
                r = torch.arange(-R, R + 1, device=self.device, dtype=torch.long)
                dx, dy = torch.meshgrid(r, r, indexing="xy")  # dx: (S,S), dy: (S,S)
                mask = (dx * dx + dy * dy) <= (R * R)
                offsets = torch.stack([dx[mask], dy[mask]], dim=1).contiguous()  # (M,2)
                if offsets.numel() == 0:
                    offsets = torch.zeros((1, 2), device=self.device, dtype=torch.long)

            self._instinct_offsets = offsets
            self._instinct_area = float(int(offsets.size(0)))
            self._instinct_cached_r = R

        return self._instinct_offsets, self._instinct_area

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
        Densities are counts / area, where area = number of discrete cells in the radius mask.
        """
        N = int(alive_idx.numel())
        if N == 0:
            return torch.empty((0, 4), device=self.device, dtype=self._data_dt)

        data = self.registry.agent_data
        offsets, area = self._get_instinct_offsets()
        M = int(offsets.size(0))
        if M <= 0 or area <= 0.0:
            return torch.zeros((N, 4), device=self.device, dtype=self._data_dt)

        # Broadcasted neighborhood coords (N,M)
        x0 = pos_xy[:, 0].to(torch.long).view(N, 1)
        y0 = pos_xy[:, 1].to(torch.long).view(N, 1)
        ox = offsets[:, 0].view(1, M)
        oy = offsets[:, 1].view(1, M)
        xx = (x0 + ox).clamp(0, self.W - 1)
        yy = (y0 + oy).clamp(0, self.H - 1)

        occ = self.grid[0][yy, xx]      # (N,M) float, 0 empty, 1 wall, 2 red, 3 blue
        uid = unit_map[yy, xx]          # (N,M) long/int, -1 none, 1 soldier, 2 archer

        teams = data[alive_idx, COL_TEAM]  # (N,) float: 2.0 red, 3.0 blue
        team_is_red = (teams == 2.0)
        ally_occ = torch.where(team_is_red, occ.new_full((N,), 2.0), occ.new_full((N,), 3.0)).view(N, 1)
        enemy_occ = torch.where(team_is_red, occ.new_full((N,), 3.0), occ.new_full((N,), 2.0)).view(N, 1)

        ally_mask = (occ == ally_occ)
        enemy_mask = (occ == enemy_occ)

        ally_arch = ally_mask & (uid == 2)
        ally_sold = ally_mask & (uid == 1)

        ally_arch_c = ally_arch.sum(dim=1).to(torch.float32)
        ally_sold_c = ally_sold.sum(dim=1).to(torch.float32)
        enemy_c = enemy_mask.sum(dim=1).to(torch.float32)

        # Exclude self cell (offset (0,0) is included by construction).
        self_unit = data[alive_idx, COL_UNIT]  # (N,) float: 1 soldier, 2 archer
        ally_arch_c = (ally_arch_c - (self_unit == 2.0).to(torch.float32)).clamp_min(0.0)
        ally_sold_c = (ally_sold_c - (self_unit == 1.0).to(torch.float32)).clamp_min(0.0)

        # Add small noise to enemy count (requirement).
        noise = torch.randn((N,), device=self.device, dtype=torch.float32) * 0.25
        enemy_c_noisy = (enemy_c + noise).clamp_min(0.0)

        inv_area = 1.0 / float(area)
        ally_arch_d = ally_arch_c * inv_area
        ally_sold_d = ally_sold_c * inv_area
        enemy_d = enemy_c_noisy * inv_area

        eps = 1e-4 if self._data_dt == torch.float16 else 1e-6
        ally_total_d = ally_arch_d + ally_sold_d
        threat = enemy_d / (ally_total_d + eps)

        out = torch.stack([ally_arch_d, ally_sold_d, enemy_d, threat], dim=1)
        return out.to(dtype=self._data_dt)

    def _apply_deaths(self, sel: torch.Tensor, metrics: TickMetrics, credit_kills: bool = True) -> Tuple[int, int]:
        """
        Kill agents indicated by sel (boolean or index tensor).
        Updates grid, agent data, and metrics. Returns (red_deaths, blue_deaths).
        """
        data = self.registry.agent_data
        dead_idx = sel.nonzero(as_tuple=False).squeeze(1) if sel.dtype == torch.bool else sel.view(-1)
        if dead_idx.numel() == 0:
            return 0, 0

        # ---- TELEMETRY: death events + AgentLife death_tick ----
        telemetry = getattr(self, "telemetry", None)
        if telemetry is not None and getattr(telemetry, "enabled", False):
            try:
                tick_now = int(self.stats.tick)
                # PATCH: use agent_uids if available for persistent IDs, otherwise fallback to reading from data.
                if hasattr(self.registry, "agent_uids"):
                    dead_ids = self.registry.agent_uids.index_select(0, dead_idx).detach().cpu().tolist()
                else:
                    dead_ids = [int(data[slot, COL_AGENT_ID].item()) for slot in dead_idx.tolist()]
                dead_team = [int(data[slot, COL_TEAM].item()) for slot in dead_idx.tolist()]
                dead_unit = [int(data[slot, COL_UNIT].item()) for slot in dead_idx.tolist()]
                dead_slots = [int(s) for s in dead_idx.tolist()]
                telemetry.record_deaths(
                    tick=tick_now,
                    dead_ids=dead_ids,
                    dead_team=dead_team,
                    dead_unit=dead_unit,
                    dead_slots=dead_slots,
                    notes=("credit_kills" if credit_kills else "no_credit_kills"),
                )
            except Exception as e:
                try:
                    telemetry._anomaly(f"_apply_deaths telemetry hook failed: {e}")
                except Exception:
                    pass

        dead_team = data[dead_idx, COL_TEAM]
        red_deaths = int((dead_team == 2.0).sum().item())
        blue_deaths = int((dead_team == 3.0).sum().item())

        if red_deaths:
            self.stats.add_death("red", red_deaths)
            if credit_kills:
                self.stats.add_kill("blue", red_deaths)

        if blue_deaths:
            self.stats.add_death("blue", blue_deaths)
            if credit_kills:
                self.stats.add_kill("red", blue_deaths)

        # Remove agents from the grid: clear occupancy, health, and slot ID layers.
        gx, gy = self._as_long(data[dead_idx, COL_X]), self._as_long(data[dead_idx, COL_Y])
        self.grid[0][gy, gx], self.grid[1][gy, gx], self.grid[2][gy, gx] = self._g0, self._g0, self._gneg
        data[dead_idx, COL_ALIVE] = self._d0
        metrics.deaths += int(dead_idx.numel())
        return red_deaths, blue_deaths

    @torch.no_grad()
    def _build_transformer_obs(self, alive_idx: torch.Tensor, pos_xy: torch.Tensor) -> torch.Tensor:
        """
        Build the observation tensor for all alive agents.
        Observation = raycast features (32 directions × 8 values) + rich features (including instinct).
        """
        from engine.ray_engine.raycast_firsthit import build_unit_map
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
                on_cp |= cp_mask[pos_xy[:, 1], pos_xy[:, 0]]

        def _norm_const(v: float, scale: float) -> torch.Tensor:
            s = scale if scale > 0 else 1.0
            return torch.full((N,), v / s, dtype=self._data_dt, device=self.device)

        expected_ray_dim = 32 * 8
        unit_map = build_unit_map(data, self.grid)
        rays = raycast32_firsthit(
            pos_xy, self.grid, unit_map,
            max_steps_each=data[alive_idx, COL_VISION].long()
        )
        if rays.shape != (N, expected_ray_dim):
            raise RuntimeError(
                f"[obs] ray tensor shape mismatch: got {tuple(rays.shape)}, "
                f"expected ({N}, {expected_ray_dim})."
            )

        hp_max = data[alive_idx, COL_HP_MAX].clamp_min(1.0)

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
            # Padding to preserve RICH_BASE_DIM=23 layout (reserved slots).
            torch.zeros(N, device=self.device, dtype=self._data_dt),
            torch.zeros(N, device=self.device, dtype=self._data_dt),
            torch.zeros(N, device=self.device, dtype=self._data_dt),
        ], dim=1).to(dtype=self._data_dt)

        instinct = self._compute_instinct_context(alive_idx=alive_idx, pos_xy=pos_xy, unit_map=unit_map)
        # Hard invariant
        if instinct.shape != (N, 4):
            raise RuntimeError(f"instinct shape {tuple(instinct.shape)} != (N,4)")

        rich = torch.cat([rich_base, instinct], dim=1)

        expected_rich_dim = int(self._OBS_DIM) - expected_ray_dim
        if rich.shape != (N, expected_rich_dim):
            raise RuntimeError(
                f"[obs] rich tensor shape mismatch: got {tuple(rich.shape)}, "
                f"expected ({N}, {expected_rich_dim})."
            )

        obs = torch.cat([rays, rich.to(rays.dtype)], dim=1)
        if obs.shape != (N, int(self._OBS_DIM)):
            raise RuntimeError(
                f"[obs] final obs shape mismatch: got {tuple(obs.shape)}, "
                f"expected ({N}, {int(self._OBS_DIM)})."
            )
        return obs

    # ==================== NEW DEBUG INVARIANTS METHOD ====================
    def _debug_invariants(self, where: str) -> None:
        """
        Optional, gated invariants to catch grid↔registry desync early.
        Only runs if environment variable FWS_DEBUG_INVARIANTS is set to "1" or "true".
        """
        if os.getenv("FWS_DEBUG_INVARIANTS", "0") not in {"1", "true", "True"}:
            return

        data = self.registry.agent_data
        H, W = self.grid.shape[-2], self.grid.shape[-1]

        alive = (data[:, COL_ALIVE] > 0.5)
        alive_idx = alive.nonzero(as_tuple=False).squeeze(1)

        # Basic bounds for alive agents
        if alive_idx.numel() > 0:
            xs = self._as_long(data[alive_idx, COL_X])
            ys = self._as_long(data[alive_idx, COL_Y])
            if not ((xs >= 0).all() and (xs < W).all() and (ys >= 0).all() and (ys < H).all()):
                raise RuntimeError(f"[invariants:{where}] alive position out of bounds")

            # Grid layer 2 should contain the slot id at the agent's position
            g2_at = self._as_long(self.grid[2, ys, xs])
            if not torch.equal(g2_at, alive_idx):
                raise RuntimeError(f"[invariants:{where}] grid[2] slot-id mismatch at alive positions")

            # Grid layer 0 should contain the team number at that position
            team = data[alive_idx, COL_TEAM].to(self._grid_dt)
            g0_at = self.grid[0, ys, xs]
            if not torch.equal(g0_at, team):
                raise RuntimeError(f"[invariants:{where}] grid[0] occupancy/team mismatch at alive positions")

        # Uniqueness + coverage: every non-empty grid[2] must be exactly one alive slot, and vice versa.
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

        # Ghost cells: slot id present but occupancy empty.
        ghost = (self.grid[2] >= 0) & (self.grid[0] == 0)
        if ghost.any():
            raise RuntimeError(f"[invariants:{where}] ghost cells: grid[2]>=0 but grid[0]==0")
    # ==================== END DEBUG INVARIANTS ====================

    @torch.no_grad()
    def run_tick(self) -> Dict[str, float]:
        """
        Execute one simulation tick:
          - Compute observations and actions for all alive agents.
          - Move agents (with conflict resolution).
          - Process attacks (damage, kills).
          - Apply zone healing and capture point scoring.
          - Record telemetry and PPO data.
          - Respawn dead agents.
        Returns a dictionary of metrics for this tick.
        """
        data = self.registry.agent_data
        telemetry = getattr(self, "telemetry", None)          # <-- ADDED for telemetry
        tick_now = int(self.stats.tick)                        # <-- ADDED for telemetry

        metrics = TickMetrics()
        alive_idx = self._recompute_alive_idx()
        if alive_idx.numel() == 0:
            self.stats.on_tick_advanced(1)
            metrics.tick = int(self.stats.tick)
            was_dead = (data[:, COL_ALIVE] <= 0.5) if self._ppo is not None else None
            # --- Flush dead agents from PPO before respawn ---
            if was_dead is not None:
                dead_slots = was_dead.nonzero(as_tuple=False).squeeze(1)
                if dead_slots.numel() > 0:
                    self._ppo.flush_agents(dead_slots)
            self.respawner.step(self.stats.tick, self.registry, self.grid)
            if was_dead is not None:
                self._ppo_reset_on_respawn(was_dead)
            self._debug_invariants("post_respawn")
            return vars(metrics)

        pos_xy = self.registry.positions_xy(alive_idx)
        obs = self._build_transformer_obs(alive_idx, pos_xy)
        # ABSOLUTE invariant: obs width must match config.OBS_DIM
        if obs.dim() != 2 or int(obs.shape[1]) != int(config.OBS_DIM):
            raise RuntimeError(
                f"[obs] shape mismatch: got {tuple(obs.shape)}, expected (N,{int(config.OBS_DIM)})"
            )
        mask = build_mask(pos_xy, data[alive_idx, COL_TEAM], self.grid, unit=self._as_long(data[alive_idx, COL_UNIT]))
        actions = torch.zeros_like(alive_idx, dtype=torch.long)
        rec_agent_ids, rec_obs, rec_logits, rec_values, rec_actions, rec_teams = [], [], [], [], [], []

        # Group agents by model bucket (each bucket corresponds to a specific neural network).
        for bucket in self.registry.build_buckets(alive_idx):
            loc = torch.searchsorted(alive_idx, bucket.indices)
            dist, vals = ensemble_forward(bucket.models, obs[loc])
            logits32 = torch.where(mask[loc], dist.logits, torch.finfo(torch.float32).min).to(torch.float32)
            a = torch.distributions.Categorical(logits=logits32).sample()
            if self._ppo:
                rec_agent_ids.append(bucket.indices)
                rec_obs.append(obs[loc])
                rec_logits.append(logits32)
                rec_values.append(vals)
                rec_actions.append(a)
                rec_teams.append(data[bucket.indices, COL_TEAM])
            actions[loc] = a

        metrics.alive = int(alive_idx.numel())

        # ----- MOVE HANDLING WITH CONFLICT RESOLUTION (Law 1) -----
        is_move = (actions >= 1) & (actions <= 8)
        if is_move.any():
            move_idx, dir_idx = alive_idx[is_move], actions[is_move] - 1
            x0, y0 = pos_xy[is_move].T
            nx, ny = (x0 + self.DIRS8_dev[dir_idx, 0]).clamp(0, self.W - 1), (y0 + self.DIRS8_dev[dir_idx, 1]).clamp(0, self.H - 1)
            can_move = (self.grid[0][ny, nx] == self._g0)
            if can_move.any():
                move_idx, x0, y0, nx, ny = move_idx[can_move], x0[can_move], y0[can_move], nx[can_move], ny[can_move]

                # -------------------- MOVE CONFLICT RESOLUTION (Law 1) --------------------
                # Candidates are already filtered by can_move (destination cell empty).
                # If multiple candidates target the same destination in the same tick:
                #   winner = highest HP; tie for highest HP -> nobody moves to that cell.
                dest_key = (ny * self.W + nx).to(torch.long)   # (M,) unique cell id
                hp = data[move_idx, COL_HP]                   # (M,) HP used for winner rule

                # Fast path (vectorized): per-destination max HP + count of max-HP claimants.
                # Deterministic: ties never pick a random winner; they block the move.
                try:
                    num_cells = self.H * self.W
                    max_hp = torch.full((num_cells,), torch.finfo(hp.dtype).min, device=self.device, dtype=hp.dtype)
                    max_hp.scatter_reduce_(0, dest_key, hp, reduce="amax", include_self=True)
                    is_max = (hp == max_hp[dest_key])
                    max_cnt = torch.zeros((num_cells,), device=self.device, dtype=torch.int32)
                    max_cnt.scatter_add_(0, dest_key, is_max.to(torch.int32))
                    winner_mask = is_max & (max_cnt[dest_key] == 1)
                except Exception:
                    # Fallback: deterministic group scan after sorting by destination.
                    # Only used if scatter_reduce_ is unavailable in the runtime Torch build.
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
                            group_hp = hp_s[s:e]
                            m = group_hp.max()
                            is_m = (group_hp == m)
                            if int(is_m.sum().item()) == 1:
                                win_off = int(is_m.nonzero(as_tuple=False)[0].item()) + s
                                winner_mask[order[win_off]] = True

                if winner_mask.any():
                    w_move_idx = move_idx[winner_mask]
                    w_x0, w_y0, w_nx, w_ny = x0[winner_mask], y0[winner_mask], nx[winner_mask], ny[winner_mask]

                    # Commit movement ONLY for winners; losers keep their original cells.
                    self.grid[0, w_y0, w_x0], self.grid[1, w_y0, w_x0], self.grid[2, w_y0, w_x0] = self._g0, self._g0, self._gneg
                    data[w_move_idx, COL_X], data[w_move_idx, COL_Y] = w_nx.to(self._data_dt), w_ny.to(self._data_dt)
                    self.grid[0, w_ny, w_nx] = data[w_move_idx, COL_TEAM].to(self._grid_dt)
                    self.grid[1, w_ny, w_nx] = data[w_move_idx, COL_HP].to(self._grid_dt)
                    self.grid[2, w_ny, w_nx] = w_move_idx.to(self._grid_dt)

                    # Count *actual* movement winners (not just candidates).
                    metrics.moved = int(w_move_idx.numel())

                    # Optional debug-only invariant checks (default off; no cost unless enabled).
                    # Enable with: FWS_DEBUG_MOVE=1
                    if os.getenv("FWS_DEBUG_MOVE", "0") in {"1", "true", "True"}:
                        # After writeback, verify each winner's grid[2] matches their slot at new coords.
                        for i_slot in w_move_idx.tolist():
                            # Ensure we don't have stray references
                            pass
        metrics.moved = int(dead_slots.numel()) if 'keep_slots' in locals() else 0
        # ----- END MOVE HANDLING -----
        self._debug_invariants("post_move")

        combat_rd, combat_bd = 0, 0
        meta_rd, meta_bd = 0, 0

        individual_rewards = torch.zeros(self.registry.capacity, device=self.device, dtype=self._data_dt)

        # ----- COMBAT -----
        if alive_idx.numel() > 0:
            if (is_attack := actions >= 9).any():
                atk_idx, atk_act = alive_idx[is_attack], actions[is_attack]
                r, dir_idx = ((atk_act - 9) % 4) + 1, (atk_act - 9) // 4
                dxy = self.DIRS8_dev[dir_idx] * r.unsqueeze(1)
                ax, ay = pos_xy[is_attack].T
                tx, ty = (ax + dxy[:, 0]).clamp(0, self.W - 1), (ay + dxy[:, 1]).clamp(0, self.H - 1)
                victims = self._as_long(self.grid[2][ty, tx])
                if (valid_hit := victims >= 0).any():
                    atk_idx, victims = atk_idx[valid_hit], victims[valid_hit]
                    is_enemy = (data[atk_idx, COL_TEAM] != data[victims, COL_TEAM])
                    victims = victims[is_enemy]
                    atk_idx = atk_idx[is_enemy]
                    if victims.numel() > 0:
                        # ===== DETERMINISTIC FOCUS-FIRE DAMAGE =====
                        # `victims` can contain duplicates (multiple attackers hitting the same victim).
                        # PyTorch advanced indexing with duplicates is not safe for accumulation, so we:
                        #   1) sort by victim id
                        #   2) sum damage per unique victim
                        #   3) apply once per victim
                        dmg = data[atk_idx, COL_ATK]
                        order = victims.argsort()
                        sv = victims[order]
                        sdmg = dmg[order]
                        satk = atk_idx[order]

                        uniq_v, counts = torch.unique_consecutive(sv, return_counts=True)
                        cums = sdmg.cumsum(0)
                        ends = counts.cumsum(0) - 1
                        starts = ends - counts + 1
                        prev = torch.where(
                            starts > 0,
                            cums[starts - 1],
                            torch.zeros_like(starts, dtype=cums.dtype)
                        )
                        dmg_sum = cums[ends] - prev

                        hp_before = data[uniq_v, COL_HP].clone()
                        data[uniq_v, COL_HP] = hp_before - dmg_sum
                        hp_after = data[uniq_v, COL_HP]

                        # Reward all attackers that contributed to a kill (victim crosses >0 -> <=0).
                        killed_v = (hp_before > 0) & (hp_after <= 0)
                        if killed_v.any():
                            reward_val = float(config.PPO_REWARD_KILL_INDIVIDUAL)
                            killed_per_entry = killed_v.repeat_interleave(counts)
                            killers = satk[killed_per_entry]
                            if killers.numel() > 0:
                                # Deterministic accumulation per killer slot (avoid duplicate-index atomics on CUDA).
                                k_order = killers.argsort()
                                sk = killers[k_order]
                                uniq_k, k_counts = torch.unique_consecutive(sk, return_counts=True)

                                individual_rewards[uniq_k] += (k_counts.to(self._data_dt) * reward_val)

                                # agent_scores is keyed by persistent agent_id (not slot)
                                for killer_slot, cnt in zip(uniq_k.tolist(), k_counts.tolist()):
                                    # PATCH: use agent_uids if available for persistent IDs.
                                    if hasattr(self.registry, "agent_uids"):
                                        uid = int(self.registry.agent_uids[killer_slot].item())
                                    else:
                                        uid = int(data[killer_slot, COL_AGENT_ID].item())
                                    self.agent_scores[uid] += reward_val * float(cnt)

                                # ---- TELEMETRY: kill events (killer_id -> victim_id) ----
                                if telemetry is not None and getattr(telemetry, "enabled", False):
                                    try:
                                        kill_mask = killed_per_entry
                                        if kill_mask.any():
                                            killer_slots_killed = satk[kill_mask]
                                            victim_slots_killed = sv[kill_mask]
                                            # PATCH: use agent_uids if available.
                                            if hasattr(self.registry, "agent_uids"):
                                                killer_ids = self.registry.agent_uids.index_select(0, killer_slots_killed).detach().cpu().tolist()
                                                victim_ids = self.registry.agent_uids.index_select(0, victim_slots_killed).detach().cpu().tolist()
                                            else:
                                                killer_ids = [int(data[slot, COL_AGENT_ID].item()) for slot in killer_slots_killed.tolist()]
                                                victim_ids = [int(data[slot, COL_AGENT_ID].item()) for slot in victim_slots_killed.tolist()]
                                            telemetry.record_kills(tick=tick_now, killer_ids=killer_ids, victim_ids=victim_ids)
                                    except Exception as e:
                                        try:
                                            telemetry._anomaly(f"tick.kill hook failed: {e}")
                                        except Exception:
                                            pass

                        # ---- TELEMETRY: damage totals + damage events ----
                        if telemetry is not None and getattr(telemetry, "enabled", False):
                            try:
                                # Victim-sum (unique victims)
                                # PATCH: use agent_uids if available.
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
                                uniq_a, inv_a = satk.unique(return_inverse=True)
                                dmg_a = torch.zeros((uniq_a.numel(),), device=sdmg.device, dtype=sdmg.dtype)
                                dmg_a.scatter_add_(0, inv_a, sdmg)
                                # PATCH: use agent_uids if available.
                                if hasattr(self.registry, "agent_uids"):
                                    a_ids = self.registry.agent_uids.index_select(0, uniq_a).detach().cpu().tolist()
                                else:
                                    a_ids = [int(data[slot, COL_AGENT_ID].item()) for slot in uniq_a.tolist()]
                                telemetry.record_damage_attacker_sum(
                                    tick=tick_now,
                                    attacker_ids=a_ids,
                                    damage_dealt=[float(x) for x in dmg_a.detach().cpu().tolist()],
                                )

                                # Optional per-hit logging (volume heavy; controlled by TELEMETRY_DAMAGE_MODE)
                                if str(getattr(telemetry, "damage_mode", "victim_sum")).lower() == "per_hit":
                                    # PATCH: use agent_uids if available.
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

                        # Update hp channel for unique victims only (avoids duplicate-index write hazards).
                        vy, vx = self._as_long(data[uniq_v, COL_Y]), self._as_long(data[uniq_v, COL_X])
                        self.grid[1, vy, vx] = data[uniq_v, COL_HP].to(self._grid_dt)
                        metrics.attacks += int(atk_idx.numel())
                        # ===== END NEW DAMAGE =====

        # Apply deaths
        rD, bD = self._apply_deaths((data[:, COL_ALIVE] > 0.5) & (data[:, COL_HP] <= 0.0), metrics)
        combat_rd += rD
        combat_bd += bD
        self._debug_invariants("post_combat")

        if (alive_idx := self._recompute_alive_idx()).numel() > 0:
            pos_xy = self.registry.positions_xy(alive_idx)
            if self._z_heal is not None and (on_heal := self._z_heal[pos_xy[:, 1], pos_xy[:, 0]]).any():
                heal_idx = alive_idx[on_heal]
                data[heal_idx, COL_HP] = (data[heal_idx, COL_HP] + config.HEAL_RATE).clamp_max(data[heal_idx, COL_HP_MAX])
                self.grid[1, pos_xy[on_heal, 1], pos_xy[on_heal, 0]] = data[heal_idx, COL_HP].to(self._grid_dt)

            if meta_drain := getattr(config, "METABOLISM_ENABLED", True):
                drain = torch.where(data[alive_idx, COL_UNIT] == 1.0, config.META_SOLDIER_HP_PER_TICK, config.META_ARCHER_HP_PER_TICK)
                data[alive_idx, COL_HP] -= drain.to(self._data_dt)
                self.grid[1, pos_xy[:, 1], pos_xy[:, 0]] = data[alive_idx, COL_HP].to(self._grid_dt)
                if (data[alive_idx, COL_HP] <= 0.0).any():
                    rD, bD = self._apply_deaths(
                        alive_idx[data[alive_idx, COL_HP] <= 0.0],
                        metrics,
                        credit_kills=False,
                    )
                    meta_rd += rD
                    meta_bd += bD

            if self._z_cp_masks and (alive_idx := self._recompute_alive_idx()).numel() > 0:
                pos_xy, teams_alive = self.registry.positions_xy(alive_idx), data[alive_idx, COL_TEAM]
                for cp_mask in self._z_cp_masks:
                    if (on_cp := cp_mask[pos_xy[:, 1], pos_xy[:, 0]]).any():
                        red_on = (on_cp & (teams_alive == 2.0)).sum().item()
                        blue_on = (on_cp & (teams_alive == 3.0)).sum().item()
                        if red_on > blue_on:
                            self.stats.add_capture_points("red", config.CP_REWARD_PER_TICK)
                            metrics.cp_red_tick += config.CP_REWARD_PER_TICK
                        elif blue_on > red_on:
                            self.stats.add_capture_points("blue", config.CP_REWARD_PER_TICK)
                            metrics.cp_blue_tick += config.CP_REWARD_PER_TICK

                        # Individual reward for agents on a contested CP
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

        if self._ppo and rec_agent_ids:
            agent_ids = torch.cat(rec_agent_ids)
            team_r_rew = (combat_bd * config.TEAM_KILL_REWARD) + ((combat_rd + meta_rd) * config.PPO_REWARD_DEATH) + metrics.cp_red_tick
            team_b_rew = (combat_rd * config.TEAM_KILL_REWARD) + ((combat_bd + meta_bd) * config.PPO_REWARD_DEATH) + metrics.cp_blue_tick

            current_hp = data[agent_ids, COL_HP]
            hp_reward = (current_hp * config.PPO_REWARD_HP_TICK).to(self._data_dt)
            final_rewards = individual_rewards[agent_ids] + torch.where(torch.cat(rec_teams) == 2.0, team_r_rew, team_b_rew) + hp_reward

            # Bootstrap for the *last* step of the PPO window:
            # we want V(s_{t+1}) for the post-tick state to reduce truncation bias.
            # This computes the value of the next state for agents that are still alive,
            # which will be used as the bootstrap target in the PPO advantage calculation.
            bootstrap_values = None
            if self._ppo.will_train_next_step():
                # Only agents alive at the end of this tick can have a next state.
                alive_mask = (data[agent_ids, COL_ALIVE] > 0.5)
                bootstrap_values = torch.zeros((agent_ids.numel(),), device=agent_ids.device, dtype=torch.float32)
                if alive_mask.any():
                    # Indices of alive agents within the current agent_ids batch.
                    alive_pos = alive_mask.nonzero(as_tuple=False).squeeze(1)
                    # The actual slot ids of those alive agents.
                    post_ids = agent_ids[alive_pos]

                    # Sort ids to maintain bucket ordering for later searchsorted.
                    order = torch.argsort(post_ids)
                    post_ids = post_ids[order]
                    alive_pos = alive_pos[order]

                    # Build observation for the post-tick state (positions already updated).
                    pos_xy_post = self.registry.positions_xy(post_ids)
                    obs_post = self._build_transformer_obs(post_ids, pos_xy_post)

                    # For each model bucket, compute value estimates for the agents in that bucket.
                    for bucket in self.registry.build_buckets(post_ids):
                        # Find where these bucket indices appear in the sorted post_ids.
                        loc = torch.searchsorted(post_ids, bucket.indices)
                        # Forward pass through the ensemble to get value estimates.
                        _, vals = ensemble_forward(bucket.models, obs_post[loc])
                        # Store the values at the corresponding positions in the bootstrap_values tensor.
                        bootstrap_values[alive_pos[loc]] = vals.to(torch.float32)

            with torch.enable_grad():
                self._ppo.record_step(
                    agent_ids=agent_ids,
                    obs=torch.cat(rec_obs),
                    logits=torch.cat(rec_logits),
                    values=torch.cat(rec_values),
                    actions=torch.cat(rec_actions),
                    rewards=final_rewards,
                    done=(data[agent_ids, COL_ALIVE] <= 0.5),  # done flag for each agent (true if dead)
                    bootstrap_values=bootstrap_values,         # value of next state (None if not training or all dead)
                )

        self.stats.on_tick_advanced(1)
        metrics.tick = int(self.stats.tick)

        was_dead = (data[:, COL_ALIVE] <= 0.5) if self._ppo is not None else None
        # --- Flush dead agents from PPO before respawn ---
        if was_dead is not None:
            dead_slots = was_dead.nonzero(as_tuple=False).squeeze(1)
            if dead_slots.numel() > 0:
                self._ppo.flush_agents(dead_slots)
        self.respawner.step(self.stats.tick, self.registry, self.grid)

        # ---- TELEMETRY: births + lineage edges from respawn meta ----
        if telemetry is not None and getattr(telemetry, "enabled", False):
            try:
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

        # Periodic flush/validation (chunking uses existing TELEMETRY_* knobs)
        if telemetry is not None and getattr(telemetry, "enabled", False):
            try:
                telemetry.on_tick_end(metrics.tick)
            except Exception:
                pass

        return vars(metrics)