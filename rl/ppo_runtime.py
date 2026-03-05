# =============================================================================
# Per-Agent PPO Runtime (No Parameter Sharing, Slot-Local Optimizers)
# =============================================================================
#
# This module implements a **minimal** Proximal Policy Optimization (PPO)
# training runtime that is designed for a **multi-agent grid simulation**.
#
# Key design choice:
#   - **Per-slot independence** ("no hive mind"):
#       Each agent slot has its own model parameters AND its own optimizer.
#       There is no parameter sharing between slots.
#
# Here, each slot learns separately, which increases compute cost but allows
# divergent behaviors and avoids “global homogenization”.
#
# What this runtime does:
#   1) Collects per-agent trajectories (observations, actions, rewards, etc.)
#   2) Every PPO window (T steps), trains each agent independently
#   3) Supports manual flushing for dead agents before respawn
#   4) Supports checkpoint save/load of PPO runtime state
#
# -----------------------------------------------------------------------------
# PPO THEORY SUMMARY (FORMAL)
# -----------------------------------------------------------------------------
# PPO is a policy-gradient method that optimizes:
#
#   L(θ) = E[ min(r_t(θ) * A_t, clip(r_t(θ), 1-ε, 1+ε) * A_t) ]
#
# where:
#   - θ are policy parameters
#   - r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)  (probability ratio)
#   - A_t is the advantage estimate (how good action a_t was vs baseline)
#   - ε is the clipping parameter (self.clip)
#
# PPO also trains a critic (value function) by minimizing:
#   (V(s_t) - R_t)^2
# and often includes:
#   - entropy bonus (encourages exploration)
#
# This implementation includes:
#   - clipped policy objective
#   - clipped value loss
#   - entropy bonus
#   - minibatches
#   - gradient clipping
#   - optional KL-based early stopping
#   - GAE for advantage estimation
#
# =============================================================================


from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional
import math  # (imported; may be unused in this snippet but kept as-is)

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from .. import config
from ..engine.agent_registry import AgentsRegistry


# ----------------------------------------------------------------------
# Simple buffer for one agent's trajectory segment
# ----------------------------------------------------------------------
@dataclass
class _Buf:
    """
    Holds rollout data for a single agent between training windows.

    Each field is a list of tensors for each timestep in the window.

    The 'bootstrap' field stores the value estimate for the state *after* the
    last recorded transition in a window: V(s_{t+1}).

    Why bootstrap exists:
    - GAE (Generalized Advantage Estimation) needs a "next value" term.
    - At the end of a rollout window, you may not have stored the next state's
      value in the buffer.
    - If you know V(s_{T}) for the state after the last action, you can compute
      the final advantage correctly without assuming 0.
    """
    obs: List[torch.Tensor]          # observations (obs_dim,)
    act: List[torch.Tensor]          # actions taken (scalar)
    logp: List[torch.Tensor]         # log-probabilities of those actions (scalar, fp32)
    val: List[torch.Tensor]          # value estimates from the critic (scalar, fp32)
    rew: List[torch.Tensor]          # rewards received (scalar, fp32)
    done: List[torch.Tensor]         # done flags (episode end) (bool)
    act_mask: List[torch.Tensor]     # legal-action mask used during rollout sampling (bool, shape=(act_dim,))
    bootstrap: Optional[torch.Tensor] = None  # V(s_{t+1}) for the last step of a rollout window


# ----------------------------------------------------------------------
# Main PPO runtime – one instance per simulation (shared by all agents)
# ----------------------------------------------------------------------
class PerAgentPPORuntime:
    """
    Minimal per-agent PPO runtime.

    Collects trajectories for every agent, trains them independently
    (no parameter sharing between slots), and supports immediate
    flushing of dead agents before respawn.

    Terminology:
    - "agent id" here refers to the *slot index* in the simulation registry.
      This is important:
        - A slot may contain different "individuals" across time due to respawn.
        - Therefore, reset_agent/reset_agents exists to prevent inheriting
          stale optimizer/buffer state after respawn.
    """

    def __init__(self, registry: AgentsRegistry, device: torch.device,
                 obs_dim: int, act_dim: int):
        """
        Args:
            registry:
                Holds all agent brains (models). Expected usage:
                    registry.brains[aid] -> nn.Module for that slot

            device:
                Torch device where runtime tensors should live.

            obs_dim:
                Dimension of the observation vector.

            act_dim:
                Number of discrete actions.
        """
        self.registry = registry
        self.device = device
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)

        # ------------------------------------------------------------------
        # Hyperparameters (read from config; defaults are conservative)
        # ------------------------------------------------------------------
        # self.T: length of a PPO rollout window (in ticks/steps).
        # self.epochs: number of PPO epochs per window update.
        # self.lr: optimizer learning rate.
        # self.clip: PPO clipping epsilon (ε).
        # self.ent_coef: entropy coefficient (encourage exploration).
        # self.vf_coef: value loss coefficient.
        # self.gamma: discount factor γ.
        # self.lam: GAE parameter λ.
        # self.T_max / eta_min: cosine annealing LR scheduler settings.
        self.T        = int(getattr(config, "PPO_WINDOW_TICKS", 64))
        self.epochs   = int(getattr(config, "PPO_EPOCHS", 2))
        self.lr       = float(getattr(config, "PPO_LR", 3e-4))
        self.clip     = float(getattr(config, "PPO_CLIP", 0.2))
        self.ent_coef = float(getattr(config, "PPO_ENTROPY_COEF", 0.01))
        self.vf_coef  = float(getattr(config, "PPO_VALUE_COEF", 0.5))
        self.gamma    = float(getattr(config, "PPO_GAMMA", 0.99))
        self.lam      = float(getattr(config, "PPO_LAMBDA", 0.95))
        self.T_max    = int(getattr(config, "PPO_LR_T_MAX", 500_000))
        self.eta_min  = float(getattr(config, "PPO_LR_ETA_MIN", 1e-6))

        # New config parameters (introduced by a patch in your codebase):
        # - minibatches: number of minibatches per epoch
        # - max_grad_norm: gradient clipping norm (stabilizes training)
        # - target_kl: early stop if KL divergence exceeds this (trust region safety)
        self.minibatches = int(getattr(config, "PPO_MINIBATCHES", 1))
        self.max_grad_norm = float(getattr(config, "PPO_MAX_GRAD_NORM", 1.0))
        self.target_kl = float(getattr(config, "PPO_TARGET_KL", 0.0))

        # ------------------------------------------------------------------
        # Per-slot PPO state
        # ------------------------------------------------------------------
        # _buf: rollout buffer per slot (aid -> _Buf)
        # _opt: optimizer per slot (aid -> torch.optim.Optimizer)
        # _sched: LR scheduler per slot (aid -> CosineAnnealingLR)
        # _step: global decision step counter (increases each record_step call)
        self._buf: Dict[int, _Buf] = {}
        self._opt: Dict[int, optim.Optimizer] = {}
        self._sched: Dict[int, CosineAnnealingLR] = {}
        self._step = 0

        # Additive telemetry cache (read-only to external observers).
        # Updated only after successful train/update work.
        self.last_train_summary: Optional[Dict[str, float]] = None
        self._train_update_seq: int = 0

        # Dedicated rich PPO telemetry rows (append-only queue drained by TelemetrySession).
        # Kept lightweight and observational only.
        self._rich_telemetry_rows_pending: List[Dict[str, Any]] = []
        self._rich_telemetry_row_seq: int = 0

    def drain_rich_telemetry_rows(self) -> List[Dict[str, Any]]:
        """
        Return and clear pending rich PPO telemetry rows.
        TelemetrySession polls this periodically and appends to CSV.
        """
        if not self._rich_telemetry_rows_pending:
            return []
        rows = self._rich_telemetry_rows_pending
        self._rich_telemetry_rows_pending = []
        return rows

    # ------------------------------------------------------------------
    # Shape & device assertions (fail loudly on mismatch)
    # ------------------------------------------------------------------
    def _assert_record_shapes(
        self,
        agent_ids: torch.Tensor,
        obs: torch.Tensor,
        logits: torch.Tensor,
        values: torch.Tensor,
        actions: torch.Tensor,
        action_masks: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Verify that all tensors have expected dimensions and are on the correct device.
        Called inside record_step().

        Why this matters:
        - Silent shape mismatch corrupts learning (wrong alignment between agents and data).
        - Device mismatch can trigger implicit device copies or runtime crashes.
        """
        dev = self.device

        # Ensure all tensors are already on the correct device.
        if (agent_ids.device != dev or obs.device != dev or
            logits.device != dev or values.device != dev or
            actions.device != dev):
            raise RuntimeError(
                f"[ppo] device mismatch: ids={agent_ids.device} obs={obs.device} "
                f"logits={logits.device} values={values.device} "
                f"actions={actions.device} expected={dev}"
            )

        # agent_ids must be a batch vector: (B,)
        if agent_ids.dim() != 1:
            raise RuntimeError(f"[ppo] agent_ids must be (B,), got {tuple(agent_ids.shape)}")
        B = int(agent_ids.size(0))

        # obs must be (B, obs_dim)
        if obs.dim() != 2 or int(obs.size(0)) != B or int(obs.size(1)) != int(self.obs_dim):
            raise RuntimeError(f"[ppo] obs must be (B,{int(self.obs_dim)}), got {tuple(obs.shape)}")

        # logits must be (B, act_dim)
        if logits.dim() != 2 or int(logits.size(0)) != B or int(logits.size(1)) != int(self.act_dim):
            raise RuntimeError(f"[ppo] logits must be (B,{int(self.act_dim)}), got {tuple(logits.shape)}")

        # values can be (B,) or (B,1)
        if values.dim() == 2 and (int(values.size(0)) == B and int(values.size(1)) == 1):
            pass
        elif values.dim() == 1 and int(values.size(0)) == B:
            pass
        else:
            raise RuntimeError(f"[ppo] values must be (B,) or (B,1), got {tuple(values.shape)}")

        # actions must be (B,)
        if actions.dim() != 1 or int(actions.size(0)) != B:
            raise RuntimeError(f"[ppo] actions must be (B,), got {tuple(actions.shape)}")

        # action_masks (optional) must be (B, act_dim) bool-like and on same device.
        if action_masks is not None:
            if action_masks.device != dev:
                raise RuntimeError(
                    f"[ppo] action_masks device mismatch: got {action_masks.device}, expected {dev}"
                )
            if action_masks.dim() != 2 or int(action_masks.size(0)) != B or int(action_masks.size(1)) != int(self.act_dim):
                raise RuntimeError(
                    f"[ppo] action_masks must be (B,{int(self.act_dim)}), got {tuple(action_masks.shape)}"
                )

    def _assert_no_optimizer_sharing(self, aids: List[int]) -> None:
        """
        Defensive check: ensure the same optimizer object is never shared between
        two different slots.

        Why:
        - Optimizers maintain internal state (Adam moments, step count, etc.).
        - Sharing an optimizer would unintentionally couple learning between slots,
          violating the “no hive mind” design.
        """
        seen = {}  # optimizer_object_id -> aid
        for aid in aids:
            opt = self._opt.get(int(aid), None)
            if opt is None:
                continue
            key = id(opt)
            if key in seen and seen[key] != int(aid):
                raise RuntimeError(
                    f"[ppo] optimizer object shared between slots {seen[key]} and {aid} (forbidden)."
                )
            seen[key] = int(aid)

    # ------------------------------------------------------------------
    # Agent reset (called when a new agent respawns into an old slot)
    # ------------------------------------------------------------------
    def reset_agent(self, aid: int) -> None:
        """
        Hard reset PPO state for a single slot.

        This discards:
        - rollout buffer
        - optimizer
        - scheduler

        Rationale:
        - After respawn, a "new individual" appears in the same slot.
        - If we keep optimizer/buffer state, the new individual inherits old
          gradients/moments and training context, which is logically incorrect.
        """
        assert 0 <= int(aid) < int(self.registry.capacity), f"aid out of range: {aid}"

        # Order matters: scheduler references optimizer.
        self._sched.pop(int(aid), None)
        self._opt.pop(int(aid), None)
        self._buf.pop(int(aid), None)

    def reset_agents(self, aids: torch.Tensor | List[int]) -> None:
        """
        Reset many agents at once.

        Accepts:
        - LongTensor of slot indices
        - Python list of slot indices
        """
        if aids is None:
            return

        if isinstance(aids, torch.Tensor):
            if aids.numel() == 0:
                return
            lst = aids.to("cpu").tolist()
        else:
            if len(aids) == 0:
                return
            lst = list(aids)

        for a in lst:
            self.reset_agent(int(a))

    # ------------------------------------------------------------------
    # Lazy buffer / optimizer creation
    # ------------------------------------------------------------------
    def _get_buf(self, aid: int) -> _Buf:
        """
        Return rollout buffer for slot `aid`, creating it if missing.

        Buffers store *lists* because data arrives step-by-step.
        At training time we stack lists into batched tensors.
        """
        if aid not in self._buf:
            self._buf[aid] = _Buf([], [], [], [], [], [], [])
        return self._buf[aid]

    def _get_opt(self, aid: int, model: nn.Module) -> optim.Optimizer:
        """
        Return Adam optimizer for slot `aid`, creating it (and its scheduler)
        if missing.

        CosineAnnealingLR:
        - Learning rate follows cosine curve from lr -> eta_min over T_max steps.
        - This can stabilize long training runs by slowly reducing LR.
        """
        if aid not in self._opt:
            self._opt[aid] = optim.Adam(model.parameters(), lr=self.lr)
            self._sched[aid] = CosineAnnealingLR(
                self._opt[aid], T_max=self.T_max, eta_min=self.eta_min
            )
        return self._opt[aid]

    # ------------------------------------------------------------------
    # Window timing helper
    # ------------------------------------------------------------------
    def will_train_next_step(self) -> bool:
        """
        Return True if the *next* record_step call will trigger a PPO window train.

        Logic:
        - self._step counts how many record_step calls have occurred.
        - Training triggers when self._step % self.T == 0.
        - So, the next step triggers training if (self._step + 1) is a multiple of T.
        """
        return ((self._step + 1) % self.T) == 0

    # ------------------------------------------------------------------
    # Record one step (called by environment after each tick)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def record_step(
        self,
        agent_ids: torch.Tensor,
        obs: torch.Tensor,
        logits: torch.Tensor,
        values: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        done: torch.Tensor,
        action_masks: Optional[torch.Tensor] = None,
        bootstrap_values: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Append a single decision step for all agents that acted in this tick.

        Data stored per agent:
        - obs_t
        - act_t
        - logp_old_t (log probability of chosen action under behavior policy)
        - val_t (critic value estimate)
        - rew_t
        - done_t

        Training trigger:
        - When global step reaches a multiple of self.T, we train and clear.

        bootstrap_values:
        - Optional tensor V(s_{t+1}) for each agent in this batch.
        - Intended to be supplied on the **last step of a PPO window** so GAE
          can bootstrap correctly at the boundary.
        """
        # Validate shapes/devices early.
        self._assert_record_shapes(agent_ids, obs, logits, values, actions, action_masks=action_masks)

        if action_masks is None:
            # Backward-compatible fallback for callers not yet passing masks.
            action_masks = torch.ones(
                (int(actions.size(0)), int(self.act_dim)),
                device=logits.device,
                dtype=torch.bool,
            )
        else:
            action_masks = action_masks.to(dtype=torch.bool)

        if not bool(action_masks.any(dim=-1).all().item()):
            raise RuntimeError("[ppo] action_masks contains a row with no legal actions")
        ar = torch.arange(actions.numel(), device=actions.device)
        if not bool(action_masks[ar, actions.long()].all().item()):
            raise RuntimeError("[ppo] chosen action is illegal under rollout action_masks")

        # Compute log probabilities of chosen actions under the SAME masked rollout policy
        # and keep PPO bookkeeping in fp32.
        logits_masked = self._mask_logits(logits, action_masks)
        logp_a = F.log_softmax(logits_masked.to(torch.float32), dim=-1).gather(1, actions.view(-1, 1)).squeeze(1)

        # Append each agent transition into its own buffer.
        for i in range(agent_ids.numel()):
            aid = int(agent_ids[i].item())
            b = self._get_buf(aid)

            b.obs.append(obs[i])
            b.act.append(actions[i].detach())
            b.logp.append(logp_a[i].detach().to(torch.float32))

            # Ensure value is stored as a (1,) tensor for consistent concatenation later.
            b.val.append(values[i].detach().reshape(1).to(torch.float32))

            b.rew.append(rewards[i].detach().to(torch.float32))
            b.done.append(done[i].detach().to(torch.bool))
            b.act_mask.append(action_masks[i].detach().to(torch.bool))

            # Store bootstrap V(s_{t+1}) if provided.
            if bootstrap_values is not None:
                b.bootstrap = bootstrap_values[i].detach().reshape(1).to(torch.float32)

        # Advance global step.
        self._step += 1

        # Train if we reached end of window.
        if self._step % self.T == 0:
            self._train_window_and_clear()

    # ------------------------------------------------------------------
    # Public flush method – train and clear specific agents immediately
    # ------------------------------------------------------------------
    def flush_agents(self, agent_ids: Any) -> None:
        """
        Train + clear rollout buffers for given agents *immediately*.

        Motivation:
        - When an agent dies, its buffer might contain valuable terminal info.
        - If a respawn reuses the same slot before the next PPO window,
          we risk overwriting that buffer.
        - Flushing preserves the learning signal before slot reuse.
        """
        if agent_ids is None:
            return

        # Accept both tensors and Python iterables.
        if isinstance(agent_ids, torch.Tensor):
            aids = agent_ids.detach().to("cpu").to(torch.long).tolist()
        else:
            aids = [int(a) for a in agent_ids]

        if len(aids) == 0:
            return

        self._train_aids_and_clear(aids)

    # ------------------------------------------------------------------
    # Core training routine – shared by window flush and manual flush
    # ------------------------------------------------------------------
    def _train_aids_and_clear(self, aids: List[int]) -> None:
        """
        For each slot in `aids`, if it has collected data, run PPO updates
        and clear its buffer.

        This method includes:
        - GAE advantages + returns
        - minibatch PPO updates
        - clipped policy objective
        - clipped value loss
        - entropy bonus
        - gradient clipping
        - optional KL early stopping
        - cosine LR scheduling
        """
        if len(aids) == 0:
            return

        # Ensure no optimizer sharing (design invariant).
        self._assert_no_optimizer_sharing(aids)

        # Additive PPO telemetry aggregation (observational only).
        log_ppo_telem = bool(getattr(config, "TELEMETRY_LOG_PPO", False))
        rich_ppo_telem = bool(getattr(config, "TELEMETRY_PPO_RICH_CSV", False))
        rich_level_requested = str(getattr(config, "TELEMETRY_PPO_RICH_LEVEL", "update")).strip().lower()
        rich_level_effective = "update"  # minimal-disturbance implementation emits update-level rows
        collect_diag = bool(log_ppo_telem or rich_ppo_telem)

        def _agg_tensor_stats(dst: Optional[Dict[str, Any]], prefix: str, x: torch.Tensor) -> None:
            if dst is None:
                return
            try:
                t = x.detach().to(torch.float32).reshape(-1)
                if t.numel() <= 0:
                    return
                finite = torch.isfinite(t)
                if not bool(finite.all().item()):
                    t = t[finite]
                    if t.numel() <= 0:
                        return
                n = int(t.numel())
                s = float(t.sum().item())
                ss = float((t * t).sum().item())
                mn = float(t.min().item())
                mx = float(t.max().item())
            except Exception:
                return

            dst[f"{prefix}_count"] = int(dst.get(f"{prefix}_count", 0)) + n
            dst[f"{prefix}_sum"] = float(dst.get(f"{prefix}_sum", 0.0)) + s
            dst[f"{prefix}_sumsq"] = float(dst.get(f"{prefix}_sumsq", 0.0)) + ss
            dst[f"{prefix}_min"] = mn if (f"{prefix}_min" not in dst) else min(float(dst[f"{prefix}_min"]), mn)
            dst[f"{prefix}_max"] = mx if (f"{prefix}_max" not in dst) else max(float(dst[f"{prefix}_max"]), mx)

        def _agg_scalar(dst: Optional[Dict[str, Any]], key: str, value: float) -> None:
            if dst is None:
                return
            try:
                v = float(value)
            except Exception:
                return
            if not math.isfinite(v):
                return
            dst[f"{key}_sum"] = float(dst.get(f"{key}_sum", 0.0)) + v
            dst[f"{key}_count"] = int(dst.get(f"{key}_count", 0)) + 1
            dst[f"{key}_min"] = v if (f"{key}_min" not in dst) else min(float(dst[f"{key}_min"]), v)
            dst[f"{key}_max"] = v if (f"{key}_max" not in dst) else max(float(dst[f"{key}_max"]), v)

        agg = {
            "trained_slots": 0,
            "samples": 0,
            "optimizer_steps": 0,
            "epochs_run": 0,
            # Existing summary fields
            "loss_pi_sum": 0.0,
            "loss_v_sum": 0.0,
            "loss_ent_sum": 0.0,
            "entropy_sum": 0.0,
            "approx_kl_sum": 0.0,
            "approx_kl_count": 0,
            "approx_kl_max": 0.0,
            "lr_before_sum": 0.0,
            "lr_after_sum": 0.0,
            # Rich diagnostics (all additive / observational only)
            "loss_total_sum": 0.0,
            "loss_total_count": 0,
            "clip_frac_sum": 0.0,
            "clip_frac_count": 0,
            "clip_frac_max": 0.0,
            "grad_norm_sum": 0.0,
            "grad_norm_count": 0,
            "grad_norm_max": 0.0,
            "explained_var_sum": 0.0,
            "explained_var_count": 0,
        } if collect_diag else None

        for aid in aids:
            aid = int(aid)
            b = self._buf.get(aid, None)

            # Skip if empty buffer.
            if b is None or not b.obs:
                continue

            # Retrieve this slot's brain (policy/value network).
            model = self.registry.brains[aid]
            if model is None:
                # No model -> discard data (cannot train).
                self._buf.pop(aid, None)
                continue

            model.train()

            # Ensure optimizer + scheduler exist.
            opt = self._get_opt(aid, model)
            lr_before = float(opt.param_groups[0]["lr"]) if len(getattr(opt, "param_groups", [])) > 0 else float(self.lr)

            # Stack rollout lists into tensors:
            # T here is "buffer length" (could be < self.T for flush cases).
            obs = torch.stack(b.obs)                         # (T, obs_dim)
            act = torch.stack(b.act).long()                  # (T,)
            logp_old = torch.stack(b.logp).to(torch.float32) # (T,)
            val_old = torch.cat(b.val).to(torch.float32)     # (T,)
            rew = torch.stack(b.rew).to(torch.float32)       # (T,)
            done = torch.stack(b.done).bool()                # (T,)

            # Rollout action masks (new checkpoints). Older checkpoints may not have them.
            if len(getattr(b, "act_mask", [])) == len(b.obs):
                act_mask = torch.stack(b.act_mask).bool()    # (T, act_dim)
            elif len(getattr(b, "act_mask", [])) == 0:
                # Cannot recover exact rollout masks from old checkpoints (pre-patch schema).
                # Drop this buffered segment instead of training with mismatched PPO semantics.
                b.obs.clear()
                b.act.clear()
                b.logp.clear()
                b.val.clear()
                b.rew.clear()
                b.done.clear()
                if hasattr(b, "act_mask"):
                    b.act_mask.clear()
                b.bootstrap = None
                continue
            else:
                raise RuntimeError(
                    f"[ppo] act_mask buffer length mismatch for aid={aid}: "
                    f"{len(getattr(b, 'act_mask', []))} vs {len(b.obs)}"
                )

            # Compute advantages and returns using GAE.
            last_v = b.bootstrap
            adv, ret = self._gae(rew, val_old, done, last_value=last_v)
            if adv.dtype != torch.float32 or ret.dtype != torch.float32:
                raise RuntimeError(f"[ppo] GAE outputs must be float32, got adv={adv.dtype} ret={ret.dtype}")
            if not bool(torch.isfinite(logp_old).all().item()):
                raise RuntimeError("[ppo] non-finite logp_old in rollout buffer")
            if not bool(torch.isfinite(adv).all().item()) or not bool(torch.isfinite(ret).all().item()):
                raise RuntimeError("[ppo] non-finite adv/ret from GAE")

            # Minibatch preparation
            B = int(obs.size(0))

            # Number of minibatches is constrained:
            # - at least 1
            # - at most self.minibatches
            # - cannot exceed B (each minibatch must have >= 1 item)
            n_mb = max(1, min(int(self.minibatches), B))
            mb_size = max(1, B // n_mb)

            if collect_diag and agg is not None:
                _agg_tensor_stats(agg, "rew", rew)
                _agg_tensor_stats(agg, "adv_norm", adv)
                _agg_tensor_stats(agg, "ret", ret)
                _agg_tensor_stats(agg, "value_old", val_old)

            aid_epochs_run = 0
            aid_optimizer_steps = 0
            aid_loss_pi_sum = 0.0
            aid_loss_v_sum = 0.0
            aid_loss_ent_sum = 0.0
            aid_entropy_sum = 0.0
            aid_approx_kl_max = 0.0
            with torch.enable_grad():
                for _ in range(self.epochs):
                    if collect_diag:
                        aid_epochs_run += 1
                    # Shuffle indices each epoch.
                    idx = torch.randperm(B, device=obs.device)

                    # Track KL for optional early stop.
                    approx_kl_epoch = 0.0

                    # Iterate minibatches
                    for start_i in range(0, B, mb_size):
                        mb_idx = idx[start_i:start_i + mb_size]
                        if mb_idx.numel() == 0:
                            continue

                        # Gather minibatch tensors
                        obs_mb = obs[mb_idx]
                        act_mb = act[mb_idx]
                        act_mask_mb = act_mask[mb_idx]
                        logp_old_mb = logp_old[mb_idx].detach()
                        val_old_mb = val_old[mb_idx].detach()
                        adv_mb = adv[mb_idx].detach()
                        ret_mb = ret[mb_idx].detach()

                        # Forward pass on minibatch (masked policy for PPO consistency)
                        logits, values, entropy = self._policy_value(model, obs_mb, action_masks=act_mask_mb)

                        # Compute logp under current masked policy for taken actions (fp32 math)
                        logp = F.log_softmax(logits.to(torch.float32), dim=-1).gather(1, act_mb.view(-1, 1)).squeeze(1)

                        # --------------------------------------------------
                        # PPO policy loss (clipped surrogate)
                        # --------------------------------------------------
                        ratio = torch.exp(logp - logp_old_mb)
                        surr1 = ratio * adv_mb
                        surr2 = torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip) * adv_mb
                        loss_pi = -torch.min(surr1, surr2).mean()

                        # --------------------------------------------------
                        # PPO value loss (clipped)
                        # --------------------------------------------------
                        v_pred = values
                        v_clipped = val_old_mb + torch.clamp(v_pred - val_old_mb, -self.clip, self.clip)
                        loss_v1 = (v_pred - ret_mb) ** 2
                        loss_v2 = (v_clipped - ret_mb) ** 2
                        loss_v = torch.max(loss_v1, loss_v2).mean()

                        # --------------------------------------------------
                        # Entropy bonus
                        # --------------------------------------------------
                        # entropy encourages non-deterministic action selection early in training.
                        # Loss form: -entropy (because we *maximize* entropy, but optimizers minimize)
                        loss_ent = -entropy.mean()

                        # Total combined loss
                        loss = loss_pi + self.vf_coef * loss_v + self.ent_coef * loss_ent

                        # Gradient step
                        opt.zero_grad(set_to_none=True)
                        loss.backward()

                        # Gradient clipping protects against exploding gradients.
                        grad_norm_raw = torch.nn.utils.clip_grad_norm_(model.parameters(), float(self.max_grad_norm))
                        if torch.is_tensor(grad_norm_raw):
                            grad_norm_val = float(grad_norm_raw.detach().item())
                        else:
                            grad_norm_val = float(grad_norm_raw)
                        opt.step()

                        # Approx KL divergence for early stopping:
                        # KL(π_old || π_new) approx ≈ E[logp_old - logp_new]
                        approx_kl = (logp_old_mb - logp).mean().item()
                        approx_kl_epoch = max(approx_kl_epoch, approx_kl)

                        if collect_diag and agg is not None:
                            ratio_det = ratio.detach().to(torch.float32)
                            v_pred_det = v_pred.detach().to(torch.float32)
                            _agg_tensor_stats(agg, "ratio", ratio_det)
                            _agg_tensor_stats(agg, "value_pred", v_pred_det)
                            _agg_tensor_stats(agg, "mask_valid_actions", act_mask_mb.to(torch.float32).sum(dim=-1))

                            clip_frac = float(((ratio_det - 1.0).abs() > float(self.clip)).to(torch.float32).mean().item())
                            agg["clip_frac_sum"] += clip_frac
                            agg["clip_frac_count"] += 1
                            agg["clip_frac_max"] = max(float(agg["clip_frac_max"]), clip_frac)

                            agg["loss_total_sum"] += float(loss.detach().item())
                            agg["loss_total_count"] += 1
                            agg["approx_kl_sum"] += float(approx_kl)
                            agg["approx_kl_count"] += 1

                            if math.isfinite(grad_norm_val):
                                agg["grad_norm_sum"] += grad_norm_val
                                agg["grad_norm_count"] += 1
                                agg["grad_norm_max"] = max(float(agg["grad_norm_max"]), grad_norm_val)

                            # Explained variance (current minibatch, using current value predictions)
                            try:
                                ret_var_t = torch.var(ret_mb.detach().to(torch.float32), unbiased=False)
                                ret_var = float(ret_var_t.item())
                                if ret_var > 1e-12:
                                    resid_var = float(torch.var((ret_mb.detach().to(torch.float32) - v_pred_det), unbiased=False).item())
                                    ev = 1.0 - (resid_var / ret_var)
                                    if math.isfinite(ev):
                                        _agg_scalar(agg, "explained_var", ev)
                            except Exception:
                                pass

                        if collect_diag:
                            # NOTE:
                            # Rich PPO CSV (TELEMETRY_PPO_RICH_CSV) also depends on these per-step
                            # counters/sums. If we gate on `log_ppo_telem` only, then runs with:
                            #   TELEMETRY_LOG_PPO = False
                            #   TELEMETRY_PPO_RICH_CSV = True
                            # will keep aid_optimizer_steps == 0 and never emit rich rows.
                            #
                            # This remains observational-only and does not affect optimizer semantics.
                            aid_optimizer_steps += 1
                            aid_loss_pi_sum += float(loss_pi.detach().item())
                            aid_loss_v_sum += float(loss_v.detach().item())
                            aid_loss_ent_sum += float(loss_ent.detach().item())
                            aid_entropy_sum += float(entropy.mean().detach().item())
                            aid_approx_kl_max = max(float(aid_approx_kl_max), float(approx_kl))

                    # Early stopping if KL exceeds target
                    if self.target_kl > 0.0 and approx_kl_epoch > self.target_kl:
                        break

            # Learning rate schedule update
            if aid in self._sched:
                self._sched[aid].step()
            lr_after = float(opt.param_groups[0]["lr"]) if len(getattr(opt, "param_groups", [])) > 0 else lr_before

            if collect_diag and agg is not None:
                n_steps = int(aid_optimizer_steps)
                if n_steps > 0:
                    agg["trained_slots"] += 1
                    agg["samples"] += int(B)
                    agg["optimizer_steps"] += n_steps
                    agg["epochs_run"] += int(aid_epochs_run)
                    agg["loss_pi_sum"] += float(aid_loss_pi_sum)
                    agg["loss_v_sum"] += float(aid_loss_v_sum)
                    agg["loss_ent_sum"] += float(aid_loss_ent_sum)
                    agg["entropy_sum"] += float(aid_entropy_sum)
                    agg["approx_kl_max"] = max(float(agg["approx_kl_max"]), float(aid_approx_kl_max))
                    agg["lr_before_sum"] += float(lr_before)
                    agg["lr_after_sum"] += float(lr_after)

            # Clear buffer after training
            b.obs.clear()
            b.act.clear()
            b.logp.clear()
            b.val.clear()
            b.rew.clear()
            b.done.clear()
            b.act_mask.clear()
            b.bootstrap = None

        def _mean_from_agg(prefix: str) -> float:
            c = int(agg.get(f"{prefix}_count", 0)) if agg is not None else 0
            return (float(agg.get(f"{prefix}_sum", 0.0)) / float(c)) if c > 0 else float("nan")

        def _std_from_agg(prefix: str) -> float:
            c = int(agg.get(f"{prefix}_count", 0)) if agg is not None else 0
            if c <= 0:
                return float("nan")
            s = float(agg.get(f"{prefix}_sum", 0.0))
            ss = float(agg.get(f"{prefix}_sumsq", 0.0))
            m = s / float(c)
            var = max(0.0, (ss / float(c)) - (m * m))
            return math.sqrt(var)

        # Publish/update telemetry summary only after successful training work.
        if collect_diag and agg is not None and int(agg["trained_slots"]) > 0 and int(agg["optimizer_steps"]) > 0:
            self._train_update_seq += 1
            slots = max(1, int(agg["trained_slots"]))
            steps = max(1, int(agg["optimizer_steps"]))
            summary_row = {
                "update_seq": float(self._train_update_seq),
                "ppo_step": float(self._step),
                "trained_slots": float(agg["trained_slots"]),
                "samples": float(agg["samples"]),
                "optimizer_steps": float(agg["optimizer_steps"]),
                "epochs_run_total": float(agg["epochs_run"]),
                "loss_pi_mean": float(agg["loss_pi_sum"]) / float(steps),
                "loss_v_mean": float(agg["loss_v_sum"]) / float(steps),
                "loss_ent_mean": float(agg["loss_ent_sum"]) / float(steps),
                "entropy_mean": float(agg["entropy_sum"]) / float(steps),
                "loss_total_mean": (float(agg["loss_total_sum"]) / float(max(1, int(agg["loss_total_count"]))))
                    if int(agg.get("loss_total_count", 0)) > 0 else float("nan"),
                "approx_kl_mean": (float(agg["approx_kl_sum"]) / float(max(1, int(agg["approx_kl_count"]))))
                    if int(agg.get("approx_kl_count", 0)) > 0 else float("nan"),
                "approx_kl_max": float(agg["approx_kl_max"]),
                "lr_before_mean": float(agg["lr_before_sum"]) / float(slots),
                "lr_after_mean": float(agg["lr_after_sum"]) / float(slots),
                "clip_frac_mean": (float(agg["clip_frac_sum"]) / float(max(1, int(agg["clip_frac_count"]))))
                    if int(agg.get("clip_frac_count", 0)) > 0 else float("nan"),
                "clip_frac_max": float(agg.get("clip_frac_max", 0.0)),
                "ratio_mean": _mean_from_agg("ratio"),
                "ratio_std": _std_from_agg("ratio"),
                "ratio_min": float(agg.get("ratio_min", float("nan"))),
                "ratio_max": float(agg.get("ratio_max", float("nan"))),
                "adv_norm_mean": _mean_from_agg("adv_norm"),
                "adv_norm_std": _std_from_agg("adv_norm"),
                "adv_norm_min": float(agg.get("adv_norm_min", float("nan"))),
                "adv_norm_max": float(agg.get("adv_norm_max", float("nan"))),
                "ret_mean": _mean_from_agg("ret"),
                "ret_std": _std_from_agg("ret"),
                "ret_min": float(agg.get("ret_min", float("nan"))),
                "ret_max": float(agg.get("ret_max", float("nan"))),
                "rew_mean": _mean_from_agg("rew"),
                "rew_std": _std_from_agg("rew"),
                "rew_min": float(agg.get("rew_min", float("nan"))),
                "rew_max": float(agg.get("rew_max", float("nan"))),
                "value_old_mean": _mean_from_agg("value_old"),
                "value_old_std": _std_from_agg("value_old"),
                "value_old_min": float(agg.get("value_old_min", float("nan"))),
                "value_old_max": float(agg.get("value_old_max", float("nan"))),
                "value_pred_mean": _mean_from_agg("value_pred"),
                "value_pred_std": _std_from_agg("value_pred"),
                "value_pred_min": float(agg.get("value_pred_min", float("nan"))),
                "value_pred_max": float(agg.get("value_pred_max", float("nan"))),
                "grad_norm_mean": (float(agg["grad_norm_sum"]) / float(max(1, int(agg["grad_norm_count"]))))
                    if int(agg.get("grad_norm_count", 0)) > 0 else float("nan"),
                "grad_norm_max": float(agg.get("grad_norm_max", float("nan"))),
                "explained_var_mean": _mean_from_agg("explained_var"),
                "explained_var_min": float(agg.get("explained_var_min", float("nan"))),
                "explained_var_max": float(agg.get("explained_var_max", float("nan"))),
                "mask_valid_actions_mean": _mean_from_agg("mask_valid_actions"),
                "mask_valid_actions_min": float(agg.get("mask_valid_actions_min", float("nan"))),
                "mask_valid_actions_max": float(agg.get("mask_valid_actions_max", float("nan"))),
                "clip_epsilon": float(self.clip),
                "target_kl": float(self.target_kl),
                "ppo_epochs_cfg": float(self.epochs),
                "ppo_minibatches_cfg": float(self.minibatches),
                "max_grad_norm_cfg": float(self.max_grad_norm),
            }
            # Preserve existing lightweight cache used by telemetry_summary.csv.
            self.last_train_summary = dict(summary_row)

            # Dedicated rich CSV row (separate file; append-safe and drained by TelemetrySession).
            if rich_ppo_telem:
                self._rich_telemetry_row_seq += 1
                note = ""
                if rich_level_requested != "update":
                    note = f"requested_granularity={rich_level_requested}; emitted=update (minimal patch path)"
                self._rich_telemetry_rows_pending.append({
                    "row_seq": float(self._rich_telemetry_row_seq),
                    "granularity_requested": rich_level_requested,
                    "granularity_effective": rich_level_effective,
                    **summary_row,
                    "notes": note,
                })

    def _train_window_and_clear(self) -> None:
        """
        Called when global step hits multiple of self.T.
        Trains on all agents that currently have buffered data.
        """
        if len(self._buf) == 0:
            return
        self._train_aids_and_clear(list(self._buf.keys()))

    # ------------------------------------------------------------------
    # Checkpointing – save/load all runtime state
    # ------------------------------------------------------------------
    def get_checkpoint_state(self) -> Dict[str, Any]:
        """
        Return a portable checkpoint payload.

        Contains:
          - rollout buffers per agent id
          - optimizer state_dict per agent id
          - scheduler state_dict per agent id
          - global step counter

        Portability:
        - All tensors are moved to CPU to allow loading on different machines/devices.
        """
        def cpuize(x: Any) -> Any:
            """
            Recursively move tensors to CPU for portability.
            """
            if torch.is_tensor(x):
                return x.detach().to("cpu")
            if isinstance(x, dict):
                return {k: cpuize(v) for k, v in x.items()}
            if isinstance(x, list):
                return [cpuize(v) for v in x]
            return x

        buf_out: Dict[int, Any] = {}
        for aid, b in self._buf.items():
            buf_out[int(aid)] = {
                "obs":  cpuize(list(b.obs)),
                "act":  cpuize(list(b.act)),
                "logp": cpuize(list(b.logp)),
                "val":  cpuize(list(b.val)),
                "rew":  cpuize(list(b.rew)),
                "done": cpuize(list(b.done)),
                "act_mask": cpuize(list(b.act_mask)),
                # bootstrap not saved (ephemeral / window-boundary specific)
            }

        opt_out = {int(aid): cpuize(opt.state_dict()) for aid, opt in self._opt.items()}
        sched_out = {int(aid): cpuize(s.state_dict()) for aid, s in self._sched.items()}

        return {
            "step":  int(self._step),
            "train_update_seq": int(self._train_update_seq),
            "rich_telemetry_row_seq": int(self._rich_telemetry_row_seq),
            "buf":   buf_out,
            "opt":   opt_out,
            "sched": sched_out,
        }

    def load_checkpoint_state(
        self,
        state: Dict[str, Any],
        *,
        registry: Any,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Restore PPO runtime from a checkpoint produced by get_checkpoint_state().

        Args:
            state:
                Checkpoint dictionary.

            registry:
                Agent registry (needed to access models at registry.brains[aid]).

            device:
                Target device. If None, uses self.device.
        """
        dev = device or self.device

        def to_dev(x: Any) -> Any:
            """
            Recursively move tensors to the target device.
            """
            if torch.is_tensor(x):
                return x.to(dev)
            if isinstance(x, list):
                return [to_dev(v) for v in x]
            return x

        # Restore global step
        self._step = int(state.get("step", 0))
        self._train_update_seq = int(state.get("train_update_seq", 0))
        self._rich_telemetry_row_seq = int(state.get("rich_telemetry_row_seq", 0))

        # Restore buffers
        self._buf.clear()
        buf_in: Dict[int, Any] = state.get("buf", {})
        for aid, payload in buf_in.items():
            aid_i = int(aid)
            self._buf[aid_i] = _Buf(
                obs=to_dev(payload.get("obs", [])),
                act=to_dev(payload.get("act", [])),
                logp=to_dev(payload.get("logp", [])),
                val=to_dev(payload.get("val", [])),
                rew=to_dev(payload.get("rew", [])),
                done=to_dev(payload.get("done", [])),
                act_mask=to_dev(payload.get("act_mask", [])),
            )

        # Restore optimizers and schedulers
        opt_in: Dict[int, Any] = state.get("opt", {})
        sched_in: Dict[int, Any] = state.get("sched", {})

        self._opt.clear()
        self._sched.clear()

        # Recreate optimizers first
        for aid, opt_sd in opt_in.items():
            aid_i = int(aid)
            model = registry.brains[aid_i]
            if model is None:
                continue

            opt = self._get_opt(aid_i, model)
            opt.load_state_dict(opt_sd)

            # Move tensors inside optimizer state to target device
            for state_group in opt.state.values():
                for k, v in list(state_group.items()):
                    if torch.is_tensor(v):
                        state_group[k] = v.to(dev)

        # Restore schedulers next
        for aid, sch_sd in sched_in.items():
            aid_i = int(aid)
            if aid_i not in self._sched:
                continue
            sch = self._sched[aid_i]
            sch.load_state_dict(sch_sd)

    # ------------------------------------------------------------------
    # Helper methods for PPO computations
    # ------------------------------------------------------------------
    def _mask_logits(self, logits: torch.Tensor, action_masks: torch.Tensor) -> torch.Tensor:
        """
        Apply rollout legality mask to policy logits.
        action_masks=True means legal action.
        """
        if logits.dim() != 2:
            raise RuntimeError(f"[ppo] logits must be (B,A) for masking, got {tuple(logits.shape)}")
        if action_masks.dim() != 2 or tuple(action_masks.shape) != tuple(logits.shape):
            raise RuntimeError(
                f"[ppo] action_masks shape mismatch: logits={tuple(logits.shape)} masks={tuple(action_masks.shape)}"
            )
        masks_bool = action_masks.to(device=logits.device, dtype=torch.bool)
        if not bool(masks_bool.any(dim=-1).all().item()):
            raise RuntimeError("[ppo] action mask row has no legal actions")
        neg = torch.tensor(torch.finfo(logits.dtype).min, device=logits.device, dtype=logits.dtype)
        return torch.where(masks_bool, logits, neg)

    def _gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        last_value: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generalized Advantage Estimation (GAE).

        Inputs:
            rewards: (T,) reward sequence r_t
            values:  (T,) critic values V(s_t)
            dones:   (T,) boolean episode-end flags
            last_value:
                Optional bootstrap value V(s_{T}) for state after last step.

        Outputs:
            adv: (T,) advantage estimates A_t
            ret: (T,) return targets R_t = A_t + V(s_t)

        ------------------------------------------------------------------
        MATHEMATICS (FORMAL)
        ------------------------------------------------------------------
        Temporal-difference residual (TD error):
            δ_t = r_t + γ * V(s_{t+1}) * (1 - done_t) - V(s_t)

        GAE recursion:
            A_t = δ_t + γ * λ * (1 - done_t) * A_{t+1}

        Returns:
            R_t = A_t + V(s_t)

        Notes:
        - If done_t == True, the episode terminated at time t, so future values
          do not contribute. The factor (1 - done_t) zeros them out.
        - At the final time step, V(s_{t+1}) is unknown unless bootstrapped.
        """
        # PPO bookkeeping/GAE math is intentionally forced to fp32 even when
        # model forward/backward may use AMP half precision.
        rewards32 = rewards.detach().to(torch.float32)
        values32 = values.detach().to(torch.float32)
        dones_b = dones.detach().to(torch.bool)

        T = rewards32.numel()
        adv = torch.zeros((T,), device=values32.device, dtype=torch.float32)
        last_gae = torch.zeros((), device=values32.device, dtype=torch.float32)

        if last_value is not None:
            last_value_t = last_value.detach().to(device=values32.device, dtype=torch.float32).reshape(())
        else:
            last_value_t = torch.zeros((), device=values32.device, dtype=torch.float32)

        for t in reversed(range(T)):
            # mask = 0 if done, else 1 (kept as tensor to avoid GPU->CPU sync)
            mask = (~dones_b[t]).to(dtype=torch.float32)

            # Choose next value:
            # - interior steps: values[t+1]
            # - final step: bootstrap value (if provided), else 0
            if t < T - 1:
                next_val_t = values32[t + 1]
            else:
                next_val_t = last_value_t

            delta = rewards32[t] + self.gamma * next_val_t * mask - values32[t]
            last_gae = delta + self.gamma * self.lam * mask * last_gae
            adv[t] = last_gae

        # Return targets:
        ret = adv + values32

        # Advantage normalization stabilizes training:
        # It rescales advantages to roughly zero-mean, unit-variance.
        if adv.numel() > 1:
            adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

        return adv, ret

    def _policy_value(
        self,
        model: nn.Module,
        obs: torch.Tensor,
        action_masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run model to obtain logits, values, and policy entropy.

        Expected model contract:
            logits, values = model(obs)

        Shapes:
            obs:    (B, obs_dim)
            logits: (B, act_dim)
            values: (B,) or (B,1)

        Entropy:
            For a categorical distribution with probabilities p(a):
                H(p) = - Σ_a p(a) log p(a)

            High entropy = more randomness (exploration)
            Low entropy  = more certainty (exploitation)

        Returns:
            logits:  (B, act_dim)
            values:  (B,)
            entropy: (B,)
        """
        logits, values = model(obs)
        values = values.squeeze(-1).to(torch.float32)  # enforce (B,) and keep PPO loss math in fp32

        if action_masks is not None:
            logits = self._mask_logits(logits, action_masks)

        # Compute entropy in fp32 for numerical stability (especially under AMP).
        logp = F.log_softmax(logits.to(torch.float32), dim=-1)
        entropy = -(logp.exp() * logp).sum(-1)

        return logits, values, entropy