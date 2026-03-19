"""Per-slot PPO runtime for independent agent training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional
import math
import os

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR

try:
    from torch.func import functional_call, vmap, stack_module_state
except Exception:
    functional_call = None
    vmap = None
    stack_module_state = None

import config
from engine.agent_registry import AgentsRegistry


# Simple buffer for one agent's trajectory segment
@dataclass
class _RolloutBuffer:
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


@dataclass
class _PreparedTrainSlot:
    """
    Fully materialized PPO training payload for one slot.

    This isolates all rollout ownership per agent before any grouped execution
    begins. Grouped execution may batch computation across compatible slots, but
    each _PreparedTrainSlot still carries only that slot's own trajectory,
    optimizer, scheduler, and model reference.
    """
    aid: int
    model: nn.Module
    buf: _RolloutBuffer
    obs: torch.Tensor
    act: torch.Tensor
    logp_old: torch.Tensor
    val_old: torch.Tensor
    rew: torch.Tensor
    done: torch.Tensor
    act_mask: torch.Tensor
    adv: torch.Tensor
    ret: torch.Tensor
    batch_size: int
    n_mb: int
    mb_size: int


def _is_torchscript_module(m: nn.Module) -> bool:
    """Return True when torch.func transforms are unsafe for the module."""
    return isinstance(m, torch.jit.ScriptModule) or isinstance(m, torch.jit.RecursiveScriptModule)


# Main PPO runtime – one instance per simulation (shared by all agents)
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

        # Hyperparameters (read from config; defaults are conservative)
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
        self._batched_ppo_attention_warned = False
        # Additional PPO stability and batching controls.
        # - minibatches: number of minibatches per epoch
        # - max_grad_norm: gradient clipping norm (stabilizes training)
        # - target_kl: early stop if KL divergence exceeds this (trust region safety)
        self.minibatches = int(getattr(config, "PPO_MINIBATCHES", 1))
        self.max_grad_norm = float(getattr(config, "PPO_MAX_GRAD_NORM", 1.0))
        self.target_kl = float(getattr(config, "PPO_TARGET_KL", 0.0))

        # Strict rollout validation is kept available, but it is opt-in on the
        # record_step() hot path unless the broader config strict mode is enabled.
        # Fast-path correctness is still preserved because rollout sampling and
        # log-prob recording both use the same masked policy logits.
        self.strict_rollout_validation = (
            bool(getattr(config, "CONFIG_STRICT", False))
            or os.getenv("FWS_PPO_STRICT_ROLLOUT_VALIDATE", "0") in ("1", "true", "True")
        )

        # Per-slot PPO state
        # _buf: rollout buffer per slot (aid -> _Buf)
        # _opt: optimizer per slot (aid -> torch.optim.Optimizer)
        # _sched: LR scheduler per slot (aid -> CosineAnnealingLR)
        # _step: global decision step counter (increases each record_step call)
        self._buf: Dict[int, _Buf] = {}
        self._opt: Dict[int, optim.Optimizer] = {}
        self._sched: Dict[int, CosineAnnealingLR] = {}
        self._step = 0

        # Slot-local value cache used to bootstrap PPO window boundaries without
        # a second post-step observation/inference pass. Semantics:
        # - cache[slot] stores the most recent value estimate V(s_t) produced by the
        #   *normal* main inference path for that slot
        # - valid[slot] tells us whether that cache line belongs to the current slot occupant
        # - pending window state remembers which slots need V(s_{t+1}) on the next tick
        # IMPORTANT:
        # - This remains strictly slot-local (no sharing / no mixing across slots).
        # - reset/flush paths explicitly invalidate reused/dead slots.
        cap = int(self.registry.capacity)
        self._value_cache = torch.zeros((cap,), device=self.device, dtype=torch.float32)
        self._value_cache_valid = torch.zeros((cap,), device=self.device, dtype=torch.bool)
        self._pending_window_agent_ids: Optional[torch.Tensor] = None
        self._pending_window_done: Optional[torch.Tensor] = None

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

    # Shape & device assertions (fail loudly on mismatch)
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

    # Agent reset (called when a new agent respawns into an old slot)
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
        self.invalidate_value_cache([int(aid)])

        # If a pending boundary still references this slot, keep the pending metadata
        # intact; finalize_pending_window_from_cache() will ignore the slot if its
        # buffer was already cleared (e.g. terminal flush before respawn).

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

    def update_value_cache(self, agent_ids: torch.Tensor, values: torch.Tensor) -> None:
        """
        Update the persistent slot-local value cache from the normal main inference pass.

        Inputs:
        - agent_ids: (B,) registry slot indices aligned with `values`
        - values:    (B,) or (B,1) critic outputs V(s_t) for those slots

        This method is intentionally lightweight:
        - one batched fp32 cast
        - one indexed write into the cache
        - no cross-slot aggregation or mixing
        """
        if agent_ids is None or agent_ids.numel() == 0:
            return
        if agent_ids.device != self.device:
            raise RuntimeError(
                f"[ppo] value-cache agent_ids device mismatch: got {agent_ids.device}, expected {self.device}"
            )
        if values.device != self.device:
            raise RuntimeError(
                f"[ppo] value-cache values device mismatch: got {values.device}, expected {self.device}"
            )
        if agent_ids.dim() != 1:
            raise RuntimeError(f"[ppo] value-cache agent_ids must be (B,), got {tuple(agent_ids.shape)}")

        vals_f32 = values.detach().reshape(-1).to(torch.float32)
        if vals_f32.numel() != int(agent_ids.numel()):
            raise RuntimeError(
                f"[ppo] value-cache values length mismatch: ids={int(agent_ids.numel())} values={int(vals_f32.numel())}"
            )

        slot_ids = agent_ids.detach().to(torch.long)
        self._value_cache[slot_ids] = vals_f32
        self._value_cache_valid[slot_ids] = True

    def invalidate_value_cache(self, agent_ids: Any) -> None:
        """
        Invalidate cached values for the specified slots.

        This is called on slot flush/reset paths so reused slots never inherit a
        stale cached value from a previous occupant.
        """
        if agent_ids is None:
            return

        if isinstance(agent_ids, torch.Tensor):
            if agent_ids.numel() == 0:
                return
            slot_ids = agent_ids.detach().to(device=self.device, dtype=torch.long).reshape(-1)
        else:
            ids_list = [int(a) for a in agent_ids]
            if len(ids_list) == 0:
                return
            slot_ids = torch.tensor(ids_list, device=self.device, dtype=torch.long)

        self._value_cache_valid[slot_ids] = False
        self._value_cache[slot_ids] = 0.0

    def has_pending_window_bootstrap(self) -> bool:
        """
        Return True if the previous boundary step is waiting for cached V(s_{t+1}).
        """
        return (
            self._pending_window_agent_ids is not None and
            self._pending_window_done is not None
        )

    def finalize_pending_window_from_cache(self) -> bool:
        """
        Complete a deferred window-boundary bootstrap using the persistent value cache.

        Design:
        - record_step() stages the final boundary batch when it reaches the end of a
          PPO window without explicit bootstrap_values.
        - On the next tick, TickEngine updates the cache from the normal main forward
          pass (current state's V(s_t)).
        - This method copies those cached slot-local values into the pending buffers
          as V(s_{t+1}) and then runs the normal train-and-clear path.

        Done slots are intentionally bootstrapped with zero:
        - their final transition already has done=True, so GAE masks out the bootstrap term
        - this also prevents stale-cache contamination if the slot gets reused before
          this method runs
        """
        if not self.has_pending_window_bootstrap():
            return False

        aids = self._pending_window_agent_ids
        done = self._pending_window_done
        assert aids is not None and done is not None

        if aids.numel() != done.numel():
            raise RuntimeError(
                f"[ppo] pending bootstrap shape mismatch: aids={tuple(aids.shape)} done={tuple(done.shape)}"
            )

        bootstrap = torch.zeros((int(aids.numel()),), device=self.device, dtype=torch.float32)

        survivors = ~done
        if survivors.any():
            survivor_ids = aids[survivors]
            valid = self._value_cache_valid[survivor_ids]
            if not bool(valid.all().item()):
                bad_ids = survivor_ids[~valid][:8].detach().cpu().to(torch.int64).tolist()
                raise RuntimeError(
                    "[ppo] missing cached bootstrap values for pending boundary slots: "
                    f"{bad_ids}"
                )
            bootstrap[survivors] = self._value_cache[survivor_ids]

        for i, aid in enumerate(aids.detach().cpu().to(torch.int64).tolist()):
            b = self._buf.get(int(aid), None)
            if b is None or len(b.obs) == 0:
                continue
            b.bootstrap = bootstrap[i].reshape(1)

        self._pending_window_agent_ids = None
        self._pending_window_done = None
        self._train_window_and_clear()
        return True

    # Lazy buffer / optimizer creation
    def _get_buf(self, aid: int) -> _Buf:
        """
        Return rollout buffer for slot `aid`, creating it if missing.

        Buffers store *lists* because data arrives step-by-step.
        At training time we stack lists into batched tensors.
        """
        if aid not in self._buf:
            self._buf[aid] = _RolloutBuffer([], [], [], [], [], [], [])
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

    # Window timing helper
    def will_train_next_step(self) -> bool:
        """
        Return True if the *next* record_step call will trigger a PPO window train.

        Logic:
        - self._step counts how many record_step calls have occurred.
        - Training triggers when self._step % self.T == 0.
        - So, the next step triggers training if (self._step + 1) is a multiple of T.
        """
        return ((self._step + 1) % self.T) == 0

    # Record one step (called by environment after each tick)
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

        if self.has_pending_window_bootstrap():
            raise RuntimeError(
                "[ppo] pending window bootstrap must be finalized from cached values "
                "before recording another PPO step"
            )

        if action_masks is None:
            # Backward-compatible fallback for callers not yet passing masks.
            action_masks = torch.ones(
                (int(actions.size(0)), int(self.act_dim)),
                device=logits.device,
                dtype=torch.bool,
            )
        else:
            action_masks = action_masks.to(dtype=torch.bool)

        # Expensive rollout legality assertions remain available, but they are
        # guarded so the common record_step() fast path does not force an
        # unconditional device->host synchronization every tick. In normal
        # runtime the chosen action is still produced from the exact same masked
        # logits recorded below, so agent behavior and PPO semantics do not change.
        if self.strict_rollout_validation:
            bad_rows = (~action_masks.any(dim=-1)).nonzero(as_tuple=False).squeeze(1)
            if bad_rows.numel() != 0:
                bad_rows_list = bad_rows[:8].detach().cpu().to(torch.int64).tolist()
                raise RuntimeError(
                    f"[ppo] action_masks contains row(s) with no legal actions: first_bad_rows={bad_rows_list}"
                )

            ar = torch.arange(actions.numel(), device=actions.device)
            bad_actions = (~action_masks[ar, actions.long()]).nonzero(as_tuple=False).squeeze(1)
            if bad_actions.numel() != 0:
                bad_actions_list = bad_actions[:8].detach().cpu().to(torch.int64).tolist()
                raise RuntimeError(
                    f"[ppo] chosen action is illegal under rollout action_masks: first_bad_rows={bad_actions_list}"
                )

        # Compute log probabilities of chosen actions under the SAME masked rollout policy (fp32).
        logits_masked = self._mask_logits(logits, action_masks)
        logp_a = F.log_softmax(logits_masked.to(torch.float32), dim=-1).gather(1, actions.view(-1, 1)).squeeze(1)

        # Rationale: calling agent_ids[i].item() in a Python loop creates N individual
        # GPU→CPU sync points per tick (one per agent). Similarly, per-element .detach()
        # and .to() calls inside a loop launch one tiny kernel per agent per field.
        # By batch-casting all tensors once before the loop we reduce:
        #   • GPU syncs:      N .item() → 1 .tolist()
        #   • Cast kernels:   7×N → 7 batch ops + N slice reads (no cast)
        # Stored data and concatenation semantics are identical.
        obs_d    = obs.detach()                                          # (B, obs_dim)
        acts_d   = actions.detach()                                      # (B,)
        logp_f32 = logp_a.detach().to(torch.float32)                     # (B,)
        vals_f32 = values.detach().reshape(-1).to(torch.float32)         # (B,)
        rews_f32 = rewards.detach().to(torch.float32)                    # (B,)
        done_b   = done.detach().to(torch.bool)                          # (B,)
        masks_b  = action_masks.detach().to(torch.bool)                  # (B, act_dim)
        bs_f32   = (bootstrap_values.detach().to(torch.float32)
                    if bootstrap_values is not None else None)            # (B,) or None

        # Single .tolist() → one GPU→CPU transfer for all agent IDs.
        aids_list = agent_ids.tolist()

        # Append each agent transition into its own buffer.
        for i, aid in enumerate(aids_list):
            aid = int(aid)
            b = self._get_buf(aid)

            b.obs.append(obs_d[i])
            b.act.append(acts_d[i])
            b.logp.append(logp_f32[i])

            # unsqueeze(0) gives (1,) shape — same as the old reshape(1) but avoids
            # a copy; slicing a float32 tensor after batch cast is already correct dtype.
            b.val.append(vals_f32[i].unsqueeze(0))

            b.rew.append(rews_f32[i])
            b.done.append(done_b[i])
            b.act_mask.append(masks_b[i])

            # Store bootstrap V(s_{t+1}) if provided.
            if bs_f32 is not None:
                b.bootstrap = bs_f32[i].reshape(1)

        # Advance global step.
        self._step += 1

        # Train if we reached end of window.
        # Two supported modes:
        # 1) Immediate bootstrap path:
        #    Caller provided explicit bootstrap_values for this boundary step, so
        #    we can train immediately (legacy / compatibility path).
        # 2) Cached bootstrap path:
        #    Caller omitted bootstrap_values at the boundary. We defer training
        #    until the next tick's normal main forward updates the per-slot cache
        #    with V(s_{t+1}) for the surviving slots from this batch.
        if self._step % self.T == 0:
            if bs_f32 is not None:
                self._train_window_and_clear()
            else:
                self._pending_window_agent_ids = agent_ids.detach().to(self.device, dtype=torch.long).clone()
                self._pending_window_done = done_b.detach().to(self.device, dtype=torch.bool).clone()

    # Public flush method – train and clear specific agents immediately
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
        self.invalidate_value_cache(aids)


    # PPO training helpers
    def _clear_rollout_buffer(self, b: _RolloutBuffer) -> None:
        """Clear one rollout buffer in-place without touching optimizer state."""
        b.obs.clear()
        b.act.clear()
        b.logp.clear()
        b.val.clear()
        b.rew.clear()
        b.done.clear()
        b.act_mask.clear()
        b.bootstrap = None

    def _agg_tensor_stats(self, dst: Optional[Dict[str, Any]], prefix: str, x: torch.Tensor) -> None:
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

    def _agg_scalar(self, dst: Optional[Dict[str, Any]], key: str, value: float) -> None:
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

    def _mean_from_agg(self, agg: Optional[Dict[str, Any]], prefix: str) -> float:
        c = int(agg.get(f"{prefix}_count", 0)) if agg is not None else 0
        return (float(agg.get(f"{prefix}_sum", 0.0)) / float(c)) if c > 0 else float("nan")

    def _std_from_agg(self, agg: Optional[Dict[str, Any]], prefix: str) -> float:
        c = int(agg.get(f"{prefix}_count", 0)) if agg is not None else 0
        if c <= 0:
            return float("nan")
        s = float(agg.get(f"{prefix}_sum", 0.0))
        ss = float(agg.get(f"{prefix}_sumsq", 0.0))
        m = s / float(c)
        var = max(0.0, (ss / float(c)) - (m * m))
        return math.sqrt(var)

    def _make_agg_dict(self, collect_diag: bool) -> Optional[Dict[str, Any]]:
        if not collect_diag:
            return None
        return {
            "trained_slots": 0,
            "samples": 0,
            "optimizer_steps": 0,
            "epochs_run": 0,
            "loss_pi_sum": 0.0,
            "loss_v_sum": 0.0,
            "loss_ent_sum": 0.0,
            "entropy_sum": 0.0,
            "approx_kl_sum": 0.0,
            "approx_kl_count": 0,
            "approx_kl_max": 0.0,
            "lr_before_sum": 0.0,
            "lr_after_sum": 0.0,
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
        }

    def _prepare_train_slot(
        self,
        aid: int,
        *,
        agg: Optional[Dict[str, Any]],
        collect_diag: bool,
    ) -> Optional[_PreparedTrainSlot]:
        """
        Materialize one slot's rollout tensors before grouped execution.

        This preserves strict rollout ownership: every returned payload contains
        tensors from exactly one slot and nothing is mixed here.
        """
        aid = int(aid)
        b = self._buf.get(aid, None)
        if b is None or not b.obs:
            return None

        model = self.registry.brains[aid]
        if model is None:
            self._buf.pop(aid, None)
            return None

        obs = torch.stack(b.obs)
        act = torch.stack(b.act).long()
        logp_old = torch.stack(b.logp).to(torch.float32)
        val_old = torch.cat(b.val).to(torch.float32)
        rew = torch.stack(b.rew).to(torch.float32)
        done = torch.stack(b.done).bool()

        if len(getattr(b, "act_mask", [])) == len(b.obs):
            act_mask = torch.stack(b.act_mask).bool()
        elif len(getattr(b, "act_mask", [])) == 0:
            self._clear_rollout_buffer(b)
            return None
        else:
            raise RuntimeError(
                f"[ppo] act_mask buffer length mismatch for aid={aid}: "
                f"{len(getattr(b, 'act_mask', []))} vs {len(b.obs)}"
            )

        last_v = b.bootstrap
        adv, ret = self._gae(rew, val_old, done, last_value=last_v)
        if adv.dtype != torch.float32 or ret.dtype != torch.float32:
            raise RuntimeError(f"[ppo] GAE outputs must be float32, got adv={adv.dtype} ret={ret.dtype}")
        if not bool(torch.isfinite(logp_old).all().item()):
            raise RuntimeError("[ppo] non-finite logp_old in rollout buffer")
        if not bool(torch.isfinite(adv).all().item()) or not bool(torch.isfinite(ret).all().item()):
            raise RuntimeError("[ppo] non-finite adv/ret from GAE")

        B = int(obs.size(0))
        n_mb = max(1, min(int(self.minibatches), B))
        mb_size = max(1, B // n_mb)

        if collect_diag and agg is not None:
            self._agg_tensor_stats(agg, "rew", rew)
            self._agg_tensor_stats(agg, "adv_norm", adv)
            self._agg_tensor_stats(agg, "ret", ret)
            self._agg_tensor_stats(agg, "value_old", val_old)

        model.train()
        return _PreparedTrainSlot(
            aid=aid,
            model=model,
            buf=b,
            obs=obs,
            act=act,
            logp_old=logp_old,
            val_old=val_old,
            rew=rew,
            done=done,
            act_mask=act_mask,
            adv=adv,
            ret=ret,
            batch_size=B,
            n_mb=n_mb,
            mb_size=mb_size,
        )

    def _group_prepared_slots_by_arch(
        self,
        prepared: List[_PreparedTrainSlot],
    ) -> List[List[_PreparedTrainSlot]]:
        """Group prepared slots by persistent architecture metadata when available."""
        if not prepared:
            return []

        groups: Dict[Any, List[_PreparedTrainSlot]] = {}
        arch_tensor = getattr(self.registry, "brain_arch_ids", None)
        if torch.is_tensor(arch_tensor):
            aids_t = torch.tensor(
                [int(p.aid) for p in prepared],
                device=arch_tensor.device,
                dtype=torch.long,
            )
            arch_ids = arch_tensor.index_select(0, aids_t).detach().to("cpu").to(torch.int64).tolist()
        else:
            arch_ids = [int(-1) for _ in prepared]

        for slot, arch_id in zip(prepared, arch_ids):
            arch_id_i = int(arch_id)
            if arch_id_i >= 0:
                key = ("arch", arch_id_i)
            elif hasattr(self.registry, "_signature"):
                key = ("sig", getattr(self.registry, "_signature")(slot.model))
            else:
                key = ("type", type(slot.model).__module__, type(slot.model).__qualname__)
            groups.setdefault(key, []).append(slot)

        return list(groups.values())
    

    def _model_uses_multihead_attention(self, model: nn.Module) -> bool:
        """
        Return True if the model contains nn.MultiheadAttention anywhere.

        Why this guard exists:
        - The grouped PPO training path uses torch.func functional_call + vmap
          across whole models.
        - On CUDA, attention-heavy models can hit backend alignment issues during
          backward (for example: 'LSE is not correctly aligned (strideH)').
        - We do NOT want to redesign the brains here.
        - We simply route such models to the already-existing safe sequential PPO
          training path.
        """
        for mod in model.modules():
            if isinstance(mod, nn.MultiheadAttention):
                return True
        return False
    def _can_batched_train_group(self, group: List[_PreparedTrainSlot]) -> bool:
        if len(group) <= 1:
            return False
        if functional_call is None or vmap is None or stack_module_state is None:
            return False

        models = [slot.model for slot in group]
        base_type = type(models[0])

        # Keep the existing homogeneity guard.
        if any(type(m) is not base_type for m in models):
            return False

        # TorchScript remains unsupported for this path.
        if any(_is_torchscript_module(m) for m in models):
            return False

        # NEW SAFETY GUARD:
        # Grouped/vmap PPO backward is currently not safe for attention-based brains.
        # Fall back to the already-existing sequential per-slot PPO trainer.
        if any(self._model_uses_multihead_attention(m) for m in models):
            if not self._batched_ppo_attention_warned:
                print("[ppo] grouped batched PPO disabled for attention-based brains; using safe sequential trainer")
                self._batched_ppo_attention_warned = True
            return False

        return True

    def _policy_value_group_batched(
        self,
        models: List[nn.Module],
        obs: torch.Tensor,
        action_masks: torch.Tensor,
    ) -> Tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Batched per-model forward/loss ingredients for one homogeneous lane.

        Each leading-dimension slice still belongs to one independent model.
        We only batch the math; we do not share parameters or optimizer state.
        """
        if functional_call is None or vmap is None or stack_module_state is None:
            raise RuntimeError("[ppo] torch.func is not available for batched PPO training")
        if len(models) == 0:
            raise RuntimeError("[ppo] empty model group for batched PPO training")
        if obs.dim() != 3:
            raise RuntimeError(f"[ppo] batched obs must be (G,M,F), got {tuple(obs.shape)}")
        if action_masks.dim() != 3:
            raise RuntimeError(
                f"[ppo] batched action_masks must be (G,M,A), got {tuple(action_masks.shape)}"
            )
        if int(obs.size(0)) != len(models) or int(action_masks.size(0)) != len(models):
            raise RuntimeError(
                f"[ppo] batched model/obs alignment mismatch: models={len(models)} obs={tuple(obs.shape)} masks={tuple(action_masks.shape)}"
            )
        if int(action_masks.size(-1)) != int(self.act_dim):
            raise RuntimeError(
                f"[ppo] batched action_masks last dim mismatch: got {int(action_masks.size(-1))}, expected {int(self.act_dim)}"
            )
        if not bool(action_masks.any(dim=-1).all().item()):
            raise RuntimeError("[ppo] batched action mask row has no legal actions")

        base = models[0]
        params_batched, buffers_batched = stack_module_state(models)

        def _f(params_i: Any, buffers_i: Any, obs_i: torch.Tensor, mask_i: torch.Tensor):
            out = functional_call(base, (params_i, buffers_i), (obs_i,))
            if not (isinstance(out, tuple) and len(out) == 2):
                raise RuntimeError("Brain.forward must return (logits_or_dist, value)")
            head_i, values_i = out
            logits_i = head_i.logits if hasattr(head_i, "logits") else head_i
            values_i = values_i.squeeze(-1).to(torch.float32)
            neg = logits_i.new_tensor(torch.finfo(logits_i.dtype).min)
            logits_i = torch.where(mask_i.to(device=logits_i.device, dtype=torch.bool), logits_i, neg)
            logp_i = F.log_softmax(logits_i.to(torch.float32), dim=-1)
            entropy_i = -(logp_i.exp() * logp_i).sum(-1)
            return logits_i, values_i, entropy_i

        logits, values, entropy = vmap(_f, in_dims=(0, 0, 0, 0))(params_batched, buffers_batched, obs, action_masks)
        return params_batched, logits, values, entropy

    def _assign_stacked_grads_to_models(self, models: List[nn.Module], params_batched: Any) -> None:
        """
        Scatter per-model gradients from stacked torch.func parameters back onto
        the original nn.Module parameters before the independent optimizer steps.
        """
        for row_i, model in enumerate(models):
            for name, param in model.named_parameters():
                g_stacked = params_batched[name].grad
                if g_stacked is None:
                    param.grad = None
                else:
                    param.grad = g_stacked[row_i].detach().clone()

    def _finalize_slot_training_stats(
        self,
        slot: _PreparedTrainSlot,
        *,
        agg: Optional[Dict[str, Any]],
        collect_diag: bool,
        aid_epochs_run: int,
        aid_optimizer_steps: int,
        aid_loss_pi_sum: float,
        aid_loss_v_sum: float,
        aid_loss_ent_sum: float,
        aid_entropy_sum: float,
        aid_approx_kl_max: float,
        lr_before: float,
        lr_after: float,
    ) -> None:
        if collect_diag and agg is not None:
            n_steps = int(aid_optimizer_steps)
            if n_steps > 0:
                agg["trained_slots"] += 1
                agg["samples"] += int(slot.batch_size)
                agg["optimizer_steps"] += n_steps
                agg["epochs_run"] += int(aid_epochs_run)
                agg["loss_pi_sum"] += float(aid_loss_pi_sum)
                agg["loss_v_sum"] += float(aid_loss_v_sum)
                agg["loss_ent_sum"] += float(aid_loss_ent_sum)
                agg["entropy_sum"] += float(aid_entropy_sum)
                agg["approx_kl_max"] = max(float(agg["approx_kl_max"]), float(aid_approx_kl_max))
                agg["lr_before_sum"] += float(lr_before)
                agg["lr_after_sum"] += float(lr_after)

        self._clear_rollout_buffer(slot.buf)

    def _train_prepared_slot_sequential(
        self,
        slot: _PreparedTrainSlot,
        *,
        agg: Optional[Dict[str, Any]],
        collect_diag: bool,
    ) -> None:
        """Original one-slot PPO path, kept as the safe fallback baseline."""
        model = slot.model
        model.train()

        opt = self._get_opt(slot.aid, model)
        lr_before = float(opt.param_groups[0]["lr"]) if len(getattr(opt, "param_groups", [])) > 0 else float(self.lr)

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
                idx = torch.randperm(slot.batch_size, device=slot.obs.device)
                approx_kl_epoch = 0.0

                for start_i in range(0, slot.batch_size, slot.mb_size):
                    mb_idx = idx[start_i:start_i + slot.mb_size]
                    if mb_idx.numel() == 0:
                        continue

                    obs_mb = slot.obs[mb_idx]
                    act_mb = slot.act[mb_idx]
                    act_mask_mb = slot.act_mask[mb_idx]
                    logp_old_mb = slot.logp_old[mb_idx].detach()
                    val_old_mb = slot.val_old[mb_idx].detach()
                    adv_mb = slot.adv[mb_idx].detach()
                    ret_mb = slot.ret[mb_idx].detach()

                    logits, values, entropy = self._policy_value(model, obs_mb, action_masks=act_mask_mb)
                    logp = F.log_softmax(logits.to(torch.float32), dim=-1).gather(1, act_mb.view(-1, 1)).squeeze(1)

                    ratio = torch.exp(logp - logp_old_mb)
                    surr1 = ratio * adv_mb
                    surr2 = torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip) * adv_mb
                    loss_pi = -torch.min(surr1, surr2).mean()

                    v_pred = values
                    v_clipped = val_old_mb + torch.clamp(v_pred - val_old_mb, -self.clip, self.clip)
                    loss_v1 = (v_pred - ret_mb) ** 2
                    loss_v2 = (v_clipped - ret_mb) ** 2
                    loss_v = torch.max(loss_v1, loss_v2).mean()

                    loss_ent = -entropy.mean()
                    loss = loss_pi + self.vf_coef * loss_v + self.ent_coef * loss_ent

                    opt.zero_grad(set_to_none=True)
                    loss.backward()

                    grad_norm_raw = torch.nn.utils.clip_grad_norm_(model.parameters(), float(self.max_grad_norm))
                    if torch.is_tensor(grad_norm_raw):
                        grad_norm_val = float(grad_norm_raw.detach().item())
                    else:
                        grad_norm_val = float(grad_norm_raw)
                    opt.step()

                    approx_kl = float((logp_old_mb - logp).mean().item())
                    approx_kl_epoch = max(approx_kl_epoch, approx_kl)

                    if collect_diag and agg is not None:
                        ratio_det = ratio.detach().to(torch.float32)
                        v_pred_det = v_pred.detach().to(torch.float32)
                        self._agg_tensor_stats(agg, "ratio", ratio_det)
                        self._agg_tensor_stats(agg, "value_pred", v_pred_det)
                        self._agg_tensor_stats(agg, "mask_valid_actions", act_mask_mb.to(torch.float32).sum(dim=-1))

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

                        try:
                            ret_var_t = torch.var(ret_mb.detach().to(torch.float32), unbiased=False)
                            ret_var = float(ret_var_t.item())
                            if ret_var > 1e-12:
                                resid_var = float(torch.var((ret_mb.detach().to(torch.float32) - v_pred_det), unbiased=False).item())
                                ev = 1.0 - (resid_var / ret_var)
                                if math.isfinite(ev):
                                    self._agg_scalar(agg, "explained_var", ev)
                        except Exception:
                            pass

                    if collect_diag:
                        aid_optimizer_steps += 1
                        aid_loss_pi_sum += float(loss_pi.detach().item())
                        aid_loss_v_sum += float(loss_v.detach().item())
                        aid_loss_ent_sum += float(loss_ent.detach().item())
                        aid_entropy_sum += float(entropy.mean().detach().item())
                        aid_approx_kl_max = max(float(aid_approx_kl_max), float(approx_kl))

                if self.target_kl > 0.0 and approx_kl_epoch > self.target_kl:
                    break

        if slot.aid in self._sched:
            self._sched[slot.aid].step()
        lr_after = float(opt.param_groups[0]["lr"]) if len(getattr(opt, "param_groups", [])) > 0 else lr_before

        self._finalize_slot_training_stats(
            slot,
            agg=agg,
            collect_diag=collect_diag,
            aid_epochs_run=aid_epochs_run,
            aid_optimizer_steps=aid_optimizer_steps,
            aid_loss_pi_sum=aid_loss_pi_sum,
            aid_loss_v_sum=aid_loss_v_sum,
            aid_loss_ent_sum=aid_loss_ent_sum,
            aid_entropy_sum=aid_entropy_sum,
            aid_approx_kl_max=aid_approx_kl_max,
            lr_before=lr_before,
            lr_after=lr_after,
        )

    def _train_prepared_group_batched(
        self,
        group: List[_PreparedTrainSlot],
        *,
        agg: Optional[Dict[str, Any]],
        collect_diag: bool,
    ) -> None:
        """
        Train one homogeneous architecture group in grouped execution lanes.

        Independence guarantees:
        - every slot still uses only its own rollout tensors
        - gradients are scattered back onto each original model separately
        - optimizer and scheduler steps remain per slot
        - there is no parameter averaging or shared optimizer state
        """
        if not self._can_batched_train_group(group):
            raise RuntimeError("[ppo] incompatible group passed to batched PPO trainer")

        local_stats: List[Dict[str, Any]] = []
        for slot in group:
            slot.model.train()
            opt = self._get_opt(slot.aid, slot.model)
            lr_before = float(opt.param_groups[0]["lr"]) if len(getattr(opt, "param_groups", [])) > 0 else float(self.lr)
            local_stats.append({
                "opt": opt,
                "lr_before": lr_before,
                "epochs_run": 0,
                "optimizer_steps": 0,
                "loss_pi_sum": 0.0,
                "loss_v_sum": 0.0,
                "loss_ent_sum": 0.0,
                "entropy_sum": 0.0,
                "approx_kl_max": 0.0,
            })

        active = [True] * len(group)

        with torch.enable_grad():
            for _ in range(self.epochs):
                epoch_active = [i for i, is_active in enumerate(active) if is_active]
                if not epoch_active:
                    break

                if collect_diag:
                    for i in epoch_active:
                        local_stats[i]["epochs_run"] += 1

                perms = {
                    i: torch.randperm(group[i].batch_size, device=group[i].obs.device)
                    for i in epoch_active
                }
                approx_kl_epoch = {i: 0.0 for i in epoch_active}
                max_chunks = max(
                    (group[i].batch_size + group[i].mb_size - 1) // group[i].mb_size
                    for i in epoch_active
                )

                for chunk_i in range(max_chunks):
                    participants: List[int] = []
                    mb_indices: List[torch.Tensor] = []
                    max_len = 0
                    for i in epoch_active:
                        start_i = chunk_i * group[i].mb_size
                        if start_i >= group[i].batch_size:
                            continue
                        mb_idx = perms[i][start_i:start_i + group[i].mb_size]
                        if mb_idx.numel() == 0:
                            continue
                        participants.append(i)
                        mb_indices.append(mb_idx)
                        max_len = max(max_len, int(mb_idx.numel()))

                    if not participants:
                        continue

                    lane_models = [group[i].model for i in participants]
                    lane_count = len(participants)
                    obs_batch = torch.zeros(
                        (lane_count, max_len, self.obs_dim),
                        device=self.device,
                        dtype=group[participants[0]].obs.dtype,
                    )
                    act_batch = torch.zeros((lane_count, max_len), device=self.device, dtype=torch.long)
                    mask_batch = torch.ones((lane_count, max_len, self.act_dim), device=self.device, dtype=torch.bool)
                    logp_old_batch = torch.zeros((lane_count, max_len), device=self.device, dtype=torch.float32)
                    val_old_batch = torch.zeros((lane_count, max_len), device=self.device, dtype=torch.float32)
                    adv_batch = torch.zeros((lane_count, max_len), device=self.device, dtype=torch.float32)
                    ret_batch = torch.zeros((lane_count, max_len), device=self.device, dtype=torch.float32)
                    valid_batch = torch.zeros((lane_count, max_len), device=self.device, dtype=torch.bool)

                    for row_i, (slot_i, mb_idx) in enumerate(zip(participants, mb_indices)):
                        slot = group[slot_i]
                        n_i = int(mb_idx.numel())
                        obs_batch[row_i, :n_i] = slot.obs[mb_idx]
                        act_batch[row_i, :n_i] = slot.act[mb_idx]
                        mask_batch[row_i, :n_i] = slot.act_mask[mb_idx]
                        logp_old_batch[row_i, :n_i] = slot.logp_old[mb_idx]
                        val_old_batch[row_i, :n_i] = slot.val_old[mb_idx]
                        adv_batch[row_i, :n_i] = slot.adv[mb_idx]
                        ret_batch[row_i, :n_i] = slot.ret[mb_idx]
                        valid_batch[row_i, :n_i] = True

                    for slot_i in participants:
                        local_stats[slot_i]["opt"].zero_grad(set_to_none=True)

                    params_batched, logits, values, entropy = self._policy_value_group_batched(
                        lane_models,
                        obs_batch,
                        mask_batch,
                    )
                    logp = F.log_softmax(logits.to(torch.float32), dim=-1).gather(2, act_batch.unsqueeze(-1)).squeeze(-1)

                    ratio = torch.exp(logp - logp_old_batch)
                    surr1 = ratio * adv_batch
                    surr2 = torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip) * adv_batch
                    valid_f = valid_batch.to(torch.float32)
                    counts = valid_f.sum(dim=1).clamp_min(1.0)

                    loss_pi_vec = -(torch.min(surr1, surr2) * valid_f).sum(dim=1) / counts
                    v_pred = values
                    v_clipped = val_old_batch + torch.clamp(v_pred - val_old_batch, -self.clip, self.clip)
                    loss_v1 = (v_pred - ret_batch) ** 2
                    loss_v2 = (v_clipped - ret_batch) ** 2
                    loss_v_vec = (torch.max(loss_v1, loss_v2) * valid_f).sum(dim=1) / counts
                    loss_ent_vec = -(entropy * valid_f).sum(dim=1) / counts
                    loss_vec = loss_pi_vec + self.vf_coef * loss_v_vec + self.ent_coef * loss_ent_vec
                    total_loss = loss_vec.sum()
                    total_loss.backward()

                    self._assign_stacked_grads_to_models(lane_models, params_batched)
                    approx_kl_vec = ((logp_old_batch - logp) * valid_f).sum(dim=1) / counts

                    for row_i, (slot_i, mb_idx) in enumerate(zip(participants, mb_indices)):
                        slot = group[slot_i]
                        opt = local_stats[slot_i]["opt"]
                        grad_norm_raw = torch.nn.utils.clip_grad_norm_(slot.model.parameters(), float(self.max_grad_norm))
                        if torch.is_tensor(grad_norm_raw):
                            grad_norm_val = float(grad_norm_raw.detach().item())
                        else:
                            grad_norm_val = float(grad_norm_raw)
                        opt.step()

                        approx_kl = float(approx_kl_vec[row_i].detach().item())
                        approx_kl_epoch[slot_i] = max(float(approx_kl_epoch.get(slot_i, 0.0)), approx_kl)

                        if collect_diag and agg is not None:
                            n_i = int(mb_idx.numel())
                            ratio_det = ratio[row_i, :n_i].detach().to(torch.float32)
                            v_pred_det = v_pred[row_i, :n_i].detach().to(torch.float32)
                            ret_mb = ret_batch[row_i, :n_i]
                            act_mask_mb = mask_batch[row_i, :n_i]
                            self._agg_tensor_stats(agg, "ratio", ratio_det)
                            self._agg_tensor_stats(agg, "value_pred", v_pred_det)
                            self._agg_tensor_stats(agg, "mask_valid_actions", act_mask_mb.to(torch.float32).sum(dim=-1))

                            clip_frac = float(((ratio_det - 1.0).abs() > float(self.clip)).to(torch.float32).mean().item())
                            agg["clip_frac_sum"] += clip_frac
                            agg["clip_frac_count"] += 1
                            agg["clip_frac_max"] = max(float(agg["clip_frac_max"]), clip_frac)

                            agg["loss_total_sum"] += float(loss_vec[row_i].detach().item())
                            agg["loss_total_count"] += 1
                            agg["approx_kl_sum"] += float(approx_kl)
                            agg["approx_kl_count"] += 1

                            if math.isfinite(grad_norm_val):
                                agg["grad_norm_sum"] += grad_norm_val
                                agg["grad_norm_count"] += 1
                                agg["grad_norm_max"] = max(float(agg["grad_norm_max"]), grad_norm_val)

                            try:
                                ret_var_t = torch.var(ret_mb.detach().to(torch.float32), unbiased=False)
                                ret_var = float(ret_var_t.item())
                                if ret_var > 1e-12:
                                    resid_var = float(torch.var((ret_mb.detach().to(torch.float32) - v_pred_det), unbiased=False).item())
                                    ev = 1.0 - (resid_var / ret_var)
                                    if math.isfinite(ev):
                                        self._agg_scalar(agg, "explained_var", ev)
                            except Exception:
                                pass

                        if collect_diag:
                            local_stats[slot_i]["optimizer_steps"] += 1
                            local_stats[slot_i]["loss_pi_sum"] += float(loss_pi_vec[row_i].detach().item())
                            local_stats[slot_i]["loss_v_sum"] += float(loss_v_vec[row_i].detach().item())
                            local_stats[slot_i]["loss_ent_sum"] += float(loss_ent_vec[row_i].detach().item())
                            local_stats[slot_i]["entropy_sum"] += float((entropy[row_i, :int(mb_idx.numel())].mean()).detach().item())
                            local_stats[slot_i]["approx_kl_max"] = max(
                                float(local_stats[slot_i]["approx_kl_max"]),
                                float(approx_kl),
                            )

                if self.target_kl > 0.0:
                    for slot_i, kl_val in approx_kl_epoch.items():
                        if float(kl_val) > float(self.target_kl):
                            active[slot_i] = False

        for slot, st in zip(group, local_stats):
            if slot.aid in self._sched:
                self._sched[slot.aid].step()
            lr_after = float(st["opt"].param_groups[0]["lr"]) if len(getattr(st["opt"], "param_groups", [])) > 0 else float(st["lr_before"])
            self._finalize_slot_training_stats(
                slot,
                agg=agg,
                collect_diag=collect_diag,
                aid_epochs_run=int(st["epochs_run"]),
                aid_optimizer_steps=int(st["optimizer_steps"]),
                aid_loss_pi_sum=float(st["loss_pi_sum"]),
                aid_loss_v_sum=float(st["loss_v_sum"]),
                aid_loss_ent_sum=float(st["loss_ent_sum"]),
                aid_entropy_sum=float(st["entropy_sum"]),
                aid_approx_kl_max=float(st["approx_kl_max"]),
                lr_before=float(st["lr_before"]),
                lr_after=float(lr_after),
            )

    # Core training routine – shared by window flush and manual flush
    def _train_aids_and_clear(self, aids: List[int]) -> None:
        """
        Train and clear the specified slot buffers.

        Refactor intent:
        - rollout extraction/GAE is still slot-local
        - compatible slots are grouped into architecture-homogeneous lanes
        - forward/loss math is batched per lane when torch.func is available
        - optimizer and scheduler ownership remain strictly per slot
        """
        if len(aids) == 0:
            return

        self._assert_no_optimizer_sharing(aids)

        log_ppo_telem = bool(getattr(config, "TELEMETRY_LOG_PPO", False))
        rich_ppo_telem = bool(getattr(config, "TELEMETRY_PPO_RICH_CSV", False))
        rich_level_requested = str(getattr(config, "TELEMETRY_PPO_RICH_LEVEL", "update")).strip().lower()
        rich_level_effective = "update"
        collect_diag = bool(log_ppo_telem or rich_ppo_telem)
        agg = self._make_agg_dict(collect_diag)

        prepared: List[_PreparedTrainSlot] = []
        for aid in aids:
            slot = self._prepare_train_slot(int(aid), agg=agg, collect_diag=collect_diag)
            if slot is not None:
                prepared.append(slot)

        for group in self._group_prepared_slots_by_arch(prepared):
            if self._can_batched_train_group(group):
                self._train_prepared_group_batched(group, agg=agg, collect_diag=collect_diag)
            else:
                for slot in group:
                    self._train_prepared_slot_sequential(slot, agg=agg, collect_diag=collect_diag)

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
                "ratio_mean": self._mean_from_agg(agg, "ratio"),
                "ratio_std": self._std_from_agg(agg, "ratio"),
                "ratio_min": float(agg.get("ratio_min", float("nan"))),
                "ratio_max": float(agg.get("ratio_max", float("nan"))),
                "adv_norm_mean": self._mean_from_agg(agg, "adv_norm"),
                "adv_norm_std": self._std_from_agg(agg, "adv_norm"),
                "adv_norm_min": float(agg.get("adv_norm_min", float("nan"))),
                "adv_norm_max": float(agg.get("adv_norm_max", float("nan"))),
                "ret_mean": self._mean_from_agg(agg, "ret"),
                "ret_std": self._std_from_agg(agg, "ret"),
                "ret_min": float(agg.get("ret_min", float("nan"))),
                "ret_max": float(agg.get("ret_max", float("nan"))),
                "rew_mean": self._mean_from_agg(agg, "rew"),
                "rew_std": self._std_from_agg(agg, "rew"),
                "rew_min": float(agg.get("rew_min", float("nan"))),
                "rew_max": float(agg.get("rew_max", float("nan"))),
                "value_old_mean": self._mean_from_agg(agg, "value_old"),
                "value_old_std": self._std_from_agg(agg, "value_old"),
                "value_old_min": float(agg.get("value_old_min", float("nan"))),
                "value_old_max": float(agg.get("value_old_max", float("nan"))),
                "value_pred_mean": self._mean_from_agg(agg, "value_pred"),
                "value_pred_std": self._std_from_agg(agg, "value_pred"),
                "value_pred_min": float(agg.get("value_pred_min", float("nan"))),
                "value_pred_max": float(agg.get("value_pred_max", float("nan"))),
                "grad_norm_mean": (float(agg["grad_norm_sum"]) / float(max(1, int(agg["grad_norm_count"]))))
                    if int(agg.get("grad_norm_count", 0)) > 0 else float("nan"),
                "grad_norm_max": float(agg.get("grad_norm_max", float("nan"))),
                "explained_var_mean": self._mean_from_agg(agg, "explained_var"),
                "explained_var_min": float(agg.get("explained_var_min", float("nan"))),
                "explained_var_max": float(agg.get("explained_var_max", float("nan"))),
                "mask_valid_actions_mean": self._mean_from_agg(agg, "mask_valid_actions"),
                "mask_valid_actions_min": float(agg.get("mask_valid_actions_min", float("nan"))),
                "mask_valid_actions_max": float(agg.get("mask_valid_actions_max", float("nan"))),
                "clip_epsilon": float(self.clip),
                "target_kl": float(self.target_kl),
                "ppo_epochs_cfg": float(self.epochs),
                "ppo_minibatches_cfg": float(self.minibatches),
                "max_grad_norm_cfg": float(self.max_grad_norm),
            }
            self.last_train_summary = dict(summary_row)

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

    # Checkpointing – save/load all runtime state
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
            "value_cache": cpuize(self._value_cache),
            "value_cache_valid": cpuize(self._value_cache_valid),
            "pending_window_agent_ids": cpuize(self._pending_window_agent_ids),
            "pending_window_done": cpuize(self._pending_window_done),
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

        # Restore value-cache state. Older checkpoints may not have these keys.
        cap = int(self.registry.capacity)
        vc = state.get("value_cache", None)
        vcv = state.get("value_cache_valid", None)
        if torch.is_tensor(vc) and int(vc.numel()) == cap:
            self._value_cache = vc.to(device=dev, dtype=torch.float32).reshape(cap)
        else:
            self._value_cache = torch.zeros((cap,), device=dev, dtype=torch.float32)

        if torch.is_tensor(vcv) and int(vcv.numel()) == cap:
            self._value_cache_valid = vcv.to(device=dev, dtype=torch.bool).reshape(cap)
        else:
            self._value_cache_valid = torch.zeros((cap,), device=dev, dtype=torch.bool)

        pwa = state.get("pending_window_agent_ids", None)
        pwd = state.get("pending_window_done", None)
        self._pending_window_agent_ids = (
            pwa.to(device=dev, dtype=torch.long).reshape(-1)
            if torch.is_tensor(pwa) else None
        )
        self._pending_window_done = (
            pwd.to(device=dev, dtype=torch.bool).reshape(-1)
            if torch.is_tensor(pwd) else None
        )

        # Restore buffers
        self._buf.clear()
        buf_in: Dict[int, Any] = state.get("buf", {})
        for aid, payload in buf_in.items():
            aid_i = int(aid)
            self._buf[aid_i] = _RolloutBuffer(
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

    # Helper methods for PPO computations
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
            head_or_logits, values = model(obs)

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
        out = model(obs)
        if not (isinstance(out, tuple) and len(out) == 2):
            raise RuntimeError("Brain.forward must return (logits_or_dist, value)")

        head, values = out
        logits = head.logits if hasattr(head, "logits") else head
        values = values.squeeze(-1).to(torch.float32)

        if action_masks is not None:
            logits = self._mask_logits(logits, action_masks)

        logp = F.log_softmax(logits.to(torch.float32), dim=-1)
        entropy = -(logp.exp() * logp).sum(-1)

        return logits, values, entropy
