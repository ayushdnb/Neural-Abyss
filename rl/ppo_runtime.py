from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional
import math

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
    The bootstrap field is used to store the value of the state after the last
    recorded step (V(s_{t+1})), which is needed for correct GAE calculation
    at window boundaries.
    """
    obs: List[torch.Tensor]          # observations (obs_dim,)
    act: List[torch.Tensor]          # actions taken (scalar)
    logp: List[torch.Tensor]         # log‑probabilities of those actions (scalar)
    val: List[torch.Tensor]          # value estimates from the critic (scalar)
    rew: List[torch.Tensor]          # rewards received (scalar)
    done: List[torch.Tensor]         # done flags (episode end) (bool)
    bootstrap: Optional[torch.Tensor] = None  # V(s_{t+1}) for the last step of a rollout window


# ----------------------------------------------------------------------
# Main PPO runtime – one instance per simulation (shared by all agents)
# ----------------------------------------------------------------------
class PerAgentPPORuntime:
    """
    Minimal per‑agent PPO runtime.
    Collects trajectories for every agent, trains them independently
    (no parameter sharing between slots), and supports immediate
    flushing of dead agents before respawn.
    """

    def __init__(self, registry: AgentsRegistry, device: torch.device,
                 obs_dim: int, act_dim: int):
        """
        Args:
            registry: holds all agent brains (models)
            device:   torch device for all tensors
            obs_dim:  dimension of observation vectors
            act_dim:  number of discrete actions
        """
        self.registry = registry
        self.device = device
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)

        # Hyperparameters from config (with safe defaults)
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

        # New config parameters introduced by the patch
        self.minibatches = int(getattr(config, "PPO_MINIBATCHES", 1))      # number of minibatches per epoch
        self.max_grad_norm = float(getattr(config, "PPO_MAX_GRAD_NORM", 1.0))  # gradient clipping norm
        self.target_kl = float(getattr(config, "PPO_TARGET_KL", 0.0))      # early stopping KL threshold (0 = disabled)

        # Per‑agent data structures
        self._buf: Dict[int, _Buf] = {}               # rollout buffers keyed by agent id
        self._opt: Dict[int, optim.Optimizer] = {}    # optimizers keyed by agent id
        self._sched: Dict[int, CosineAnnealingLR] = {}# schedulers keyed by agent id
        self._step = 0                                 # global tick counter

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
    ) -> None:
        """
        Verify that all tensors have the expected dimensions and are on the
        correct device.  Called inside record_step().
        """
        dev = self.device
        # All tensors must be on the runtime device
        if (agent_ids.device != dev or obs.device != dev or
            logits.device != dev or values.device != dev or
            actions.device != dev):
            raise RuntimeError(
                f"[ppo] device mismatch: ids={agent_ids.device} obs={obs.device} "
                f"logits={logits.device} values={values.device} "
                f"actions={actions.device} expected={dev}"
            )

        # agent_ids must be 1D (batch)
        if agent_ids.dim() != 1:
            raise RuntimeError(f"[ppo] agent_ids must be (B,), got {tuple(agent_ids.shape)}")
        B = int(agent_ids.size(0))

        # observation shape: (B, obs_dim)
        if obs.dim() != 2 or int(obs.size(0)) != B or int(obs.size(1)) != int(self.obs_dim):
            raise RuntimeError(f"[ppo] obs must be (B,{int(self.obs_dim)}), got {tuple(obs.shape)}")

        # logits shape: (B, act_dim)
        if logits.dim() != 2 or int(logits.size(0)) != B or int(logits.size(1)) != int(self.act_dim):
            raise RuntimeError(f"[ppo] logits must be (B,{int(self.act_dim)}), got {tuple(logits.shape)}")

        # values can be (B,) or (B,1) – we will squeeze later
        if values.dim() == 2 and (int(values.size(0)) == B and int(values.size(1)) == 1):
            pass
        elif values.dim() == 1 and int(values.size(0)) == B:
            pass
        else:
            raise RuntimeError(f"[ppo] values must be (B,) or (B,1), got {tuple(values.shape)}")

        # actions must be (B,) with integer class indices
        if actions.dim() != 1 or int(actions.size(0)) != B:
            raise RuntimeError(f"[ppo] actions must be (B,), got {tuple(actions.shape)}")

    def _assert_no_optimizer_sharing(self, aids: List[int]) -> None:
        """
        Defensive check: ensure the same optimizer object is never used for
        two different agents.  Under the "no hive mind" design this should
        never happen, but we catch it early if it does.
        """
        seen = {}                      # map from optimizer id -> agent id
        for aid in aids:
            opt = self._opt.get(int(aid), None)
            if opt is None:
                continue
            key = id(opt)               # unique identifier of the optimizer object
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
        Hard‑reset PPO state for a single slot.
        Discards the buffer, optimizer, and scheduler so that the new agent
        does not inherit old statistics.
        """
        assert 0 <= int(aid) < int(self.registry.capacity), f"aid out of range: {aid}"
        # Order matters: scheduler holds a reference to the optimizer.
        self._sched.pop(int(aid), None)
        self._opt.pop(int(aid), None)
        self._buf.pop(int(aid), None)

    def reset_agents(self, aids: torch.Tensor | List[int]) -> None:
        """
        Vectorised helper – reset many agents at once.
        Accepts a LongTensor or a plain Python list of slot indices.
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
        """Return the rollout buffer for agent `aid`, creating it if missing."""
        if aid not in self._buf:
            self._buf[aid] = _Buf([], [], [], [], [], [])
        return self._buf[aid]

    def _get_opt(self, aid: int, model: nn.Module) -> optim.Optimizer:
        """
        Return the Adam optimizer for agent `aid`, creating it (and its
        cosine scheduler) if not already present.
        """
        if aid not in self._opt:
            self._opt[aid] = optim.Adam(model.parameters(), lr=self.lr)
            self._sched[aid] = CosineAnnealingLR(
                self._opt[aid], T_max=self.T_max, eta_min=self.eta_min
            )
        return self._opt[aid]

    # New method added by the patch: tells if the next record_step will trigger training
    def will_train_next_step(self) -> bool:
        """Return True if the *next* record_step call will trigger a PPO window train."""
        return ((self._step + 1) % self.T) == 0

    # ------------------------------------------------------------------
    # Record one step (called by the environment after each tick)
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
        # New optional argument added by the patch: bootstrap values for the next state
        bootstrap_values: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Append a single decision step for all agents that acted in this tick.
        The data is stored in the per‑agent buffers.  When the global step
        counter reaches a multiple of `self.T`, a training window is triggered.

        The optional bootstrap_values tensor provides V(s_{t+1}) for each agent,
        which is needed for correct GAE calculation at the end of a window.
        It should be provided when the step is the last of a window.
        """
        # Crash loudly if shapes/devices are wrong – prevents silent corruption.
        self._assert_record_shapes(agent_ids, obs, logits, values, actions)

        # Compute log‑probabilities of the taken actions
        logp_a = F.log_softmax(logits, dim=-1).gather(1, actions.view(-1, 1)).squeeze(1)

        # Append each agent's transition to its own buffer
        for i in range(agent_ids.numel()):
            aid = int(agent_ids[i].item())
            b = self._get_buf(aid)
            b.obs.append(obs[i])               # observation
            b.act.append(actions[i])            # chosen action
            b.logp.append(logp_a[i])             # log‑prob of that action
            b.val.append(values[i].reshape(1))   # value estimate (keep as 1‑element tensor)
            b.rew.append(rewards[i])              # reward
            b.done.append(done[i])                 # done flag

            # If bootstrap values were provided, store them for later use at window boundary
            if bootstrap_values is not None:
                # Bootstrap value for the *post-step* state (used only at window boundary)
                b.bootstrap = bootstrap_values[i].detach().reshape(1)

        self._step += 1
        if self._step % self.T == 0:
            # Time to train on all agents that have collected data
            self._train_window_and_clear()

    # ------------------------------------------------------------------
    # Public flush method – train and clear specific agents immediately
    # ------------------------------------------------------------------
    def flush_agents(self, agent_ids: Any) -> None:
        """
        Train + clear rollout buffers for the given agents *right now*.

        Why this exists:
        - The TickEngine may respawn a new agent into a slot that was previously
          occupied by a dead agent.
        - If we wait for the next natural PPO window, the terminal trajectory
          of the dead agent could be overwritten before it is learned from.
        - Flushing the dead slot just before respawn preserves the learning
          signal without altering the core PPO logic.
        """
        if agent_ids is None:
            return

        # Convert input to a plain list of integers (accepts tensors or iterables)
        if isinstance(agent_ids, torch.Tensor):
            aids = agent_ids.detach().to("cpu").to(torch.long).tolist()
        else:
            aids = [int(a) for a in agent_ids]

        if len(aids) == 0:
            return

        # Perform training and clearing for these agents
        self._train_aids_and_clear(aids)

    # ------------------------------------------------------------------
    # Core training routine – shared by window flush and manual flush
    # ------------------------------------------------------------------
    def _train_aids_and_clear(self, aids: List[int]) -> None:
        """
        For each agent in `aids`, if it has collected enough data, run PPO
        updates on its trajectory and then clear its buffer.
        This method contains the full PPO update logic with minibatches,
        value clipping, and optional early stopping based on KL divergence.
        """
        if len(aids) == 0:
            return

        # Defensive check: no optimizer sharing across the selected agents
        self._assert_no_optimizer_sharing(aids)

        for aid in aids:
            aid = int(aid)
            b = self._buf.get(aid, None)
            # Skip agents with empty buffers
            if b is None or not b.obs:
                continue

            model = self.registry.brains[aid]
            if model is None:
                # Model missing – agent probably dead, discard its buffer
                self._buf.pop(aid, None)
                continue

            model.train()
            opt = self._get_opt(aid, model)      # ensures optimizer & scheduler exist

            # Stack all collected tensors into single batch tensors
            obs = torch.stack(b.obs)               # (T, obs_dim)
            act = torch.stack(b.act).long()        # (T,)
            logp_old = torch.stack(b.logp)         # (T,)
            val_old = torch.cat(b.val)             # (T,) – cat because values were stored as (1,)
            rew = torch.stack(b.rew)                # (T,)
            done = torch.stack(b.done).bool()       # (T,)

            # Compute advantages and returns using GAE
            # Use a bootstrap value for the last step if this buffer was closed by a window boundary.
            last_v = b.bootstrap
            adv, ret = self._gae(rew, val_old, done, last_value=last_v)

            # PPO training epochs (shuffle + minibatches)
            B = int(obs.size(0))
            # Determine number of minibatches: at most self.minibatches, but cannot exceed B
            n_mb = max(1, min(int(self.minibatches), B))
            mb_size = max(1, B // n_mb)

            with torch.enable_grad():
                for _ in range(self.epochs):
                    # Shuffle the indices for this epoch
                    idx = torch.randperm(B, device=obs.device)
                    approx_kl_epoch = 0.0

                    # Iterate over minibatches
                    for start_i in range(0, B, mb_size):
                        mb_idx = idx[start_i:start_i + mb_size]
                        if mb_idx.numel() == 0:
                            continue

                        # Gather minibatch data
                        obs_mb = obs[mb_idx]
                        act_mb = act[mb_idx]
                        logp_old_mb = logp_old[mb_idx].detach()
                        val_old_mb = val_old[mb_idx].detach()
                        adv_mb = adv[mb_idx].detach()
                        ret_mb = ret[mb_idx].detach()

                        # Forward pass through the model for this minibatch
                        logits, values, entropy = self._policy_value(model, obs_mb)

                        # New log-probabilities of the actions that were actually taken
                        logp = F.log_softmax(logits, dim=-1).gather(1, act_mb.view(-1, 1)).squeeze(1)

                        # PPO clipped surrogate objective
                        ratio = torch.exp(logp - logp_old_mb)                     # importance sampling ratio
                        surr1 = ratio * adv_mb                                   # unclipped objective
                        surr2 = torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip) * adv_mb  # clipped
                        loss_pi = -torch.min(surr1, surr2).mean()               # policy loss

                        # Clipped value loss (PPO-style trust region for critic)
                        v_pred = values
                        v_clipped = val_old_mb + torch.clamp(v_pred - val_old_mb, -self.clip, self.clip)
                        loss_v1 = (v_pred - ret_mb) ** 2
                        loss_v2 = (v_clipped - ret_mb) ** 2
                        loss_v = torch.max(loss_v1, loss_v2).mean()              # value loss

                        # Entropy bonus (encourage exploration)
                        loss_ent = -entropy.mean()

                        # Total loss
                        loss = loss_pi + self.vf_coef * loss_v + self.ent_coef * loss_ent

                        # Gradient step
                        opt.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), float(self.max_grad_norm))
                        opt.step()

                        # Approx KL for early stopping (optional)
                        approx_kl = (logp_old_mb - logp).mean().item()
                        approx_kl_epoch = max(approx_kl_epoch, approx_kl)

                    # Early stopping if KL exceeds target (only if target_kl > 0)
                    if self.target_kl > 0.0 and approx_kl_epoch > self.target_kl:
                        # Trust region violated; stop remaining epochs for this window.
                        break

            # Step the learning rate scheduler (cosine annealing)
            if aid in self._sched:
                self._sched[aid].step()

            # Clear the buffer – we have trained on this data and it must not be reused
            b.obs.clear()
            b.act.clear()
            b.logp.clear()
            b.val.clear()
            b.rew.clear()
            b.done.clear()
            b.bootstrap = None   # Reset bootstrap field as well

    def _train_window_and_clear(self) -> None:
        """
        Called when the global tick counter reaches a multiple of `self.T`.
        Trains on all agents that currently have buffered data and then clears them.
        """
        if len(self._buf) == 0:
            return
        # Delegate to the common helper
        self._train_aids_and_clear(list(self._buf.keys()))

    # ------------------------------------------------------------------
    # Checkpointing – save/load all runtime state
    # ------------------------------------------------------------------
    def get_checkpoint_state(self) -> Dict[str, Any]:
        """
        Return a portable checkpoint payload:
          - rollout buffers per agent id
          - optimizer/scheduler state_dict per agent id
          - global step counter
        All tensors are moved to CPU for portability.
        """
        def cpuize(x: Any) -> Any:
            """
            Recursively move tensors to CPU.
            - torch.Tensor -> detached CPU tensor
            - dict/list    -> recursively process values/elements
            - others       -> unchanged
            """
            if torch.is_tensor(x):
                return x.detach().to("cpu")
            if isinstance(x, dict):
                return {k: cpuize(v) for k, v in x.items()}
            if isinstance(x, list):
                return [cpuize(v) for v in x]
            return x

        # Buffer data: for each agent, convert each list of tensors to CPU
        buf_out: Dict[int, Any] = {}
        for aid, b in self._buf.items():
            buf_out[int(aid)] = {
                "obs":  cpuize(list(b.obs)),
                "act":  cpuize(list(b.act)),
                "logp": cpuize(list(b.logp)),
                "val":  cpuize(list(b.val)),
                "rew":  cpuize(list(b.rew)),
                "done": cpuize(list(b.done)),
                # Note: bootstrap field is not saved because it is ephemeral;
                # it will be recomputed on the next window boundary.
            }

        # Optimizer state dicts (already contain tensors on whatever device they were on)
        opt_out = {int(aid): cpuize(opt.state_dict()) for aid, opt in self._opt.items()}

        # Scheduler state dicts (mostly integers/floats, but cpuize for safety)
        sched_out = {int(aid): cpuize(s.state_dict()) for aid, s in self._sched.items()}

        return {
            "step":  int(self._step),          # global step counter
            "buf":   buf_out,                    # buffers
            "opt":   opt_out,                     # optimizers
            "sched": sched_out,                    # schedulers
        }

    def load_checkpoint_state(
        self,
        state: Dict[str, Any],
        *,
        registry: Any,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Restore PPO runtime from a checkpoint produced by `get_checkpoint_state()`.

        Args:
            state:    the checkpoint dictionary
            registry: agent registry (needed to access brains/models)
            device:   target device; if None, self.device is used
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

        # Restore global step counter
        self._step = int(state.get("step", 0))

        # --- Restore rollout buffers ---
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
            )

        # --- Restore optimizers and schedulers ---
        opt_in: Dict[int, Any] = state.get("opt", {})
        sched_in: Dict[int, Any] = state.get("sched", {})

        # Clear existing ones
        self._opt.clear()
        self._sched.clear()

        # First restore optimizers (schedulers depend on them)
        for aid, opt_sd in opt_in.items():
            aid_i = int(aid)

            # Get the model for this agent
            model = registry.brains[aid_i]          # may be None if slot empty – that's fine
            if model is None:
                # No model -> cannot recreate optimizer; skip.
                continue

            # Create fresh optimizer and scheduler (they will be stored by _get_opt)
            opt = self._get_opt(aid_i, model)       # this creates both opt and sched

            # Load the saved optimizer state
            opt.load_state_dict(opt_sd)

            # Move all tensors inside the optimizer state to the target device
            for state_group in opt.state.values():
                for k, v in list(state_group.items()):
                    if torch.is_tensor(v):
                        state_group[k] = v.to(dev)

            # (The scheduler will be overwritten later, but we keep it for now)

        # Now restore scheduler states (they must refer to the optimizers we just restored)
        for aid, sch_sd in sched_in.items():
            aid_i = int(aid)
            if aid_i not in self._sched:
                # Scheduler missing – maybe model was None? Skip.
                continue
            sch = self._sched[aid_i]
            sch.load_state_dict(sch_sd)

    # ------------------------------------------------------------------
    # Helper methods for PPO computations
    # ------------------------------------------------------------------
    def _gae(self, rewards: torch.Tensor, values: torch.Tensor,
             dones: torch.Tensor, last_value: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generalized Advantage Estimation.
        Computes advantages and returns (target values) using the GAE formula.
        Args:
            rewards: (T,) tensor of rewards
            values:  (T,) tensor of value estimates
            dones:   (T,) boolean tensor indicating episode termination
            last_value: optional (1,) tensor containing V(s_{T}) for the state after the last step.
                       If not provided, it is assumed to be 0 (no bootstrap).
        Returns:
            advantages: (T,) tensor of advantages
            returns:    (T,) tensor of returns (advantages + values)
        """
        T = rewards.numel()
        adv = torch.zeros_like(rewards)
        last_gae = 0.0

        for t in reversed(range(T)):
            # If episode ended at t, mask = 0, otherwise 1
            mask = 1.0 - float(dones[t].item())

            # Next value: use values[t+1] for interior steps.
            # For the last step, bootstrap from last_value if provided, else assume 0.
            if t < T - 1:
                next_val_t = values[t + 1]
            else:
                # Last step: use the bootstrap value if given, otherwise 0
                next_val_t = float(last_value.item()) if last_value is not None else 0.0

            delta = rewards[t] + self.gamma * next_val_t * mask - values[t]
            last_gae = delta + self.gamma * self.lam * mask * last_gae
            adv[t] = last_gae

        ret = adv + values                # returns = advantages + values

        # Normalise advantages (optional but helps stability)
        if adv.numel() > 1:
            adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

        return adv, ret

    def _policy_value(self, model: nn.Module,
                      obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run the model to obtain logits, values, and the entropy of the policy.
        Args:
            model: the agent's brain (should return (logits, values))
            obs:   (B, obs_dim) observation batch
        Returns:
            logits: (B, act_dim) raw logits
            values: (B,) value estimates
            entropy: (B,) entropy of the policy distribution
        """
        logits, values = model(obs)        # forward pass
        values = values.squeeze(-1)         # ensure (B,) not (B,1)
        logp = F.log_softmax(logits, dim=-1)
        entropy = -(logp.exp() * logp).sum(-1)   # policy entropy
        return logits, values, entropy  