# Learning and Optimization

This document explains the implemented learning stack in the inspected repository and separates code-backed behavior from background PPO theory.

## Purpose

The repository contains actual learning code. It should therefore be documented precisely. The goal here is to explain:

- what is implemented
- how it is wired into the runtime
- what reward terms exist
- what should **not** be inferred from the mere presence of PPO code

## Implemented behavior

### Learning runtime exists and is wired into the tick loop

`TickEngine` conditionally creates `PerAgentPPORuntime` when:

- `PPO_ENABLED` is true
- the runtime module imports successfully

The engine then interacts with PPO at several points:

- after decision-time inference, it updates a slot-local value cache
- after the environment phase, it packages rewards and calls `record_step(...)`
- after tick advance, it flushes dead slots
- after respawn, it resets PPO state for repopulated slots

This is not dead code. It is part of the main inspected runtime path.

### The PPO design is slot-local

This is the most important architectural fact about learning in this repository.

The runtime is designed so that:

- each slot owns its own model parameters
- each slot owns its own optimizer
- each slot owns its own scheduler
- rollout buffers are slot-local
- optimizer state is not shared between slots

The runtime includes explicit checks to forbid optimizer sharing between slots.

This means the current learning path is materially different from a shared-policy MARL design.

### Grouped math does not imply shared learning

The code can batch compatible slots for forward passes and some grouped PPO math, but those grouped operations are only computational grouping. They do **not** introduce shared parameters or a shared optimizer.

## PPO data flow

### Decision-time collection

During each tick:

1. the engine builds observations for alive slots
2. it runs bucketed forward inference
3. it samples actions from masked logits
4. it records observations, logits, values, actions, masks, and team ids for PPO

### Reward packaging

After combat, movement, and environment effects, the engine computes a final per-slot reward for the acted slots. It then calls:

```python
self._ppo.record_step(...)
```

with:

- observations
- old logits
- old values
- sampled actions
- final reward
- done flag
- action masks
- bootstrap values handled via the slot-local cache path

### Window completion and training

The runtime accumulates a rollout window of length `PPO_WINDOW_TICKS`. When the window boundary is reached, it prepares slot-local training payloads, computes returns and advantages, and runs PPO optimization.

### Dead-slot flushing and respawn reset

Dead slots are explicitly flushed before respawn. After respawn, slot-local PPO state is reset for slots that changed occupant.

This is an important safeguard in a slot-based world where physical slot identity and biological/lineage identity are not the same thing.

## Reward wiring in the current implementation

The engine maintains several reward components. The practical reward seen by PPO depends on config.

### Individual reward channels implemented in code

These channels are present in the tick engine:

- individual kill reward
- individual damage-dealt shaping
- individual damage-taken penalty
- individual contested-control-point reward
- individual healing-recovery reward

### Team-level reward channels implemented in code

These channels are also present in PPO reward assembly:

- team kill reward
- team death reward
- team control-point reward

### HP-based shaping term

The engine also computes an HP-based shaping term:

- base term: current HP × `PPO_REWARD_HP_TICK`
- optional `threshold_ramp` mode

### Important default-config implication

The inspected default config sets:

- `PPO_HP_REWARD_MODE = "threshold_ramp"`
- `PPO_HP_REWARD_THRESHOLD = 10`

The engine clamps the threshold into `[0, 1]`. A default of `10` therefore clamps to `1.0`, which makes the threshold-ramp branch produce zero HP reward unless the threshold is overridden to a value below `1.0`.

That is a code-backed operational detail, not an interpretation.

### Team score and PPO reward are not identical objects

`simulation/stats.py` maintains cumulative team score using configured team reward weights. PPO reward packaging in `TickEngine.run_tick()` builds a slot-local reward tensor for acted slots. These are related but distinct accounting paths.

## Implemented PPO ingredients

The runtime docstring and code show the following implemented ingredients:

- clipped policy objective
- clipped value loss
- entropy term
- minibatches
- gradient clipping
- optional target-KL early stop
- GAE
- cosine annealing learning-rate scheduler
- checkpoint save and restore for PPO state

## Background formula: PPO objective

The standard clipped PPO surrogate can be written as

$$
L_{\text{clip}}(\theta)
=
\mathbb{E}_t
\left[
\min\left(
r_t(\theta) A_t,\,
\operatorname{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon) A_t
\right)
\right],
$$

where

$$
r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}.
$$

This background equation is included because it matches the implemented runtime structure. It should not be read as a promise that every surrounding design choice is textbook-identical.

## Background formula: return and advantage path

The runtime computes generalized advantage estimates and returns from recorded rollout tensors.

At a high level:

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t),
$$

and

$$
A_t = \delta_t + \gamma \lambda \delta_{t+1} + \gamma^2 \lambda^2 \delta_{t+2} + \cdots
$$

with return

$$
R_t = A_t + V(s_t).
$$

Again, this is background aligned to the implemented runtime, not a claim that every external paper convention is reproduced exactly.

## Implemented deviations and repository-specific design choices

### 1. No shared policy

This is the most substantial deviation from common MARL practice.

### 2. Slot-local bootstrap cache

The runtime uses a value-cache mechanism so that ordinary decision-time inference can provide the bootstrap information needed at rollout boundaries. This avoids an extra post-step inference pass.

### 3. Bucketed grouped execution

The code groups structurally compatible models for computational efficiency, but optimizer ownership remains slot-local.

### 4. Action masks are first-class PPO inputs

The runtime stores action masks per step, which matters because actions are not simply unconstrained categorical outputs.

## What should not be inferred

The following claims are **not** warranted solely from the inspected code:

- that PPO is converging
- that emergence has occurred
- that the reward design is well-shaped for the intended behaviors
- that the current defaults are tuned
- that per-slot PPO is computationally efficient enough for large-scale runs
- that the repository has already been experimentally validated

Those are empirical questions. The code only shows that the machinery is present.

## Operational cautions

### Dense reward terms are config-sensitive

Several reward channels default to zero, while others default to nonzero values. A small config change can materially alter the learning signal.

### Per-slot PPO is heavy

The architecture is expressive, but it is also expensive. Every slot can carry its own model, optimizer, scheduler, buffer, and checkpoint state.

### Schema changes invalidate policy-bearing checkpoints

Observation-schema compatibility is enforced before policy state is restored.

## Related documents

- [Agents, observations, and actions](06-agents-observations-actions.md)
- [Mathematical foundations](12-mathematical-foundations.md)
- [Limitations, validation, and open questions](14-limitations-validation-and-open-questions.md)
