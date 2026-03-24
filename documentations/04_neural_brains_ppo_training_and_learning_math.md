# Volume 4 — Neural brains, PPO training, and learning math

## Document Purpose

This volume explains the learning subsystem of the repository as it exists in the verified codebase.

Its job is to answer, precisely and operationally, the following question:

**How does this codebase turn observations into actions, collect rollout data, compute PPO training targets, update per-agent neural policies, and preserve learning state across checkpoint/resume cycles?**

This document is intentionally narrower than a general reinforcement-learning textbook. It does not try to teach every variant of actor-critic learning. It explains:

- the active neural brain family implemented in `agent/mlp_brain.py`,
- the observation-to-action path driven by `engine/tick.py`,
- the slot-local PPO runtime in `rl/ppo_runtime.py`,
- the persistence and resume logic in `utils/checkpointing.py`,
- the places where the code is mathematically central,
- the places where the code is mostly engineering scaffolding,
- the invariants that must remain true if the learning system is modified.

It also separates three things that are often blurred together:

1. **general PPO theory,**
2. **what this repository actually implements,**
3. **what the code comments or legacy names suggest but the active runtime does not currently do.**

That separation matters. This codebase contains some legacy naming such as `_build_transformer_obs`, and some comments that refer to older architecture ideas. The active creation, inference, training, checkpoint, and restore paths verified here are built around the **MLP brain family**, not an active transformer policy path.

---

## Audience and How to Read This Volume

This volume is written for two audiences at the same time.

The first audience is a serious beginner who can read code slowly but cannot yet reconstruct a full actor-critic pipeline from raw source. That reader should read this document in order.

The second audience is an experienced engineer who wants a truthful map of the learning system so they can audit it, modify it, or debug training. That reader can skim section headers, tables, and appendices first, then return to the sections on target computation, loss construction, and persistence integrity.

The most important sections are:

- **Section 3** for the forward and inference path,
- **Section 5** for reward flow and rollout collection,
- **Section 6** for returns and advantages,
- **Section 7** for the PPO objective actually optimized here,
- **Section 10** for checkpoint/resume integrity,
- **Section 11** for modification safety.

---

## 1. Why the neural learning layer exists

### 1.1 Plain-language intuition

The simulation is not driven by a fixed rules engine that directly chooses one hard-coded action for each agent. Instead, the environment computes a structured observation for each alive agent, and a neural policy converts that observation into a probability distribution over discrete actions.

That design exists because the system wants agents to react to:

- local ray-based perception,
- non-ray scalar context,
- team and world conditions,
- capture-point and healing-zone context,
- their own learned policy parameters.

A rules-only controller would require the developer to hand-write a large number of coupled heuristics for movement, combat, and situational tradeoffs. The neural layer allows the action policy to be learned from reward signals rather than fully authored in advance.

### 1.2 Formal technical explanation

At runtime, the simulation performs the following broad loop:

1. identify alive slots,
2. build a fixed-width observation tensor for those slots,
3. build a legal-action mask,
4. run each slot’s brain to get action logits and a value estimate,
5. sample one action per slot from the masked categorical policy,
6. advance the environment,
7. compute rewards and terminal status,
8. record the resulting transition into a PPO rollout buffer,
9. periodically train the policy and value networks from those stored transitions.

The policy network and value network are not separate top-level objects in this codebase. Each brain is an **actor-critic module** that emits both:

- policy logits for the discrete action distribution, and
- a scalar value estimate for the same observation.

### 1.3 What role PPO plays

PPO is the gradient-based learning algorithm used for the active neural policies. It is not the entire learning story, because the codebase also has respawn-time cloning and mutation logic. However, PPO is the component that performs repeated gradient updates based on stored trajectories.

There are therefore **two distinct mechanisms by which policy parameters can change over time**:

1. **within-lifetime gradient updates** through PPO, and
2. **respawn-time inheritance/mutation** through the respawn subsystem.

The PPO runtime lives in `rl/ppo_runtime.py`. The respawn/evolution layer lives primarily in `engine/respawn.py`.

### 1.4 Relationship to the observation/action interface

This volume assumes the reader has already seen the observation and action contract, but it restates the verified facts needed for training analysis:

- `config.OBS_DIM = RAY_TOKEN_COUNT * RAY_FEAT_DIM + RICH_TOTAL_DIM`
- resolved default values in the active config are:
  - `RAY_TOKEN_COUNT = 32`
  - `RAY_FEAT_DIM = 8`
  - `RICH_BASE_DIM = 23`
  - `INSTINCT_DIM = 4`
  - `OBS_DIM = 283`
- `config.NUM_ACTIONS = 41`

Those are not cosmetic choices. The code treats them as schema contracts. Changing them affects:

- neural module shapes,
- checkpoint compatibility,
- PPO buffer semantics,
- action-mask shape,
- rollout log-prob computation,
- inference-time and training-time masking.

### 1.5 Design consequence

The learning layer is not loosely attached to the simulation. It is tightly coupled to:

- observation layout,
- action count,
- slot identity,
- respawn semantics,
- checkpoint format.

That tight coupling is why so much of the code is defensive and full of explicit shape checks.

---

## 2. Brain and policy module overview

## 2.1 Verified module map

The active learning stack is spread across a small set of files:

| Path | Role |
| --- | --- |
| `agent/mlp_brain.py` | Defines the actor-critic brain family and shared input preprocessing |
| `agent/obs_spec.py` | Authoritative observation splitting for the MLP path |
| `agent/ensemble.py` | Bucketed multi-model inference helper for independent brains |
| `engine/agent_registry.py` | Stores per-slot brain modules and architecture metadata |
| `engine/tick.py` | Builds observations, samples actions, computes rewards, records PPO transitions |
| `rl/ppo_runtime.py` | Slot-local PPO rollout storage, GAE, loss construction, optimizer/scheduler state, checkpointable PPO state |
| `engine/respawn.py` | Creates new brains, clones parent brains, perturbs parameters, resets slot-local PPO state on reuse |
| `utils/checkpointing.py` | Saves and restores brain weights, PPO runtime state, RNG state, and runtime metadata |

## 2.2 What the active brain actually is

The active brain family is an **MLP actor-critic family**.

The factory function `create_mlp_brain(kind, obs_dim, act_dim)` constructs one of five classes:

- `WhisperingAbyssBrain`
- `VeilOfEchoesBrain`
- `CathedralOfAshBrain`
- `DreamerInBlackFogBrain`
- `ObsidianPulseBrain`

Every one of these classes inherits from `_BaseMLPBrain`, which enforces the same interface:

```python
(obs: Tensor[B, obs_dim]) -> (logits: Tensor[B, act_dim], value: Tensor[B, 1])
```

The trunk architecture changes by variant, but the input contract and head contract remain the same.

## 2.3 Important clarification: active MLP path versus legacy names

Some code and comments still contain legacy names such as:

- `_build_transformer_obs`
- older comments in respawn docstrings that mention transformer-like paths

However, the verified active creation and restore path is this:

- `engine/respawn.py::_new_brain(...) -> create_mlp_brain(...)`
- `engine/respawn.py::_clone_brain(...) -> create_mlp_brain(...)`
- `utils/checkpointing.py::_make_brain(...) -> create_mlp_brain(...)`

That means the runtime, respawn, and checkpoint restore paths all reconstruct **MLP brains**.

A reader should therefore not infer the existence of an active recurrent or transformer PPO path unless they separately verify it in code. This volume does not claim such a path exists, because the verified active path is MLP-based.

## 2.4 Shared actor-critic structure

All three brain variants share the same overall pattern:

```text
flat observation (default self-centric: 276; legacy full: 283)
    -> split into ray block and rich block
    -> encode rays with the ray tower
    -> pool rays into one summary vector
    -> encode the rich tail with the scalar tower
    -> concatenate the two summaries
    -> pass through variant-specific fusion trunk
    -> actor head -> logits over 41 actions
    -> critic head -> scalar value
```

The actor and critic are therefore **shared-body, separate-head** modules:

- shared preprocessing,
- shared trunk,
- separate `actor_head`,
- separate `critic_head`.

This is not a separate actor network and a separate critic network with independent backbones. It is one actor-critic module per slot.

## 2.5 Brain variants

The resolved default tower widths are:

- `BRAIN_MLP_RAY_WIDTH = 96`
- `BRAIN_MLP_SCALAR_WIDTH = 96`
- `BRAIN_MLP_FUSION_WIDTH = 128`
- final fusion input width = `ray_width + scalar_width = 192`

Using those defaults, the verified variants are:

| Brain kind | Trunk shape | Notes |
| --- | --- | --- |
| `throne_of_ashen_dreams` | `ray(8 -> 96) + scalar(20 -> 96) -> 192 -> 3 fusion residual blocks -> 128` | clean late-fusion baseline |
| `veil_of_the_hollow_crown` | `same towers -> 192 -> 3 fusion residual blocks with scalar reinjection -> 128` | preserves scalar context through the trunk |
| `black_grail_of_nightfire` | `same towers -> scalar-gated ray tokens -> 192 -> 3 gated fusion blocks -> 128` | lets scalar context modulate perception and fusion |

All variants terminate in:

- `actor_head: Linear(fusion_width, act_dim)`
- `critic_head: Linear(fusion_width, 1)`

## 2.6 Shared preprocessing before the trunk

The brain does not feed the raw flattened observation directly into the trunk.

Instead `_BaseMLPBrain` performs a structured conversion:

1. split the observation with `obs_spec.split_obs_for_mlp(obs)`,
2. reshape the ray block from `(B, 256)` to `(B, 32, 8)`,
3. run the ray tower over all 32 ray tokens,
4. pool the encoded rays into one summary vector (default `mean_max` pooling),
5. normalize and project the rich tail into the scalar tower,
6. run the scalar tower over the full non-ray tail,
7. concatenate `[ray_summary, scalar_summary]` to obtain `(B, 192)` under the default widths,
8. run the variant-specific fusion stack,
9. emit actor logits and a scalar critic value.

Under the default self-centric schema the rich tail is `(B, 20)`. The legacy full schema still exists for compatibility and uses `(B, 27)`, but the active default runtime path is the self-centric 276-wide observation.

## 2.7 Weight initialization

The code initializes all linear layers orthogonally, then overrides the output heads:

- hidden layers: orthogonal with gain `sqrt(2)`
- actor head: orthogonal with gain `BRAIN_MLP_ACTOR_INIT_GAIN` (default `0.01`)
- critic head: orthogonal with gain `BRAIN_MLP_CRITIC_INIT_GAIN` (default `1.0`)
- biases: zeros

This matters because the actor head is intentionally initialized with a much smaller gain than the critic head. Operationally, that means the initial logits are kept relatively modest, which reduces overly sharp action probabilities at startup.

## 2.8 What a newcomer should understand first

A newcomer should understand these three facts before anything else:

1. **each slot owns its own brain instance,**
2. **each brain emits both logits and value,**
3. **all variants obey the same interface, so PPO and inference code do not care which concrete variant produced the outputs.**

That uniform contract is what makes the bucketed inference and grouped PPO training code possible.

---

## 3. Forward pass and inference flow

## 3.1 The runtime path at a glance

The decision path inside `engine/tick.py` can be summarized as:

```text
alive slots
    -> positions
    -> observation tensor via _build_transformer_obs(...)
    -> legal-action mask via build_mask(...)
    -> bucket alive slots by architecture
    -> run ensemble_forward(bucket.models, bucket_obs)
    -> mask illegal logits
    -> sample discrete action from Categorical(logits=masked_logits)
    -> write actions back into alive-slot order
```

This is the inference path. PPO training is a separate path that reuses the same model contract later.

## 3.2 Observation creation

`TickEngine._build_transformer_obs(alive_idx, pos_xy)` builds the per-agent observation tensor.

Despite the legacy function name, the verified output is a flat tensor of shape:

```text
(N_alive, OBS_DIM)
```

The observation contains two main parts:

- a raycast block of size `32 * 8 = 256`,
- a rich block of size `27`.

The rich block includes normalized health, normalized position, team flags, unit flags, normalized attack and vision values, zone flags, normalized global stats, and a four-dimensional instinct context.

The code checks the resulting shape explicitly and raises if it does not match `(N_alive, config.OBS_DIM)`.

## 3.3 Legal-action masking

Immediately after building observations, `TickEngine` builds a legal-action mask:

```text
mask.shape == (N_alive, NUM_ACTIONS)
```

The convention is:

- `mask[i, a] == True` means action `a` is legal for agent `i`,
- illegal actions must not be sampled,
- the same legality semantics must later be reused during PPO training.

This point is mathematically important. The policy is not defined here as a categorical distribution over all 41 actions unconditionally. It is a categorical distribution over the subset of **currently legal** actions, represented by masking the logits.

## 3.4 Architecture bucketing and why it exists

Agents can have different concrete brain variants. Running them naively in a Python loop would work, but it would be slow.

The registry therefore groups alive agents into **buckets** whose brains share the same architecture signature. `engine/agent_registry.py::build_buckets` uses persistent architecture metadata `brain_arch_ids` to do this efficiently.

A bucket contains:

- `indices`: the slot ids in that bucket,
- `models`: the corresponding brain modules,
- `locs`: where those slot ids sit inside the current `alive_idx` order.

This grouping does **not** mean the parameters are shared. It only means the shapes are compatible for batched inference optimization.

## 3.5 `ensemble_forward` and independent-model batched inference

`agent/ensemble.py::ensemble_forward(models, obs)` is the public inference helper.

It has two implementations:

1. a safe sequential loop, `_ensemble_forward_loop`,
2. an optional `torch.func`/`vmap` path, `_ensemble_forward_vmap`.

The function chooses the `vmap` path only if:

- `config.USE_VMAP` is enabled,
- the bucket size is large enough,
- `torch.func` APIs are available,
- the models are not TorchScript modules,
- the path succeeds without exception.

If any condition fails, it falls back to the loop.

### Important design point

The `vmap` path is **not** parameter sharing. It stacks parameter sets for independent models and evaluates them together. Each row still belongs to a different model instance.

This distinction matters throughout the codebase:

- **batched execution** is an efficiency technique,
- **shared parameters** would be a learning design choice.

This repository uses the first, not the second.

## 3.6 Brain forward pass inside the bucket

For each model, the forward contract is:

```text
obs:    (B, obs_dim)
logits: (B, act_dim)
value:  (B, 1)
```

`ensemble_forward` normalizes the model contract so the bucket-level outputs become:

- `dist.logits`: `(K, A)` where `K` is bucket size and `A = act_dim`,
- `values`: `(K,)`

Internally, `_DistWrap` exists only to give the result a `.logits` attribute. The active MLP brains themselves return raw logits tensors, not a full `torch.distributions.Categorical` object.

## 3.7 Action selection

Once `TickEngine` has `dist.logits` and the legal-action mask for a bucket, it applies masking:

```python
logits32 = torch.where(bucket_mask, dist.logits.to(torch.float32), neg_inf)
```

where `neg_inf` is `torch.finfo(torch.float32).min`.

Then it samples:

```python
a = torch.distributions.Categorical(logits=logits32).sample()
```

### Meaning

This means the runtime action policy is:

- **stochastic**, not greedy,
- defined by masked logits,
- sampled from a categorical distribution.

The active decision path is therefore **not** `argmax` action selection. The only exceptions are debug overrides such as forced actions.

## 3.8 Plain-language explanation of logits

A logit is an unnormalized preference score. It is not yet a probability.

If the legal actions for one agent are `[a0, a4, a8]`, then the model produces scores for all 41 actions, but the mask forces illegal actions to an extremely negative value. After softmax, those illegal actions receive effectively zero probability, and the remaining legal actions form the valid categorical distribution.

So the forward pass does **not** directly output “the chosen action.” It outputs scores from which a distribution is formed.

## 3.9 Formal data-flow summary

The end-to-end inference path for one alive agent is:

```text
s_t
  -> obs_t in R^283
  -> policy/value brain
  -> logits_t in R^41, value_t in R
  -> legality mask m_t in {0,1}^41
  -> masked logits
  -> π(a|s_t, m_t)
  -> sampled action a_t
```

At the same time, the critic head produces `V(s_t)` for the same observation. That value is not used to choose the action directly. It is stored for learning.

## 3.10 Important design consequences

1. **The action mask is part of the policy semantics.**
   If rollout uses one masking rule and PPO training uses another, log-prob ratios become invalid.

2. **The logits are not probabilities.**
   Any code that treats raw logits as action probabilities is wrong.

3. **Sampling is intentional.**
   The runtime depends on stochastic action selection for exploration and for PPO’s on-policy data collection.

4. **The critic is evaluated on the same observation that produced the action.**
   This is a standard actor-critic pattern, but it matters because bootstrap timing later depends on these values.

## 3.11 Common failure modes in this section

- changing `NUM_ACTIONS` without updating masks, heads, and buffer expectations,
- changing observation width without updating brain constructors and checkpoint compatibility,
- replacing sampling with `argmax` during rollout while still training with PPO as if the behavior policy were stochastic,
- changing legal-action rules in rollout but not in training-time masking.

---

## 4. Policy, value, and distribution semantics

## 4.1 What the policy output means

For each observation `s_t`, the actor head produces a vector of logits:

\[
z_t \in \mathbb{R}^{A}
\]

where `A = NUM_ACTIONS = 41`.

After applying the legality mask, these logits define a masked categorical policy:

\[
\pi_\theta(a \mid s_t, m_t) = \text{softmax}(z_t^{\text{masked}})_a
\]

where `m_t` is the action mask at time `t`.

### Plain-language meaning

The policy head answers:

> “Given what this agent currently sees, how strongly does the model prefer each legal action?”

It does **not** answer:

> “What reward will happen next?”
That is the critic’s job.

## 4.2 What the value output means

For each observation `s_t`, the critic head produces a scalar:

\[
V_\theta(s_t)
\]

Operationally, this is the model’s estimate of the discounted future return from the current state, under the current policy and current environment/reward design.

In this codebase the critic output is stored as shape `(B, 1)` at the brain boundary, then squeezed to `(B,)` in PPO helpers and bucketed inference.

### Plain-language meaning

The critic is a baseline estimator. It tries to predict how good the current state is before the actual future unfolds.

It is not:

- the action probability,
- the immediate reward,
- the Q-value of each action,
- a centralized team value.

It is a single scalar state-value estimate for that slot’s current observation.

## 4.3 Why both outputs are needed

PPO here is an **actor-critic** method.

The actor needs to answer:

- which actions should become more likely or less likely?

The critic needs to answer:

- how good was the state expected to be?

The gap between observed outcome and predicted value becomes the raw material for the advantage estimate.

## 4.4 Log-probabilities and why they matter

During rollout recording, the runtime computes the log-probability of the chosen action under the masked behavior policy:

```python
logp_a = F.log_softmax(logits_masked.to(torch.float32), dim=-1).gather(1, actions.view(-1, 1)).squeeze(1)
```

For PPO, the important stored quantity is **not the whole old distribution**, but the chosen-action log-probability under the old policy:

\[
\log \pi_{\theta_{\text{old}}}(a_t \mid s_t)
\]

Later, during training, the model is run again on the stored observation to produce a new log-probability:

\[
\log \pi_{\theta}(a_t \mid s_t)
\]

and the probability ratio is formed from their difference.

This is enough because PPO’s clipped surrogate loss only needs the ratio for the sampled action.

## 4.5 Entropy

The PPO helper computes entropy as:

\[
H(\pi) = -\sum_a p(a)\log p(a)
\]

using the masked logits after `log_softmax`.

### Interpretation

- high entropy means the policy is more spread out and uncertain,
- low entropy means the policy is more concentrated.

In this repository, entropy is used as an exploration-supporting regularizer in the total loss.

## 4.6 Distribution semantics in this repository versus general RL theory

General RL theory allows many policy types:

- Gaussian policies,
- Beta policies,
- autoregressive policies,
- recurrent policies,
- masked or unmasked categorical policies.

This repository’s verified implementation is much narrower:

- discrete action space,
- categorical policy,
- legality masking applied both during rollout and during PPO recomputation,
- scalar state-value critic,
- no verified recurrent hidden state,
- no verified centralized critic,
- no verified multi-action factorization.

That narrowness is a strength when documenting the system, because it lets the contracts stay explicit.

## 4.7 One subtle but critical point: the mask changes the distribution

A common beginner mistake is to think the action mask is just an external rejection rule layered on top of the policy. That is not what this code does.

The code changes the logits **before** softmax and log-prob computation. Therefore the mask changes the actual policy distribution:

\[
\pi_\theta(a \mid s_t, m_t) \neq \pi_\theta(a \mid s_t)
\]

If you change the mask, you have changed the policy that PPO is optimizing.

---

## 5. Reward flow and rollout collection

## 5.1 What a rollout means here

In this repository, a PPO rollout is not one global tensor that stores every agent’s experience in a single shared buffer.

Instead, the PPO runtime stores a **slot-local trajectory buffer** per registry slot.

That buffer type is `_Buf` in `rl/ppo_runtime.py`, with fields:

| Field | Meaning |
| --- | --- |
| `obs` | stored observation `s_t` |
| `act` | sampled discrete action `a_t` |
| `logp` | old chosen-action log-probability |
| `val` | old critic value `V(s_t)` |
| `rew` | reward assigned to that transition |
| `done` | terminal flag after the tick |
| `act_mask` | legal-action mask used at rollout time |
| `bootstrap` | optional `V(s_{t+1})` for final step of the window |

So a rollout window is a stack of per-slot transitions collected over time until the PPO window boundary or an earlier flush event.

## 5.2 Why the buffers are slot-local

The runtime is explicitly designed around **per-slot independence**:

- separate model per slot,
- separate optimizer per slot,
- separate scheduler per slot,
- separate rollout buffer per slot.

This is important because registry slots can be reused after death and respawn. The runtime therefore treats slot identity as the ownership key for learning state, not a persistent universal “agent object.”

That is why `reset_agent()` and `flush_agents()` exist.

## 5.3 Where rollout collection happens

Rollout collection is done by `PerAgentPPORuntime.record_step(...)`, which is called from `TickEngine.run_tick(...)` after action execution, reward construction, and terminal-status determination.

The tick engine passes:

- slot ids,
- observations captured at decision time,
- masked logits,
- value estimates,
- chosen actions,
- final rewards,
- done flags,
- action masks,
- `bootstrap_values=None` in the active path.

`record_step()` is decorated with `@torch.no_grad()`, which is the right design here. The simulation step is used to collect data, not to retain a giant autograd graph across environment time.

## 5.4 Reward construction in the tick engine

The reward stream used by PPO is constructed in `engine/tick.py`. The verified final reward recorded for PPO is:

\[
r_t^{\text{final}} = r_t^{\text{individual}} + r_t^{\text{team}} + r_t^{\text{hp}}
\]

where:

### Individual reward components

The tick engine accumulates slot-local individual components such as:

- kill reward,
- damage dealt shaping,
- damage taken penalty,
- contested control-point reward,
- healing recovered reward.

These are accumulated into tensors such as:

- `reward_kill_individual`
- `reward_damage_dealt_individual`
- `reward_damage_taken_penalty`
- `reward_contested_cp_individual`
- `reward_healing_recovered`

### Team reward components

The code also constructs team-scoped terms and assigns them to each acting slot based on team identity:

- `team_kill_reward`
- `team_death_reward`
- `team_cp_reward`

A subtle point matters here: `PPO_REWARD_DEATH` multiplies own-team death counts into the reward stream. Therefore whether that term acts as a penalty or reward depends on the sign of the configured coefficient.

### HP shaping

A dense HP term is added as either:

- raw linear current-HP shaping, or
- threshold-ramped shaping if `PPO_HP_REWARD_MODE == "threshold_ramp"`.

## 5.5 Important separation: reward logic is in the environment, not in PPO

PPO itself does not define the reward function. PPO only consumes the rewards it is given.

The reward design here is embedded in `engine/tick.py`. Therefore when someone says “PPO is learning from reward,” the precise implementation truth is:

- the simulation computes reward components after the tick’s effects,
- those scalar results are handed into the PPO runtime,
- PPO uses those scalars to compute returns and advantages.

That separation matters when debugging. Poor learning can come from:

- bad PPO hyperparameters,
- incorrect log-prob bookkeeping,
- or simply a poor reward definition.

## 5.6 How `record_step()` stores the transition

Inside `record_step()` the runtime:

1. validates shapes and devices,
2. defaults the action mask to all-legal only if none was passed,
3. optionally validates that each chosen action is legal under the stored mask,
4. recomputes the chosen-action log-probability from the masked logits,
5. batch-detaches and casts rollout tensors,
6. appends each row into the corresponding slot-local buffer.

A critical detail is that the buffer stores **old chosen-action log-probabilities**, not full old logits.

This is enough for PPO’s ratio term and reduces storage.

## 5.7 When rollout collection triggers training

The runtime maintains a global step counter `self._step` inside `PerAgentPPORuntime`. It increments once per `record_step()` call.

Training is triggered when:

```python
self._step % self.T == 0
```

where `self.T = config.PPO_WINDOW_TICKS`.

In the active tick engine, `record_step()` is called once per tick when there are recorded agents, so the PPO window is effectively driven by decision ticks, not by per-slot episode boundaries.

## 5.8 Terminal flush before respawn

This is one of the most important engineering details in the entire learning system.

At the end of the tick, before respawn, the engine does:

1. identify dead slots,
2. `self._ppo.flush_agents(dead_slots)`,
3. respawn dead units,
4. `self._ppo_reset_on_respawn(was_dead)`.

Why?

Because the slot id is the key for buffer, optimizer, scheduler, and value-cache ownership. If a dead slot were immediately reused without flushing/resetting, the new occupant could inherit stale rollout state and stale optimizer moments from the previous occupant.

That would silently corrupt learning.

## 5.9 Respawn and brain inheritance

The respawn subsystem may:

- create a fresh brain,
- clone a parent brain,
- perturb parameters with Gaussian noise.

So the policy parameters of a newly spawned agent may come from:

- a fresh initialization,
- inherited parent weights,
- inherited weights plus mutation noise.

But the PPO runtime still resets slot-local optimizer, scheduler, buffer, and value-cache state when the slot is reused. That means **weights can be inherited while optimizer moments are not**.

This is a very important design choice. It separates:

- inherited policy parameters,
- from inherited training-state momentum.

---

## 6. Returns, advantages, and training targets

## 6.1 Why raw rewards are not enough

A raw reward `r_t` tells you what happened immediately after one action.

But PPO does not optimize only immediate reward. It wants to estimate how good the action was relative to the long-term outcome and relative to the critic’s baseline.

That is why the runtime computes:

- returns,
- advantages.

## 6.2 The verified target computation path

When a slot is prepared for training in `_prepare_train_slot(...)`, the runtime stacks its buffer into tensors:

- `obs`
- `act`
- `logp_old`
- `val_old`
- `rew`
- `done`
- `act_mask`

Then it takes the stored bootstrap value `b.bootstrap` if present and computes:

```python
adv, ret = self._gae(rew, val_old, done, last_value=last_v)
```

So the central target computation happens in `_gae(...)`.

## 6.3 Formal definitions used by the code

The code implements Generalized Advantage Estimation (GAE) with:

- discount factor `gamma = config.PPO_GAMMA`
- trace parameter `lambda = config.PPO_LAMBDA`

The TD residual is:

\[
\delta_t = r_t + \gamma V(s_{t+1})(1 - d_t) - V(s_t)
\]

where `d_t` is the done flag treated as `1` on terminal transitions.

The recursive GAE estimate is:

\[
A_t = \delta_t + \gamma \lambda (1-d_t) A_{t+1}
\]

and the return target is:

\[
R_t = A_t + V(s_t)
\]

## 6.4 What `_gae(...)` actually does

The verified implementation does the following:

1. casts rewards and values to `float32`,
2. casts dones to boolean,
3. initializes `adv` as zeros,
4. initializes `last_gae` as scalar zero,
5. sets `last_value_t` to the provided bootstrap value or zero,
6. iterates backward over the trajectory,
7. uses `values[t+1]` for interior next-state values,
8. uses `last_value_t` on the final step,
9. zeros future contributions when `done[t]` is true,
10. computes `ret = adv + values32`,
11. normalizes `adv` to zero mean and unit-ish variance if length > 1.

### Important subtlety

The code computes:

```python
ret = adv + values32
```

**before** normalizing advantages.

That means:

- the value target `ret` is based on the unnormalized GAE result,
- only the policy-side advantage is normalized afterward.

This is the correct and common pattern. A frequent beginner error is to assume the normalized advantage is also the value target correction term. It is not.

## 6.5 Plain-language explanation of advantage

The advantage estimates whether the chosen action turned out better or worse than the critic expected.

- positive advantage means “better than baseline,”
- negative advantage means “worse than baseline.”

In PPO, the policy loss uses advantage to decide whether to push the action probability up or down.

## 6.6 Bootstrap semantics

The final step of a rollout window needs a “next value” term:

\[
V(s_{T})
\]

for the state after the last recorded action in that window.

This codebase supports two conceptual ways to obtain it:

1. an explicit `bootstrap_values` argument to `record_step(...)`,
2. the active deferred cached-bootstrap path.

The active tick engine uses the second.

## 6.7 The deferred cached-bootstrap design

This is one of the most interesting engineering decisions in the repository.

### General problem

Suppose a PPO window ends on tick `t`. To compute the final GAE term for the last recorded transition, you need `V(s_{t+1})` for surviving agents.

A naive approach is to run an extra forward pass after the step just to get these bootstrap values.

### What this repository does instead

The runtime keeps a persistent slot-local value cache:

- `_value_cache[slot]`
- `_value_cache_valid[slot]`

This cache is updated from the **normal main inference pass** each tick.

At a window boundary, if no explicit bootstrap is supplied, `record_step()` stores:

- the slot ids of the boundary batch,
- the done flags of the boundary batch,

into:

- `_pending_window_agent_ids`
- `_pending_window_done`

Then, on the next tick:

1. the normal main forward pass computes values for currently alive slots,
2. `update_value_cache(...)` updates the cache,
3. `finalize_pending_window_from_cache()` copies cached values into the boundary buffers for surviving slots,
4. done slots receive bootstrap zero,
5. `_train_window_and_clear()` is called.

### Why this matters

This design removes a duplicate bootstrap-only forward pass from the hot path.

Mathematically, it still supplies the needed `V(s_{t+1})`. Engineering-wise, it avoids extra inference work.

## 6.8 Boundary conditions and dead agents

If a slot is terminal on the final step of the window, the code uses zero bootstrap for that slot.

That is correct because the done mask already prevents the future-value term from contributing:

\[
(1 - d_t) = 0
\]

for terminal transitions.

The runtime also invalidates cached values for flushed/reset slots so reused slots cannot accidentally bootstrap from the wrong occupant’s cached value.

## 6.9 What assumptions the target computation makes

The GAE computation assumes:

1. `rew[t]`, `val[t]`, `done[t]`, `act[t]`, and `logp[t]` are aligned for the same slot and same time step,
2. the stored `done[t]` reflects whether that transition ended the slot’s trajectory,
3. `last_value` corresponds to the state immediately after the final recorded action for the same slot occupant,
4. slot reuse does not occur before flush/reset logic has protected the buffer.

If any of these assumptions is violated, the PPO targets become mathematically wrong even if tensor shapes still look valid.

## 6.10 Common misunderstandings corrected

### Misunderstanding 1
“Advantage is just reward.”

No. Advantage is a discounted, baseline-corrected estimate built from rewards, values, done flags, and possibly a bootstrap value.

### Misunderstanding 2
“Return means cumulative reward only.”

In this code, the return target used for value regression is:

\[
R_t = A_t + V(s_t)
\]

where `A_t` is GAE-derived, not merely a raw sum of future rewards.

### Misunderstanding 3
“If the trajectory window ends, future value is zero.”

Only for terminal transitions. For surviving slots at a nonterminal window boundary, the runtime bootstraps from the next state value.

---

## 7. PPO objective and loss construction

## 7.1 The central PPO ratio idea

For the chosen action at time `t`, PPO compares:

- the new policy probability,
- the old policy probability recorded during rollout.

The code forms the ratio using log-prob differences:

\[
r_t(\theta) = \exp\left(\log \pi_\theta(a_t \mid s_t) - \log \pi_{\theta_{\text{old}}}(a_t \mid s_t)\right)
\]

This is computed in both the sequential and grouped training paths.

## 7.2 Why the ratio matters

If `r_t > 1`, the new policy makes the chosen action more likely than before.
If `r_t < 1`, it makes it less likely than before.

Advantage then determines whether that direction is good:

- positive `A_t`: increasing probability is desirable,
- negative `A_t`: decreasing probability is desirable.

## 7.3 The clipped surrogate policy loss

The code computes:

\[
surr1 = r_t A_t
\]

\[
surr2 = \text{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t
\]

and then uses:

\[
L_{\pi} = - \mathbb{E}\left[\min(surr1, surr2)\right]
\]

with `epsilon = self.clip = config.PPO_CLIP`.

This is the standard PPO clipping idea: allow policy movement, but limit the incentive to move too far in one update.

## 7.4 The value loss

The code also clips the value update around the old value prediction:

\[
V_{\text{clipped}} = V_{\text{old}} + \text{clip}(V_{\text{new}} - V_{\text{old}}, -\epsilon, \epsilon)
\]

Then it computes two candidate squared errors:

\[
L_{v1} = (V_{\text{new}} - R_t)^2
\]

\[
L_{v2} = (V_{\text{clipped}} - R_t)^2
\]

and uses:

\[
L_v = \mathbb{E}[\max(L_{v1}, L_{v2})]
\]

This is more conservative than an unclipped MSE regression and is consistent with PPO-style value clipping.

## 7.5 The entropy term

The helper computes entropy per sample as:

\[
H_t = -\sum_a p_t(a)\log p_t(a)
\]

Then the loss term stored in code is:

\[
L_{\text{ent}} = -\mathbb{E}[H_t]
\]

and the total loss is:

\[
L = L_\pi + c_v L_v + c_e L_{\text{ent}}
\]

where:

- `c_v = self.vf_coef = config.PPO_VALUE_COEF`
- `c_e = self.ent_coef = config.PPO_ENTROPY_COEF`

Because `L_ent` is negative entropy, a positive entropy coefficient encourages higher entropy by reducing total loss when entropy is large.

## 7.6 The exact masked-policy detail during training

A very important implementation fact is this:

**training-time logits are masked again using the stored rollout action mask before log-prob and entropy are computed.**

This happens in:

- `_policy_value(...)`
- `_policy_value_group_batched(...)`

So the policy used for PPO ratio recomputation is the same masked policy family that was used during rollout sampling.

That keeps old and new log-probabilities comparable.

## 7.7 What is actually optimized here

The verified total training objective per minibatch is:

\[
L = L_\pi + \text{vf\_coef} \cdot L_v + \text{ent\_coef} \cdot L_{\text{ent}}
\]

with:

- clipped policy objective,
- clipped value objective,
- entropy regularization,
- gradient clipping,
- optional approximate-KL early stopping.

There is no verified auxiliary imitation loss, curiosity loss, reconstruction loss, centralized value mixing loss, or recurrent-state loss.

## 7.8 Approximate KL and what it is here

The sequential path computes:

\[
\text{approx\_kl} \approx \mathbb{E}[\log p_{\text{old}} - \log p_{\text{new}}]
\]

on the sampled actions in the minibatch.

This is not the exact full-distribution KL divergence between two categorical distributions over all actions. It is a sampled-action approximation used as a trust-region-like safety signal.

If `self.target_kl > 0`, the runtime may stop further epochs early when this quantity becomes too large.

## 7.9 Why the old logits are not stored

A common question is:

> Why does the buffer store only `logp_old` instead of the full old logits?

Because PPO only needs the ratio for the chosen action. The chosen-action log-probability is enough for:

\[
r_t = \exp(\log p_{\text{new}} - \log p_{\text{old}})
\]

This saves storage and simplifies checkpointing.

## 7.10 Loss-construction diagram

```text
stored rollout:
    obs_t, act_t, logp_old_t, val_old_t, ret_t, adv_t, act_mask_t
        |
        v
run current model on obs_t with same act_mask_t
        |
        +--> logits_new -> logp_new(act_t) -> ratio -> clipped policy loss
        |
        +--> value_new ---------------------> clipped value loss
        |
        +--> policy entropy ----------------> entropy regularizer
        |
        v
total PPO loss
```

## 7.11 Failure modes in loss construction

1. **Using unmasked logits during training.**
   This would make `logp_new` belong to a different policy than the one used during rollout.

2. **Changing action indexing without changing stored actions.**
   The log-prob lookup would point to the wrong action dimension.

3. **Reusing stale `logp_old`.**
   PPO assumes `logp_old` came from the behavior policy that generated the trajectory.

4. **Using a different observation schema for training than for rollout.**
   The ratio would still compute, but it would no longer represent a valid on-policy update.

---

## 8. Optimization loop and update schedule

## 8.1 When updates happen

The PPO runtime updates when `self._step % self.T == 0`, where:

- `self._step` increments once per `record_step()` call,
- `self.T = config.PPO_WINDOW_TICKS`.

With the current config defaults, the resolved rollout horizon is `256` steps.

## 8.2 What “window” means operationally

A window here means:

- store each slot’s transitions for roughly `T` decision steps,
- then train on the accumulated slot-local buffers,
- then clear those buffers.

This is an on-policy design. Buffers are not retained as a replay memory after training.

## 8.3 Slot preparation before training

Before any optimization begins, `_train_aids_and_clear(...)` calls `_prepare_train_slot(...)` for each selected slot.

This method:

1. stacks the slot-local lists into tensors,
2. validates mask alignment,
3. computes `adv` and `ret`,
4. determines minibatch count and minibatch size,
5. packages everything into `_PreparedTrainSlot`.

This is an important separation point:

- rollout ownership remains slot-local,
- grouped execution happens only after each slot’s data is materialized separately.

## 8.4 Sequential path

The safe baseline path is `_train_prepared_slot_sequential(...)`.

For one slot, it does:

1. get or create that slot’s `Adam` optimizer,
2. run for `self.epochs`,
3. shuffle the trajectory with `torch.randperm`,
4. iterate over minibatches,
5. recompute masked logits, values, and entropy,
6. compute policy/value/entropy losses,
7. `zero_grad`,
8. `backward`,
9. clip gradient norm,
10. `optimizer.step`,
11. track approximate KL,
12. optionally early-stop remaining epochs,
13. `scheduler.step()` once after the slot’s training work is done,
14. clear the slot’s rollout buffer.

## 8.5 Grouped batched path

The more advanced path is `_train_prepared_group_batched(...)`.

It is used only when a group of slots is compatible for grouped execution:

- more than one slot,
- `torch.func` available,
- same Python model type,
- not TorchScript,
- no `nn.MultiheadAttention` present.

### What grouped training does

It batches **the math**, not the learning ownership.

For a compatible group:

- each slot still has its own model,
- each slot still has its own optimizer,
- each slot still has its own scheduler,
- each slot still has its own rollout tensors.

The function pads minibatches to a common length per group chunk, uses a `valid_batch` mask, computes stacked forward/loss terms with `vmap`, then scatters stacked gradients back to each original model before independent optimizer steps.

### Why this is not shared learning

No parameter averaging occurs. No optimizer state is shared. No cross-slot gradient sum is applied to a common model.

This is better understood as **vectorized independent PPO**, not shared-policy PPO.

## 8.6 Optimizer and scheduler

Each slot lazily gets:

- `optim.Adam(model.parameters(), lr=self.lr)`
- `CosineAnnealingLR(optimizer, T_max=self.T_max, eta_min=self.eta_min)`

So the optimizer and LR scheduler are **slot-local**, not global.

That means two different slots can have different Adam moment histories and different scheduler states depending on when they were created or reset.

## 8.7 Gradient clipping

After backpropagation, the code applies:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
```

This is important in PPO because policy and value gradients can become unstable, especially with poorly scaled rewards or stale ratio geometry.

## 8.8 Early stopping by target KL

If `target_kl > 0`, the sequential path stops remaining epochs for that slot when the epoch-level approximate KL exceeds the target.

The grouped path uses a slightly different but related mechanism:

- each slot in the grouped lane tracks its own max approximate KL,
- if a slot exceeds the target, that slot is marked inactive for later epochs while other slots may continue.

That is a subtle but important difference between the sequential and grouped implementations.

## 8.9 Resolved default optimization hyperparameters

The active config resolves the following defaults:

| Hyperparameter | Value |
| --- | --- |
| `PPO_WINDOW_TICKS` | `256` |
| `PPO_LR` | `3e-4` |
| `PPO_CLIP` | `0.2` |
| `PPO_ENTROPY_COEF` | `0.05` |
| `PPO_VALUE_COEF` | `0.5` |
| `PPO_EPOCHS` | `4` |
| `PPO_MINIBATCHES` | `8` |
| `PPO_MAX_GRAD_NORM` | `0.5` |
| `PPO_TARGET_KL` | `0.02` |
| `PPO_GAMMA` | `0.995` |
| `PPO_LAMBDA` | `0.95` |
| `PPO_LR_T_MAX` | `10_000_000` |
| `PPO_LR_ETA_MIN` | `1e-6` |

## 8.10 One configuration caveat

The config also exposes `PPO_UPDATE_TICKS`, but the core runtime update trigger verified in `rl/ppo_runtime.py` is driven by `PPO_WINDOW_TICKS` and `self._step % self.T == 0`.

A modifier should therefore not assume that changing `PPO_UPDATE_TICKS` alone will change the actual optimizer cadence. In the verified learning runtime, the cadence is controlled by `PPO_WINDOW_TICKS`.

## 8.11 Engineering scaffolding versus mathematical core

The mathematical core of the update loop is:

- GAE target computation,
- clipped surrogate objective,
- clipped value loss,
- entropy regularization.

The engineering scaffolding is:

- architecture grouping,
- vmap path selection,
- gradient scattering back to original models,
- scheduler state management,
- rich telemetry aggregation.

The scaffolding matters for performance and operability, but if you want to understand “what is PPO optimizing,” focus first on the mathematical core.

---

## 9. Multi-agent and multi-brain training organization

## 9.1 The central design choice: no parameter sharing

`PerAgentPPORuntime` is explicitly documented and enforced as a no-sharing design:

- one model per slot,
- one optimizer per slot,
- one scheduler per slot,
- one rollout buffer per slot.

The code even contains `_assert_no_optimizer_sharing(...)` to defend against accidental optimizer reuse across slots.

This is the opposite of the common multi-agent pattern where all agents share one policy network.

## 9.2 What “multi-agent PPO” means here

In many papers, “multi-agent PPO” might mean:

- one shared policy for all agents,
- one shared critic,
- possibly centralized training with decentralized execution.

That is **not** what this repository implements.

Here, “multi-agent” means:

- the simulation contains many agents at once,
- but their learning objects are slot-local and independent,
- only execution is batched opportunistically.

## 9.3 Team-aware architecture assignment

The respawn and spawn subsystems support team-aware architecture selection:

- fixed per-team assignment,
- mixed per-spawn assignment,
- weighted random or alternating architecture mixes.

This means different agents can not only have different weights, but different **brain variants**.

That heterogeneity is why:

- inference buckets must group by architecture,
- grouped PPO training must only batch compatible models,
- checkpointing must save the brain kind per slot.

## 9.4 Weight inheritance across generations

The respawn subsystem can clone a parent brain and, if the target architecture kind matches the parent kind, load the parent’s `state_dict()` into the child.

Then it may optionally perturb parameters with Gaussian noise.

So the long-run policy population evolves under both:

- gradient updates during an individual’s lifetime,
- inheritance and mutation across respawn events.

This hybrid dynamic is important for interpretation. If behavior changes over long timescales, not all of that change comes from PPO alone.

## 9.5 Why slot identity still matters

Although persistent agent UIDs exist in the registry, the PPO runtime uses slot ids for ownership because the live engine state is organized by slots.

Therefore:

- buffers are keyed by slot,
- optimizers are keyed by slot,
- schedulers are keyed by slot,
- value cache is keyed by slot.

That is why dead slots must be flushed and reset before respawn.

## 9.6 Architecture grouping does not imply policy coupling

The registry maintains `brain_arch_ids` and groups alive agents by architecture signature. PPO training can also group compatible prepared slots.

This only means:

- same interface shape,
- same computational graph shape,
- same vectorization lane.

It does **not** mean:

- shared loss,
- shared parameter tensor,
- shared optimizer moments,
- shared value baseline.

That distinction must remain mentally clear.

## 9.7 What is absent

The verified code does **not** show:

- a centralized critic over multiple agents,
- a single shared team policy,
- parameter averaging across slots,
- joint action modeling,
- communication actions inside the PPO runtime,
- a recurrent hidden-state carry across ticks.

These things are all possible in multi-agent RL generally, but they are not part of the verified active learning path described here.

---

## 10. Persistence of learning state and resume integrity

## 10.1 Why persistence matters for PPO

Resuming a PPO run faithfully requires more than saving neural weights.

If you restore only model parameters and ignore:

- optimizer moments,
- scheduler state,
- rollout buffers,
- pending boundary bootstrap metadata,
- value cache,
- RNG state,

then the resumed run is no longer the same training process. It may still run, but its learning trajectory will diverge.

This repository explicitly tries to avoid that error.

## 10.2 What the checkpoint saves

`CheckpointManager.save_atomic(...)` stores a large structured payload. For the learning subsystem, the important parts are:

### Per-slot brain payload
For each slot in `registry.brains`:

- brain kind,
- brain `state_dict()`.

### PPO runtime payload
From `ppo.get_checkpoint_state()`:

- global PPO step counter,
- train update sequence,
- rich telemetry row sequence,
- per-slot rollout buffers:
  - `obs`
  - `act`
  - `logp`
  - `val`
  - `rew`
  - `done`
  - `act_mask`
- per-slot optimizer `state_dict`,
- per-slot scheduler `state_dict`,
- `value_cache`,
- `value_cache_valid`,
- `pending_window_agent_ids`,
- `pending_window_done`.

### Global resume integrity payload
The checkpoint also stores:

- registry tensors,
- generations,
- agent UIDs,
- next agent id,
- statistics,
- respawn controller state,
- catastrophe controller state,
- RNG state for Python, NumPy, Torch CPU, and best-effort CUDA RNG.

## 10.3 One subtle persistence detail: bootstrap is not saved directly

Inside `get_checkpoint_state()`, the slot buffer payload does **not** serialize `b.bootstrap` directly. The comment explicitly says bootstrap is “ephemeral / window-boundary specific.”

Instead, the design preserves the active deferred-boundary mechanism through:

- `value_cache`,
- `value_cache_valid`,
- `pending_window_agent_ids`,
- `pending_window_done`.

That is an important design choice. It means exact resume of boundary state depends on those fields, not on serializing a per-buffer bootstrap scalar.

## 10.4 How restore happens

On resume, `main.py` performs the following broad order:

1. load checkpoint on CPU,
2. restore world grid and zones,
3. create empty registry and stats objects,
4. create the tick engine,
5. call `CheckpointManager.apply_loaded_checkpoint(...)`,
6. restore runtime state into registry, engine, stats, PPO runtime,
7. restore RNG **last**.

That last point is critical and intentional.

## 10.5 Why RNG is restored last

`utils/checkpointing.py` explicitly documents that RNG restoration must happen last, because constructors and setup code can consume random numbers.

If RNG were restored too early, then model creation, object setup, or tensor initialization during restore could shift the random sequence. The resumed run would then diverge even though the checkpoint looked complete.

This is a serious operational detail, not a cosmetic one.

## 10.6 Brain reconstruction on load

During restore, checkpointing does not simply attach opaque saved brain objects.

It reconstructs each brain by:

1. reading the stored brain kind,
2. calling `_make_brain(kind, device)`,
3. loading the saved `state_dict()`,
4. assigning the brain into the registry with `registry.set_brain(...)`.

After that, the registry rebuilds architecture metadata so the inference bucket logic remains consistent.

## 10.7 Optimizer and scheduler restoration

PPO restore recreates optimizers by calling `_get_opt(aid, model)`, then loads each optimizer `state_dict`, then moves optimizer-state tensors onto the target device.

Schedulers are restored afterward by loading their saved `state_dict`s.

This is the correct order because schedulers reference optimizers.

## 10.8 Resume-integrity diagram

```text
checkpoint
    -> world/grid/zones
    -> registry tensors and brain kinds/weights
    -> engine controller state
    -> PPO buffers/optimizers/schedulers/value-cache/pending-window state
    -> stats
    -> RNG state restored last
```

## 10.9 What can silently break continuity

A resume can silently stop being faithful if any of the following are changed between save and load:

- observation width,
- action count,
- brain architecture shapes,
- head dimensions,
- checkpoint brain-kind factory mapping,
- optimizer parameter structure,
- slot-capacity expectations,
- legality-mask semantics,
- slot reuse/reset timing.

The code defends against some of these with explicit runtime errors. Others can still produce semantic drift rather than immediate crashes.

## 10.10 Why checkpoint compatibility is a learning issue, not just an I/O issue

If checkpoint compatibility is broken, the result is not merely “cannot load file.” It can be:

- wrong bucket metadata,
- wrong optimizer state alignment,
- stale value cache,
- incorrect pending-window bootstrap,
- wrong policy/output shape,
- resumed training on a mathematically different process.

That is why checkpoint integrity belongs inside a learning-system volume.

---

## 11. How to modify the learning system without breaking PPO

This section is intentionally direct. It is written as safe-modification guidance for a future engineer.

## 11.1 If you change observation width

You must re-check all of the following:

- `config.RAY_TOKEN_COUNT`
- `config.RAY_FEAT_DIM`
- `config.RICH_BASE_DIM`
- `config.INSTINCT_DIM`
- `config.OBS_DIM`
- `agent/obs_spec.py::split_obs_flat`
- `agent/obs_spec.py::split_obs_for_mlp`
- `_BaseMLPBrain` constructor validation
- the two-token embedding logic
- checkpoint compatibility for saved brains
- any existing rollout buffers or checkpoints

This is not a “small edit.” The code explicitly treats observation layout as a schema contract.

## 11.2 If you change action count

You must re-check:

- `config.NUM_ACTIONS`
- actor head output width,
- action-mask builders,
- categorical sampling,
- PPO training-time masking,
- stored action indices,
- checkpoint compatibility.

If rollout and training disagree on action indexing or mask width, PPO ratios become invalid.

## 11.3 If you change mask semantics

You must keep rollout and training aligned.

Specifically:

- `TickEngine` samples from masked logits,
- `record_step()` stores the action mask,
- `_policy_value()` and `_policy_value_group_batched()` re-mask logits during training.

If one side changes and the other does not, then `logp_old` and `logp_new` no longer belong to the same policy family.

That silently corrupts the clipped-ratio update.

## 11.4 If you change the brain forward interface

The current contract is:

```python
(obs) -> (logits_or_dist, value)
```

with the active MLP brains returning raw logits tensor and value tensor.

The following code assumes that contract:

- `ensemble_forward`
- `_policy_value`
- `_policy_value_group_batched`
- bucket inference
- PPO training

If you add auxiliary heads or change return order, you must update all consumers.

## 11.5 If you add recurrence or hidden state

The current PPO runtime stores:

- observations,
- actions,
- old log-probs,
- values,
- rewards,
- dones,
- action masks.

It does **not** currently store a recurrent hidden-state trajectory contract.

Therefore adding recurrence is not local. You would need to redesign:

- rollout storage,
- training minibatch slicing,
- checkpoint state,
- bucket inference,
- grouped batched PPO logic,
- reset logic on respawn,
- sequence-boundary handling.

Do not treat this as a one-line model swap.

## 11.6 If you want shared policies

The current runtime is structurally organized around per-slot models and per-slot optimizers.

If you want one shared policy, you must redesign at least:

- buffer ownership,
- optimizer ownership,
- scheduler ownership,
- checkpoint payload shape,
- slot reset logic,
- grouped training,
- respawn inheritance semantics.

The current grouped execution helpers are performance tools for independent models, not a partial implementation of parameter sharing.

## 11.7 If you change reward scale

Reward scale affects:

- return magnitude,
- critic regression difficulty,
- ratio-weighted policy gradient magnitude,
- entropy/value balance,
- gradient norm clipping behavior.

This code does normalize advantages, which helps, but it does **not** make reward scale irrelevant. Large reward-scale changes can still destabilize value loss and long-term optimization behavior.

## 11.8 If you modify respawn behavior

Remember that policy weights can change in two ways:

- PPO updates,
- respawn cloning/mutation.

If you change respawn inheritance, mutation noise, or architecture reassignment, you are changing the effective learning dynamics even if PPO code stays untouched.

For example:

- increasing mutation noise changes policy continuity across generations,
- switching architecture kinds on clone can prevent weight transfer,
- removing PPO reset on slot reuse can leak stale optimizer moments into a new occupant.

## 11.9 If you edit checkpointing

At minimum, maintain consistency among:

- brain kind reconstruction,
- model `state_dict`,
- optimizer `state_dict`,
- scheduler `state_dict`,
- value cache,
- pending window metadata,
- RNG state.

Restoring only weights is not enough for faithful continuation.

## 11.10 If you edit grouped PPO training

Be careful with:

- padding logic,
- `valid_batch`,
- per-slot counts,
- gradient scattering back to original models,
- early-stop semantics,
- no-sharing guarantees.

The grouped path is easy to damage because it compresses many slot-local assumptions into one vectorized computation.

## 11.11 Invariants worth preserving

The following invariants are especially important:

1. rollout action must be legal under the stored action mask,
2. training-time logits must be masked with the same semantics,
3. stored `logp_old` must match the behavior policy that sampled `act`,
4. `val_old[t]` must correspond to the same observation as `obs[t]`,
5. bootstrap value must correspond to the next state of the same slot occupant,
6. slot resets must happen before reused slots inherit new occupants,
7. checkpoint reconstruction must restore brain kind and state before optimizer state is loaded.

---

## 12. Common beginner misreadings

## 12.1 “The highest logit is the action probability.”

Incorrect.

A logit is an unnormalized score. Probabilities are obtained only after masking and softmax. Moreover, the runtime samples stochastically from the masked categorical distribution; it does not simply take the largest logit.

## 12.2 “Reward equals advantage.”

Incorrect.

Reward is the immediate scalar fed into the PPO runtime. Advantage is a discounted, value-corrected quantity computed later by `_gae(...)`.

## 12.3 “PPO just means policy gradient.”

Too vague to be useful here.

The verified implementation includes:

- chosen-action old log-prob storage,
- GAE,
- clipped policy loss,
- clipped value loss,
- entropy regularization,
- minibatches,
- gradient clipping,
- approximate-KL early stopping,
- slot-local optimizers and schedulers.

That is much more specific than “policy gradient.”

## 12.4 “Value loss is optional noise.”

Incorrect.

The critic is not decorative. It provides the baseline used to form advantages and is trained against return targets. Poor value estimates can degrade advantage quality and destabilize policy updates.

## 12.5 “Changing observation size is a small edit.”

Incorrect.

In this repository, observation size is a cross-cutting schema contract. It affects model shapes, checkpoint compatibility, buffer alignment, and semantic meaning.

## 12.6 “Checkpoint restore only needs weights.”

Incorrect.

Faithful PPO continuation also needs:

- optimizer state,
- scheduler state,
- buffered trajectories,
- value cache,
- pending boundary metadata,
- RNG state.

## 12.7 “Grouped training means the agents share one brain.”

Incorrect.

Grouped training only batches independent models of compatible architecture. It does not merge their parameters or optimizers.

## 12.8 “The runtime trains one shared team policy.”

Incorrect.

The verified runtime is slot-local and per-agent. Team reward terms exist, but the model ownership remains per slot.

## 12.9 “If a slot dies, the PPO runtime can just keep going.”

Incorrect.

The slot-local design makes death and respawn a training-boundary event. Dead slots are flushed and reset to avoid leaking old trajectory and optimizer state into new occupants.

## 12.10 “Approximate KL here is the exact KL divergence.”

Incorrect.

The code uses a sampled-action log-prob difference as an approximate trust-region signal. It is useful, but it is not the full exact categorical KL over all actions.

---

## 13. What this volume establishes for later volumes

This volume establishes the internal learning mechanics:

- how observations enter the brains,
- how logits and values are produced,
- how actions are sampled,
- how rewards are attached,
- how PPO targets are computed,
- how updates are applied,
- how learning state is checkpointed and resumed safely.

It intentionally does **not** fully cover:

- viewer/operator workflows,
- run-directory conventions and artifact inspection,
- telemetry surface details,
- long-run operational procedures,
- checkpoint browser usage,
- extension workflow outside the learning stack.

Those belong to later operational and tooling volumes.

---

## Appendix A. Verified learning pipeline at a glance

```text
TickEngine.run_tick
    |
    +--> identify alive slots
    |
    +--> build obs: _build_transformer_obs(...) -> (N_alive, OBS_DIM)
    |
    +--> build legal-action mask: (N_alive, 41)
    |
    +--> group alive slots by architecture: registry.build_buckets(...)
    |
    +--> per bucket:
    |       ensemble_forward(models, obs_bucket)
    |           -> logits
    |           -> value estimates
    |       mask illegal logits
    |       sample action from Categorical(masked_logits)
    |
    +--> execute environment dynamics
    |
    +--> compute reward components
    |       individual + team + hp shaping
    |
    +--> record_step(...)
    |       store obs, action, logp_old, value_old, reward, done, act_mask
    |       increment PPO step counter
    |       if boundary reached:
    |           either train immediately (explicit bootstrap path)
    |           or stage pending-window metadata (active cached-bootstrap path)
    |
    +--> flush dead slots before respawn
    |
    +--> respawn new occupants
    |       reset slot-local PPO state for reused slots
    |
    +--> next tick normal forward updates value cache
    |
    +--> finalize_pending_window_from_cache()
    |       write bootstrap values for surviving slots
    |       run PPO training
    |
    +--> PPO training:
            stack slot-local rollout tensors
            compute GAE advantages and returns
            run epochs/minibatches
            recompute masked logits and values
            compute clipped policy/value/entropy losses
            clip gradients
            optimizer.step per slot
            scheduler.step per slot
            clear rollout buffers
```

---

## Appendix B. Verified loss components and their roles

| Component | Mathematical meaning | Where it comes from | Where it is computed | Why it matters |
| --- | --- | --- | --- | --- |
| `logp_old` | old chosen-action log-probability | rollout-time masked policy | `record_step()` | anchors PPO ratio to the behavior policy |
| `adv` | generalized advantage estimate | rewards, values, dones, bootstrap value | `_gae()` | tells policy whether sampled action was better or worse than baseline |
| `ret` | value regression target | `adv_raw + value_old` before advantage normalization | `_gae()` | supervises critic |
| `ratio` | new/old action-probability ratio | `exp(logp_new - logp_old)` | sequential and grouped training paths | core PPO trust-region-like quantity |
| `loss_pi` | clipped policy loss | `-mean(min(ratio*adv, clip(ratio)*adv))` | sequential and grouped training paths | pushes policy in good directions without unlimited step incentive |
| `loss_v` | clipped value loss | max of unclipped and clipped squared error to `ret` | sequential and grouped training paths | trains the critic conservatively |
| `entropy` | categorical policy entropy | masked logits -> log-softmax | `_policy_value()` and grouped variant | measures policy spread/randomness |
| `loss_ent` | negative entropy mean | `-mean(entropy)` | sequential and grouped training paths | regularizes toward exploration when weighted positively |
| total loss | `loss_pi + vf_coef*loss_v + ent_coef*loss_ent` | assembled from above | sequential and grouped training paths | actual scalar backpropagated objective |
| `approx_kl` | sampled-action log-prob difference | `mean(logp_old - logp_new)` | sequential and grouped training paths | early-stop safety signal, not the optimized loss itself |

---

## Appendix C. Compact mental model of PPO in this repository

A compact, implementation-faithful mental model is:

1. **Every live slot has its own actor-critic brain.**
2. **The brain reads the active observation contract (`OBS_DIM`, default 276 under the self-centric schema) and emits masked-policy logits plus one scalar value.**
3. **The engine samples a legal discrete action from the masked categorical policy.**
4. **After the environment step, the engine computes a shaped reward and a done flag for that same slot.**
5. **The PPO runtime stores that transition in the slot’s own buffer.**
6. **After a window of steps, the runtime computes GAE advantages and return targets for each slot separately.**
7. **The same slot’s current brain is run again on stored observations, using the same legality masks, to obtain new log-probs and new values.**
8. **PPO compares new chosen-action log-probs to stored old chosen-action log-probs, clips the update incentive, trains the critic against returns, adds entropy regularization, clips gradients, and steps that slot’s optimizer.**
9. **Dead slots are flushed and reset before respawn so new occupants do not inherit stale rollout or optimizer state.**
10. **Checkpointing preserves enough state to resume the same learning process rather than merely the same neural weights.**

That is the core learning system.

Everything else in this volume is detail, guardrail, or performance engineering around that core.
