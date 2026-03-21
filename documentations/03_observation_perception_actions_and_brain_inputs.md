# Volume 3 — Observation, Perception, Actions, and Brain Inputs

## Document Purpose

This volume defines the interface between the simulation world and the policy network. In this repository, that interface is not an informal idea. It is a concrete contract with fixed widths, fixed ordering, fixed action indices, and multiple layers of runtime validation. The code treats it as a schema, not as an incidental implementation detail. `config.py` fixes the top-level observation and action dimensions; `engine/tick.py` constructs the live observation tensor each tick; `agent/obs_spec.py` re-splits that tensor under strict checks; and the MLP brain family validates the same layout again before producing logits and values.

This document is written cumulatively on purpose. Good long-form engineering documentation benefits from context-first ordering, clear section hierarchy, and short, idea-focused paragraphs; Write the Docs and GitHub’s documentation guidance both recommend that kind of progressive structure for technical manuals.

## Audience and How to Read This Volume

This volume is for two readers at once.

The first reader is new to reinforcement learning observation design and needs a clean answer to a simple question: *what exactly does one agent know at decision time, and what exactly can it choose to do next?*

The second reader is the engineer who may later edit the schema. That reader needs a different answer: *which constants, slices, masks, and model expectations are coupled tightly enough that one careless edit can silently corrupt training or inference?*

Read Sections 1 through 5 if the goal is conceptual understanding. Read Sections 6 through 11 if the goal is safe modification.

---

## 1. Why the Observation–Action Interface Exists

The policy cannot consume the raw world directly. The world state is stored as multiple tensors with simulation-specific meanings: a three-channel grid, a registry of per-agent attributes, runtime statistics, zone masks, and other engine-side state. The neural model, by contrast, expects a fixed-width numeric tensor per agent. The observation system exists to compress and organize the relevant subset of world state into that fixed-width per-agent vector. On the output side, the action system exists to turn one discrete sampled index back into an engine-understood intention such as idle, move in one of eight directions, or attack along a direction/range combination.

This layer is one of the highest-risk contracts in the repository because it carries both semantics and shape. Changing observation width changes more than one file. It changes model construction, checkpoint compatibility, PPO rollout storage, and any code that assumes a specific split between rays and non-ray features. `config.py` says this explicitly for both `OBS_DIM` and `NUM_ACTIONS`.

A useful beginner mental model is this:

```text
simulation world state
    -> observation builder
        -> fixed-width per-agent tensor
            -> policy/value network
                -> discrete action id
                    -> engine decode
                        -> attempted world operation
```

That diagram is small, but each arrow hides a separate correctness condition. This volume opens those arrows one by one.

---

## 2. Observation Pipeline Overview

### 2.1 Where observation construction starts

Observation construction happens inside `TickEngine._build_transformer_obs()`. Despite the historical function name, the live code builds the flat observation consumed by the current MLP brain family. The function receives the alive slot indices and their `(x, y)` positions, then builds one observation row per alive agent. The final shape must be `(N_alive, OBS_DIM)`, and the code raises immediately if that shape does not match.

### 2.2 When observations are computed during runtime

`run_tick()` follows a stable order:

1. find alive agents,
2. build observations,
3. build legality masks,
4. run policy inference,
5. sample actions,
6. execute combat,
7. execute movement,
8. continue with environment effects and bookkeeping.

This matters. The policy chooses using the state at the start of the tick, not after combat resolution or post-move updates. That is the decision-time interface.

### 2.3 What source state the builder reads

The builder reads from:

- `self.grid`, the three-channel world grid,
- `self.registry.agent_data`, the per-slot attribute table,
- zone masks such as heal/control-point masks,
- global statistics in `self.stats`,
- and a derived `unit_map` built from the registry plus grid occupancy.

### 2.4 Per-agent or batched?

The implementation is batched and heavily vectorized. `_build_transformer_obs()` produces one tensor for all alive agents at once. Raycasting computes all rays for all alive agents in one call. Instinct features are also computed in batch using broadcasted offset grids. This is not a Python loop that builds one observation object at a time.

### 2.5 High-level observation layout

At the highest level, the observation is:

```text
obs = [ rays_flat | rich_base | instinct ]
```

with:

- `RAY_TOKEN_COUNT = 32`
- `RAY_FEAT_DIM = 8`
- `RAYS_FLAT_DIM = 32 * 8 = 256`
- `RICH_BASE_DIM = 23`
- `INSTINCT_DIM = 4`
- `OBS_DIM = 256 + 23 + 4 = 283`

That 283-wide flat vector is the canonical contract for the current codebase.

---

## 3. Observation Schema and Feature Families

The observation has three major families, but one of them is itself internally structured.

### 3.1 Family 1 — Ray features (`256` floats)

The first `256` columns are the spatial perception block. They come from `raycast32_firsthit()` and represent `32` rays with `8` features each. The function returns shape `(N, 256)`, then the engine concatenates that block directly into the front of the observation.

### 3.2 Family 2 — Rich base features (`23` floats)

The next `23` columns are scalar, flag, and global-context features. They include self-state, normalized position, team/unit indicators, zone occupancy flags, global tick/score/combat counters, and padding at the tail. These values are written explicitly column by column in `_build_transformer_obs()`.

### 3.3 Family 3 — Instinct features (`4` floats)

The final `4` columns are a local neighborhood summary produced by `_compute_instinct_context()`. They are not learned memory. They are engine-computed density-style features describing nearby allied archers, nearby allied soldiers, noisy enemy density, and a threat ratio derived from those counts.

### 3.4 Why the split matters

This split is not only conceptual. `obs_spec.split_obs_flat()` and `obs_spec.split_obs_for_mlp()` depend on it exactly. The brain then relies on the same split when it reshapes the first `256` floats into `(B, 32, 8)` and treats the remaining `27` floats as the rich vector.

---

## 4. Spatial Perception and Directional/Local Sensing

### 4.1 What “ray-based perception” means here

In this codebase, ray-based perception means the engine sends out a fixed set of sensor directions from the agent’s current position and records only the **first meaningful hit** per direction. It does not return a full occupancy strip. It does not build an image-like crop. It compresses each direction down to one categorical hit description plus two normalized scalars.

### 4.2 Number of directions

The 32-ray system uses `32` directions evenly spaced around the unit circle, generated from evenly spaced angles in `[0, 2π)`. These are continuous `(cos θ, sin θ)` directions computed once at import time, then reused.

This is an important detail for beginners: the sensor is not limited to only eight compass directions. The **action space** uses eight compass directions, but the **observation space** uses 32 sensing directions.

### 4.3 Per-ray feature layout

Each ray emits exactly eight features in this order:

```text
[ onehot6(none, wall, red-soldier, red-archer, blue-soldier, blue-archer),
  dist_norm,
  hp_norm ]
```

That is why `RAY_FEAT_DIM = 8`: six categorical slots plus two scalar slots. The class coding is explicit in `raycast_32.py`, and the assembly logic writes the one-hot block first, then normalized distance, then normalized HP.

### 4.4 What is considered a “hit”

The ray marcher gathers, along every ray path:

- occupancy (`grid[0]`),
- HP (`grid[1]`),
- agent id (`grid[2]`).

A wall hit is any cell where `grid[0] == 1`. An agent hit is any cell where `grid[2] >= 0`. The function computes the first wall index and first agent index separately, then chooses the earlier one. If both occur at the same step, wall wins because the comparison is `<=`. After that, agent hits are refined into one of four team-plus-unit classes using `grid[0]` for team color and `unit_map` for subtype.

### 4.5 Distance and HP normalization

For a valid hit, `dist_norm` is `(first_hit_step / max_steps_each_agent)`. If no hit exists within the agent’s vision range, the value is zeroed by multiplying with the validity mask. `hp_norm` is the HP stored at the hit location divided by `MAX_HP`, clamped into `[0, 1]` before output.

### 4.6 Per-agent vision range

The raycast does not blindly use a single global depth for everyone. `_build_transformer_obs()` passes `alive_data[:, COL_VISION]` into `raycast32_firsthit()` as `max_steps_each`, so each agent’s observation depth is clipped by its own current vision stat. The config derives the maximum possible raycast depth from the largest class-specific vision range.

### 4.7 Limitations of the ray system

There are several important limits.

First, the ray feature is **first-hit only**. If a wall or unit blocks the ray, farther structure on that direction is not represented.

Second, the directions are continuous but cell coordinates are obtained by truncating to integer indices. The code explicitly notes that shallow angles can revisit the same cell across multiple steps because of integer casting. That is a discretization artifact of this implementation.

Third, the ray block is not team-relative. It encodes absolute team colors (`red-*`, `blue-*`), not “ally” and “enemy”. That means the policy must combine ray classes with the later team indicators in `rich_base` to interpret a seen unit as friend or foe from the acting agent’s perspective.

That last point is easy to miss and matters a great deal.

---

## 5. Self-State, Scalar, and Semantic Features

### 5.1 The 23-column `rich_base` block

The code writes the `rich_base` block directly and in fixed order. The verified mapping is:

| Index | Feature | Meaning |
|---:|---|---|
| 0 | `hp / hp_max` | normalized self HP |
| 1 | `x / (W-1)` | normalized x position |
| 2 | `y / (H-1)` | normalized y position |
| 3 | `team == red` | red flag |
| 4 | `team == blue` | blue flag |
| 5 | `unit == soldier` | soldier flag |
| 6 | `unit == archer` | archer flag |
| 7 | `atk / MAX_ATK` | normalized attack strength |
| 8 | `vision / RAYCAST_MAX_STEPS` | normalized vision |
| 9 | `on_heal` | current cell is heal zone |
| 10 | `on_cp` | current cell is control point |
| 11 | `tick / 50000` | normalized global time |
| 12 | `red.score / 1000` | normalized red score |
| 13 | `blue.score / 1000` | normalized blue score |
| 14 | `red.cp_points / 500` | normalized red control-point total |
| 15 | `blue.cp_points / 500` | normalized blue control-point total |
| 16 | `red.kills / 500` | normalized red kills |
| 17 | `blue.kills / 500` | normalized blue kills |
| 18 | `red.deaths / 500` | normalized red deaths |
| 19 | `blue.deaths / 500` | normalized blue deaths |
| 20 | padding | hard zero |
| 21 | padding | hard zero |
| 22 | padding | hard zero |

The padding at indices `20:23` is deliberate. The comments say those slots exist to keep the layout exactly matched to required dimensions. They carry no active feature content in the current build.

### 5.2 How to think about `rich_base`

For a beginner, it helps to group this block conceptually:

- **Self state**: health, attack, vision, unit type.
- **Self location**: normalized x and y.
- **Identity flags**: red/blue, soldier/archer.
- **Zone state**: heal and control-point occupancy.
- **Global world summary**: tick, scores, capture totals, kills, deaths.
- **Reserved slots**: trailing zeros.

This is why the block is called “rich”: it is not one feature family. It is a compact mixture of self-local, world-global, and game-progress information.

### 5.3 The 4-column instinct block

The instinct block is computed under `@torch.no_grad()` and is explicitly defined as:

1. ally archer density,
2. ally soldier density,
3. noisy enemy density,
4. threat ratio = enemy density / (ally total density + epsilon).

The engine constructs a discrete circle of offsets using `dx^2 + dy^2 <= R^2`, where `R = INSTINCT_RADIUS`, samples the grid at those offsets, counts unit types, subtracts self from allied counts, adds Gaussian noise to enemy count, and divides by the number of offsets to turn counts into densities.

A beginner should not think of “instinct” as hidden neural memory. It is a deterministic engine-computed neighborhood summary with one deliberately noisy channel.

### 5.4 Semantic grouping helper in `obs_spec.py`

`config.py` defines semantic index groups inside the 23-column `rich_base` block:

- `own_context`
- `world_context`
- `zone_context`
- `team_context`
- `combat_context`
- `instinct_context` (by convention, the separate 4-wide instinct block)

`obs_spec.build_semantic_tokens()` can materialize these groups using `torch.index_select()`. However, the current MLP path does **not** consume those semantic tokens directly. The current brain path uses `split_obs_for_mlp()`, which returns only `rays_raw` and one concatenated `rich_vec`. That distinction matters: semantic grouping exists in the schema utilities, but the active model input path does not presently process `own_context`, `world_context`, and the rest as separate learned tokens.

That is a subtle but important implementation fact.

---

## 6. Feature Layout, Ordering, and Tensor Contracts

### 6.1 The authoritative flat layout

The authoritative flat layout is:

```text
0 ............................................... 255 | 256 .......... 278 | 279 .. 282
[                32 rays × 8 dims                ] | [ rich_base 23 ] | [ instinct 4 ]
```

or more formally:

```text
obs ∈ R^(283)
obs = concat(rays_flat ∈ R^(256), rich_base ∈ R^(23), instinct ∈ R^(4))
```

### 6.2 Ray flattening order

`raycast32_firsthit()` assembles `feat` with shape `(N, 32, 8)` and returns `feat.reshape(N, 32 * 8)`. That means the flat ordering is contiguous by ray, then by feature within ray. In other words, the first eight columns belong to ray 0, the next eight to ray 1, and so on through ray 31.

So the flat indices for ray `r` are:

```text
start = r * 8
columns start : start+8
```

and inside each eight-column block the order is:

```text
[class_none, class_wall, class_red_sold, class_red_arch, class_blue_sold, class_blue_arch, dist_norm, hp_norm]
```

### 6.3 `obs_spec` as schema enforcer

`split_obs_flat()` checks rank, checks `F == OBS_DIM`, checks `RAYS_FLAT_DIM + RICH_TOTAL_DIM == F`, and only then slices the tensor. `split_obs_for_mlp()` then checks that the ray block width equals `RAY_TOKEN_COUNT * RAY_FEAT_DIM`, reshapes it to `(B, 32, 8)`, and concatenates `rich_base` with `instinct` into a `27`-wide vector.

That means there are two independent ordering contracts:

1. the **flat observation contract** exposed by the engine,
2. the **reinterpretation contract** used by the brain.

If either side changes without the other, the code will either raise or, worse, learn on semantically wrong columns.

### 6.4 Brain-side input contract

The MLP brain family validates:

- `obs_dim == 32*8 + 27`,
- `config.OBS_DIM` matches that expectation,
- the input tensor rank is `(B, F)`,
- and the feature width is exactly `self.obs_dim`.

Then it builds:

- `rays_raw`: `(B, 32, 8)`
- `rich_vec`: `(B, 27)`
- `ray_token`: `(B, D)`
- `rich_token`: `(B, D)`
- concatenated flat trunk input: `(B, 2D)` where `D = BRAIN_MLP_D_MODEL`, default `32`, so final trunk input is `(B, 64)`.

This means the model does **not** process all 283 raw features at once in the trunk. It first compresses the ray block into one learned summary token and the full non-ray block into one learned rich token.

### 6.5 What breaks if ordering changes

If you reorder columns inside `rich_base`, at least four things become unsafe:

1. `SEMANTIC_RICH_BASE_INDICES` may point to wrong content.
2. Any future code using `build_semantic_tokens()` will silently receive wrong groups.
3. The current brain will still run, but its learned interpretation of the 27-wide rich vector becomes invalid relative to old checkpoints.
4. Any previously trained model weights become semantically stale even if tensor shapes still match.

If you change `RAY_TOKEN_COUNT`, `RAY_FEAT_DIM`, `RICH_BASE_DIM`, or `INSTINCT_DIM`, the breakage is more direct: `OBS_DIM`, the brain’s constructor checks, `split_obs_for_mlp()`, and checkpoint compatibility all change together. The code comments explicitly describe these values as schema contracts rather than ordinary tuning knobs.

---

## 7. Action Space Overview

### 7.1 The configured action space

`config.NUM_ACTIONS` defaults to `41`. The move-mask module documents two layouts:

- legacy `17`-action layout,
- current `41`-action layout.

The default, active configuration is `41`, and the live engine decode path is clearly aligned with that layout. This is the end-to-end verified schema for the current repository state.

### 7.2 High-level action families

Under the 41-action schema:

- `0` = idle,
- `1..8` = movement in eight compass directions,
- `9..40` = attacks in eight direction blocks, each with four range choices.

The eight directions are ordered as:

```text
0 N, 1 NE, 2 E, 3 SE, 4 S, 5 SW, 6 W, 7 NW
```

and this direction table is reused consistently for move masking, movement decode, and attack decode.

### 7.3 Flat indexing scheme for attacks

Attack columns begin at `9`. For direction `d` and range `r ∈ {1,2,3,4}`, the mask builder writes columns:

```text
column = 9 + d*4 + (r-1)
```

That is the same quotient/remainder structure later used by `run_tick()` to recover direction and range from a sampled attack action.

### 7.4 What the action output actually is

The model outputs one logit per action, shape `(B, act_dim)`. The brain does **not** directly move units or apply combat. The brain only emits preferences over discrete IDs. Those logits are masked externally and sampled from a categorical distribution.

That is the correct boundary: **policy output is an intention code, not the world effect itself**.

---

## 8. Action Decoding and Engine Mapping

### 8.1 Legality first, execution later

Before inference results are sampled, `build_mask()` produces a boolean legality mask of shape `(N_alive, NUM_ACTIONS)`. `True` means the action is currently legal. At inference time, illegal logits are replaced with a very negative number, then a categorical sample is drawn from the masked logits. PPO training uses the same masking idea in `_mask_logits()`.

This is a crucial distinction:

- **action validity** is decided by the mask,
- **action choice** is decided by the sampled masked policy,
- **action success** still depends on later engine resolution.

Those are three different stages.

### 8.2 Idle

`0` is always valid if the action space has at least one action. The mask builder marks `mask[:, 0] = True` immediately.

### 8.3 Movement decode

Movement actions are `1..8`. The engine decodes them as:

```text
dir_idx = action - 1
destination = current_position + DIRS8[dir_idx]
```

Agents killed earlier in the same tick do not move because movement resolution uses `alive_after`, not the original pre-combat alive mask. The code calls this “combat-first semantics.”

Movement is then subject to two layers of world resolution:

1. the target cell must be empty,
2. if multiple agents want the same empty cell, conflict resolution applies:
   - highest HP wins,
   - exact top-HP ties mean nobody moves.

So a move action means “attempt to move in direction `d`,” not “guaranteed translation.”

### 8.4 Attack decode

Attack actions are `>= 9`. The engine decodes them with:

```text
r       = ((action - 9) % 4) + 1
dir_idx =  (action - 9) // 4
```

Then it scales `DIRS8[dir_idx]` by the decoded range to obtain the target offset.

Under the verified 41-action layout, that means each direction owns a block of four contiguous range actions.

### 8.5 Soldier versus archer range rules

The legality mask does not give every unit the same attack ranges. In the 41-action path, soldiers are allowed only range `1`, while archers are allowed ranges `1..ARCHER_RANGE`, clipped to at most `4`. That gating is encoded directly in the mask builder through `allow_r`.

### 8.6 Optional line-of-sight wall blocking

If `ARCHER_LOS_BLOCKS_WALLS` is enabled, ranged attacks are intended to be masked out when a wall exists in an intermediate cell. The engine also rechecks this during combat execution before applying damage. That second check is important because even forced or incorrectly permitted actions are filtered again before damage lands.

### 8.7 Two fragility points that must be stated plainly

#### Fragility 1 — The 17-action path is not end-to-end cleanly verified

`build_mask()` documents and implements a legacy 17-action layout. But `run_tick()` decodes every attack using the 41-action quotient/remainder scheme with four ranges per direction. That means the mask builder’s “legacy 17” support should not be treated as a fully aligned end-to-end engine contract in the current repository state. The safe reading is: **41 actions are the verified live schema**.

#### Fragility 2 — The mask-time LOS helper call is shape-inconsistent with its own documented contract

`_los_blocked_by_walls_grid0()` documents its first argument as `occ0: (H, W)` occupancy channel 0. But the call site inside `build_mask()` passes `occ`, which that same function has just defined as the neighbor occupancy tensor of shape `(N, 8)`, not the full world occupancy plane. The code comment even notes this mismatch. That means the legality-mask LOS computation is not cleanly aligned with the helper’s documented input contract. The later engine-side LOS recheck reduces the damage risk, but the observation-to-action legality contract is still fragile here.

A manual should say that directly.

---

## 9. Observation-to-Brain Interface

### 9.1 The model-facing object

At inference time, each alive agent contributes one row in `obs` of shape `(N_alive, OBS_DIM)`. Buckets of agents with the same architecture signature are then extracted from that observation tensor and forwarded together. `ensemble_forward()` expects:

- `models`: list of length `K`,
- `obs`: tensor `(K, F)` aligned to the same ordering.

The alignment point is important. `obs[i]` must correspond to `models[i]`.

### 9.2 What the brain does with the flat observation

The current brain family is MLP-based. Its shared front end does **not** ingest the 283-wide vector as an undifferentiated flat input to the trunk.

Instead it:

1. validates the width,
2. splits the flat observation through `obs_spec.split_obs_for_mlp()`,
3. reshapes the first 256 values to `(B, 32, 8)`,
4. projects each ray feature vector to width `D`,
5. averages across all 32 rays to obtain one `ray_token`,
6. projects the whole 27-wide rich block into one `rich_token`,
7. concatenates the two tokens into `(B, 2D)`,
8. normalizes,
9. runs the variant-specific MLP trunk,
10. outputs policy logits and one scalar value.

With default config `D = 32`, so the trunk input width is `64`, not `283`.

### 9.3 Shared schema across brain variants

All current MLP variants share the same observation semantics and output semantics. They differ only in the downstream trunk architecture. The input contract is intentionally held constant across variants so that architecture choice does not change feature meaning.

### 9.4 Where masking happens

The brain outputs raw logits for all actions. Legality masking happens outside the brain:

- in `run_tick()` during online action sampling,
- in PPO runtime during training-time policy/value evaluation.

This separation matters operationally. A model checkpoint is not “already legality-aware” by architecture alone. It relies on the environment-side action mask.

### 9.5 Training-time batched interface

The PPO runtime also contains a batched-per-model path where observations and masks are shaped `(G, M, F)` and `(G, M, A)` for `G` homogeneous model lanes and minibatch length `M`. The math is batched, but parameters are still independent per model. This is an implementation optimization, not a semantic change to the observation/action contract.

---

## 10. Implementation Strategy and Performance Considerations

### 10.1 Observation scratch is preallocated

The tick engine preallocates multiple capacity-sized scratch tensors for observation building:

- zone booleans,
- `rich_base`,
- the combined non-ray block,
- instinct scratch,
- movement conflict buffers.

This means the observation builder is written as a hot-path system, not as one-off readable code.

### 10.2 Raycasting scratch is cached per device and dtype

`raycast32_firsthit()` maintains a per-device, per-feature-dtype scratch cache containing directions, step tensors, coordinate workspaces, active masks, and final feature buffers. Capacity grows only when needed. That design reduces repeated allocations and makes the ray path stable for long runs.

### 10.3 Instinct uses cached discrete-circle offsets

The instinct system caches the discrete circle offsets for the configured radius and reuses `(N, M)` scratch tensors for neighborhood coordinates and masks. This avoids recomputing the offset geometry every tick.

### 10.4 Bucketed inference reduces Python overhead

Alive agents are grouped by persistent architecture class through `registry.build_buckets()`, and then `ensemble_forward()` fuses per-agent forward passes for each homogeneous bucket. When enabled and large enough, the system can use `torch.func` plus `vmap`; otherwise it falls back to a safe loop. The performance goal is batching math without sharing parameters across agents.

### 10.5 Design tradeoff

The implementation optimizes the **mechanics** of feature assembly, not the **semantics**. The preallocation and vectorization do not change what one row of `obs` means, but they make the code harder to edit casually. Anyone modifying this layer must respect both semantics and scratch-buffer discipline.

---

## 11. How to Modify Features or Actions Without Breaking the System

### 11.1 If you add, remove, or reorder observation features

At minimum, inspect and usually update all of the following together:

1. `config.py` observation constants:
   - `RAY_TOKEN_COUNT`
   - `RAY_FEAT_DIM`
   - `RICH_BASE_DIM`
   - `INSTINCT_DIM`
   - `RAYS_FLAT_DIM`
   - `RICH_TOTAL_DIM`
   - `OBS_DIM`

2. `_build_transformer_obs()` if the live engine feature assembly changes.

3. `obs_spec.split_obs_flat()` and `split_obs_for_mlp()` if the slice boundaries change.

4. `SEMANTIC_RICH_BASE_INDICES` if any `rich_base` column meaning changes.

5. The MLP brain constructor and input checks if the expected observation width changes.

6. Any existing checkpoints or saved models, because shape-compatible is not the same as meaning-compatible. The config comments explicitly classify these dimensions as checkpoint contracts.

7. PPO rollout storage assumptions, because rollout entries store observations of fixed dimensionality and policy heads of fixed action width.

A safe operational rule is: **never edit observation columns in only one place**.

### 11.2 If you change ray semantics

You must decide whether the change is:

- a shape change,
- an ordering change,
- or only a changed normalization/value rule.

A shape change propagates into config constants, `obs_spec`, the brain front end, and checkpoints. An ordering change additionally invalidates any learned weights even if the dimensions stay equal. A normalization-only change keeps shapes intact but still changes model input distribution and therefore should be treated as a training-distribution change, not a harmless refactor.

### 11.3 If you change the action schema

At minimum, update together:

1. `config.NUM_ACTIONS`.
2. `build_mask()` so legality columns match the new IDs.
3. `run_tick()` decode logic for attacks and movement.
4. Brain construction, because `act_dim` comes from `NUM_ACTIONS` and determines the actor-head width.
5. PPO legality masking and rollout data, because `action_masks` are expected to match the logits’ last dimension exactly.
6. Any telemetry or analysis code that assumes current action IDs in event logs. Movement telemetry currently records raw action IDs.

### 11.4 Silent bug classes to expect

This layer can fail in ways that look like “training got worse” rather than “code crashed.” Common silent failure classes are:

- feature reorder with unchanged dimension,
- wrong semantic indices for `rich_base`,
- changing ray class meaning while keeping eight features,
- action mask columns no longer matching engine decode,
- model checkpoints loaded against semantically changed schemas,
- padding slots repurposed without updating all downstream assumptions.

### 11.5 Practical safe-change checklist

Before trusting a schema change, verify all of the following:

- `obs.shape[1] == config.OBS_DIM`
- `split_obs_for_mlp()` still reconstructs `(B, 32, 8)` and `(B, rich_dim)` correctly
- actor logits have width `NUM_ACTIONS`
- every action-mask row has at least one legal action
- sampled actions are never out of range or masked out
- forced debug actions still decode sensibly
- old checkpoints are either rejected or intentionally migrated.

---

## 12. Common Beginner Misreadings

### Misreading 1 — “The agent sees the whole world.”

False. The agent receives a fixed vector built from 32 first-hit rays, 23 rich scalar/global features, and 4 local-density instinct features. There is no full map tensor input in the current policy path.

### Misreading 2 — “Ray classes already know friend versus enemy.”

False. The ray classes are absolute (`red-*`, `blue-*`), not relative (`ally`, `enemy`). Relative interpretation requires combining those ray classes with the acting agent’s team flags in `rich_base`.

### Misreading 3 — “The action itself guarantees the outcome.”

False. The action is only an attempted intention. Movement can fail because of walls, occupancy, or same-destination conflict resolution. Attacks can fail because there is no valid victim, because the target is not an enemy, or because LOS checks reject the shot.

### Misreading 4 — “Feature order does not matter because the model will learn around it.”

False. This codebase is explicit that schema drift is dangerous. `obs_spec` exists to defend against off-by-one and split errors, and the brain validates the expected widths for the same reason. If order changes while shape stays constant, the model will not “figure it out” from old weights. It will receive different semantics under the same parameterization.

### Misreading 5 — “Instinct means memory.”

False. Instinct is a four-float engineered neighborhood summary computed fresh from nearby occupancy and unit types each tick. It is not a recurrent state and is not stored inside the model across time.

### Misreading 6 — “Semantic tokens are the current model interface.”

Not in the active MLP path. `build_semantic_tokens()` exists, but the current brain front end uses one concatenated `rich_vec`, not separate semantic-token projections.

### Misreading 7 — “The repository fully supports both 17 and 41 actions.”

That would be too strong a claim. The mask builder contains both layouts, but the live engine decode path is clearly written around the 41-action quotient/remainder mapping. The reliable documented contract for the present codebase is the 41-action layout.

---

## 13. What This Volume Establishes for Later Volumes

This volume establishes the answer to one precise systems question:

> what numeric contract connects world state to the policy, and what discrete contract connects policy output back to world operations?

It does **not** fully cover:

- the detailed architecture differences between the five MLP brain variants,
- PPO loss mathematics, advantage estimation, and optimizer flow,
- checkpoint serialization and resume compatibility in depth,
- viewer overlays and operator workflows,
- long-run telemetry and reporting design.

Those belong to later volumes. What this volume provides is the stable foundation those later discussions depend on.

---

## Appendix A — Verified Observation Layout at a Glance

### A.1 Top-level block layout

| Block | Flat indices | Width | Source |
|---|---:|---:|---|
| Ray block | `0..255` | `256` | `raycast32_firsthit()` |
| Rich base | `256..278` | `23` | `_build_transformer_obs()` |
| Instinct | `279..282` | `4` | `_compute_instinct_context()` |

### A.2 Per-ray layout

For ray `r`, flat columns are `8*r .. 8*r+7`.

| In-ray offset | Meaning |
|---:|---|
| 0 | no hit |
| 1 | wall |
| 2 | red soldier |
| 3 | red archer |
| 4 | blue soldier |
| 5 | blue archer |
| 6 | normalized first-hit distance |
| 7 | normalized HP at first-hit cell |

### A.3 Rich-base layout

| Flat index | Meaning | Why it matters |
|---:|---|---|
| 256 | normalized HP | immediate self survivability |
| 257 | normalized x | horizontal position context |
| 258 | normalized y | vertical position context |
| 259 | red flag | identity disambiguation |
| 260 | blue flag | identity disambiguation |
| 261 | soldier flag | unit capability disambiguation |
| 262 | archer flag | unit capability disambiguation |
| 263 | normalized attack | offensive strength |
| 264 | normalized vision | perception horizon |
| 265 | on heal zone | local environment affordance |
| 266 | on control point | objective occupancy |
| 267 | normalized tick | temporal phase |
| 268 | normalized red score | global team progress |
| 269 | normalized blue score | global team progress |
| 270 | normalized red CP points | objective pressure |
| 271 | normalized blue CP points | objective pressure |
| 272 | normalized red kills | combat history summary |
| 273 | normalized blue kills | combat history summary |
| 274 | normalized red deaths | combat history summary |
| 275 | normalized blue deaths | combat history summary |
| 276 | zero padding | reserved / schema filler |
| 277 | zero padding | reserved / schema filler |
| 278 | zero padding | reserved / schema filler |

### A.4 Instinct layout

| Flat index | Meaning | Source state |
|---:|---|---|
| 279 | ally archer density | neighborhood sample + unit map |
| 280 | ally soldier density | neighborhood sample + unit map |
| 281 | noisy enemy density | neighborhood sample + Gaussian perturbation |
| 282 | threat ratio | enemy density divided by ally total density |

### A.5 Semantic grouping helper for `rich_base`

| Semantic group | `rich_base` indices |
|---|---|
| `own_context` | `0,1,2,5,6,7,8` |
| `world_context` | `11,20,21,22` |
| `zone_context` | `9,10` |
| `team_context` | `3,4,12,13,14,15` |
| `combat_context` | `16,17,18,19` |
| `instinct_context` | separate 4-wide instinct block |

---

## Appendix B — Verified Action Map at a Glance

### B.1 Default 41-action map

| Action id(s) | Meaning | Engine-side interpretation | Caveat |
|---|---|---|---|
| `0` | idle | do nothing | always legal if action space exists |
| `1..8` | move | `dir_idx = action - 1`; attempt one-cell move in `DIRS8` | may fail due to wall, occupancy, or conflict |
| `9..12` | attack north, range `1..4` | decode by quotient/remainder | soldier only legal at range 1 |
| `13..16` | attack northeast, range `1..4` | same | same |
| `17..20` | attack east, range `1..4` | same | same |
| `21..24` | attack southeast, range `1..4` | same | same |
| `25..28` | attack south, range `1..4` | same | same |
| `29..32` | attack southwest, range `1..4` | same | same |
| `33..36` | attack west, range `1..4` | same | same |
| `37..40` | attack northwest, range `1..4` | same | same |

### B.2 Direction order used by moves and attacks

| Direction index | Vector |
|---:|---|
| 0 | `(0, -1)` |
| 1 | `(1, -1)` |
| 2 | `(1, 0)` |
| 3 | `(1, 1)` |
| 4 | `(0, 1)` |
| 5 | `(-1, 1)` |
| 6 | `(-1, 0)` |
| 7 | `(-1, -1)` |

### B.3 Legality rules that matter most

| Rule | Where enforced |
|---|---|
| idle always allowed | `build_mask()` |
| move only into empty in-bounds cell | `build_mask()` |
| attack only if target cell contains enemy | `build_mask()` |
| soldier range restricted to 1 | `build_mask()` |
| archer range restricted to `1..ARCHER_RANGE` | `build_mask()` |
| optional LOS wall blocking | intended in `build_mask()`, rechecked in `run_tick()` |
| masked logits before sampling/training | `run_tick()` and PPO runtime |

---

## Appendix C — Compact Mental Model of Perception-to-Action Flow

One alive agent goes through the following chain each tick:

```text
registry row + grid state + zones + global stats
    -> 32 first-hit rays (256 floats)
    -> 23 rich scalar/global features
    -> 4 instinct neighborhood features
    -> flat observation row (283 floats)
    -> split into rays_raw (32x8) and rich_vec (27)
    -> ray token + rich token
    -> MLP trunk
    -> 41 raw action logits + 1 value
    -> legality mask applied outside the brain
    -> categorical sample
    -> idle / move / attack intention
    -> engine decode and world resolution
```

The most important facts to remember are these:

1. the observation is fixed-width and schema-sensitive,
2. the ray block is absolute-color first-hit sensing, not a full world view,
3. the non-ray block mixes self, zone, and global progress information,
4. the brain outputs intentions, not guaranteed outcomes,
5. legality masking is external to the brain,
6. schema drift can silently invalidate training even when tensor shapes still match.
