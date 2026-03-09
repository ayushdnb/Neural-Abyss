# Neural-Abyss  
## Agent Intelligence Stack

### Abstract

This document reconstructs the agent intelligence stack implemented in `Neural-Abyss` from repository evidence. Its subject is the path that turns live simulation state into policy outputs and sampled actions. The stack is organized around a strict flat observation schema, a 32-ray first-hit perceptual encoder, a 27-dimensional non-ray feature tail, a family of five MLP actor-critic brains that all consume the same two-token interface, an external legality-mask layer, and per-slot execution paths that preserve agent-level parameter independence while still exploiting architecture-homogeneous batching when possible.

The current executable path is narrower and more concrete than some legacy names and comments suggest. The repository contains only MLP brain implementations in the provided agent package, while live observation construction still uses the historical method name `_build_transformer_obs`. Likewise, semantic-token utilities exist in `agent.obs_spec`, but the current MLP path does not consume them directly. Those mismatches are not defects by themselves, but they matter for correct documentation because this document is concerned with implementation truth, not naming aspiration.

The subsystem’s most important design characteristics are these:

1. **Observation is fixed-width and tightly validated.** Each alive agent receives `32 × 8 = 256` ray features plus `23 + 4 = 27` rich features, for a total observation width of `283`.
2. **Perception is hybrid.** Rays provide directional first-hit structure; rich features provide local state, unit identity, zone flags, global score statistics, and a four-value instinct summary.
3. **Brains are actor-critic MLPs with a shared front end.** Every brain variant embeds rays into one learned token and rich features into one learned token, concatenates them into a `2D` vector, and then applies a variant-specific trunk followed by actor and critic heads.
4. **Masking is external to the network.** The network emits logits for the whole discrete action space. Legality is imposed afterward by `build_mask`, and invalid logits are replaced with a large negative sentinel before categorical sampling.
5. **There is no parameter sharing between slots.** Each slot owns its own module instance and, when PPO is enabled, its own optimizer and scheduler. The repository still groups agents by architecture for inference and batched PPO math, but not for weight sharing.

The resulting implementation is easy to inspect, strongly shape-checked, and operationally aligned with mixed-precision execution and checkpoint resume; it also appears to prioritize explicit contracts over architectural novelty.

---

## Reader orientation

### What this document covers

This document covers the intelligence-specific path inside `Neural-Abyss`:

- observation construction
- ray and rich feature design
- instinct feature computation
- observation splitting and token preparation
- MLP brain families
- batched forward execution and `vmap` fallback behavior
- action-logit semantics
- legality masking
- slot organization and parameter ownership
- interfaces into PPO, checkpointing, telemetry, and viewer tooling

### What this document does not cover deeply

This document does not attempt full subsystem coverage of:

- world simulation semantics
- combat resolution rules
- movement conflict resolution
- full PPO optimization theory and telemetry schema internals
- UI rendering internals
- end-to-end checkpointing mechanics outside intelligence-relevant payloads

Those appear only when they directly constrain the intelligence stack.

### Evidence discipline used here

This document distinguishes three claim types:

**Implementation.**  
Behavior directly evidenced by code in the provided repository dump.

**Theory.**  
General mathematical framing used to explain the implementation.

**Reasoned inference.**  
A conservative interpretation of why a design appears to exist when the code does not state that motivation explicitly.

When the repository contains naming residue or comments broader than the executable path, the executable path is treated as authoritative.

---

## Executive subsystem view

The intelligence stack sits between live simulation state and discrete action selection. It does not act on full privileged world tensors in raw form. Instead, the engine constructs a per-agent observation tensor, routes aligned agent subsets through their slot-owned brains, applies legal-action masking outside the brains, samples one action per alive agent, and then records the resulting observation, masked logits, values, and actions for PPO if PPO is enabled.

```mermaid
flowchart LR
    A[Registry + Grid + Zones + Stats] --> B[_build_transformer_obs]
    B --> C[obs: N x 283]
    A --> D[build_mask]
    D --> E[mask: N x 41]
    C --> F[build_buckets(alive_idx)]
    F --> G[Architecture-homogeneous buckets]
    G --> H[ensemble_forward]
    H --> I[logits: K x A, values: K]
    E --> J[External masking]
    I --> J
    J --> K[Categorical sampling]
    K --> L[Discrete actions]
    I --> M[PPO value cache + rollout recording]
```

**Figure 1.** High-level intelligence flow in the live tick path. Observation assembly, mask generation, bucketed forward execution, masking, and sampling are separate stages. This separation is one of the subsystem’s defining architectural choices.

### Inputs to the subsystem

The intelligence stack consumes:

- slot-filtered agent state from `registry.agent_data`
- alive slot indices
- positions from `registry.positions_xy`
- grid occupancy, HP, and slot-id channels
- zone masks
- global statistics such as scores, capture points, kills, and deaths
- per-slot brain modules in `registry.brains`

### Outputs from the subsystem

The subsystem produces:

- a flat observation tensor per alive agent
- actor logits and critic values per agent
- a boolean legality mask per agent
- one sampled discrete action per agent
- PPO rollout records: observations, masked logits, values, actions, and masks
- slot-local value-cache updates used for PPO boundary bootstrapping

---

## Subsystem source map

| Module | Responsibility inside the intelligence stack |
| --- | --- |
| `engine.tick` | Builds observations, calls masks, groups agents, runs forward inference, samples actions, records PPO inputs |
| `engine.ray_engine.raycast_32` | Produces the 32-ray first-hit perceptual block |
| `engine.ray_engine.raycast_firsthit` | Builds `unit_map`, which disambiguates soldier versus archer hits |
| `engine.game.move_mask` | Computes the discrete legal-action mask |
| `agent.obs_spec` | Defines the authoritative observation splitting contract and semantic rich-token utilities |
| `agent.mlp_brain` | Defines all executable brain families and their shared input/output interface |
| `agent.ensemble` | Executes architecture-homogeneous per-slot brains via loop or `torch.func`/`vmap` |
| `engine.agent_registry` | Stores per-slot brain modules and persistent architecture metadata; builds forward buckets |
| `rl.ppo_runtime` | Consumes logits, values, actions, and masks; preserves per-slot training ownership |
| `utils.checkpointing` | Saves and restores per-slot brain kind and weights, plus PPO runtime state |
| `ui.viewer` | Inspects brain identity and model summaries for human inspection |

---

## Conceptual framing

### Implementation

The implementation solves a standard partially observed control problem, but in a specifically engineered way.

Each agent does **not** consume the full simulation state. Instead, the engine builds an observation vector that combines:

- **directional perception** from 32 raycasts,
- **entity-local scalar information** such as normalized health and position,
- **global scalar context** such as team scores and kill counts,
- **zone occupancy flags**, and
- **instinctive local-density estimates** around the agent.

A brain maps that observation to:

- `logits ∈ R^A` over a discrete action space of size `A = NUM_ACTIONS`, and
- `value ∈ R` for actor-critic training.

Legal-action constraints are enforced **after** the brain produces logits.

### Theory

A useful abstract form is:

\[
o_t^i = \phi(s_t, i)
\]

where:

- \(s_t\) is the simulation state at tick \(t\),
- \(i\) identifies an alive slot,
- \(\phi\) is the engineered observation function.

The policy and critic then take the form:

\[
\pi_{\theta_i}(a \mid o_t^i), \qquad V_{\theta_i}(o_t^i)
\]

with the crucial repository-specific detail that each slot \(i\) may own its **own** parameter set \(\theta_i\), not merely share one global \(\theta\).

The usable policy is not the raw categorical over all logits. It is the masked policy:

\[
\pi_{\theta_i}^{\text{legal}}(a \mid o_t^i, m_t^i)
\propto
\exp(\ell_a) \cdot \mathbf{1}[m_t^i(a)=1]
\]

where:

- \(\ell_a\) is the raw logit for action \(a\),
- \(m_t^i(a)\) is the legal-action indicator.

### Reasoned inference

The code strongly suggests that the system prefers **explicit information bottlenecks** over end-to-end latent perception. Rays are reduced to one summary token, rich features are reduced to one token, and masking is kept outside the network. That combination usually reflects a preference for debuggability, shape stability, and operational control rather than maximal representational flexibility.

---

## Observation system

This section is mandatory because the intelligence stack is primarily an observation-to-action system.

## Observation contract

### Implementation

The observation contract is fixed by configuration and enforced by code:

- `RAY_TOKEN_COUNT = 32`
- `RAY_FEAT_DIM = 8`
- `RAYS_FLAT_DIM = 32 * 8 = 256`
- `RICH_BASE_DIM = 23`
- `INSTINCT_DIM = 4`
- `RICH_TOTAL_DIM = 27`
- `OBS_DIM = 256 + 27 = 283`

`agent.obs_spec.split_obs_flat` and `_BaseMLPBrain.__init__` both validate these relations. `engine.tick._build_transformer_obs` builds exactly this layout and raises on mismatch.

A crucial implementation truth follows from the code, not from configuration names: although `config.RAY_TOKEN_COUNT` exists as an environment knob, the live observation builder and ray engine are hard-wired to 32 rays. `engine.tick` allocates rich scratch using `32 * 8`, `expected_ray_dim` is hard-coded as `32 * 8`, and the live ray builder is `raycast32_firsthit`. The subsystem is therefore **effectively fixed at 32 rays** in the provided codebase.

### Observation layout

\[
o = [rays\_flat \;|\; rich\_base \;|\; instinct]
\]

with shapes:

- `rays_flat`: `(B, 256)`
- `rich_base`: `(B, 23)`
- `instinct`: `(B, 4)`

`split_obs_for_mlp` reshapes the ray block into `(B, 32, 8)` and concatenates `rich_base` with `instinct` into `(B, 27)`.

---

## Ray features

### Implementation

`engine.tick._build_transformer_obs` calls `raycast32_firsthit(pos_xy, grid, unit_map, max_steps_each=alive_data[:, COL_VISION].long())`.

The ray subsystem therefore uses:

- **32 directions** uniformly spaced over the circle,
- **per-agent maximum range** determined by `COL_VISION`,
- **first-hit semantics**, not dense occupancy profiles.

Each ray emits exactly eight features:

1. one-hot `none`
2. one-hot `wall`
3. one-hot `red-soldier`
4. one-hot `red-archer`
5. one-hot `blue-soldier`
6. one-hot `blue-archer`
7. normalized hit distance
8. normalized HP at the first-hit cell

This yields `32 × 8 = 256` floats per agent.

### First-hit resolution

The ray engine marches all agents, all 32 rays, and all allowed steps in one vectorized batch. It determines:

- the first wall hit index,
- the first agent hit index,
- whichever happens earlier,
- with **ties broken in favor of walls**.

Agent hits are disambiguated using both:

- `grid[0]` for team encoding (`2` red, `3` blue),
- `unit_map` for subtype encoding (`1` soldier, `2` archer).

### Normalization

- `dist_norm` is first-hit distance divided by the agent’s own `max_steps_each`, clamped by `clamp_min(1)`.
- `hp_norm` is first-hit HP divided by `MAX_HP`.

### Perceptual biases introduced by the ray implementation

The ray encoding is compact and useful, but it is not a physically exact visibility model.

1. **Only the first hit is retained.**  
   Everything behind the first encountered wall or agent on a ray is discarded.

2. **Continuous directions are discretized by truncation to integer grid indices.**  
   Shallow angles can map multiple consecutive steps to the same cell.

3. **Coordinates are clamped to map bounds rather than terminated by explicit out-of-bounds logic.**  
   Near edges, later ray steps can repeatedly sample the boundary cell.

These are not documentation conjectures; they follow from the actual `raycast32_firsthit` implementation.

---

## Rich scalar/context features

### Implementation

After ray construction, `_build_transformer_obs` fills a 23-column `rich_base` block directly into preallocated tensor storage. The feature definitions are exact:

| Index | Meaning | Formula / source |
| --- | --- | --- |
| 0 | normalized HP | `COL_HP / clamp_min(COL_HP_MAX, 1)` |
| 1 | normalized x-position | `COL_X / (W - 1)` |
| 2 | normalized y-position | `COL_Y / (H - 1)` |
| 3 | is red team | `COL_TEAM == 2.0` |
| 4 | is blue team | `COL_TEAM == 3.0` |
| 5 | is soldier | `COL_UNIT == 1.0` |
| 6 | is archer | `COL_UNIT == 2.0` |
| 7 | normalized attack power | `COL_ATK / MAX_ATK` |
| 8 | normalized vision | `COL_VISION / RAYCAST_MAX_STEPS` |
| 9 | on heal zone | zone lookup |
| 10 | on capture point | OR over capture-point masks |
| 11 | normalized tick | `stats.tick / 50000` |
| 12 | normalized red score | `stats.red.score / 1000` |
| 13 | normalized blue score | `stats.blue.score / 1000` |
| 14 | normalized red capture points | `stats.red.cp_points / 500` |
| 15 | normalized blue capture points | `stats.blue.cp_points / 500` |
| 16 | normalized red kills | `stats.red.kills / 500` |
| 17 | normalized blue kills | `stats.blue.kills / 500` |
| 18 | normalized red deaths | `stats.red.deaths / 500` |
| 19 | normalized blue deaths | `stats.blue.deaths / 500` |
| 20 | zero padding | explicitly zeroed |
| 21 | zero padding | explicitly zeroed |
| 22 | zero padding | explicitly zeroed |

**Table 1.** Exact `rich_base` feature inventory.

### Interpretation

The rich block mixes three conceptually different information classes:

- **self-state:** health, position, unit type, attack, vision
- **zone state:** heal/capture-point occupancy
- **global context:** tick, team scores, team kills, team deaths

This means the policy is not purely egocentric. Every alive agent receives the same global-score features during a given tick.

### Important implementation note

Columns `20:23` are explicitly zero-filled every tick. They are not missing by accident; they are part of the stable schema and therefore part of the downstream contract.

---

## Instinct features

### Implementation

The final four observation components are not copied from raw agent state. They are computed by `_compute_instinct_context`, which returns:

1. `ally_archer_density`
2. `ally_soldier_density`
3. `noisy_enemy_density`
4. `threat_ratio = enemy_density / (ally_total_density + eps)`

The function builds a discrete circle of offsets of radius `INSTINCT_RADIUS`, samples the map around every alive agent in a vectorized broadcast, classifies nearby units by team and subtype, subtracts self-count from ally totals, injects Gaussian noise into enemy count, normalizes by discrete circle area, and forms the threat ratio.

### Exact mechanics

Let `offsets = {(dx,dy): dx² + dy² ≤ R²}`.

For each alive agent:

- neighborhood coordinates are generated by broadcast addition,
- coordinates are clamped to bounds,
- `grid[0]` supplies team occupancy,
- `unit_map` supplies subtype,
- ally and enemy counts are accumulated over the sampled offsets.

The code then computes:

\[
ally\_arch\_d = \frac{ally\_arch\_count}{area}, \quad
ally\_sold\_d = \frac{ally\_sold\_count}{area}, \quad
enemy\_d = \frac{\max(enemy\_count + \epsilon_{noise}, 0)}{area}
\]

and

\[
threat = \frac{enemy\_d}{ally\_arch\_d + ally\_sold\_d + \varepsilon}
\]

where:

- Gaussian noise has standard deviation `0.25`,
- `eps = 1e-4` in `float16`, else `1e-6`.

### Consequences

This instinct block is a hand-engineered mesoscale context feature. It gives the network population-pressure and local-force-ratio information without requiring the MLP to infer such structure solely from ray hits.

### Biases and limitations

1. **Stochastic observation.**  
   Enemy density includes Gaussian noise, so the same world state can produce different instinct features if RNG state differs.

2. **Border distortion through clamping.**  
   Out-of-bounds offsets are clamped back to the map edge, so border agents can sample the same edge cell multiple times. This is a real implementation bias.

3. **Subtype asymmetry.**  
   Ally subtype densities are separated into archer and soldier, while enemy subtype is collapsed into one noisy density.

---

## Semantic-token scaffolding

### Implementation

`agent.obs_spec.build_semantic_tokens` can materialize six semantic groups:

- `own_context`
- `world_context`
- `zone_context`
- `team_context`
- `combat_context`
- `instinct_context`

The configured index groups are:

- `own_context`: `(0, 1, 2, 5, 6, 7, 8)`
- `world_context`: `(11, 20, 21, 22)`
- `zone_context`: `(9, 10)`
- `team_context`: `(3, 4, 12, 13, 14, 15)`
- `combat_context`: `(16, 17, 18, 19)`
- `instinct_context`: direct instinct tensor

### Critical scope note

This semantic-token utility exists, but the current MLP path does **not** call it. `_BaseMLPBrain` uses `split_obs_for_mlp`, not `build_semantic_tokens`. Therefore, semantic tokenization is presently **available as schema infrastructure**, not part of the live brain forward path.

### Reasoned inference

This looks like architectural residue or reserved infrastructure for more structured models. That inference is conservative because:

- the semantic grouping code is real,
- the current MLP path bypasses it,
- no executable non-MLP brain classes appear in the provided repository dump.

---

## Feature representation and preprocessing

## Observation splitting

### Implementation

The authoritative observation parser lives in `agent.obs_spec`:

- `split_obs_flat(obs)` verifies rank-2 `(B,F)` input and exact `OBS_DIM`
- splits the flat observation into `(rays_flat, rich_base, instinct)`
- `split_obs_for_mlp(obs)` reshapes rays into `(B, 32, 8)` and concatenates rich blocks into `(B, 27)`

This is an important design decision: the brain module does **not** duplicate slice boundaries.

## Shared two-token front end

Every MLP brain uses exactly the same front end:

1. `rays_raw ∈ R^{B×32×8}`
2. layer-normalize each ray over its 8 feature channels
3. linearly project each ray `8 → d_model`
4. mean-reduce over the 32 rays to get one `ray_token ∈ R^{B×d_model}`
5. layer-normalize the 27-dim rich vector
6. linearly project rich `27 → d_model`
7. concatenate `[ray_token, rich_token]`
8. apply final normalization
9. feed the resulting `B × (2d_model)` tensor into the variant trunk

With default configuration, `d_model = 32`, so the final shared input width is `64`.

```mermaid
flowchart LR
    A[obs: B x 283] --> B[split_obs_for_mlp]
    B --> C[rays_raw: B x 32 x 8]
    B --> D[rich_vec: B x 27]
    C --> E[LayerNorm(8)]
    E --> F[Linear 8->32]
    F --> G[Mean over 32 rays]
    D --> H[LayerNorm(27)]
    H --> I[Linear 27->32]
    G --> J[Concat]
    I --> J
    J --> K[input_norm]
    K --> L[final flat input: B x 64]
```

**Figure 2.** Shared MLP input pipeline. All five brain families differ only after the final `B × 64` input vector has been constructed.

## Normalization and dtype behavior

### Implementation

The code uses a deliberate dtype pattern:

- ray and rich normalization are done in `float()`
- normalized activations are then cast back toward the destination tensor or layer dtype
- actor and critic heads explicitly cast hidden activations to the head weight dtype before applying `Linear`

This makes the subsystem compatible with mixed precision while keeping numerically sensitive normalization steps in full precision.

### Important nuance

`BRAIN_MLP_NORM` does **not** control every normalization site.

- `ray_in_norm` is always `LayerNorm(8)`
- `rich_in_norm` is always `LayerNorm(27)`
- `input_norm` uses `_maybe_norm`, so config can disable it
- residual/gated/bottleneck block norms also use `_maybe_norm`

Likewise, `BRAIN_MLP_ACTIVATION` only affects the custom block internals. The plain sequential trunks use hard-coded `GELU`.

That distinction matters because configuration names are broader than their actual effect.

## Feature completeness invariants

The subsystem aggressively fails on schema drift. It raises hard runtime errors for:

- wrong observation rank
- wrong observation width
- wrong ray width after split
- wrong rich width after split
- wrong final shared input width
- wrong actor or critic output shape

This is not incidental defensive programming. In a slot-local PPO system, silent misalignment between observation layout and model contract would corrupt training quickly and opaquely.

---

## Model and brain families

## Shared contract

All executable brain classes inherit from `_BaseMLPBrain` and obey the same forward interface:

\[
forward(obs: B \times obs\_dim) \rightarrow (logits: B \times act\_dim,\ value: B \times 1)
\]

Every brain therefore shares:

- the same observation layout
- the same ray/rich tokenization logic
- the same actor and critic head semantics
- the same output dimensionality

Only the trunk differs.

## Variant inventory

| Brain kind | Trunk shape after shared input | Structural idea | Trunk output width |
| --- | --- | --- | --- |
| `whispering_abyss` | `64 → 96 → 96` | plain compact MLP | 96 |
| `veil_of_echoes` | `64 → 128 → 96 → 64` | deeper narrowing MLP | 64 |
| `cathedral_of_ash` | `64 → 80 → 3 residual blocks` | fixed-width residual refinement | 80 |
| `dreamer_in_black_fog` | `64 → 80 → 2 gated blocks` | gated residual refinement | 80 |
| `obsidian_pulse` | `64 → 128 → 2 bottleneck residual blocks` | wide outer state with compressed inner path | 128 |

**Table 2.** Executable MLP brain families.

### Whispering Abyss

A plain two-layer GELU MLP:

- `Linear(64,96)`
- `GELU`
- `Linear(96,96)`
- `GELU`

This is the simplest variant and serves as the cleanest baseline.

### Veil of Echoes

A deeper narrowing MLP:

- `Linear(64,128)`
- `GELU`
- `Linear(128,96)`
- `GELU`
- `Linear(96,64)`
- `GELU`

This introduces one extra compression stage before the heads.

### Cathedral of Ash

A fixed-width residual stack:

- entry `Linear(64,80)` + `GELU`
- three `_ResidualBlock(80)`

Each residual block performs:

\[
y = x + W_2(\sigma(W_1(Norm(x))))
\]

with configurable normalization and activation inside the block.

### Dreamer in Black Fog

A gated residual stack:

- entry `Linear(64,80)` + `GELU`
- two `_GatedBlock(80)`

Each gated block computes:

\[
y = x + W_o(\sigma(W_v(Norm(x))) \odot sigmoid(W_g(Norm(x))))
\]

This is still an MLP, but it introduces feature-wise modulation.

### Obsidian Pulse

A bottleneck residual stack:

- entry `Linear(64,128)` + `GELU`
- two `_BottleneckResidualBlock(128,48)`

Each block compresses to `48`, applies nonlinearity, expands back to `128`, and adds the residual.

## Actor and critic heads

Every trunk feeds two heads:

- `actor_head: Linear(trunk_out, act_dim)`
- `critic_head: Linear(trunk_out, 1)`

So the family variation changes latent processing, not output semantics.

## Initialization policy

All linear layers are orthogonally initialized, with special gains:

- hidden layers: `sqrt(2)`
- actor head: `0.01`
- critic head: `1.0`

This is a common actor-critic choice: small initial actor gain prevents excessively confident starting policies, while critic magnitude is left less suppressed.

---

## Forward inference path

## High-level path

The live forward path in `engine.tick.run_tick` is:

1. compute `alive_idx`
2. build `obs = _build_transformer_obs(alive_idx, pos_xy)`
3. build `mask = build_mask(...)`
4. group alive slots into architecture-homogeneous buckets
5. run `ensemble_forward(bucket.models, bucket_obs)` for each bucket
6. apply mask outside the model
7. sample from `Categorical(logits=masked_logits)`

## Bucket construction

### Implementation

`AgentsRegistry.build_buckets(alive_idx)` groups alive slots by persistent architecture class id, not by parameter identity. The registry maintains:

- `brains`: per-slot `nn.Module` list
- `brain_arch_ids`: per-slot compact architecture id
- `signature_to_arch_id` and `arch_id_to_signature`

The architecture signature is built from:

- model class name
- all `nn.Linear(in_features,out_features)` pairs encountered in named modules

This makes the grouping key structural rather than weight-based.

The bucket output contains:

- `indices`: slot ids in the bucket
- `models`: corresponding brain objects
- `locs`: positions of those slots within `alive_idx`

The `locs` tensor exists specifically so the caller does not need a per-tick `searchsorted` alignment step.

## `ensemble_forward`

### Implementation

`agent.ensemble.ensemble_forward(models, obs)` is the public execution gateway. It chooses between:

- `_ensemble_forward_loop`
- `_ensemble_forward_vmap`

depending on:

- `USE_VMAP`
- bucket size `K >= VMAP_MIN_BUCKET`
- availability of `torch.func`
- absence of TorchScript modules
- runtime success of the `vmap` path

If `vmap` is unavailable or fails, execution falls back to the canonical safe Python loop.

### `_DistWrap`

The brains themselves return raw logits and values. `ensemble_forward` wraps the batched logits in a lightweight `_DistWrap` object exposing only `.logits`. That wrapper is not a full probability distribution. It exists to satisfy downstream expectations without tying the code to a specific `torch.distributions` object at the forward boundary.

## Loop path

The loop path:

- feeds each model one observation row at a time using `obs[i].unsqueeze(0)`
- validates the return contract
- preallocates output tensors after inspecting the first model output
- concatenates logits into `(K,A)` and values into `(K,)`

This path is slower but robust.

## `vmap` path

The `vmap` path uses `torch.func.functional_call`, `vmap`, and `stack_module_state` to evaluate multiple independent parameter sets in parallel.

The key idea is:

\[
f(\theta_i, x_i) \rightarrow (logits_i, value_i)
\]

and then

\[
vmap(f)(\Theta, X)
\]

over the leading bucket dimension.

### Cache design

The stacked module state is cached using:

- ordered model identity
- parameter and buffer object identity
- tensor version counters
- device, dtype, and shape information

This is unusually careful and important. It allows cache reuse across ticks while still invalidating automatically after in-place parameter or buffer mutation such as optimizer steps or checkpoint loads.

### Output contract

Both loop and `vmap` paths enforce:

- logits shape `(K,A)`
- values shape `(K,)`

Any mismatch raises immediately.

---

## Forward-pass pseudocode

```text
alive_idx = recompute_alive_idx()
pos_xy = registry.positions_xy(alive_idx)
obs = _build_transformer_obs(alive_idx, pos_xy)
mask = build_mask(pos_xy, teams, grid, unit)

for bucket in registry.build_buckets(alive_idx):
    bucket_obs = obs[bucket.locs]
    bucket_mask = mask[bucket.locs]

    dist, vals = ensemble_forward(bucket.models, bucket_obs)
    logits32 = where(bucket_mask, dist.logits.float(), -inf)
    a = Categorical(logits=logits32).sample()

    record bucket_obs, logits32, vals, a, bucket_mask if PPO is enabled
    actions[bucket.locs] = a
```

**Figure 3.** Exact logical structure of the decision-time path. The mask is applied after forward inference, not inside the brain.

---

## Action-space interface and masking

## Discrete action layout

### Implementation

`engine.game.move_mask.build_mask` defines the action legality interface.

For the 41-action layout used by current configuration:

- action `0`: idle
- actions `1..8`: movement in 8 directions
- actions `9..40`: attack actions arranged as 8 directional blocks × 4 ranges

For attack actions:

- direction index: `(action - 9) // 4`
- range: `((action - 9) % 4) + 1`

### Unit gating

For the 41-action layout:

- soldiers can attack only at range 1
- archers can attack at ranges `1..ARCHER_RANGE`, clipped to `4`

### LOS gating

If `ARCHER_LOS_BLOCKS_WALLS` is enabled, attack legality is additionally constrained by wall checks along intermediate cells.

### Important discrepancy

The move-mask docstring still describes a default of 17 actions in one place, but the live configuration default is `NUM_ACTIONS = 41`. The implementation itself is clear and supports both layouts.

## External mask application

### Implementation

Masking occurs in `engine.tick`, not in the brain:

```python
logits32 = torch.where(bucket_mask, dist.logits.to(torch.float32), neg_inf)
a = torch.distributions.Categorical(logits=logits32).sample()
```

So the brain always emits the full action-space logits, and the environment layer is responsible for suppressing invalid actions.

This same masked policy is then recorded for PPO. `PerAgentPPORuntime.record_step` stores the logits and masks, recomputes masked log-probabilities, and optionally validates that:

- every mask row has at least one legal action
- every chosen action is legal under that mask

### Why this matters conceptually

External masking creates a clean separation:

- the **brain** scores all actions,
- the **environment interface** decides which scores are admissible.

That keeps legality logic centralized and guarantees that combat, movement, and unit-class constraints are not duplicated inside the neural architecture.

### Failure-mode implications

This design also introduces a hard invariant:

\[
\text{mask width} = \text{logit width} = NUM\_ACTIONS
\]

If the action encoding changes without synchronized changes to:

- mask construction,
- actor-head width,
- action decoding in `run_tick`,
- PPO rollout validation,

the subsystem will break or train on nonsense.

---

## Policy and value semantics

### Implementation

The brain outputs:

- `logits`: unnormalized action preferences
- `value`: scalar critic estimate

No deterministic greedy execution path is evidenced in the live tick path. The main loop samples from a categorical distribution over the masked logits.

### Value semantics

The value output is used in two ways:

1. recorded for PPO rollout and GAE-based training
2. copied into a persistent slot-local value cache

That cache is important because it allows the engine to bootstrap PPO window boundaries from the **next tick’s normal forward pass** instead of running a second dedicated bootstrap-only inference pass.

### Distribution semantics

The effective behavior policy is:

- raw actor logits from the brain
- converted to `float32`
- invalid actions replaced by a very negative sentinel
- sampled via `torch.distributions.Categorical`

The repository therefore treats masked logits, not raw logits, as the behavior-policy truth recorded into PPO.

---

## Parameter sharing and agent organization

## No-hive-mind design

### Implementation

The strongest intelligence-stack design choice in the repository is explicit slot-local independence.

`PerAgentPPORuntime` states this directly:

- each slot owns its own model parameters
- each slot owns its own optimizer
- each slot owns its own scheduler
- rollout buffers are also slot-local

`AgentsRegistry.brains` is a Python list of per-slot modules. `reset_agent` discards slot-local optimizer, scheduler, and rollout buffer state when a new individual occupies an old slot.

### Architectural grouping without weight sharing

Even though parameters are independent, the runtime still groups by architecture:

- `build_buckets` groups by structural signature for inference
- batched PPO groups compatible slots into architecture-homogeneous lanes for forward/loss math

This is a subtle but important hybrid:

- **no sharing of learned parameters**
- **sharing of computational shape and dispatch structure**

## Brain-family assignment across the population

### Implementation

New brains are created through `create_mlp_brain(kind, obs_dim, act_dim)`. The kind can be chosen from configuration via:

- `BRAIN_KIND`
- `TEAM_BRAIN_ASSIGNMENT`
- `TEAM_BRAIN_ASSIGNMENT_MODE`
- `TEAM_BRAIN_MIX_STRATEGY`
- team-specific exclusive kinds
- probabilistic mixture weights

Spawn and respawn code can therefore produce:

- one global architecture for all slots
- exclusive team-specific architectures
- alternating or probabilistically mixed architectures within teams

### Clone behavior

Respawn cloning is conservative:

- if the child’s chosen kind matches the parent kind, `state_dict` is loaded
- if not, a fresh brain of the target kind is created without incompatible weight transfer

That avoids accidental cross-architecture `state_dict` misuse.

---

## Tensorization, vectorization, and device flow

## Tensor contracts

| Interface | Shape | Meaning |
| --- | --- | --- |
| `obs` in tick path | `(N_alive, 283)` | flat observation for each alive agent |
| `rays_raw` in brain front end | `(B, 32, 8)` | structured ray tokens |
| `rich_vec` in brain front end | `(B, 27)` | full non-ray tail |
| shared MLP input | `(B, 64)` by default | concatenated ray/rich tokens |
| brain logits | `(B, 41)` by default | raw action scores |
| brain values | `(B, 1)` | scalar critic output |
| mask | `(N_alive, 41)` | legal-action indicator |
| `ensemble_forward` values | `(K,)` | flattened per-agent values |
| PPO action masks | `(B, act_dim)` | stored rollout legality mask |

**Table 3.** Core tensor interfaces across the intelligence stack.

## Device flow

### Implementation

The live engine keeps intelligence computation on the simulation device:

- registry tensors live on `config.TORCH_DEVICE`
- observations are built on `grid.device`
- brains are moved onto the registry device through `set_brain`
- raycasting, instinct computation, masks, and inference all occur on device

The code repeatedly avoids host synchronization in hot paths by:

- using preallocated device tensors,
- batching `.tolist()` transfers where unavoidable,
- caching vectorization state,
- keeping output buffers reusable.

## Mixed precision behavior

`config.TORCH_DTYPE` becomes `float16` when CUDA and AMP are enabled, else `float32`. The intelligence stack respects that choice but selectively casts to `float32` for normalization and PPO bookkeeping.

## Grouped PPO math

### Implementation

The PPO runtime contains a grouped batched execution path:

- it batches compatible slots into homogeneous lanes,
- uses `functional_call` and `vmap`,
- computes logits, values, and entropies in batch,
- scatters gradients back to original models,
- still steps each slot’s optimizer independently.

The code explicitly forbids optimizer sharing and disables grouped batched PPO for attention-based brains. In the provided dump, no attention-based brain classes exist, but the guard is present.

### Why this matters

Inference and PPO both follow the same architectural principle:

- **batch the math when architecture matches**
- **do not share learning state**

That principle defines much of the subsystem’s engineering character.

---

## Mathematical and conceptual framing

## Observation function

The implemented observation map can be written as:

\[
o_i = \phi_i(s)
      =
      \big[
      \rho_i(s),\;
      \eta_i(s),\;
      \kappa_i(s)
      \big]
\]

where:

- \(\rho_i(s) \in \mathbb{R}^{256}\) is the 32-ray first-hit block,
- \(\eta_i(s) \in \mathbb{R}^{23}\) is the rich scalar/context block,
- \(\kappa_i(s) \in \mathbb{R}^{4}\) is the instinct block.

## Two-token projection

The MLP front end then computes:

\[
R_i \in \mathbb{R}^{32 \times 8}
\rightarrow
\hat{R}_i \in \mathbb{R}^{32 \times D}
\rightarrow
r_i = \frac{1}{32} \sum_{j=1}^{32} \hat{R}_{ij}
\]

and

\[
q_i = W_{rich} \, LN(rich_i) \in \mathbb{R}^{D}
\]

then concatenates

\[
x_i = LN([r_i ; q_i]) \in \mathbb{R}^{2D}
\]

With default configuration, \(D = 32\), so \(x_i \in \mathbb{R}^{64}\).

## Policy and value heads

Each architecture-specific trunk \(T_k\) produces hidden state \(h_i\), then

\[
\ell_i = W_{\pi} h_i + b_{\pi}, \qquad
v_i = W_V h_i + b_V
\]

where:

- \(\ell_i \in \mathbb{R}^{A}\) are raw logits,
- \(v_i \in \mathbb{R}\) is the critic prediction.

## External legality mask

Let \(m_i \in \{0,1\}^{A}\) be the legal-action mask. The effective logits are:

\[
\tilde{\ell}_{ia}
=
\begin{cases}
\ell_{ia}, & m_{ia}=1 \\
-\infty_{\text{approx}}, & m_{ia}=0
\end{cases}
\]

where the code uses the minimum finite representable value of the chosen floating dtype rather than literal negative infinity.

The sampled action is then:

\[
a_i \sim Categorical(\tilde{\ell}_i)
\]

## Slot-local parameterization

Because the repository uses per-slot modules, one may write the live policy family as:

\[
\pi_{\theta_1}, \pi_{\theta_2}, \ldots, \pi_{\theta_n}
\]

rather than one globally shared \(\pi_{\theta}\). Grouping is computational, not statistical.

---

## Interfaces to other subsystems

## Interface to the simulation engine

The intelligence stack reads from:

- `registry.agent_data`
- `registry.positions_xy`
- `grid`
- `zones`
- `stats`

and writes back only indirectly via sampled action integers. The engine then decodes these integers into combat and movement semantics.

For attacks in the 41-action layout:

- direction is `(a - 9) // 4`
- range is `((a - 9) % 4) + 1`

For movement:

- actions `1..8` select one of `DIRS8`

This means the actor head is tightly coupled to the engine’s discrete action codec.

## Interface to PPO runtime

The intelligence stack supplies PPO with:

- `agent_ids`
- `obs`
- masked `logits`
- `values`
- sampled `actions`
- `rewards`
- `done`
- `action_masks`

It also updates a persistent slot-local value cache from the normal forward pass, which PPO uses to finalize pending rollout windows.

This is a notable interface refinement: it eliminates a duplicate bootstrap-only forward branch.

## Interface to checkpointing

Checkpointing saves, per slot:

- brain kind string
- brain `state_dict`

and restores them through `_make_brain(kind, device)` plus `load_state_dict`.

PPO checkpoint state additionally stores:

- per-slot rollout buffers
- per-slot optimizer state
- per-slot scheduler state
- value cache
- pending window bootstrap metadata

So the intelligence stack is fully resumable, including training momentum and unfinished PPO windows.

## Interface to telemetry and viewer systems

The intelligence stack is visible operationally in at least three ways:

1. viewer inspection uses `brain_kind_from_module`, display names, short labels, and model summaries;
2. telemetry records PPO reward-component data by slot;
3. viewer tooling can save a selected brain’s `state_dict`.

These interfaces do not change policy semantics, but they are part of the subsystem’s inspectability.

---

## Design decisions and trade-offs

## 1. Fixed 32-ray first-hit perception

**Implementation.**  
The live path uses `raycast32_firsthit` and hard-coded 32-ray assumptions in the engine.

**Likely motivation.**  
A fixed-width directional sensor is simple, cheap, and easy to normalize.

**Enables.**  
Compact spatial awareness with predictable tensor sizes.

**Costs.**  
Loss of detail behind first hits, discretization artifacts, and effective hard-coding of ray count.

**Alternative.**  
Dense local spatial grids, learned convolutions, or multi-hit ray histories.

## 2. Explicit instinct features

**Implementation.**  
The observation builder computes ally-type densities, noisy enemy density, and threat ratio.

**Likely motivation.**  
Give the policy direct local-force-ratio information without requiring deep inference from sparse ray hits.

**Enables.**  
Faster access to mesoscale tactical context.

**Costs.**  
Hand-crafted bias, stochastic observation noise, and border distortion via clamping.

**Alternative.**  
Purely learned local aggregation from raw spatial input.

## 3. Two-token MLP interface

**Implementation.**  
Rays become one learned token, rich features become one learned token, then both are concatenated.

**Likely motivation.**  
Retain some modality separation while keeping the trunk MLP-compatible and cheap.

**Enables.**  
A single shared input contract across all variants.

**Costs.**  
Aggressive compression of ray structure. The model cannot attend differently to individual rays after the mean summary.

**Alternative.**  
Per-ray attention, sequence models, or direct flattened-ray MLPs.

## 4. External legality masking

**Implementation.**  
Masks are computed by the engine and applied to logits after forward inference.

**Likely motivation.**  
Centralize legality logic and avoid teaching the network engine rules implicitly.

**Enables.**  
Clean separation between scoring and rule enforcement.

**Costs.**  
Strong coupling between actor width, mask width, and engine action codec.

**Alternative.**  
Internal masked heads or rule-conditioned network outputs.

## 5. No parameter sharing across slots

**Implementation.**  
Per-slot brains, per-slot optimizers, per-slot schedulers, per-slot rollout buffers.

**Likely motivation.**  
Avoid homogenization and allow true individual divergence.

**Enables.**  
Evolution-like behavioral diversity and slot-local learning histories.

**Costs.**  
Higher memory, more optimizer state, and more complicated batched execution.

**Alternative.**  
One shared policy with agent-conditioning features.

## 6. Architecture grouping without learning-state sharing

**Implementation.**  
Buckets and grouped PPO lanes batch compatible slots while preserving independent parameters and optimizers.

**Likely motivation.**  
Recover throughput without giving up the no-hive-mind design.

**Enables.**  
A middle ground between total independence and global sharing.

**Costs.**  
More complicated dispatch logic, `vmap` compatibility guards, and cache invalidation logic.

## 7. Strong runtime validation

**Implementation.**  
Shape checks, mask checks, action-range checks, and checkpoint compatibility checks are pervasive.

**Likely motivation.**  
This codebase is intended for long-running simulations where silent corruption is expensive.

**Enables.**  
Fail-fast debugging and safer resume behavior.

**Costs.**  
Slight runtime overhead and stricter coupling between subsystems.

---

## Extension guidance

## Adding a new rich feature

A safe extension requires coordinated edits in at least these places:

1. `engine.tick._build_transformer_obs` to write the feature
2. `config.RICH_BASE_DIM` or `INSTINCT_DIM` if dimensionality changes
3. `config.OBS_DIM` through derived expressions
4. `agent.obs_spec.split_obs_flat` and any semantic-token index groups affected
5. any checkpoint compatibility expectations if old weights must still load

If the new feature changes dimensionality but the model or split code is not updated, `_BaseMLPBrain` and `obs_spec` will raise.

## Adding a new brain family

A safe path is:

1. define a new subclass of `_BaseMLPBrain`
2. implement `_build_variant_trunk`
3. add the kind to `create_mlp_brain`
4. register display and short labels in config
5. add it to architecture-selection config validation
6. ensure checkpoint `_make_brain` recognizes the new kind
7. verify `build_buckets` groups it correctly through the linear-layer signature

If the architecture includes components unsupported by grouped `vmap` paths, the system will fall back to safe sequential execution, but only if the guards are correct.

## Changing the action space

This is high-risk and requires synchronized updates to:

- `config.NUM_ACTIONS`
- actor-head width
- `build_mask`
- attack and movement decoding in `engine.tick`
- PPO rollout mask assertions
- checkpoint compatibility for head shapes

## Instrumenting intermediate activations

The safest place to instrument is inside `_BaseMLPBrain._build_flat_input` or immediately before actor/critic heads. That preserves the single shared front end and avoids duplicating instrumentation across variants.

---

## Failure modes, risks, and limitations

## 1. Schema drift risk

The subsystem is only safe because many constants move together. In practice, these must remain synchronized:

- 32 ray count
- 8 ray feature width
- 23 rich-base width
- 4 instinct width
- 41 action width

Changing only configuration is not enough.

## 2. Legacy naming can mislead readers

The live observation builder is still named `_build_transformer_obs`, and some comments elsewhere mention non-MLP families. In the provided executable agent code, only the MLP family is present. Documentation or tooling that relies on names instead of call paths could mischaracterize the subsystem.

## 3. World-context semantic token is partly dead capacity

`world_context` maps to rich-base indices `(11,20,21,22)`, but indices `20:22` are always zero in the live builder. Any future consumer of semantic tokens must understand that this token currently contains one active feature and three zeros.

## 4. Observation stochasticity from instinct noise

The instinct block injects Gaussian noise into enemy counts. This can improve robustness, but it also means observation is not purely deterministic from spatial state alone. Reproducible resumes therefore depend on restoring RNG state correctly, which the checkpoint layer does.

## 5. Border artifacts

Both instinct computation and ray marching use clamping to bounds. Border agents therefore perceive the world through a biased geometry that effectively repeats edge cells.

## 6. `vmap` compatibility fragility

The grouped execution path depends on:

- homogeneous architecture
- compatible module structure
- non-TorchScript models
- available `torch.func` support

The code handles this with fallbacks, but performance behavior can vary sharply across model families and PyTorch builds.

## 7. External masking makes interface mismatch catastrophic

Because legality is enforced outside the brain, any mismatch between:

- action encoding,
- mask construction,
- and engine action decode

can produce impossible actions, invalid PPO records, or silent training pathologies if checks are disabled.

## 8. Mean ray summarization is lossy

Averaging 32 embedded rays into one token removes directional individuality before the trunk sees the data. This is computationally efficient, but it discards potentially important anisotropic structure.

---

## Conclusion

The agent intelligence stack in `Neural-Abyss` is a deliberately engineered actor-critic decision system built around a strict observation schema, per-slot neural individuality, and computational batching without parameter sharing. Its live path is clearer and narrower than some historical names imply: the provided code executes a five-variant MLP family, not a transformer family, and its brains consume a two-token compression of a 32-ray perceptual block plus a 27-dimensional non-ray tail.

The subsystem’s central architectural idea is not model novelty. It is **contract discipline**. Observation width is fixed and validated. Brain input structure is shared across all variants. Logits and values have hard shape checks. Legal-action masking is explicit and external. Bucketed execution and grouped PPO are used to recover throughput without giving up slot-local learning ownership. Checkpointing preserves both model state and training state. Viewer and telemetry layers can inspect the resulting intelligence state without altering it.

That combination makes the intelligence stack technically modest in architecture but strong in operational semantics. It appears designed for long-running, inspectable, resumable multi-agent simulation rather than for maximum representational complexity.

---

## Glossary

**Action mask**  
A boolean tensor indicating which discrete actions are currently legal for each agent.

**Actor head**  
The final linear layer that maps trunk features to action logits.

**Brain kind**  
The string identifier for a concrete MLP architecture variant such as `whispering_abyss` or `obsidian_pulse`.

**Bucket**  
A group of alive agents whose brains share the same structural architecture signature and can therefore be executed together.

**Critic head / value head**  
The final linear layer that maps trunk features to a scalar value estimate.

**First-hit ray feature**  
A ray encoding that records only the earliest wall or agent encountered along a direction.

**Instinct context**  
A four-value engineered local-density summary: ally-archer density, ally-soldier density, noisy enemy density, and threat ratio.

**Logits**  
Unnormalized action scores produced by the actor head before masking and categorical sampling.

**Observation**  
The flat per-agent tensor input to the brain, width `283` in the current implementation.

**Policy**  
The action-selection rule induced by masked logits and categorical sampling.

**Rich features**  
The non-ray observation tail containing scalar local state, zone flags, and global statistics.

**Semantic tokens**  
Named subsets of the rich feature block defined in `obs_spec`; present as infrastructure but not used by the live MLP path.

**Slot**  
A fixed registry index that owns agent state and, when occupied, one brain module.

**Two-token interface**  
The shared MLP front end in which all rays are reduced to one learned token and all non-ray features are reduced to one learned token before trunk processing.

**`vmap` path**  
A `torch.func`-based execution path that evaluates multiple independent models in batch without sharing parameters.

---

## Appendix A. Module-to-responsibility table

| Module | Intelligence-relevant responsibility | Notes |
| --- | --- | --- |
| `engine.tick` | builds observations, runs action selection, updates PPO value cache | method name `_build_transformer_obs` is legacy |
| `engine.ray_engine.raycast_32` | emits the live 32-ray perceptual block | fixed 32-ray implementation |
| `engine.ray_engine.raycast_firsthit` | builds `unit_map` used by ray and instinct code | resolves soldier vs archer subtype |
| `agent.obs_spec` | authoritative observation slicing and semantic grouping helpers | semantic groups currently not consumed by MLP path |
| `agent.mlp_brain` | shared front end and five MLP variants | only executable brain family in provided dump |
| `agent.ensemble` | loop/`vmap` batched forward over independent models | returns logits wrapper plus values |
| `engine.agent_registry` | stores brains, architecture ids, and builds homogeneous buckets | no weights live in tensors here |
| `rl.ppo_runtime` | per-slot PPO ownership and grouped math interface | no optimizer sharing |
| `utils.checkpointing` | save/load brain kinds, weights, and PPO state | restores architecture metadata |
| `ui.viewer` | displays brain labels and model summaries | operational inspection only |

---

## Appendix B. Model-family comparison

| Kind | Entry projection | Core block type | Depth pattern | Uses configurable block activation? | Uses configurable block norm? |
| --- | --- | --- | --- | --- | --- |
| Whispering Abyss | `64→96` | plain linear | 2 hidden layers | no | no |
| Veil of Echoes | `64→128` | plain linear | 3 hidden layers | no | no |
| Cathedral of Ash | `64→80` | residual | 3 residual blocks | yes | yes |
| Dreamer in Black Fog | `64→80` | gated residual | 2 gated blocks | yes | yes |
| Obsidian Pulse | `64→128` | bottleneck residual | 2 bottleneck blocks | yes | yes |

---

## Appendix C. Tensor/interface checklist

| Stage | Tensor | Shape | Hard invariant |
| --- | --- | --- | --- |
| live observation | `obs` | `(N, 283)` | width must equal `OBS_DIM` |
| ray split | `rays_flat` | `(B, 256)` | must equal `32×8` |
| ray structured | `rays_raw` | `(B, 32, 8)` | reshape must match config and live builder |
| rich combined | `rich_vec` | `(B, 27)` | must equal `23+4` |
| shared front end | `x` | `(B, 64)` by default | must equal `2*d_model` |
| actor output | `logits` | `(B, act_dim)` | width must equal `NUM_ACTIONS` |
| critic output | `value` | `(B, 1)` | scalar per sample |
| legality mask | `mask` | `(N, act_dim)` | mask width must equal logit width |
| ensemble values | `vals` | `(K,)` | flattened value vector in batched execution |
| PPO action mask | `act_mask` | `(B, act_dim)` | every row must contain at least one legal action |

---

## Appendix D. Intelligence-relevant configuration variables

| Variable | Role |
| --- | --- |
| `RAY_TOKEN_COUNT` | nominal ray-token count; live system is effectively fixed to 32 |
| `RAY_FEAT_DIM` | per-ray feature width, fixed at 8 |
| `RICH_BASE_DIM` | rich-base width, fixed at 23 |
| `INSTINCT_DIM` | instinct width, fixed at 4 |
| `OBS_DIM` | total observation width |
| `NUM_ACTIONS` | actor-head width and mask width |
| `BRAIN_KIND` | default brain family |
| `TEAM_BRAIN_ASSIGNMENT*` | team-specific or mixed architecture assignment |
| `BRAIN_MLP_D_MODEL` | token embedding width |
| `BRAIN_MLP_FINAL_INPUT_WIDTH` | derived shared MLP input width |
| `BRAIN_MLP_NORM` | controls `input_norm` and block norms |
| `BRAIN_MLP_ACTIVATION` | controls activation inside custom residual/gated/bottleneck blocks |
| `BRAIN_MLP_RAY_SUMMARY` | currently only `mean` is supported |
| `USE_VMAP` | enables `torch.func` batched inference path |
| `VMAP_MIN_BUCKET` | bucket-size threshold before trying `vmap` |
| `ARCHER_RANGE` | ranged-attack legality cap for archers |
| `ARCHER_LOS_BLOCKS_WALLS` | whether walls block ranged legality and combat resolution |
| `PPO_WINDOW_TICKS` and PPO hyperparameters | training-side consumers of intelligence outputs |
