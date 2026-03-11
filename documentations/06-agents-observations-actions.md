# Agents, Observations, and Actions

This document explains how agents are represented in the inspected code, what information they observe, and what discrete actions they can take.

## Purpose

The repository does not expose a generic “agent class” with all behavior inside it. Instead, agent-related state is split between:

- a dense slot-major tensor in `AgentsRegistry`
- a per-slot neural module list
- runtime observation construction inside `TickEngine`

## Agent representation

### Slot-based registry

Each live or dead slot exists in a fixed-capacity registry tensor. Important columns include:

| Column | Meaning |
|---|---|
| `COL_ALIVE` | Alive flag encoded as float-like `1.0` or `0.0` |
| `COL_TEAM` | Team id, with `2.0` for red and `3.0` for blue |
| `COL_X`, `COL_Y` | Grid coordinates |
| `COL_HP` | Current HP |
| `COL_UNIT` | Unit type |
| `COL_HP_MAX` | Maximum HP |
| `COL_VISION` | Vision range used by raycasting |
| `COL_ATK` | Attack power |
| `COL_AGENT_ID` | Display/compatibility field for persistent id |

In addition to `agent_data`, the registry stores:

- `brains`: one neural module per slot or `None`
- `agent_uids`: persistent integer identities
- `generations`: lineage generation counters
- `brain_arch_ids`: architecture-class ids used for inference bucketing

### Unit types

The inspected config defines two unit types:

- soldier
- archer

Initial spawn and respawn logic can choose between them probabilistically or inherit them along clone paths, depending on config.

### Per-slot brain ownership

The current inspected architecture is **per-slot**. Each slot can own its own model parameters. The repository does not implement a shared team policy in the main PPO path.

## Current brain family

`agent/mlp_brain.py` defines five MLP actor-critic variants:

- `WhisperingAbyssBrain`
- `VeilOfEchoesBrain`
- `CathedralOfAshBrain`
- `DreamerInBlackFogBrain`
- `ObsidianPulseBrain`

All of them share the same input and output interface:

- input: flat observation vector of width `OBS_DIM`
- output: action logits and scalar critic value

The differences are in the trunk structure:

- plain MLP
- narrowing MLP
- residual MLP
- gated residual MLP
- bottleneck residual MLP

## Observation construction

The current observation is built in `TickEngine._build_transformer_obs()`.

### Overall width

The config defines:

- `RAY_TOKEN_COUNT = 32`
- `RAY_FEAT_DIM = 8`
- `RICH_BASE_DIM = 23`
- `INSTINCT_DIM = 4`

Therefore:

- ray part: `32 × 8 = 256`
- rich tail: `23 + 4 = 27`
- total: `OBS_DIM = 283`

### Ray features

The observation starts with 32 first-hit ray features. The ray engine documents six first-hit classes:

- none
- wall
- red soldier
- red archer
- blue soldier
- blue archer

The exact 8-value-per-ray encoding is implemented by the ray engine rather than being redefined by the MLP module. The important point for the rest of the system is that the ray block is a fixed-width flat tensor of size 256.

### Rich base features

The 23 rich-base features are named explicitly in `config.py`.

| Index | Feature |
|---:|---|
| 0 | `hp_ratio` |
| 1 | `x_norm` |
| 2 | `y_norm` |
| 3 | `is_red_team` |
| 4 | `is_blue_team` |
| 5 | `is_soldier` |
| 6 | `is_archer` |
| 7 | `atk_norm` |
| 8 | `vision_norm` |
| 9 | `zone_effect_local` |
| 10 | `cp_local` |
| 11 | `tick_norm` |
| 12 | `red_score_norm` |
| 13 | `blue_score_norm` |
| 14 | `red_cp_points_norm` |
| 15 | `blue_cp_points_norm` |
| 16 | `red_kills_norm` |
| 17 | `blue_kills_norm` |
| 18 | `red_deaths_norm` |
| 19 | `blue_deaths_norm` |
| 20 | `pad_0` |
| 21 | `pad_1` |
| 22 | `pad_2` |

A crucial implementation detail is that schema version 2 changed index 9 from a heal-local boolean to a signed scalar `zone_effect_local` in `[-1, +1]`.

### Instinct features

The final 4 columns are an “instinct” context derived from local neighborhood density. The engine computes these from cached local offsets and occupancy/unit patterns rather than reading them from the base state directly.

### Brain-side preprocessing

The MLP brains do not consume the raw flat observation without structure. They split it into:

- `rays_raw` with shape `(B, 32, 8)`
- `rich_vec` with shape `(B, 27)`

The current shared MLP input path then:

1. normalizes each ray’s feature vector
2. projects each ray into width `D`
3. mean-pools the 32 ray embeddings into one learned ray token
4. normalizes and projects the rich vector into one learned rich token
5. concatenates the two learned tokens into a flat width `2D` input for the trunk

## Action space

The inspected config defaults to `NUM_ACTIONS = 41`.

The repository’s action semantics are:

| Action ids | Meaning |
|---|---|
| `0` | idle / no-op |
| `1..8` | movement in the eight compass directions |
| `9..40` | directional attacks encoded as 8 directions × 4 ranges |

### Attack decoding

For action `a >= 9`:

- range is `((a - 9) % 4) + 1`
- direction index is `(a - 9) // 4`

This gives 32 attack actions:

- 8 directions
- 4 possible range codes

### Unit-specific range gating

The action mask restricts valid attack ranges by unit type:

- soldiers: range 1 only
- archers: range `1..ARCHER_RANGE`, clipped to at most 4 in the current 41-action layout

### Optional line-of-sight blocking

If `ARCHER_LOS_BLOCKS_WALLS` is true, ranged attacks are masked out when a wall lies on an intermediate cell between attacker and target.

## Decision path during one tick

For alive slots:

1. observations are built
2. legal actions are masked
3. slots are grouped into architecture buckets
4. `ensemble_forward(...)` returns logits and values for one bucket
5. masked logits are used to form a categorical distribution
6. one action is sampled per slot
7. if PPO is enabled, observation, logits, value, action, and mask are cached for training

## Reward-relevant state visible to the agent

The observation includes several reward-relevant or outcome-relevant signals directly:

- current HP ratio
- current position
- team and unit flags
- local signed zone value
- current control-point occupancy at the cell
- normalized global score, kill, death, and control-point counters
- instinct neighborhood signal

That does **not** imply the policy uses them successfully. It only means the current implementation supplies them.

## Schema and naming cautions

### The schema is strict

`agent/obs_spec.py` validates:

- total dimensional arithmetic
- feature-name length
- zone-context semantic indices
- schema version and family

That strictness exists because silent semantic drift would corrupt policy interpretation.

### The name `_build_transformer_obs()` is historical

The current inspected policy family is MLP-based. The method name remains from an earlier model lineage.

### Checkpoint compatibility depends on schema identity

Policy-bearing checkpoint payloads are rejected if the stored observation schema does not match the current runtime schema.

## Related documents

- [Simulation runtime](05-simulation-runtime.md)
- [Learning and optimization](07-learning-and-optimization.md)
- [Glossary and notation](15-glossary-and-notation.md)
