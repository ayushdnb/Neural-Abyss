# Extension Guide

This document explains where to extend the inspected repository and which coupling points require care.

## Purpose

The repository is modular enough to extend, but several invariants are strong. This guide focuses on safe extension paths grounded in the inspected code.

## General rule

Before adding a feature, decide which layer it belongs to:

- configuration
- world representation
- tick semantics
- policy model
- PPO runtime
- catastrophe pack
- telemetry
- viewer/operator tooling

Putting a feature in the wrong layer is a common source of drift.

## Adding or changing mechanics

### Tick-order mechanics

If the new mechanic changes world-state progression during a tick, the main insertion point is `engine/tick.py`.

Examples:

- new combat effects
- new environment hazards
- new post-move rules
- new reward bookkeeping tied to world events

### Grid-coupled mechanics

If the mechanic changes occupancy, HP, or slot placement, changes must keep grid and registry state synchronized. The main risk is updating only one view.

High-risk coupling points:

- `registry.agent_data`
- `grid[0]`
- `grid[1]`
- `grid[2]`

### Zone-derived mechanics

If the mechanic is conceptually a zone effect, evaluate whether it belongs in:

- canonical base-zone generation in `engine/mapgen.py`
- runtime catastrophe overlay in `engine/catastrophe.py`
- environment application in `engine/tick.py`

## Adding a new catastrophe type

The catastrophe pack is one of the clearest extension surfaces in the repository.

### Where to add it

Primary file:

- `engine/catastrophe.py`

Typical steps:

1. add a preset builder function producing a `CatastropheSpec`
2. add eligibility logic if the scheduler should be able to choose it
3. add weight config if it should participate in dynamic scheduling
4. add manual-dispatch support if it should be operator-triggerable
5. add viewer hotkey or viewer menu path only after the engine path is stable

### Things that must match

A catastrophe spec must be consistent in:

- world shape
- override map shape
- apply-mask shape
- edit-lock-mask shape
- duration semantics

### Viewer integration

If the catastrophe should be user-triggerable, wire it in `ui/viewer.py`:

- help-card text
- hotkey handler
- user-facing status messages

## Adding telemetry

### Where to add it

Primary files:

- `utils/telemetry.py`
- `engine/tick.py` for hook placement

### Safer pattern

Use the existing pattern:

- compute the event or aggregate in the engine
- guard the telemetry call behind `telemetry is not None and telemetry.enabled`
- keep telemetry best-effort so it cannot crash the simulation

### Good extension categories

- new aggregate counter rows
- new event rows
- new per-agent summary columns
- new run metadata fields
- new schema-manifest fields when the file contract changes

### Important caution

If an append-compatible file schema changes, update schema/version handling rather than silently changing CSV headers.

## Adding or changing agent logic

### Brain architectures

Primary file:

- `agent/mlp_brain.py`

Tasks may include:

- adding another MLP variant
- changing trunk depth or block type
- changing shared preprocessing
- changing initialization behavior

### Architecture grouping support

If a new model type is added, verify that `AgentsRegistry` can still form valid architecture buckets. Bucket grouping is based on architecture signature rather than on parameter values.

### Observation-interface changes

If the policy input changes:

- update `config.py` schema constants
- update `agent/obs_spec.py`
- update `TickEngine._build_transformer_obs()`
- preserve or explicitly revise checkpoint schema compatibility logic in `utils/checkpointing.py`

This is a high-risk change.

## Adding or changing learning logic

### PPO runtime changes

Primary file:

- `rl/ppo_runtime.py`

Examples:

- new optimizer policy
- alternative scheduler
- different minibatch organization
- new training diagnostics
- additional rollout validation

### Engine-to-PPO interface changes

Primary file:

- `engine/tick.py`

Any change to what is passed into `record_step(...)` must remain consistent with the rollout-buffer contract in the PPO runtime.

### Hard boundary

Do not accidentally introduce shared optimizer or parameter state if the design goal remains slot-local independence.

## Adding new spawn or respawn behavior

Primary files:

- `engine/spawn.py`
- `engine/respawn.py`

Suitable extension points:

- new parent-selection rules
- new spawn-location rules
- new clone/mutation policies
- new unit-type inheritance rules

Important caution:

- respawn interacts with lineage metadata
- respawn interacts with PPO reset behavior
- respawn interacts with telemetry birth/death semantics

## Adding viewer features

Primary file:

- `ui/viewer.py`

Good viewer extensions include:

- new inspection panels
- new overlay modes
- new hotkeys tied to existing engine APIs
- clearer status messaging

Avoid putting simulation rules directly into the viewer. The viewer should call engine methods, not become a second copy of game logic.

## Invariants and warning zones

### 1. Grid and registry must match

This is the highest-risk invariant.

### 2. Observation schema changes ripple widely

Affected areas include:

- config schema constants
- observation build
- model input assumptions
- checkpoint compatibility
- PPO restore safety

### 3. Slot identity is not lineage identity

A slot can change occupant across respawn. That matters for PPO state, telemetry, and experimental analysis.

### 4. Catastrophe base/runtime separation is deliberate

Do not collapse base zones and runtime catastrophe state into one mutable field unless the checkpoint, viewer, and edit-lock contracts are redesigned together.

### 5. Telemetry should remain non-fatal

The code consistently treats telemetry as best-effort. Preserve that pattern.

## Practical extension workflow

A safe order for a significant extension is usually:

1. add config knobs
2. add core engine behavior
3. add checkpoint/telemetry support
4. add viewer/operator exposure
5. update documentation
6. only then append into existing run directories or reuse old checkpoints

## Related documents

- [Repository map](03-repository-map.md)
- [System architecture](04-system-architecture.md)
- [Limitations, validation, and open questions](14-limitations-validation-and-open-questions.md)
