# Documentation Index and Reading Guide

## Purpose of This Index

This file is the entry point for the technical manual set that documents the `Neural Abyss` codebase. Its job is not to reteach the main volumes. Its job is to help a reader choose the correct starting point, follow a sensible reading order, and keep the manual synchronized with the code as the system evolves. The uploaded source tree shows a Python/PyTorch simulation organized around configuration, world generation, agent registration and spawning, a combat-first tick engine, per-agent brains and PPO runtime, an optional Pygame viewer, and utilities for checkpointing, persistence, profiling, sanitization, and telemetry. fileciteturn0file0

This index follows standard large-document guidance: descriptive headings, progressive disclosure, and clear internal navigation reduce reader friction in long technical sets, while GitHub-flavored Markdown supports stable section links, relative links, and readable repository-hosted manuals. citeturn563581search0turn563581search1turn727212search3turn727212search6turn563581search12

## Who This Manual Set Is For

This manual set is for readers who need more than a repository front page:

- beginners who need a controlled path from architecture to mechanics to learning and operations,
- simulation engineers working on grid state, spawning, combat, or tick behavior,
- ML engineers working on observations, brains, batching, and PPO,
- operators and maintainers working on viewer behavior, resume flows, telemetry, outputs, and safe extension,
- reviewers who need a fast but technically grounded understanding of how the system is structured. fileciteturn0file0

A README should explain what the project is, why it exists, and how to get started. A long-form manual should carry the deeper design, operations, and maintenance material. This index belongs to the second category. citeturn727212search3turn727212search21

## 1. Documentation set at a glance

The current manual set consists of six volumes:

```text
00  Documentation index and reading guide
01  System foundations and architecture
02  Simulation mechanics and world engine
03  Observation, perception, actions, and brain inputs
04  Neural brains, PPO training, and learning math
05  Operations, viewer, checkpointing, telemetry, and safe extension
```

The split matches the structure visible in the codebase: top-level orchestration and configuration; engine/world mechanics; observation and perception interfaces; neural and RL internals; and operational tooling around viewing, persistence, checkpointing, and telemetry. fileciteturn0file0

## 2. Volume-by-volume map

| Volume | File | Main purpose | Primary readers | Primary code anchors | Typical questions answered |
|---|---|---|---|---|---|
| 00 | [`00_documentation_index_and_reading_guide.md`](00_documentation_index_and_reading_guide.md) | Orient the reader and choose an efficient path through the set. | Everyone | Entire set | Where should I start? What should I skip for now? |
| 01 | [`01_system_foundations_and_architecture.md`](01_system_foundations_and_architecture.md) | Explain the system as a whole: package layout, orchestration, configuration model, runtime flow, and major subsystem boundaries. | All new readers; reviewers; maintainers | `main.py`, `config.py`, package layout | What subsystems exist? How does a run start? What is configurable? |
| 02 | [`02_simulation_mechanics_and_world_engine.md`](02_simulation_mechanics_and_world_engine.md) | Explain the world model and tick-driven simulation mechanics. | Simulation engineers; beginners after Volume 01 | `engine/grid.py`, `engine/mapgen.py`, `engine/spawn.py`, `engine/respawn.py`, `engine/tick.py`, `engine/agent_registry.py` | How does the world evolve each tick? How are agents spawned, moved, damaged, removed, and restored? |
| 03 | [`03_observation_perception_actions_and_brain_inputs.md`](03_observation_perception_actions_and_brain_inputs.md) | Explain what information reaches an agent and how action interfaces are shaped. | Simulation engineers; ML engineers | `agent/obs_spec.py`, `engine/ray_engine/*`, `engine/game/move_mask.py`, observation-related config | What is inside the observation vector? How do rays work? What action constraints exist? |
| 04 | [`04_neural_brains_ppo_training_and_learning_math.md`](04_neural_brains_ppo_training_and_learning_math.md) | Explain the model family, batched inference path, and PPO runtime. | ML engineers; researchers | `agent/mlp_brain.py`, `agent/ensemble.py`, `rl/ppo_runtime.py` | How are brains built? How are observations embedded? How does per-agent PPO work? |
| 05 | [`05_operations_viewer_checkpointing_telemetry_and_safe_extension.md`](05_operations_viewer_checkpointing_telemetry_and_safe_extension.md) | Explain how to run, inspect, resume, record, debug, and extend the system safely. | Operators; maintainers; advanced contributors | `ui/viewer.py`, `ui/camera.py`, `utils/checkpointing.py`, `utils/persistence.py`, `utils/telemetry.py`, `utils/profiler.py`, `utils/sanitize.py` | How do I inspect a run? Resume from checkpoints? Trust telemetry? Add features without corrupting runs? |

The mapping above is anchored to the actual repository layout and subsystem boundaries visible in the uploaded source tree. fileciteturn0file0

## 3. Recommended reading orders

### Beginner full path

`00 → 01 → 02 → 03 → 04 → 05`

Use this path when the codebase is new. It follows the same progression recommended for large technical documents: start with architecture, then mechanics, then interfaces, then learning internals, then operational surfaces. citeturn563581search0turn563581search12

### Experienced engineer fast path

`00 → 01 → target volume`

Read Volume 01 first even if the final target is narrow. It establishes naming, configuration boundaries, and runtime orchestration. After that:

- simulation work: jump to 02,
- observation/interface work: jump to 03,
- learning/runtime work: jump to 04,
- operations/tooling work: jump to 05.

### Mechanics-focused path

`00 → 01 → 02 → 03`

Use this when changing world state, spawning, combat, movement, or any logic that changes what the agent can perceive or do.

### ML and PPO-focused path

`00 → 01 → 03 → 04 → 05`

Read Volume 03 before Volume 04. In this codebase, the learning stack depends on the observation contract and action interface, not only on the brain classes themselves. `agent/obs_spec.py`, the ray engines, and action masking rules define what the neural and PPO components are actually learning over. fileciteturn0file0

### Maintainer and extension-focused path

`00 → 01 → 05 → 02/03/04 as needed`

Use this when the immediate goal is to keep runs stable while adding or changing features. Volume 05 should be the operational control surface; Volumes 02 to 04 are then consulted by subsystem.

## 4. Which volume to read first by goal

| Goal | Read first | Then read |
|---|---|---|
| Understand the repository as a system | 01 | 02, 03, 04, 05 |
| Understand the tick loop and world evolution | 02 | 03 if observations/actions are affected; 05 if outputs or resume behavior are affected |
| Understand the observation vector, rays, and action interface | 03 | 04 |
| Understand the brain family, batching, and PPO | 04 | 03 first if the observation contract is still unclear |
| Run, inspect, resume, or instrument experiments | 05 | 01 for architecture context |
| Review the project quickly but seriously | 01 | 02 or 04 depending on whether the review is simulation-first or ML-first |
| Decide where to begin as a beginner | 00 | 01 |

## 5. How the volumes depend on each other

### Dependency chain

```text
00  Index only
│
└── 01  Foundations and architecture
    ├── 02  Mechanics and world engine
    │   └── 03  Observation, perception, actions, brain inputs
    │       └── 04  Neural brains, PPO training, learning math
    └── 05  Operations, viewer, checkpointing, telemetry, safe extension
```

### Practical interpretation

- **Volume 01 is foundational.** It should usually come before every other volume.
- **Volume 02 comes before 03 in most serious reads.** Observation and action semantics are downstream of world mechanics and tick execution.
- **Volume 03 comes before 04.** The model and PPO stack only make sense once the input and control contracts are clear.
- **Volume 05 is cross-cutting.** It touches the top-level runtime, the viewer, persistence, checkpointing, telemetry, and safe operational practice. It can be read earlier for operational needs, but it is better understood after Volume 01. fileciteturn0file0

A reader may safely jump out of order only when the goal is narrow and operationally bounded. Even then, Volume 01 should be skimmed first.

## 6. Keeping the manual synchronized with the codebase

The manual remains trustworthy only if updates follow the code tree instead of drifting into parallel folklore. For maintainers, the reliable workflow is:

1. Verify the behavior in code first.
2. Update the volume that owns the changed concept.
3. Update any dependent volume whose explanation, assumptions, or cross-links were affected.
4. Update this index when file names, scope boundaries, or recommended reading paths change.

This reflects standard documentation practice for large sets: keep content scannable, link to the right depth instead of repeating background, and preserve stable, descriptive headings and link targets so navigation does not decay over time. citeturn563581search12turn563581search14turn563581search11turn563581search19

### Change-to-volume guidance

| Code change area | Update first | Then review |
|---|---|---|
| `main.py`, `config.py` | 01 | 05, 00 |
| `engine/grid.py`, `engine/mapgen.py`, `engine/spawn.py`, `engine/respawn.py`, `engine/tick.py`, `engine/agent_registry.py` | 02 | 03, 05 |
| `agent/obs_spec.py`, `engine/ray_engine/*`, `engine/game/move_mask.py`, observation/action config | 03 | 04, 02 |
| `agent/mlp_brain.py`, `agent/ensemble.py`, `rl/ppo_runtime.py` | 04 | 03, 01 |
| `ui/viewer.py`, `ui/camera.py` | 05 | 01 |
| `utils/checkpointing.py`, `utils/persistence.py`, `utils/telemetry.py`, `utils/profiler.py`, `utils/sanitize.py` | 05 | 00, 01 |
| File renames, volume renames, or scope reshuffles | 00 | Every affected volume |

These ownership boundaries are grounded in the uploaded package structure and module responsibilities. fileciteturn0file0

## 7. Quick navigation table

| If you want to… | Read this first |
|---|---|
| understand the full codebase without getting lost | 01 |
| understand how a simulation tick works | 02 |
| understand the observation vector and perception pipeline | 03 |
| understand how the brains and PPO runtime learn | 04 |
| inspect runs, resume safely, or trust telemetry | 05 |
| choose a reading path for the manual | 00 |

## 8. Scope notes

This index does not duplicate the deep explanations from the main volumes. It exists to reduce navigation friction.

The README and the manual set should not do the same job. GitHub recommends a README as the repository’s immediate orientation surface, while longer-form project documentation belongs in dedicated documentation space. This manual set is that deeper layer. citeturn727212search3turn727212search21

The topics are separated into different files because readers do not arrive with the same goal:

- architecture readers need boundaries before details,
- mechanics readers need the world model before the learning stack,
- ML readers need the observation contract before PPO details,
- operators need resume, telemetry, and safe extension guidance without reading the entire theory stack first. citeturn563581search0turn563581search7turn563581search12

## Appendix A. One-line description of every volume

| Volume | One-line description |
|---|---|
| 00 | Navigation layer for the manual set. |
| 01 | Explains the repository’s structure, runtime orchestration, and configuration model. |
| 02 | Explains how the simulated world and tick engine behave. |
| 03 | Explains what agents perceive, how observations are structured, and how actions are constrained. |
| 04 | Explains the brain family, batched inference path, PPO runtime, and learning mathematics. |
| 05 | Explains how to run, inspect, persist, resume, debug, and extend the system safely. |

## Appendix B. Suggested update matrix for maintainers

| When this changes in code | Update these manual files |
|---|---|
| Run startup, environment overrides, config schema, naming conventions | `01`, `00`, and `05` if operator behavior changes |
| Grid channels, world generation, spawn/respawn rules, tick order, combat mechanics | `02`; also `03` if perceptions or action legality change |
| Observation width, feature ordering, ray semantics, masks, action indexing | `03`; also `04` if model inputs or PPO assumptions change |
| Brain architectures, embedding path, vmap batching, PPO loss or rollout handling | `04`; also `03` if interfaces changed |
| Viewer controls, camera behavior, checkpoint schema, results writing, telemetry format, runtime sanity checks | `05`; also `00` if navigation or file names changed |
| Any markdown file rename or any shift in volume boundaries | `00` first, then fix relative links in every affected volume |

## Manual maintenance note

Keep this manual set code-determined. Verify from the source tree, update the owning volume, fix cross-links when headings or file names change, and do not let operational behavior, schema definitions, or naming conventions drift out of sync with the written manual. fileciteturn0file0

