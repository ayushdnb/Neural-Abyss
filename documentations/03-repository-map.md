# Repository Map

This document describes how the inspected repository is organized and where to read specific behaviors in the code.

## Purpose

This is a navigation guide rather than a full design document. It should be read alongside [System architecture](04-system-architecture.md) and [Simulation runtime](05-simulation-runtime.md).

## Public root layout

The public repository root exposes the following major paths:

```text
.
├── agent/
├── documentations/                # public prose folder present in the repository
├── engine/
│   ├── game/
│   └── ray_engine/
├── recorder/
├── rl/
├── simulation/
├── ui/
├── utils/
├── LICENSE
├── README.md
├── config.py
└── main.py
```

The inspected source snapshot also included a package wrapper path, but the public repository layout should be read as root-level modules and directories.

## Major files and directories

### `main.py`

Top-level runtime orchestrator.

Primary responsibilities:

- seed setup
- fresh world creation or checkpoint restore
- tick engine construction
- telemetry startup
- results-writer startup
- viewer loop or headless loop
- signal handling
- final summary and on-exit checkpoint

Read this first if the goal is to understand how the runtime boots and shuts down.

### `config.py`

Central configuration module.

Primary responsibilities:

- environment-variable parsing
- runtime defaults
- profile overrides
- config invariant checks
- one-line runtime summary string
- action, observation, PPO, catastrophe, respawn, UI, and telemetry knobs

Read this first if the goal is to identify the experiment-control surface.

### `agent/`

Policy and observation-interface helpers.

Important files:

- `mlp_brain.py`: current actor-critic brain family
- `ensemble.py`: bucketed forward path, including optional `torch.func`/`vmap`
- `obs_spec.py`: observation schema contract and token/group helpers

### `engine/`

Simulation mechanics.

Important files:

- `tick.py`: core tick engine
- `agent_registry.py`: slot-based state store and architecture bucket construction
- `mapgen.py`: wall and zone generation, canonical `Zones` container
- `spawn.py`: initial population generation
- `respawn.py`: floor-based and periodic respawn logic
- `catastrophe.py`: catastrophe controller, scheduler, and preset builders
- `grid.py`: base grid creation
- `game/move_mask.py`: legal-action masking
- `ray_engine/*`: ray feature extraction

### `rl/`

Learning runtime.

Important file:

- `ppo_runtime.py`: per-slot PPO collection, update, bootstrap caching, checkpoint state

### `ui/`

Viewer and runtime operator surface.

Important files:

- `viewer.py`: main viewer, controls, HUD, catastrophe actions, checkpoint hotkeys
- `camera.py`: camera and view transforms

### `simulation/`

- `stats.py`: cumulative scoring, elapsed time, death log, and snapshot/delta helpers

### `utils/`

Operational infrastructure.

Important files:

- `checkpointing.py`: atomic save/load/resume
- `persistence.py`: background CSV writer process
- `telemetry.py`: telemetry tree and analysis-oriented outputs
- `sanitize.py`: runtime sanity checks
- `profiler.py`: optional profiling and GPU summary helpers

### `recorder/`

Adjacent recording/analytics utilities.

Important files:

- `recorder.py`
- `schemas.py`
- `video_writer.py`

These modules are present in the repository, but the inspected `main.py` path currently uses its own `_SimpleRecorder` instead of this package. They should therefore be treated as repository utilities rather than as a confirmed part of the main execution path.

## Where to look for specific questions

### “Where does one tick actually happen?”

- `engine/tick.py`
- specifically `TickEngine.run_tick()`

### “Where are actions defined and masked?”

- `engine/game/move_mask.py`
- attack decoding in `engine/tick.py`

### “Where is the observation vector built?”

- `engine/tick.py` in `_build_transformer_obs()`
- schema contract in `agent/obs_spec.py`
- config names in `config.py`

### “Where are models created?”

- `agent/mlp_brain.py`
- `engine/spawn.py`
- checkpoint restore helpers in `utils/checkpointing.py`

### “Where is PPO actually wired?”

- `engine/tick.py`
- `rl/ppo_runtime.py`

### “Where do catastrophes live?”

- `engine/catastrophe.py`
- integration points in `engine/tick.py`
- viewer hooks in `ui/viewer.py`

### “Where do checkpoints and resume semantics live?”

- `utils/checkpointing.py`
- resume boot path in `main.py`

### “Where are results and telemetry written?”

- `utils/persistence.py`
- `utils/telemetry.py`
- startup and shutdown flow in `main.py`

### “Where are viewer controls defined?”

- `ui/viewer.py`, especially `InputHandler.handle()`

## Recommended maintainer reading order

A new maintainer can usually read the repository in this order:

1. `config.py`
2. `main.py`
3. `engine/agent_registry.py`
4. `engine/tick.py`
5. `agent/mlp_brain.py`
6. `engine/game/move_mask.py`
7. `engine/ray_engine/raycast_firsthit.py`
8. `rl/ppo_runtime.py`
9. `engine/catastrophe.py`
10. `utils/checkpointing.py`
11. `utils/telemetry.py`
12. `ui/viewer.py`

That order moves from boot, to state layout, to tick semantics, to learning, to observability, to operator tooling.

## Notes on naming drift

A few names are historical seams:

- `Infinite_War_Simulation` appears in source-path comments and the uploaded snapshot path.
- `Neural Siege` still appears in the runtime summary string.
- `_build_transformer_obs()` currently builds the inspected observation vector even though the current public brain family is MLP-based.

These names matter when reading the code, but they should not be mistaken for separate subsystems.
