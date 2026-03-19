# 01 System Foundations and Architecture

## Document Purpose

This document is the opening architecture volume for the `Neural-Abyss` repository. Its purpose is not to teach every mechanic, neural-network detail, or PPO equation in full. Its purpose is to establish the correct system model before the reader studies those deeper subjects. In other words, this volume explains what the system is, where its main responsibilities live, how the runtime is assembled, what persists across ticks and across resumes, and how the repository is divided into operational layers. The discussion is anchored to the uploaded codebase, not to assumed intent outside the repository. fileciteturn0file0L12021-L13176 fileciteturn0file0L1720-L3288

Editorially, this volume follows a discipline that is appropriate for long-form technical explanation rather than tutorial or marketing copy: explicit scope, hierarchical headings, progressive disclosure, and an audience model that fills the gap between what readers need and what they already know. That approach is consistent with established documentation guidance from Diátaxis, GitHub Docs, Google’s technical writing courses, and Write the Docs. citeturn998813search0turn232117search14turn232117search4turn439726search1turn439726search3

## Audience and How to Read This Volume

This volume is written for two readers at once.

The first reader is new to the codebase and possibly new to large simulation or reinforcement-learning repositories. That reader should read this document sequentially, because later sections assume the mental model established earlier: orchestration before mechanics, runtime ownership before implementation detail, and boundaries before optimizations.

The second reader is already comfortable reading code, but wants a reliable architecture map before inspecting implementation details. That reader can skim headings and diagrams, then jump to the sections on runtime lifecycle, subsystem boundaries, and the guided reading order.

One design choice is important to state explicitly: this document is an **explanation volume**. It is not a step-by-step tutorial and it is not a command reference. That distinction matters because explanation documents should build understanding and mental models, while tutorials and references serve different jobs. citeturn998813search0turn439726search7turn439726search9

---

## 1. Project Orientation

### 1.1 What this repository is

At repository level, this is a **grid-based, multi-agent simulation system** implemented in Python and PyTorch, with optional GPU execution, optional interactive visualization through Pygame, optional video capture through OpenCV, integrated per-agent PPO training, structured telemetry, and checkpoint/resume support. The system advances in discrete ticks. On each tick, alive agents observe the world, choose masked discrete actions through their own policy modules, interact through combat and movement rules, receive reward signals, and may later be respawned into reused registry slots. fileciteturn0file0L12021-L13176 fileciteturn0file0L9785-L12018 fileciteturn0file0L13191-L15075

### 1.2 The naming situation is not singular

A newcomer should notice a naming fact immediately, because it affects search and interpretation. The uploaded source is organized under the directory `Neural-Abyss`. However, internal file docstrings also use the names `Infinite_War_Simulation` and `Infinite War Simulation`, while the runtime summary banner printed by `config.summary_str()` is `[Neural Siege: Custom]`. These names coexist in the code. This volume therefore uses **`Neural-Abyss`** as the repository name, because that is the concrete directory/package identity in the uploaded source, while treating the other names as internal labels rather than collapsing them into an unverified single branding claim. fileciteturn0file0L1720-L3288 fileciteturn0file0L12021-L13176

### 1.3 The shortest correct mental description

In plain language, the system is a **world simulator plus learning runtime plus operator shell**.

Formally, the repository contains:

1. a world model built around a three-channel spatial grid,
2. an authoritative agent registry storing per-slot numerical state plus per-slot brain modules,
3. a tick engine that constructs observations, masks actions, runs inference, resolves combat and movement, applies environment effects, records reward components, and triggers respawn,
4. an optional PPO runtime that trains slot-local brains,
5. an optional viewer that drives the same engine interactively,
6. an output layer consisting of a background CSV writer, in-process telemetry, checkpointing, summaries, and optional video capture. fileciteturn0file0L3372-L4183 fileciteturn0file0L5438-L5650 fileciteturn0file0L9785-L12018 fileciteturn0file0L13191-L15075 fileciteturn0file0L15921-L18178 fileciteturn0file0L18181-L23985

### 1.4 What a newcomer should understand before anything else

Before reading any detail, five facts matter.

First, `main.py` is the **application orchestrator**, not the place where the core mechanics live. Second, `engine.tick.TickEngine` is the object that actually advances the simulation. Third, the world state lives in **two synchronized representations**: the spatial grid and the agent registry. Fourth, learning is integrated but not allowed to erase operational boundaries: the PPO runtime is optional and the simulation still has clear non-learning mechanics. Fifth, the viewer is a consumer and driver of engine ticks, not the owner of simulation truth. fileciteturn0file0L9785-L12018 fileciteturn0file0L12021-L13176 fileciteturn0file0L15921-L18178

---

## 2. System Snapshot

### 2.1 Verified snapshot

The following snapshot is directly supported by the uploaded code.

| Item | Verified description |
|---|---|
| Repository identity used in this volume | `Neural-Abyss` |
| Primary entry point | `main.py` |
| Runtime summary banner | `[Neural Siege: Custom]` |
| Core simulation stepper | `engine.tick.TickEngine` |
| World representation | `grid` tensor shaped `(3, H, W)` |
| Registry representation | `AgentsRegistry.agent_data` shaped `(capacity, 10)` plus side structures |
| Default grid size | `100 x 100` |
| Default initial population | `150` agents per team |
| Maximum registry capacity by default | `400` |
| Observation width | `283` |
| Discrete action count | `41` |
| Default results root | `results/` |
| UI library | Pygame |
| Optional video library | OpenCV if import succeeds |
| Training runtime | per-slot PPO runtime |
| Checkpoint format | atomic checkpoint directories under `run_dir/checkpoints/` |

This snapshot combines defaults from `config.py` with concrete wiring in `main.py`, `engine.tick.py`, and related runtime modules. fileciteturn0file0L1720-L3288 fileciteturn0file0L5451-L5544 fileciteturn0file0L3539-L4183 fileciteturn0file0L9785-L12018 fileciteturn0file0L12021-L13176 fileciteturn0file0L18657-L19370

### 2.2 The most important invariant in the whole repository

A shallow reading might suggest that the world is just the grid. That is not correct. The engine’s own top-level comments state that the world state exists in two places: the **agent registry** and the **grid tensor**. The registry stores the authoritative per-slot attributes and Python brain modules. The grid stores fast spatial occupancy, hit points, and slot-id lookup. The system works only if these remain synchronized. This is the central architectural invariant of the runtime. fileciteturn0file0L9785-L9950 fileciteturn0file0L3372-L4183 fileciteturn0file0L5438-L5650

### 2.3 The action and observation contracts

The active policy interface is a fixed-length observation vector and a fixed discrete action space. The observation size is `OBS_DIM = RAYS_FLAT_DIM + RICH_TOTAL_DIM = 283`, where the code defines `32` ray tokens with `8` features each, plus `23` base rich features and `4` instinct features. The action space is `NUM_ACTIONS = 41`. The move-mask module documents the practical layout as idle, directional movement, and directional attacks over multiple ranges; the tick engine decodes attack actions from integer codes using quotient and remainder logic. These are not incidental constants. They are schema-level contracts that affect observations, model heads, masking, PPO storage, and checkpoint compatibility. fileciteturn0file0L1411-L1717 fileciteturn0file0L1720-L3288 fileciteturn0file0L4988-L5435 fileciteturn0file0L10791-L12018

---

## 3. Architectural Overview

### 3.1 Plain-language intuition

The repository is easiest to understand if you picture three layers.

The **foundation layer** defines state and contracts: configuration, grid shape, registry columns, observation schema, action schema, zone structures, and brain families.

The **runtime layer** advances the world: startup, spawning, ticking, combat, movement, reward accumulation, PPO bookkeeping, respawn, checkpoint save/load, telemetry flush, and shutdown.

The **operator layer** lets a human run or inspect the system: viewer UI, headless print summaries, manual checkpoints, telemetry files, and optional video recording. fileciteturn0file0L1411-L1717 fileciteturn0file0L1720-L3288 fileciteturn0file0L3372-L4183 fileciteturn0file0L9785-L13176 fileciteturn0file0L15921-L23985

### 3.2 Formal subsystem map

```text
+-------------------+
|     config.py     |
| env -> defaults   |
| profiles -> checks|
+---------+---------+
          |
          v
+-------------------+
|      main.py      |
| startup orchestral|
| mode selection    |
+----+---------+----+
     |         |
     |         +------------------------------------+
     |                                              |
     v                                              v
+------------+   +----------------+        +-------------------+
| world grid  |   |AgentsRegistry |        | Results / outputs |
| grid[3,H,W] |   |slots + brains |        | writer/telemetry  |
+------+-----+   +--------+-------+        | checkpoints/video |
       |                  |                +-------------------+
       +---------+--------+
                 |
                 v
          +--------------+
          | TickEngine   |
          | observe      |
          | mask actions |
          | infer        |
          | combat/move  |
          | rewards      |
          | respawn      |
          +------+-------+
                 |
      +----------+-----------+
      |                      |
      v                      v
+-------------+      +----------------+
| PPO runtime |      | Viewer (opt.)  |
| per-slot RL |      | UI frame loop  |
+-------------+      +----------------+
```

This diagram is deliberately conservative. It reflects what the code shows: `main.py` assembles the world, registry, stats, engine, output systems, and optional UI; `TickEngine` owns the per-tick simulation sequence; the PPO runtime is attached from inside the engine when enabled; the viewer drives ticks but does not replace the engine; persistence and telemetry are side systems observing or snapshotting runtime state. fileciteturn0file0L9785-L12018 fileciteturn0file0L12021-L13176 fileciteturn0file0L13191-L15075 fileciteturn0file0L15921-L18178 fileciteturn0file0L18181-L23985

### 3.3 What is central and what is optional

The following elements are central to the repository’s identity:

- `config.py`
- `main.py`
- `engine.tick.TickEngine`
- `engine.agent_registry.AgentsRegistry`
- `engine.grid.make_grid`
- `engine.spawn` and `engine.respawn`
- the active brain family in `agent.mlp_brain`
- the observation contract in `agent.obs_spec`

The following elements are optional or mode-dependent:

- `ui.viewer.Viewer`
- OpenCV recording in `_SimpleRecorder`
- `TelemetrySession` if telemetry is disabled
- `PerAgentPPORuntime` if PPO is disabled or import fails
- catastrophe scheduling if catastrophe config is off
- inspector no-output mode for visual inspection without results side effects. fileciteturn0file0L1720-L3288 fileciteturn0file0L506-L1717 fileciteturn0file0L4186-L4985 fileciteturn0file0L9785-L13176 fileciteturn0file0L15921-L18178 fileciteturn0file0L20782-L23985

### 3.4 Why the architecture is divided this way

The code repeatedly reflects one design principle: **separate high-frequency simulation logic from orchestration, rendering, and file I/O**. The tick engine is written for vectorized hot-path execution. File writing is pushed into a background process for CSV outputs. Telemetry is attached observationally instead of taking control of the simulation. Viewer logic keeps a cached CPU snapshot to avoid unnecessary GPU synchronization every frame. Checkpointing is isolated in its own utility module and uses atomic filesystem patterns. These are all concrete examples of architectural separation serving performance, operational safety, or both. fileciteturn0file0L9785-L12018 fileciteturn0file0L15921-L18178 fileciteturn0file0L18181-L19994 fileciteturn0file0L20782-L23985

---

## 4. Runtime Lifecycle

This is the strongest section of the volume because the repository is easiest to understand as a lifecycle.

### 4.1 Phase 0 — Configuration is resolved at import time

#### Plain-language intuition

Before the simulation starts, the repository decides what world it is going to run, what device it will use, whether PPO is enabled, whether UI is enabled, where outputs will go, and what schemas are in force.

#### Formal explanation

`config.py` is an import-time configuration module. It reads `FWS_*` environment variables, applies profile overrides only when explicit environment variables are absent, validates invariants, exposes resolved globals, and provides a summary banner. The precedence is explicitly documented as `environment variable > profile override > hard-coded default`. This means configuration is mostly frozen before `main()` begins, because many runtime decisions depend on already-resolved module globals. fileciteturn0file0L1720-L3288

#### Design consequence

A reader should treat `config.py` not as a passive constant file, but as a **configuration compiler**: it parses, normalizes, validates, and derives runtime contracts. That is why changing schema-level values such as observation dimensions, action counts, or brain widths is not morally equivalent to changing a print frequency. Some knobs change meaning; others merely change operation. fileciteturn0file0L1720-L3288

### 4.2 Phase 1 — `main()` establishes deterministic startup state

At the beginning of `main()`, PyTorch matmul precision is set to `high`, seeding is applied from config, and a one-line summary string is printed. The code also evaluates whether explicit inspector no-output mode is active. This is the first operational fork in the program, because inspector mode suppresses normal run-output creation later in startup. fileciteturn0file0L12747-L13176 fileciteturn0file0L1720-L3288

### 4.3 Phase 2 — The program chooses between fresh-start and resume-start

#### Fresh-start path

In a fresh run, `main.py` creates a new grid with `make_grid()`, creates an empty `AgentsRegistry`, creates a fresh `SimulationStats`, adds random walls, builds zones, then spawns the initial population with either `spawn_symmetric()` or `spawn_uniform_random()` based on `SPAWN_MODE`. At this stage there is world state, but there is not yet a `TickEngine`. The point of this phase is to materialize initial state, not to start advancing it. fileciteturn0file0L5451-L5544 fileciteturn0file0L5653-L6426 fileciteturn0file0L9154-L9782 fileciteturn0file0L12777-L12824

#### Resume path

In resume mode, `main.py` loads a checkpoint on CPU first, restores the world grid, reconstructs zones from the checkpoint payload, and creates empty registry and stats containers that are populated later by `CheckpointManager.apply_loaded_checkpoint()`. This order matters. The engine constructor wants a world and a registry object. The checkpoint utility then repopulates those objects with agent tensors, brains, statistics, controller state, PPO state, and RNG state. The code explicitly restores RNG last. That ordering is one of the strongest signals that resume fidelity is treated as a correctness issue rather than as a convenience feature. fileciteturn0file0L12773-L12808 fileciteturn0file0L19021-L19274

### 4.4 Phase 3 — The tick engine is constructed

The `TickEngine` is created with four primary inputs: registry, grid, stats, and zones. Construction time does more than store references. The engine derives height and width from the grid, creates a `RespawnController`, initializes zone caches, creates a `HealZoneCatastropheController`, caches direction tables and constants on device, allocates reusable movement and reward buffers, allocates observation scratch space, prepares instinct scratch state, and conditionally instantiates the PPO runtime. In other words, engine construction is a substantial runtime assembly phase, not a trivial object creation call. fileciteturn0file0L9950-L10208 fileciteturn0file0L12794-L12830

### 4.5 Phase 4 — If resuming, runtime state is applied into live objects

`CheckpointManager.apply_loaded_checkpoint()` repopulates the registry’s `agent_data`, `agent_uids`, generations, next-agent-id counter, and per-slot brains. It rebuilds architecture metadata so bucket grouping remains valid. It restores engine-owned agent scores, respawn-controller state, catastrophe-controller state, statistics, optional PPO state, and finally RNG state. This means a checkpoint in this repository is not merely “model weights plus world image.” It is an attempt to restore a **live experimental state**, including policy optimizers, rollout buffers, controller timers, and identity counters. fileciteturn0file0L18684-L19274 fileciteturn0file0L14756-L15048

### 4.6 Phase 5 — Outputs and side systems are attached

Once a live engine exists, `main.py` creates the operational side systems.

If inspector no-output mode is active, it deliberately skips normal results, telemetry, and checkpoint infrastructure and uses a no-op recorder instead. Otherwise, it creates a `ResultsWriter`, decides whether resume output should append into the original run directory or branch into a new run directory, constructs a `CheckpointManager`, and then tries to initialize a `TelemetrySession`. If telemetry is enabled, the engine receives a `telemetry` attribute, context is attached, schema manifests and run metadata are written, resume events are recorded, and a bootstrap lineage pass can materialize birth events for the initial population. Optional video recording is also attached in this phase. fileciteturn0file0L12315-L13025 fileciteturn0file0L18181-L19994 fileciteturn0file0L20558-L23985

### 4.7 Phase 6 — `engine.run_tick()` is wrapped, then signals are wired

A notable design detail appears before the main loop begins: `main.py` stores the original `engine.run_tick`, defines a wrapper that calls the original tick and then conditionally records video frames, and reassigns `engine.run_tick` to this wrapper. That is a runtime decoration pattern rather than a modification of engine code. After that, signal handlers install a shared shutdown flag and expose that flag back to the engine object. This creates a uniform shutdown protocol that both UI and headless paths can observe between ticks. fileciteturn0file0L12928-L13033

### 4.8 Phase 7A — UI mode runtime

When UI is enabled, the viewer becomes the driver loop. `Viewer.run()` binds the engine, registry, and stats; optionally creates a checkpoint manager for the run directory; creates layout, renderer, HUD, side panel, input handler, animation manager, and minimap objects; builds cached zone render state; optionally monkey-patches PPO record-step calls to accumulate per-agent score overlays; primes CPU caches; and then enters the frame loop. Each frame processes input, optionally saves a checkpoint, decides how many simulation ticks to execute based on pause state and speed multiplier, runs those ticks, refreshes cached CPU state on a configured cadence, renders the world and panels, flips the display, caps frame rate, and stops at tick limit if configured. The simulation logic is therefore still engine-owned, while pacing and rendering are viewer-owned. fileciteturn0file0L17527-L18178

### 4.9 Phase 7B — Headless runtime

The headless path is optimized for long unattended runs. `_headless_loop()` repeatedly checks the shutdown flag, executes exactly one tick, writes the latest flat stats row to the background writer, drains and writes death logs, optionally emits headless telemetry summaries, performs periodic runtime sanity checks, advances an optional profiler, handles trigger-file and periodic checkpoint saves, prints periodic status lines, and exits cleanly on signal or tick limit. This path has a different driver loop from the viewer, but it uses the same `engine.run_tick()` contract and the same registry/grid/stats objects. That is a key architectural point: **mode differences are driver differences, not world-model differences**. fileciteturn0file0L12523-L12744 fileciteturn0file0L19552-L19994 fileciteturn0file0L22589-L22667

### 4.10 Phase 8 — What happens inside one tick

A single call to `TickEngine.run_tick()` performs the core simulation lifecycle.

1. It establishes telemetry phase context and recomputes alive indices.
2. It refreshes zone runtime caches and catastrophe state if needed.
3. If no agents are alive, it advances time, flushes PPO dead slots, respawns, and returns early.
4. Otherwise it builds observations, builds legal-action masks, groups alive agents into architecture buckets, runs per-bucket inference with `ensemble_forward()`, samples actions, and updates PPO value caches.
5. It resets per-tick reward buffers.
6. It resolves combat first.
7. It applies combat deaths before movement.
8. It resolves movement with explicit conflict handling.
9. It applies zone scoring and other environment effects.
10. It records PPO reward components and training-step data if PPO is enabled.
11. It advances tick count.
12. It flushes dead slots from PPO and runs respawn.
13. It ingests respawn metadata into telemetry and ends the tick. fileciteturn0file0L10971-L12018

#### Why “combat-first” matters

The engine’s documented semantics are combat-first. This means an agent killed in combat does not move later in the same tick. That is not a small implementation detail. It shapes gameplay dynamics, reward timing, and the interpretation of event order in telemetry. A reader who assumes movement-first sequencing will misunderstand causal chains in logs and in future mechanic analyses. fileciteturn0file0L9785-L9950 fileciteturn0file0L10971-L11648

### 4.11 Phase 9 — Shutdown and finalization

On normal exit, interrupt, or crash, `main.py` executes a strong cleanup sequence: drain remaining death logs, close telemetry before freezing the final horizon, attempt an on-exit checkpoint if configured, write `summary.json` atomically, close the background results writer, close the recorder, and print shutdown completion. On crash, it also writes `crash_trace.txt` when a run directory exists. The system therefore treats shutdown as an explicit operational phase, not as accidental interpreter teardown. fileciteturn0file0L13074-L13176

### 4.12 End-to-end lifecycle diagram

```text
import config
   -> parse env / apply profile / validate
   -> main()
      -> seed + banner
      -> fresh world OR load checkpoint
      -> construct TickEngine
      -> restore runtime state if resuming
      -> create run_dir / writer / telemetry / checkpoint manager
      -> attach optional recorder
      -> install signal handling
      -> run viewer loop OR headless loop
           -> repeatedly call engine.run_tick()
      -> final logs / telemetry close / on-exit checkpoint / summary / close
```

The reason this diagram matters is that it preserves the repository’s real ownership structure. State does not begin inside the viewer. Checkpoints are not merely a final afterthought. Telemetry is not the same thing as the CSV results writer. PPO is downstream of engine inference and reward generation, not upstream of them. fileciteturn0file0L12021-L13176 fileciteturn0file0L18181-L23985

---

## 5. Repository Structure and Responsibility Map

This section is curated. It explains the files that matter first, not every file with equal weight.

### 5.1 Top-level responsibility map

| Path | Responsibility |
|---|---|
| `config.py` | import-time configuration resolution, defaults, profiles, validation, summary string |
| `main.py` | startup orchestration, mode selection, resource wiring, top-level run/shutdown lifecycle |
| `engine/` | simulation runtime and world mechanics |
| `agent/` | brain definitions, observation schema helpers, inference bucketing logic |
| `rl/` | per-slot PPO runtime |
| `simulation/` | cumulative statistics and export-friendly counters |
| `ui/` | viewer, camera, layout, rendering, input handling |
| `utils/` | checkpointing, persistence, telemetry, and operational helpers |

This grouping is not speculative. It follows the import graph and ownership patterns visible in `main.py` and the subsystem modules it constructs. fileciteturn0file0L12021-L12100

### 5.2 The files a serious reader should care about first

#### `config.py`

This file controls almost every runtime dimension that matters operationally: experiment identity, result paths, resume behavior, telemetry, device and AMP policy, vmap policy, world size, wall and zone generation, unit stats, observation schema, action count, respawn behavior, reward shaping, PPO hyperparameters, brain family selection, UI behavior, inspector mode, and video recording. It also applies profile presets and validates invariants. For architecture study, this is the repository’s policy surface. fileciteturn0file0L1720-L3288

#### `main.py`

This is the composition root. It does not define the combat system, but it does define the runtime order in which the whole application exists: seed, create or restore state, attach engine, attach outputs, choose UI or headless loop, and close safely. A reader who skips `main.py` often misunderstands where responsibility begins and ends in the rest of the repository. fileciteturn0file0L12021-L13176

#### `engine/tick.py`

This is the most important single runtime file. It expresses the exact tick semantics, the grid–registry synchronization discipline, the observation construction path, the action-mask path, the inference path, reward staging, death ordering, movement conflict resolution, zone effects, PPO data capture, respawn, and tick-end telemetry behavior. If the reader wants the true operational heart of the repository, it is here. fileciteturn0file0L9785-L12018

#### `engine/agent_registry.py`

This file explains how the repository thinks about agent identity. It distinguishes **registry slots** from persistent **agent UIDs**, stores numerical state in a dense tensor, stores brains separately as Python modules, and maintains architecture metadata for inference bucketing. This file is necessary to understand respawn, PPO state reset, lineage, and checkpoint restoration. fileciteturn0file0L3372-L4183

#### `agent/mlp_brain.py`, `agent/obs_spec.py`, and `agent/ensemble.py`

These three files jointly define the active policy interface.

- `obs_spec.py` defines how a flat observation is partitioned.
- `mlp_brain.py` defines the current brain family and its two-token input contract.
- `ensemble.py` defines how multiple independent models are fused into a batched inference path, optionally through `torch.func`/`vmap` with caching.

Together, they explain what the engine means when it says it is doing policy inference. fileciteturn0file0L44-L503 fileciteturn0file0L506-L1408 fileciteturn0file0L1411-L1717

#### `rl/ppo_runtime.py`

This file is the learning engine, but with an important caveat: it is **slot-based**, not persistent-individual-based. Its rollout buffers, optimizers, schedulers, and value caches are keyed by registry slot. That choice is coherent with the rest of the runtime, because the live simulation itself is slot-oriented. It also explains why respawn must explicitly reset PPO state for reused slots. fileciteturn0file0L13331-L15075

#### `ui/viewer.py`

This is the operator console. It does not own simulation truth, but it matters operationally because it determines how a human observes, pauses, steps, saves checkpoints, inspects agents and zones, and manually triggers catastrophe-related controls. It also illustrates how the code cleanly separates frame pacing from simulation mechanics. fileciteturn0file0L15921-L18178

#### `utils/checkpointing.py`, `utils/persistence.py`, and `utils/telemetry.py`

These files explain how runtime state leaves memory.

- `checkpointing.py` is about whole-state snapshots and restoration.
- `persistence.py` is about background CSV writing for per-tick stats and death logs.
- `telemetry.py` is about rich, structured observational output under `run_dir/telemetry/`.

A reader who wants to understand operational reproducibility, post-run analysis, or long-run survivability needs these files. fileciteturn0file0L18181-L23985

### 5.3 Important supporting files

`engine/grid.py` defines the grid shape and base semantics. `engine/mapgen.py` defines heal zones, capture-point masks, and zone state serialization. `engine/spawn.py` defines initial population creation and team-aware brain assignment. `engine/respawn.py` defines floor-based and periodic respawn, parent selection, brain cloning or replacement, and rare mutation logic. `engine/game/move_mask.py` defines legal action masking. `engine/ray_engine/*` defines ray-based perceptual feature extraction. `simulation/stats.py` defines cumulative counters and the flat row exported to `stats.csv`. fileciteturn0file0L4988-L7720 fileciteturn0file0L7723-L9782 fileciteturn0file0L15078-L15578

---

## 6. Configuration Model

### 6.1 Plain-language intuition

The configuration system answers two different kinds of questions.

Some questions are **schema questions**: How wide is the observation vector? How many actions exist? Which brain family is active? Is the grid `100x100` or something else? Those answers shape what the runtime is.

Other questions are **operational questions**: Should the UI run? Should telemetry be enabled? How often should checkpoints be written? Where should results go? Those answers shape how the runtime is operated.

`config.py` contains both kinds of questions in one place. The reader must learn to separate them mentally. fileciteturn0file0L1720-L3288

### 6.2 Resolution model

The code documents a clear precedence model:

1. explicit `FWS_*` environment variable,
2. profile override,
3. hard-coded default.

Profiles currently include `default`, `debug`, `train_fast`, and `train_quality`. Profile overrides are applied only when the corresponding environment variable was not explicitly set. This matters because it means profiles are convenience bundles, not hidden overrides that can silently defeat an operator’s explicit environment settings. fileciteturn0file0L1720-L3288

### 6.3 Categories that matter operationally

The repository’s config surface can be grouped into the following practical categories.

| Category | Examples |
|---|---|
| Experiment identity and outputs | `PROFILE`, `EXPERIMENT_TAG`, `RESULTS_DIR`, `CHECKPOINT_PATH` |
| Resume and checkpoint policy | `RESUME_OUTPUT_CONTINUITY`, `RESUME_FORCE_NEW_RUN`, `CHECKPOINT_EVERY_TICKS`, `CHECKPOINT_ON_EXIT` |
| Telemetry policy | `TELEMETRY_ENABLED`, schema/version options, move sampling, rich PPO telemetry, validation flags |
| Device and performance policy | `USE_CUDA`, `TORCH_DEVICE`, `TORCH_DTYPE`, `AMP_ENABLED`, `USE_VMAP` |
| World shape and map generation | `GRID_WIDTH`, `GRID_HEIGHT`, wall settings, heal-zone and CP settings |
| Unit and mechanic parameters | HP, attack, vision, line-of-sight, metabolism |
| Observation and action schema | `RAY_TOKEN_COUNT`, `RAY_FEAT_DIM`, `RICH_BASE_DIM`, `INSTINCT_DIM`, `OBS_DIM`, `NUM_ACTIONS` |
| Respawn and mutation policy | floors, budgets, parent selection, location policy, mutation knobs |
| Learning policy | PPO rollout length, optimizer settings, clipping, gamma, lambda, minibatches |
| Brain policy | `BRAIN_KIND`, team-assignment mode, mix strategy, MLP token width and normalization |
| UI and inspection policy | `ENABLE_UI`, inspector mode, viewer cadence, cell size, HUD width, recording |

This table is a reader-facing classification, but every item in it is grounded in concrete config globals present in `config.py`. fileciteturn0file0L1720-L3288

### 6.4 Which configuration changes are safe and which are dangerous

A safe beginner rule is this.

Changes to print cadence, output directories, telemetry toggles, UI toggles, or checkpoint frequency are usually **operational**.

Changes to `OBS_DIM`, `NUM_ACTIONS`, `RAY_TOKEN_COUNT`, `RICH_BASE_DIM`, `INSTINCT_DIM`, `BRAIN_MLP_D_MODEL`, or brain-kind selections are often **schema-affecting** and can break assumptions in models, checkpoints, telemetry compatibility, or both. The code reinforces this distinction through explicit invariant checks such as `OBS_DIM == RAYS_FLAT_DIM + RICH_TOTAL_DIM`, `RAYS_FLAT_DIM == RAY_TOKEN_COUNT * RAY_FEAT_DIM`, and `BRAIN_MLP_FINAL_INPUT_WIDTH == 2 * BRAIN_MLP_D_MODEL`. fileciteturn0file0L3047-L3275 fileciteturn0file0L881-L1118 fileciteturn0file0L1411-L1675

### 6.5 Configuration honesty in this codebase

A subtle but important detail appears in both comments and implementation: some configuration surfaces are broader than the currently executed runtime behavior. For example, `TelemetrySession` reads an events format config knob, but the implementation comments state that the current implementation supports JSONL only for events. This is a good example of why readers should trust the runtime code path more than the abstract possibility implied by a knob name. The repository itself signals this design honesty in `config.py`, where the file states that comments should document observed runtime behavior rather than aspirational behavior. fileciteturn0file0L1720-L1760 fileciteturn0file0L20782-L21090

### 6.6 Inspector mode is a real operational mode, not a UI cosmetic toggle

`INSPECTOR_MODE` and `INSPECTOR_UI_NO_OUTPUT` are especially worth understanding. They do not merely change rendering. In `main.py`, inspector no-output mode suppresses creation of normal results, telemetry, checkpoint, and file-output side effects while still allowing a world or resumed checkpoint to be viewed interactively. This is architecturally significant because it defines a clean inspection mode separated from experimental lineage creation. fileciteturn0file0L1189-L1233 fileciteturn0file0L12476-L13025

---

## 7. State, Control, Rendering, Learning, and Persistence Boundaries

### 7.1 Where state lives

The system does not have one monolithic state object.

Instead, state is distributed across several owners:

- **grid state** in the `(3, H, W)` world tensor,
- **agent state** in `AgentsRegistry.agent_data`, `agent_uids`, `brains`, and `generations`,
- **cumulative score/counter state** in `SimulationStats`,
- **zone state** in `Zones` and catastrophe controller state,
- **learning state** in PPO buffers, optimizers, schedulers, and value caches,
- **output state** in results files, telemetry buffers, and checkpoint directories,
- **operator state** in viewer selection, camera, speed, and viewer-state checkpoint payloads. fileciteturn0file0L3372-L4183 fileciteturn0file0L5653-L6426 fileciteturn0file0L8930-L9151 fileciteturn0file0L13331-L15075 fileciteturn0file0L15238-L15578 fileciteturn0file0L15921-L18178 fileciteturn0file0L18181-L23985

### 7.2 The grid–registry boundary

#### Plain-language intuition

The grid answers spatial questions quickly. The registry answers agent questions correctly.

#### Formal explanation

`engine/grid.py` defines the grid as three channels: occupancy/category, HP, and agent-id/slot channel. `AgentsRegistry` stores alive flags, team, position, HP, unit type, max HP, vision, attack, and display-compatible agent id in `agent_data`, while the authoritative permanent UID lives in `agent_uids`. Per-slot brains live in a Python list because modules cannot live inside a tensor. During a tick, the engine reads and writes both structures. That is why it also contains explicit helper logic like `_sync_grid_hp_for_slots()` and debug invariants around post-combat and post-respawn consistency. fileciteturn0file0L3372-L4183 fileciteturn0file0L5438-L5650 fileciteturn0file0L10306-L10969

#### Common failure mode

A beginner may think the grid is authoritative because it is visual and spatial. That is wrong. The registry is where the engine keeps properties that the grid cannot express cleanly, such as generation, brain object, permanent UID, and full slot-local runtime semantics. The grid is a synchronized spatial cache with important operational semantics, not a full replacement for the registry. fileciteturn0file0L3372-L4183 fileciteturn0file0L9785-L9950

### 7.3 The observation boundary

Observation construction belongs to the engine, but its schema is shared with the agent package.

`TickEngine._build_transformer_obs()` constructs the live observation tensor for alive agents. It builds a per-cell `unit_map`, raycasts in `32` directions with first-hit features, computes rich scalar/context features, computes four instinct features, and concatenates those parts into a final `(N_alive, OBS_DIM)` tensor. `agent.obs_spec` then acts as the authoritative splitter for the flat observation layout, and `agent.mlp_brain` validates that the configured observation width matches the expected contract before any model is used. This is a good example of architecture-level separation: **the engine produces observations, but it does not own the schema alone**. fileciteturn0file0L6419-L7720 fileciteturn0file0L10791-L10969 fileciteturn0file0L1411-L1717 fileciteturn0file0L881-L1372

### 7.4 The action boundary

Legal action masking lives in `engine/game/move_mask.py`, not in the brain modules. The move-mask file documents the practical action-space layout and builds a boolean mask over legal actions based on position, team, grid occupancy, unit type, and optional wall-blocked line-of-sight for ranged actions. The tick engine then applies this mask by replacing illegal logits with a very negative value before sampling from a categorical distribution. The boundary is therefore clean: the policy proposes preferences over the full action space, while the environment decides which actions are legal in the current world state. fileciteturn0file0L4988-L5435 fileciteturn0file0L10971-L11105

### 7.5 The brain boundary

The active brain family is defined in `agent/mlp_brain.py`. All concrete brain variants inherit from a shared base that enforces one observation contract and one output contract. The file’s central architectural choice is the **two-token input interface**:

- ray features are summarized into one learned token,
- rich features are projected into one learned token,
- those two tokens are concatenated into one flat input vector,
- a variant-specific MLP trunk processes that vector,
- actor and critic heads emit action logits and value estimates.

The five concrete kinds are `whispering_abyss`, `veil_of_echoes`, `cathedral_of_ash`, `dreamer_in_black_fog`, and `obsidian_pulse`. Brain selection is then controlled by config and by spawn/respawn policy. fileciteturn0file0L506-L1408 fileciteturn0file0L1009-L1167

A useful beginner correction belongs here: the engine method is still named `_build_transformer_obs()`, but the active policy modules in the uploaded code are MLP-based. A method name with older history should not be misread as proof that the current runtime is transformer-based. The actual active brain family in the uploaded source is the MLP family defined in `agent/mlp_brain.py`. fileciteturn0file0L10791-L10969 fileciteturn0file0L506-L1408

### 7.6 The inference boundary

Inference bridging lives in `agent.ensemble.py` and `AgentsRegistry.build_buckets()`. The registry groups alive slots by persistent architecture class. Each bucket contains aligned slot indices, models, and local positions in the alive-index array. `ensemble_forward()` then chooses between a safe Python loop and an optional `torch.func`/`vmap` path depending on config and bucket size. This architecture matters because it allows the simulation to preserve **per-agent model individuality** while still extracting batching efficiency where architecture compatibility exists. It is not classic shared-policy inference. It is bucketed multi-model inference. fileciteturn0file0L44-L503 fileciteturn0file0L4034-L4183 fileciteturn0file0L11020-L11088

### 7.7 The learning boundary

The PPO runtime is attached inside the engine and records steps during normal tick execution, but it remains conceptually separate from world mechanics.

Three points are essential.

First, PPO is **slot-local**. The runtime explicitly says that “agent id” in this context means registry slot index, not a persistent individual identity. Second, dead or reused slots must be flushed or reset before respawn, because otherwise optimizer or buffer state would leak from one occupant to the next. Third, the PPO runtime keeps its own rollout buffers, optimizer instances, schedulers, and value caches, and those are included in checkpoints when PPO is enabled. The learning subsystem is therefore tightly integrated operationally, but it still has a crisp ownership boundary. The world creates observations and rewards. PPO consumes them and manages training state. fileciteturn0file0L13331-L13838 fileciteturn0file0L10971-L12018 fileciteturn0file0L18684-L18813

### 7.8 The rendering boundary

The viewer is built as an **operator-facing read-and-drive shell** around the simulation. Its architecture reflects this.

- `Viewer` owns the window and frame loop.
- `InputHandler` owns input translation.
- `WorldRenderer`, `HudPanel`, `SidePanel`, and `Minimap` own different visual responsibilities.
- `Camera` owns spatial viewing transforms.
- The viewer maintains cached CPU copies of selected runtime state to reduce GPU-to-CPU sync frequency.
- The viewer can request checkpoints or manual zone changes between ticks.

This is not the structure of a game engine that embeds all simulation state into the render layer. It is closer to an operator console layered over a simulation engine. fileciteturn0file0L15921-L18178

### 7.9 The persistence boundary

The repository has **three** distinct persistence/output channels, and they should not be conflated.

1. `ResultsWriter` writes `config.json`, `stats.csv`, and `dead_agents_log.csv` through a background process and queue.
2. `TelemetrySession` writes structured telemetry under `run_dir/telemetry/`, including lineage, agent-life, event, summary, movement, reward, and mutation-related files.
3. `CheckpointManager` writes atomic full-state snapshots under `run_dir/checkpoints/`.

These channels differ in durability intent, data granularity, and operational role. ResultsWriter is intentionally non-blocking and may drop messages if its queue is full. Telemetry is richer and in-process. Checkpoints are whole-state recovery artifacts, guarded by `DONE` markers and `latest.txt`. fileciteturn0file0L19552-L19994 fileciteturn0file0L20782-L23985 fileciteturn0file0L18657-L19067

### 7.10 The checkpoint boundary

A checkpoint in this repository is a directory, not just a file. It contains `checkpoint.pt`, `manifest.json`, a `DONE` marker, and optionally a `PINNED` marker, while the checkpoint root keeps `latest.txt`. The save path is atomic: write into a temporary directory, write files, write `DONE`, rename into final place, then update `latest.txt`. On load, the absence of `DONE` is treated as grounds for refusing the checkpoint. This is exactly the kind of pattern a serious long-running simulation should use when partial writes would be dangerous. fileciteturn0file0L18657-L19020

### 7.11 Optional catastrophe and zone-control boundary

Zones are owned by `engine.mapgen.Zones`, while catastrophe scheduling is owned by `engine.catastrophe.HealZoneCatastropheController`. The tick engine does not implement catastrophe logic directly; it constructs a runtime signal and asks the catastrophe controller to update zone suppression state. The viewer can also expose manual catastrophe controls. This is another example of boundary discipline: zones are world-state structures, catastrophe scheduling is controller logic, and the engine is the place where they are synchronized each tick. fileciteturn0file0L4186-L4985 fileciteturn0file0L5653-L6426 fileciteturn0file0L10971-L11015 fileciteturn0file0L16648-L17076

---

## 8. Guided Reading Order for the Codebase

A beginner can easily get lost by opening the biggest file first. The better approach is to read by dependency and by ownership.

### 8.1 First pass: establish the runtime map

1. Open `config.py`.
2. Open `main.py`.
3. Open `engine/tick.py`.

This order gives the reader configuration semantics first, top-level orchestration second, and tick ownership third. By the end of that pass, the reader should know what the repository runs, how it starts, and what a tick is. fileciteturn0file0L1720-L3288 fileciteturn0file0L12021-L13176 fileciteturn0file0L9785-L12018

### 8.2 Second pass: establish world and identity structures

4. Read `engine/agent_registry.py`.
5. Read `engine/grid.py`.
6. Read `simulation/stats.py`.
7. Read `engine/mapgen.py`.

This pass tells the reader where state lives, how agent identity is encoded, how the world tensor is shaped, what cumulative counters exist, and how zones are represented. Without this pass, later study of movement, checkpointing, or telemetry becomes confusing. fileciteturn0file0L3372-L4183 fileciteturn0file0L5438-L5650 fileciteturn0file0L5653-L6426 fileciteturn0file0L15078-L15578

### 8.3 Third pass: establish the policy interface

8. Read `agent/obs_spec.py`.
9. Read `agent/mlp_brain.py`.
10. Read `agent/ensemble.py`.
11. Read `engine/game/move_mask.py`.
12. Read the ray-engine files.

This pass explains what the policy sees, what the policy emits, how legality is enforced, and how multi-model inference is batched. It is the right place to learn why the observation is shaped the way it is and why the engine can support different brain kinds. fileciteturn0file0L44-L1717 fileciteturn0file0L4988-L5435 fileciteturn0file0L6429-L7720

### 8.4 Fourth pass: establish population lifecycle

13. Read `engine/spawn.py`.
14. Read `engine/respawn.py`.
15. Return to `engine/tick.py` and focus on the respawn and death phases.

This pass is where the reader learns how fresh agents are created, how brains are assigned initially, how parent selection works during respawn, how rare mutation is injected, and why registry slots and persistent UIDs must be separated. fileciteturn0file0L7723-L9782 fileciteturn0file0L10971-L12018

### 8.5 Fifth pass: establish learning and outputs

16. Read `rl/ppo_runtime.py`.
17. Read `utils/persistence.py`.
18. Read `utils/telemetry.py`.
19. Read `utils/checkpointing.py`.

This pass is for operational seriousness. It shows how rewards become PPO updates, how logs are written, how lineage and event telemetry are captured, and how the whole runtime is snapshotted and restored. fileciteturn0file0L13191-L15075 fileciteturn0file0L18181-L23985

### 8.6 Sixth pass: study the operator shell

20. Read `ui/viewer.py`.
21. Read `ui/camera.py` only after `viewer.py` if needed.

The viewer is large, but it is easier to understand once the reader already knows what the engine, registry, and outputs mean. Opening it too early usually causes confusion because many visual features are just presentations of concepts defined elsewhere. fileciteturn0file0L15921-L18178

### 8.7 What to ignore on first reading

A newcomer can postpone fine details of the following on the first pass:

- exact raymarch implementation details,
- detailed PPO training statistics aggregation,
- rare mutation details,
- catastrophe pattern generation internals,
- UI paint details,
- helper scripts such as `dump_py_to_text.py`.

These are real parts of the repository, but they are not the right entry point for building the first reliable architecture map. fileciteturn0file0L3289-L3371 fileciteturn0file0L4186-L4985 fileciteturn0file0L6429-L7720 fileciteturn0file0L13191-L15075 fileciteturn0file0L15921-L18178

---

## 9. Glossary of Core Terms

The following glossary is intentionally compact and repository-specific.

**Tick**  
One discrete simulation step. In this repository, a tick includes observation building, action masking, inference, combat resolution, death application, movement, zone effects, PPO logging, tick advancement, and respawn. fileciteturn0file0L10971-L12018

**Grid**  
A three-channel tensor shaped `(3, H, W)` storing occupancy/category, HP, and slot-id information for spatial queries and rendering. fileciteturn0file0L5438-L5544

**Registry slot**  
A fixed row index in `AgentsRegistry`. Slots are reused over time and are the live identities used by the engine and PPO runtime. fileciteturn0file0L3539-L3790 fileciteturn0file0L13331-L13409

**Agent UID**  
A persistent unique identifier stored authoritatively in `agent_uids`. Unlike slots, UIDs are intended to remain meaningful across death, respawn, lineage, and telemetry. fileciteturn0file0L3539-L3789

**Brain kind**  
The normalized string identifier for a concrete policy/value architecture, such as `whispering_abyss` or `obsidian_pulse`. fileciteturn0file0L506-L1408 fileciteturn0file0L1009-L1167

**Bucket**  
A grouping of alive agents whose brains share the same architecture class, allowing batched inference while preserving per-agent model individuality. fileciteturn0file0L3498-L3517 fileciteturn0file0L4034-L4183

**Rich features**  
The non-ray portion of the observation vector, consisting of base context features plus instinct features. fileciteturn0file0L1411-L1675 fileciteturn0file0L10791-L10969

**Instinct context**  
A four-value feature block computed from local ally/enemy density and related neighborhood statistics rather than direct ray hits. fileciteturn0file0L10275-L10460 fileciteturn0file0L10857-L10969

**Combat-first semantics**  
The rule that combat damage and combat deaths are applied before movement, so an agent killed in combat does not move later in that tick. fileciteturn0file0L9785-L9950 fileciteturn0file0L11320-L11648

**Respawn controller**  
The stateful controller that manages floor-based and periodic respawn policy, cooldowns, and spawn metadata. fileciteturn0file0L8930-L9151

**Inspector no-output mode**  
A UI inspection mode that intentionally avoids creation of normal output side effects such as results, telemetry, checkpoints, and files. fileciteturn0file0L1189-L1233 fileciteturn0file0L12476-L13025

**Telemetry schema manifest**  
A structured metadata file describing expected lineage fields, reward fields, death causes, and selected mechanic flags for telemetry consumers. fileciteturn0file0L12222-L12270 fileciteturn0file0L21488-L21530

**Pinned checkpoint**  
A checkpoint marked with a `PINNED` file so pruning will not delete it. fileciteturn0file0L18657-L18915

---

## 10. Common Beginner Misreadings

### Misreading 1 — “`main.py` is the simulation engine.”

**Correction:** `main.py` is the application orchestrator. It seeds, restores or creates world state, attaches side systems, chooses UI versus headless mode, and coordinates shutdown. The simulation engine is `TickEngine.run_tick()`. This distinction matters because most mechanic changes belong in `engine/`, not in `main.py`. fileciteturn0file0L12021-L13176 fileciteturn0file0L9785-L12018

### Misreading 2 — “The viewer owns the simulation.”

**Correction:** the viewer owns frame pacing, input, and rendering. It calls `engine.run_tick()`; it does not replace it. The same engine contract also drives headless mode. If the reader mentally promotes the viewer to “simulation owner,” the repository will become harder to understand. fileciteturn0file0L12523-L12744 fileciteturn0file0L17988-L18178

### Misreading 3 — “The grid is the whole world state.”

**Correction:** the grid is only one half of the runtime truth. The registry stores the rest: alive flags, vision, attack, persistent UID, generation, and the actual brain modules. The engine repeatedly relies on both structures and contains helper logic precisely because desynchronization would be dangerous. fileciteturn0file0L3372-L4183 fileciteturn0file0L9785-L9950

### Misreading 4 — “One PPO agent equals one biological individual across the whole run.”

**Correction:** PPO state is keyed by registry slot. Slots are reused. The repository addresses this explicitly through slot resets and dead-slot flushes on respawn. Persistent individuality for analytics lives in UIDs and lineage telemetry, not in the PPO slot key. fileciteturn0file0L13331-L13409 fileciteturn0file0L13770-L13883 fileciteturn0file0L11943-L11992

### Misreading 5 — “A checkpoint is just weights.”

**Correction:** checkpoints contain world state, zones, registry tensors, brains, generations, agent-id counters, engine controller state, PPO runtime state, stats, viewer state, and RNG state. Treating them as weights-only artifacts would make resume behavior incomprehensible. fileciteturn0file0L18684-L19274

### Misreading 6 — “Telemetry and `stats.csv` are the same thing.”

**Correction:** `stats.csv` and `dead_agents_log.csv` come from the background `ResultsWriter`. Rich telemetry lives in `run_dir/telemetry/` and includes many other streams, such as lineage edges, event chunks, per-agent life records, movement summaries, and PPO-related telemetry. These systems are adjacent but not identical. fileciteturn0file0L19552-L19994 fileciteturn0file0L20782-L23985

### Misreading 7 — “The method name `_build_transformer_obs()` proves the active architecture is transformer-based.”

**Correction:** the uploaded repository’s active brain family is the MLP family in `agent/mlp_brain.py`. The engine method name appears to be historical. The current policy path is observation construction plus MLP-brain inference, not a transformer module defined in the uploaded source. fileciteturn0file0L506-L1408 fileciteturn0file0L10791-L10969

### Misreading 8 — “All config knobs are equally safe to change mid-project.”

**Correction:** some knobs are operational and some are schema-defining. Observation widths, action counts, and brain-shape knobs affect deep compatibility assumptions. The code contains explicit invariant validation because the repository treats such mismatches as dangerous. fileciteturn0file0L3047-L3275 fileciteturn0file0L881-L1118

### Misreading 9 — “Logging is fully durable because it is asynchronous.”

**Correction:** the CSV `ResultsWriter` is intentionally non-blocking and explicitly allows message drops when the queue is full, because the design prefers simulation progress over guaranteed logging durability. That is a conscious tradeoff, not a bug to assume away. fileciteturn0file0L19771-L19994

### Misreading 10 — “UI mode and headless mode are different simulation systems.”

**Correction:** they are different **driver loops** around the same engine contract. That is why checkpointing, telemetry, and tick semantics remain coherent across both modes. fileciteturn0file0L12523-L12744 fileciteturn0file0L17988-L18178

---

## 11. What This Volume Establishes for Later Volumes

This volume establishes the repository’s system shape.

It has explained:

- what kind of repository this is,
- where primary responsibilities live,
- how startup, runtime, output, and shutdown are ordered,
- why the grid–registry split is fundamental,
- how observation, action, inference, learning, rendering, and persistence are separated,
- how to read the repository without getting lost.

It has intentionally not gone deep into:

- exact combat math and movement rules,
- detailed ray feature semantics and perception geometry,
- the full semantics of rich and instinct features,
- the differences among the five brain variants in training behavior,
- the internals of PPO loss computation and batched training,
- detailed viewer operations and UI controls,
- telemetry schema interpretation for downstream analysis.

Those belong in later volumes precisely because they make more sense after the architectural map is stable. fileciteturn0file0L44-L1717 fileciteturn0file0L4988-L7720 fileciteturn0file0L9785-L15075 fileciteturn0file0L15921-L23985

---

## Appendix A. Important Verified Files at a Glance

The following table is a practical map, not an exhaustive directory listing. It highlights the files that carry the most architectural weight in the uploaded source. fileciteturn0file0L44-L23985

| File / module | Role | Why a reader should care |
|---|---|---|
| `config.py` | Runtime configuration surface | Explains the entire policy surface and many compatibility assumptions |
| `main.py` | Composition root | Shows startup order, mode branching, output wiring, and shutdown order |
| `engine/tick.py` | Core simulation stepper | Defines what a tick actually does |
| `engine/agent_registry.py` | Agent truth store | Explains slots, UIDs, brains, and bucket metadata |
| `engine/grid.py` | Spatial world tensor | Defines the world tensor’s base semantics |
| `engine/mapgen.py` | Zone definitions and masks | Explains heal zones, CP masks, and zone serialization |
| `engine/spawn.py` | Initial population logic | Explains how a fresh world gets populated |
| `engine/respawn.py` | Ongoing population regeneration | Explains floor and periodic respawn, parent selection, and mutation |
| `engine/game/move_mask.py` | Action legality | Explains what actions can be chosen at all |
| `engine/ray_engine/raycast_32.py` | 32-ray feature extractor | Explains directional sensing representation |
| `engine/ray_engine/raycast_firsthit.py` | First-hit ray features and unit map | Connects world occupancy to perceptual categories |
| `agent/obs_spec.py` | Observation schema authority | Prevents silent feature-order drift |
| `agent/mlp_brain.py` | Active brain family | Defines policy/value networks and shared contracts |
| `agent/ensemble.py` | Bucketed inference bridge | Explains batched inference over independent models |
| `rl/ppo_runtime.py` | Learning runtime | Explains per-slot PPO state, rollout storage, and training |
| `simulation/stats.py` | Cumulative counters | Explains exported flat stats and death-log buffering |
| `ui/viewer.py` | Operator-facing UI | Explains inspection, pacing, overlays, and manual controls |
| `utils/persistence.py` | Async CSV writer | Explains `stats.csv` and death-log output behavior |
| `utils/telemetry.py` | Rich telemetry recorder | Explains the structured analysis sidecar |
| `utils/checkpointing.py` | Full-state snapshot system | Explains save/load/recovery and checkpoint durability |

---

## Appendix B. A Compact End-to-End Mental Model

A short recall model after finishing this volume is the following.

1. `config.py` resolves runtime policy before anything else.
2. `main.py` either creates a fresh world or loads one from checkpoint.
3. The world is split between a spatial grid and an agent registry.
4. `TickEngine` is the only place that truly advances simulation time.
5. A tick means: build observations, mask actions, run inference, resolve combat, apply deaths, resolve movement, apply environment effects, record learning data, advance time, respawn.
6. Brains are per-slot modules with a shared observation/output contract.
7. PPO is integrated but slot-local, so respawn must reset reused-slot learning state.
8. UI mode and headless mode are different shells around the same engine.
9. Results, telemetry, and checkpoints are three different output systems.
10. Resume means reconstructing a live runtime, not merely loading weights.

A final compressed diagram:

```text
config -> main -> {fresh world | checkpoint restore}
      -> registry + grid + stats + zones
      -> TickEngine
      -> {viewer loop | headless loop}
      -> repeated run_tick()
      -> {results writer, telemetry, checkpoints, optional video}
      -> clean shutdown + summary
```

If that model is clear, the reader is ready for later volumes on mechanics, observation semantics, brain architecture, PPO learning behavior, and operations. fileciteturn0file0L1720-L3288 fileciteturn0file0L9785-L13176 fileciteturn0file0L13191-L23985
