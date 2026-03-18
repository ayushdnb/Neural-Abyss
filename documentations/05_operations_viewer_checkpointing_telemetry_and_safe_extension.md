# 05 Operations, Viewer, Checkpointing, Telemetry, and Safe Extension

## Document Purpose

This volume explains how to operate this repository as a running system rather than as a pile of source files.

Earlier volumes can explain what the simulation is, how the world works, how observations are built, and how PPO training is organized. This volume answers a different question:

**How does a maintainer or operator actually run the repository, observe it, resume it, inspect its outputs, and modify it without breaking the hidden contracts that make long runs and faithful resumes possible?**

The emphasis here is operational truth, not aspiration. Every important claim in this document is anchored to the provided codebase, especially:

- `main.py`
- `config.py`
- `ui/viewer.py`
- `utils/checkpointing.py`
- `utils/persistence.py`
- `utils/telemetry.py`
- `utils/profiler.py`
- `engine/tick.py`
- `engine/agent_registry.py`
- `rl/ppo_runtime.py`
- `simulation/stats.py`

Where the code contains a configuration knob but the corresponding behavior is not used in the provided runtime path, this volume says so explicitly.

---

## Audience and How to Read This Volume

This volume is for two readers at once.

The first reader is new to running larger simulations. That reader needs a precise mental model of what “run”, “checkpoint”, “resume”, “viewer”, and “telemetry” actually mean in this repository.

The second reader is a future maintainer. That reader needs to know where hidden contracts live, where append-in-place assumptions exist, what state must remain synchronized across save and restore, and which edits are local versus cross-cutting.

A good way to read this volume is:

1. Read Sections 1 through 5 in order if you are new to the repository.
2. Use Sections 6 through 12 as an operational reference when running or modifying the project.
3. Keep the appendices nearby when making changes that touch persistence, viewer behavior, or telemetry schema.

---

## 1. Operating Model of the Repository

### 1.1 Plain-language intuition

Running this repository means starting one orchestrator, letting it either build a fresh world or restore a previous world, then repeatedly advancing simulation ticks while optional side systems observe and record what happens.

Those side systems are not all the same thing.

At minimum, the runtime has a simulation core. Around that core, the code can attach:

- a graphical viewer,
- a background CSV writer,
- a structured telemetry writer,
- a checkpoint manager,
- a video recorder,
- a profiler,
- console status reporting,
- crash and summary side files.

The code is organized so that the simulation can run with or without the viewer, with or without telemetry, and with or without output creation. That is why it is important to separate the words **simulation**, **viewer**, **results**, **telemetry**, and **checkpoint**. In this codebase, they are connected, but they are not interchangeable.

### 1.2 Formal technical explanation

`main.py` is the application orchestrator. Its job is to connect subsystems, not to define the full simulation mechanics itself. The runtime sequence in the provided code is:

1. Import and resolve configuration from `config.py`.
2. Seed Python, NumPy, and PyTorch RNGs.
3. Decide whether startup is a fresh run or a resume from checkpoint.
4. Create or restore the world grid and zones.
5. Create the `AgentsRegistry` and `SimulationStats`.
6. Build the `TickEngine`.
7. If resuming, apply runtime state into the engine, registry, stats, PPO runtime, and RNG.
8. Decide whether outputs are enabled or whether inspector no-output mode is active.
9. If outputs are enabled, create a run directory, a `ResultsWriter`, a `CheckpointManager`, and an optional `TelemetrySession`.
10. Optionally create a video recorder.
11. Replace `engine.run_tick` with a small wrapper that can record video frames after each tick.
12. Run either the viewer loop or the headless loop.
13. On shutdown, flush death logs, close telemetry, optionally save an on-exit checkpoint, write a final summary, close the writer and recorder, and exit.

### 1.3 Code mapping

The central code anchors are:

- `main.main()` — startup, orchestration, shutdown.
- `config.py` — configuration resolution from `FWS_*` environment variables.
- `engine.tick.TickEngine` — the runtime unit that advances the world by one tick.
- `ui.viewer.Viewer.run(...)` — interactive loop for UI-driven operation.
- `main._headless_loop(...)` — non-UI loop for long runs.
- `utils.persistence.ResultsWriter` — background process writing `config.json`, `stats.csv`, and `dead_agents_log.csv`.
- `utils.telemetry.TelemetrySession` — richer structured telemetry under `run_dir/telemetry/`.
- `utils.checkpointing.CheckpointManager` — checkpoint save/load, periodic save, trigger-file save, pruning, and restore.
- `utils.profiler.torch_profiler_ctx(...)` — optional PyTorch profiling context.
- `utils.profiler.nvidia_smi_summary()` — optional live GPU status string.
- `utils.sanitize.runtime_sanity_check(...)` — periodic runtime validation in headless mode.

### 1.4 The repository’s operational separations

An operator should mentally separate five layers.

#### A. Simulation truth

The true evolving state lives in the world grid, the registry, the engine, stats, PPO runtime, zone state, catastrophe controller state, respawn controller state, and RNG state.

This is the state that checkpointing attempts to preserve.

#### B. Viewer state

The viewer has camera position, zoom, pause state, speed multiplier, selection state, marked agents, overlay toggles, cached CPU copies for rendering and picking, and some UI-only bookkeeping.

This state is not the same as simulation truth.

The code does save a `viewer` payload in manual checkpoints from the UI, but the provided resume path does not restore that payload into the viewer on startup. That means viewer state is currently checkpoint-adjacent, not part of the effective resume contract.

#### C. Lightweight results writing

`ResultsWriter` writes a small legacy-style surface:

- `config.json`
- `stats.csv`
- `dead_agents_log.csv`

This path is intentionally non-blocking and may drop messages if its queue is full.

#### D. Rich telemetry

`TelemetrySession` writes a larger and more structured output tree under `telemetry/`. It tracks schema compatibility, append-safe event chunking, lineage, richer death information, optional PPO telemetry, and headless summary surfaces.

#### E. Checkpointing

Checkpointing is the only mechanism in the provided code intended to preserve enough runtime state for faithful continuation. It stores far more than model weights.

### 1.5 Why this separation matters

Beginners often merge these categories into one mental object called “the run”. That causes operational mistakes.

For example:

- A viewer overlay bug does not necessarily mean the simulation state is wrong.
- A missing `stats.csv` row does not necessarily mean the tick did not happen.
- A saved model weight file from the viewer is not a full checkpoint.
- A full checkpoint can restore training buffers and RNG, while a plain `.pth` brain export cannot.
- Telemetry append compatibility matters even when the simulation itself can technically resume.

That separation is the foundation for everything else in this volume.

---

## 2. Launch Flow and Runtime Modes

## 2.1 Entry point

The code is launched through the standard Python entry guard in `main.py`:

```python
if __name__ == "__main__":
    main()
```

In operational terms, this repository is launched as `python main.py`, with behavior controlled primarily by `FWS_*` environment variables resolved in `config.py`.

There is no `argparse`-style command-line interface in the provided code. Runtime choice is environment-driven, not flag-driven.


## 2.1.1 Runtime-mode map

The repository’s main operational modes can be summarized like this:

| Mode | Key settings | Main loop | Run directory? | Telemetry? | Checkpoints? | Typical use |
|---|---|---|---:|---:|---:|---|
| Fresh UI run | `FWS_UI=1`, no checkpoint path | `Viewer.run(...)` | Yes | Optional | Yes | interactive exploration |
| Fresh headless run | `FWS_UI=0`, no checkpoint path | `_headless_loop(...)` | Yes | Optional | Yes | long training / benchmarking |
| Resume UI run | `FWS_UI=1`, `FWS_CHECKPOINT_PATH=...` | `Viewer.run(...)` | Yes, unless inspector no-output | Optional | Yes, unless inspector no-output | interactive continuation |
| Resume headless run | `FWS_UI=0`, `FWS_CHECKPOINT_PATH=...` | `_headless_loop(...)` | Yes | Optional | Yes | unattended continuation |
| Inspector no-output run | `FWS_INSPECTOR_MODE=ui_no_output` or equivalent | `Viewer.run(...)` | No | No | No | visual inspection without side effects |

A useful compact flow is:

```text
python main.py
    ->
config import + env resolution
    ->
seed everything
    ->
fresh world OR checkpoint load
    ->
TickEngine creation
    ->
optional checkpoint application
    ->
outputs on?  yes ------------------------ no (inspector no-output)
    ->                                          ->
run_dir / writer / telemetry / ckpt_mgr         no output side systems
    ->
viewer loop OR headless loop
    ->
telemetry close
    ->
optional on-exit checkpoint
    ->
summary + shutdown
```

## 2.2 Startup sequence

The startup flow in `main.main()` can be summarized as:

```text
import config
    ->
seed RNGs
    ->
decide fresh vs resume
    ->
create grid/zones/registry/stats or load checkpoint payload
    ->
create TickEngine
    ->
if resume: apply loaded runtime state
    ->
decide output mode
    ->
if outputs enabled:
    create ResultsWriter
    create run_dir
    create CheckpointManager
    create TelemetrySession
    maybe create video recorder
    ->
wrap engine.run_tick for recording
    ->
viewer loop OR headless loop
    ->
flush / checkpoint / summary / shutdown
```

That ordering matters. The engine is created before checkpoint application, and checkpoint application mutates the already-created runtime objects in place.

## 2.3 Fresh-run mode

Fresh-run mode is chosen when `config.CHECKPOINT_PATH` is empty.

In that path:

- `make_grid(config.TORCH_DEVICE)` creates the grid.
- `AgentsRegistry(grid)` creates the registry.
- `SimulationStats()` creates the stats collector.
- `add_random_walls(grid)` mutates the map.
- `make_zones(...)` creates control-point and heal-zone structures.
- Initial agents are spawned by either:
  - `spawn_symmetric(...)` when `SPAWN_MODE == "symmetric"`, or
  - `spawn_uniform_random(...)` otherwise.

The engine is then created as:

```python
engine = TickEngine(registry, grid, stats, zones=zones)
```

This means the engine always starts from already-created world objects, whether those objects are fresh or restored.

## 2.4 Resume mode

Resume mode is chosen when `config.CHECKPOINT_PATH` is non-empty.

The resume path in `main.py` does the following:

1. Load checkpoint data on CPU first with `CheckpointManager.load(checkpoint_path, map_location="cpu")`.
2. Restore the grid from `ckpt["world"]["grid"]` onto `config.TORCH_DEVICE`.
3. Reconstruct zones through `CheckpointManager.zones_from_checkpoint(...)`.
4. Create empty runtime containers:
   - `AgentsRegistry(grid)`
   - `SimulationStats()`
5. Create the `TickEngine`.
6. Apply the checkpoint into those objects with `CheckpointManager.apply_loaded_checkpoint(...)`.
7. Verify that restored `stats.tick` matches checkpoint metadata tick.

This is important: resume is not “load model and continue”. Resume is “load a serialized runtime payload, create new runtime objects, then mutate them into restored state”.

## 2.5 Output mode versus no-output inspector mode

The repository distinguishes between normal output-producing execution and a special inspection mode.

`main._inspector_no_output_mode_active()` returns `True` when either:

- `FWS_INSPECTOR_MODE` is set to one of:
  - `ui_no_output`
  - `inspect`
  - `inspector`
  - `no_output`
  - `viewer_no_output`
- or `FWS_INSPECTOR_UI_NO_OUTPUT` is true.

When that mode is active, `main.py` prints that no results, telemetry, checkpoints, or files will be created. In that branch:

- no run directory is created,
- no `ResultsWriter` is started,
- no `TelemetrySession` is created,
- no `CheckpointManager` is created,
- the recorder is `_NoopRecorder()`.

The simulation can still be viewed, but operational side effects are intentionally suppressed.

This is a meaningful runtime mode. It is the cleanest built-in way to visually inspect a checkpoint or world state without polluting result trees.

## 2.6 UI mode versus headless mode

`config.ENABLE_UI` controls whether the repository uses the viewer or the headless loop.

- If `ENABLE_UI` is true, `Viewer.run(...)` is used.
- If `ENABLE_UI` is false, `main._headless_loop(...)` is used.

Inspector no-output mode still goes through the viewer path; it is not a separate loop implementation.

Operationally, that means there are two main execution loops:

- `Viewer.run(...)` for interactive work,
- `_headless_loop(...)` for unattended long runs.

## 2.7 Resume output continuity and resume branching

The code recognizes that “resume from checkpoint” and “append into original run directory” are related but separate decisions.

The relevant config knobs are:

- `CHECKPOINT_PATH`
- `RESUME_OUTPUT_CONTINUITY`
- `RESUME_FORCE_NEW_RUN`
- `RESUME_APPEND_STRICT_CSV_SCHEMA`

In `main.py`, resume-in-place append becomes active only when:

- a checkpoint is requested,
- `RESUME_OUTPUT_CONTINUITY` is true,
- `RESUME_FORCE_NEW_RUN` is false.

When that condition is true, `main.py` infers the original run directory from the checkpoint path and starts `ResultsWriter` in append mode on that directory.

When it is false, resume still restores runtime state, but outputs go into a fresh run directory.

This is an important distinction.

**Resume** answers: “What state does the simulation continue from?”

**Output continuity** answers: “Do new artifacts continue the original lineage tree, or do they branch into a new run directory?”

## 2.8 Configuration profiles

`config.py` defines profile presets through `FWS_PROFILE` with documented values:

- `default`
- `debug`
- `train_fast`
- `train_quality`

These profiles do not override explicit environment variables. They only replace defaults that the operator did not set manually.

Operationally useful profile facts from the provided config:

- `debug` keeps UI enabled, reduces grid/population scale, disables video recording, and disables vmap.
- `train_fast` disables UI and enables vmap.
- `train_quality` disables UI and enables vmap with a larger `BRAIN_MLP_D_MODEL`.

These are convenience defaults, not separate code paths.

## 2.9 Launch-time consequences that propagate downstream

The operator’s launch decisions affect later behavior in concrete ways.

### If you change `FWS_UI`

You are not only changing whether a window appears. You are changing which main loop owns tick advancement, where manual checkpoint hotkeys exist, and whether frame cadence matters.

### If you change `FWS_CHECKPOINT_PATH`

You are not only loading weights. You are switching to a state-restoration path that also restores registry data, PPO runtime state, engine state, catastrophe and respawn state, and RNG state.

### If you change `FWS_INSPECTOR_MODE`

You are not only suppressing file writes. You are preventing run-directory creation, telemetry creation, checkpoint creation, and recorder output.

### If you change `FWS_RESUME_OUTPUT_CONTINUITY`

You are changing not simulation truth, but artifact lineage. This affects append compatibility of CSVs and telemetry chunk numbering.

### If you change telemetry schema-related settings

You may make resume-in-place append invalid even if the checkpoint itself is still loadable.

---

## 3. Viewer / UI Responsibilities and Operator Controls

## 3.1 What the viewer is

The viewer lives in `ui/viewer.py`. Its responsibility is to provide interactive visualization and operator controls around a running simulation.

It does **not** own the simulation truth. The simulation truth lives in the engine, registry, grid, zones, and stats. The viewer reads from those objects, advances them by calling `engine.run_tick()`, and caches CPU-side snapshots for rendering and picking.

This distinction is essential.

## 3.2 Plain-language intuition

The viewer is a window that lets the operator:

- look at the world,
- pause or advance it,
- inspect agents and zones,
- toggle overlays,
- trigger certain zone/catastrophe controls,
- request checkpoints,
- export one selected brain’s weights,
- change viewing speed and camera state.

It is a control shell around the runtime, not the runtime itself.

## 3.3 Formal runtime role

`Viewer.run(engine, registry, stats, tick_limit=0, target_fps=None, run_dir=None)` executes the UI loop. Inside that loop, the viewer:

1. Processes input events.
2. Optionally performs a manual checkpoint save if requested.
3. Decides how many simulation ticks to advance this frame.
4. Runs those ticks through `engine.run_tick()`.
5. Performs automatic trigger-file and periodic checkpoint checks when checkpointing is enabled.
6. Refreshes cached CPU state on configured frame cadences.
7. Renders the world and side panels.
8. Caps frame rate with `pygame.time.Clock`.
9. Stops when `tick_limit` is reached or when the operator exits.

If `run_dir` is provided, the viewer constructs its own `CheckpointManager(run_dir)` and can save checkpoints. If `run_dir` is `None`, manual checkpoint requests remain UI-visible but cannot actually write checkpoint files.

## 3.4 Viewer state that is separate from simulation state

The viewer has its own internal state, including:

- `paused`
- `speed_multiplier`
- camera offsets and zoom
- `show_rays`
- `threat_vision_mode`
- `battle_view_enabled`
- `show_brain_types`
- `fullscreen`
- selected slot and remembered selected UID
- a list of marked slots
- selected zone state
- checkpoint request flag
- cached CPU state for rendering and picking
- per-frame refresh cadence counters

This viewer state is operationally useful, but it is not the authoritative simulation state.

## 3.5 Cached rendering and picking data

The viewer intentionally caches CPU-side data.

`ui/viewer.py` keeps:

- `_cached_state_data`
- `_cached_id_np`
- `_cached_occ_np`
- `_cached_alive_indices`
- `_cached_agent_map`

These are refreshed according to:

- `VIEWER_STATE_REFRESH_EVERY`
- `VIEWER_PICK_REFRESH_EVERY`

The purpose is to reduce repeated GPU-to-CPU synchronization work every frame.

This has an important consequence: what you see or click in the viewer is based on periodic CPU snapshots, not on a promise that every render call directly re-reads every live GPU tensor in real time.

That is a performance and architecture choice, not a bug.

## 3.6 What the viewer reads and what it does not own

The viewer reads from:

- `grid`
- `registry`
- `stats`
- `engine` accessors or engine-linked helpers
- zone and catastrophe state
- internal `agent_scores` used for display and checkpoint payloads

The viewer does not own:

- the registry schema,
- the simulation tick algorithm,
- the PPO runtime,
- the results writer,
- the structured telemetry session,
- final shutdown summary creation.

It can trigger actions that interact with those systems, but it does not replace them.

## 3.7 Verified operator controls

The event handling in `ui/viewer.py` supports the following verified controls.

### Basic navigation and simulation control

- `ESC` — exit the viewer loop.
- `SPACE` — toggle pause.
- `.` when paused — advance exactly one tick.
- `W`, `A`, `S`, `D` or arrow keys — pan camera.
- Mouse wheel up/down — zoom in/out through `cam.zoom_at(...)`.
- `+` or `=` — double `speed_multiplier`, capped at `16`.
- `-` — halve `speed_multiplier`, floored at `0.25`.
- `F11` — toggle fullscreen.

### Selection and inspection

- Left click on an agent — inspect/select via grid picking.
- Left click on a heal zone — select zone.
- `M` with an active selected slot — mark or unmark the selected slot, up to 10 markers.

### Overlay toggles

- `R` — toggle ray overlay.
- `T` — toggle threat vision mode.
- `B` — toggle battle view.
- `N` — toggle brain type labels.

### Zone and catastrophe controls

- `Y` — toggle catastrophe scheduler.
- `0` — restore all zones to normal effective state.
- `1` — manual catastrophe `"random_small"`.
- `2` — manual catastrophe `"random_medium"`.
- `3` — manual catastrophe `"left_side"`.
- `4` — manual catastrophe `"right_side"`.
- `5` — manual catastrophe `"cluster_survives"`.
- Right click or shift-click on a zone — toggle that zone state.

### Persistence-related controls

- `F9` — request a manual checkpoint save.
- `S` — save the selected agent brain weights to a `.pth` file.

These controls are also reflected in the viewer legend and side-panel help text.

## 3.8 The important difference between `F9` and `S`

These two keys can look similar to a beginner because both “save something”. They are operationally very different.

### `F9` manual checkpoint save

This triggers `CheckpointManager.save_atomic(...)` from the viewer loop, between ticks, and includes:

- world/grid state,
- zones,
- registry tensors,
- agent UIDs,
- generations,
- per-slot brains,
- engine runtime state,
- PPO runtime state,
- stats,
- RNG state,
- viewer payload.

This is a resume-oriented save.

### `S` save selected brain

This calls `save_selected_brain()` and writes only the selected brain’s `state_dict()` to a file named like:

```text
brain_agent_<uid>_t_<tick>.pth
```

This is **not** a full checkpoint. It is only a PyTorch weight export for one selected agent brain.

There is another operational detail that matters: this brain export is written into the current working directory, not into the run directory. That is easy to overlook.

## 3.9 Manual checkpoint behavior inside the viewer

When `F9` is pressed, the viewer sets `save_requested = True`. The actual save happens later in the viewer loop, before tick advancement for that frame. That makes the save occur at a safe between-ticks boundary.

When checkpointing is enabled, the viewer saves an additional `viewer_state` payload containing:

- `paused`
- `speed_mult`
- camera `offset_x`
- camera `offset_y`
- camera `zoom`
- `agent_scores`

The checkpoint note is `"manual_hotkey_F9"` and pinning behavior follows:

- `CHECKPOINT_PIN_ON_MANUAL`
- `CHECKPOINT_PIN_TAG`

If checkpointing is not enabled because `run_dir` is absent, the viewer records a status string saying that checkpoint save was requested but `run_dir` is not set.

## 3.10 What a beginner should understand before editing the viewer

Before changing `ui/viewer.py`, understand these operational facts:

1. The viewer is not the source of simulation truth.
2. Viewer rendering depends on periodic CPU caches.
3. Viewer controls can trigger real runtime side effects, including checkpoints and zone state changes.
4. The viewer loop performs automatic checkpoint checks when checkpointing is enabled.
5. Viewer manual checkpoint payloads include viewer data, but the provided resume path does not restore that viewer data.
6. Viewer brain export is a convenience weight save, not a resume surface.
7. UI changes can still affect operations if they alter control flow, checkpoint requests, or zone-toggle logic.

---

## 4. Run Directories, Output Files, and Artifact Interpretation

## 4.1 Where outputs are created

Normal output-producing runs create a run directory and write multiple artifact families into it.

The default fresh-run path comes from `ResultsWriter.start(config_obj=_config_snapshot())`, which in turn falls back to `ResultsWriter._timestamp_dir(base="results")`.

That helper produces directories shaped like:

```text
results/sim_YYYY-MM-DD_HH-MM-SS
```

### Important code-truth note

`config.py` defines `RESULTS_DIR` from `FWS_RESULTS_DIR`, but in the provided fresh-run path `main.py` does not pass that config value into `ResultsWriter.start()`. `ResultsWriter._timestamp_dir()` defaults to the literal base `"results"`.

So, in the provided code path, fresh runs are created under `results/...` unless an explicit run directory is supplied through another path such as resume-in-place append.

That is an operationally important detail because a reader might expect `FWS_RESULTS_DIR` to control fresh-run directory placement directly.

## 4.2 Run directory naming and lineage

A normal new run gets a timestamped folder like:

```text
results/sim_2026-03-17_18-35-04
```

A resume can behave in two ways:

- **resume-in-place append** — continue writing into the original run directory inferred from the checkpoint path,
- **resume-into-new-run** — restore state from a checkpoint but create a new timestamped run directory for subsequent artifacts.

This is controlled by output continuity settings, not by checkpoint loadability itself.

## 4.3 High-level output tree

A typical output tree can look like this:

```text
results/
└── sim_YYYY-MM-DD_HH-MM-SS/
    ├── config.json
    ├── stats.csv
    ├── dead_agents_log.csv
    ├── summary.json
    ├── crash_trace.txt                  # only on crash
    ├── summary_fallback.txt             # only if summary JSON write fails
    ├── simulation_raw.avi               # only if video recording enabled and available
    ├── telemetry/
    │   ├── schema_manifest.json
    │   ├── run_meta.json
    │   ├── agent_life.csv
    │   ├── lineage_edges.csv
    │   ├── agent_static.csv             # optional
    │   ├── tick_summary.csv             # optional cadence-based
    │   ├── move_summary.csv
    │   ├── counters.csv
    │   ├── telemetry_summary.csv        # headless live summary sidecar
    │   ├── ppo_training_telemetry.csv   # optional
    │   ├── mutation_events.csv          # optional
    │   ├── dead_agents_log_detailed.csv
    │   └── events/
    │       ├── events_000000.jsonl
    │       ├── events_000001.jsonl
    │       └── ...
    └── checkpoints/
        ├── latest.txt
        ├── ckpt_t<tick>_<timestamp>/
        │   ├── checkpoint.pt
        │   ├── manifest.json
        │   ├── DONE
        │   └── PINNED                  # optional
        └── ...
```

Not every file exists for every run. Several surfaces are conditional on configuration and runtime mode.

## 4.4 What is written once versus continuously

### Written once at startup or initialization

- `config.json` — written by `ResultsWriter` initialization for new runs.
- `telemetry/schema_manifest.json` — written for fresh telemetry lineage.
- `telemetry/run_meta.json` — written for fresh telemetry lineage when enabled.
- `telemetry/agent_static.csv` header and data surface — initialized when enabled.
- initial bootstrap birth events — written through telemetry when not doing resume-in-place append.

### Written continuously during the run

- `stats.csv` — per-tick rows through `ResultsWriter`.
- `dead_agents_log.csv` — death batches through `ResultsWriter`.
- telemetry event chunks under `telemetry/events/`.
- `telemetry/agent_life.csv` snapshots.
- `telemetry/move_summary.csv`.
- `telemetry/counters.csv`.
- `telemetry/tick_summary.csv` on configured cadence.
- `telemetry/telemetry_summary.csv` in headless mode on cadence.
- `telemetry/ppo_training_telemetry.csv` when enabled.
- `telemetry/mutation_events.csv` when enabled.
- periodic checkpoint directories under `checkpoints/`.
- optional video frames into `simulation_raw.avi`.

### Written at shutdown

- `summary.json`
- `summary_fallback.txt` on JSON failure
- `crash_trace.txt` on crash path
- on-exit checkpoint if enabled

## 4.5 Legacy writer surface versus telemetry surface

The repository has two different artifact styles.

### Legacy/background writer surface

Produced by `utils.persistence.ResultsWriter`:

- `config.json`
- `stats.csv`
- `dead_agents_log.csv`

This path is simple and lightweight. It is also intentionally non-blocking and may drop writes if the inter-process queue is full.

### Rich telemetry surface

Produced by `utils.telemetry.TelemetrySession` under `telemetry/`.

This path is more structured, append-aware, schema-aware, and analysis-oriented.

A newcomer should not assume that these two systems are redundant. They overlap in theme, but not in exact purpose or fidelity.

## 4.6 How to interpret the most important artifacts first

If you are new to a run directory, inspect in roughly this order:

1. `summary.json` — final status, duration, final tick, scores, error field.
2. `config.json` — configuration snapshot for the run.
3. `checkpoints/latest.txt` — latest checkpoint lineage pointer.
4. `telemetry/run_meta.json` — static run metadata if telemetry is enabled.
5. `telemetry/schema_manifest.json` — the append-compatibility and lineage schema definition.
6. `stats.csv` — coarse tick-by-tick score and damage progression.
7. `telemetry/tick_summary.csv` and `telemetry/telemetry_summary.csv` — denser operational summaries.
8. `dead_agents_log.csv` and `telemetry/dead_agents_log_detailed.csv` — death surfaces.
9. `telemetry/events/` — fine-grained event chunks if deeper reconstruction is needed.

## 4.7 Artifact interpretation warnings

### Do not assume `stats.csv` is a perfect ground-truth ledger

`ResultsWriter` uses a bounded queue and intentionally drops messages instead of blocking the simulation when the queue is full.

That makes `stats.csv` operationally useful, but not a strict durability guarantee.

### Do not assume the absence of a fresh `run_meta.json` means telemetry is broken

On resume-in-place append, `main.py` intentionally preserves the original telemetry `run_meta.json` by default rather than overwriting it.

### Do not assume an exported `.pth` brain file is part of the run tree

Viewer brain exports go to the current working directory, not the run directory.

### Do not assume a checkpoint directory without `DONE` is loadable

`CheckpointManager.load(...)` explicitly refuses to load a checkpoint when the `DONE` marker is absent.

---

## 5. Checkpointing and Resume: What Is Saved, When, and Why

## 5.1 Plain-language intuition

A checkpoint in this repository is a snapshot of the runtime, not just of neural-network weights.

The code is designed around long-running simulations. Because of that, a faithful resume needs to preserve:

- the world,
- the agents,
- their brains,
- training buffers,
- optimizer and scheduler state,
- game-controller state,
- stats,
- and randomness state.

That is why the checkpoint payload is large and multi-part.

## 5.2 Checkpoint directory structure

`utils.checkpointing.CheckpointManager` stores checkpoints under:

```text
<run_dir>/checkpoints/
```

Each completed checkpoint is a directory:

```text
ckpt_t<tick>_<YYYY-MM-DD_HH-MM-SS>/
```

Inside it, the code writes:

- `checkpoint.pt`
- `manifest.json`
- `DONE`
- `PINNED` optionally

The `checkpoints/` root also contains:

- `latest.txt`

`latest.txt` points to the latest checkpoint directory name.

## 5.3 Atomic save design

`CheckpointManager.save_atomic(...)` uses a cautious pattern:

1. Build a temporary directory ending in `__tmp`.
2. Write the payload files into that temp directory.
3. Write the `DONE` marker only after content is ready.
4. Atomically rename the temp directory into its final checkpoint name.
5. Update `latest.txt`.

This design is important because partially written checkpoint directories are dangerous. The `DONE` marker is the explicit signal that a checkpoint completed successfully.

The load path respects that by refusing to load when `DONE` is missing.

## 5.4 What the checkpoint payload contains

The top-level checkpoint dictionary written by `save_atomic(...)` contains these sections:

- `checkpoint_version`
- `meta`
- `world`
- `registry`
- `engine`
- `ppo`
- `stats`
- `viewer`
- `rng`

### `meta`

Includes:

- current tick,
- timestamp,
- notes,
- saved device,
- runtime device,
- git commit.

### `world`

Includes:

- CPU copy of the grid,
- serialized zones payload or `None`.

### `registry`

Includes:

- `agent_data`
- `agent_uids`
- `generations`
- `next_agent_id`
- `brains` as a per-slot list of either `None` or `{kind, state_dict}`

### `engine`

Includes:

- `agent_scores`
- respawn controller subset
- catastrophe controller payload

### `ppo`

Contains PPO runtime checkpoint state when PPO is enabled.

### `stats`

Contains serialized statistics state.

### `viewer`

Contains viewer state if supplied by the caller, otherwise an empty dictionary.

### `rng`

Contains:

- Python random state,
- NumPy RNG state,
- torch CPU RNG state,
- torch CUDA RNG states.

## 5.5 What PPO checkpointing means in this repository

The PPO portion of the checkpoint is far richer than “policy weights”.

`rl/ppo_runtime.py` exposes `get_checkpoint_state()` and `load_checkpoint_state()`. The checkpointed PPO state includes:

- per-slot rollout buffers with observation, action, log-probability, value, reward, done, and action-mask information,
- optimizer state per agent,
- scheduler state per agent,
- training step counters,
- update sequence numbers,
- rich telemetry row sequence,
- value caches,
- cache validity flags,
- pending window agent IDs,
- pending window done state.

This is one of the most important operational truths in the repository.

A faithful resume is trying to preserve learning continuity, not just inference continuity.

## 5.6 When checkpoints are written

The provided code supports four checkpoint timings.

### A. Manual viewer-triggered checkpoint

From the UI with `F9`.

### B. Manual external trigger-file checkpoint

Through `CheckpointManager.maybe_save_trigger_file(...)`, using a trigger file in the run directory root whose filename defaults to `checkpoint.now`.

### C. Periodic tick-based checkpoint

Through `CheckpointManager.maybe_save_periodic(...)`, controlled by `CHECKPOINT_EVERY_TICKS`.

### D. On-exit checkpoint

In the final shutdown block of `main.py`, controlled by `CHECKPOINT_ON_EXIT`.

## 5.7 Trigger-file checkpoint behavior

`maybe_save_trigger_file(...)` checks whether a trigger file exists. If it does:

- it reads the text content if possible,
- decides whether the checkpoint should be pinned,
- uses the file content as notes or falls back to `"manual_trigger"`,
- saves the checkpoint,
- deletes the trigger file only after a successful save,
- prunes older checkpoints according to keep-last-N rules.

The pinning rule is:

- pinned if `default_pin` is true,
- or if the trigger file text contains `"pin"` or `"keep"`.

This is a useful operator mechanism for long headless runs because it allows a manual “save now” signal without shutting down the process.

## 5.8 Periodic checkpoint behavior

`maybe_save_periodic(...)` saves only when:

- the configured period is positive,
- current tick is greater than zero,
- current tick is an exact multiple of the save period,
- the same tick was not already saved in the same periodic path.

After saving, it prunes older checkpoints according to `CHECKPOINT_KEEP_LAST_N`.

## 5.9 Pruning behavior

Checkpoint pruning has protective rules.

`prune_keep_last_n(...)`:

- only considers completed checkpoint directories,
- ignores temp or incomplete directories,
- never deletes pinned checkpoints,
- never deletes the checkpoint currently named by `latest.txt`.

This means “keep last N” does not mean “destroy everything else without exception”.

## 5.10 How resume is initiated

Resume is initiated by setting `CHECKPOINT_PATH`.

`CheckpointManager.load(...)` accepts:

- a direct path to `checkpoint.pt`,
- a checkpoint directory containing `checkpoint.pt`,
- a checkpoints root directory containing `latest.txt`, which it resolves to the latest checkpoint subdirectory.

That makes the load interface flexible at the path level.

## 5.11 How restore is applied

After load, `CheckpointManager.apply_loaded_checkpoint(...)` mutates runtime objects in place.

It validates and restores, among other things:

- registry tensor rank and dimensions,
- registry capacity and column count,
- brain list length,
- generation list length,
- UID list length if present,
- registry tensor contents onto target device,
- `agent_uids`,
- `generations`,
- next agent ID,
- per-slot brains re-created by `kind` and loaded from `state_dict`,
- architecture metadata rebuild where supported,
- engine `agent_scores`,
- respawn controller state,
- statistics state,
- catastrophe controller state,
- PPO runtime state,
- RNG state last.

The order matters.

RNG state is restored last because other restore steps may allocate or mutate tensors. Restoring RNG too early could break deterministic continuation assumptions.

## 5.12 What is restored conceptually

The resume contract is trying to restore four conceptual layers:

### A. World continuity

Grid and zones.

### B. Agent continuity

Registry tensor state, unique IDs, generations, brains, architecture metadata.

### C. Learning continuity

PPO buffers, optimizer state, scheduler state, value caches, pending windows.

### D. Stochastic continuity

Python, NumPy, and torch RNG states.

That is why “resume” in this repository is a strong term.

## 5.13 Viewer state and resume truth

The checkpoint format includes a `viewer` section, and the UI path saves viewer payload for manual `F9` checkpoints.

However, the provided startup and restore path does not apply checkpoint `viewer` payload back into the viewer on resume.

That means:

- the checkpoint file may contain viewer metadata,
- but viewer camera, pause, speed, and similar UI state are not part of the effective restore path in the provided code.

A maintainer should not promise users that viewer state is restored unless they add the missing restore path.

## 5.14 Resume continuity versus append continuity

A checkpoint can be correctly loadable while append continuity still fails or becomes unsafe.

Examples:

- `stats.csv` header mismatch under strict append mode,
- telemetry schema manifest mismatch,
- a changed event schema or reward/death field contract,
- a changed checkpoint schema that loader no longer matches.

A run can resume in simulation terms but still be unsafe to append into the exact same artifact lineage.

The code treats those concerns separately for a reason.

## 5.15 Why persistence is not “just save weights”

A common beginner statement is:

> “If I save model weights, I can always resume training later.”

That is false for this repository.

Here, meaningful continuity also depends on:

- rollout buffers,
- optimizer state,
- scheduler state,
- value cache state,
- pending PPO window state,
- registry slot identity and unique IDs,
- world and zone state,
- catastrophe controller state,
- respawn controller state,
- stats,
- RNG.

Without those, a resumed run may continue in some loose sense, but it would not be the same continuation contract the provided checkpointing system is designed to preserve.

---

## 6. Telemetry, Logs, Metadata, and Inspection Surfaces

## 6.1 The repository has multiple inspection surfaces

This codebase emits information through more than one path.

The main inspection surfaces are:

- console prints,
- `stats.csv`,
- `dead_agents_log.csv`,
- final `summary.json`,
- `crash_trace.txt` on failure,
- rich telemetry files under `telemetry/`,
- checkpoint manifests,
- optional PyTorch profiler traces,
- optional raw video recording.

These surfaces are not identical. Some are for quick live monitoring. Some are for later analysis. Some are for crash recovery. Some are for lineage integrity.

## 6.2 Console output

### Startup console output

`main.py` prints:

- seed information,
- compact configuration summary via `config.summary_str()`,
- whether inspector no-output mode is active,
- whether it is resuming or starting fresh,
- chosen spawn mode when fresh,
- engine initialization status,
- result directory path,
- checkpoint save status on exit.

### Headless periodic console output

`_headless_loop(...)` prints periodic status every `HEADLESS_PRINT_EVERY_TICKS` ticks when that value is positive.

The message always includes:

- tick,
- red and blue scores,
- elapsed simulation seconds.

Depending on `HEADLESS_PRINT_LEVEL`, it adds more detail.

Depending on `HEADLESS_PRINT_GPU`, it can include a GPU summary line.

This console stream is useful for live monitoring, but it is not the main persistent scientific ledger.

## 6.3 `summary.json`

At shutdown, `main.py` writes `summary.json` atomically with fields including:

- `status`
- `started_at`
- `duration_sec`
- `final_tick`
- `elapsed_seconds`
- `scores`
- `error`

If JSON write fails, the code attempts `summary_fallback.txt`.

This makes `summary.json` the quickest final-status artifact for a run directory.

## 6.4 Crash tracing

On exception, the crash path writes:

```text
crash_trace.txt
```

inside the run directory if one exists.

This matters because a failed run may still have enough artifacts to diagnose the error if the crash trace is preserved.

## 6.5 The lightweight writer outputs

### `stats.csv`

Written from `SimulationStats.as_row()`, which includes fields such as:

- `tick`
- `elapsed_s`
- `red_score`, `blue_score`
- `red_kills`, `blue_kills`
- `red_deaths`, `blue_deaths`
- `red_dmg_dealt`, `blue_dmg_dealt`
- `red_dmg_taken`, `blue_dmg_taken`
- `red_cp_points`, `blue_cp_points`

This is a compact coarse timeseries.

### `dead_agents_log.csv`

Written from `SimulationStats.record_death_entry()` and `drain_dead_log()` with rows including:

- `tick`
- `agent_id`
- `team`
- `x`
- `y`
- `killer_team`

This is a simpler legacy death ledger.

### Durability warning

Because `ResultsWriter` is non-blocking and drops messages on full queue, these CSVs are useful but not strict durability surfaces.

## 6.6 Telemetry directory purpose

`utils.telemetry.TelemetrySession` writes into:

```text
<run_dir>/telemetry/
```

This subsystem is intended as a richer and more structured analysis surface than the background writer.

It also includes append-aware behavior for resume-in-place continuity.

## 6.7 Verified telemetry files and their roles

### `schema_manifest.json`

Defines the lineage schema contract for the telemetry tree.

`main.py` either writes it for a fresh lineage or validates compatibility against it for resume-in-place append.

### `run_meta.json`

Static run metadata, written for fresh lineages unless resume-in-place append intentionally preserves the original file.

### `agent_life.csv`

A snapshot-style persistent ledger of per-agent lifecycle totals and death metadata.

### `lineage_edges.csv`

Parent-child lineage relationships.

### `agent_static.csv`

Optional one-time static per-agent info when enabled.

### `tick_summary.csv`

Cadence-based summary rows across the run.

### `move_summary.csv`

Movement summary surface.

### `counters.csv`

Long-format `(tick, key, value)` style extension-friendly counters sink.

### `telemetry_summary.csv`

Headless live summary sidecar written on cadence when enabled.

### `ppo_training_telemetry.csv`

Optional PPO-rich telemetry CSV.

### `mutation_events.csv`

Optional rare mutation event log.

### `dead_agents_log_detailed.csv`

Richer death ledger than the root `dead_agents_log.csv`.

### `events/events_XXXXXX.jsonl`

Chunked event stream files.

## 6.8 Event chunking and resume continuity

Telemetry event chunks are append-safe because `TelemetrySession` discovers the next event chunk index by scanning existing chunk files.

That means resume-in-place append can continue chunk numbering instead of overwriting existing event chunk files.

This is one of the places where the telemetry system is more explicitly append-aware than the simpler background writer.

## 6.9 Schema compatibility enforcement

The telemetry system treats schema compatibility seriously.

`validate_schema_manifest_compat(...)` compares:

- `schema_version`
- `lineage_fields`
- `reward_fields`
- `death_causes`
- `mechanics`

That matters because resume-in-place append is not just “continue writing rows somewhere”. It assumes the meaning of those rows remains stable.

## 6.10 Tick summaries and headless summaries

### Tick summary

`_write_tick_summary(...)` writes per-summary-cadence rows that include, among other values:

- tick
- elapsed seconds
- alive counts by team
- mean HP by team
- kills and deaths
- damage dealt and taken

### Headless live summary

`on_headless_tick(...)` and `_write_headless_summary(...)` write `telemetry_summary.csv` when enabled. This can include:

- wall-clock elapsed time,
- TPS window and average,
- score,
- control-point totals and deltas,
- alive counts,
- HP means,
- kills and deaths,
- damage totals and deltas,
- optional tick metrics,
- optional GPU fields,
- optional PPO last-train-summary fields.

This file is particularly useful for unattended runs.

## 6.11 Movement, death, lineage, and PPO surfaces

The telemetry layer is where the code joins operational monitoring and scientific interpretation.

Examples:

- move summaries provide locomotion history,
- agent life tracks accumulated reward and death metadata,
- lineage edges preserve ancestry relationships,
- detailed death logs carry richer death interpretation,
- PPO-rich telemetry exposes training-process information,
- mutation event logging preserves rare mutation observations.

A newcomer should inspect telemetry when they need deeper explanation than `stats.csv` can provide.

## 6.12 Flush behavior and horizon alignment

Telemetry is explicitly closed **before** the final on-exit checkpoint and summary are written. `main.py` comments explain why: it wants final telemetry snapshots and events to reach the same horizon before checkpoint and summary are frozen.

This ordering is a good example of operational discipline. It reduces mismatch between:

- what the final checkpoint claims the world state is,
- and what telemetry has actually flushed to disk.

## 6.13 Optional video recording

If `RECORD_VIDEO` is enabled and OpenCV is available, `_SimpleRecorder` writes:

```text
simulation_raw.avi
```

into the run directory.

The recorder:

- reads occupancy channel `grid[0]`,
- maps occupancy values through a fixed palette,
- scales frames,
- writes every `VIDEO_EVERY_TICKS` ticks,
- is injected by monkey-patching `engine.run_tick`.

This is a live visualization artifact, not a telemetry replacement.

## 6.14 Optional profiler traces

`utils.profiler.torch_profiler_ctx(...)` can enable PyTorch profiling through `FWS_TORCH_PROFILER`.

Its default output directory is:

```text
prof
```

and it uses `torch.profiler.tensorboard_trace_handler(out_dir)`.

This is an opt-in diagnostics surface. It is not part of normal run-directory lineage.

---

## 7. Debugging Workflows and Practical Operator Playbooks

## 7.1 First debugging principle: ask which layer failed

When a run behaves strangely, identify the failing layer first.

Possible failure categories include:

- simulation truth corruption,
- viewer/rendering issue,
- background writer lossiness,
- telemetry schema mismatch,
- checkpoint save failure,
- checkpoint load incompatibility,
- resume continuity mismatch,
- crash during shutdown or flush.

Do not start by assuming everything failed at once.

## 7.2 Debugging a strange live run

If the run is still alive but suspicious:

1. Check console output first.
2. In headless mode, inspect periodic status prints and GPU summary if enabled.
3. If UI is active, use pause and single-step to see whether the anomaly is persistent or transient.
4. If possible, create a checkpoint:
   - `F9` in the viewer,
   - or trigger-file checkpoint in a run directory for headless mode.
5. Inspect `telemetry_summary.csv` and `tick_summary.csv` if telemetry is active.
6. If the issue may be state corruption, remember that headless mode periodically calls `runtime_sanity_check(...)` every 500 ticks.

## 7.3 Debugging a failed run after it exits

Inspect in this order:

1. `summary.json`
2. `crash_trace.txt` if present
3. latest checkpoint in `checkpoints/`
4. `telemetry/run_meta.json`
5. `telemetry/schema_manifest.json`
6. tail of `stats.csv`
7. tail of `telemetry/telemetry_summary.csv`
8. recent `events_*.jsonl` chunks
9. `dead_agents_log_detailed.csv` if the issue involves deaths or combat

This order separates quick status, crash reason, recoverability, schema continuity, and then detailed event analysis.

## 7.4 Debugging resume problems

When resume fails or behaves incorrectly, inspect three different contracts.

### A. Checkpoint completeness contract

- Does the checkpoint directory have `DONE`?
- Does `checkpoint.pt` load?
- Is `manifest.json` present and consistent?

### B. Restore-shape contract

- Did registry capacity change?
- Did registry column count change?
- Did brain list length change?
- Did checkpointed world structure match expected shape?

`CheckpointManager.apply_loaded_checkpoint(...)` explicitly validates these.

### C. Append-lineage contract

- Did CSV schemas drift?
- Did telemetry schema manifest drift?
- Is resume-in-place append being used when it should not be?

A checkpoint can be loadable even when append continuity should not be trusted.

## 7.5 Debugging viewer anomalies

When the viewer looks wrong, ask:

1. Is the simulation wrong, or is the render cache stale?
2. Is the selected agent or zone still alive/valid?
3. Is an overlay toggle causing a misread?
4. Is the issue only visual, or do persisted stats and telemetry show the same anomaly?
5. Was the viewer started in inspector no-output mode, making persistence features intentionally unavailable?

Remember that the viewer uses cached CPU snapshots with configurable refresh intervals.

## 7.6 Debugging artifact gaps

If you notice missing rows or uneven artifacts:

- missing or sparse `stats.csv` rows can result from `ResultsWriter` queue drops,
- telemetry gaps may reflect cadence choices or feature toggles,
- no new `run_meta.json` on resume-in-place append can be intentional,
- no checkpoint files in inspector no-output mode is expected by design,
- no video file with recording enabled can occur if OpenCV is unavailable or the writer failed to open.

## 7.7 Practical operator playbooks

### Playbook A: Safe long headless run

Use when you care about sustained unattended operation.

1. Disable UI.
2. Keep telemetry enabled.
3. Keep periodic checkpointing enabled.
4. Keep on-exit checkpoint enabled.
5. Use periodic headless prints at a cadence that is informative but not noisy.
6. Treat trigger-file checkpointing as the “save now” escape hatch.
7. Watch telemetry summaries rather than relying only on console output.

### Playbook B: Checkpoint inspection without side effects

Use when you want to open a resumed world visually without creating a new results lineage.

1. Set `CHECKPOINT_PATH`.
2. Enable UI.
3. Enable inspector no-output mode.
4. Understand that results, telemetry, checkpoints, and files are intentionally suppressed.

### Playbook C: Resume in place for one continuous lineage

Use when you want one artifact tree to represent the whole run across interruptions.

1. Set `CHECKPOINT_PATH`.
2. Keep `RESUME_OUTPUT_CONTINUITY` true.
3. Keep `RESUME_FORCE_NEW_RUN` false.
4. Preserve CSV and telemetry schemas.
5. Do not change semantics of key telemetry fields between segments.

### Playbook D: Resume as a new branch experiment

Use when you want the same runtime state but a new output lineage.

1. Set `CHECKPOINT_PATH`.
2. Disable output continuity or force a new run directory.
3. Treat new results as a branch, not as a direct continuation ledger.

---

## 8. Persistence Boundaries and Resume Integrity

## 8.1 Why persistence boundaries matter

A simulation checkpoint is only as trustworthy as the boundary it captures.

In this repository, resume integrity depends on multiple subsystems remaining aligned across save and restore. The boundary is wider than many readers first expect.


## 8.1.1 Persistence-boundary sketch

A compact mental model for this repository is:

```text
things that may be visible in the window
    !=
things that are persisted in root CSVs
    !=
things that are persisted in telemetry
    !=
things required for faithful resume

faithful resume boundary =
    world
  + registry
  + brains
  + engine controllers
  + PPO runtime internals
  + stats
  + RNG
```

That inequality chain is the safest beginner correction in this codebase.

## 8.2 The main persistence boundary

A faithful resume in this codebase spans:

```text
world/grid
+ zones
+ registry tensors
+ slot-local brain objects
+ unique IDs and generations
+ engine-side runtime fields
+ respawn controller state
+ catastrophe controller state
+ stats
+ PPO runtime state
+ RNG state
```

That boundary is what the checkpoint system is designed to preserve.

## 8.3 World and registry synchronization

The engine comments and registry design make clear that the repository uses both:

- a grid/world tensor,
- a registry tensor with agent rows,
- plus Python-side lists for brains and metadata.

These surfaces must remain synchronized.

A checkpoint that restores only the grid but not the registry would be invalid. A checkpoint that restores registry tensor rows but not the per-slot brains would also be invalid for learning continuity. A checkpoint that restores per-slot brains but not `agent_uids` or `next_agent_id` would damage lineage and identity continuity.

## 8.4 Unique IDs and generations are part of integrity

`engine.agent_registry.py` maintains:

- authoritative integer `agent_uids`,
- `generations`,
- `_next_agent_id`,
- architecture metadata (`brain_arch_ids`) alongside `brains`.

Those are not cosmetic fields. They are part of runtime identity.

If they drift, lineage tracking, selection continuity, and architecture bucketing can all become confused.

## 8.5 PPO continuity is stateful

The PPO runtime is stateful in ways that matter operationally:

- pending rollout windows,
- optimizer moments,
- scheduler position,
- cached values,
- rich telemetry row sequence,
- per-slot buffers.

A resume that resets these while claiming seamless continuation would be misleading.

The provided checkpoint system does not make that mistake; it stores them.

## 8.6 RNG state is part of reproducible continuity

Restoring RNG state is not a decorative feature.

If RNG state were not restored, then even after restoring the visible world, the continuation would diverge immediately when random-dependent logic next executes. That can affect:

- respawn details,
- mutation behavior,
- training stochasticity,
- any random policy-sampling or environment randomness.

That is why the checkpoint stores Python, NumPy, and torch RNG state and restores RNG last.

## 8.7 Resume integrity risks

The major integrity risks in this codebase are:

1. changing registry schema or capacity,
2. changing checkpoint schema without compatible load logic,
3. changing observation or action contracts while attempting to reuse old brain states,
4. changing PPO runtime internals without updating checkpoint serialization,
5. changing telemetry schema while appending into the same run directory,
6. changing zone/catastrophe payload structure without compatible restoration,
7. mutating brain objects without keeping architecture metadata synchronized.

## 8.8 Viewer state is outside the effective integrity boundary

Because the provided resume path does not reapply checkpoint `viewer` payload, viewer camera and UI state are outside the effective resume integrity boundary.

That does not break simulation continuation, but it matters when making promises about what “resume” means.

## 8.9 Append integrity versus runtime integrity

It is useful to say this explicitly:

- **Runtime integrity** means the simulation resumes faithfully.
- **Append integrity** means the artifact lineage remains semantically coherent when new data is written into old files or directories.

Those are separate concerns.

The code acknowledges that separation through:

- checkpoint load validation,
- telemetry schema manifest validation,
- strict CSV schema append options,
- output continuity controls.

---

## 9. Safe Extension Philosophy

## 9.1 First principle: change the narrowest true boundary

A safe change starts by locating the narrowest subsystem boundary that actually owns the behavior you want to change.

For example:

- changing a panel layout is probably a viewer-local change,
- changing an action count is not a viewer-local change,
- changing checkpoint payload fields is not a checkpointing-only change,
- adding a new observation feature is not just an observation-builder change,
- changing per-agent brain architecture is not just a model-file change.

## 9.2 Second principle: look for hidden contracts, not only explicit APIs

This repository contains explicit functions, but it also contains hidden contracts such as:

- registry tensor shape and column count,
- observation width contracts,
- action count contracts,
- telemetry schema fields,
- resume-in-place append assumptions,
- per-slot brain-kind restoration,
- architecture bucketing metadata,
- event-chunk append numbering,
- checkpoint completeness markers.

These are the places where naive local edits create broad failures.

## 9.3 Third principle: identify whether a change is local, cross-cutting, or lineage-breaking

### Local changes

Usually affect one subsystem and do not alter persistence or schema contracts.

Examples:
- viewer colors,
- panel layout,
- text labels,
- certain pure rendering changes.

### Cross-cutting changes

Affect more than one runtime subsystem but may remain compatible if all dependent parts are updated.

Examples:
- adding an observation feature,
- adding an action,
- changing telemetry metrics,
- modifying catastrophe logic.

### Lineage-breaking changes

Can invalidate existing checkpoints, append continuity, or model compatibility.

Examples:
- changing registry schema,
- changing observation dimension without migration logic,
- changing checkpoint payload structure incompatibly,
- changing telemetry schema while appending into the same lineage.

## 9.4 Fourth principle: never confuse “code still runs” with “contracts still hold”

A change can compile and even execute while still breaking operational correctness.

Examples:

- a telemetry CSV can keep writing while its semantic meaning silently changed,
- a checkpoint can keep saving while omitting newly important runtime state,
- a viewer can keep rendering while selection picks stale or misinterpreted IDs,
- a PPO model can keep loading while its observation semantics drifted.

Safe extension requires checking contracts, not only syntax.

## 9.5 Fifth principle: treat checkpoint schema as a real compatibility surface

If you add state that must survive resume, you must decide one of three things:

1. the new state is derived and need not be checkpointed,
2. the new state must be checkpointed and restored compatibly,
3. the new state breaks compatibility and old checkpoints should no longer be treated as load-equivalent without migration logic.

Doing nothing and hoping old checkpoints still mean the same thing is not a safe strategy.

## 9.6 Sixth principle: treat telemetry schema as a scientific contract

Telemetry data is not just for human eyeballing. In this repository it also serves lineage, analysis, and append continuity.

If you change the meaning of tracked reward components, death causes, movement metrics, or mechanics flags, you should also revisit schema manifest logic and append-lineage assumptions.

---

## 10. What Common Changes Touch Across the Codebase


## 10.0.1 Change-impact matrix

This matrix is a compact reminder of where changes usually propagate.

| Change type | Viewer | Engine | Registry | PPO runtime | Checkpoint code | Telemetry | Old checkpoints risk | Resume-in-place risk |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| UI color/layout only | Maybe | No | No | No | No | No | Low | Low |
| New overlay using existing state | Yes | Maybe | Maybe | No | No | No | Low | Low |
| New observation feature | Maybe | Maybe | Maybe | Yes | Yes | Maybe | High | High |
| New action | Maybe | Yes | Maybe | Yes | Yes | Maybe | Very high | Very high |
| New environment mechanic | Maybe | Yes | Maybe | Maybe | Yes | Yes | High | High |
| New telemetry metric only | Maybe | No | Maybe | Maybe | No | Yes | Low | Medium |
| Changed telemetry semantics | Maybe | Maybe | Maybe | Maybe | No | Yes | Medium | High |
| Changed checkpoint payload | Maybe | Maybe | Maybe | Maybe | Yes | Maybe | Very high | High |
| Changed registry schema | Maybe | Yes | Yes | Maybe | Yes | Yes | Very high | Very high |
| Changed brain architecture only | Maybe | Maybe | Yes | Yes | Yes | Maybe | High | Medium to high |

Use the matrix as a first warning, not as a substitute for reading the code path.

## 10.1 Adding a new observation feature

### What it is

A new observation feature changes what the brain sees.

### What it touches

At minimum, inspect:

- observation construction logic,
- `config.OBS_DIM` and related layout constants,
- `agent/obs_spec.py`,
- brain input expectations in `agent/mlp_brain.py`,
- any observation-dependent PPO or mask logic,
- checkpoint compatibility of old brain weights,
- documentation and telemetry that assume old observation size.

### Why this is cross-cutting

`agent/mlp_brain.py` validates observation dimensions strictly. `obs_spec.py` enforces layout splits. PPO brains are serialized by `kind` and `state_dict`, so old checkpoints may load structurally only if shapes still match. If not, old checkpoints and old exported brains cease to be compatible without migration.

### Risk level

High.

## 10.2 Adding a new action

### What it is

A new action changes action-space semantics and output width.

### What it touches

At minimum, inspect:

- `config.NUM_ACTIONS` and action masks,
- policy output heads,
- action sampling logic,
- action decoding in the tick engine,
- UI labels or overlays that assume old action semantics,
- PPO rollout storage for actions and masks,
- old checkpoint compatibility for actor heads.

### Why this is cross-cutting

The action dimension appears in model outputs, PPO buffers, and engine-side action handling. This is not a one-file change.

### Risk level

Very high.

## 10.3 Changing checkpoint schema

### What it is

Adding, removing, or renaming fields in checkpoint payloads.

### What it touches

At minimum, inspect:

- `CheckpointManager.save_atomic(...)`,
- `CheckpointManager.load(...)`,
- `CheckpointManager.apply_loaded_checkpoint(...)`,
- any helper that serializes controller state,
- any helper that reconstructs zones from checkpoint world payload,
- compatibility expectations for older checkpoints.

### Why this is dangerous

Checkpoint loadability is a recoverability contract. Schema drift can produce failures that only surface when resuming after a long run.

### Risk level

Very high.

## 10.4 Modifying viewer overlays or panel rendering

### What it is

Changing how the viewer draws information.

### What it touches

Usually:

- `ui/viewer.py`,
- camera or layout helpers,
- cached render state if new data must be pulled from GPU.

### Why this is often lower risk

Pure presentation changes usually do not alter simulation truth or persistence, provided they do not change control flow, selection logic, or save behavior.

### Hidden risk

If the overlay requires new state extraction, you may accidentally add expensive GPU syncs or stale-cache bugs.

### Risk level

Low to medium, depending on whether new runtime data is required.

## 10.5 Changing output metadata

### What it is

Adding or altering entries in config snapshots, summary files, or telemetry metadata.

### What it touches

- `_config_snapshot()` in `main.py`,
- `summary.json` generation,
- telemetry run meta generation,
- schema manifest if semantics truly changed,
- append continuity if files are reused across resumed segments.

### Risk level

Medium.

## 10.6 Adding a new environment mechanic

### What it is

Changing world rules such as zone behavior, catastrophe behavior, combat consequences, or movement constraints.

### What it touches

Potentially:

- `engine/tick.py`,
- map or zone structures,
- catastrophe controller state serialization,
- telemetry mechanics manifest,
- checkpoint world payload,
- summaries and counters,
- viewer overlays if the mechanic must be visible.

### Why this is dangerous

Mechanics affect both runtime truth and interpretation layers. If the mechanic influences rewards, death causes, or movement semantics, telemetry and checkpoint compatibility both matter.

### Risk level

High.

## 10.7 Changing model architecture

### What it is

Changing the internal network shapes or brain-kind mapping.

### What it touches

- `agent/mlp_brain.py`,
- `create_mlp_brain(...)`,
- checkpoint restore by brain `kind`,
- registry architecture metadata and bucketing,
- old checkpoint and exported `.pth` compatibility.

### Hidden risk

Changing architecture without adjusting restore assumptions can make old checkpoints unloadable or semantically inconsistent.

### Risk level

High.

## 10.8 Changing telemetry collection

### What it is

Adding, removing, or redefining telemetry outputs.

### What it touches

- `utils/telemetry.py`,
- manifest validation logic,
- headless summary fields,
- append schema strictness,
- downstream analysis expectations.

### Why this matters

Telemetry resume-in-place append depends on compatible lineage definitions. A telemetry change can be scientifically meaningful even when it is operationally easy to code.

### Risk level

Medium to high.

## 10.9 Changing registry internals

### What it is

Changing `NUM_COLS`, slot semantics, UID handling, or architecture metadata.

### What it touches

This is one of the widest changes in the repository.

It can affect:

- engine tick logic,
- checkpoint validation,
- checkpoint restore,
- selection and picking,
- telemetry interpretation,
- architecture bucketing,
- sanity checks.

### Risk level

Very high.

---

## 11. Common Beginner Misreadings and Operational Traps

## 11.1 “Resume only needs model weights”

Incorrect.

In this repository, faithful resume also depends on world state, registry state, PPO buffers, optimizer state, scheduler state, controller state, stats, and RNG.

## 11.2 “Viewer state is the simulation state”

Incorrect.

The viewer is a UI shell with caches, camera state, overlays, and selections. It reads and advances simulation truth, but it does not define that truth.

## 11.3 “If a checkpoint contains viewer data, the viewer must restore from it”

Incorrect in the provided code.

Viewer payload may be saved into checkpoints, but the provided startup path does not restore that payload into the viewer.

## 11.4 “Changing one environment variable is isolated”

Often incorrect.

Some environment variables change broader operational behavior:

- `FWS_UI`
- `FWS_CHECKPOINT_PATH`
- `FWS_RESUME_OUTPUT_CONTINUITY`
- `FWS_INSPECTOR_MODE`
- telemetry schema and cadence variables

A configuration change can affect persistence, lineage, append behavior, and runtime side effects.

## 11.5 “A changed telemetry schema is harmless if the code still writes CSVs”

Incorrect.

Resume-in-place telemetry continuity assumes stable schema meaning. The code even validates important schema-manifest fields for that reason.

## 11.6 “UI edits cannot affect runtime assumptions”

Incorrect.

UI edits can affect:

- checkpoint request timing,
- zone toggling,
- catastrophe controls,
- selection/picking correctness,
- performance through cache refresh policy.

Purely cosmetic edits are usually safe, but not every UI edit is cosmetic.

## 11.7 “`stats.csv` is the definitive record of every tick”

Incorrect.

`ResultsWriter` intentionally drops messages when its queue is full instead of blocking the simulation.

## 11.8 “`summary.json` is enough to understand a run”

Incorrect.

It is the fastest final-status artifact, not the full operational history.

## 11.9 “`FWS_RESULTS_DIR` controls fresh run output location in the provided runtime path”

Not in the fresh-run path shown by the provided code.

Fresh runs call `ResultsWriter.start()` without a base/run_dir argument, and `ResultsWriter` defaults to the literal base `"results"`.

## 11.10 “Autosave-by-seconds is part of the active runtime behavior”

`config.py` defines `AUTOSAVE_EVERY_SEC`, but no active usage of that setting appears in the provided orchestration and checkpointing path. The active periodic save path is tick-based through `CHECKPOINT_EVERY_TICKS`.

## 11.11 “Target TPS throttling is part of the active runtime behavior”

`config.py` defines `TARGET_TPS`, but no active use of that setting appears in the provided runtime loop.

## 11.12 “Saving a selected brain is a safe substitute for checkpointing”

Incorrect.

It produces a `.pth` weight file for one brain in the current working directory. It does not preserve engine, registry, world, RNG, or PPO continuity.

---

## 12. Post-Change Verification Checklist

This checklist is for maintainers who changed code and want to verify that they did not silently break operational contracts.

## 12.1 Core startup verification

- Confirm `python main.py` still starts.
- Confirm fresh-run startup reaches engine creation.
- Confirm resume startup reaches `apply_loaded_checkpoint(...)` when given a valid checkpoint.
- Confirm inspector no-output mode starts without creating a run directory.

## 12.2 Output-tree verification

- Confirm fresh runs still create a run directory.
- Confirm `config.json`, `stats.csv`, and `dead_agents_log.csv` still appear when expected.
- Confirm `summary.json` still writes on normal shutdown.
- Confirm `crash_trace.txt` still appears on forced crash tests.

## 12.3 Checkpoint verification

- Confirm manual viewer checkpoint (`F9`) still produces a completed checkpoint directory.
- Confirm trigger-file checkpoint still works from the run directory root.
- Confirm periodic tick-based checkpoint still fires at expected tick multiples.
- Confirm `latest.txt` points to the newest checkpoint.
- Confirm incomplete checkpoint directories are still rejected by load logic.
- Confirm pruning still preserves pinned checkpoints and the `latest.txt` target.

## 12.4 Resume verification

- Load the latest checkpoint and confirm:
  - tick matches expected value,
  - registry capacity and slot data restore,
  - brains restore by kind,
  - stats restore,
  - run continues without immediate shape or schema failure.
- If PPO was active, confirm training resumes without missing-buffer or missing-optimizer-state errors.
- Confirm RNG restoration order has not been accidentally broken.

## 12.5 Resume-in-place append verification

- Resume into the original run directory with append continuity enabled.
- Confirm `stats.csv` appends rather than rewrites.
- Confirm telemetry event chunk numbering continues rather than overwrites.
- Confirm schema manifest validation still passes.
- Confirm the original `run_meta.json` preservation behavior remains as intended.

## 12.6 Viewer verification

- Confirm pause, single-step, speed controls, zoom, and selection still work.
- Confirm overlay toggles still function.
- Confirm zone toggling and catastrophe controls still behave as expected.
- Confirm manual checkpoint request still works when `run_dir` is set and correctly reports when `run_dir` is absent.
- Confirm exported brain `.pth` still writes when requested.

## 12.7 Telemetry verification

- Confirm `telemetry/` still initializes when enabled.
- Confirm `schema_manifest.json` writes for fresh lineages.
- Confirm `record_resume(...)` emits a resume event on resume.
- Confirm `agent_life.csv`, `tick_summary.csv`, and `telemetry_summary.csv` still update on their expected cadences.
- Confirm flush-on-close still writes the final horizon.

## 12.8 High-risk schema verification

After any change to observation dimension, action dimension, registry schema, checkpoint format, or telemetry meaning:

- test fresh run,
- test checkpoint save,
- test checkpoint load,
- test resume-in-place append,
- test resume-as-new-run,
- compare old and new artifacts intentionally,
- document incompatibilities explicitly.

## 12.9 Silent-regression checks

These are the regressions that are easy to miss:

- stale telemetry append into incompatible schema,
- viewer-only state appearing correct while simulation truth is wrong,
- old checkpoints loading but with semantically wrong restored state,
- root CSVs dropping rows under high pressure,
- architecture metadata desynchronizing from `brains`,
- UID or generation continuity breaking without immediate crashes,
- changed working-directory assumptions for selected brain exports.

---

## 13. How This Volume Connects the Full Documentation Set

This operations volume is the maintenance bridge for the rest of the documentation set.

The earlier conceptual volumes explain what the simulation is, what the world mechanics mean, what observations and actions mean, and how learning works.

This volume explains how those systems behave as a live program with files, state transitions, recovery paths, sidecar data, and operational hazards.

After finishing this volume, a reader should approach the repository as a system with four simultaneous views:

1. **mechanics view** — what the world and agents do,
2. **learning view** — how policies and PPO evolve,
3. **operations view** — how runs are launched, observed, checkpointed, and resumed,
4. **maintenance view** — how changes propagate across hidden contracts.

A reader who can hold all four views at once is ready to maintain the code safely.

---

## Appendix A. Verified Runtime Artifact Map

| Artifact | Where | When created | What it contains | Why it matters |
|---|---|---:|---|---|
| `config.json` | run root | startup | JSON config snapshot from `_config_snapshot()` | Baseline run context |
| `stats.csv` | run root | startup, then per tick | coarse timeseries from `SimulationStats.as_row()` | quick trend inspection |
| `dead_agents_log.csv` | run root | during run | simple death ledger from drained death buffer | basic death trace |
| `summary.json` | run root | shutdown | final status, duration, tick, scores, error | quickest final-status file |
| `summary_fallback.txt` | run root | rare shutdown fallback | plain-text summary if JSON write fails | recovery fallback |
| `crash_trace.txt` | run root | on crash | Python traceback | failure diagnosis |
| `simulation_raw.avi` | run root | during run if recording enabled | raw occupancy-based video | visual replay surface |
| `telemetry/` | run root | startup if telemetry enabled | structured telemetry tree | deeper analysis and append continuity |
| `telemetry/schema_manifest.json` | telemetry dir | fresh lineage startup | schema/mechanics contract | append compatibility |
| `telemetry/run_meta.json` | telemetry dir | fresh lineage startup | static run metadata | context for analysis |
| `telemetry/agent_life.csv` | telemetry dir | during run | per-agent lifecycle totals and outcomes | lineage and reward interpretation |
| `telemetry/lineage_edges.csv` | telemetry dir | during run | ancestry relationships | genealogy tracking |
| `telemetry/agent_static.csv` | telemetry dir | optional startup/writes | one-time agent static fields | static metadata |
| `telemetry/tick_summary.csv` | telemetry dir | cadence-based | summary rows across ticks | compact scientific summary |
| `telemetry/move_summary.csv` | telemetry dir | during run | movement totals/aggregates | locomotion analysis |
| `telemetry/counters.csv` | telemetry dir | during run | generic `(tick,key,value)` counters | extension-friendly metrics |
| `telemetry/telemetry_summary.csv` | telemetry dir | cadence-based in headless mode | live headless summary with optional TPS/GPU/PPO fields | unattended-run monitoring |
| `telemetry/ppo_training_telemetry.csv` | telemetry dir | optional | PPO rich telemetry rows | training-process inspection |
| `telemetry/mutation_events.csv` | telemetry dir | optional | rare mutation events | mutation analysis |
| `telemetry/dead_agents_log_detailed.csv` | telemetry dir | during run | richer death records | deeper death diagnosis |
| `telemetry/events/events_*.jsonl` | telemetry/events | during run | chunked event stream | fine-grained reconstruction |
| `checkpoints/latest.txt` | run root `checkpoints/` | after checkpoint saves | latest checkpoint dir name | fast latest resolution |
| `checkpoints/ckpt_t.../checkpoint.pt` | checkpoint dir | on save | full serialized checkpoint payload | true resume source |
| `checkpoints/ckpt_t.../manifest.json` | checkpoint dir | on save | human-readable checkpoint metadata | inspection and bookkeeping |
| `checkpoints/ckpt_t.../DONE` | checkpoint dir | after successful save | completion marker | load safety gate |
| `checkpoints/ckpt_t.../PINNED` | checkpoint dir | optional | retention marker | protects from pruning |
| `brain_agent_<uid>_t_<tick>.pth` | current working directory | viewer `S` action | selected brain `state_dict()` | one-brain export, not full resume |

---

## Appendix B. Verified Checkpoint/Resume Flow at a Glance

```text
SAVE SIDE
---------
tick boundary reached
    ->
manual F9 OR trigger-file OR periodic OR on-exit path
    ->
CheckpointManager.save_atomic(...)
    ->
serialize:
    meta
    world
    registry
    engine
    ppo
    stats
    viewer (optional)
    rng
    ->
write checkpoint.pt + manifest.json into temp dir
    ->
write DONE
    ->
atomic rename temp dir -> final checkpoint dir
    ->
update latest.txt
    ->
optional prune_keep_last_n()

LOAD SIDE
---------
set CHECKPOINT_PATH
    ->
CheckpointManager.load(...)
    ->
resolve directory/file/latest.txt target
    ->
refuse load if DONE missing
    ->
load checkpoint.pt
    ->
main.py restores grid and zones first
    ->
create empty registry + stats + TickEngine
    ->
CheckpointManager.apply_loaded_checkpoint(...)
    ->
restore registry tensors, uids, generations, brains
    ->
restore engine-side state
    ->
restore stats
    ->
restore catastrophe/respawn/PPO state
    ->
restore RNG last
    ->
continue via viewer loop or headless loop
```

Key truth: this flow restores far more than model weights.

---

## Appendix C. Compact Safe-Extension Mental Model

Use this mental model before changing anything:

```text
What do I want to change?
    ->
Is it only presentation?
    -> viewer-only, usually low risk
Is it runtime behavior?
    -> check engine + stats + telemetry + checkpoint impact
Is it agent schema or policy contract?
    -> check registry + brains + PPO + checkpoint compatibility
Is it artifact meaning?
    -> check telemetry schema + append continuity
Is it state that must survive resume?
    -> add save + load + validation, or declare incompatibility
```

A practical rule set:

1. Change the smallest real owner.
2. Search for every place that serializes, restores, summarizes, or validates the thing you changed.
3. Test fresh run and resume separately.
4. Treat append continuity as a separate verification target.
5. Never promise compatibility the code does not explicitly preserve.
