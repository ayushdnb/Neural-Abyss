# Neural Abyss

Neural Abyss is a Python/PyTorch grid simulation with per-agent neural policies, a vectorized tick engine, an interactive Pygame viewer, and built-in persistence for long runs. The codebase combines world simulation, policy inference, per-agent PPO training, checkpoint/resume, telemetry, and optional video recording in one repository. Runtime configuration is environment-driven through `FWS_*` variables, and runs are written into timestamped directories under `results/`.

## System snapshot

| Item | Verified from code |
| --- | --- |
| Language | Python |
| Core runtime | PyTorch |
| UI | Pygame viewer in `ui/viewer.py` |
| Entry point | `main.py` |
| Default grid | `100x100` |
| Default initial population | `150` agents per team |
| Observation width | `283` |
| Action count | `41` |
| Output root | `results/` |

## Key technical points

- **Combat-first tick loop.** `engine/tick.py` resolves observation, action selection, combat, deaths, movement, environment effects, PPO logging, and respawn in discrete ticks.
- **Tensor-first simulation state.** `engine/agent_registry.py` stores agent state in a dense struct-of-arrays tensor and keeps one brain per registry slot.
- **Per-agent actor-critic brains.** `agent/mlp_brain.py` defines five MLP variants behind one shared observation contract and shared actor/critic interface.
- **Grouped inference path.** `agent/ensemble.py` batches inference by architecture bucket and can switch to a `torch.func`/`vmap` path when enabled.
- **Per-slot PPO runtime.** `rl/ppo_runtime.py` maintains rollout buffers, optimizers, schedulers, and training payloads per slot rather than using a shared policy.
- **Environment-driven operation.** `config.py` resolves most settings from `FWS_*` environment variables and supports profiles such as `default`, `debug`, `train_fast`, and `train_quality`.
- **Operational durability.** `utils/checkpointing.py`, `utils/persistence.py`, and `utils/telemetry.py` implement atomic checkpoint saves, background CSV writing, telemetry sidecars, and resume-in-place workflows.

## Repository structure

```text
.
├── main.py                    # runtime entry point and orchestration
├── config.py                  # environment-driven configuration surface
├── agent/
│   ├── mlp_brain.py           # actor-critic brain family
│   ├── ensemble.py            # grouped inference and optional vmap path
│   └── obs_spec.py            # observation schema and tokenization helpers
├── engine/
│   ├── tick.py                # core simulation loop
│   ├── agent_registry.py      # slot registry and agent tensor layout
│   ├── spawn.py / respawn.py  # initial spawn and respawn logic
│   ├── mapgen.py              # walls, zones, map features
│   ├── ray_engine/            # raycasting backends
│   └── catastrophe.py         # heal-zone catastrophe controller
├── rl/
│   └── ppo_runtime.py         # per-agent PPO runtime
├── ui/
│   ├── viewer.py              # interactive viewer
│   └── camera.py              # camera and viewport handling
├── simulation/
│   └── stats.py               # run statistics
└── utils/
    ├── checkpointing.py       # save/load/resume utilities
    ├── persistence.py         # background results writer
    ├── telemetry.py           # telemetry and event logs
    ├── profiler.py            # optional profiling hooks
    └── sanitize.py            # runtime sanity checks
```

## Start here

- Read `main.py` first for runtime orchestration and the end-to-end control flow.
- Read `engine/tick.py` next for the simulation semantics.
- Read `config.py` before changing behavior; most operational knobs live there.

## Quick start

### Prerequisites

The provided source dump does not include a lockfile or dependency manifest. The imports show these direct runtime dependencies:

- `torch`
- `numpy`
- `pygame`
- `opencv-python` only if you want video recording via `cv2`

A minimal setup is therefore:

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch numpy pygame opencv-python
```

### Run the viewer

From the repository root:

```bash
python main.py
```

The default configuration enables the UI. The viewer is implemented in `ui/viewer.py` and drives the simulation by calling `engine.run_tick()`.

Useful built-in viewer controls:

- `Space` pause or resume
- `.` single-step while paused
- mouse wheel zoom
- `F9` save a manual checkpoint
- `F11` toggle fullscreen

### Run headless

```bash
FWS_UI=0 python main.py
```

This routes execution through the headless loop in `main.py`, keeps the writer process active, and supports periodic status printing, telemetry, and checkpoint triggers.

### Resume from a checkpoint

```bash
FWS_CHECKPOINT_PATH="results/sim_YYYY-MM-DD_HH-MM-SS/checkpoints/ckpt_t..." python main.py
```

`utils/checkpointing.py` accepts any of the following as `FWS_CHECKPOINT_PATH`:

- a checkpoint directory
- a direct `checkpoint.pt` path
- a `checkpoints/` directory that contains `latest.txt`

By default, resume continues writing into the original run directory when possible.

```bash
FWS_CHECKPOINT_PATH="results/sim_YYYY-MM-DD_HH-MM-SS/checkpoints/ckpt_t..." \
FWS_RESUME_OUTPUT_CONTINUITY=1 \
FWS_RESUME_FORCE_NEW_RUN=0 \
python main.py
```

### Inspect a checkpoint without creating outputs

```bash
FWS_CHECKPOINT_PATH="results/sim_YYYY-MM-DD_HH-MM-SS/checkpoints/ckpt_t..." \
FWS_INSPECTOR_MODE=ui_no_output \
python main.py
```

This enables the viewer while disabling results creation, telemetry, and checkpoint writes for the inspection session.

## Configuration

`config.py` is the authoritative configuration surface. Most settings are resolved from environment variables prefixed with `FWS_`.

Common workflow-critical variables:

- `FWS_PROFILE` — profile preset such as `debug`, `train_fast`, or `train_quality`
- `FWS_UI` — enable or disable the Pygame viewer
- `FWS_SEED` — deterministic seed used at startup
- `FWS_CUDA` and `FWS_AMP` — device and mixed-precision controls
- `FWS_CHECKPOINT_PATH` — checkpoint to resume from
- `FWS_RESUME_OUTPUT_CONTINUITY` and `FWS_RESUME_FORCE_NEW_RUN` — resume output policy
- `FWS_INSPECTOR_MODE` / `FWS_INSPECTOR_UI_NO_OUTPUT` — no-output inspection mode
- `FWS_CHECKPOINT_EVERY_TICKS` and `FWS_CHECKPOINT_ON_EXIT` — checkpoint cadence and exit behavior
- `FWS_TELEMETRY` — enable or disable telemetry output

## Checkpoints and outputs

A normal run creates `results/sim_YYYY-MM-DD_HH-MM-SS/`. From the provided code, the main output layout is:

```text
results/sim_YYYY-MM-DD_HH-MM-SS/
├── config.json
├── stats.csv
├── dead_agents_log.csv
├── summary.json
├── simulation_raw.avi              # only when recording is enabled and OpenCV is available
├── checkpoints/
│   ├── latest.txt
│   └── ckpt_t.../
│       ├── checkpoint.pt
│       ├── manifest.json
│       ├── DONE
│       └── PINNED                 # optional
└── telemetry/
    ├── run_meta.json
    ├── schema_manifest.json
    ├── agent_life.csv
    ├── lineage_edges.csv
    ├── tick_summary.csv
    ├── move_summary.csv
    ├── ppo_training_telemetry.csv
    └── events/
```

Operational notes:

- Checkpoints are written atomically into `run_dir/checkpoints/`.
- `latest.txt` points to the latest complete checkpoint directory.
- Headless runs can trigger a manual checkpoint by creating the configured trigger file (default: `checkpoint.now`) in the run directory.
- The viewer can request a manual checkpoint with `F9`.
- On normal exit, `main.py` writes `summary.json` and, by default, an on-exit checkpoint.

## License

Intended license: MIT.

From the provided source dump, a repository-root `LICENSE` file is not verifiable. Add the standard MIT `LICENSE` file at repository root if it is not already present.
