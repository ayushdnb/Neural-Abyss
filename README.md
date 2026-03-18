# Neural-Abyss

Neural-Abyss is a PyTorch-based two-team grid combat simulation with per-agent PPO training. `main.py` starts a fresh world or resumes from a checkpoint, runs the simulation loop via `engine.tick.TickEngine`, and can operate with the Pygame viewer or headless.

All configuration is environment-driven through `config.py`; there is no CLI argument parser and no dependency manifest. The runtime requires `torch`, `numpy`, and `pygame`; `cv2` is optional (video recording only). `pygame` is required even in headless mode because `main.py` imports `ui.viewer.Viewer` at startup.

## Important Files And Folders

| Path | Purpose |
| --- | --- |
| `main.py` | Startup, fresh vs. resume flow, headless loop, viewer launch, shutdown, final summary. |
| `config.py` | Environment-variable configuration, profiles, runtime defaults, and validation. |
| `agent/` | Observation splitting, MLP brain variants, and bucketed inference helpers. |
| `engine/` | Grid state, spawning, map generation, catastrophe control, respawn, and tick execution. |
| `rl/` | Per-agent PPO runtime. |
| `simulation/` | Run statistics and death-log buffering. |
| `ui/` | Pygame viewer and camera. |
| `utils/` | Checkpointing, telemetry, persistence, profiling, and sanitization utilities. |
| `documentations/` | Focused technical documents for the engine, RL runtime, viewer, telemetry, and operations. |

## Quick Start

Install `torch`, `numpy`, and `pygame` into a Python environment, then run:

```bash
python main.py                          # viewer + PPO training (default)
FWS_UI=0 FWS_PROFILE=train_fast python main.py   # headless training
FWS_PROFILE=debug python main.py                  # small grid, viewer at 30 FPS
```

All runtime control is through `FWS_*` environment variables resolved in `config.py`.

## Common Configuration

**Profiles and basics**

- `FWS_PROFILE` — preset bundle: `default`, `debug`, `train_fast`, or `train_quality`.
- `FWS_SEED` — master random seed (default `32`).
- `FWS_UI` — `1` enables the viewer (default), `0` runs headless.
- `FWS_TICK_LIMIT` — maximum tick count; `0` = unlimited (default).
- `FWS_SPAWN_MODE` — initial layout: `uniform` (default) or `symmetric`.

**Training**

- `FWS_PPO_ENABLED` — enable per-agent PPO training (default `1`). Instantiated by `TickEngine`.
- `FWS_TELEMETRY` — structured telemetry output (default `1`).
- `FWS_RECORD_VIDEO` — raw occupancy video recording; requires `cv2` (default `0`).

**Checkpointing and resume**

- `FWS_CHECKPOINT_PATH` — resume source. Accepts a checkpoint directory, a direct `checkpoint.pt` path, or a `checkpoints/` directory with `latest.txt`.
- `FWS_CHECKPOINT_EVERY_TICKS` — periodic save interval in ticks (default `50000`).
- `FWS_CHECKPOINT_ON_EXIT` — save on clean shutdown (default `1`).
- `FWS_CHECKPOINT_KEEP_LAST_N` — retention count (default `1`).
- `FWS_RESUME_OUTPUT_CONTINUITY` — resume appends into the original run directory (default `1`).
- `FWS_RESUME_FORCE_NEW_RUN` — force a new output tree on resume (default `0`).

## Runtime Outputs

Fresh runs are created under `results/sim_YYYY-MM-DD_HH-MM-SS/`. Normal runs write `config.json` at startup and `summary.json` at shutdown.

Headless mode populates `stats.csv` and `dead_agents_log.csv` via `utils.persistence.ResultsWriter`. When `FWS_RECORD_VIDEO=1` and `cv2` is available, `simulation_raw.avi` is also written.

When telemetry is enabled, a `telemetry/` subdirectory is created containing:

- `schema_manifest.json`, `run_meta.json` — schema and run metadata.
- `events/events_*.jsonl` — chunked event log (births, deaths, damage, kills).
- `agent_life.csv`, `agent_static.csv` — per-agent snapshots and static attributes.
- `tick_summary.csv`, `move_summary.csv`, `counters.csv` — time-series summaries.
- `telemetry_summary.csv` — headless sidecar with windowed metrics.
- `ppo_training_telemetry.csv` — per-agent PPO metrics (when PPO is enabled).
- `lineage_edges.csv`, `mutation_events.csv` — respawn lineage and mutation records.
- `dead_agents_log_detailed.csv` — detailed death information.

## Checkpoints And Resume

Checkpoints are written under `run_dir/checkpoints/ckpt_t{tick}_{timestamp}/`. Each completed checkpoint directory contains `checkpoint.pt`, `manifest.json`, and `DONE`; manual or pinned saves also add `PINNED`. `run_dir/checkpoints/latest.txt` points to the latest completed checkpoint.

Periodic checkpointing is driven by `FWS_CHECKPOINT_EVERY_TICKS`. A clean shutdown saves an additional checkpoint when `FWS_CHECKPOINT_ON_EXIT=1`. Manual saves can be triggered by creating the configured trigger file in the run directory (`checkpoint.now` by default), and the viewer also exposes a manual checkpoint hotkey in normal UI mode.

Resume-in-place is enabled by default through `FWS_RESUME_OUTPUT_CONTINUITY=1`. When it is active, `main.py` infers the original run directory from the checkpoint path and appends compatible outputs there. `FWS_RESUME_FORCE_NEW_RUN=1` disables that behavior.

## Viewer

The viewer is enabled by default through `FWS_UI=1`. Verified controls in `ui/viewer.py` include:

- `Space`: pause or resume.
- `.`: single-step while paused.
- `W`, `A`, `S`, `D` or arrow keys: pan the camera.
- Mouse wheel or `+` / `-`: zoom.
- Left click: select an agent or heal zone.
- Right click or `Shift` + left click: toggle the heal zone under the cursor through the catastrophe controller.
- `F9`: manual checkpoint save request in normal UI mode.
- `F11`: fullscreen toggle.

`FWS_INSPECTOR_MODE=ui_no_output` or `FWS_INSPECTOR_UI_NO_OUTPUT=1` launches the viewer without creating results, telemetry, checkpoints, or other output files.
