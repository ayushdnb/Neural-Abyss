# Getting Started

This document describes the verified first-run path for the inspected repository.

No `requirements.txt`, `pyproject.toml`, or `environment.yml` file was present in the inspected source snapshot. The setup guidance below is therefore inferred conservatively from actual imports used by the runtime.

## Prerequisites

### Required for the main runtime path

The main inspected path imports these libraries directly:

- Python 3
- `torch`
- `numpy`
- `pygame`

### Optional for specific features

These imports are optional or feature-gated:

- `cv2` / OpenCV for raw AVI recording in `main.py`
- `pyarrow` in `recorder/` utilities
- `imageio` in `recorder/video_writer.py`

The current `main.py` path does **not** require `pyarrow` or `imageio` for a normal run.

## Repository entrypoint

The inspected public repository layout exposes `main.py` at the repository root. The main entry command is therefore:

```bash
python main.py
```

## First-run installation example

A conservative environment setup is:

```bash
python -m venv .venv
```

### Windows PowerShell

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch numpy pygame-ce
```

### Linux or macOS

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch numpy pygame-ce
```

If raw AVI recording is needed, add OpenCV:

```bash
python -m pip install opencv-python
```

## Running the project

### Default launch

By default, the inspected config enables the UI:

```bash
python main.py
```

Expected startup behavior:

- deterministic seed print
- one-line runtime summary
- fresh world creation unless a checkpoint path is configured
- wall and zone generation
- initial spawn
- tick engine initialization
- a results directory under `results/` unless no-output inspector mode is enabled
- Pygame viewer window

### Headless launch

To disable the UI and run the headless loop:

#### Windows PowerShell

```powershell
$env:FWS_UI='0'
python main.py
```

#### Linux or macOS

```bash
FWS_UI=0 python main.py
```

Expected behavior in headless mode:

- no viewer window
- periodic console status lines controlled by headless print config
- continuous writing of results and optional telemetry
- periodic and on-exit checkpoint behavior if enabled

## First successful run: what to expect

For a normal run with outputs enabled, the runtime creates a run directory under `results/`, typically named like:

```text
results/sim_YYYY-MM-DD_HH-MM-SS
```

The runtime then creates or appends artifacts such as:

- `config.json`
- `stats.csv`
- `dead_agents_log.csv`
- `summary.json`
- `checkpoints/`
- `telemetry/` if telemetry is enabled

More detail appears in [Checkpointing, results, and telemetry](10-checkpointing-results-and-telemetry.md).

## Resume from checkpoint

The resume path is controlled by `FWS_CHECKPOINT_PATH`.

The checkpoint resolver accepts:

- a checkpoint directory containing `checkpoint.pt`
- a direct path to `checkpoint.pt`
- a `checkpoints/` root directory if `latest.txt` exists there

### Windows PowerShell

```powershell
$env:FWS_CHECKPOINT_PATH='results\sim_2026-03-08_01-42-11\checkpoints\ckpt_t230749_2026-03-10_09-22-06'
python main.py
```

### Linux or macOS

```bash
FWS_CHECKPOINT_PATH='results/sim_2026-03-08_01-42-11/checkpoints/ckpt_t230749_2026-03-10_09-22-06' python main.py
```

If `FWS_RESUME_OUTPUT_CONTINUITY=1` and `FWS_RESUME_FORCE_NEW_RUN=0`, the runtime appends into the original run directory instead of starting a new one.

## Viewer-only inspection with no new output files

The inspected runtime supports an explicit no-output inspector mode. This is useful when a checkpoint should be opened in the viewer without creating a new results folder, telemetry tree, or checkpoint artifacts.

### Windows PowerShell

```powershell
$env:FWS_CHECKPOINT_PATH='results\sim_2026-03-08_01-42-11\checkpoints\ckpt_t230749_2026-03-10_09-22-06'
$env:FWS_INSPECTOR_MODE='ui_no_output'
python main.py
```

### Linux or macOS

```bash
FWS_CHECKPOINT_PATH='results/sim_2026-03-08_01-42-11/checkpoints/ckpt_t230749_2026-03-10_09-22-06' FWS_INSPECTOR_MODE='ui_no_output' python main.py
```

In this mode the runtime prints that inspector no-output mode is enabled and suppresses results, telemetry, checkpoints, and video output.

## Common first-run caveats

### Legacy internal naming still appears

The public repository name is `Neural-Abyss`, but the runtime banner still prints `Neural Siege: Custom` in `config.summary_str()`. That label is internal legacy output, not a separate project.

### UI mode is the config default

The inspected config defaults to `FWS_UI=1`. A user expecting an immediate headless run should explicitly disable UI.

### Some optional dependencies fail gracefully

- If OpenCV is missing, video recording is simply disabled.
- The `recorder/` package has additional optional imports, but it is not the main path used by `main.py`.

### Checkpoint compatibility is schema-checked

Policy-bearing checkpoints validate the observation schema before restoring policy and PPO state. A structurally loadable world checkpoint may still be rejected if the policy interface changed.

### Public prose may not match current code

The repository currently contains older public prose in `README.md` and `documentations/`. The runtime behavior documented here is aligned to the inspected code rather than to older descriptive text.

## Suggested next documents

- [Repository map](03-repository-map.md)
- [Viewer and operator guide](09-viewer-and-operator-guide.md)
- [Checkpointing, results, and telemetry](10-checkpointing-results-and-telemetry.md)
