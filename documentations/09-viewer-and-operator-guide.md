# Viewer and Operator Guide

This document describes the verified viewer behavior in the inspected repository.

It covers only controls and HUD semantics that are directly supported by `ui/viewer.py`.

## Purpose

The viewer is not only a renderer. It is also an operator console for:

- inspection
- pause and single-step control
- speed adjustment
- catastrophe control
- zone editing
- checkpoint triggering
- selected-agent export

## Viewer responsibilities

The viewer loop:

- displays the world, side panel, HUD, and minimap
- polls keyboard and mouse input
- steps the engine when unpaused or when single-step is requested
- exposes selected-agent and selected-cell inspection panels
- can request checkpoints through `F9`
- can save the selected brain state dict with `S`

## Verified controls

| Control | Verified behavior |
|---|---|
| `Esc` | exit viewer loop |
| `Space` | pause or unpause |
| `.` | single-step one tick when paused |
| `WASD` / arrow keys | pan camera |
| mouse wheel | zoom |
| left mouse button | inspect/select in world |
| `M` | mark selected slot, up to a capped list |
| `R` | toggle rays |
| `T` | toggle threat vision mode |
| `B` | toggle HP-bar battle view |
| `N` | toggle brain labels |
| `Z` | toggle signed-zone overlay |
| `[` / `]` | decrease or increase selected-cell base-zone value |
| `0`, keypad `0`, `Backspace`, or `Delete` | reset selected-cell base-zone value |
| `1` | trigger `global_attenuation` catastrophe |
| `2` | trigger `positive_band_dormancy` catastrophe |
| `3` | trigger `polarity_split_left_negative` catastrophe |
| `4` | trigger `polarity_split_right_negative` catastrophe |
| `5` | trigger `regional_attenuation_left` catastrophe |
| `6` | trigger `regional_attenuation_right` catastrophe |
| `Shift+I` | trigger experimental `inversion` catastrophe |
| `Shift+O` | trigger experimental `full_dormancy` catastrophe |
| `C` | clear active catastrophe |
| `G` | toggle catastrophe master system |
| `Shift+G` | toggle dynamic catastrophe scheduler |
| `S` | save selected agent brain `state_dict()` to a `.pth` file in the current working directory |
| `F9` | save a checkpoint into the run checkpoint directory when `run_dir` is available |
| `F11` | toggle fullscreen |
| `+` / `-` | increase or decrease speed multiplier |

## Legend and help card

The help card in the side panel includes:

- unit-color legend for red and blue soldiers/archers
- beneficial, harmful, and dormant zone legend
- a compact control summary

The help card is therefore an operator aid, not just decorative UI.

## Side-panel sections

The side panel is organized around at least these functions:

- selected agent inspector
- selected cell / zone inspector
- legend and controls

## Selected agent inspection

When a live slot is selected, the viewer can show:

- agent id
- cumulative reward and kill-credit summaries if enabled
- position
- unit type
- brain description
- HP and HP max
- attack
- vision
- model summary
- parameter count

This is useful because the repository uses multiple MLP variants and slot-local model ownership.

## Selected cell / zone inspection

The selected-cell inspector can show:

- cell coordinates
- terrain label
- occupant label if present
- canonical base-zone value and interpreted state
- runtime-effective value and interpreted state
- catastrophe system on/off status
- active catastrophe display name
- whether catastrophe applies at the selected cell
- control-point mask membership
- edit-lock state
- base-zone edit step
- auto-scheduler summary including pressure, hazard, and cooldown

This is the correct operator-facing way to inspect catastrophe state.

## HUD and runtime semantics

The viewer code explicitly computes catastrophe and scheduler lines for the zone section and exposes a display-friendly catastrophe status payload through `get_catastrophe_status()`.

Important semantic distinction:

- **System on/off** indicates whether the catastrophe subsystem is allowed to run at all.
- **Auto-scheduler on/off** indicates whether scheduler-driven activation is enabled when the system itself is on.
- **Catastrophe active** indicates whether a concrete catastrophe is currently in force.

Those are different flags.

## Checkpoint hotkey behavior

`F9` requests a manual checkpoint save. When `run_dir` is available, the viewer constructs a `CheckpointManager` and saves:

- engine state
- registry state
- stats
- viewer state including pause, speed multiplier, camera, and score caches

If `run_dir` is unavailable, the viewer reports that checkpoint save was requested but no run directory is set.

## Selected-brain export

Pressing `S` saves the selected brain’s `state_dict()` to a file named like:

```text
brain_agent_<uid>_t_<tick>.pth
```

This write goes to the current working directory, not necessarily to the run directory.

## Zone editing behavior

The viewer edits the **canonical base-zone layer**, not the runtime catastrophe overlay.

Operations supported:

- add or subtract one configured edit step
- reset selected cell to zero
- respect runtime edit locks when present

That means the viewer can be used to inspect or modify base zone configuration while still preserving the catastrophe/base separation.

## Operational cautions

### `G` and `Shift+G` are intentionally different

`G` toggles the catastrophe master system. `Shift+G` toggles the dynamic scheduler only.

### Experimental catastrophe hotkeys are gated

`inversion` and `full_dormancy` require the shift-modified hotkeys. Plain `I` and `O` do not activate them.

### The viewer is a live runtime controller

Actions such as manual catastrophe activation, zone edits, and checkpoints affect the current live run. They are not sandboxed.

### Inspector no-output mode changes outer runtime behavior

The viewer itself still works in no-output inspector mode, but the outer runtime suppresses results, telemetry, checkpoints, and video output.

## Related documents

- [Catastrophe mechanics](08-catastrophe-mechanics.md)
- [Checkpointing, results, and telemetry](10-checkpointing-results-and-telemetry.md)
- [Configuration and experiment control](11-configuration-and-experiment-control.md)
