# Checkpointing, Results, and Telemetry

This document describes how the inspected runtime persists state and writes analysis artifacts.

## Purpose

The repository has unusually strong operational instrumentation relative to many simulation repositories. This document explains the output tree, checkpoint contract, and resume behavior without inventing schemas that are not present in code.

## Results directory layout

A normal run creates a directory under `results/` named like:

```text
results/sim_YYYY-MM-DD_HH-MM-SS
```

The exact base path is controlled by `FWS_RESULTS_DIR`.

## Core run artifacts

The main run directory can contain:

- `config.json`
- `stats.csv`
- `dead_agents_log.csv`
- `summary.json`
- `crash_trace.txt` on crash
- `simulation_raw.avi` if raw video recording is enabled and OpenCV is available
- `checkpoints/`
- `telemetry/`

## Asynchronous CSV writing

`utils/persistence.py` implements `ResultsWriter` as a background writer process.

Its core responsibilities are:

- write `stats.csv`
- write `dead_agents_log.csv`
- optionally write model metadata payloads

This is done out-of-process so the simulation loop does not block on disk I/O.

## Checkpoint directory structure

`CheckpointManager` writes checkpoints under:

```text
<run_dir>/checkpoints/
```

Each checkpoint is a directory named like:

```text
ckpt_t<tick>_YYYY-MM-DD_HH-MM-SS
```

A completed checkpoint directory contains at least:

- `checkpoint.pt`
- `manifest.json`
- `DONE`

It may also contain:

- `PINNED`

The `checkpoints/` root also contains `latest.txt`.

## Why checkpoints are directory-based

The implementation uses a directory per checkpoint so it can atomically finalize:

- the main binary payload
- the human-readable manifest
- completion markers
- optional retention markers

This makes pruning and resume resolution easier than a single monolithic filename scheme.

## Checkpoint contents

The checkpoint payload includes these major sections:

- `meta`
- `world`
- `registry`
- `engine`
- `ppo`
- `stats`
- `viewer`
- `rng`

### `meta`

Includes items such as:

- tick
- timestamp
- notes
- saved device
- runtime device
- git commit if available
- observation schema payload

### `world`

Includes:

- grid tensor
- zone payloads, including canonical base zones and control-point masks

### `registry`

Includes:

- `agent_data`
- `agent_uids`
- `generations`
- next agent id
- per-slot brain payloads

### `engine`

Includes:

- agent score accumulators
- agent reward totals
- respawn-controller state
- catastrophe-controller state

### `ppo`

Includes PPO runtime state when PPO is enabled.

### `rng`

Includes saved random state for Python, NumPy, and PyTorch.

## Portability and safety features

The checkpoint implementation explicitly emphasizes:

- CPU conversion for tensors before writing
- atomic write patterns
- DONE-marker validation on load
- manifest generation
- schema checks for policy-bearing checkpoint payloads

## Resume semantics

### Accepted checkpoint path forms

The checkpoint resolver accepts:

- checkpoint directory
- direct `checkpoint.pt` path
- checkpoints root directory resolved via `latest.txt`

### Resume-in-place output continuity

If:

- a checkpoint is loaded
- `FWS_RESUME_OUTPUT_CONTINUITY` is true
- `FWS_RESUME_FORCE_NEW_RUN` is false

then `main.py` infers the original run directory from the checkpoint path and appends into that run directory instead of creating a new one.

That behavior affects:

- CSV append mode
- telemetry append/compat checks
- run-directory continuity
- later checkpoint placement

### Schema validation on resume

The checkpoint loader performs explicit observation-schema checks before restoring:

- per-slot brain payloads
- PPO runtime state

A checkpoint can therefore be structurally present but still rejected if the policy interface changed.

## Telemetry directory

When telemetry is enabled, `TelemetrySession` writes under:

```text
<run_dir>/telemetry/
```

The inspected code creates or references these key outputs:

- `agent_life.csv`
- `lineage_edges.csv`
- `schema_manifest.json`
- `run_meta.json`
- `agent_static.csv`
- `tick_summary.csv`
- `move_summary.csv`
- `counters.csv`
- `telemetry_summary.csv`
- `ppo_training_telemetry.csv`
- `mutation_events.csv`
- `dead_agents_log_detailed.csv`
- `events/` chunk files

## Telemetry categories

### Snapshot-style files

Examples:

- `agent_life.csv`
- `tick_summary.csv`

These support analysis without reading a full event stream.

### Append-style lineage files

Example:

- `lineage_edges.csv`

### Chunked event logs

The telemetry code writes event logs into chunk files under `events/`, typically in JSONL format.

### Schema and run metadata

Examples:

- `schema_manifest.json`
- `run_meta.json`

These are especially important when appending into an existing run directory after resume.

## What telemetry records

The code supports recording, depending on config:

- births
- deaths
- damage aggregates
- per-hit damage events
- kill events
- movement summaries
- sampled movement events
- PPO reward components
- catastrophe transitions
- resume events
- counters
- rare mutation events

## High-level interpretation guidance

### `stats.csv`

This is the coarse run-level tick log written by `ResultsWriter`. It is useful for quick trend inspection and basic plotting.

### `dead_agents_log.csv`

This is a batch-written death log oriented around the main runtime loop rather than the richer telemetry event model.

### `agent_life.csv`

This is the long-horizon per-agent summary file. It is useful for lineage and lifetime analysis.

### `tick_summary.csv`

This is a lower-frequency telemetry summary file intended for analysis-friendly time series.

### `events/*.jsonl`

These are the highest-detail event records and can become large.

## Schema caveats

### Instrumentation can evolve

The repository contains explicit schema-version and manifest machinery. That is a direct signal that telemetry shape and interpretation can evolve across runs or across code revisions.

### Resume append can be strict

Both persistence and telemetry paths include schema-compatibility checks for append mode. That is helpful for safety, but it also means old runs may need migration rather than blind append.

### Not every output is always present

Many telemetry outputs are config-gated. Absence of a file does not imply a broken run.

## Recorder package note

The repository also contains a `recorder/` package with Arrow schema and video utilities. However, the inspected `main.py` path currently uses `_SimpleRecorder` instead of this package. The `recorder/` package should therefore be treated as adjacent repository tooling unless it is explicitly wired into the main launch path.

## Related documents

- [Getting started](02-getting-started.md)
- [Configuration and experiment control](11-configuration-and-experiment-control.md)
- [Limitations, validation, and open questions](14-limitations-validation-and-open-questions.md)
