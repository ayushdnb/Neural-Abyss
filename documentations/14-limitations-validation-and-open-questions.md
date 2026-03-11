# Limitations, Validation, and Open Questions

This document states the technical limits of what the inspected repository and this documentation suite can currently assert.

## Purpose

A repository with simulation, learning, catastrophe logic, resume semantics, and telemetry can easily be over-described. This document exists to prevent that.

## Confirmed limits of the inspected evidence base

### No automated test suite was present in the inspected source snapshot

No `tests/` directory or equivalent automated test module was present in the inspected code snapshot used for this documentation.

This does **not** prove the repository has never been tested. It does mean the documentation cannot point to an in-repository automated validation harness.

### No dependency manifest was present in the inspected snapshot

No `requirements.txt`, `pyproject.toml`, or `environment.yml` file was present in the inspected source snapshot.

Setup guidance therefore had to be inferred from actual imports.

### Public prose and current code may diverge

The public repository contains `README.md` and a `documentations/` folder. The inspected code also contains legacy internal names such as `Neural Siege` and older method names such as `_build_transformer_obs()`.

The documentation suite is intentionally aligned to the inspected code path rather than to older descriptive prose.

## Learning-related limits

### The presence of PPO code is not evidence of learning success

The repository clearly implements a PPO runtime. That does not establish:

- convergence
- policy quality
- stable emergence
- good reward shaping
- good hyperparameter tuning

Those require experiments, not code inspection alone.

### Per-slot PPO may be computationally expensive

The repository chooses slot-local model and optimizer ownership. This supports independence, but it also raises obvious efficiency questions for large populations. This document suite cannot resolve that trade-off empirically.

### Reward defaults may create unintuitive pressure

Some score and team-reward defaults are zero while some PPO individual reward defaults are nonzero. In addition, the default HP threshold-ramp setting can effectively zero HP shaping unless overridden. That is a configuration fact, not an outcome verdict, but it is a real validation concern.

## Runtime and mechanics limits

### The repository does not expose a standard environment API

The inspected runtime is built around `main.py` and `TickEngine`, not around a public `reset()/step()` environment class with a stable external API contract. That limits out-of-the-box interoperability claims.

### Some repository packages are adjacent rather than fully integrated

The `recorder/` package exists, but the main launch path currently uses `_SimpleRecorder` inside `main.py`. Until a clear integration path exists, those modules should be treated as utilities rather than as guaranteed runtime path components.

### Historical naming remains in active code

That includes:

- `Neural Siege` in the runtime summary
- `Infinite_War_Simulation` in source-path comments and snapshot paths
- `_build_transformer_obs()` for an MLP-era observation path

This is not inherently a bug, but it increases the chance of interpretive drift.

## Checkpoint and schema limits

### Policy-bearing checkpoints are schema-sensitive

The repository does the correct conservative thing by refusing incompatible policy-bearing checkpoints. That protects correctness, but it also means long-lived experiments must manage schema evolution explicitly.

### Resume-in-place append is only safe when schemas match

Persistence and telemetry append paths contain compatibility checks. That is a strength, but it also means users should not assume append into older runs is always safe.

## Telemetry limits

### Telemetry coverage is broad but config-dependent

The telemetry system can emit many files, but not every file is always present. Analyses must therefore start by checking what was actually enabled.

### Instrumentation evolution must be expected

The existence of schema manifest logic and append checks indicates that telemetry layouts can evolve. Downstream analysis should therefore verify schema identity before merging runs.

## Open technical questions a contributor should validate experimentally

### 1. Learning behavior under current defaults

Questions:

- Does per-slot PPO learn useful policies under the default reward mix?
- Are reward scales balanced?
- Does slot-local optimizer ownership produce acceptable throughput?

### 2. Sensitivity to catastrophe settings

Questions:

- How strongly do catastrophe scheduler parameters alter survival dynamics?
- Which presets materially change control-point or combat behavior?
- Does scheduler pressure create the intended activation cadence?

### 3. Observation usefulness

Questions:

- Which rich features are actually used by trained policies?
- Is the current instinct signal informative enough to justify its cost?
- Are 32 rays with 8 features each the right trade-off?

### 4. Respawn lineage semantics

Questions:

- Does the current respawn policy preserve the intended lineages?
- Are clone and mutation settings aligned with experimental goals?
- Is near-parent spawning producing desirable spatial dynamics?

### 5. Viewer/operator semantics under edge cases

Questions:

- Are manual catastrophe controls and edit locks always intuitive in long runs?
- Are no-output inspector semantics sufficient for all inspection workflows?
- Are manual checkpoint semantics still correct under repeated resume-in-place runs?

## What this documentation suite intentionally does not claim

This suite does not claim that the repository is:

- experimentally validated
- benchmarked
- tuned
- complete as a research platform
- stable across future schema changes
- free of latent performance or correctness issues

It claims only what the inspected code supports.

## Related documents

- [Learning and optimization](07-learning-and-optimization.md)
- [Checkpointing, results, and telemetry](10-checkpointing-results-and-telemetry.md)
- [Configuration and experiment control](11-configuration-and-experiment-control.md)
