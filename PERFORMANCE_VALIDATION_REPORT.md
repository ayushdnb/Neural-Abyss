# PERFORMANCE_VALIDATION_REPORT

## Tests Run

Focused regression runs:

```text
pytest -q tests/test_config_contracts.py tests/test_ensemble.py tests/test_obs_and_brains.py tests/test_tick_engine_mechanics.py
```

Result:
- `45 passed in 7.34s`

Full repository regression run:

```text
pytest -q
```

Result:
- `87 passed in 22.97s`

## Seed-Based Comparison Strategy

Two levels of parity checking were used:

1. Existing unit parity:
   - `tests/test_ensemble.py` already checks that loop and `vmap` grouped inference produce matching logits/values and refresh correctly after in-place parameter mutation.

2. Added end-to-end seeded tick parity:
   - `tests/test_tick_engine_mechanics.py::test_run_tick_matches_between_loop_and_vmap_inference`
   - Builds the same world twice on CPU with the same seed.
   - Runs one tick with `USE_VMAP=False` and one tick with `USE_VMAP=True`.
   - Verifies:
     - identical `TickMetrics`
     - identical `registry.agent_data`
     - identical `grid`
     - identical score/kill totals

This specifically validates that the grouped `vmap` path preserves simulation meaning for a seeded tick.

## Observation Contract Validation

Added validation coverage:

- `tests/test_obs_and_brains.py::test_validate_obs_contract_cache_rechecks_after_config_drift`

What it checks:
- a successful cached validation does not suppress future failure
- if the config-derived observation dims drift afterward, `validate_obs_contract()` still raises loudly

This protects the new contract-token cache from masking schema drift.

## Checkpoint Validation Results

Checkpoint code and schema were not changed in this patch.

Validation performed:
- full `pytest -q` run includes existing checkpoint tests
- headless and PPO smoke runs completed without checkpoint-related runtime errors while `CHECKPOINT_EVERY_TICKS=0` and `CHECKPOINT_ON_EXIT=0`

Result:
- no checkpoint regressions were detected
- backward compatibility for checkpoint payloads is unchanged by this patch

## Telemetry Validation Results

Telemetry payload semantics were not changed in this patch.

Validation performed:
- full `pytest -q` run includes existing telemetry tests
- PPO-enabled headless smoke completed with telemetry enabled

Result:
- no telemetry regressions were detected
- no telemetry schema or field-order changes were introduced

## Smoke Runs

Headless smoke:

```text
$env:FWS_UI='0'; $env:FWS_PROFILE='debug'; $env:FWS_CUDA='0'; $env:FWS_TICK_LIMIT='2'; ...
python main.py
```

Result:
- completed successfully

PPO headless smoke:

```text
$env:FWS_PROFILE='debug'; $env:FWS_UI='0'; $env:FWS_TICK_LIMIT='96'; $env:FWS_TELEMETRY='1'; $env:FWS_PPO_TICKS='8'; ...
python main.py
```

Result:
- completed successfully

Dummy-SDL UI smoke:

```text
$env:SDL_VIDEODRIVER='dummy'; $env:SDL_AUDIODRIVER='dummy'; $env:FWS_PROFILE='debug'; $env:FWS_UI='1'; ...
python main.py
```

Result:
- completed successfully

## Benchmark / Profiling Commands Used

Reproducible benchmark harness added:

```text
python benchmarks/benchmark_runtime.py --profile debug --use-vmap on --grid-w 64 --grid-h 64 --start-per-team 32 --max-agents 128 --ticks 32 --warmup 8
```

Representative post-patch outputs:

- debug/default-style headless benchmark with `USE_VMAP=True`:
  - `4.77 ticks/s`
- same benchmark with `USE_VMAP=False`:
  - `1.36 ticks/s`
- PPO-windowed benchmark (`48x48`, `16/team`, `PPO_TICKS=8`, `PPO_EPOCHS=1`, `PPO_MINIB=1`):
  - `2.45 ticks/s`

Targeted cProfile verification also showed the `vmap` cache-validation path shrinking materially after the ensemble cache change.

## Remaining Uncertainty

- The explicit `USE_VMAP=True` steady-state benchmark improved clearly in subphase profiling, but end-to-end TPS in that already-fast mode stayed within a relatively tight band. The largest measured user-visible gain comes from making the debug profile use `vmap` by default.
- Persistence-writer flush policy and telemetry-heavy runs were not changed in this patch. They should be benchmarked separately before any durability or batching policy changes are shipped.
- PPO boundary work remains a meaningful cost center once updates trigger. This patch improves the inference side of that path but does not yet redesign rollout storage or optimizer ownership.
