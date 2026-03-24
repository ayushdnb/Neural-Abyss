# PERFORMANCE_OPTIMIZATION_PATCH_REPORT

## Executive Summary

This patch targeted the highest-confidence performance bottleneck that showed up in real profiling: per-agent brain execution and the overhead around grouped `torch.func` / `vmap` inference.

The patch does three things:

1. Keeps the `debug` profile on the grouped `vmap` inference path by default.
2. Reduces `vmap` cache-validation overhead by caching each model's flattened parameter/buffer references and by reusing a stable cache key based on ordered model identity instead of embedding full fingerprints into the key.
3. Removes repeated observation-contract revalidation work from hot-path calls by caching the validated contract token and clearing semantic-index caches only when the underlying config-derived contract changes.

The result is a material throughput improvement for the default debug/headless runtime profile and a smaller but still real improvement for a short PPO-windowed training scenario.

## Baseline Hotspots Found

Pre-patch cProfile on a direct headless benchmark (`64x64`, `32/team`, `max_agents=128`, telemetry off, respawn off):

- `agent.ensemble._ensemble_forward_loop` dominated runtime when `USE_VMAP=False`.
- With `USE_VMAP=True`, the main residual hotspot was no longer brain math. It was cache-validation:
  - `_stacked_module_state_cached`: `1.628s`
  - `_make_stack_cache_key`: `1.066s`
  - `_model_state_fingerprint`: `1.062s`
  - sample window: `8` ticks, total elapsed `3.553s`

Pre-patch PPO-windowed benchmark (`48x48`, `16/team`, `max_agents=64`, `PPO_TICKS=8`, `PPO_EPOCHS=1`, `PPO_MINIB=1`, telemetry off, respawn off):

- `rl.ppo_runtime.finalize_pending_window_from_cache` and `_train_aids_and_clear` dominated the training boundary.
- Inference-side grouped execution and cache-validation still consumed a meaningful slice of that path.

## Optimizations Applied

### 1. Debug profile now enables grouped `vmap`

File:
- `config.py`

Change:
- The `debug` profile now preserves `USE_VMAP=True` instead of forcing the slow loop path.

Why this preserves semantics:
- The repo already had a tested `vmap` path with loop fallback.
- Agent ordering, masking, sampling, and per-agent ownership remain unchanged.
- The new validation test compares a seeded `run_tick()` result with `USE_VMAP=False` vs `USE_VMAP=True` and verifies identical world state after the tick.

### 2. Faster `vmap` cache validation

File:
- `agent/ensemble.py`

Change:
- Added a weak-key cache for each model's flattened parameter/buffer tuples.
- Switched the stacked-state cache key to `(device, ordered_model_ids)` and kept freshness validation as a separate fingerprint comparison.
- `clear_stacked_vmap_cache()` now clears both the stacked-state cache and the per-model tensor-reference cache.

Why this preserves semantics:
- Freshness still comes from each tensor's in-place version counter.
- In-place parameter updates, `load_state_dict`, and mutation still invalidate reuse correctly.
- Brain replacement still misses safely because the ordered model identities change.
- No logits, values, sampling, or optimizer ownership rules changed.

Measured effect:
- In the profiled 8-tick `vmap` sample, `_stacked_module_state_cached` dropped from `1.628s` to `0.991s`.
- `_make_stack_cache_key` dropped from `1.066s` to `0.363s`.

### 3. Observation-contract validation now caches by config-derived token

File:
- `agent/obs_spec.py`

Change:
- Added a validated-contract token cache keyed by schema, dims, feature names, and semantic index mapping.
- Semantic index cache is cleared only when the contract token changes.
- Removed one redundant `validate_obs_contract()` call from `split_obs_for_mlp()` because `split_obs_flat()` already validates the same contract.

Why this preserves semantics:
- All fail-loud validation still happens whenever the contract actually changes.
- A new unit test proves the cached validator re-runs and raises when config-derived dimensions drift after an earlier successful validation.
- This changes validation frequency, not feature order, tensor layout, or runtime math.

## Benchmark Results Before/After

### Benchmark A: headless direct tick throughput

Scenario:
- direct `TickEngine.run_tick()` benchmark
- `64x64`
- `32/team`
- `max_agents=128`
- telemetry off
- respawn off
- `32` measured ticks after `8` warmup ticks

Before patch:
- debug/default behavior (`USE_VMAP=False` in the profile): `1.30 ticks/s`
- explicit `USE_VMAP=True`: `4.69 ticks/s`

After patch:
- debug/default behavior (now `USE_VMAP=True`): `4.77 ticks/s`
- explicit `USE_VMAP=False`: `1.36 ticks/s`

Interpretation:
- The main end-user win is that the default debug runtime now uses the fast grouped inference path.
- Relative to the previous debug-profile default, the measured throughput gain is about `3.7x`.
- The explicit `vmap`-on steady-state path remained in the same performance band while its cache-validation subphase got materially cheaper under profiling.

### Benchmark B: short PPO-windowed run

Scenario:
- direct `TickEngine.run_tick()` benchmark
- `48x48`
- `16/team`
- `max_agents=64`
- telemetry off
- respawn off
- `PPO_TICKS=8`
- `PPO_EPOCHS=1`
- `PPO_MINIB=1`
- `24` measured ticks after `4` warmup ticks

Before patch:
- `2.09 ticks/s`

After patch:
- `2.45 ticks/s`

Interpretation:
- The PPO-windowed case improved by about `17%` in this measured setup.
- This is smaller than the default debug-profile gain because PPO boundary work still dominates once training triggers.

## What Was Intentionally Not Changed

- No combat, movement, death, respawn, mutation, or checkpoint ordering semantics were changed.
- No observation feature order, observation width, action meaning, reward meaning, or tie-breaking logic was changed.
- No checkpoint schema or resume semantics were changed.
- No viewer/UI render semantics were changed.
- I did not ship broader world-update or respawn refactors because profiling did not justify taking on that much semantic risk in this pass.
- I did not change persistence durability defaults or telemetry payload semantics in this patch because the measured dominant bottleneck was still inference-side orchestration.

## Validation Methodology

- Full unit/integration regression suite: `87 passed`
- New/updated tests:
  - debug profile keeps `USE_VMAP` enabled
  - observation-contract cache revalidates after config drift
  - seeded `run_tick()` parity between loop inference and `vmap` inference
- Runtime smoke validation:
  - headless `main.py` smoke
  - PPO-enabled headless `main.py` smoke
  - dummy-SDL UI `main.py` smoke
- Benchmarks:
  - direct headless throughput benchmark via `benchmarks/benchmark_runtime.py`
  - targeted cProfile samples before and after the cache change

## Residual Risks

- The grouped inference fast path still depends on `torch.func` availability and compatible model structure. The loop fallback remains the safety net.
- The per-model tensor-reference cache assumes the brain module graph is structurally stable after construction, which matches current repo behavior. If future code starts swapping live parameter objects on existing model instances in new ways, the safe response is to clear the stacked `vmap` cache at that mutation boundary.
- PPO boundary cost is still significant once training triggers. This patch improves the inference side of that path but does not redesign per-slot optimizer ownership or rollout storage.

## Follow-up Opportunities

- Profile and optimize PPO boundary work after the first optimizer-creation/import cost is amortized.
- Benchmark persistence/telemetry-heavy runs separately from telemetry-off runs before changing flush or batching policy.
- Revisit observation-builder and respawn hotspots only if a new profile shows they have become dominant after these inference-side fixes.
