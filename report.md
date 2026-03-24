# Performance Optimization Report (Approved-Scope Pass)
Date: March 22, 2026

## 1. Scope and Guardrails
This pass was limited to the approved low-risk subset:
- A) inference-only `no_grad -> inference_mode` swaps
- B) CPU/GPU sync cleanup in hot tick/telemetry paths
- C) VMAP threshold benchmarking/tuning (keep only if proven)
- D) tiny allocation cleanup in reward/helper paths
- E) objective assessment of current `config.py` performance posture

Out-of-scope architectural rewrites were not applied.

## 2. Phase 0: Call-Path Audit and Plan
Hot paths and contracts inspected before edits:
- `engine/tick.py`: `run_tick`, `_build_transformer_obs`, combat/death processing, movement telemetry, PPO reward assembly
- `agent/ensemble.py`: bucket inference loop/vmap path
- `engine/ray_engine/*`, `engine/game/move_mask.py`: per-tick inference/masking functions
- `utils/telemetry.py`: snapshot flushing (`agent_life`) and per-slot counters
- `config.py`: `USE_VMAP`, `VMAP_MIN_BUCKET` defaults and profile behavior

Baseline benchmark command family (config-equivalent throughput scenario):
```powershell
python benchmarks/benchmark_runtime.py `
  --profile default --ticks 64 --warmup 16 `
  --grid-w 100 --grid-h 100 --start-per-team 180 --max-agents 560 `
  --use-vmap on --respawn --telemetry `
  --ppo-ticks 192 --ppo-epochs 4 --ppo-minibatches 8
```

Equivalence checks planned:
- fixed-seed deterministic state digest comparison (before/after)
- checkpoint/resume continuity checks + checkpoint/persistence test suite
- telemetry file/header sanity on a real `main.py` run

## 3. Baseline (Before Any Edits)
Resolved current config snapshot:
- `PROFILE=default`
- `TORCH_DEVICE=cuda` (`USE_CUDA=True`, `AMP_ENABLED=True`)
- `USE_VMAP=True`, `VMAP_MIN_BUCKET=8`
- `GRID=100x100`, `START_PER_TEAM=180`, `MAX_AGENTS=560`
- `RESPAWN_ENABLED=True`, `TELEMETRY_ENABLED=True`

### 3.1 Baseline throughput (5 runs, same command/settings)
`ticks_per_s`:
- 1.854308
- 1.895200
- 1.931365
- 1.929497
- 1.840509

Aggregate:
- mean: **1.890176**
- median: 1.895200
- best: 1.931365
- min: 1.840509
- stdev: 0.041896

### 3.2 Baseline VMAP threshold probe (pre-edit, 64-tick runs)
3-run means:
- `VMAP_MIN_BUCKET=8`: 1.893795
- `VMAP_MIN_BUCKET=4`: 1.991107
- `VMAP_MIN_BUCKET=2`: 1.974996

## 4. Changes Applied

### 4.1 Accepted change A: inference-only path optimization
Converted decorator from `@torch.no_grad()` to `@torch.inference_mode()` only in verified inference-only call paths:
- `agent/ensemble.py`
  - `_ensemble_forward_loop`
  - `_ensemble_forward_vmap`
  - `ensemble_forward`
- `engine/game/move_mask.py`
  - `build_mask`
- `engine/ray_engine/raycast_32.py`
  - `raycast32_firsthit`
- `engine/ray_engine/raycast_64.py`
  - `raycast64_firsthit`
- `engine/ray_engine/raycast_firsthit.py`
  - `build_unit_map`
  - `raycast8_firsthit`
- `engine/tick.py`
  - `_get_instinct_offsets`
  - `_compute_instinct_context`
  - `_build_transformer_obs`

Safety note:
- `run_tick` and PPO training/update functions were not converted.
- no changes to reward/combat semantics or ordering.

Observed throughput after A (3 runs, threshold=8):
- mean: **2.057610** (`[2.101142, 2.061772, 2.009915]`)

### 4.2 Accepted change B: CPU/GPU sync cleanup in hot paths
#### `engine/tick.py`
- `_apply_deaths`: combined repeated team/unit and XY host conversions into packed batched transfers.
- combat telemetry block:
  - removed per-element `.item()` loops for victim/attacker/per-hit metadata
  - switched to batched `index_select -> to(int64/float32) -> cpu -> tolist`
- move-event telemetry extraction:
  - replaced 6 separate transfer calls with a single packed transfer.
- merit/reward UID extraction:
  - explicit batched int64 conversion.

#### `utils/telemetry.py`
- `_flush_agent_life_snapshot`:
  - movement counters: collapsed 8 separate `.detach().cpu().tolist()` calls into one stacked CPU transfer
  - reward counters: collapsed 12 separate `.detach().cpu().tolist()` calls into one stacked CPU transfer
  - fallback behavior preserved if tensor ops fail

Safety note:
- payload fields, ordering, and schema remained unchanged.
- conversions remain explicit and deterministic.

### 4.3 Accepted change D: tiny safe allocation cleanup
In `engine/tick.py` PPO reward assembly:
- replaced repeated `torch.where(... torch.full ... torch.full ...)` allocations with one preallocated tensor per reward channel + fill + boolean assign:
  - `team_kill_reward`
  - `team_death_reward`
  - `team_cp_reward`

Semantics preserved:
- identical per-agent scalar values, only allocation strategy changed.

### 4.4 Attempted/assessed change C: VMAP threshold tuning
Extensive post-edit benchmarking was run for `8`, `4`, `2`.
Results were workload-sensitive and inconsistent across horizons:
- some 64-tick sweeps favored `8`
- other short sweeps favored `4`
- 96-tick runs favored `8`
- 128-tick runs favored `2`

Decision:
- **No `config.py` threshold change kept** (default `VMAP_MIN_BUCKET=8` retained) because no stable cross-workload winner was proven.

### 4.5 Config posture (E)
- No config defaults changed.
- Current `config.py` defaults are already near a reasonable operating point for mixed workloads.
- Threshold tuning is not one-size-fits-all in this repo; best value varies with horizon/population dynamics.

## 5. Final Combined Throughput (After Accepted Edits)
Same benchmark command/settings as baseline, 5 runs:

`ticks_per_s`:
- 2.123544
- 2.085058
- 2.054441
- 2.024336
- 2.019754

Aggregate:
- mean: **2.061427**
- median: 2.054441
- best: 2.123544
- min: 2.019754
- stdev: 0.043526

Baseline vs final (mean):
- baseline: 1.890176
- final: 2.061427
- delta: **+0.171251 ticks/s** (**+9.06%**)

## 6. Validation Evidence

### 6.1 Targeted regression tests
Executed:
```powershell
pytest -q tests/test_ensemble.py tests/test_tick_engine_mechanics.py tests/test_ppo_runtime.py
pytest -q tests/test_tick_engine_mechanics.py tests/test_telemetry.py tests/test_ppo_runtime.py
pytest -q tests/test_ensemble.py tests/test_tick_engine_mechanics.py tests/test_telemetry.py tests/test_ppo_runtime.py tests/test_checkpointing.py
pytest -q tests/test_checkpointing.py tests/test_persistence.py
```

Result highlights:
- `22 passed` on final touched-subsystem suite
- checkpoint/persistence subset: `12 passed`

### 6.2 Deterministic before/after parity (fixed seed, CPU scenario)
Same scenario executed before and after edits; summaries matched exactly:
- `tick=48`
- `alive_total=120`, `red_alive=62`, `blue_alive=58`
- kills/deaths: red `4/3`, blue `3/4`
- state digest:
  - `b168117afa27c4912f360aad0f9ae6f1d7986169bd9a9d73af892be3f0d54817`

### 6.3 Checkpoint/resume
Custom strict bitwise-equivalence resume harness (with/without PPO) did not produce exact final-state identity in this repository setup.
Official checkpoint/persistence regression tests passed, and no failures/regressions were introduced by this optimization patch.

### 6.4 Telemetry sanity
Short real run with telemetry enabled completed and produced expected files/headers:
- `results_perf_validation_post/sim_2026-03-22_17-41-57/telemetry/*`
- `tick_summary.csv`, `agent_life.csv`, `ppo_training_telemetry.csv`, events chunks, etc.
- field integrity remained intact.

## 7. Files Touched
- `agent/ensemble.py`
- `engine/game/move_mask.py`
- `engine/ray_engine/raycast_32.py`
- `engine/ray_engine/raycast_64.py`
- `engine/ray_engine/raycast_firsthit.py`
- `engine/tick.py`
- `utils/telemetry.py`
- `report.md`

## 8. Rejected/Not-Kept Ideas
- `VMAP_MIN_BUCKET` default change: rejected due inconsistent winner across realistic run horizons.
- no out-of-scope PPO/system architecture rewrites were attempted.

## 9. Final Recommendation
- Keep the accepted code-level optimizations in this patch (A + B + D).
- Keep `config.VMAP_MIN_BUCKET=8` default unless you profile for a specific fixed workload envelope and pin the threshold per experiment profile.
- For future tuning, prefer profile-specific VMAP threshold overrides (`FWS_VMAP_MIN_BUCKET`) rather than a global default shift.
