# Test Execution Report

## Environment Assumptions
- OS: Windows
- Shell: PowerShell
- Python:
  - sandboxed pytest runs used Python `3.10`,
  - direct runtime and escalated external runs used Python `3.14`
- Torch/CUDA:
  - CUDA was available in real runtime smokes,
  - deterministic unit/regression tests were kept on CPU.
- Sandbox boundary:
  - Windows multiprocessing named-pipe creation is blocked in the sandbox,
  - real `ResultsWriter` and `main.py` multiprocessing verification therefore required escalated execution.

## Exact Commands Run

### Phase 1 Baseline and Prior Audit
```powershell
pytest -q
pytest -q tests/test_tick_engine_mechanics.py tests/test_checkpointing.py tests/test_persistence.py tests/test_telemetry.py tests/test_ppo_runtime.py tests/test_config_contracts.py tests/test_obs_and_brains.py tests/test_ensemble.py
1..3 | ForEach-Object { pytest -q tests/test_tick_engine_mechanics.py::test_run_tick_movement_conflict_highest_hp_wins tests/test_tick_engine_mechanics.py::test_run_tick_combat_kill_updates_stats_and_registry tests/test_tick_engine_mechanics.py::test_apply_deaths_records_root_dead_log_for_metabolism_regression tests/test_checkpointing.py::test_checkpoint_save_load_and_apply_roundtrip tests/test_persistence.py::test_results_writer_surfaces_worker_schema_mismatch tests/test_telemetry.py::test_validate_schema_manifest_compat_rejects_append_drift tests/test_ppo_runtime.py::test_pending_window_bootstrap_finalizes_from_value_cache }
```

### Phase 2 Focused Subset
```powershell
pytest -q tests/test_viewer_runtime.py tests/test_persistence.py tests/test_ppo_runtime.py
```

### Phase 2 Adjacent Resume / Telemetry / Mechanics Subset
```powershell
pytest -q tests/test_viewer_runtime.py tests/test_persistence.py tests/test_ppo_runtime.py tests/test_checkpointing.py tests/test_telemetry.py tests/test_tick_engine_mechanics.py
```

### Phase 2 Repeated Adversarial Subset
```powershell
1..3 | ForEach-Object { pytest -q tests/test_viewer_runtime.py::test_viewer_run_headless_smoke_handles_events_and_manual_checkpoint tests/test_persistence.py::test_results_writer_migrates_legacy_dead_log_for_resume_append tests/test_ppo_runtime.py::test_checkpoint_resume_matches_continuous_training_over_pending_boundary }
```

### Final Repo Test Tree
```powershell
pytest -q tests
```

### Real Windows-Process Harness
```powershell
pytest -q tests/test_persistence.py::test_results_writer_real_process_schema_mismatch_harness
```

### Real SDL-Dummy UI Smoke
```powershell
$env:SDL_VIDEODRIVER='dummy'; $env:SDL_AUDIODRIVER='dummy'; $env:FWS_PROFILE='debug'; $env:FWS_UI='1'; $env:FWS_PYGAME_CE_STRICT_RUNTIME='1'; $env:FWS_TICK_LIMIT='2'; $env:FWS_GRID_W='20'; $env:FWS_GRID_H='20'; $env:FWS_START_PER_TEAM='4'; $env:FWS_MAX_AGENTS='12'; $env:FWS_RESULTS_DIR='results_phase2_ui_smoke'; $env:FWS_CHECKPOINT_EVERY_TICKS='0'; $env:FWS_CHECKPOINT_ON_EXIT='0'; $env:FWS_TELEMETRY='0'; python main.py
```

### Real Headless PPO Smoke
```powershell
$env:FWS_PROFILE='debug'; $env:FWS_UI='0'; $env:FWS_PYGAME_CE_STRICT_RUNTIME='1'; $env:FWS_TICK_LIMIT='96'; $env:FWS_GRID_W='20'; $env:FWS_GRID_H='20'; $env:FWS_START_PER_TEAM='4'; $env:FWS_MAX_AGENTS='12'; $env:FWS_RESULTS_DIR='results_phase2_ppo_smoke'; $env:FWS_CHECKPOINT_EVERY_TICKS='0'; $env:FWS_CHECKPOINT_ON_EXIT='0'; $env:FWS_TELEMETRY='1'; $env:FWS_TELEM_TICK_SUMMARY_EVERY='24'; $env:FWS_TELEM_TICK_EVERY='24'; $env:FWS_TELEM_FLUSH_EVERY='24'; $env:FWS_PPO_TICKS='8'; $env:FWS_PPO_EPOCHS='1'; $env:FWS_PPO_MINIB='1'; python main.py
```

## Failures Observed
- Historical resume defect reproduced:
  - root `dead_agents_log.csv` using header `tick,agent_id,team,x,y` failed on append when the first resumed death row included `killer_team`.
- Sandbox boundary reproduced repeatedly:
  - `ResultsWriter()` / `main.py` failed with `PermissionError: [WinError 5] Access is denied` when Windows multiprocessing pipes were created inside the sandbox.
- Repo-root `pytest -q` became unreliable after a sandbox investigation created an ACL-broken orphan directory.
  - final suite execution therefore used `pytest -q tests`, which cleanly targets the repository test tree.

## Fixes Applied
- Added append preflight and migration for historical root death logs in `utils/persistence.py`.
- Added fail-fast rejection for unknown incompatible root death-log headers before background writer startup.
- Added viewer runtime tests, PPO soak/resume-equivalence tests, persistence migration tests, and a real-process harness test.

## Re-Run Results
- Focused Phase 2 subset:
  - `12 passed, 1 skipped`
- Adjacent resume/telemetry/mechanics subset:
  - `24 passed, 1 skipped`
- Repeated adversarial subset:
  - run 1: `3 passed`
  - run 2: `3 passed`
  - run 3: `3 passed`
- Final repository test tree:
  - `50 passed, 1 skipped`
- Real Windows-process harness outside sandbox:
  - `1 passed`
- Real SDL-dummy UI smoke:
  - passed,
  - `summary.json` recorded `status = ok`, `final_tick = 2`
- Real headless PPO smoke:
  - passed,
  - `summary.json` recorded `status = ok`, `final_tick = 96`,
  - telemetry directory contained `ppo_training_telemetry.csv`, `tick_summary.csv`, `agent_life.csv`, and event chunks.

## Final Pass / Fail Status By Subsystem
- Config / env parsing: Pass
- Observation / agent / model contracts: Pass
- Viewer helper compatibility: Pass
- Viewer non-interactive runtime path: Pass
- Viewer full interactive path: Partial
- Tick mechanics / death bookkeeping: Pass
- Checkpoint save/load/apply: Pass
- PPO runtime contracts: Pass
- PPO multi-window soak / resume-equivalence: Pass
- Telemetry schema validation: Pass
- Results writer strict-schema surfacing: Pass
- Historical root death-log append continuity: Pass for supported legacy schema, fail-fast for unsupported schemas
- Real `main.py` headless PPO path: Pass
- Real `main.py` UI path under SDL dummy: Pass

## Skipped Tests With Justification
- `tests/test_persistence.py::test_results_writer_real_process_schema_mismatch_harness`
  - skipped during sandboxed `pytest -q tests` because Windows multiprocessing pipes are blocked there,
  - separately executed outside the sandbox and passed.

## Notes on Boundaries
- Viewer tests inside pytest use `PYGAME_CE_STRICT_RUNTIME = False` to avoid a pytest-only package-metadata anomaly while still exercising the real viewer object and SDL-dummy runtime.
- Strict runtime validation itself was still verified by the external SDL-dummy `main.py` smoke with `FWS_PYGAME_CE_STRICT_RUNTIME='1'`.
