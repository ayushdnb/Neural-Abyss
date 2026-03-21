# Fix Log

## Production Patches

| File | Change | Why It Was Necessary | Evidence | Re-Tested With |
| --- | --- | --- | --- | --- |
| `utils/persistence.py` | Added append-mode preflight for `dead_agents_log.csv`; auto-migrates supported legacy header `tick,agent_id,team,x,y` to add `killer_team`; rejects unknown incompatible headers before starting the worker | historical resume continuity would otherwise fail late on the first resumed death write, which is unsafe for experiment output integrity | reproduced with a thread-backed `ResultsWriter` probe that failed on the first resumed death batch | `tests/test_persistence.py::test_results_writer_migrates_legacy_dead_log_for_resume_append`, `tests/test_persistence.py::test_results_writer_rejects_unknown_dead_log_header_before_worker_start`, `pytest -q tests/test_persistence.py::test_results_writer_real_process_schema_mismatch_harness` |

## Test Files Added / Expanded

| File | Purpose | Why It Was Added / Updated | Evidence Produced |
| --- | --- | --- | --- |
| `tests/test_viewer_runtime.py` | viewer runtime hardening | Phase 2 required fuller automated viewer verification | verified zone controls, legacy checkpoint-state restore, non-interactive event loop, manual checkpoint save |
| `tests/test_persistence.py` | persistence and resume continuity hardening | needed regression coverage for legacy root death-log migration and a real Windows-process schema-mismatch harness | verified migration, fail-fast unknown headers, real-process error surfacing |
| `tests/test_ppo_runtime.py` | PPO soak and resume hardening | needed deterministic multi-window soak coverage and exact checkpoint/resume equivalence | verified finite summaries over many windows and matched resumed weights against continuous training |

## Existing Phase 1 Fixes Retained

| File | Change | Why It Was Necessary | Re-Tested With |
| --- | --- | --- | --- |
| `simulation/stats.py` | forward-compatible `record_death_entry(...)` and safe missing-killer handling | root death logging was silently dropping events | full suite, tick regression, real death-path smoke |
| `utils/telemetry.py` | fixed manifest mismatch error formatting | schema drift was raising the wrong exception | full suite |
| `utils/persistence.py` | worker-error propagation and health checks | child-process output failures needed to surface to callers | full suite, real-process harness |
| `main.py` | explicit writer-close failure logging | output pipeline errors should not be silently swallowed | full suite, real smokes |
| `engine/tick.py` | corrected root death-log comment | code comment contradicted the actual legacy CSV contract | full suite |

## Re-Test Summary
- Focused Phase 2 subset: `12 passed, 1 skipped`
- Adjacent resume/telemetry/mechanics subset: `24 passed, 1 skipped`
- Final test tree: `50 passed, 1 skipped`
- Repeated adversarial subset: `3 passed` x `3`
- Real Windows-process strict-schema harness: `1 passed`
- Real `main.py` SDL-dummy UI smoke: passed
- Real `main.py` headless PPO smoke: passed
