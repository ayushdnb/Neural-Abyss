# Test Coverage Matrix

## Strategy Matrix

| Subsystem | Risk | What Can Go Wrong | Target Invariant | Test Level | Deterministic Strategy | Failure Injection | Expected Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Config/env parsing | High | env drift silently mutates experiment semantics | warnings, defaults, strict mode, and profile precedence are stable | Unit | reload `config` with controlled `FWS_*` values | bad ints, bad bools, invalid profiles | warnings or hard failures match contract |
| Observation / model contracts | High | shape drift silently poisons rollout/training | obs and model outputs match configured contracts | Unit | synthetic tensors and direct brain factory calls | wrong dims, bad tokens, unknown kinds | explicit failures or exact output shapes |
| Viewer runtime | High | UI path regresses without crashing unit helpers | viewer state, event handling, checkpoint save, and non-interactive run loop stay coherent | Unit / Integration / Smoke | SDL dummy runtime with posted events and tiny worlds | pause/single-step, toggle events, scheduler toggle, manual save | exact state toggles, checkpoint dir created, viewer run exits cleanly |
| Tick mechanics | High | movement/combat/heal/death logic drifts silently | world-state mutation and stats bookkeeping remain deterministic | Integration | tiny CPU worlds and forced actions | move conflicts, kills, healing, forced death | exact positions, counters, and death rows |
| Catastrophe / zones | Medium | manual override and catastrophe state desync | effective zone state is derived consistently from geometry + runtime control | Unit / Integration | tiny masks and explicit cell selection | manual toggle, scheduler toggle, restore-all | zone status snapshots and payload roundtrips |
| PPO runtime | High | pending bootstrap, optimizer state, or resume drift silently corrupt training | multi-window training remains finite and checkpoint/resume is state-correct | Unit / Integration | deterministic synthetic rollouts with actual models | illegal masks, boundary pending state, split/resume across window boundary | finite summaries, empty pending state, weight equivalence |
| Telemetry / persistence | High | schema drift or writer failure corrupts outputs | append continuity either succeeds safely or fails loudly | Unit / Integration / External harness | temp dirs, thread-backed seam, real Windows process | stats schema mismatch, legacy death-log header, unknown death-log header | surfaced runtime error, migrated CSV, real-process failure visibility |
| Checkpoint / resume | High | stale IDs or malformed state corrupt reload | save/load/apply either roundtrip or reject loudly | Integration | tiny saved runs and explicit mutation of payloads | missing `DONE`, stale `next_agent_id` | `CheckpointError` or exact state restore |
| CLI / entrypoint behavior | Medium | unit seams differ from real runtime wiring | `main.py` boot, output creation, viewer/headless modes, and PPO telemetry remain wired correctly | Smoke | small fixed-seed runs | SDL dummy UI, headless PPO run | successful `summary.json`, output tree, telemetry files |

## Coverage Summary

| Subsystem | Unit | Property-Style | Integration | Fault Injection | Regression | Resume / Serialization | Smoke | Performance-Smoke | Blind Spots |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Config and schema contracts | Yes | Partial | Partial | Yes | Yes | n/a | n/a | n/a | no exhaustive env cross-product |
| Importability and module integrity | Partial | n/a | Partial | No | No | n/a | Partial | n/a | no dedicated all-core-module import sweep |
| Observation and agent contracts | Yes | Partial | Partial | Yes | Yes | n/a | n/a | n/a | limited dtype/device matrix |
| Spawn / respawn / doctrine | Yes | Partial | Partial | Partial | Yes | Partial | n/a | n/a | no long-run population soak |
| Simulation rules and world mechanics | Yes | Partial | Yes | Partial | Yes | n/a | Partial | No | no large-map stress profile |
| PPO / training plumbing | Yes | Partial | Yes | Yes | Yes | Yes | Yes | Partial | no convergence-quality oracle |
| Telemetry / persistence / outputs | Yes | Partial | Yes | Yes | Yes | Partial | Yes | No | no single external script covering every resume branch |
| Checkpoint / resume | Yes | n/a | Yes | Yes | Yes | Yes | Partial | No | no one-shot legacy-run `main.py` resume scenario |
| UI / viewer / compatibility | Yes | n/a | Yes | Partial | Yes | Yes | Yes | No | no full interactive manual-input automation |
| CLI / entrypoint behavior | Partial | n/a | Partial | Partial | No | Partial | Yes | Partial | resume-in-place through `main.py` not fully externalized |
| Determinism / stability probes | Partial | Partial | Yes | Yes | Yes | Yes | Yes | Partial | no long CUDA golden-artifact baseline |

## Explicit Subsystem Status

| Subsystem | Unit | Property | Integration | Fault Injection | Regression | Smoke | Resume | Serialization | Performance-Smoke | Honest Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Viewer / pygame compatibility | Yes | n/a | Yes | Partial | Yes | Yes | Yes | Yes | No | strong non-interactive coverage, interactive rendering still partial |
| PPO runtime | Yes | Partial | Yes | Yes | Yes | Yes | Yes | Yes | Partial | substantially improved, not a long-convergence proof |
| ResultsWriter multiprocessing | Yes | n/a | Yes | Yes | Yes | n/a | Partial | Partial | n/a | real Windows process harness verified |
| Historical root death-log continuity | Yes | n/a | Yes | Yes | Yes | n/a | Yes | Yes | n/a | supported legacy v1 now migrated automatically |

## Honest Blind Spots
- No automated screenshot oracle or human-like interactive viewer session.
- No multi-thousand-tick PPO soak with checkpoint cycling and output diffs.
- No external `main.py` resume-from-checkpoint into a pre-seeded historical run directory as one single scripted scenario.
- Unknown historical root death-log schemas are intentionally fail-fast, not auto-migrated.
