# Risk Register

| Rank | Risk | Severity | Likelihood | Impact | Detection Difficulty | Mitigation Recommendation |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Full human-interactive viewer behavior is still only partially automated | Medium | Medium | platform-specific input/render regressions could escape | High | add display-backed scripted viewer sessions or screenshot assertions on a stable runner |
| 2 | PPO long-horizon scientific quality is not proven beyond contract tests, a 64-step soak, and a 96-tick runtime smoke | Medium | Medium | silent learning-quality drift can still bias conclusions | High | add 1k+ tick seeded PPO soak with checkpoint cycles and artifact sanity checks |
| 3 | Cross-hardware CUDA determinism remains unproven | Medium | Medium | repeated experiments may diverge across drivers or machines | High | add golden seeded headless runs with normalized output diffs across target hardware |
| 4 | External `main.py` resume continuity for legacy runs is not yet covered as one single end-to-end scripted scenario | Medium | Low | operational resume regressions could still surface only in field use | Medium | add a file-backed scripted resume test that starts from a seeded checkpoint and historical run directory |
| 5 | Unknown historical root `dead_agents_log.csv` schemas remain fail-fast rather than auto-migrated | Low | Medium | some older runs may refuse append continuity after upgrade | Low | keep fail-fast behavior, document supported legacy schema, add explicit one-shot migration utility if operators need it |
| 6 | Repo-root `pytest -q` is currently blocked by an ACL-broken orphan probe directory created during sandbox investigation | Low | Low | local audit reruns from repo root can fail during collection | Low | manually remove the orphan outside the repository workflow or continue using `pytest -q tests` |
