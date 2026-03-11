# Neural-Abyss Documentation

This index is the navigation hub for the repository documentation suite.

The repository implements a grid-based multi-agent combat simulation with tensor-backed world state, per-slot neural policies, a per-slot PPO runtime, signed zone mechanics with runtime catastrophe overrides, a Pygame viewer, checkpointing, and structured telemetry. The suite below is written against the inspected code paths in `main.py`, `config.py`, `agent/`, `engine/`, `rl/`, `simulation/`, `ui/`, and `utils/`.

## Status note

The inspected codebase is runnable and instrumented. It includes simulation, UI, persistence, checkpointing, and learning code. The documentation does **not** claim benchmarked learning quality, algorithmic convergence, or experimental validation beyond what the code itself makes explicit.

## Suggested reading paths

### For a new reader

1. [Project overview](01-project-overview.md)
2. [Getting started](02-getting-started.md)
3. [Repository map](03-repository-map.md)
4. [Glossary and notation](15-glossary-and-notation.md)

### For an operator

1. [Getting started](02-getting-started.md)
2. [Viewer and operator guide](09-viewer-and-operator-guide.md)
3. [Checkpointing, results, and telemetry](10-checkpointing-results-and-telemetry.md)
4. [Configuration and experiment control](11-configuration-and-experiment-control.md)

### For a developer

1. [Repository map](03-repository-map.md)
2. [System architecture](04-system-architecture.md)
3. [Simulation runtime](05-simulation-runtime.md)
4. [Extension guide](13-extension-guide.md)
5. [Limitations, validation, and open questions](14-limitations-validation-and-open-questions.md)

### For an algorithm reader

1. [Agents, observations, and actions](06-agents-observations-actions.md)
2. [Learning and optimization](07-learning-and-optimization.md)
3. [Catastrophe mechanics](08-catastrophe-mechanics.md)
4. [Mathematical foundations](12-mathematical-foundations.md)

## File map

| File | Purpose |
|---|---|
| [01-project-overview.md](01-project-overview.md) | Plain-language technical overview of what the repository implements. |
| [02-getting-started.md](02-getting-started.md) | First-run setup, launch, resume, and inspection flow. |
| [03-repository-map.md](03-repository-map.md) | Codebase navigation guide and reading order. |
| [04-system-architecture.md](04-system-architecture.md) | Subsystem boundaries and data/control flow. |
| [05-simulation-runtime.md](05-simulation-runtime.md) | One-tick execution model from initialization through respawn. |
| [06-agents-observations-actions.md](06-agents-observations-actions.md) | Agent representation, observation schema, and action semantics. |
| [07-learning-and-optimization.md](07-learning-and-optimization.md) | Implemented PPO runtime, reward wiring, and training flow. |
| [08-catastrophe-mechanics.md](08-catastrophe-mechanics.md) | Signed zones, catastrophe lifecycle, scheduler, and viewer triggers. |
| [09-viewer-and-operator-guide.md](09-viewer-and-operator-guide.md) | Verified viewer controls, HUD semantics, and operator behavior. |
| [10-checkpointing-results-and-telemetry.md](10-checkpointing-results-and-telemetry.md) | Output layout, resume semantics, checkpoints, and telemetry artifacts. |
| [11-configuration-and-experiment-control.md](11-configuration-and-experiment-control.md) | Config sources, profiles, and experiment-control surfaces. |
| [12-mathematical-foundations.md](12-mathematical-foundations.md) | Formal view of state, actions, rewards, PPO, and catastrophe overlays. |
| [13-extension-guide.md](13-extension-guide.md) | Safe extension points and coupling constraints. |
| [14-limitations-validation-and-open-questions.md](14-limitations-validation-and-open-questions.md) | Honest boundary document for what is still uncertain or unvalidated. |
| [15-glossary-and-notation.md](15-glossary-and-notation.md) | Compact terminology and notation reference. |

## Related repository files

- Repository license: [`../LICENSE`](../LICENSE)
- Public README: [`../README.md`](../README.md)
- Public legacy prose folder, if present in the repository root: `../documentations/`

## Documentation conventions

- “Implemented” means the behavior is directly supported by inspected code.
- “Background” means theory needed to read the code, not a guarantee of textbook completeness.
- When a name differs between public identity and internal strings, the public repository name **Neural-Abyss** is used, and internal legacy names are called out only where needed for clarity.
- When a behavior depends on config, the documentation describes the code path first and then the main configuration knobs that alter it.
