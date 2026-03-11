# Glossary and Notation

This document is a compact reference for recurring repository terms.

## Core repository terms

### Agent

A live or dead individual represented by one slot in the registry and, when present, one per-slot brain module.

### Slot

A fixed-capacity registry position. A slot can change occupant over time because respawn can repopulate it.

### Persistent agent id / UID

An integer identity tracked separately from slot position. This is the more stable identity for lineage and long-run analysis.

### Registry

The dense slot-major tensor store plus auxiliary per-slot metadata such as brains, generations, architecture ids, and persistent ids.

### Grid

The spatial tensor representation of the world, documented by the tick engine as a three-channel tensor for occupancy, HP, and slot id.

### Tick

One discrete simulation step.

### Tick engine

`TickEngine`, the core state transition engine that applies observations, actions, combat, movement, environment, PPO bookkeeping, and respawn.

### Alive index

The 1D tensor of slot ids currently marked alive and therefore participating in the tick.

### Brain

A per-slot policy/value network module.

### Architecture bucket

A group of alive slots whose brains share the same architecture signature and can therefore be processed together for grouped inference.

### Observation schema

The exact dimensional and semantic contract for the policy input vector. In the inspected code this includes an explicit schema version and family.

### Ray features

The 32-direction first-hit perception block that occupies the first 256 dimensions of the observation.

### Rich features

The 23-column structured scalar/context block in the observation tail.

### Instinct features

The 4-column local neighborhood-density block appended after the rich-base block.

### Signed base zones

The canonical persistent zone field with values in \([-1, +1]\), where positive is beneficial, negative is harmful, and zero is dormant.

### Control-point masks

Boolean masks stored separately from the base-zone field and used for control-point logic.

### Catastrophe

A transient runtime overlay on top of the canonical signed base-zone field.

### Catastrophe master system

The top-level catastrophe enable/disable switch.

### Dynamic catastrophe scheduler

The deterministic hazard-law scheduler that can trigger catastrophes automatically when enabled.

### Apply mask

The boolean mask selecting which cells are replaced by catastrophe override values.

### Edit-lock mask

The boolean mask preventing manual base-zone edits while a catastrophe is active over selected cells.

### Effective zone field

The resolved zone field actually consumed during environment application after combining the base layer and any active catastrophe overlay.

### Respawn

The post-tick process that repopulates dead slots according to floor-based and periodic rules.

### Clone path

A respawn path in which a new occupant is derived from an existing parent slot rather than from a fully fresh brain choice.

### Rare mutation

A config-driven mutation path in the respawn system for inherited brain noise or physical drift.

### PPO runtime

`PerAgentPPORuntime`, the slot-local PPO collection and update subsystem.

### Slot-local PPO

The design in which each slot has its own model, optimizer, scheduler, buffer, and cached bootstrap state.

### Rollout window

The fixed-length PPO collection interval before a training update is prepared.

### Value cache

The slot-local cache used to avoid an extra post-step inference pass when finishing PPO windows.

### ResultsWriter

The background writer process that writes `stats.csv` and `dead_agents_log.csv`.

### Telemetry session

The structured telemetry subsystem that writes snapshots, events, counters, lineage files, and summary files.

### Resume-in-place

The run mode in which a resumed checkpoint continues appending into the original run directory instead of creating a fresh output folder.

### Inspector no-output mode

A UI inspection mode that suppresses creation of results, telemetry, checkpoints, and video artifacts.

## Notation reference

| Notation | Meaning |
|---|---|
| \(t\) | tick index |
| \(s\) | slot index |
| \(i\) | local live-agent index inside an alive subset |
| \(H, W\) | grid height and width |
| \(o_t^{(s)}\) | observation for slot \(s\) at tick \(t\) |
| \(a_t^{(s)}\) | action for slot \(s\) at tick \(t\) |
| \(\pi_{\theta_s}\) | slot-local policy of slot \(s\) |
| \(V_{\theta_s}\) | slot-local value function of slot \(s\) |
| \(r_t^{(s)}\) | PPO reward assigned to slot \(s\) |
| \(Z_{\text{base}}\) | canonical signed base-zone map |
| \(Z_{\text{override}}\) | catastrophe override field |
| \(M_{\text{apply}}\) | catastrophe apply mask |
| \(Z_{\text{eff}}\) | runtime-effective zone field |
| \(\gamma\) | discount factor |
| \(\lambda\) | GAE parameter |
| \(\varepsilon\) | PPO clip coefficient |

## Naming cautions

### `Neural-Abyss`

Public repository identity.

### `Neural Siege`

Legacy internal runtime/banner label still visible in code.

### `Infinite_War_Simulation`

Legacy path/comment label visible in source comments and snapshot paths.

### `_build_transformer_obs()`

Historical method name for the current observation builder, even though the current inspected public brain family is MLP-based.

## See also

- [Project overview](01-project-overview.md)
- [Agents, observations, and actions](06-agents-observations-actions.md)
- [Mathematical foundations](12-mathematical-foundations.md)
