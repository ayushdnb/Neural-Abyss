# Volume 2 — Simulation mechanics and world engine

## Document purpose

This volume explains the simulated world implemented by the `Neural-Abyss` codebase and, more specifically, the mechanics that transform world state from one tick to the next.

The goal is not to restate source code line by line. The goal is to make the runtime understandable.

A reader who finishes this volume should be able to answer all of the following without reopening the engine first:

- What the world contains.
- Which data structures hold that world.
- What an agent is in storage terms, not only gameplay terms.
- What happens during one tick, in exact phase order.
- How movement, combat, healing, metabolism, scoring, and respawn interact.
- Which assumptions the engine depends on.
- Which edits are safe, and which edits can silently break invariants.

This document is grounded in the implementation in modules such as `engine/tick.py`, `engine/agent_registry.py`, `engine/grid.py`, `engine/spawn.py`, `engine/respawn.py`, `engine/mapgen.py`, `engine/game/move_mask.py`, `engine/ray_engine/raycast_firsthit.py`, `simulation/stats.py`, and `main.py`.

---

## Audience and how to read this volume

This volume is written for two readers at once.

The first reader is new to simulation engines, tensor-based game state, and reinforcement-learning-adjacent environments. That reader needs plain-language explanations before formal ones.

The second reader is already technically mature and wants implementation truth: real state ownership, real phase order, and real failure modes. That reader needs exactness, not simplification that becomes false.

The document therefore repeats a deliberate teaching pattern whenever possible:

1. Plain-language intuition.
2. Formal mechanical description.
3. Code mapping.
4. State-transition explanation.
5. Design consequence.
6. Modification risk.

The document stays inside world mechanics. It does not try to be the neural-network volume, the PPO volume, or the persistence volume, except where those systems touch the mechanics boundary.

---

## 1. What world this codebase simulates

### 1.1 Plain-language picture

The codebase simulates a two-team battle on a discrete two-dimensional grid.

Each tick, alive agents observe the map, choose one discrete action, and then the engine applies the rules in a strict order:

1. combat,
2. death cleanup,
3. movement,
4. environment effects,
5. objective accounting,
6. tick advance,
7. respawn.

The world is not continuous physics. There is no momentum, no acceleration, no projectile flight model, and no persistent body after death. The simulation is a rule-based grid world with discrete occupancy, discrete actions, configurable range-limited attacks, and a lifecycle system that recycles dead slots back into living agents.

### 1.2 What the world does model

At the mechanical level, the world models the following:

- A bounded rectangular map with border walls and optional random interior wall segments.
- Two teams, encoded as red and blue.
- Two unit classes, soldier and archer.
- Per-agent location, health, attack power, unit class, team, vision range, and identity metadata.
- Discrete movement in eight compass directions.
- Discrete attack actions in eight directions and up to four range steps.
- Healing zones and capture-point regions stored as off-grid masks.
- Optional catastrophe scheduling that suppresses heal zones temporarily.
- Continuous attrition through metabolism.
- Death, immediate spatial removal, and later respawn.
- Team-level score, kill, death, damage, and capture counters.
- A persistent per-slot agent registry plus a spatial grid used for fast lookup.

### 1.3 What the world does not model

The implementation does **not** model the following as first-class mechanics:

- Continuous trajectories or sub-cell positions.
- Multi-cell bodies or orientation state.
- Projectile travel time.
- Push, knockback, or collision damage during movement.
- Persistent corpses or loot.
- Resource gathering or inventory.
- Terrain costs beyond wall versus non-wall.
- Poison zones or damage-over-time environmental hazards in the current runtime path.
- Direct catastrophe damage. The catastrophe system suppresses healing; it does not itself injure agents.

Some names in the codebase suggest broader future ideas, but this volume documents only behavior that is actually wired into the current engine.

### 1.4 World creation before the first tick

On a fresh start, `main.py` constructs the world in this order:

1. `engine.grid.make_grid(...)` creates a `3 x H x W` grid with boundary walls and `grid[2] = -1` everywhere.
2. `engine.mapgen.add_random_walls(...)` adds random internal one-cell-thick wall traces.
3. `engine.mapgen.make_zones(...)` creates heal zones and capture-point masks.
4. `engine.spawn.spawn_symmetric(...)` or `engine.spawn.spawn_uniform_random(...)` places the initial population.
5. `engine.tick.TickEngine(...)` binds the registry, grid, stats object, zones, catastrophe controller, and respawn controller into the runtime stepper.

That ordering matters. Initial spawn is performed after map geometry exists, so spawn logic respects walls and existing occupancy.

---

## 2. World state and representation

### 2.1 The core idea: world state lives in multiple structures

The simulation uses more than one state container, and each container serves a different purpose.

At the highest level there are four important state families:

1. **Spatial state** — the grid tensor.
2. **Per-agent state** — the registry tensor and side structures.
3. **Global mechanical state** — scores, tick count, counters, catastrophe runtime.
4. **Off-grid map overlays** — heal zones and capture-point masks.

A beginner mistake is to think the grid alone is the world. It is not. The grid is only one part of the world state.

### 2.2 The grid tensor

`engine/grid.py` defines the canonical grid shape as `(3, H, W)`.

The three channels are:

| Channel | Meaning | Typical values |
|---|---|---|
| `grid[0]` | Occupancy and tile code | `0.0` empty, `1.0` wall, `2.0` red occupant, `3.0` blue occupant |
| `grid[1]` | HP channel | Current health at occupied cells, `0.0` otherwise |
| `grid[2]` | Occupying slot index | `-1.0` if empty, otherwise registry slot id |

This is a channel-first spatial tensor, conceptually similar to an image with three feature planes.

### 2.3 Why three channels instead of one

A single occupancy map would not be enough.

The engine wants to answer spatial questions quickly:

- Is the target cell empty, a wall, or occupied?
- If occupied, which team is there?
- If occupied, which slot in the registry owns that cell?
- What is that occupant's current HP?

Keeping those answers directly on the grid lets movement, combat, raycasting, and debugging operate without rebuilding spatial state from the registry on every query.

### 2.4 The agent registry

`engine/agent_registry.py` stores dense per-slot agent data in `registry.agent_data`, shaped `(capacity, NUM_COLS)`.

The important columns are:

| Column constant | Meaning |
|---|---|
| `COL_ALIVE` | `1.0` for alive, `0.0` for dead |
| `COL_TEAM` | `2.0` red, `3.0` blue |
| `COL_X`, `COL_Y` | Grid coordinates |
| `COL_HP` | Current HP |
| `COL_UNIT` | Unit class: `1.0` soldier, `2.0` archer |
| `COL_HP_MAX` | Maximum HP |
| `COL_VISION` | Vision range |
| `COL_ATK` | Attack power |
| `COL_AGENT_ID` | Float-backed display/compatibility id |

This tensor is not the only per-agent storage. The registry also owns:

- `agent_uids`: authoritative `int64` unique identifiers.
- `brains`: Python list of per-slot neural modules.
- `generations`: per-slot generation metadata.
- `brain_arch_ids`: compact architecture ids used for bucketed inference.

### 2.5 Spatial truth versus per-agent truth

The engine itself states a critical invariant in `engine/tick.py`:

- The **registry** is the per-agent truth.
- The **grid** is the fast spatial truth used for queries.

These two representations must stay synchronized.

A healthy live agent therefore has a matched pair of facts:

- Registry row says the slot is alive and located at `(x, y)`.
- Grid cell `(y, x)` says that slot occupies the cell.

The engine treats desynchronization as dangerous enough to justify optional invariant checks in `_debug_invariants(...)`.

### 2.6 Off-grid zone state

Special map regions are not stored by overwriting grid channels.

Instead, `engine/mapgen.py` defines a `Zones` object that holds:

- `heal_zones`: explicit base heal-zone patches.
- `cp_masks`: a list of capture-point boolean masks.
- `manual_zone_enabled`: optional manual overrides.
- `catastrophe_state`: a single suppression slot for active catastrophe scheduling.
- `heal_mask` / `effective_heal_mask`: the derived runtime heal truth.

This design is important. Heal zones and capture points are **additive overlays**, not alternate occupancy codes.

The consequence is that a cell can simultaneously be:

- empty or occupied in the grid,
- inside a heal zone,
- inside one or more capture-point masks.

### 2.7 Global mechanical state

The main global state holders are:

- `SimulationStats` in `simulation/stats.py`.
- `TickMetrics` created fresh each tick in `engine/tick.py`.
- `HealZoneCatastropheController` in `engine/catastrophe.py`.
- `RespawnController` in `engine/respawn.py`.

These structures do not define where agents are. They define how the world evolves over time and how that evolution is counted.

### 2.8 Compact ownership map

```text
+--------------------+----------------------------------------------+
| Structure           | Owns                                         |
+--------------------+----------------------------------------------+
| grid                | cell occupancy, wall code, spatial hp, slot |
| registry.agent_data | per-slot life state, team, hp, atk, x, y    |
| registry.agent_uids | persistent identity                          |
| registry.brains     | per-slot controller module                   |
| zones               | heal/capture geometry and runtime heal mask |
| stats               | cumulative team counters and tick           |
| tick metrics        | one-tick aggregates                         |
| catastrophe ctrl    | heal-zone suppression scheduling            |
| respawn ctrl        | team refill and lineage spawning            |
+--------------------+----------------------------------------------+
```

### 2.9 Design consequence

Because state is intentionally split, no safe modification can change only one representation unless that representation is explicitly the sole owner.

A change to position, occupancy, death, healing, or spawn must be checked across:

- registry row,
- grid cell,
- possibly stats,
- sometimes telemetry and PPO side effects.

That is the single most important mental model for this engine.

---

## 3. Entities, teams, slots, and identity

### 3.1 What an “agent” is in this engine

In conceptual terms, an agent is a combat unit.

In implementation terms, an agent is a **currently alive registry slot** whose slot index is also written into `grid[2]` at one spatial location and whose behavior module is stored in `registry.brains[slot]`.

This distinction matters because slots survive death; the individual agent occupying that slot does not.

### 3.2 Teams

Team identity is encoded numerically, not symbolically:

- red team = `2.0`
- blue team = `3.0`

These values are used in both registry and grid occupancy conventions. That shared convention is why action masking and combat can compare `grid[0]` directly to team identity.

### 3.3 Unit classes

The active world defines two unit classes:

- soldier = `1`
- archer = `2`

By default in `config.py`:

- soldier HP = `1.0`
- soldier ATK = `0.15`
- soldier vision = `6`
- archer HP = `0.55`
- archer ATK = `0.10`
- archer vision = `14`
- archer attack range limit = `4`

These are configuration defaults, not mathematical laws of the engine. The runtime mechanics consume the configured values.

### 3.4 Slot identity versus agent identity

This codebase uses **two notions of identity**.

#### Slot id
A slot id is the row index in the registry.

Properties:

- Fixed-size storage location.
- Reused after death.
- Stored on the grid in `grid[2]`.
- Used heavily by runtime code because it is a compact, local index.

#### Agent UID
An agent UID is a monotonically increasing identity stored in `registry.agent_uids`.

Properties:

- Intended to remain unique across births and deaths.
- Better for telemetry and lineage.
- Not reused when a slot is recycled.

### 3.5 Why both identities exist

Slots are efficient for simulation. Persistent UIDs are efficient for analysis.

If only slot ids existed, a long run would blur many different lives into the same identifier. If only UIDs existed, the engine would need a more complex indirection layer for nearly every tensor operation.

The project therefore separates:

- **storage identity**: slot id,
- **historical identity**: UID.

### 3.6 Alive, dead, inactive, respawnable

This engine mostly distinguishes only two slot-life states in the registry tensor:

- alive: `COL_ALIVE > 0.5`
- dead: `COL_ALIVE <= 0.5`

A dead slot is still a valid storage row. It is not removed from the registry tensor. Its grid presence is cleared, but the slot remains available for later respawn.

There is no separate corpse state on the map.

### 3.7 How death affects identity

When an agent dies:

- the slot stays,
- the UID of the dead individual remains in historical logs,
- the slot may later receive a **new** UID when respawn writes a new agent into it.

This means:

- slot continuity is **not** life continuity,
- respawn is not resurrection of the same identity by default,
- lineage must use metadata, not slot reuse alone.

### 3.8 Brains are attached to slots

Every slot may also own a brain module stored outside the tensor in `registry.brains`.

This matters mechanically because decision-making is done per alive slot, and because respawn can either:

- clone and perturb a same-team parent brain, or
- create a fresh brain according to team-aware configuration.

The world is therefore not only a population of bodies. It is a population of bodies **plus** attached controllers.

### 3.9 Generations

The registry also stores a per-slot `generation`.

Observed behavior:

- initial spawn uses `generation=1`,
- clone-based respawn increments from the parent generation,
- fresh respawn sets generation to `0`.

This is not a combat rule, but it is part of the lifecycle model and helps explain how the engine treats “descent” versus “fresh injection”.

### 3.10 Common beginner confusion in this section

The most common misunderstandings are:

1. Thinking `grid[2]` stores persistent identity. It does not; it stores slot id.
2. Thinking a dead agent disappears from all state. It disappears from spatial occupancy, not from registry capacity.
3. Thinking a respawned agent is the same individual because it reused a slot. It is usually not.
4. Thinking the float column `COL_AGENT_ID` is the authoritative id. It is not; the `int64` `agent_uids` tensor is the authoritative store.

### 3.11 Practical mental model

```text
slot 37
  ├─ storage row in registry.agent_data
  ├─ may currently be alive or dead
  ├─ may currently hold a brain module
  ├─ if alive, exactly one grid cell should point to slot 37
  └─ over a long run, may represent many different UIDs across many lives
```

---

## 4. The tick loop: phase by phase

### 4.1 Why the tick loop is the heart of the engine

A “tick” is one discrete world update.

Everything mechanically important happens because `TickEngine.run_tick()` executes one ordered sequence of reads and writes. If the sequence changed, the world behavior would change even if the same individual rules remained in place.

For this codebase, update order is not an implementation detail. It is a gameplay rule.

### 4.2 Top-level sequence

The main runtime order in `engine/tick.py` is:

```text
tick start
  -> refresh zone runtime cache if needed
  -> catastrophe scheduler update
  -> if nobody alive: advance tick and respawn
  -> build observations
  -> build legal action mask
  -> run per-agent inference and sample actions
  -> combat
  -> apply combat deaths
  -> movement
  -> environment phase:
       heal
       metabolism
       metabolism deaths
       capture points
  -> PPO bookkeeping (if enabled)
  -> advance tick counter
  -> flush dead PPO slots
  -> respawn
  -> post-tick telemetry / invariant checks
```

### 4.3 Fast-path when everyone is dead

If `alive_idx` is empty at the start of the tick, the engine does **not** run observation, action, combat, movement, or environment phases.

Instead it:

1. finalizes pending PPO window state if applicable,
2. advances tick by one,
3. flushes dead PPO slots,
4. calls the respawner,
5. resets PPO state for any newly spawned slots,
6. returns.

This means total extinction does not stall the simulation. The system can repopulate from a dead world.

### 4.4 Phase 1: build observations

For alive slots only, the engine computes:

- `pos_xy` from registry positions,
- `obs = _build_transformer_obs(alive_idx, pos_xy)`.

Mechanically, this phase does not change the world. It reads the world.

The observation contains:

- raycast features,
- normalized self features,
- zone flags,
- global counters,
- instinct context built from local neighborhood density.

The exact observation schema belongs to later volumes, but one world-mechanics point matters here: action selection depends on **current** pre-combat, pre-movement world state.

### 4.5 Phase 2: build action mask

`engine/game/move_mask.py` computes a boolean legality mask.

This mask constrains which actions may be sampled.

Mechanically it prevents ordinary policy sampling from choosing:

- movement into non-empty cells,
- attacks that do not target a valid enemy cell,
- out-of-range attacks for the unit class.

This mask is part of the world rule system because later combat and movement depend on the assumption that most actions were drawn from this legal set.

### 4.6 Phase 3: policy inference and action sampling

Alive agents are grouped by architecture bucket via `registry.build_buckets(alive_idx)`. Each bucket runs through `ensemble_forward(...)`, which returns per-agent logits and values. Illegal actions are masked to negative infinity-like logits, and then the engine samples one discrete action per alive slot.

The important mechanical fact is this:

- the world state is still unchanged at this point,
- the action vector is now fixed for this tick,
- later phases consume these sampled actions.

### 4.7 Phase 4: combat

Combat runs before movement.

This is one of the most load-bearing choices in the engine.

Consequences:

- An agent killed in combat does not move later in the same tick.
- Movement conflict resolution uses HP **after** combat damage.
- Cells vacated by combat deaths can become empty targets for the movement phase of surviving agents.

### 4.8 Phase 5: apply combat deaths

After combat damage is written, the engine finds all alive agents whose HP is now `<= 0` and clears them through `_apply_deaths(...)`.

This is not the same as the combat phase itself. Combat changes HP. Death application removes the dead from spatial occupancy and updates counters.

That split is important because:

- several attackers may reduce a victim below zero in aggregate,
- only after aggregated damage is known can death resolution occur cleanly.

### 4.9 Phase 6: movement

Movement runs only for agents that:

- were alive at decision time,
- remain alive after combat,
- selected a directional move action in `1..8`.

The movement phase uses the post-combat world.

That means movement can see cells cleared by combat death, but it does **not** see cells that other movers are about to vacate later in the same movement phase.

### 4.10 Phase 7: environment

The environment phase contains three live mechanics:

1. healing zones,
2. metabolism drain,
3. capture-point accounting.

Order matters inside this phase too:

- healing happens before metabolism,
- metabolism can then kill agents,
- capture-point accounting uses the post-metabolism survivor set.

A cell standing on heal terrain therefore receives heal first and attrition second in the same tick.

### 4.11 Phase 8: PPO bookkeeping

If PPO is enabled, the engine computes per-slot reward components and records the step.

This phase mostly matters to learning, not to world evolution, but it does read mechanical counters produced by earlier phases.

### 4.12 Phase 9: advance tick and respawn

Only after the world step is otherwise complete does the engine:

1. increment `stats.tick`,
2. flush dead PPO slots,
3. respawn into dead slots,
4. reset PPO state for newly alive slots.

Respawned agents therefore do **not** participate in the tick that created them. They appear after the tick is already over and will first act on the next tick.

### 4.13 One-tick control-flow diagram

```text
pre-tick world
   |
   v
observe -> mask -> choose action
   |
   v
combat damage
   |
   v
combat deaths cleared
   |
   v
movement winners move
   |
   v
healing -> metabolism -> metabolism deaths -> capture points
   |
   v
tick counter advances
   |
   v
respawn into dead slots
   |
   v
post-tick world
```

### 4.14 Why this order matters so much

Changing phase order would change the game even if no arithmetic changed.

Examples:

- If movement came before combat, agents could escape lethal range on the same tick.
- If metabolism came before healing, heal-zone value would be lower.
- If respawn came before capture-point scoring, newborn agents could influence objectives immediately.
- If combat deaths were applied after movement, dead agents could still move.

In this engine, phase order is a first-class mechanic.

---

## 5. Movement mechanics

### 5.1 Plain-language intuition

Movement is simple in surface form: choose one of eight directions and attempt to step one cell.

What makes it interesting is not the step itself. What matters is the conflict rule:

- only empty destinations may be entered,
- if several agents want the same empty destination, highest current HP wins,
- if the highest HP is tied, nobody moves.

### 5.2 Action encoding

Movement actions are:

- `0` = idle,
- `1..8` = move one step in one of the eight directions in `DIRS8`.

`DIRS8` uses the standard order:

1. north,
2. northeast,
3. east,
4. southeast,
5. south,
6. southwest,
7. west,
8. northwest.

### 5.3 Movement legality before sampling

The legality mask in `build_mask(...)` allows a move only if the destination cell is:

- in bounds,
- empty in `grid[0]`.

Because the world has border walls, many outward moves near the edge naturally become blocked by wall occupancy rather than by a special out-of-bounds state.

### 5.4 Formal movement rule

For an alive post-combat mover at position `(x, y)` with chosen move direction `d`:

1. Compute tentative `(nx, ny) = (x, y) + DIRS8[d]`.
2. Clamp to map bounds in runtime code.
3. Read destination occupancy from `grid[0, ny, nx]`.
4. If destination is not empty, movement is blocked.
5. If destination is empty and there is only one claimant, the move succeeds.
6. If destination is empty and there are multiple claimants:
   - compare current HP values,
   - if one claimant has unique highest HP, that claimant moves,
   - if the highest HP is shared, none of the tied claimants move.

### 5.5 Post-combat life gating

Movement is filtered by `alive_after = (data[alive_idx, COL_ALIVE] > 0.5)`.

Therefore an agent that selected a move but died in combat is removed from the movement phase entirely.

This is the concrete implementation of the combat-first promise.

### 5.6 Destination emptiness rule

Only cells with `grid[0] == 0.0` are movement-eligible.

Consequences:

- walls block movement,
- enemies block movement,
- allies block movement,
- a cell that another agent is about to leave this tick still counts as occupied during eligibility checking.

That last consequence means the engine does **not** support simultaneous swaps or chain movement through same-tick vacancies.

### 5.7 Conflict resolution is HP-based

When several can-move agents target the same empty cell, the engine constructs a destination key `ny * W + nx` and groups claimants by destination.

The winner is the unique claimant with maximal current HP at that destination.

Important details:

- “Current HP” here means HP **after combat damage** and **before metabolism**.
- A previously wounded agent may lose a movement contest it would have won one tick earlier.
- The rule is deterministic for unequal HP.
- Equal max HP produces a tie and no move.

### 5.8 How the move is written

For movement winners, the engine performs three operations in order:

1. Clear the old grid cell:
   - `grid[0] = 0`
   - `grid[1] = 0`
   - `grid[2] = -1`
2. Update registry coordinates `COL_X`, `COL_Y`.
3. Write the new grid cell:
   - `grid[0] = team code`
   - `grid[1] = current HP`
   - `grid[2] = slot id`

This is the same dual-state synchronization pattern seen everywhere else.

### 5.9 What movement does not do

Movement does **not**:

- inflict damage,
- push occupants,
- swap two occupied cells,
- combine with attack in the same action,
- use momentum or speed,
- consume stamina,
- pass through walls,
- pass through allies or enemies,
- resolve by initiative or randomness.

### 5.10 Edge-case consequences

#### Border behavior
Because runtime coordinates are clamped and border cells are walls, an outward move near the edge usually becomes “blocked by wall”.

#### Vacated combat cells
A cell cleared by combat death earlier in the tick may now count as empty and can be entered by movement.

#### Vacated mover cells
A cell occupied at movement-eligibility time stays blocked even if its current occupant later moves away, because the engine does not recompute eligibility mid-phase.

#### HP ties
If two or more highest-HP claimants tie, nobody moves. There is no tiebreak by slot id, team, or randomness.

### 5.11 Metrics emitted by movement

Movement phase aggregates include:

- `move_attempted`
- `move_can_move`
- `moved`
- `move_blocked_wall`
- `move_blocked_occupied`
- `move_conflict_lost`
- `move_conflict_tie`

A subtle point: `move_attempted` counts only post-combat alive movers, not every agent that originally chose a move action.

### 5.12 Design consequence

The movement system rewards survival and current durability, not only spatial intent. In effect, combat and movement are coupled because combat wounds directly affect conflict outcomes later in the same tick.

---

## 6. Combat, damage, and elimination

### 6.1 Plain-language intuition

An attack action means: choose a direction and a discrete range, then attempt to damage whatever enemy occupies that target cell.

This is not projectile simulation. The attack resolves in the same tick as a direct state update.

### 6.2 Attack action layout

For the `41`-action layout used by default:

- `0` = idle
- `1..8` = movement
- `9..40` = attacks

Attack actions encode:

- one of eight directions,
- one of four range values.

The runtime decoding is:

- `r = ((action - 9) % 4) + 1`
- `dir_idx = (action - 9) // 4`

So each direction owns a block of four range-specific actions.

### 6.3 Unit-specific attack reach

The legality mask encodes unit-class reach:

- soldiers may attack only at range `1`,
- archers may attack at ranges `1..ARCHER_RANGE`, clipped to `4`.

This unit gating is implemented in `engine/game/move_mask.py`.

A critical design detail follows from that statement: **normal gameplay relies on action masking to prevent illegal reach**. The combat phase itself does not re-derive full unit legality beyond optional debug checks. If a caller forces illegal actions and debug validation is off, behavior can diverge from intended unit constraints.

### 6.4 Target coordinate calculation

For each attacking agent:

1. read attacker position `(ax, ay)`,
2. decode direction and range,
3. compute `dxy = DIRS8[dir_idx] * r`,
4. compute tentative target `(tx, ty) = (ax, ay) + dxy`,
5. clamp target coordinates to map bounds.

In ordinary play the legality mask already enforces in-bounds enemy targeting. Clamping mainly protects forced or invalid action paths.

### 6.5 Optional archer line-of-sight wall blocking

If `ARCHER_LOS_BLOCKS_WALLS` is enabled, the combat phase performs an additional check for archers:

- inspect intermediate cells from step `1` through `r-1`,
- if any intermediate cell is out of bounds or a wall, the shot is blocked,
- blocked shots are removed before damage application.

This matters only for archers because melee range `1` has no intermediate cells.

### 6.6 Enemy filtering

After target coordinates are computed, combat reads `grid[2][ty, tx]` to find victim slot ids.

The attack is only retained if:

- the target cell contains an agent (`slot >= 0`),
- attacker team and victim team differ.

Therefore:

- attacks against empty cells do nothing,
- attacks against same-team occupants are filtered out.

### 6.7 Damage model

Damage dealt by an attacker is simply `data[attacker_slot, COL_ATK]`.

There is no armor, dodge, defense stat, or damage falloff in the current combat path.

### 6.8 Multi-attacker focus fire

The engine explicitly handles duplicate-victim cases.

If multiple attackers hit the same victim in one tick, the engine does **not** subtract damage one write at a time in arbitrary order. Instead it:

1. sorts hits by victim slot,
2. groups consecutive equal victims,
3. sums damage per unique victim,
4. subtracts total damage once per victim.

This is a major correctness property. It prevents duplicate-index race behavior and makes same-tick focus fire deterministic.

### 6.9 When a kill is recognized

After grouped damage is applied, the engine checks:

- `hp_before > 0`
- `hp_after <= 0`

Victims meeting both conditions are treated as newly killed by combat in this tick.

Death is not spatially applied yet at that exact line. At this point the engine has only determined that lethal damage happened.

### 6.10 Kill credit is not shared among all contributors

The engine assigns **exactly one credited killer per killed victim** for per-kill reward/credit purposes.

The selection rule is deterministic:

1. choose the attacker who dealt the highest same-tick damage to that victim,
2. if several attackers tie on damage, choose the smallest attacker slot id.

This choice is stored in `combat_killer_slot_by_victim` and later passed into `_apply_deaths(...)`.

This is important because the engine distinguishes between:

- **damage contribution**, which can be shared,
- **kill credit**, which is singular.

### 6.11 Damage and kill telemetry order

The combat phase intentionally emits damage telemetry before kill-credit telemetry, and death telemetry is emitted later during `_apply_deaths(...)`.

The causal order is therefore:

```text
damage aggregation
   -> kill credit
      -> death application
```

That ordering matters for replay and analysis.

### 6.12 Combat deaths are applied after damage, before movement

Once combat damage is done, the engine applies deaths by selecting all still-alive slots with `HP <= 0` and calling:

`_apply_deaths(..., credit_kills=True, death_cause="combat", killer_slot_by_victim=...)`

Consequences:

- combat deaths increment kills and deaths,
- dead cells are cleared before movement,
- surviving movers can step into newly empty cells.

### 6.13 What `_apply_deaths(...)` actually does

For dead slots, `_apply_deaths(...)`:

1. snapshots metadata needed for logs,
2. updates `SimulationStats` kill and death counters,
3. clears spatial occupancy:
   - `grid[0] = 0`
   - `grid[1] = 0`
   - `grid[2] = -1`
4. sets `COL_ALIVE = 0` in the registry,
5. emits death telemetry,
6. increments per-tick death metrics.

Notably, it does **not** delete the registry row. The slot remains available for later respawn.

### 6.14 What combat does not use

Combat in the current engine does **not** use:

- defender armor,
- attack cooldown state,
- probabilistic hit chance,
- separate projectile entities,
- vision range as a combat gate,
- initiative ordering within the same phase.

Vision is used for observation construction, not for combat target legality in `run_tick()`.

### 6.15 Subtle but important consequence

Because action masking uses current grid state and combat resolves before movement, the attacker always aims at the pre-movement location of the target. There is no dodge-through-motion in the same tick.

---

## 7. Health, death, respawn, and agent lifecycle

### 7.1 Lifecycle overview

A slot can be viewed as moving through this sequence:

```text
dead slot
   -> respawn writes new agent
      -> alive slot acts for many ticks
         -> HP falls to zero or below
            -> death clears spatial presence
               -> slot returns to dead pool
```

### 7.2 Health storage

Health exists in two places:

- authoritative per-agent HP in `registry.agent_data[:, COL_HP]`,
- mirrored per-cell HP in `grid[1]`.

The engine treats the registry value as the primary value and explicitly syncs the grid HP channel after important mutations.

### 7.3 Why `grid[1]` exists at all

`grid[1]` is not redundant decoration. It supports:

- fast spatial reads,
- debug visibility,
- perception/raycast utilities,
- consistent world snapshots without reconstructing HP from slot lookups.

### 7.4 Immediate causes of death in the current tick path

In the current runtime path, a live slot becomes dead when HP drops to `<= 0` due to:

1. combat damage,
2. metabolism drain.

The helper `_apply_deaths(...)` can label deaths as `"environmental"` or `"collision"`, but those causes are not currently exercised by `run_tick()`.

### 7.5 No corpse persistence

When death is applied:

- occupancy is cleared,
- HP channel is cleared,
- slot-id channel is cleared.

The world does not keep a corpse object, corpse timer, or corpse tile state.

A dead agent therefore disappears from the playable map immediately at death application time.

### 7.6 Respawn is slot recycling, not map-level object creation

Respawn does not expand capacity and does not allocate a new row. It looks for dead slots and reuses them.

That is why understanding slot identity versus agent identity is essential.

### 7.7 Respawn controller responsibilities

`RespawnController.step(...)` manages two refill mechanisms:

1. **floor-based respawn**
   - if a team’s alive count falls below `floor_per_team`, spawn toward the floor, subject to cap and cooldown;
2. **periodic respawn budget**
   - every `period_ticks`, allocate a total budget across teams inversely proportional to their current alive counts.

By default in `config.py`, the relevant values are:

- floor per team: `100`
- max per tick: `5`
- periodic window: `20_000` ticks
- periodic budget: `40`
- cooldown hysteresis: `100` ticks

### 7.8 Team balancing logic

Periodic respawn uses an inverse split function.

If one team has fewer alive agents, it receives a larger share of the periodic budget.

That is a balancing mechanism, not a spatial mechanic, but it strongly shapes long-run population dynamics.

### 7.9 Choosing dead slots

Respawn scans for dead slots:

`dead_slots = (~alive).nonzero(...)`

It then iterates through dead slots in order and writes new agents into them until the requested count or available dead-slot count is exhausted.

### 7.10 Choosing spawn locations

Respawn uses `_pick_location(...)`, which supports:

- uniform placement,
- near-parent placement for clone paths if configured.

A cell is spawnable only if:

- it lies inside the configured wall margin,
- `grid[0, y, x] == 0.0`,
- `grid[2, y, x] == -1.0`.

So respawn respects both occupancy and slot-id emptiness.

### 7.11 Choosing whether the child is cloned or fresh

For each respawn attempt:

- if same-team parents exist and a random draw is below `clone_prob`, the controller clones a parent brain,
- otherwise it creates a fresh brain.

Default `clone_prob` in `RespawnCfg` is `0.50`.

### 7.12 Parent selection

When clone mode is chosen, parent selection can be:

- random/uniform,
- top-k weighted by local scalar score `hp + atk`.

The default mode is `topk_weighted` according to `RespawnCfg` field default factories, though the helper supports simpler modes as well.

### 7.13 Child unit class

Respawn separates “brain inheritance” from “unit inheritance”.

By default, `child_unit_mode` is `inherit_parent_on_clone`, which means:

- clone births inherit the parent unit class,
- fresh births sample the unit class from the configured archer ratio.

### 7.14 Mutation layers

The respawn path supports several mutation mechanisms:

- standard Gaussian brain perturbation on clone path,
- rare mutation pathways that can alter physical stats,
- optional extra brain noise for inherited anomaly births.

These mutation layers affect the future population, but the world engine still treats the resulting child as an ordinary alive agent once registration is complete.

### 7.15 What gets reset on respawn

Respawn writes a new agent into the slot, including:

- alive flag,
- team,
- position,
- HP,
- HP max,
- vision,
- attack,
- unit class,
- new UID,
- new or cloned brain,
- generation metadata.

If optimizers exist, they are cleared for that slot.

If PPO runtime is active, `TickEngine` also resets PPO state for slots that were dead before respawn and alive after respawn.

### 7.16 What persists through death

At the world level, very little persists through death of an individual.

The main things that persist are:

- the slot container itself,
- global team counters,
- telemetry history,
- agent lineage metadata if logged externally,
- possibly architecture metadata attached once a new brain is written.

The dead individual’s spatial presence does not persist.

### 7.17 Important lifecycle consequence

Because respawn happens after tick advancement and after all action phases, newborn agents are invisible to the just-finished tick’s mechanics and visible to the next tick’s mechanics.

This makes respawn a clean tick boundary event rather than a mid-tick intrusion.

### 7.18 A verified wiring discrepancy worth noting

`config.py` defines `RESPAWN_ENABLED`, but `TickEngine` constructs its controller as `RespawnController(RespawnCfg())`, and `RespawnCfg.enabled` is hardcoded to `True` rather than derived from `config.RESPAWN_ENABLED`.

As written, the persistent respawn controller used by the engine does not appear to read that config flag directly. That means “respawn disabled” is not a documented safe assumption unless the construction path is changed or a different controller configuration is injected.

This is not a guess about intention. It is a code-wiring observation and should be treated as a modification hazard.

---

## 8. Environmental rules, events, and special mechanics

### 8.1 Plain-language overview

After movement, the engine applies non-combat world rules:

1. heal zones restore HP,
2. metabolism drains HP,
3. capture points award objective progress,
4. catastrophe scheduling can suppress heal zones across ticks.

These are the ambient world mechanics.

### 8.2 Heal zones

Heal zones are generated in `engine/mapgen.make_zones(...)` as rectangular boolean masks.

The `Zones` object stores explicit per-zone geometry, then derives one effective runtime heal mask by combining:

- base zone geometry,
- catastrophe suppression,
- manual override state.

`TickEngine` uses the effective heal mask cached on the simulation device.

### 8.3 Heal-zone application rule

During the environment phase, if a live agent stands on a heal cell:

1. read `hp_before`,
2. add `config.HEAL_RATE`,
3. clamp to `COL_HP_MAX`,
4. write new HP back to registry,
5. sync `grid[1]`.

By default, `HEAL_RATE = 0.003`.

### 8.4 Overlapping heal zones do not stack

This is an important design consequence of using a single derived boolean `heal_mask`.

If several base heal patches overlap, the effective heal truth is still a boolean union. A cell is either heal-active or not. The runtime does not apply one heal increment per overlapping heal zone.

### 8.5 Metabolism

If `METABOLISM_ENABLED` is on, the engine subtracts a class-specific drain every tick from each alive agent:

- soldier drain default: `0.0015`
- archer drain default: `0.0010`

The engine then syncs the HP grid channel and applies metabolism deaths for any survivors whose HP is now `<= 0`.

### 8.6 Order between heal and metabolism

The order is:

1. heal,
2. metabolism.

Therefore the net HP change for a heal-zone occupant in one tick is:

```text
net change = +HEAL_RATE - metabolism_drain(unit)
```

With default values, both soldiers and archers gain net HP while standing on an active heal tile, though the gain differs by class.

### 8.7 Capture points

Capture-point masks are also generated as rectangular patches in `make_zones(...)` and stored in `zones.cp_masks`.

During the environment phase, the engine iterates over each capture-point mask separately and counts how many living red and blue agents stand inside that mask.

If one team outnumbers the other on that mask, that team gains `CP_REWARD_PER_TICK` capture points and the corresponding per-tick metric.

### 8.8 Capture-point scoring defaults

`config.py` sets:

- `CP_COUNT = 7`
- `CP_SIZE_RATIO = 0.08`
- `CP_REWARD_PER_TICK = 0.0`

So the mechanic exists by default, but its contribution to team score is zero unless reconfigured.

This means capture points are structurally part of the world even when they are not currently score-driving.

### 8.9 Contested capture-point reward path

If both teams are present on a capture point and one side has strictly greater count, the winning side’s agents on that point can receive an individual contested reward used by PPO bookkeeping.

That is not a world-state mutation beyond reward accounting, but it is mechanically tied to the same occupancy counts.

### 8.10 Capture-point overlap can stack

Unlike heal zones, capture points are processed as a **list of masks**, not as one boolean union.

Therefore, if two capture-point masks overlap spatially, a unit standing in the overlap can be counted in both loops in the same tick. Because `make_zones(...)` samples rectangles independently and does not enforce separation, such overlap is possible.

This means capture-point effects can stack across overlapping masks. That is a real consequence of the implementation.

### 8.11 Catastrophe system

The catastrophe system in `engine/catastrophe.py` does not directly damage units or alter combat. It temporarily suppresses selected heal zones by writing catastrophe suppression state into `Zones`.

The controller can operate in:

- periodic mode,
- dynamic mode.

It can trigger patterns such as:

- random small suppression,
- random medium suppression,
- left-side bias off,
- right-side bias off,
- one-cluster-survives.

### 8.12 What catastrophe actually changes

Mechanically, catastrophe changes only this:

- which heal zones contribute to the effective runtime heal mask.

It does **not** change:

- walls,
- occupancy,
- attack rules,
- movement rules,
- capture-point masks,
- PPO buffer structure.

### 8.13 When catastrophe updates happen

`TickEngine.run_tick()` updates catastrophe scheduling at the **start** of the tick, before observations are built and before the environment phase.

If catastrophe state changes, the engine refreshes cached zone tensors immediately.

The practical effect is that same-tick healing availability reflects the newly updated catastrophe state.

### 8.14 Manual zone overrides

The `Zones` and catastrophe controller code support manual zone enabling, disabling, and restoration. Those paths are important for viewer or operator control.

The precedence rule in `Zones.rebuild_effective_heal_mask()` is:

1. base zone active by default,
2. catastrophe suppression may disable it,
3. manual override wins last.

### 8.15 Random walls

Map walls are not only the border. `engine/mapgen.add_random_walls(...)` can add meandering internal wall traces.

For each wall cell written:

- `grid[0]` becomes wall code `1.0`,
- `grid[1]` becomes `0.0`,
- `grid[2]` becomes `-1.0`.

So a wall cell is guaranteed not to contain residual HP or slot-id state.

### 8.16 What is absent from the environment phase

The environment phase currently does **not** include:

- zone-based damage,
- poison,
- slowing tiles,
- resource production,
- ownership persistence on zones,
- weather,
- collision death.

If such mechanics are added later, they should be treated as genuinely new rules, not as already-existing hidden behavior.

---

## 9. Mechanical counters and runtime signals

### 9.1 Why counters matter

Counters are not cosmetic in this engine. They are the compact summary of what the world just did.

Some counters are cumulative across the whole run. Others are per-tick metrics.

### 9.2 `SimulationStats` cumulative team counters

`simulation/stats.py` maintains, for each team:

- `score`
- `kills`
- `deaths`
- `dmg_dealt`
- `dmg_taken`
- `cp_points`

It also stores the global simulation tick and a structured death log.

### 9.3 What directly changes score

The score object is updated through explicit methods:

- `add_kill(...)`
- `add_death(...)`
- `add_damage_dealt(...)`
- `add_damage_taken(...)`
- `add_capture_points(...)`

In the current tick path, combat deaths and capture-point awards visibly call into the stats manager. Damage-based score updates exist as stats APIs but are not the central world-state write in the current combat path shown here.

### 9.4 Per-tick metrics

`TickMetrics` stores one-tick aggregates such as:

- alive count,
- moved count,
- attacks,
- deaths,
- death cause breakdown,
- capture-point totals for the tick,
- movement outcome breakdown.

These are returned as a plain dictionary at the end of `run_tick()`.

### 9.5 What `attacks` counts

A subtle but important detail: `metrics.attacks` is incremented after attack filtering and hit aggregation logic, not immediately when an attack action is chosen.

So this metric reflects realized attack executions that survived legality-style filtering, not raw “I selected an attack symbol” intents.

### 9.6 Death cause breakdown

`TickMetrics` supports cause-specific counters:

- `deaths_combat`
- `deaths_metabolism`
- `deaths_environmental`
- `deaths_collision`
- `deaths_unknown`

In the current runtime path, combat and metabolism are the active causes.

### 9.7 Capture-point tick metrics

`metrics.cp_red_tick` and `metrics.cp_blue_tick` measure the amount of capture reward credited during the tick.

Because the default `CP_REWARD_PER_TICK` is `0.0`, these metrics may remain zero unless the configuration is changed.

### 9.8 Runtime catastrophe signal

Before catastrophe scheduling, the engine builds a small runtime signal bundle containing:

- current tick,
- alive count,
- count of alive agents standing on heal cells.

The controller can use this signal in dynamic scheduling mode to decide whether catastrophe conditions are sustained.

This signal does not directly change the world. It is scheduler input.

### 9.9 Telemetry hooks versus mechanics

The code contains many telemetry calls. Those should not be confused with mechanics.

A safe rule is:

- if a value changes registry, grid, zones, or stats, it is part of world progression;
- if it only records what already happened, it is instrumentation.

The engine tries to keep telemetry additive and failure-tolerant so that instrumentation does not destabilize simulation.

---

## 10. Engine invariants, ordering assumptions, and edge cases

### 10.1 The deepest invariant: registry and grid must agree

The engine’s optional `_debug_invariants(...)` checks enforce several critical truths:

- alive positions must be in bounds,
- `grid[2]` at alive positions must equal the alive slot ids,
- `grid[0]` at alive positions must equal team codes,
- no duplicate slot ids may appear on the grid,
- the set of alive slots must match the set of slot ids present on the grid,
- there must be no “ghost cells” where `grid[2] >= 0` but `grid[0] == 0`.

Any mechanic edit that touches position, death, spawn, or occupancy must preserve these truths.

### 10.2 Ordering assumption: combat before movement

Many later rules depend on combat-first semantics.

If you change this order, you must re-audit:

- movement eligibility,
- HP-based move conflict resolution,
- capture-point occupancy timing,
- reward accounting,
- telemetry ordering,
- viewer expectations.

### 10.3 Ordering assumption: heal before metabolism

Heal-zone value currently includes “you get healed before you are drained”.

Reversing this order changes survival thresholds and zone value immediately.

### 10.4 Ordering assumption: respawn after tick advance

Respawned agents currently act only on the next tick.

Changing that would require rethinking:

- observation timing,
- capture-point counts,
- PPO step ownership,
- replay semantics.

### 10.5 Slot-based world assumptions

Large parts of the engine assume that `grid[2]` stores slot ids, not persistent UIDs.

Affected systems include:

- death application,
- build-unit-map,
- raycast feature generation,
- combat victim lookup,
- movement occupancy repair,
- telemetry conversion helpers.

Replacing slot ids with UIDs on the grid would require a full indirection layer and should not be attempted casually.

### 10.6 The world relies on legal action masking more than full runtime revalidation

The engine mask builder is not just an optimization. It is part of the correctness story.

Examples:

- movement legality is largely determined before sampling,
- unit-specific range legality is encoded in the mask,
- combat contains some debug-only assertions rather than universal runtime guards.

Therefore any code path that bypasses or forces actions must be treated as operating outside normal world assumptions.

### 10.7 Verified LOS consistency risk

`engine/game/move_mask.py` defines `_los_blocked_by_walls_grid0(...)` to expect the occupancy plane `grid[0]`, but the LOS-enabled attack-mask path calls it with `occ`, which is only the immediate neighbor occupancy tensor from the move computation, not the full `(H, W)` wall map.

The combat phase in `engine/tick.py` performs its own LOS check against the real grid, which protects final damage application better than it protects policy-mask consistency.

The practical risk is not necessarily wrong damage. The more immediate risk is a mismatch between:

- what the action mask says is legal,
- what the combat phase actually allows to deal damage.

This should be treated as a real fragility in LOS-enabled runs.

### 10.8 Verified respawn-config wiring risk

As noted earlier, `RESPAWN_ENABLED` is defined in configuration but does not appear to be wired into the `RespawnCfg.enabled` used by the persistent `TickEngine` respawner construction path.

A maintainer should not assume the config flag alone disables respawn without verifying the actual controller configuration path.

### 10.9 Overlapping capture points can multiply objective effects

Because capture points are processed mask-by-mask, overlap can stack control effects. This is not inherently wrong, but it is a property that many readers would not expect unless it is stated explicitly.

### 10.10 Overlapping heal zones do not multiply healing

Because heal zones collapse into one effective boolean mask, overlap does not stack. This asymmetry with capture points is intentional in implementation effect, even if not explicitly named as a gameplay design.

### 10.11 No same-tick swap movement

Two adjacent agents cannot exchange cells in one movement phase because each sees the other’s current cell as occupied during eligibility testing.

A future maintainer adding swap-like movement must redesign movement semantics rather than patch one line.

### 10.12 Cells vacated by combat are visible to movement

Because death cleanup runs before movement, a surviving mover can enter a cell that was occupied at tick start but cleared by combat death earlier in the same tick.

This is a genuine gameplay consequence of the chosen order.

### 10.13 HP grid sync windows are intentionally minimized, not eliminated by magic

The engine includes `_sync_grid_hp_for_slots(...)` and explicit post-heal/post-metabolism HP writes to reduce windows where registry HP and grid HP disagree.

That is good defensive engineering, but it also reveals an assumption: HP can temporarily desynchronize if a maintainer adds new HP-mutating code and forgets the sync path.

### 10.14 Brainless alive slots are treated as inconsistent

`registry.build_buckets(...)` contains a repair path that can mark a supposedly alive slot dead if its brain is `None` or metadata is invalid.

That is not a primary combat mechanic, but it is an important runtime consistency assumption: an alive agent is expected to have a usable brain/controller.

---

## 11. How to modify the mechanics without breaking the engine

### 11.1 First principle: change the rule, then audit the ownership graph

Before editing any mechanic, answer two questions:

1. Which state is the rule **supposed** to change?
2. Which structures currently mirror or derive that state?

For this engine, the answer is often “more than one structure”.

### 11.2 If you change occupancy or position

You must audit at least:

- `registry.agent_data` position columns,
- `grid[0]` occupancy,
- `grid[1]` HP location,
- `grid[2]` slot id location,
- `_debug_invariants(...)`,
- raycast consumers,
- action mask logic.

A position change that only updates the registry is not safe.

### 11.3 If you change combat

You must audit at least:

- action encoding assumptions in `build_mask(...)`,
- target decoding in `run_tick()`,
- unit reach rules,
- LOS behavior,
- grouped damage semantics,
- kill-credit assignment,
- death application ordering,
- reward/telemetry coupling.

Combat is not one function. It spans mask construction, action decoding, state mutation, and death resolution.

### 11.4 If you change movement

You must audit at least:

- legality mask,
- conflict resolution,
- old-cell clearing,
- new-cell writes,
- registry coordinate updates,
- movement telemetry,
- invariant checks,
- downstream occupancy-dependent observation code.

Movement is especially easy to break because the grid is both gameplay state and sensor input.

### 11.5 If you change healing or metabolism

You must audit at least:

- registry HP writes,
- `grid[1]` sync,
- death thresholds,
- zone mask caching,
- catastrophe interaction,
- capture-point post-heal survivor assumptions,
- PPO reward side effects if enabled.

### 11.6 If you change respawn

You must audit at least:

- dead-slot selection,
- spawn location validity,
- registry writes,
- grid writes,
- UID generation,
- generation metadata,
- brain replacement,
- PPO reset-on-respawn behavior,
- checkpoint compatibility if runtime state is persisted elsewhere.

### 11.7 If you change any config-backed contract

Some config knobs are ordinary tuning parameters. Others are schema or mechanic contracts.

Treat the following as high-risk:

- `OBS_DIM`
- `NUM_ACTIONS`
- unit id conventions
- team id conventions
- grid channel semantics
- slot-id-in-grid convention

Changing them is a cross-module migration, not a local edit.

### 11.8 Use invariants aggressively after mechanic edits

`FWS_DEBUG_INVARIANTS` exists for a reason. After editing movement, combat, spawn, death, or respawn, run with invariants enabled and verify:

- no ghost cells,
- no duplicate slot ids,
- alive slot set equals grid-present slot set.

If those fail, the world state is already inconsistent even if the viewer still looks plausible.

### 11.9 Beware of changes that are mechanically small but semantically large

Examples:

- changing death timing by only a few lines,
- moving respawn earlier in the tick,
- letting movement look at a post-move occupancy view,
- changing capture masks from list semantics to union semantics,
- replacing slot ids with UIDs on the grid.

Each of those is a design change disguised as a refactor.

### 11.10 Recommended safe workflow for mechanic edits

1. State the intended rule change in plain language.
2. Identify which phase should own the change.
3. Identify every state structure touched.
4. Implement the write path.
5. Implement the mirror/sync path.
6. Re-run invariant checks.
7. Re-check action legality assumptions.
8. Re-check spawn and death paths.
9. Re-check telemetry only after mechanics are correct.

---

## 12. What this volume establishes for later volumes

This volume has established the answer to:

**What exists in the world, what state owns it, and what order transforms it?**

That foundation supports later volumes, which should go deeper into topics intentionally only touched lightly here:

- observation encoding and ray features,
- action semantics as model interface,
- per-agent brain architectures,
- PPO runtime and reward accounting,
- viewer, controls, and operator tooling,
- checkpointing and resume state,
- telemetry and analytics.

This volume stops at the mechanics boundary. It explains the world engine, not every learning system built around it.

---

## Appendix A. Tick lifecycle at a glance

```text
1. Recompute alive slots
2. Refresh runtime heal-mask cache if zone revision changed
3. Update catastrophe scheduler
4. If nobody alive:
     a. advance tick
     b. flush dead PPO slots
     c. respawn
     d. return
5. Build observations for alive slots
6. Build action mask
7. Run bucketed inference and sample actions
8. Resolve combat damage
9. Apply combat deaths
10. Resolve movement for surviving movers
11. Apply healing
12. Apply metabolism
13. Apply metabolism deaths
14. Award capture-point progress
15. Record PPO step if enabled
16. Advance global tick
17. Flush dead PPO slots
18. Respawn into dead slots
19. Reset PPO state for respawned slots
20. Run end-of-tick telemetry/invariant hooks
```

---

## Appendix B. Verified mechanics responsibility map

| Mechanic | Primary file / function | State touched | Why it matters |
|---|---|---|---|
| Grid creation | `engine/grid.py` → `make_grid` | `grid` | Defines channel semantics and border walls |
| Internal walls | `engine/mapgen.py` → `add_random_walls` | `grid[0:3]` | Creates static obstacles and clears conflicting spatial state |
| Heal/capture geometry | `engine/mapgen.py` → `make_zones` | `Zones` | Defines off-grid environmental overlays |
| Agent storage | `engine/agent_registry.py` → `AgentsRegistry` | `agent_data`, `agent_uids`, `brains` | Defines per-slot truth |
| Initial spawn | `engine/spawn.py` → `spawn_*` | registry + grid | Populates the initial world |
| Tick control | `engine/tick.py` → `run_tick` | registry, grid, stats, zones | Owns world progression order |
| Observation read path | `engine/tick.py` → `_build_transformer_obs` | reads grid, registry, zones, stats | Converts world state into agent inputs |
| Legal action mask | `engine/game/move_mask.py` → `build_mask` | reads grid and team/unit info | Encodes normal action legality |
| Combat damage | `engine/tick.py` → combat block in `run_tick` | registry HP, grid HP | Applies grouped focus-fire damage |
| Death cleanup | `engine/tick.py` → `_apply_deaths` | registry alive flag, grid occupancy, stats | Removes dead agents and updates counters |
| Movement | `engine/tick.py` → movement block in `run_tick` | registry position, grid occupancy | Resolves directional motion and conflicts |
| Healing | `engine/tick.py` → environment block | registry HP, grid HP | Restores health on active heal cells |
| Metabolism | `engine/tick.py` → environment block | registry HP, grid HP | Continuous attrition and non-combat deaths |
| Capture points | `engine/tick.py` → environment block | stats, tick metrics | Objective control accounting |
| Catastrophe scheduling | `engine/catastrophe.py` → `HealZoneCatastropheController` | `Zones` catastrophe slot, effective heal mask | Temporarily suppresses healing regions |
| Respawn | `engine/respawn.py` → `RespawnController.step` | dead slots, registry, grid | Replenishes teams and recycles slots |
| Team counters | `simulation/stats.py` → `SimulationStats` | cumulative counters and tick | Long-run mechanical accounting |

---

## Appendix C. Compact mental model of one tick

Think of the engine as a four-layer machine:

1. **Read the world**
   Alive agents sense the current grid, current zones, and current counters.

2. **Choose intents**
   Each alive agent samples exactly one legal action from its masked policy distribution.

3. **Apply rules in order**
   Combat changes HP. Death clears the dead. Movement changes positions. Environment changes HP and objective counters.

4. **Repair population at the boundary**
   The tick counter advances, dead slots are flushed, and respawn writes new agents into available storage for the next tick.

A shorter version is:

```text
sense -> choose -> fight -> remove dead -> move -> environment -> advance time -> respawn
```

The deeper version is this:

- the grid is spatial truth,
- the registry is per-agent truth,
- slot ids are runtime storage identity,
- UIDs are historical identity,
- order is itself a mechanic,
- and every safe edit must preserve the grid/registry contract.
