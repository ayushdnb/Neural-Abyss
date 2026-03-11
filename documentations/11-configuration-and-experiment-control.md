# Configuration and Experiment Control

This document explains how the inspected repository is configured and which control surfaces affect runtime behavior.

## Purpose

The repository is heavily config-driven. Many semantics are stable in code, but practical behavior depends on environment-variable overrides. This document organizes those knobs by function rather than by the order in `config.py`.

## Configuration source

The inspected runtime uses a single central config module: `config.py`.

Its resolution order is:

1. hard-coded defaults
2. environment-variable parsing
3. profile overrides for values not explicitly set in the environment
4. config invariant validation

This means explicit environment variables take precedence over profile presets.

## Profiles

The inspected config supports these named profiles:

- `default`
- `debug`
- `train_fast`
- `train_quality`

Profiles are applied after defaults but do not override explicitly set environment variables.

## Static config versus runtime controls

A useful distinction is:

- **static config**: values resolved at startup from `config.py`
- **runtime controls**: actions taken after startup through the viewer or trigger files

### Static config examples

- grid size
- start population
- PPO hyperparameters
- catastrophe scheduler parameters
- telemetry cadence
- UI enable flag

### Runtime control examples

- pause / single-step
- speed multiplier
- manual checkpoint hotkey
- catastrophe master toggle
- catastrophe scheduler toggle
- manual catastrophe activation
- base-zone editing
- trigger-file checkpoint save

## Major config families

### 1. Run identity and resume

Important variables include:

- `FWS_PROFILE`
- `FWS_EXPERIMENT_TAG`
- `FWS_SEED`
- `FWS_RESULTS_DIR`
- `FWS_CHECKPOINT_PATH`
- `FWS_RESUME_OUTPUT_CONTINUITY`
- `FWS_RESUME_FORCE_NEW_RUN`
- `FWS_RESUME_APPEND_STRICT_CSV_SCHEMA`

### 2. Checkpointing

Important variables include:

- `FWS_CHECKPOINT_EVERY_TICKS`
- `FWS_CHECKPOINT_ON_EXIT`
- `FWS_CHECKPOINT_KEEP_LAST_N`
- `FWS_CHECKPOINT_PIN_ON_MANUAL`
- `FWS_CHECKPOINT_PIN_TAG`
- `FWS_CHECKPOINT_TRIGGER_FILE`

### 3. Headless logging cadence

Important variables include:

- `FWS_HEADLESS_PRINT_EVERY_TICKS`
- `FWS_HEADLESS_PRINT_LEVEL`
- `FWS_HEADLESS_PRINT_GPU`

### 4. Telemetry

The telemetry section is extensive. Important variables include:

- `FWS_TELEMETRY`
- `FWS_TELEM_SCHEMA`
- `FWS_TELEM_RUN_META`
- `FWS_TELEM_TICK_SUMMARY_EVERY`
- `FWS_TELEM_TICK_EVERY`
- `FWS_TELEM_SNAPSHOT_EVERY`
- `FWS_TELEM_VALIDATE_EVERY`
- `FWS_TELEM_FLUSH_EVERY`
- `FWS_TELEM_MOVE_EVERY`
- `FWS_TELEM_MOVE_MAX`
- `FWS_TELEM_MOVE_RATE`
- `FWS_TELEM_DMG_MODE`
- `FWS_TELEM_EVENTS_FMT`
- `FWS_TELEM_SNAP_FMT`
- `FWS_TELEM_REPORT`
- `FWS_TELEM_PPO`
- `FWS_TELEM_PPO_RICH_CSV`

### 5. Device and execution mode

Important variables include:

- `FWS_CUDA`
- `FWS_AMP`
- `FWS_USE_VMAP`
- `FWS_VMAP_MIN_BUCKET`
- `FWS_UI`
- `FWS_INSPECTOR_MODE`
- `FWS_INSPECTOR_UI_NO_OUTPUT`

### 6. World geometry and population

Important variables include:

- `FWS_GRID_W`
- `FWS_GRID_H`
- `FWS_START_PER_TEAM`
- `FWS_MAX_AGENTS`
- `FWS_TICK_LIMIT`
- `FWS_TARGET_TPS`

### 7. Walls and zone generation

Important variables include:

- `FWS_RAND_WALLS`
- `FWS_WALL_SEG_MIN`
- `FWS_WALL_SEG_MAX`
- `FWS_WALL_MARGIN`
- `FWS_HEAL_COUNT`
- `FWS_HEAL_SIZE_RATIO`
- `FWS_CP_COUNT`
- `FWS_CP_SIZE_RATIO`
- `FWS_CP_REWARD`

### 8. Unit and combat parameters

Important variables include:

- `FWS_MAX_HP`
- `FWS_SOLDIER_HP`
- `FWS_ARCHER_HP`
- `FWS_BASE_ATK`
- `FWS_SOLDIER_ATK`
- `FWS_ARCHER_ATK`
- `FWS_ARCHER_RANGE`
- `FWS_ARCHER_BLOCK_LOS`
- `FWS_VISION_SOLDIER`
- `FWS_VISION_ARCHER`

### 9. Observation parameters

Important variables include:

- `FWS_RAY_TOKENS`
- `OBS_SCHEMA_VERSION`
- `OBS_SCHEMA_FAMILY`

The last two are internal constants rather than environment knobs, but they directly affect checkpoint compatibility.

### 10. Respawn and spawn policy

Important variables include:

- `FWS_RESPAWN`
- `FWS_RESP_FLOOR_PER_TEAM`
- `FWS_RESP_MAX_PER_TICK`
- `FWS_RESP_PERIOD_TICKS`
- `FWS_RESP_PERIOD_BUDGET`
- `FWS_RESP_HYST_COOLDOWN_TICKS`
- `FWS_SPAWN_MODE`
- `FWS_SPAWN_ARCHER_RATIO`
- `FWS_CLONE_PROB`
- `FWS_RESP_PARENT_SELECT_MODE`
- `FWS_RESP_PARENT_TOPK_FRAC`
- `FWS_RESP_SPAWN_LOCATION_MODE`
- `FWS_RESP_SPAWN_NEAR_PARENT_RADIUS`

### 11. Reward shaping

Important variables include:

- `FWS_REW_KILL`
- `FWS_REW_DMG_DEALT`
- `FWS_REW_DEATH`
- `FWS_REW_DMG_TAKEN`
- `FWS_PPO_REW_HP_TICK`
- `FWS_PPO_HP_REWARD_MODE`
- `FWS_PPO_HP_REWARD_THRESHOLD`
- `FWS_PPO_REW_DMG_DEALT_AGENT`
- `FWS_PPO_PEN_DMG_TAKEN_AGENT`
- `FWS_PPO_REW_KILL_AGENT`
- `FWS_PPO_REW_DEATH`
- `FWS_PPO_REW_CONTEST`

### 12. PPO hyperparameters

Important variables include:

- `FWS_PPO_ENABLED`
- `FWS_PPO_TICKS`
- `FWS_PPO_LR`
- `FWS_PPO_CLIP`
- `FWS_PPO_ENTROPY`
- `FWS_PPO_VCOEF`
- `FWS_PPO_EPOCHS`
- `FWS_PPO_MINIB`
- `FWS_PPO_MAXGN`
- `FWS_PPO_TKL`
- `FWS_PPO_GAMMA`
- `FWS_PPO_LAMBDA`
- `FWS_PPO_UPDATE_TICKS`

### 13. Brain selection

Important variables include:

- `FWS_BRAIN`
- `FWS_TEAM_BRAIN_ASSIGNMENT`
- `FWS_TEAM_BRAIN_MODE`
- `FWS_TEAM_BRAIN_MIX_STRATEGY`
- `FWS_TEAM_BRAIN_RED`
- `FWS_TEAM_BRAIN_BLUE`
- `FWS_TEAM_BRAIN_P_*`
- `FWS_BRAIN_MLP_DMODEL`
- `FWS_BRAIN_MLP_ACT`
- `FWS_BRAIN_MLP_NORM`
- `FWS_BRAIN_MLP_RAY_SUMMARY`

### 14. Catastrophe system

Important variables include:

- `FWS_CATASTROPHE_ENABLED`
- `FWS_CATASTROPHE_DYNAMIC_SCHEDULER_ENABLED`
- `FWS_CATASTROPHE_ALLOW_OVERLAP`
- `FWS_CATASTROPHE_OVERRIDE_LOCKS_EDIT_MASK`
- `FWS_CATASTROPHE_MANUAL_REPLACE_EXISTING`
- `FWS_CATASTROPHE_CLEAR_ENABLED`
- duration variables for each preset
- scheduler pressure and hazard-law variables
- dynamic preset weights

### 15. UI and recording

Important variables include:

- `FWS_TARGET_FPS`
- `FWS_CELL_SIZE`
- `FWS_VIEWER_STATE_REFRESH_EVERY`
- `FWS_VIEWER_PICK_REFRESH_EVERY`
- `FWS_VIEWER_SIDE_PANEL_WIDTH`
- `FWS_VIEWER_BOTTOM_PANEL_HEIGHT`
- `FWS_VIEWER_ZONE_OVERLAY_DEFAULT`
- `FWS_VIEWER_HUD_SHOW_SCHEDULER`
- `FWS_VIEWER_BASE_ZONE_EDIT_STEP`
- `FWS_RECORD_VIDEO`
- `FWS_VIDEO_FPS`
- `FWS_VIDEO_SCALE`
- `FWS_VIDEO_EVERY_TICKS`

## Config validation

`config.py` performs invariant checks such as:

- positive dimensions and capacities
- observation-width consistency
- allowed brain names
- catastrophe parameter ranges
- profile-name sanity
- probability bounds
- max-interval versus min-interval consistency

Warnings can be escalated by strict mode.

## Practical control guidance

### Stable experiment setup

A stable experiment record should generally pin at least:

- seed
- grid size
- start population
- checkpoint cadence
- PPO hyperparameters
- reward settings
- catastrophe settings
- UI/headless mode
- output continuity behavior

### Separate runtime control from config control

A viewer keypress is not the same thing as a config change. For example:

- `G` toggles catastrophe system state during a run
- `FWS_CATASTROPHE_ENABLED` changes the startup default

Both matter, but they operate at different timescales.

### Be careful with reward defaults

Several score and team-reward knobs default to zero, while some PPO individual reward knobs default to large nonzero values. That balance can change the behavioral pressure materially.

### Be careful with schema-affecting changes

Changes to observation width, ordering, or schema identity can invalidate checkpoints that contain policies or PPO state.

## Suggested next documents

- [Getting started](02-getting-started.md)
- [Learning and optimization](07-learning-and-optimization.md)
- [Catastrophe mechanics](08-catastrophe-mechanics.md)
