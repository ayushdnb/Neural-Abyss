from __future__ import annotations

"""
Neural-Abyss configuration.

This file is written to serve two purposes at once:
1. executable runtime configuration, and
2. a readable operating manual for researchers and engineers.

Philosophy
==========
A reader should be able to answer the following questions by reading only this
file:
- What does each knob control?
- Which values are valid?
- When should I increase or decrease it?
- What tradeoff does it create?
- Which settings are strict schema contracts and should rarely be changed?
- Which settings are merely operator choices for speed, scale, and reporting?

Override model
==============
Most settings are resolved from environment variables whose names start with
``FWS_``. Resolution precedence is:

    explicit environment variable > profile override > hard-coded default

This means:
- hard-coded defaults define the baseline behavior,
- profiles can reshape groups of defaults for common operating modes,
- but an explicit environment variable always wins.

Important honesty rule
======================
This file tries to document actual observed runtime behavior, not only intended
behavior. Where the runtime currently supports only a narrower implementation
than a knob name might suggest, the comments say so explicitly.
"""

import math
import os
from typing import Any

import torch


# =============================================================================
# INTERNAL WARNING COLLECTION
# =============================================================================
# Why this exists:
# - Configuration is part of the experiment state.
# - Silent fallback is dangerous in research systems.
# - We therefore keep a warning buffer that can later be exported into
#   telemetry, reports, debug panels, or tests.
# =============================================================================

_CONFIG_WARNINGS: list[str] = []


def _config_warn(msg: str) -> None:
    """Record and print a configuration warning."""
    msg = str(msg)
    _CONFIG_WARNINGS.append(msg)
    print(f"[config][WARN] {msg}")


# =============================================================================
# ENVIRONMENT PARSING UTILITIES
# =============================================================================
# These helpers convert string environment variables into typed Python values.
# Centralizing parsing keeps the module consistent and audit-friendly.
# =============================================================================


def _env_bool(key: str, default: bool) -> bool:
    """Read an environment variable as bool.

    Accepted truthy tokens:
        1, true, yes, y, on, t

    Accepted falsy tokens:
        0, false, no, n, off, f

    Missing or invalid values fall back to ``default``.
    Invalid values emit a warning instead of failing immediately.
    """
    raw = os.getenv(key)
    if raw is None:
        return bool(default)

    norm = raw.strip().lower()
    if norm in ("1", "true", "yes", "y", "on", "t"):
        return True
    if norm in ("0", "false", "no", "n", "off", "f"):
        return False

    _config_warn(f"Unknown boolean env {key}={raw!r}; using default={bool(default)}")
    return bool(default)


def _env_float(key: str, default: float) -> float:
    """Read an environment variable as float.

    Missing or invalid values fall back to ``default``.
    Range validation happens later in the invariant-validation stage.
    """
    raw = os.getenv(key)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except Exception:
        _config_warn(f"Invalid float env {key}={raw!r}; using default={float(default)}")
        return float(default)


def _env_int(key: str, default: int) -> int:
    """Read an environment variable as int.

    Missing or invalid values fall back to ``default``.
    """
    raw = os.getenv(key)
    if raw is None:
        return int(default)
    try:
        return int(raw)
    except Exception:
        _config_warn(f"Invalid int env {key}={raw!r}; using default={int(default)}")
        return int(default)


def _env_str(key: str, default: str) -> str:
    """Read an environment variable as string.

    This helper intentionally does not strip whitespace automatically because
    some callers may want exact text. Most configuration assignments below call
    ``.strip()`` where surrounding whitespace has no meaning.
    """
    raw = os.getenv(key)
    return default if raw is None else str(raw)


def _env_is_set(key: str) -> bool:
    """Return True if the environment variable exists at all.

    Important detail:
    An empty string still counts as explicitly set.
    This matters for override precedence logic.
    """
    return os.getenv(key) is not None


# =============================================================================
# CONFIGURATION DIAGNOSTICS
# =============================================================================
# These helpers let other systems inspect the resolved configuration state.
# =============================================================================

# What it does:
#   Turns config warnings into hard errors.
# When to enable:
#   - CI
#   - sweep launchers
#   - production-like experiment pipelines
# When to leave off:
#   - exploratory local iteration where fallback is acceptable
CONFIG_STRICT: bool = _env_bool("FWS_CONFIG_STRICT", False)


def _config_issue(msg: str) -> None:
    """Warn by default, or raise immediately in strict mode."""
    if bool(CONFIG_STRICT):
        raise ValueError(f"[config] {msg}")
    _config_warn(msg)


def config_warnings() -> tuple[str, ...]:
    """Return an immutable snapshot of warnings collected during import."""
    return tuple(_CONFIG_WARNINGS)


def dump_config_dict() -> dict[str, Any]:
    """Return a JSON-friendly snapshot of resolved public config globals."""
    out: dict[str, Any] = {}
    for key, value in globals().items():
        if key.startswith("_"):
            continue
        if key in {"os", "math", "torch", "Any"}:
            continue
        if callable(value):
            continue

        if isinstance(value, (str, int, float, bool)) or value is None:
            out[key] = value
        elif isinstance(value, torch.device):
            out[key] = str(value)

    out["CONFIG_WARNINGS"] = list(_CONFIG_WARNINGS)
    return out


# =============================================================================
# EXPERIMENT IDENTITY, OUTPUTS, AND RESUME POLICY
# =============================================================================

# PROFILE
# -------
# Valid values:
#   "default", "debug", "train_fast", "train_quality"
#
# What it does:
#   Applies a bundle of preset overrides for values that were not explicitly set
#   via environment variables.
#
# When to set:
#   - use "debug" for small, quick, interactive runs
#   - use "train_fast" when throughput matters more than representation width
#   - use "train_quality" when you want somewhat higher model capacity
#   - use "default" when you want only the literal per-knob defaults
PROFILE: str = _env_str("FWS_PROFILE", "default").strip().lower()

# EXPERIMENT_TAG
# --------------
# Free-form label used for organization and human traceability.
# Set this whenever you are running multiple named experiments.
EXPERIMENT_TAG: str = _env_str("FWS_EXPERIMENT_TAG", "War").strip()

# RNG_SEED / SEED
# ---------------
# Master seed for reproducible randomness.
# Set a fixed integer for comparable experiments.
# Change it when you want independent random trials.
RNG_SEED: int = _env_int("FWS_SEED", 32)
SEED: int = int(RNG_SEED)

# RESULTS_DIR
# -----------
# Base directory where run folders, checkpoints, telemetry, and reports live.
# Change this when you want outputs on a different disk or storage root.
RESULTS_DIR: str = _env_str("FWS_RESULTS_DIR", "results").strip()

# CHECKPOINT_PATH
# ---------------
# Path to a checkpoint for resume.
# Leave empty to start fresh.
CHECKPOINT_PATH: str = _env_str("FWS_CHECKPOINT_PATH", "").strip()

# RESUME_OUTPUT_CONTINUITY
# ------------------------
# If True, a resumed run appends back into the original run directory.
# Use True when you want one continuous lineage.
# Use False when you want resume to produce separate output trees.
RESUME_OUTPUT_CONTINUITY: bool = _env_bool("FWS_RESUME_OUTPUT_CONTINUITY", True)

# RESUME_FORCE_NEW_RUN
# --------------------
# Stronger override than continuity: even if resuming from a checkpoint, create
# a fresh run folder.
# Use this when a resumed branch should be treated as a new experiment branch.
RESUME_FORCE_NEW_RUN: bool = _env_bool("FWS_RESUME_FORCE_NEW_RUN", False)

# RESUME_APPEND_STRICT_CSV_SCHEMA
# -------------------------------
# If True, append-mode CSV writers refuse schema/header mismatch.
# Enable for scientific rigor.
# Disable only if you intentionally accept schema drift in ad-hoc analysis.
RESUME_APPEND_STRICT_CSV_SCHEMA: bool = _env_bool("FWS_RESUME_APPEND_STRICT_CSV_SCHEMA", True)

# AUTOSAVE_EVERY_SEC
# ------------------
# Wall-clock autosave period.
# Increase it to reduce I/O.
# Decrease it to reduce potential work loss after hard crashes.
AUTOSAVE_EVERY_SEC: int = _env_int("FWS_AUTOSAVE_EVERY_SEC", 3600)


# =============================================================================
# CHECKPOINTING
# =============================================================================

# CHECKPOINT_EVERY_TICKS
# ----------------------
# Save a checkpoint every N ticks.
# 0 disables tick-based periodic checkpoints.
# Set lower values for safer long runs; set higher values for less I/O.
CHECKPOINT_EVERY_TICKS: int = _env_int("FWS_CHECKPOINT_EVERY_TICKS", 50_000)

# CHECKPOINT_ON_EXIT
# ------------------
# Save a checkpoint on clean shutdown.
# Recommended True for almost all serious runs.
CHECKPOINT_ON_EXIT: bool = _env_bool("FWS_CHECKPOINT_ON_EXIT", True)

# CHECKPOINT_KEEP_LAST_N
# ----------------------
# Number of latest non-pinned checkpoints to retain during pruning.
# Increase when you want more rollback options.
# Decrease when disk usage matters more than history depth.
CHECKPOINT_KEEP_LAST_N: int = _env_int("FWS_CHECKPOINT_KEEP_LAST_N", 1)

# CHECKPOINT_PIN_ON_MANUAL
# ------------------------
# Manual or trigger-file checkpoints may be marked pinned.
# Pinned checkpoints are protected from ordinary keep-last-N pruning.
CHECKPOINT_PIN_ON_MANUAL: bool = _env_bool("FWS_CHECKPOINT_PIN_ON_MANUAL", True)

# CHECKPOINT_PIN_TAG
# ------------------
# Naming tag for pinned/manual checkpoints.
# Useful for quickly identifying hand-curated save points.
CHECKPOINT_PIN_TAG: str = _env_str("FWS_CHECKPOINT_PIN_TAG", "manual").strip()

# CHECKPOINT_TRIGGER_FILE
# -----------------------
# If runtime sees this filename inside the run directory, it performs a manual
# checkpoint.
# Use when you want an external "save now" signal without stopping the process.
CHECKPOINT_TRIGGER_FILE: str = _env_str("FWS_CHECKPOINT_TRIGGER_FILE", "checkpoint.now").strip()


# =============================================================================
# HEADLESS CONSOLE REPORTING
# =============================================================================

# HEADLESS_PRINT_EVERY_TICKS
# --------------------------
# Print status every N ticks in headless mode.
# 0 disables periodic prints.
# Lower values give more feedback but more console noise.
HEADLESS_PRINT_EVERY_TICKS: int = _env_int("FWS_HEADLESS_PRINT_EVERY_TICKS", 2000)

# HEADLESS_PRINT_LEVEL
# --------------------
# Current convention:
#   0 = minimal
#   1 = standard
#   2 = more detail
# Raise it when diagnosing behavior.
# Lower it when you want cleaner logs.
HEADLESS_PRINT_LEVEL: int = _env_int("FWS_HEADLESS_PRINT_LEVEL", 2)

# HEADLESS_PRINT_GPU
# ------------------
# Include GPU probe information in headless prints.
# Useful when monitoring resource usage.
# Turn off if polling overhead or cleaner logs matter more.
HEADLESS_PRINT_GPU: bool = _env_bool("FWS_HEADLESS_PRINT_GPU", False)


# =============================================================================
# SCIENTIFIC RECORDING / TELEMETRY
# =============================================================================

# TELEMETRY_ENABLED
# -----------------
# Master switch for telemetry.
# Keep True for serious experiments.
# Turn off only for very lightweight debugging or when output suppression is the
# explicit goal.
TELEMETRY_ENABLED: bool = _env_bool("FWS_TELEMETRY", True)

# TELEMETRY_TAG
# -------------
# Free-form label for telemetry files and reports.
TELEMETRY_TAG: str = _env_str("FWS_TELEMETRY_TAG", "").strip()

# TELEMETRY_SCHEMA_VERSION
# ------------------------
# Parser/report schema version string.
# Change only when you intentionally redefine downstream data contracts.
TELEMETRY_SCHEMA_VERSION: str = _env_str("FWS_TELEM_SCHEMA", "2").strip()

# TELEMETRY_WRITE_RUN_META
# ------------------------
# Write static run metadata once.
# Keep True unless you explicitly want no metadata sidecar.
TELEMETRY_WRITE_RUN_META: bool = _env_bool("FWS_TELEM_RUN_META", True)

# TELEMETRY_WRITE_AGENT_STATIC
# ----------------------------
# Write one-time per-agent static info.
TELEMETRY_WRITE_AGENT_STATIC: bool = _env_bool("FWS_TELEM_AGENT_STATIC", True)

# TELEMETRY_TICK_SUMMARY_EVERY
# ----------------------------
# Frequency of summary rows.
# Lower values = denser summaries.
# Higher values = lighter I/O.
TELEMETRY_TICK_SUMMARY_EVERY: int = _env_int("FWS_TELEM_TICK_SUMMARY_EVERY", 200)

# Periodic sampling / flush cadences.
# Lower = more frequent measurement, better temporal resolution, more overhead.
# Higher = lighter logging, coarser history.
TELEMETRY_TICK_METRICS_EVERY: int = _env_int("FWS_TELEM_TICK_EVERY", 200)
TELEMETRY_SNAPSHOT_EVERY: int = _env_int("FWS_TELEM_SNAPSHOT_EVERY", 500)
TELEMETRY_REGISTRY_SNAPSHOT_EVERY: int = _env_int("FWS_TELEM_REG_EVERY", 200)
TELEMETRY_VALIDATE_EVERY: int = _env_int("FWS_TELEM_VALIDATE_EVERY", 500)
TELEMETRY_PERIODIC_FLUSH_EVERY: int = _env_int("FWS_TELEM_FLUSH_EVERY", 1000)

# Buffer sizes.
# Increase for throughput and fewer writes.
# Decrease for lower memory pressure and smaller crash-loss windows.
TELEMETRY_EVENT_CHUNK_SIZE: int = _env_int("FWS_TELEM_EVENT_CHUNK", 200_000)
TELEMETRY_TICK_CHUNK_SIZE: int = _env_int("FWS_TELEM_TICK_CHUNK", 20_000)

# Event-type toggles.
# Enable what you need to study.
# Disable high-volume streams when output size matters more.
TELEMETRY_LOG_BIRTHS: bool = _env_bool("FWS_TELEM_BIRTHS", True)
TELEMETRY_LOG_DEATHS: bool = _env_bool("FWS_TELEM_DEATHS", True)
TELEMETRY_LOG_DAMAGE: bool = _env_bool("FWS_TELEM_DAMAGE", True)
TELEMETRY_LOG_KILLS: bool = _env_bool("FWS_TELEM_KILLS", True)
TELEMETRY_LOG_MOVES: bool = _env_bool("FWS_TELEM_MOVES", False)
TELEMETRY_LOG_PPO: bool = _env_bool("FWS_TELEM_PPO", True)

# TELEMETRY_PPO_RICH_CSV
# ----------------------
# Enables dedicated PPO diagnostics CSV.
# Use when training dynamics matter.
# Disable only if PPO detail output is unnecessary.
TELEMETRY_PPO_RICH_CSV: bool = _env_bool("FWS_TELEM_PPO_RICH_CSV", True)

# TELEMETRY_PPO_RICH_LEVEL
# ------------------------
# Requested values:
#   "update", "epoch", "minibatch"
# Runtime truth today:
#   the implementation emits update-level rows only and annotates when a finer
#   granularity was requested.
# Set to "update" unless you are intentionally documenting a richer request for
# future comparison.
TELEMETRY_PPO_RICH_LEVEL: str = _env_str("FWS_TELEM_PPO_RICH_LEVEL", "update").strip().lower()

# TELEMETRY_PPO_RICH_FLUSH_EVERY
# ------------------------------
# Flush frequency for PPO-rich CSV rows.
# 1 is safest and most immediate.
# Larger values reduce write calls.
TELEMETRY_PPO_RICH_FLUSH_EVERY: int = _env_int("FWS_TELEM_PPO_RICH_FLUSH_EVERY", 1)

# TELEMETRY_APPEND_SCHEMA_STRICT
# ------------------------------
# Require matching telemetry CSV schema when appending.
TELEMETRY_APPEND_SCHEMA_STRICT: bool = _env_bool("FWS_TELEM_APPEND_SCHEMA_STRICT", True)

# Headless live summary sidecar switches.
TELEMETRY_HEADLESS_LIVE_SUMMARY: bool = _env_bool("FWS_TELEM_HEADLESS_SUMMARY", True)
TELEMETRY_HEADLESS_SUMMARY_INCLUDE_TPS: bool = _env_bool("FWS_TELEM_SUMMARY_TPS", True)
TELEMETRY_HEADLESS_SUMMARY_INCLUDE_GPU: bool = _env_bool("FWS_TELEM_SUMMARY_GPU", True)
TELEMETRY_HEADLESS_SUMMARY_INCLUDE_TICK_METRICS: bool = _env_bool("FWS_TELEM_SUMMARY_TICK_METRICS", True)
TELEMETRY_HEADLESS_SUMMARY_INCLUDE_PPO: bool = _env_bool("FWS_TELEM_SUMMARY_PPO", True)

# Rare mutation event logging.
TELEMETRY_LOG_RARE_MUTATIONS: bool = _env_bool("FWS_TELEM_MUTATIONS", True)

# Move-event sampling controls.
# Use these when movement telemetry is enabled but raw per-move logging would be
# too large.
TELEMETRY_MOVE_EVENTS_EVERY: int = _env_int("FWS_TELEM_MOVE_EVERY", 100)
TELEMETRY_MOVE_EVENTS_MAX_PER_TICK: int = _env_int("FWS_TELEM_MOVE_MAX", 256)
TELEMETRY_MOVE_EVENTS_SAMPLE_RATE: float = _env_float("FWS_TELEM_MOVE_RATE", 0.4)

# Counter emission cadence.
# 0 disables counter emission.
TELEMETRY_COUNTERS_EVERY: int = _env_int("FWS_TELEM_COUNTERS_EVERY", 500)

# TELEMETRY_DAMAGE_MODE
# ---------------------
# Observed useful values:
#   "victim_sum" : aggregate damage per victim
#   "per_hit"    : additionally emit per-hit damage events
# Choose "victim_sum" for lighter output.
# Choose "per_hit" only when detailed causal combat traces matter.
TELEMETRY_DAMAGE_MODE: str = _env_str("FWS_TELEM_DMG_MODE", "victim_sum").strip().lower()

# TELEMETRY_EVENTS_FORMAT
# -----------------------
# Current runtime effectively supports JSONL only and safely degrades other
# requests back to JSONL with a note.
# Set to "jsonl" unless you are preserving metadata about a requested format.
TELEMETRY_EVENTS_FORMAT: str = _env_str("FWS_TELEM_EVENTS_FMT", "jsonl").strip().lower()

# TELEMETRY_EVENTS_GZIP
# ---------------------
# Reserved/future-facing in current runtime comments.
# Leave False unless your telemetry implementation path explicitly handles it.
TELEMETRY_EVENTS_GZIP: bool = _env_bool("FWS_TELEM_EVENTS_GZIP", False)

# TELEMETRY_TICKS_FORMAT / TELEMETRY_SNAPSHOT_FORMAT
# --------------------------------------------------
# These exist as configuration metadata and defaults.
# Operationally, current code paths behave with CSV ticks and NPZ snapshots as
# the practical defaults.
TELEMETRY_TICKS_FORMAT: str = _env_str("FWS_TELEM_TICKS_FMT", "csv").strip().lower()
TELEMETRY_SNAPSHOT_FORMAT: str = _env_str("FWS_TELEM_SNAP_FMT", "npz").strip().lower()

# TELEMETRY_VALIDATE_LEVEL
# ------------------------
# 0 = off, 1 = basic, 2+ = stricter.
# Raise it when checking data integrity.
# Lower it when maximizing throughput matters more.
TELEMETRY_VALIDATE_LEVEL: int = _env_int("FWS_TELEM_VALIDATE", 1)

# Abort on telemetry anomaly.
# Good for strict pipelines, less ideal for exploratory runs.
TELEMETRY_ABORT_ON_ANOMALY: bool = _env_bool("FWS_TELEM_ABORT", False)

# End-of-run report artifact switches.
TELEMETRY_REPORT_ENABLE: bool = _env_bool("FWS_TELEM_REPORT", True)
TELEMETRY_REPORT_EXCEL: bool = _env_bool("FWS_TELEM_EXCEL", False)
TELEMETRY_REPORT_PNG: bool = _env_bool("FWS_TELEM_PNG", True)


# =============================================================================
# HARDWARE ACCELERATION AND TORCH EXECUTION MODE
# =============================================================================

# USE_CUDA
# --------
# Becomes True only if the knob requests CUDA AND CUDA is actually available.
# Set False when forcing CPU runs for debugging or reproducibility checks.
USE_CUDA: bool = _env_bool("FWS_CUDA", True) and torch.cuda.is_available()

# Canonical device used throughout the codebase.
DEVICE: torch.device = torch.device("cuda" if USE_CUDA else "cpu")
TORCH_DEVICE: torch.device = DEVICE

# AMP_ENABLED
# -----------
# Automatic Mixed Precision.
# Turn on for faster/lighter CUDA runs.
# Turn off when diagnosing numerical issues or when strict full-precision
# behavior is desired.
AMP_ENABLED: bool = _env_bool("FWS_AMP", True)


def amp_enabled() -> bool:
    """Return the resolved AMP policy."""
    return AMP_ENABLED


# Default tensor dtype based on device + AMP policy.
TORCH_DTYPE = torch.float16 if (USE_CUDA and AMP_ENABLED) else torch.float32

# VMAP controls.
# These affect batched execution across many independent models.
USE_VMAP: bool = _env_bool("FWS_USE_VMAP", True)
VMAP_MIN_BUCKET: int = _env_int("FWS_VMAP_MIN_BUCKET", 8)
VMAP_DEBUG: bool = _env_bool("FWS_VMAP_DEBUG", False)

# VMAP_STACK_CACHE_MAX
# --------------------
# Max cached stacked-state entries for torch.func/vmap inference.
# <= 0 disables the cache.
# Increase when repeated bucket structures are common and memory is available.
# Decrease when memory pressure matters more than cache reuse.
VMAP_STACK_CACHE_MAX: int = _env_int("FWS_VMAP_STACK_CACHE_MAX", 128)


# =============================================================================
# WORLD SCALE, CAPACITY, AND RUNTIME LENGTH
# =============================================================================

# Grid size.
# Increase for larger, sparser, slower worlds.
# Decrease for faster, denser experiments.
GRID_WIDTH: int = _env_int("FWS_GRID_W", 128)
GRID_HEIGHT: int = _env_int("FWS_GRID_H", 128)

# Starting population per team.
# Increase for richer population dynamics and heavier compute.
# Decrease for fast debugging.
START_AGENTS_PER_TEAM: int = _env_int("FWS_START_PER_TEAM", 200)

# Total registry slot capacity.
# Must be large enough for the starting population plus respawn dynamics.
MAX_AGENTS: int = _env_int("FWS_MAX_AGENTS", 500)

# Run length cap.
# 0 means unlimited.
TICK_LIMIT: int = _env_int("FWS_TICK_LIMIT", 0)

# Throttle target ticks per second.
# 0 means unthrottled.
TARGET_TPS: int = _env_int("FWS_TARGET_TPS", 0)

# Strict agent-registry schema size. Do not casually change.
AGENT_FEATURES: int = 10


# =============================================================================
# MAP TOPOLOGY, OBJECTIVES, AND HEALING LANDSCAPE
# =============================================================================

# Random wall count.
# Raise for more obstacle structure and tactical routing.
# Lower for more open battlefields.
RANDOM_WALLS: int = _env_int("FWS_RAND_WALLS", 18)

# Wall segment length bounds.
WALL_SEG_MIN: int = _env_int("FWS_WALL_SEG_MIN", 5)
WALL_SEG_MAX: int = _env_int("FWS_WALL_SEG_MAX", 40)

# Boundary margin for wall placement.
WALL_AVOID_MARGIN: int = _env_int("FWS_WALL_MARGIN", 3)

# Straightness and gap probabilities for wall generation.
# Higher straight probability -> longer corridors.
# Higher gap probability -> more openings/passability.
MAP_WALL_STRAIGHT_PROB: float = _env_float("FWS_MAP_WALL_STRAIGHT_PROB", 0.65)
MAP_WALL_GAP_PROB: float = _env_float("FWS_MAP_WALL_GAP_PROB", 0.20)

# Heal-zone parameters.
# Raise count/size/rate when sustain should matter more.
# Lower them when combat lethality or territorial churn should dominate.
HEAL_ZONE_COUNT: int = _env_int("FWS_HEAL_COUNT", 8)
HEAL_ZONE_SIZE_RATIO: float = _env_float("FWS_HEAL_SIZE_RATIO", 5 / 64)
HEAL_RATE: float = _env_float("FWS_HEAL_RATE", 0.004)


# =============================================================================
# HEAL-ZONE CATASTROPHE SCHEDULER
# =============================================================================

# Enable or disable the scheduler entirely.
# Turn on when heal camping is too stable.
# Leave off when studying the base heal-zone ecology.
CATASTROPHE_SCHEDULER_ENABLED: bool = _env_bool("FWS_CATASTROPHE_ENABLED", True)

# Scheduler mode.
# Valid values:
#   "periodic" : timing/cooldown driven
#   "dynamic"  : triggered by sustained heal occupancy signal
# Use periodic for simple controlled experiments.
# Use dynamic when you want response to emergent behavior.
CATASTROPHE_SCHEDULER_MODE: str = _env_str("FWS_CATASTROPHE_MODE", "dynamic").strip().lower()
if CATASTROPHE_SCHEDULER_MODE not in ("periodic", "dynamic"):
    _config_issue(
        f"Invalid FWS_CATASTROPHE_MODE={CATASTROPHE_SCHEDULER_MODE!r}; using 'periodic'"
    )
    CATASTROPHE_SCHEDULER_MODE = "periodic"

# Cooldown after one catastrophe clears.
CATASTROPHE_COOLDOWN_TICKS: int = max(0, _env_int("FWS_CATASTROPHE_COOLDOWN_TICKS", 8_000))

# Duration of one catastrophe event.
CATASTROPHE_DURATION_TICKS: int = max(1, _env_int("FWS_CATASTROPHE_DURATION_TICKS", 1_800))

# Never suppress all heal zones; keep at least this many active.
CATASTROPHE_MIN_ACTIVE_HEAL_ZONES: int = max(1, _env_int("FWS_CATASTROPHE_MIN_ACTIVE_HEAL_ZONES", 2))

# Minimum total zone count required before catastrophe logic is allowed.
CATASTROPHE_MIN_ZONE_COUNT_TO_TRIGGER: int = max(
    CATASTROPHE_MIN_ACTIVE_HEAL_ZONES + 1,
    _env_int("FWS_CATASTROPHE_MIN_ZONE_COUNT_TO_TRIGGER", 3),
)

# If True, suppression must leave at least one active zone on each map half.
# Use when you want catastrophes to reduce camping without fully collapsing map
# accessibility across one side.
CATASTROPHE_REQUIRE_BOTH_HALVES_COVERED: bool = _env_bool(
    "FWS_CATASTROPHE_REQUIRE_BOTH_HALVES_COVERED",
    True,
)

# Dynamic-mode occupancy trigger threshold.
CATASTROPHE_DYNAMIC_HEAL_OCCUPANCY_THRESHOLD: float = _env_float(
    "FWS_CATASTROPHE_DYNAMIC_HEAL_OCCUPANCY_THRESHOLD",
    0.28,
)

# Number of consecutive ticks the dynamic signal must persist.
CATASTROPHE_DYNAMIC_SUSTAIN_TICKS: int = max(
    1,
    _env_int("FWS_CATASTROPHE_DYNAMIC_SUSTAIN_TICKS", 90),
)

# Fractional severity knobs.
# Raise them to suppress more healing support.
# Lower them to make catastrophes milder.
CATASTROPHE_SMALL_SUPPRESS_FRACTION: float = _env_float(
    "FWS_CATASTROPHE_SMALL_SUPPRESS_FRACTION",
    0.25,
)
CATASTROPHE_MEDIUM_SUPPRESS_FRACTION: float = _env_float(
    "FWS_CATASTROPHE_MEDIUM_SUPPRESS_FRACTION",
    0.50,
)
CATASTROPHE_CLUSTER_SURVIVOR_FRACTION: float = _env_float(
    "FWS_CATASTROPHE_CLUSTER_SURVIVOR_FRACTION",
    0.25,
)

# Log catastrophe trigger/clear events to console.
CATASTROPHE_LOG_EVENTS: bool = _env_bool("FWS_CATASTROPHE_LOG_EVENTS", True)


# Capture points.
# Raise count/size/reward when territorial control should matter more.
# Lower reward to make CP mostly informational/spatial rather than score-driving.
CP_COUNT: int = _env_int("FWS_CP_COUNT", 6)
CP_SIZE_RATIO: float = _env_float("FWS_CP_SIZE_RATIO", 0.08)
CP_REWARD_PER_TICK: float = _env_float("FWS_CP_REWARD", 0.0)


# =============================================================================
# COMBAT BIOLOGY, UNIT CLASSES, METABOLISM, AND SENSING
# =============================================================================

UNIT_SOLDIER_ID: int = 1
UNIT_ARCHER_ID: int = 2
UNIT_SOLDIER: int = UNIT_SOLDIER_ID
UNIT_ARCHER: int = UNIT_ARCHER_ID

# HP and damage scales.
# Increase HP for longer engagements.
# Increase ATK for higher lethality.
MAX_HP: float = _env_float("FWS_MAX_HP", 1.0)
SOLDIER_HP: float = _env_float("FWS_SOLDIER_HP", 1.0)
ARCHER_HP: float = _env_float("FWS_ARCHER_HP", 0.65)
BASE_ATK: float = _env_float("FWS_BASE_ATK", 0.18)
SOLDIER_ATK: float = _env_float("FWS_SOLDIER_ATK", 0.15)
ARCHER_ATK: float = _env_float("FWS_ARCHER_ATK", 0.10)
MAX_ATK: float = max(SOLDIER_ATK, ARCHER_ATK, BASE_ATK, 1e-6)

# Archer-specific mechanics.
ARCHER_RANGE: int = _env_int("FWS_ARCHER_RANGE", 4)
ARCHER_LOS_BLOCKS_WALLS: bool = _env_bool("FWS_ARCHER_BLOCK_LOS", True)

# Metabolism drain.
# Raise it to make healing/objectives/combat more urgent.
# Lower it to allow more passive roaming or longer standoffs.
METABOLISM_ENABLED: bool = _env_bool("FWS_META_ON", True)
META_SOLDIER_HP_PER_TICK: float = _env_float("FWS_META_SOLDIER", 0.0020)
META_ARCHER_HP_PER_TICK: float = _env_float("FWS_META_ARCHER", 0.0015)

# Vision ranges.
# Raise to make agents more informed and long-range aware.
# Lower to make combat more local and uncertain.
VISION_RANGE_SOLDIER: int = _env_int("FWS_VISION_SOLDIER", 8)
VISION_RANGE_ARCHER: int = _env_int("FWS_VISION_ARCHER", 16)

# Class-specific vision lookup used elsewhere in the engine.
VISION_RANGE_BY_UNIT = {
    UNIT_SOLDIER_ID: VISION_RANGE_SOLDIER,
    UNIT_ARCHER_ID: VISION_RANGE_ARCHER,
}

# Maximum raycast depth derived from the largest vision range.
RAYCAST_MAX_STEPS: int = max(max(VISION_RANGE_BY_UNIT.values()), 1)
RAY_MAX_STEPS: int = RAYCAST_MAX_STEPS

# INSTINCT_RADIUS
# ---------------
# Radius for broader neighborhood context features.
# Raise when richer local-context sensing is desired.
# Lower when simplifying observation context.
INSTINCT_RADIUS: int = _env_int("FWS_INSTINCT_RADIUS", 20)


# =============================================================================
# OBSERVATION LAYOUT CONTRACT
# =============================================================================
# These values are schema contracts. Changing them is not just a tuning choice.
# It changes the meaning and dimensionality of observations, which affects model
# architecture, checkpoint compatibility, PPO buffer shapes, and feature split
# logic in other modules.
# =============================================================================

# Number of ray tokens per observation.
RAY_TOKEN_COUNT: int = _env_int("FWS_RAY_TOKENS", 32)

# Features carried by each ray token.
# Strict schema constant.
RAY_FEAT_DIM: int = 8

# Flattened ray block size.
RAYS_FLAT_DIM: int = RAY_TOKEN_COUNT * RAY_FEAT_DIM

# Rich non-ray feature dimensions.
RICH_BASE_DIM: int = 23
INSTINCT_DIM: int = 4
RICH_TOTAL_DIM: int = RICH_BASE_DIM + INSTINCT_DIM

# Final observation size.
OBS_DIM: int = RAYS_FLAT_DIM + RICH_TOTAL_DIM

# Semantic grouping used by observation splitting/token semantics.
# These indices must stay aligned with actual feature construction elsewhere.
SEMANTIC_RICH_BASE_INDICES = {
    "own_context": (0, 1, 2, 5, 6, 7, 8),
    "world_context": (11, 20, 21, 22),
    "zone_context": (9, 10),
    "team_context": (3, 4, 12, 13, 14, 15),
    "combat_context": (16, 17, 18, 19),
}

# Canonical semantic token order expected by token-building logic.
SEMANTIC_TOKEN_ORDER = (
    "own_context",
    "world_context",
    "zone_context",
    "team_context",
    "combat_context",
    "instinct_context",
)

# Number of discrete actions.
# Another critical architecture/checkpoint contract.
NUM_ACTIONS: int = _env_int("FWS_NUM_ACTIONS", 41)


# =============================================================================
# POPULATION CONTROL, INITIAL SPAWNING, AND RESPAWN EVOLUTION
# =============================================================================

# RESPAWN_ENABLED
# ---------------
# Master switch for reinforcement / repopulation.
# Keep True when studying long-lived population dynamics.
# Turn off when you want pure extinction-style combat.
RESPAWN_ENABLED: bool = _env_bool("FWS_RESPAWN", True)

# Population floor per team.
# When a team drops below this level, respawn logic can help replenish it.
RESP_FLOOR_PER_TEAM: int = _env_int("FWS_RESP_FLOOR_PER_TEAM", 100)

# Hard cap on respawns per tick.
RESP_MAX_PER_TICK: int = _env_int("FWS_RESP_MAX_PER_TICK", 3)

# Periodic reinforcement window.
RESP_PERIOD_TICKS: int = _env_int("FWS_RESP_PERIOD_TICKS", 10_000)
RESP_PERIOD_BUDGET: int = _env_int("FWS_RESP_PERIOD_BUDGET", 40)

# Cooldown hysteresis to avoid rapid refill oscillation.
RESP_HYST_COOLDOWN_TICKS: int = _env_int("FWS_RESP_HYST_COOLDOWN_TICKS", 100)

# Wall margin for spawn placement.
RESP_WALL_MARGIN: int = _env_int("FWS_RESP_WALL_MARGIN", 2)

# SPAWN_MODE
# ----------
# Initial population spawn layout mode.
# Valid values:
#   "uniform"   : uniform random spawning across valid map space
#   "symmetric" : mirrored / symmetry-aware opening layout
# Use uniform for organic starts.
# Use symmetric for more controlled comparability between teams.
SPAWN_MODE: str = _env_str("FWS_SPAWN_MODE", "uniform").strip().lower()
if SPAWN_MODE not in ("uniform", "symmetric"):
    _config_warn(f"Unknown SPAWN_MODE={SPAWN_MODE!r}; falling back to 'uniform'")
    SPAWN_MODE = "uniform"

# Fraction of initial spawns that are archers.
SPAWN_ARCHER_RATIO: float = _env_float("FWS_SPAWN_ARCHER_RATIO", 0.35)

# Core respawn probabilities and placement search budget.
RESPAWN_PROB_PER_DEAD: float = _env_float("FWS_RESPAWN_PROB", 0.05)
RESPAWN_SPAWN_TRIES: int = _env_int("FWS_RESPAWN_TRIES", 200)
RESPAWN_MUTATION_STD: float = _env_float("FWS_MUT_STD", 0.05)
RESPAWN_CLONE_PROB: float = _env_float("FWS_CLONE_PROB", 1.0)
RESPAWN_USE_TEAM_ELITE: bool = _env_bool("FWS_TEAM_ELITE", True)
RESPAWN_RESET_OPT_ON_RESPAWN: bool = _env_bool("FWS_RESET_OPT", True)
RESPAWN_JITTER_RADIUS: int = _env_int("FWS_RESP_JITTER", 1)
RESPAWN_COOLDOWN_TICKS: int = _env_int("FWS_RESPAWN_CD", 300)
RESPAWN_BATCH_PER_TEAM: int = _env_int("FWS_RESPAWN_BATCH", 2)
RESPAWN_ARCHER_SHARE: float = _env_float("FWS_RESPAWN_ARCHER_SHARE", 0.50)
RESPAWN_INTERIOR_BIAS: float = _env_float("FWS_RESPAWN_INTERIOR_BIAS", 0.10)

# Rare-mutation evolution layer.
# These knobs govern occasional stronger mutation events for evolutionary
# variety. Increase them when diversity pressure matters more. Lower them when
# stability and inheritance fidelity matter more.
RESPAWN_RARE_MUTATION_TICK_WINDOW_ENABLE: bool = _env_bool("FWS_RESP_RARE_TICK_WINDOW_ENABLE", True)
RESPAWN_RARE_MUTATION_TICK_WINDOW_TICKS: int = _env_int("FWS_RESP_RARE_TICK_WINDOW", 7000)
RESPAWN_RARE_MUTATION_PHYSICAL_ENABLE: bool = _env_bool("FWS_RESP_RARE_PHYS_ENABLE", True)
RESPAWN_RARE_MUTATION_PHYSICAL_DRIFT_STD_FRAC: float = _env_float("FWS_RESP_RARE_PHYS_STD", 0.10)
RESPAWN_RARE_MUTATION_PHYSICAL_DRIFT_CLIP_FRAC: float = _env_float("FWS_RESP_RARE_PHYS_CLIP", 0.15)
RESPAWN_RARE_MUTATION_INHERITED_BRAIN_NOISE_ENABLE: bool = _env_bool("FWS_RESP_RARE_BRAIN_NOISE_ENABLE", True)
RESPAWN_RARE_MUTATION_INHERITED_BRAIN_NOISE_STD: float = _env_float("FWS_RESP_RARE_BRAIN_NOISE_STD", 0.15)

# RESPAWN_PARENT_SELECTION_MODE
# -----------------------------
# Valid values:
#   "random"         : choose a parent candidate randomly
#   "topk_weighted"  : choose from a top-k pool with weighted bias
# Use random for diversity.
# Use topk_weighted when you want stronger selection pressure.
RESPAWN_PARENT_SELECTION_MODE: str = _env_str("FWS_RESP_PARENT_SELECT_MODE", "topk_weighted").strip().lower()
if RESPAWN_PARENT_SELECTION_MODE not in ("random", "topk_weighted"):
    _config_warn(
        f"Unknown RESPAWN_PARENT_SELECTION_MODE={RESPAWN_PARENT_SELECTION_MODE!r}; "
        "falling back to 'random'"
    )
    RESPAWN_PARENT_SELECTION_MODE = "random"

# Fraction of parent candidates retained in the top-k pool.
RESPAWN_PARENT_SELECTION_TOPK_FRAC: float = _env_float("FWS_RESP_PARENT_TOPK_FRAC", 0.08)

# Exponent controlling how strongly weights favor higher scores within the pool.
RESPAWN_PARENT_SELECTION_SCORE_POWER: float = _env_float("FWS_RESP_PARENT_SCORE_POWER", 2.0)

# Doctrine of Birth / The Closed Cradle
# Ongoing births can be forced to remain bloodline-bound:
# - if True, a live same-team parent must exist for every birth
# - if False, legacy parentless fresh births remain possible
RESPAWN_REQUIRE_PARENT_FOR_BIRTH: bool = _env_bool("FWS_RESP_REQUIRE_PARENT_FOR_BIRTH", True)

# Active doctrine for scoring alive same-team parents when
# RESPAWN_PARENT_SELECTION_MODE="topk_weighted".
RESPAWN_BIRTH_DOCTRINE_MODE: str = _env_str("FWS_RESP_BIRTH_DOCTRINE_MODE", "personal_points").strip().lower()

# Pool sampled by random_per_birth.
RESPAWN_BIRTH_RANDOM_DOCTRINE_POOL = tuple(
    s.strip().lower()
    for s in _env_str("FWS_RESP_BIRTH_RANDOM_DOCTRINE_POOL", "overall").split(",")
    if s.strip()
)

# Explicit top-k parent count. Use 0 to retain legacy fraction-based sizing.
RESPAWN_BIRTH_TOPK_SIZE: int = _env_int("FWS_RESP_BIRTH_TOPK_SIZE", 12)

# Behavior when a doctrine yields no positive candidate score.
RESPAWN_BIRTH_ZERO_SCORE_FALLBACK: str = _env_str(
    "FWS_RESP_BIRTH_ZERO_SCORE_FALLBACK",
    "uniform_candidates",
).strip().lower()

# Relative blend weights for multi-axis doctrines.
RESPAWN_BIRTH_BLEND_WEIGHT_KILL: float = _env_float("FWS_RESP_BIRTH_BLEND_WEIGHT_KILL", 1.0)
RESPAWN_BIRTH_BLEND_WEIGHT_CP: float = _env_float("FWS_RESP_BIRTH_BLEND_WEIGHT_CP", 1.0)
RESPAWN_BIRTH_BLEND_WEIGHT_HEALTH: float = _env_float("FWS_RESP_BIRTH_BLEND_WEIGHT_HEALTH", 1.0)
RESPAWN_BIRTH_BLEND_WEIGHT_PERSONAL: float = _env_float("FWS_RESP_BIRTH_BLEND_WEIGHT_PERSONAL", 1.0)

# RESPAWN_SPAWN_LOCATION_MODE
# ---------------------------
# Valid values:
#   "uniform"     : place children via general spatial sampling
#   "near_parent" : try to place children near the chosen parent
# Use uniform when you want redistribution and mixing.
# Use near_parent when you want local lineage continuity or colony behavior.
RESPAWN_SPAWN_LOCATION_MODE: str = _env_str("FWS_RESP_SPAWN_LOCATION_MODE", "near_parent").strip().lower()
if RESPAWN_SPAWN_LOCATION_MODE not in ("uniform", "near_parent"):
    _config_warn(
        f"Unknown RESPAWN_SPAWN_LOCATION_MODE={RESPAWN_SPAWN_LOCATION_MODE!r}; "
        "falling back to 'uniform'"
    )
    RESPAWN_SPAWN_LOCATION_MODE = "uniform"

# Search radius around the parent when using near-parent spawning.
RESPAWN_SPAWN_NEAR_PARENT_RADIUS: int = _env_int(
    "FWS_RESP_SPAWN_NEAR_PARENT_RADIUS",
    3,
)


# =============================================================================
# REWARD SHAPING
# =============================================================================
# These knobs influence team-level and PPO-agent-level rewards.
# Raise with care: reward scale strongly affects training dynamics.
# =============================================================================

# Team-level combat rewards/penalties.
TEAM_KILL_REWARD: float = _env_float("FWS_REW_KILL", 0.0)
TEAM_DMG_DEALT_REWARD: float = _env_float("FWS_REW_DMG_DEALT", 0.0)
TEAM_DEATH_PENALTY: float = _env_float("FWS_REW_DEATH", 0.0)
TEAM_DMG_TAKEN_PENALTY: float = _env_float("FWS_REW_DMG_TAKEN", 0.0)

# PPO dense HP reward.
PPO_REWARD_HP_TICK: float = _env_float("FWS_PPO_REW_HP_TICK", 0.007)

# PPO_HP_REWARD_MODE
# ------------------
# Valid / observed values:
#   "raw"            : reward scales linearly with current HP
#   "threshold_ramp" : no HP reward below a threshold, then ramps upward
# Use raw for simple dense shaping.
# Use threshold_ramp when you want to reward staying substantially healthy,
# rather than merely being slightly above zero.
PPO_HP_REWARD_MODE: str = _env_str("FWS_PPO_HP_REWARD_MODE", "threshold_ramp").strip().lower()

# Threshold used only when PPO_HP_REWARD_MODE == "threshold_ramp".
# Runtime clamps it into [0, 1].
# A value around 0.60 means: no HP reward at or below 60% HP, then linearly
# increasing reward above that point.
PPO_HP_REWARD_THRESHOLD: float = _env_float("FWS_PPO_HP_REWARD_THRESHOLD", 0.60)

# Individual PPO reward terms.
PPO_REWARD_DMG_DEALT_INDIVIDUAL: float = _env_float("FWS_PPO_REW_DMG_DEALT_AGENT", 0.30)
# Runtime subtracts this coefficient, so positive values create a true penalty.
PPO_PENALTY_DMG_TAKEN_INDIVIDUAL: float = _env_float("FWS_PPO_PEN_DMG_TAKEN_AGENT", 0.45)
PPO_REWARD_KILL_INDIVIDUAL: float = _env_float("FWS_PPO_REW_KILL_AGENT", 1.5)
# Despite the name, the current PPO path applies this as a same-team death aggregate.
PPO_REWARD_DEATH: float = _env_float("FWS_PPO_REW_DEATH", 0.0)
PPO_REWARD_CONTESTED_CP: float = _env_float("FWS_PPO_REW_CONTEST", 0.35)
PPO_REWARD_HEALING_RECOVERY: float = _env_float("FWS_PPO_REW_HEALING_RECOVERY", 6.0)


# =============================================================================
# REINFORCEMENT LEARNING (PPO)
# =============================================================================

# Master PPO switch.
PPO_ENABLED: bool = _env_bool("FWS_PPO_ENABLED", True)

# Reset/log PPO state on startup/resume.
PPO_RESET_LOG: bool = _env_bool("FWS_PPO_RESET_LOG", False)

# Rollout horizon in ticks.
# Larger windows improve long-horizon information but delay updates.
# Smaller windows update more frequently but with shorter trajectories.
PPO_WINDOW_TICKS: int = _env_int("FWS_PPO_TICKS", 192)

# Optimizer learning rate.
PPO_LR: float = _env_float("FWS_PPO_LR", 3e-4)

# LR scheduler horizon and floor.
PPO_LR_T_MAX: int = _env_int("FWS_PPO_T_MAX", 10_000_000)
PPO_LR_ETA_MIN: float = _env_float("FWS_PPO_ETA_MIN", 1e-6)

# PPO clipping and coefficients.
PPO_CLIP: float = _env_float("FWS_PPO_CLIP", 0.2)
PPO_CLIP_EPS: float = PPO_CLIP
PPO_ENTROPY_COEF: float = _env_float("FWS_PPO_ENTROPY", 0.025)
PPO_VALUE_COEF: float = _env_float("FWS_PPO_VCOEF", 0.5)
PPO_EPOCHS: int = _env_int("FWS_PPO_EPOCHS", 4)
PPO_MINIBATCHES: int = _env_int("FWS_PPO_MINIB", 8)
PPO_MAX_GRAD_NORM: float = _env_float("FWS_PPO_MAXGN", 1.0)
PPO_TARGET_KL: float = _env_float("FWS_PPO_TKL", 0.02)
PPO_GAMMA: float = _env_float("FWS_PPO_GAMMA", 0.995)
PPO_LAMBDA: float = _env_float("FWS_PPO_LAMBDA", 0.95)
PPO_UPDATE_TICKS: int = _env_int("FWS_PPO_UPDATE_TICKS", 3)

# Compatibility alias.
# If the older env variable is set but the newer one is not, backfill it.
PPO_ENTROPY_BONUS: float = _env_float("FWS_PPO_ENTROPY_BONUS", PPO_ENTROPY_COEF)
if _env_is_set("FWS_PPO_ENTROPY_BONUS") and not _env_is_set("FWS_PPO_ENTROPY"):
    PPO_ENTROPY_COEF = float(PPO_ENTROPY_BONUS)

# Per-agent brain ownership.
# The codebase is designed around independent brains rather than one shared
# policy object for all agents.
PER_AGENT_BRAINS: bool = _env_bool("FWS_PER_AGENT_BRAINS", True)

# Global periodic mutation event across alive agents.
MUTATION_PERIOD_TICKS: int = _env_int("FWS_MUTATE_EVERY", 1_000_000_000)
MUTATION_FRACTION_ALIVE: float = _env_float("FWS_MUTATE_FRAC", 0.02)


# =============================================================================
# BRAIN FAMILY AND TEAM-ASSIGNMENT POLICY
# =============================================================================

# Default brain kind used when no team-specific override applies.
BRAIN_KIND: str = _env_str("FWS_BRAIN", "whispering_abyss").strip().lower()

# TEAM_BRAIN_ASSIGNMENT
# ---------------------
# If True, team-aware brain assignment logic may override the single default
# brain kind. If False, the default BRAIN_KIND is used more uniformly.
TEAM_BRAIN_ASSIGNMENT: bool = _env_bool("FWS_TEAM_BRAIN_ASSIGNMENT", True)

# TEAM_BRAIN_ASSIGNMENT_MODE
# --------------------------
# Synonym groups recognized by runtime:
#   fixed-per-team:
#       "exclusive", "split", "team"
#   mixed-per-spawn:
#       "mix", "hybrid", "both"
#
# Fixed-per-team mode:
#   each team uses its configured team-specific architecture.
# Mixed-per-spawn mode:
#   architecture can vary per spawn according to mix strategy.
TEAM_BRAIN_ASSIGNMENT_MODE: str = _env_str("FWS_TEAM_BRAIN_MODE", "mix").strip().lower()

# TEAM_BRAIN_MIX_STRATEGY
# -----------------------
# Synonym groups recognized by runtime:
#   deterministic cycling:
#       "alternate", "roundrobin", "rr"
#   weighted random draw:
#       "random", "prob", "probabilistic"
#
# Use alternate when you want exact repeatable architecture cycling.
# Use random/probabilistic when you want stochastic architecture mixtures.
TEAM_BRAIN_MIX_STRATEGY: str = _env_str("FWS_TEAM_BRAIN_MIX_STRATEGY", "random").strip().lower()

# Team-specific fixed architectures used in exclusive/split/team modes.
TEAM_BRAIN_EXCLUSIVE_RED: str = _env_str("FWS_TEAM_BRAIN_RED", "whispering_abyss").strip().lower()
TEAM_BRAIN_EXCLUSIVE_BLUE: str = _env_str("FWS_TEAM_BRAIN_BLUE", "veil_of_echoes").strip().lower()

# Sequence used by alternate/round-robin mixing.
TEAM_BRAIN_MIX_SEQUENCE = tuple(
    s.strip().lower()
    for s in _env_str(
        "FWS_TEAM_BRAIN_MIX_SEQUENCE",
        "whispering_abyss,veil_of_echoes,cathedral_of_ash,dreamer_in_black_fog,obsidian_pulse",
    ).split(",")
    if s.strip()
)

# Weighted probabilities used by random/probabilistic mixing.
TEAM_BRAIN_MIX_P_WHISPERING_ABYSS: float = _env_float("FWS_TEAM_BRAIN_P_WHISPERING_ABYSS", 0.10)
TEAM_BRAIN_MIX_P_VEIL_OF_ECHOES: float = _env_float("FWS_TEAM_BRAIN_P_VEIL_OF_ECHOES", 0.15)
TEAM_BRAIN_MIX_P_CATHEDRAL_OF_ASH: float = _env_float("FWS_TEAM_BRAIN_P_CATHEDRAL_OF_ASH", 0.25)
TEAM_BRAIN_MIX_P_DREAMER_IN_BLACK_FOG: float = _env_float("FWS_TEAM_BRAIN_P_DREAMER_IN_BLACK_FOG", 0.25)
TEAM_BRAIN_MIX_P_OBSIDIAN_PULSE: float = _env_float("FWS_TEAM_BRAIN_P_OBSIDIAN_PULSE", 0.25)

# Separate seed for team-brain mixing logic.
# Set explicitly when you want architecture-mixture randomness decoupled from the
# main run seed.
TEAM_BRAIN_MIX_SEED: int = _env_int("FWS_TEAM_BRAIN_MIX_SEED", int(globals().get("RNG_SEED", 0)))


# =============================================================================
# SIMPLE MLP BRAIN FAMILY
# =============================================================================
# These are architecture knobs for the MLP-based brain family.
# They are more structural than ordinary tuning knobs; changing them affects
# model shape and checkpoint compatibility.
# =============================================================================

# Shared embedding width for the two-token input contract.
# Increase when you want more representational capacity.
# Decrease for lighter, faster models.
BRAIN_MLP_D_MODEL: int = _env_int("FWS_BRAIN_MLP_DMODEL", 48)

# Derived final input width = 2 tokens * d_model.
BRAIN_MLP_FINAL_INPUT_WIDTH: int = 2 * BRAIN_MLP_D_MODEL

# Activation choice.
# Valid values:
#   "gelu", "relu", "silu"
# GELU is a good default for smooth MLPs.
# ReLU is simpler and common.
# SiLU can be useful for smoother gating-like behavior.
BRAIN_MLP_ACTIVATION: str = _env_str("FWS_BRAIN_MLP_ACT", "gelu").strip().lower()

# Normalization choice.
# Valid values:
#   "layernorm", "none"
# Use layernorm for stability.
# Use none only if you intentionally want a more bare architecture.
BRAIN_MLP_NORM: str = _env_str("FWS_BRAIN_MLP_NORM", "layernorm").strip().lower()

# Ray aggregation rule.
# Current supported value in runtime: "mean"
BRAIN_MLP_RAY_SUMMARY: str = _env_str("FWS_BRAIN_MLP_RAY_SUMMARY", "mean").strip().lower()

# Actor/critic head initialization gains.
# Smaller actor gain keeps early logits modest.
# Critic gain usually stays around 1.0.
BRAIN_MLP_ACTOR_INIT_GAIN: float = _env_float("FWS_BRAIN_MLP_ACTOR_GAIN", 0.01)
BRAIN_MLP_CRITIC_INIT_GAIN: float = _env_float("FWS_BRAIN_MLP_CRITIC_GAIN", 1.0)

# Stable architecture ordering used by validation, UI, telemetry, and mixing.
BRAIN_MLP_KIND_ORDER = (
    "whispering_abyss",
    "veil_of_echoes",
    "cathedral_of_ash",
    "dreamer_in_black_fog",
    "obsidian_pulse",
)

# Human-facing names.
BRAIN_KIND_DISPLAY_NAMES = {
    "whispering_abyss": "Whispering Abyss",
    "veil_of_echoes": "Veil of Echoes",
    "cathedral_of_ash": "Cathedral of Ash",
    "dreamer_in_black_fog": "Dreamer in the Black Fog",
    "obsidian_pulse": "Obsidian Pulse",
}

# Short labels used in compact displays.
BRAIN_KIND_SHORT_LABELS = {
    "whispering_abyss": "WA",
    "veil_of_echoes": "VE",
    "cathedral_of_ash": "CA",
    "dreamer_in_black_fog": "DB",
    "obsidian_pulse": "OP",
}

# Architecture metadata table.
# This is descriptive and used for documentation/UI/reporting alignment.
BRAIN_MLP_VARIANTS = {
    "whispering_abyss": {
        "family": "plain",
        "hidden": (96, 96),
    },
    "veil_of_echoes": {
        "family": "plain",
        "hidden": (128, 96, 64),
    },
    "cathedral_of_ash": {
        "family": "residual",
        "width": 80,
        "blocks": 3,
    },
    "dreamer_in_black_fog": {
        "family": "gated",
        "width": 80,
        "blocks": 2,
    },
    "obsidian_pulse": {
        "family": "bottleneck",
        "outer_width": 128,
        "inner_width": 48,
        "blocks": 2,
    },
}


# =============================================================================
# UI, VIEWER, INSPECTOR MODE, AND SCREEN RECORDING
# =============================================================================

# ENABLE_UI
# ---------
# False means pure headless mode.
# Use True for interactive visualization.
# Use False for maximum throughput or remote non-GUI environments.
ENABLE_UI: bool = _env_bool("FWS_UI", True)

# INSPECTOR_MODE
# --------------
# Documented values here:
#   "off"
#   "ui_no_output"
# Runtime also treats several aliases as equivalent to no-output inspector mode:
#   "inspect", "inspector", "no_output", "viewer_no_output"
#
# Use no-output inspector mode when you want to visually inspect a world or a
# resumed checkpoint without creating telemetry/results side effects.
INSPECTOR_MODE: str = _env_str("FWS_INSPECTOR_MODE", "off").strip().lower()

# Backward-compatible boolean switch for no-output inspector behavior.
INSPECTOR_UI_NO_OUTPUT: bool = _env_bool("FWS_INSPECTOR_UI_NO_OUTPUT", False)

# Viewer refresh cadences.
# Lower values refresh more often but cost more UI work.
VIEWER_STATE_REFRESH_EVERY: int = _env_int("FWS_VIEWER_STATE_REFRESH_EVERY", 3)
VIEWER_PICK_REFRESH_EVERY: int = _env_int("FWS_VIEWER_PICK_REFRESH_EVERY", 3)

# Font used by the UI.
UI_FONT_NAME: str = _env_str("FWS_UI_FONT", "consolas")

# Center the window on startup.
VIEWER_CENTER_WINDOW: bool = _env_bool("FWS_VIEWER_CENTER_WINDOW", True)

# Require the pygame-ce distribution when the UI is used.
PYGAME_CE_STRICT_RUNTIME: bool = _env_bool("FWS_PYGAME_CE_STRICT_RUNTIME", True)

# Bound the number of rendered text surfaces cached by the viewer.
VIEWER_TEXT_CACHE_MAX_SURFACES: int = _env_int("FWS_VIEWER_TEXT_CACHE_MAX_SURFACES", 2048)

# Relative export directory (inside run_dir) for saved agent brains.
VIEWER_BRAIN_EXPORT_DIRNAME: str = _env_str("FWS_VIEWER_BRAIN_EXPORT_DIRNAME", "exports/brains").strip()

# Renderer dimensions.
# Increase CELL_SIZE for a larger visual map.
# Increase HUD_WIDTH for more side-panel room.
CELL_SIZE: int = _env_int("FWS_CELL_SIZE", 5)
HUD_WIDTH: int = _env_int("FWS_HUD_W", 340)

# Target render FPS in UI mode.
TARGET_FPS: int = _env_int("FWS_TARGET_FPS", 60)

# Video recording settings.
# Enable only when you explicitly want captured output; recording costs storage
# and some runtime overhead.
RECORD_VIDEO: bool = _env_bool("FWS_RECORD_VIDEO", False)
VIDEO_FPS: int = _env_int("FWS_VIDEO_FPS", 60)
VIDEO_SCALE: int = _env_int("FWS_VIDEO_SCALE", 4)
VIDEO_EVERY_TICKS: int = _env_int("FWS_VIDEO_EVERY_TICKS", 1)

# UI color palette.
# Rendering-only aesthetics; changing these does not alter simulation logic.
UI_COLORS = {
    "bg": (15, 17, 22),
    "hud_bg": (10, 12, 16),
    "side_bg": (14, 16, 20),
    "grid": (35, 37, 42),
    "border": (80, 85, 95),
    "wall": (100, 105, 115),
    "empty": (20, 22, 28),
    "red_soldier": (240, 50, 50),
    "red_archer": (255, 120, 0),
    "red": (240, 50, 50),
    "blue_soldier": (30, 160, 255),
    "blue_archer": (0, 220, 180),
    "blue": (30, 160, 255),
    "archer_glyph": (255, 245, 120),
    "marker": (255, 255, 255),
    "text": (240, 240, 245),
    "text_dim": (160, 165, 175),
    "green": (50, 220, 120),
    "warn": (255, 170, 0),
    "bar_bg": (30, 35, 40),
    "bar_fg": (50, 220, 120),
    "graph_red": (240, 50, 50, 180),
    "graph_blue": (30, 160, 255, 180),
    "graph_grid": (50, 50, 60),
    "pause_text": (255, 200, 50),
}


# =============================================================================
# PROFILE OVERRIDE INJECTION
# =============================================================================
# Profiles are convenience presets. They do NOT override explicitly set env
# variables. They only replace hard-coded defaults that the operator did not
# override manually.
# =============================================================================


def _apply_profile_overrides() -> None:
    """Apply profile presets without overriding explicitly-set env vars."""
    if PROFILE == "default":
        return

    presets = {
        "debug": [
            ("FWS_GRID_W", "GRID_WIDTH", 80),
            ("FWS_GRID_H", "GRID_HEIGHT", 80),
            ("FWS_START_PER_TEAM", "START_AGENTS_PER_TEAM", 30),
            ("FWS_MAX_AGENTS", "MAX_AGENTS", 160),
            ("FWS_RAND_WALLS", "RANDOM_WALLS", 6),
            ("FWS_UI", "ENABLE_UI", True),
            ("FWS_TARGET_FPS", "TARGET_FPS", 30),
            ("FWS_RECORD_VIDEO", "RECORD_VIDEO", False),
            ("FWS_USE_VMAP", "USE_VMAP", False),
        ],
        "train_fast": [
            ("FWS_UI", "ENABLE_UI", False),
            ("FWS_USE_VMAP", "USE_VMAP", True),
            ("FWS_BRAIN_MLP_DMODEL", "BRAIN_MLP_D_MODEL", 24),
        ],
        "train_quality": [
            ("FWS_UI", "ENABLE_UI", False),
            ("FWS_USE_VMAP", "USE_VMAP", True),
            ("FWS_BRAIN_MLP_DMODEL", "BRAIN_MLP_D_MODEL", 40),
        ],
    }

    rows = presets.get(PROFILE)
    if not rows:
        return

    g = globals()
    for env_key, var_name, value in rows:
        if not _env_is_set(env_key):
            g[var_name] = value

    if int(g.get("BRAIN_MLP_D_MODEL", 0)) <= 0:
        raise ValueError(f"BRAIN_MLP_D_MODEL must be > 0 (got {g.get('BRAIN_MLP_D_MODEL')})")

    g["BRAIN_MLP_FINAL_INPUT_WIDTH"] = 2 * int(g["BRAIN_MLP_D_MODEL"])


# =============================================================================
# IMPORT-TIME VALIDATION
# =============================================================================
# The goal is fail-early correctness.
# Most validation is intentionally conservative: it checks that values make
# semantic sense without trying to guess user intent beyond that.
# =============================================================================


def _validate_config_invariants() -> None:
    """Validate important numeric ranges, enums, and schema relationships."""

    def _prob(name: str, value: float) -> None:
        try:
            x = float(value)
        except Exception:
            _config_issue(f"{name} is not numeric ({value!r})")
            return
        if not math.isfinite(x):
            _config_issue(f"{name} is not finite ({x!r})")
            return
        if x < 0.0 or x > 1.0:
            _config_issue(f"{name}={x} is outside [0, 1]")

    def _positive_int(name: str, value: int) -> None:
        try:
            x = int(value)
        except Exception:
            _config_issue(f"{name} is not int-like ({value!r})")
            return
        if x <= 0:
            _config_issue(f"{name} must be > 0 (got {x})")

    def _non_negative_int(name: str, value: int) -> None:
        try:
            x = int(value)
        except Exception:
            _config_issue(f"{name} is not int-like ({value!r})")
            return
        if x < 0:
            _config_issue(f"{name} must be >= 0 (got {x})")

    def _one_of(name: str, value: str, allowed: set[str]) -> None:
        if str(value) not in allowed:
            _config_issue(f"{name} must be one of {sorted(allowed)} (got {value!r})")

    def _non_negative_float(name: str, value: float) -> None:
        try:
            x = float(value)
        except Exception:
            _config_issue(f"{name} is not float-like ({value!r})")
            return
        if not math.isfinite(x):
            _config_issue(f"{name} is not finite ({x!r})")
            return
        if x < 0.0:
            _config_issue(f"{name} must be >= 0 (got {x})")

    # Basic world/capacity checks.
    _positive_int("GRID_WIDTH", GRID_WIDTH)
    _positive_int("GRID_HEIGHT", GRID_HEIGHT)
    _positive_int("MAX_AGENTS", MAX_AGENTS)
    _non_negative_int("START_AGENTS_PER_TEAM", START_AGENTS_PER_TEAM)

    total_initial_requested = int(START_AGENTS_PER_TEAM) * 2
    if total_initial_requested > int(MAX_AGENTS):
        _config_issue(
            f"Initial spawn request START_AGENTS_PER_TEAM*2={total_initial_requested} "
            f"exceeds MAX_AGENTS={int(MAX_AGENTS)}; runtime spawn may truncate."
        )

    # Common probabilities.
    _prob("MAP_WALL_STRAIGHT_PROB", MAP_WALL_STRAIGHT_PROB)
    _prob("MAP_WALL_GAP_PROB", MAP_WALL_GAP_PROB)
    _prob("SPAWN_ARCHER_RATIO", SPAWN_ARCHER_RATIO)
    _prob("RESPAWN_PROB_PER_DEAD", RESPAWN_PROB_PER_DEAD)
    _prob("RESPAWN_CLONE_PROB", RESPAWN_CLONE_PROB)
    _prob("RESPAWN_ARCHER_SHARE", RESPAWN_ARCHER_SHARE)
    _prob("RESPAWN_INTERIOR_BIAS", RESPAWN_INTERIOR_BIAS)
    _prob("RESPAWN_RARE_MUTATION_PHYSICAL_DRIFT_STD_FRAC", RESPAWN_RARE_MUTATION_PHYSICAL_DRIFT_STD_FRAC)
    _prob("RESPAWN_RARE_MUTATION_PHYSICAL_DRIFT_CLIP_FRAC", RESPAWN_RARE_MUTATION_PHYSICAL_DRIFT_CLIP_FRAC)
    _prob("RESPAWN_PARENT_SELECTION_TOPK_FRAC", RESPAWN_PARENT_SELECTION_TOPK_FRAC)
    _prob("MUTATION_FRACTION_ALIVE", MUTATION_FRACTION_ALIVE)
    _prob("TEAM_BRAIN_MIX_P_WHISPERING_ABYSS", TEAM_BRAIN_MIX_P_WHISPERING_ABYSS)
    _prob("TEAM_BRAIN_MIX_P_VEIL_OF_ECHOES", TEAM_BRAIN_MIX_P_VEIL_OF_ECHOES)
    _prob("TEAM_BRAIN_MIX_P_CATHEDRAL_OF_ASH", TEAM_BRAIN_MIX_P_CATHEDRAL_OF_ASH)
    _prob("TEAM_BRAIN_MIX_P_DREAMER_IN_BLACK_FOG", TEAM_BRAIN_MIX_P_DREAMER_IN_BLACK_FOG)
    _prob("TEAM_BRAIN_MIX_P_OBSIDIAN_PULSE", TEAM_BRAIN_MIX_P_OBSIDIAN_PULSE)
    _prob("TELEMETRY_MOVE_EVENTS_SAMPLE_RATE", TELEMETRY_MOVE_EVENTS_SAMPLE_RATE)
    _prob("CATASTROPHE_DYNAMIC_HEAL_OCCUPANCY_THRESHOLD", CATASTROPHE_DYNAMIC_HEAL_OCCUPANCY_THRESHOLD)
    _prob("CATASTROPHE_SMALL_SUPPRESS_FRACTION", CATASTROPHE_SMALL_SUPPRESS_FRACTION)
    _prob("CATASTROPHE_MEDIUM_SUPPRESS_FRACTION", CATASTROPHE_MEDIUM_SUPPRESS_FRACTION)
    _prob("CATASTROPHE_CLUSTER_SURVIVOR_FRACTION", CATASTROPHE_CLUSTER_SURVIVOR_FRACTION)

    # Positive integer style checks.
    _positive_int("CHECKPOINT_KEEP_LAST_N", max(1, CHECKPOINT_KEEP_LAST_N))
    _positive_int("BRAIN_MLP_D_MODEL", BRAIN_MLP_D_MODEL)
    _positive_int("BRAIN_MLP_FINAL_INPUT_WIDTH", BRAIN_MLP_FINAL_INPUT_WIDTH)
    _positive_int("RESPAWN_RARE_MUTATION_TICK_WINDOW_TICKS", RESPAWN_RARE_MUTATION_TICK_WINDOW_TICKS)
    _positive_int("RESPAWN_SPAWN_NEAR_PARENT_RADIUS", RESPAWN_SPAWN_NEAR_PARENT_RADIUS)

    # Non-negative integer style checks.
    for name, value in (
        ("AUTOSAVE_EVERY_SEC", AUTOSAVE_EVERY_SEC),
        ("CHECKPOINT_EVERY_TICKS", CHECKPOINT_EVERY_TICKS),
        ("HEADLESS_PRINT_EVERY_TICKS", HEADLESS_PRINT_EVERY_TICKS),
        ("TELEMETRY_TICK_SUMMARY_EVERY", TELEMETRY_TICK_SUMMARY_EVERY),
        ("TELEMETRY_TICK_METRICS_EVERY", TELEMETRY_TICK_METRICS_EVERY),
        ("TELEMETRY_SNAPSHOT_EVERY", TELEMETRY_SNAPSHOT_EVERY),
        ("TELEMETRY_REGISTRY_SNAPSHOT_EVERY", TELEMETRY_REGISTRY_SNAPSHOT_EVERY),
        ("TELEMETRY_VALIDATE_EVERY", TELEMETRY_VALIDATE_EVERY),
        ("TELEMETRY_PERIODIC_FLUSH_EVERY", TELEMETRY_PERIODIC_FLUSH_EVERY),
        ("TELEMETRY_EVENT_CHUNK_SIZE", TELEMETRY_EVENT_CHUNK_SIZE),
        ("TELEMETRY_TICK_CHUNK_SIZE", TELEMETRY_TICK_CHUNK_SIZE),
        ("TELEMETRY_MOVE_EVENTS_EVERY", TELEMETRY_MOVE_EVENTS_EVERY),
        ("TELEMETRY_MOVE_EVENTS_MAX_PER_TICK", TELEMETRY_MOVE_EVENTS_MAX_PER_TICK),
        ("TELEMETRY_COUNTERS_EVERY", TELEMETRY_COUNTERS_EVERY),
        ("VMAP_STACK_CACHE_MAX", VMAP_STACK_CACHE_MAX),
        ("TICK_LIMIT", TICK_LIMIT),
        ("TARGET_TPS", TARGET_TPS),
        ("RANDOM_WALLS", RANDOM_WALLS),
        ("ARCHER_RANGE", ARCHER_RANGE),
        ("INSTINCT_RADIUS", INSTINCT_RADIUS),
        ("RAY_TOKEN_COUNT", RAY_TOKEN_COUNT),
        ("NUM_ACTIONS", NUM_ACTIONS),
        ("RESP_FLOOR_PER_TEAM", RESP_FLOOR_PER_TEAM),
        ("RESP_MAX_PER_TICK", RESP_MAX_PER_TICK),
        ("RESP_PERIOD_TICKS", RESP_PERIOD_TICKS),
        ("RESP_PERIOD_BUDGET", RESP_PERIOD_BUDGET),
        ("RESP_HYST_COOLDOWN_TICKS", RESP_HYST_COOLDOWN_TICKS),
        ("RESP_WALL_MARGIN", RESP_WALL_MARGIN),
        ("RESPAWN_SPAWN_TRIES", RESPAWN_SPAWN_TRIES),
        ("RESPAWN_JITTER_RADIUS", RESPAWN_JITTER_RADIUS),
        ("RESPAWN_COOLDOWN_TICKS", RESPAWN_COOLDOWN_TICKS),
        ("RESPAWN_BATCH_PER_TEAM", RESPAWN_BATCH_PER_TEAM),
        ("RESPAWN_BIRTH_TOPK_SIZE", RESPAWN_BIRTH_TOPK_SIZE),
        ("PPO_WINDOW_TICKS", PPO_WINDOW_TICKS),
        ("PPO_LR_T_MAX", PPO_LR_T_MAX),
        ("PPO_EPOCHS", PPO_EPOCHS),
        ("PPO_MINIBATCHES", PPO_MINIBATCHES),
        ("PPO_UPDATE_TICKS", PPO_UPDATE_TICKS),
        ("MUTATION_PERIOD_TICKS", MUTATION_PERIOD_TICKS),
        ("CELL_SIZE", CELL_SIZE),
        ("HUD_WIDTH", HUD_WIDTH),
        ("TARGET_FPS", TARGET_FPS),
        ("VIEWER_TEXT_CACHE_MAX_SURFACES", VIEWER_TEXT_CACHE_MAX_SURFACES),
        ("VIDEO_FPS", VIDEO_FPS),
        ("VIDEO_SCALE", VIDEO_SCALE),
        ("VIDEO_EVERY_TICKS", VIDEO_EVERY_TICKS),
    ):
        _non_negative_int(name, value)

    if int(BRAIN_MLP_FINAL_INPUT_WIDTH) != 2 * int(BRAIN_MLP_D_MODEL):
        _config_issue(
            f"BRAIN_MLP_FINAL_INPUT_WIDTH must equal 2 * BRAIN_MLP_D_MODEL "
            f"(got {BRAIN_MLP_FINAL_INPUT_WIDTH} vs {2 * int(BRAIN_MLP_D_MODEL)})"
        )

    if float(RESPAWN_PARENT_SELECTION_SCORE_POWER) < 0.0:
        _config_issue(
            f"RESPAWN_PARENT_SELECTION_SCORE_POWER must be >= 0 "
            f"(got {RESPAWN_PARENT_SELECTION_SCORE_POWER})"
        )

    for name, value in (
        ("RESPAWN_BIRTH_BLEND_WEIGHT_KILL", RESPAWN_BIRTH_BLEND_WEIGHT_KILL),
        ("RESPAWN_BIRTH_BLEND_WEIGHT_CP", RESPAWN_BIRTH_BLEND_WEIGHT_CP),
        ("RESPAWN_BIRTH_BLEND_WEIGHT_HEALTH", RESPAWN_BIRTH_BLEND_WEIGHT_HEALTH),
        ("RESPAWN_BIRTH_BLEND_WEIGHT_PERSONAL", RESPAWN_BIRTH_BLEND_WEIGHT_PERSONAL),
    ):
        _non_negative_float(name, value)

    if (
        float(RESPAWN_BIRTH_BLEND_WEIGHT_KILL)
        + float(RESPAWN_BIRTH_BLEND_WEIGHT_CP)
        + float(RESPAWN_BIRTH_BLEND_WEIGHT_HEALTH)
        + float(RESPAWN_BIRTH_BLEND_WEIGHT_PERSONAL)
    ) <= 0.0:
        _config_issue("At least one RESPAWN_BIRTH_BLEND_WEIGHT_* value must be > 0")

    if float(RESPAWN_RARE_MUTATION_INHERITED_BRAIN_NOISE_STD) < 0.0:
        _config_issue(
            "RESPAWN_RARE_MUTATION_INHERITED_BRAIN_NOISE_STD must be >= 0 "
            f"(got {RESPAWN_RARE_MUTATION_INHERITED_BRAIN_NOISE_STD})"
        )

    # Enum-style checks.
    _one_of("PROFILE", PROFILE, {"default", "debug", "train_fast", "train_quality"})
    _one_of("SPAWN_MODE", SPAWN_MODE, {"uniform", "symmetric"})
    _one_of("RESPAWN_PARENT_SELECTION_MODE", RESPAWN_PARENT_SELECTION_MODE, {"random", "topk_weighted"})
    allowed_birth_doctrines = {
        "overall",
        "killer",
        "cp",
        "health",
        "kill_health",
        "health_cp",
        "kill_cp",
        "trinity",
        "highest_spike",
        "personal_points",
        "random_per_birth",
    }
    _one_of("RESPAWN_BIRTH_DOCTRINE_MODE", RESPAWN_BIRTH_DOCTRINE_MODE, allowed_birth_doctrines)
    _one_of(
        "RESPAWN_BIRTH_ZERO_SCORE_FALLBACK",
        RESPAWN_BIRTH_ZERO_SCORE_FALLBACK,
        {"uniform_candidates", "abort_birth"},
    )
    _one_of("RESPAWN_SPAWN_LOCATION_MODE", RESPAWN_SPAWN_LOCATION_MODE, {"uniform", "near_parent"})
    _one_of("CATASTROPHE_SCHEDULER_MODE", CATASTROPHE_SCHEDULER_MODE, {"periodic", "dynamic"})
    _one_of("BRAIN_MLP_ACTIVATION", BRAIN_MLP_ACTIVATION, {"gelu", "relu", "silu"})
    _one_of("BRAIN_MLP_NORM", BRAIN_MLP_NORM, {"layernorm", "none"})
    _one_of("BRAIN_MLP_RAY_SUMMARY", BRAIN_MLP_RAY_SUMMARY, {"mean"})
    _one_of("TELEMETRY_PPO_RICH_LEVEL", TELEMETRY_PPO_RICH_LEVEL, {"update", "epoch", "minibatch"})
    _one_of("TELEMETRY_DAMAGE_MODE", TELEMETRY_DAMAGE_MODE, {"victim_sum", "per_hit"})
    _one_of("TELEMETRY_EVENTS_FORMAT", TELEMETRY_EVENTS_FORMAT, {"jsonl", "csv"})
    _one_of("TELEMETRY_TICKS_FORMAT", TELEMETRY_TICKS_FORMAT, {"csv", "jsonl", "parquet", "npz"})
    _one_of("TELEMETRY_SNAPSHOT_FORMAT", TELEMETRY_SNAPSHOT_FORMAT, {"npz", "pt", "pickle"})
    _one_of("PPO_HP_REWARD_MODE", PPO_HP_REWARD_MODE, {"raw", "threshold_ramp"})

    # Brain-kind validation.
    allowed_brains = set(BRAIN_MLP_KIND_ORDER)
    if BRAIN_KIND not in allowed_brains:
        _config_issue(f"BRAIN_KIND must be one of {sorted(allowed_brains)} (got {BRAIN_KIND!r})")
    if TEAM_BRAIN_EXCLUSIVE_RED not in allowed_brains:
        _config_issue(
            f"TEAM_BRAIN_EXCLUSIVE_RED must be one of {sorted(allowed_brains)} "
            f"(got {TEAM_BRAIN_EXCLUSIVE_RED!r})"
        )
    if TEAM_BRAIN_EXCLUSIVE_BLUE not in allowed_brains:
        _config_issue(
            f"TEAM_BRAIN_EXCLUSIVE_BLUE must be one of {sorted(allowed_brains)} "
            f"(got {TEAM_BRAIN_EXCLUSIVE_BLUE!r})"
        )

    seq = tuple(str(x).strip().lower() for x in TEAM_BRAIN_MIX_SEQUENCE)
    if len(seq) == 0:
        _config_issue("TEAM_BRAIN_MIX_SEQUENCE must not be empty")
    else:
        bad = [x for x in seq if x not in allowed_brains]
        if bad:
            _config_issue(f"TEAM_BRAIN_MIX_SEQUENCE contains unknown kinds: {bad!r}")

    total_mix_prob = (
        float(TEAM_BRAIN_MIX_P_WHISPERING_ABYSS)
        + float(TEAM_BRAIN_MIX_P_VEIL_OF_ECHOES)
        + float(TEAM_BRAIN_MIX_P_CATHEDRAL_OF_ASH)
        + float(TEAM_BRAIN_MIX_P_DREAMER_IN_BLACK_FOG)
        + float(TEAM_BRAIN_MIX_P_OBSIDIAN_PULSE)
    )
    if total_mix_prob <= 0.0:
        _config_issue("At least one TEAM_BRAIN_MIX_P_* probability must be > 0")

    if len(RESPAWN_BIRTH_RANDOM_DOCTRINE_POOL) == 0:
        _config_issue("RESPAWN_BIRTH_RANDOM_DOCTRINE_POOL must not be empty")
    else:
        bad_birth_pool = [
            x
            for x in RESPAWN_BIRTH_RANDOM_DOCTRINE_POOL
            if x not in allowed_birth_doctrines or x == "random_per_birth"
        ]
        if bad_birth_pool:
            _config_issue(
                "RESPAWN_BIRTH_RANDOM_DOCTRINE_POOL contains invalid doctrine names "
                f"(or recursive random_per_birth): {bad_birth_pool!r}"
            )

    # Schema relationships.
    expected_rays_flat = int(RAY_TOKEN_COUNT) * int(RAY_FEAT_DIM)
    if int(RAYS_FLAT_DIM) != expected_rays_flat:
        _config_issue(
            f"RAYS_FLAT_DIM must equal RAY_TOKEN_COUNT * RAY_FEAT_DIM "
            f"(got {RAYS_FLAT_DIM} vs {expected_rays_flat})"
        )

    expected_rich_total = int(RICH_BASE_DIM) + int(INSTINCT_DIM)
    if int(RICH_TOTAL_DIM) != expected_rich_total:
        _config_issue(
            f"RICH_TOTAL_DIM must equal RICH_BASE_DIM + INSTINCT_DIM "
            f"(got {RICH_TOTAL_DIM} vs {expected_rich_total})"
        )

    expected_obs_dim = int(RAYS_FLAT_DIM) + int(RICH_TOTAL_DIM)
    if int(OBS_DIM) != expected_obs_dim:
        _config_issue(
            f"OBS_DIM must equal RAYS_FLAT_DIM + RICH_TOTAL_DIM "
            f"(got {OBS_DIM} vs {expected_obs_dim})"
        )

    if int(RAYCAST_MAX_STEPS) != max(max(VISION_RANGE_BY_UNIT.values()), 1):
        _config_issue("RAYCAST_MAX_STEPS is inconsistent with VISION_RANGE_BY_UNIT")


_apply_profile_overrides()
_validate_config_invariants()


# =============================================================================
# HUMAN-READABLE SUMMARY STRING
# =============================================================================

def summary_str() -> str:
    """Return a compact one-line summary of the active run configuration."""
    return (
        f"[Neural-Abyss] "
        f"device={DEVICE.type} "
        f"grid={GRID_WIDTH}x{GRID_HEIGHT} "
        f"start={START_AGENTS_PER_TEAM}/team "
        f"obs={OBS_DIM} actions={NUM_ACTIONS} "
        f"amp={'on' if AMP_ENABLED else 'off'}"
    )
