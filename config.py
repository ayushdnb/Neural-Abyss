from __future__ import annotations

import os
import math
import torch

# -----------------------------------------------------------------------------
# Internal config warnings buffer (used for import-time validation + parser warnings)
# -----------------------------------------------------------------------------
_CONFIG_WARNINGS = []

def _config_warn(msg: str) -> None:
    msg = str(msg)
    _CONFIG_WARNINGS.append(msg)
    print(f"[config][WARN] {msg}")

# =============================================================================
# 🛠️ ENV-PARSING UTILITIES (PRODUCTION-GRADE HELPERS)
# =============================================================================
# These helpers convert environment variables (strings) into typed values safely.
#
# Why this matters:
# - os.getenv returns strings (or None)
# - invalid env values should not crash a run unless explicitly desired
# - typed parsers centralize behavior and reduce bugs

def _env_bool(key: str, default: bool) -> bool:
    """
    Read an environment variable and interpret it as boolean.

    Accepted truthy strings (case-insensitive):
        "1", "true", "yes", "y", "on", "t"

    Accepted falsey strings (case-insensitive):
        "0", "false", "no", "n", "off", "f"

    If the variable is missing entirely, the provided default is returned.
    If the variable is present but not recognized, warn and fall back to default.

    Operational tip:
    - Use "1"/"0" or "true"/"false" consistently in scripts for readability.
    """
    v = os.getenv(key)
    if v is None:
        return bool(default)

    norm = v.strip().lower()
    if norm in ("1", "true", "yes", "y", "on", "t"):
        return True
    if norm in ("0", "false", "no", "n", "off", "f"):
        return False

    _config_warn(
        f"Unknown boolean env {key}={v!r}; using default={bool(default)}"
    )
    return bool(default)

def _env_float(key: str, default: float) -> float:
    """
    Read an environment variable as float.

    Behavior:
    - Missing => default
    - Invalid parse => default

    This "fail-soft" behavior is useful for long-running experiments where a malformed
    override should not crash the entire job unless strict validation is added elsewhere.
    """
    v = os.getenv(key)
    if v is None: return float(default)
    try: return float(v)
    except Exception: return float(default)

def _env_int(key: str, default: int) -> int:
    """
    Read an environment variable as int.

    Behavior:
    - Missing => default
    - Invalid parse => default

    Recommended usage:
    - Use for counts, periods, dimensions, toggles encoded as 0/1 (though bool parser is preferred for booleans).
    """
    v = os.getenv(key)
    if v is None: return int(default)
    try: return int(v)
    except Exception: return int(default)

def _env_str(key: str, default: str) -> str:
    """
    Read an environment variable as string.

    Behavior:
    - Missing => default
    - Present => exact string value (no validation here)

    Use cases:
    - paths, tags, profile names, format identifiers, modes.
    """
    v = os.getenv(key)
    return default if v is None else str(v)

def _env_is_set(key: str) -> bool:
    """
    Return True if the environment variable exists at all (even if empty string).

    Why this exists:
    - Sometimes config logic needs to know whether a user explicitly set a value,
      not just what the parsed fallback value is.
    - This is especially useful in precedence / compatibility logic.
    """
    return os.getenv(key) is not None


# -----------------------------------------------------------------------------
# Config validation / reproducibility helpers (opt-in strict mode)
# -----------------------------------------------------------------------------
# Env: FWS_CONFIG_STRICT
# When enabled, selected config warnings are escalated to ValueError during import.
CONFIG_STRICT: bool = _env_bool("FWS_CONFIG_STRICT", False)

def _config_issue(msg: str) -> None:
    """Warn by default; raise in strict mode."""
    if bool(CONFIG_STRICT):
        raise ValueError(f"[config] {msg}")
    _config_warn(msg)

def config_warnings():
    """Return a stable snapshot of config warnings collected during import."""
    return tuple(_CONFIG_WARNINGS)

def dump_config_dict() -> dict:
    """
    Return a JSON-friendly-ish snapshot of resolved config globals for telemetry/debugging.
    Only includes simple scalar values plus torch.device (stringified).
    """
    out = {}
    for k, v in globals().items():
        if k.startswith("_"):
            continue
        if k in {"os", "math", "torch"}:
            continue
        if callable(v):
            continue
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
        elif isinstance(v, torch.device):
            out[k] = str(v)
    out["CONFIG_WARNINGS"] = list(_CONFIG_WARNINGS)
    return out

# =============================================================================
# 🧪 EXPERIMENT TRACKING & SEEDING
# =============================================================================
# These knobs define experiment identity, reproducibility, and checkpoint resumption.
# They do not directly change the simulation physics or PPO math, but they are critical
# for run management and reproducibility.

# Profile name used later for macro-overrides (e.g., debug/train_fast/train_quality).
# Env: FWS_PROFILE
# Typical values: "default", "debug", "train_fast", "train_quality"
# How to check effect: print PROFILE or inspect summary/log startup prints.
PROFILE: str = _env_str("FWS_PROFILE", "default").strip().lower()

# Free-form tag for organizing experiments / output folders / telemetry labels.
# Env: FWS_EXPERIMENT_TAG
# Example values: "overnight_cp_pressure_test", "wallhug_fix_v3"
# This is metadata only unless downstream code uses it for folder naming/report labels.
EXPERIMENT_TAG: str = _env_str("FWS_EXPERIMENT_TAG", "god_level_run").strip()

# Global RNG seed (used by codepaths that consume this config value).
# Env: FWS_SEED
# Lower-level libraries (PyTorch/CUDA) may need explicit seeding elsewhere too.
# If reproducibility matters, also log CUDA determinism settings in runtime.
RNG_SEED: int = _env_int("FWS_SEED", 42)

# Backward-compatibility alias for older runtime/telemetry codepaths.
# Keep both names to avoid silent metadata nulls or branch-specific breakage.
SEED: int = int(RNG_SEED)

# Base results directory for run outputs (checkpoints, telemetry, reports).
# Env: FWS_RESULTS_DIR
# Keep on SSD/NVMe if possible for better telemetry/checkpoint throughput.
RESULTS_DIR: str = _env_str("FWS_RESULTS_DIR", "results").strip()

# Path to checkpoint to resume from.
# Env: FWS_CHECKPOINT_PATH
# Empty string means "start fresh".
# Validation/loading is typically handled in main/runtime, not here.
CHECKPOINT_PATH: str = _env_str("FWS_CHECKPOINT_PATH", "").strip()

# Resume output continuity: when resuming from a checkpoint, reuse the original run folder
# (same stats/telemetry/checkpoints lineage) instead of creating a new results folder.
# Env: FWS_RESUME_OUTPUT_CONTINUITY
RESUME_OUTPUT_CONTINUITY: bool = _env_bool("FWS_RESUME_OUTPUT_CONTINUITY", True)

# Force a new run folder even when CHECKPOINT_PATH is set (overrides continuity).
# Env: FWS_RESUME_FORCE_NEW_RUN
RESUME_FORCE_NEW_RUN: bool = _env_bool("FWS_RESUME_FORCE_NEW_RUN", False)

# Strict append schema guard for append-mode CSV streams where headers are expected stable.
# When enabled, append writers may fail-fast on header mismatch instead of silently corrupting rows.
# Env: FWS_RESUME_APPEND_STRICT_CSV_SCHEMA
RESUME_APPEND_STRICT_CSV_SCHEMA: bool = _env_bool("FWS_RESUME_APPEND_STRICT_CSV_SCHEMA", True)

# Autosave interval in seconds (UI mode usage note mentioned in your original comments).
# Env: FWS_AUTOSAVE_EVERY_SEC
# 0 or negative behavior depends on downstream code (not enforced here).
# Larger value => less I/O overhead, more risk of progress loss on crash.
AUTOSAVE_EVERY_SEC = _env_int("FWS_AUTOSAVE_EVERY_SEC", 3600)

# ----------------------------------------------------------------------
# Tick-based checkpointing (works in BOTH UI and headless)
# ----------------------------------------------------------------------
# These control checkpoint cadence and retention in tick units.
# Tick-based scheduling is often more stable than wall-clock in variable-TPS runs.

# Save a checkpoint every N ticks.
# Env: FWS_CHECKPOINT_EVERY_TICKS
# 0 disables periodic tick checkpoints.
# Larger => faster run (less I/O), but fewer recovery points.
# Smaller => safer recovery, but more serialization overhead.
CHECKPOINT_EVERY_TICKS = _env_int("FWS_CHECKPOINT_EVERY_TICKS", 50000)  # 0 disables

# Save a checkpoint when process exits cleanly (if runtime honors this flag).
# Env: FWS_CHECKPOINT_ON_EXIT
# Useful for manual stops / Ctrl+C workflows.
CHECKPOINT_ON_EXIT = _env_bool("FWS_CHECKPOINT_ON_EXIT", True)

# Retain only latest N non-pinned checkpoints to limit disk growth.
# Env: FWS_CHECKPOINT_KEEP_LAST_N
# 1 = minimal disk usage, little rollback history.
# Higher values = more forensic/debug capability, more disk use.
CHECKPOINT_KEEP_LAST_N = _env_int("FWS_CHECKPOINT_KEEP_LAST_N", 1)  # keep latest N (non-pinned)

# Whether manually-triggered checkpoints should be "pinned" (protected from auto-pruning).
# Env: FWS_CHECKPOINT_PIN_ON_MANUAL
CHECKPOINT_PIN_ON_MANUAL = _env_bool("FWS_CHECKPOINT_PIN_ON_MANUAL", True)

# Tag applied to pinned/manual checkpoints for naming/identification.
# Env: FWS_CHECKPOINT_PIN_TAG
CHECKPOINT_PIN_TAG = _env_str("FWS_CHECKPOINT_PIN_TAG", "manual")

# Filesystem trigger filename (watched by runtime, if implemented there).
# Env: FWS_CHECKPOINT_TRIGGER_FILE
# Typical pattern: create this file to request a checkpoint without process interaction.
CHECKPOINT_TRIGGER_FILE = _env_str("FWS_CHECKPOINT_TRIGGER_FILE", "checkpoint.now")

# ----------------------------------------------------------------------
# Headless console reporting
# ----------------------------------------------------------------------
# Console I/O can become a real bottleneck in high-TPS loops, especially on Windows terminals.
# These knobs help trade observability vs throughput.

# Print headless status every N ticks.
# Env: FWS_HEADLESS_PRINT_EVERY_TICKS
# 0 disables headless tick prints.
# For overnight speed, larger intervals usually improve TPS.
HEADLESS_PRINT_EVERY_TICKS = _env_int("FWS_HEADLESS_PRINT_EVERY_TICKS", 1000)  # 0 disables

# Print verbosity level for headless mode (downstream runtime decides exact behavior).
# Env: FWS_HEADLESS_PRINT_LEVEL
# Suggested interpretation (as documented): 0=min, 1=std, 2=detail
HEADLESS_PRINT_LEVEL = _env_int("FWS_HEADLESS_PRINT_LEVEL", 2)               # 0=min,1=std,2=detail

# Whether to include GPU info in headless prints (can add subprocess/API overhead depending on impl).
# Env: FWS_HEADLESS_PRINT_GPU
# If TPS is critical and GPU telemetry polling is expensive, set False.
HEADLESS_PRINT_GPU = _env_bool("FWS_HEADLESS_PRINT_GPU", False)

# =============================================================================
# 📡 SCIENTIFIC RECORDING / TELEMETRY (NON-INVASIVE, CONFIG-FIRST)
# =============================================================================
# These knobs govern telemetry collection, sampling frequency, buffering, and report generation.
#
# Practical truth:
# - Telemetry can dominate runtime cost if event rates are high.
# - For long runs, chunking/buffering choices matter as much as model size.

# Master telemetry switch.
# Env: FWS_TELEMETRY
# False can significantly improve throughput if telemetry is heavy.
# If disabled, ensure you still have enough observability for debugging.
TELEMETRY_ENABLED: bool = _env_bool("FWS_TELEMETRY", True)

# Optional telemetry tag for file naming / report grouping.
# Env: FWS_TELEMETRY_TAG
TELEMETRY_TAG: str = _env_str("FWS_TELEMETRY_TAG", "").strip()

# Telemetry schema version string (metadata).
# Env: FWS_TELEM_SCHEMA
# Useful for parsers/reports to interpret file layout.
TELEMETRY_SCHEMA_VERSION: str = _env_str("FWS_TELEM_SCHEMA", "2").strip()

# Write static run metadata once (strongly recommended for reproducibility).
# Env: FWS_TELEM_RUN_META
TELEMETRY_WRITE_RUN_META: bool = _env_bool("FWS_TELEM_RUN_META", True)

# Write static per-agent info (can be expensive / large depending on implementation).
# Env: FWS_TELEM_AGENT_STATIC
# False is a good default when optimizing throughput.
TELEMETRY_WRITE_AGENT_STATIC: bool = _env_bool("FWS_TELEM_AGENT_STATIC", True)

# Emit summary ticks every N ticks (coarse aggregate summaries).
# Env: FWS_TELEM_TICK_SUMMARY_EVERY
# Larger => less I/O, less granularity.
TELEMETRY_TICK_SUMMARY_EVERY: int = _env_int("FWS_TELEM_TICK_SUMMARY_EVERY", 200)

# --- Frequencies (ticks) ---
# These determine how often various telemetry streams are sampled or flushed.

# Per-tick metrics sampling period.
# Env: FWS_TELEM_TICK_EVERY
# Smaller => richer time series, more overhead.
TELEMETRY_TICK_METRICS_EVERY: int = _env_int("FWS_TELEM_TICK_EVERY", 200)

# World snapshot period (likely state snapshots / compressed arrays).
# Env: FWS_TELEM_SNAPSHOT_EVERY
# Larger intervals reduce disk usage substantially.
TELEMETRY_SNAPSHOT_EVERY: int = _env_int("FWS_TELEM_SNAPSHOT_EVERY", 500)

# Registry snapshot period (e.g., agent registry/system state snapshot).
# Env: FWS_TELEM_REG_EVERY
TELEMETRY_REGISTRY_SNAPSHOT_EVERY: int = _env_int("FWS_TELEM_REG_EVERY", 200)

# Telemetry validation cadence.
# Env: FWS_TELEM_VALIDATE_EVERY
# Useful for catching anomalies; reduce frequency if performance constrained.
TELEMETRY_VALIDATE_EVERY: int = _env_int("FWS_TELEM_VALIDATE_EVERY", 500)

# Periodic flush cadence for telemetry buffers.
# Env: FWS_TELEM_FLUSH_EVERY
# More frequent flush => safer on crash, more I/O overhead.
TELEMETRY_PERIODIC_FLUSH_EVERY: int = _env_int("FWS_TELEM_FLUSH_EVERY", 1000)

# --- Buffers / chunk sizes ---
# Chunk sizes control write batching and memory usage.
# Larger chunks usually improve throughput but increase memory and crash-loss window.

# Event log chunk size before write/rollover.
# Env: FWS_TELEM_EVENT_CHUNK
TELEMETRY_EVENT_CHUNK_SIZE: int = _env_int("FWS_TELEM_EVENT_CHUNK", 200000)

# Tick metrics chunk size before write/rollover.
# Env: FWS_TELEM_TICK_CHUNK
TELEMETRY_TICK_CHUNK_SIZE: int = _env_int("FWS_TELEM_TICK_CHUNK", 20000)

# --- Event category toggles ---
# Disable high-volume categories first when optimizing throughput.

# Birth event logging (agent spawns/reinforcements).
# Env: FWS_TELEM_BIRTHS
TELEMETRY_LOG_BIRTHS: bool = _env_bool("FWS_TELEM_BIRTHS", True)

# Death event logging.
# Env: FWS_TELEM_DEATHS
TELEMETRY_LOG_DEATHS: bool = _env_bool("FWS_TELEM_DEATHS", True)

# Damage event logging (can become very high-rate).
# Env: FWS_TELEM_DAMAGE
TELEMETRY_LOG_DAMAGE: bool = _env_bool("FWS_TELEM_DAMAGE", True)

# Kill event logging.
# Env: FWS_TELEM_KILLS
TELEMETRY_LOG_KILLS: bool = _env_bool("FWS_TELEM_KILLS", True)

# Move event logging (typically the largest event stream if per-move).
# Env: FWS_TELEM_MOVES
TELEMETRY_LOG_MOVES: bool = _env_bool("FWS_TELEM_MOVES", True)   # can be huge if you emit per-move events

# PPO telemetry logging (losses, KL, entropy, etc., depending on implementation).
# Env: FWS_TELEM_PPO
TELEMETRY_LOG_PPO: bool = _env_bool("FWS_TELEM_PPO", True)

# Dedicated rich PPO training diagnostics CSV (separate from headless summary).
# File: telemetry/ppo_training_telemetry.csv
# Env: FWS_TELEM_PPO_RICH_CSV
TELEMETRY_PPO_RICH_CSV: bool = _env_bool("FWS_TELEM_PPO_RICH_CSV", True)

# Rich PPO telemetry write granularity: "update", "epoch", or "minibatch".
# Higher detail gives better diagnostics but increases I/O/CSV size.
# Env: FWS_TELEM_PPO_RICH_LEVEL
TELEMETRY_PPO_RICH_LEVEL: str = _env_str("FWS_TELEM_PPO_RICH_LEVEL", "update").strip().lower()

# Flush rich PPO telemetry CSV every N appended rows (1 = flush every row batch append).
# Env: FWS_TELEM_PPO_RICH_FLUSH_EVERY
TELEMETRY_PPO_RICH_FLUSH_EVERY: int = _env_int("FWS_TELEM_PPO_RICH_FLUSH_EVERY", 1)

# Fail-fast on telemetry CSV header mismatch when appending to an existing file.
# Applies to telemetry append helpers (including rich PPO CSV).
# Env: FWS_TELEM_APPEND_SCHEMA_STRICT
TELEMETRY_APPEND_SCHEMA_STRICT: bool = _env_bool("FWS_TELEM_APPEND_SCHEMA_STRICT", True)

# ----------------------------------------------------------------------
# Headless live CSV summary sidecar (additive; does NOT affect console prints)
# ----------------------------------------------------------------------
# Uses existing TELEMETRY_TICK_SUMMARY_EVERY for row cadence and existing
# TELEMETRY_PERIODIC_FLUSH_EVERY for buffered streams. This sidecar writes rows
# in append mode (file close per append), so rows are "live" without extra deps.

# Master switch for additive headless live summary CSV (telemetry_summary.csv).
# Env: FWS_TELEM_HEADLESS_SUMMARY
TELEMETRY_HEADLESS_LIVE_SUMMARY: bool = _env_bool("FWS_TELEM_HEADLESS_SUMMARY", True)

# Include wall-time/TPS columns in headless live summary.
# Env: FWS_TELEM_SUMMARY_TPS
TELEMETRY_HEADLESS_SUMMARY_INCLUDE_TPS: bool = _env_bool("FWS_TELEM_SUMMARY_TPS", True)

# Include parsed GPU probe columns (util/mem/power) in headless live summary.
# GPU polling is performed only on summary ticks by the headless loop.
# Env: FWS_TELEM_SUMMARY_GPU
TELEMETRY_HEADLESS_SUMMARY_INCLUDE_GPU: bool = _env_bool("FWS_TELEM_SUMMARY_GPU", True)

# Include TickMetrics window aggregates (moved/attacks/deaths/cp deltas/etc.).
# Env: FWS_TELEM_SUMMARY_TICK_METRICS
TELEMETRY_HEADLESS_SUMMARY_INCLUDE_TICK_METRICS: bool = _env_bool("FWS_TELEM_SUMMARY_TICK_METRICS", True)

# Include latest cached PPO train summary columns (if PPO runtime exposes them).
# This is observational only and remains blank when PPO is disabled/unavailable.
# Env: FWS_TELEM_SUMMARY_PPO
TELEMETRY_HEADLESS_SUMMARY_INCLUDE_PPO: bool = _env_bool("FWS_TELEM_SUMMARY_PPO", True)

# Log rare respawn mutation events to telemetry/mutation_events.csv (append).
# Env: FWS_TELEM_MUTATIONS
TELEMETRY_LOG_RARE_MUTATIONS: bool = _env_bool("FWS_TELEM_MUTATIONS", True)
# Move event sampling controls:
# These are critical for keeping logs manageable in large populations.

# Emit per-move events every N ticks.
# Env: FWS_TELEM_MOVE_EVERY
# 0 = disable per-move event emission entirely (aggregates only).
# This is a major throughput/disk saver.
TELEMETRY_MOVE_EVENTS_EVERY: int = _env_int("FWS_TELEM_MOVE_EVERY", 100)        # 0=off (aggregates only)

# Max number of move events emitted per tick when move sampling is enabled.
# Env: FWS_TELEM_MOVE_MAX
# Helps cap worst-case event volume bursts.
TELEMETRY_MOVE_EVENTS_MAX_PER_TICK: int = _env_int("FWS_TELEM_MOVE_MAX", 256)

# Probability sampling rate for move events.
# Env: FWS_TELEM_MOVE_RATE
# Range typically [0.0, 1.0]
# - 1.0 = keep all sampled candidates
# - 0.1 = keep ~10%
TELEMETRY_MOVE_EVENTS_SAMPLE_RATE: float = _env_float("FWS_TELEM_MOVE_RATE", 0.4)

# Optional counters emission cadence.
# Env: FWS_TELEM_COUNTERS_EVERY
# 0 typically means disabled (depends on downstream runtime).
TELEMETRY_COUNTERS_EVERY: int = _env_int("FWS_TELEM_COUNTERS_EVERY", 500)

# Damage aggregation mode string (parser/runtime-defined semantics).
# Env: FWS_TELEM_DMG_MODE
# Example default: "victim_sum"
TELEMETRY_DAMAGE_MODE: str = _env_str("FWS_TELEM_DMG_MODE", "victim_sum").strip().lower()

# Events file format.
# Env: FWS_TELEM_EVENTS_FMT
# Common values might include "jsonl", "csv", etc. (runtime-dependent)
TELEMETRY_EVENTS_FORMAT: str = _env_str("FWS_TELEM_EVENTS_FMT", "jsonl").strip().lower()

# Whether to gzip-compress telemetry event stream files (if runtime/writer supports it).
# Env: FWS_TELEM_EVENTS_GZIP
TELEMETRY_EVENTS_GZIP: bool = _env_bool("FWS_TELEM_EVENTS_GZIP", False)

# Tick metrics file format.
# Env: FWS_TELEM_TICKS_FMT
TELEMETRY_TICKS_FORMAT: str = _env_str("FWS_TELEM_TICKS_FMT", "csv").strip().lower()

# Snapshot file format.
# Env: FWS_TELEM_SNAP_FMT
# npz is a good default for numeric arrays with compression support.
TELEMETRY_SNAPSHOT_FORMAT: str = _env_str("FWS_TELEM_SNAP_FMT", "npz").strip().lower()

# Telemetry validation strictness level (implementation-defined).
# Env: FWS_TELEM_VALIDATE
# Common pattern: 0=off, 1=basic, 2+=stricter
TELEMETRY_VALIDATE_LEVEL: int = _env_int("FWS_TELEM_VALIDATE", 1)

# Abort run on telemetry anomaly.
# Env: FWS_TELEM_ABORT
# False is safer for long exploration runs; True is safer for correctness-critical runs.
TELEMETRY_ABORT_ON_ANOMALY: bool = _env_bool("FWS_TELEM_ABORT", False)

# Generate end-of-run telemetry report artifacts.
# Env: FWS_TELEM_REPORT
TELEMETRY_REPORT_ENABLE: bool = _env_bool("FWS_TELEM_REPORT", True)

# Generate Excel export (can be slow/large on long runs).
# Env: FWS_TELEM_EXCEL
TELEMETRY_REPORT_EXCEL: bool = _env_bool("FWS_TELEM_EXCEL", False)  # Excel export can be slow at end of long run

# Generate PNG plots in report.
# Env: FWS_TELEM_PNG
TELEMETRY_REPORT_PNG: bool = _env_bool("FWS_TELEM_PNG", True)

# =============================================================================
# 💻 HARDWARE ACCELERATION & TENSOR COMPILER
# =============================================================================
# These knobs select device and mixed precision behavior.
# They have major performance and numerical stability implications.

# Master CUDA usage flag, AND gated by actual CUDA availability.
# Env: FWS_CUDA
# If FWS_CUDA=1 but torch.cuda.is_available() is False, CPU is used safely.
USE_CUDA = _env_bool("FWS_CUDA", True) and torch.cuda.is_available()

# Canonical torch device object used by the codebase.
# DEVICE / TORCH_DEVICE aliases allow flexible imports in other modules.
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
TORCH_DEVICE = DEVICE

# Automatic Mixed Precision toggle.
# Env: FWS_AMP
# On CUDA, this can significantly improve throughput and reduce memory use.
# If instability occurs, first try disabling AMP.
AMP_ENABLED = _env_bool("FWS_AMP", True)

def amp_enabled() -> bool:
    """
    Lightweight accessor used by codepaths that prefer a function over direct constant import.
    """
    return AMP_ENABLED

# Default tensor dtype for configured runtime paths.
# fp16 when CUDA+AMP enabled, otherwise fp32.
# Note: individual modules may still cast selectively for stability.
TORCH_DTYPE = torch.float16 if (USE_CUDA and AMP_ENABLED) else torch.float32

# VMAP controls (torch.func-based batching over independent models)
# ---------------------------------------------------------------
# Useful when many per-agent models are evaluated and Python overhead is significant.

# Master switch for vmap path (if supported by runtime and torch build).
# Env: FWS_USE_VMAP
USE_VMAP = _env_bool("FWS_USE_VMAP", True)

# Minimum bucket size before attempting vmap (below this, overhead may dominate).
# Env: FWS_VMAP_MIN_BUCKET
# Increase if vmap overhead hurts small buckets; decrease if buckets are usually moderate.
VMAP_MIN_BUCKET = _env_int("FWS_VMAP_MIN_BUCKET", 8)

# Debug prints for vmap fallback/diagnostics.
# Env: FWS_VMAP_DEBUG
VMAP_DEBUG = _env_bool("FWS_VMAP_DEBUG", True)

# =============================================================================
# 🌍 WORLD SCALE & MEMORY ALLOCATION
# =============================================================================
# These knobs define map size and population bounds. They strongly influence:
# - compute cost (more agents + larger map = more work),
# - memory usage,
# - emergent density/contact rates.

# Grid width in cells.
# Env: FWS_GRID_W
# Larger maps reduce encounter frequency at fixed population and may increase pathing/raycast work.
GRID_WIDTH  = _env_int("FWS_GRID_W", 100)

# Grid height in cells.
# Env: FWS_GRID_H
GRID_HEIGHT = _env_int("FWS_GRID_H", 100)

# Initial starting agents per team.
# Env: FWS_START_PER_TEAM
# Higher => more immediate pressure/contact, more compute.
# Ensure MAX_AGENTS can accommodate both teams + respawn dynamics.
START_AGENTS_PER_TEAM = _env_int("FWS_START_PER_TEAM", 150)

# Maximum total agent slots/capacity (global).
# Env: FWS_MAX_AGENTS
# If too low, respawn may be slot-constrained. If too high, memory/compute increases.
MAX_AGENTS  = _env_int("FWS_MAX_AGENTS", 400)

# Hard tick limit for run termination.
# Env: FWS_TICK_LIMIT
# 0 commonly means "run indefinitely" (runtime-defined but consistent with your usage).
TICK_LIMIT = _env_int("FWS_TICK_LIMIT", 0)

# Optional target TPS throttling.
# Env: FWS_TARGET_TPS
# 0 typically means unthrottled / max speed.
TARGET_TPS = _env_int("FWS_TARGET_TPS", 0)

# Agent feature schema size (static contract).
# DO NOT CHANGE unless the engine + serialization + consumers are updated together.
AGENT_FEATURES = 10  # DO NOT CHANGE (schema contract)

# =============================================================================
# 🗺️ TOPOGRAPHY & STRATEGIC OBJECTIVES
# =============================================================================
# Map topology and zone generation strongly shape emergent strategy.
# When diagnosing "AI behavior", always consider topology as a confounder.

# Number of random wall segments.
# Env: FWS_RAND_WALLS
# More walls => more choke points and possible partitioning.
# Fewer walls => cleaner topology for behavior debugging.
RANDOM_WALLS = _env_int("FWS_RAND_WALLS", 12)

# Minimum wall segment length.
# Env: FWS_WALL_SEG_MIN
# Larger minimum tends to create longer barriers/chokes.
WALL_SEG_MIN = _env_int("FWS_WALL_SEG_MIN", 5)

# Maximum wall segment length.
# Env: FWS_WALL_SEG_MAX
# If very large relative to grid size, risk of accidental partitioning increases.
WALL_SEG_MAX = _env_int("FWS_WALL_SEG_MAX", 77)

# Margin from boundaries or protected areas for wall placement (implementation-dependent exact usage).
# Env: FWS_WALL_MARGIN
# Increasing can reduce corner-locking geometry.
WALL_AVOID_MARGIN = _env_int("FWS_WALL_MARGIN", 3)

# Probability that wall generation continues straight instead of turning.
# Env: FWS_MAP_WALL_STRAIGHT_PROB
# Higher => straighter, longer directional walls.
MAP_WALL_STRAIGHT_PROB = _env_float("FWS_MAP_WALL_STRAIGHT_PROB", 0.65)

# Probability of inserting a gap in walls.
# Env: FWS_MAP_WALL_GAP_PROB
# Higher => more permeability / connectivity.
# Useful anti-stalemate lever when walls over-partition the map.
MAP_WALL_GAP_PROB      = _env_float("FWS_MAP_WALL_GAP_PROB", 0.20)

# Heal Zones ("water holes")
# -------------------------
# Placement assumptions are documented in your comments (uniform interior sampling; no center bias).
# These knobs strongly affect whether teams can maintain private safe sustain regions.

# Number of heal zones.
# Env: FWS_HEAL_COUNT
# More small zones often increase probability of uncontested camping.
# Fewer zones (especially 1) tends to force shared contention.
HEAL_ZONE_COUNT      = _env_int("FWS_HEAL_COUNT", 11)

# Heal zone size as ratio (engine interprets exact geometry).
# Env: FWS_HEAL_SIZE_RATIO
# Larger ratios make zones easier to find/reach and more likely to overlap strategic routes.
HEAL_ZONE_SIZE_RATIO = _env_float("FWS_HEAL_SIZE_RATIO", 5/64)

# HP healed per tick on heal tiles.
# Env: FWS_HEAL_RATE
# Compare against metabolism to understand net sustain:
#   net ≈ HEAL_RATE - META_*_HP_PER_TICK
HEAL_RATE            = _env_float("FWS_HEAL_RATE", 0.003)

# -----------------------------------------------------------------------------
# Catastrophe runtime framework (final reconciliation patch)
# -----------------------------------------------------------------------------
CATASTROPHE_FRAMEWORK_VERSION: str = "runtime_override_v2"
CATASTROPHE_RUNTIME_RESOLUTION: str = "base_plus_override_apply_mask_v1"
CATASTROPHE_SCHEDULER_VERSION: str = "pressure_hazard_v1"

# Master catastrophe toggles.
CATASTROPHE_ENABLED = _env_bool("FWS_CATASTROPHE_ENABLED", True)
CATASTROPHE_DYNAMIC_SCHEDULER_ENABLED = _env_bool("FWS_CATASTROPHE_DYNAMIC_SCHEDULER_ENABLED", True)
CATASTROPHE_ALLOW_OVERLAP = _env_bool("FWS_CATASTROPHE_ALLOW_OVERLAP", False)
CATASTROPHE_OVERRIDE_LOCKS_EDIT_MASK = _env_bool("FWS_CATASTROPHE_OVERRIDE_LOCKS_EDIT_MASK", True)
CATASTROPHE_MANUAL_REPLACE_EXISTING = _env_bool("FWS_CATASTROPHE_MANUAL_REPLACE_EXISTING", True)
CATASTROPHE_CLEAR_ENABLED = _env_bool("FWS_CATASTROPHE_CLEAR_ENABLED", True)

# Shared catastrophe timing / damage knobs.
CATASTROPHE_DEFAULT_DURATION_TICKS = _env_int("FWS_CATASTROPHE_DEFAULT_DURATION_TICKS", 9000)
CATASTROPHE_NEGATIVE_ZONE_DAMAGE_RATE = _env_float("FWS_CATASTROPHE_NEGATIVE_ZONE_DAMAGE_RATE", HEAL_RATE)
CATASTROPHE_SCHEDULER_SEED_OFFSET = _env_int("FWS_CATASTROPHE_SCHEDULER_SEED_OFFSET", 7919)
CATASTROPHE_DYNAMIC_COOLDOWN_TICKS = _env_int("FWS_CATASTROPHE_DYNAMIC_COOLDOWN_TICKS", 2600)
CATASTROPHE_DYNAMIC_MIN_INTERVAL_TICKS = _env_int("FWS_CATASTROPHE_DYNAMIC_MIN_INTERVAL_TICKS", 1800)
CATASTROPHE_DYNAMIC_MAX_INTERVAL_TICKS = _env_int("FWS_CATASTROPHE_DYNAMIC_MAX_INTERVAL_TICKS", 9000)

# Hazard-law pressure accumulation.
CATASTROPHE_DYNAMIC_PRESSURE_BASE_PER_TICK = _env_float("FWS_CATASTROPHE_DYNAMIC_PRESSURE_BASE_PER_TICK", 0.00016)
CATASTROPHE_DYNAMIC_PRESSURE_NONZERO_GAIN = _env_float("FWS_CATASTROPHE_DYNAMIC_PRESSURE_NONZERO_GAIN", 0.00045)
CATASTROPHE_DYNAMIC_PRESSURE_POSITIVE_GAIN = _env_float("FWS_CATASTROPHE_DYNAMIC_PRESSURE_POSITIVE_GAIN", 0.00020)
CATASTROPHE_DYNAMIC_PRESSURE_NEGATIVE_GAIN = _env_float("FWS_CATASTROPHE_DYNAMIC_PRESSURE_NEGATIVE_GAIN", 0.00010)
CATASTROPHE_DYNAMIC_PRESSURE_MEAN_ABS_GAIN = _env_float("FWS_CATASTROPHE_DYNAMIC_PRESSURE_MEAN_ABS_GAIN", 0.00035)
CATASTROPHE_DYNAMIC_PRESSURE_CAP = _env_float("FWS_CATASTROPHE_DYNAMIC_PRESSURE_CAP", 6.0)
CATASTROPHE_DYNAMIC_PRESSURE_TO_PROB_GAIN = _env_float("FWS_CATASTROPHE_DYNAMIC_PRESSURE_TO_PROB_GAIN", 0.08)
CATASTROPHE_DYNAMIC_BASE_HAZARD_PROB = _env_float("FWS_CATASTROPHE_DYNAMIC_BASE_HAZARD_PROB", 0.001)
CATASTROPHE_DYNAMIC_MAX_HAZARD_PROB = _env_float("FWS_CATASTROPHE_DYNAMIC_MAX_HAZARD_PROB", 0.18)
CATASTROPHE_DYNAMIC_PRESSURE_COOLDOWN_DECAY = _env_float("FWS_CATASTROPHE_DYNAMIC_PRESSURE_COOLDOWN_DECAY", 0.15)

# Safe default catastrophe strengths.
CATASTROPHE_GLOBAL_ATTENUATION_FACTOR = _env_float("FWS_CATASTROPHE_GLOBAL_ATTENUATION_FACTOR", 0.35)
CATASTROPHE_REGIONAL_ATTENUATION_FACTOR = _env_float("FWS_CATASTROPHE_REGIONAL_ATTENUATION_FACTOR", 0.25)
CATASTROPHE_POLARITY_SPLIT_MAGNITUDE_SCALE = _env_float("FWS_CATASTROPHE_POLARITY_SPLIT_MAGNITUDE_SCALE", 0.75)
CATASTROPHE_POSITIVE_BAND_DORMANCY_FRACTION = _env_float("FWS_CATASTROPHE_POSITIVE_BAND_DORMANCY_FRACTION", 0.22)
CATASTROPHE_DYNAMIC_BAND_AXIS_MODE: str = _env_str("FWS_CATASTROPHE_DYNAMIC_BAND_AXIS_MODE", "random").strip().lower()
CATASTROPHE_MANUAL_POSITIVE_BAND_AXIS: str = _env_str("FWS_CATASTROPHE_MANUAL_POSITIVE_BAND_AXIS", "vertical").strip().lower()

# Per-preset durations.
CATASTROPHE_DURATION_GLOBAL_ATTENUATION = _env_int("FWS_CATASTROPHE_DURATION_GLOBAL_ATTENUATION", 2600)
CATASTROPHE_DURATION_POSITIVE_BAND_DORMANCY = _env_int("FWS_CATASTROPHE_DURATION_POSITIVE_BAND_DORMANCY", 2000)
CATASTROPHE_DURATION_POLARITY_SPLIT = _env_int("FWS_CATASTROPHE_DURATION_POLARITY_SPLIT", 2400)
CATASTROPHE_DURATION_REGIONAL_ATTENUATION = _env_int("FWS_CATASTROPHE_DURATION_REGIONAL_ATTENUATION", 2200)
CATASTROPHE_DURATION_INVERSION = _env_int("FWS_CATASTROPHE_DURATION_INVERSION", 1500)
CATASTROPHE_DURATION_FULL_DORMANCY = _env_int("FWS_CATASTROPHE_DURATION_FULL_DORMANCY", 1200)

CATASTROPHE_PRESET_DURATION_TICKS = {
    "global_attenuation": int(CATASTROPHE_DURATION_GLOBAL_ATTENUATION),
    "positive_band_dormancy": int(CATASTROPHE_DURATION_POSITIVE_BAND_DORMANCY),
    "polarity_split_left_negative": int(CATASTROPHE_DURATION_POLARITY_SPLIT),
    "polarity_split_right_negative": int(CATASTROPHE_DURATION_POLARITY_SPLIT),
    "regional_attenuation_left": int(CATASTROPHE_DURATION_REGIONAL_ATTENUATION),
    "regional_attenuation_right": int(CATASTROPHE_DURATION_REGIONAL_ATTENUATION),
    "inversion": int(CATASTROPHE_DURATION_INVERSION),
    "full_dormancy": int(CATASTROPHE_DURATION_FULL_DORMANCY),
}

# Weighted scheduler distribution.
CATASTROPHE_WEIGHT_GLOBAL_ATTENUATION = _env_float("FWS_CATASTROPHE_WEIGHT_GLOBAL_ATTENUATION", 1.00)
CATASTROPHE_WEIGHT_POSITIVE_BAND_DORMANCY = _env_float("FWS_CATASTROPHE_WEIGHT_POSITIVE_BAND_DORMANCY", 0.90)
CATASTROPHE_WEIGHT_POLARITY_SPLIT_LEFT_NEGATIVE = _env_float("FWS_CATASTROPHE_WEIGHT_POLARITY_SPLIT_LEFT_NEGATIVE", 0.70)
CATASTROPHE_WEIGHT_POLARITY_SPLIT_RIGHT_NEGATIVE = _env_float("FWS_CATASTROPHE_WEIGHT_POLARITY_SPLIT_RIGHT_NEGATIVE", 0.70)
CATASTROPHE_WEIGHT_REGIONAL_ATTENUATION_LEFT = _env_float("FWS_CATASTROPHE_WEIGHT_REGIONAL_ATTENUATION_LEFT", 0.55)
CATASTROPHE_WEIGHT_REGIONAL_ATTENUATION_RIGHT = _env_float("FWS_CATASTROPHE_WEIGHT_REGIONAL_ATTENUATION_RIGHT", 0.55)

# Experimental modes remain available for manual triggering but are disabled from the
# dynamic scheduler by default because they can create brittle extinction pressure.
CATASTROPHE_WEIGHT_INVERSION = _env_float("FWS_CATASTROPHE_WEIGHT_INVERSION", 0.0)
CATASTROPHE_WEIGHT_FULL_DORMANCY = _env_float("FWS_CATASTROPHE_WEIGHT_FULL_DORMANCY", 0.0)

CATASTROPHE_DYNAMIC_WEIGHTS = {
    "global_attenuation": float(CATASTROPHE_WEIGHT_GLOBAL_ATTENUATION),
    "positive_band_dormancy": float(CATASTROPHE_WEIGHT_POSITIVE_BAND_DORMANCY),
    "polarity_split_left_negative": float(CATASTROPHE_WEIGHT_POLARITY_SPLIT_LEFT_NEGATIVE),
    "polarity_split_right_negative": float(CATASTROPHE_WEIGHT_POLARITY_SPLIT_RIGHT_NEGATIVE),
    "regional_attenuation_left": float(CATASTROPHE_WEIGHT_REGIONAL_ATTENUATION_LEFT),
    "regional_attenuation_right": float(CATASTROPHE_WEIGHT_REGIONAL_ATTENUATION_RIGHT),
    "inversion": float(CATASTROPHE_WEIGHT_INVERSION),
    "full_dormancy": float(CATASTROPHE_WEIGHT_FULL_DORMANCY),
}

# Capture Points ("King of the Hill")
# -----------------------------------
# Dense objective pressure is often the most reliable anti-camping incentive.

# Number of capture points.
# Env: FWS_CP_COUNT
# 1 large CP often creates strongest convergence to a shared objective.
CP_COUNT           = _env_int("FWS_CP_COUNT", 9)

# Capture point size ratio.
# Env: FWS_CP_SIZE_RATIO
# Larger values increase contact probability but may reduce tactical variety.
CP_SIZE_RATIO      = _env_float("FWS_CP_SIZE_RATIO", 0.06)

# Team reward per tick for owning/outnumbering on CP (fed into team reward path).
# Env: FWS_CP_REWARD
# Increasing this makes objective play more dominant relative to survival/farming.
CP_REWARD_PER_TICK = _env_float("FWS_CP_REWARD", 0)

# =============================================================================
# ⚔️ COMBAT BIOLOGY & CLASSES
# =============================================================================
# Unit IDs and combat stats. These define class asymmetry and damage/health scales.

# Unit type identifiers (schema-like constants used throughout engine/runtime).
UNIT_SOLDIER_ID = 1
UNIT_ARCHER_ID  = 2
UNIT_SOLDIER    = UNIT_SOLDIER_ID
UNIT_ARCHER     = UNIT_ARCHER_ID

# Global max HP reference (used by normalization/clamping/logic depending on engine).
# Env: FWS_MAX_HP
MAX_HP     = _env_float("FWS_MAX_HP", 1.0)

# Soldier base HP.
# Env: FWS_SOLDIER_HP
# Higher soldier HP increases frontline durability and may favor melee/objective holding.
SOLDIER_HP = _env_float("FWS_SOLDIER_HP", 1.0)

# Archer base HP.
# Env: FWS_ARCHER_HP
# Lower HP makes positioning and line-of-sight more critical.
ARCHER_HP  = _env_float("FWS_ARCHER_HP", 0.60)

# Base attack reference (generic/shared fallback).
# Env: FWS_BASE_ATK
BASE_ATK    = _env_float("FWS_BASE_ATK", 0.20)

# Soldier attack damage coefficient.
# Env: FWS_SOLDIER_ATK
SOLDIER_ATK = _env_float("FWS_SOLDIER_ATK", 0.15)

# Archer attack damage coefficient.
# Env: FWS_ARCHER_ATK
ARCHER_ATK  = _env_float("FWS_ARCHER_ATK", 0.10)

# Maximum attack among configured attack stats (with lower bound epsilon to avoid zero divisions).
# This is a derived helper constant used for normalization/scaling elsewhere.
MAX_ATK     = max(SOLDIER_ATK, ARCHER_ATK, BASE_ATK, 1e-6)

# Archer attack range in cells.
# Env: FWS_ARCHER_RANGE
# Larger range increases kiting potential and engagement radius.
ARCHER_RANGE = _env_int("FWS_ARCHER_RANGE", 4)

# Whether walls block archer line-of-sight.
# Env: FWS_ARCHER_BLOCK_LOS
# True makes topology matter more; False makes ranged classes much more globally threatening.
ARCHER_LOS_BLOCKS_WALLS = _env_bool("FWS_ARCHER_BLOCK_LOS", True)

# =============================================================================
# 🔋 METABOLISM (Anti-Stalemate Mechanic)
# =============================================================================
# Metabolism imposes HP drain over time, pressuring agents to seek healing/objectives/fights.
# This is a powerful anti-camping mechanic when balanced against HEAL_RATE.

# Master metabolism toggle.
# Env: FWS_META_ON
# Turning this off often increases stagnant equilibria if survival reward is positive.
METABOLISM_ENABLED       = _env_bool("FWS_META_ON", True)

# Soldier HP drain per tick from metabolism.
# Env: FWS_META_SOLDIER
# Compare to HEAL_RATE to estimate whether stationary healing zones create net-positive sustain.
META_SOLDIER_HP_PER_TICK = _env_float("FWS_META_SOLDIER", 0.0015)

# Archer HP drain per tick from metabolism.
# Env: FWS_META_ARCHER
META_ARCHER_HP_PER_TICK  = _env_float("FWS_META_ARCHER",  0.0010)

# =============================================================================
# 👁️ SENSORS & INSTINCT (THE AI'S "EYES")
# =============================================================================
# Perception radius and instinct radius directly affect what policies can infer and how soon
# separated agents can re-acquire enemies/objectives.

# Soldier vision range (cells/steps, depending on engine raycast semantics).
# Env: FWS_VISION_SOLDIER
# Larger values increase perception cost but can reduce aimless drifting and re-contact delay.
VISION_RANGE_SOLDIER = _env_int("FWS_VISION_SOLDIER", 6)

# Archer vision range.
# Env: FWS_VISION_ARCHER
VISION_RANGE_ARCHER  = _env_int("FWS_VISION_ARCHER", 14)

# Convenience mapping by unit type.
# This allows downstream code to query per-class vision cleanly.
VISION_RANGE_BY_UNIT = {
    UNIT_SOLDIER_ID: VISION_RANGE_SOLDIER,
    UNIT_ARCHER_ID:  VISION_RANGE_ARCHER,
}

# Maximum raycast steps derived from max vision range.
# If vision ranges change, raycast cap tracks automatically.
RAYCAST_MAX_STEPS = max(max(VISION_RANGE_BY_UNIT.values()), 1)
RAY_MAX_STEPS     = RAYCAST_MAX_STEPS

# Instinct radius (broader neighborhood context features).
# Env: FWS_INSTINCT_RADIUS
# Increasing may improve macro-context awareness but can increase feature computation cost.
INSTINCT_RADIUS = _env_int("FWS_INSTINCT_RADIUS", 16)

# =============================================================================
# 🧩 TENSOR OBSERVATION LAYOUT (STRICT CONTRACT)
# =============================================================================
# These are high-risk shape contract constants. Changing them can break:
# - model architectures,
# - checkpoint loading,
# - observation splitting/tokenization logic.

# Number of ray tokens in observation.
# Env: FWS_RAY_TOKENS
# Must stay synchronized with model tokenization assumptions and observation encoder.
RAY_TOKEN_COUNT = _env_int("FWS_RAY_TOKENS", 32)

# Number of features per ray token.
# This is a schema constant (not env-overridden here).
RAY_FEAT_DIM    = 8

# Flattened ray feature block size.
# Derived: num_rays * features_per_ray
RAYS_FLAT_DIM   = RAY_TOKEN_COUNT * RAY_FEAT_DIM

# Rich (non-ray) base feature dimension.
# Schema-like constant; changing this requires coordinated updates in obs builders/models.
RICH_BASE_DIM   = 23

# Instinct feature dimension appended to rich tail.
INSTINCT_DIM    = 4

# Total rich tail dimension = base + instinct.
RICH_TOTAL_DIM  = RICH_BASE_DIM + INSTINCT_DIM

# Final observation dimension contract.
# Used by brains, runtime buffers, validation checks.
OBS_DIM = RAYS_FLAT_DIM + RICH_TOTAL_DIM

# -----------------------------------------------------------------------------
# Observation schema versioning
# -----------------------------------------------------------------------------
# This is a POLICY-INTERFACE schema version, not a world/checkpoint tensor version.
# Increment it whenever observation feature meaning changes in a way that can make
# previously trained policy weights unsafe even if dimensionality stays constant.
#
# Version history:
#   v1: rich_base[9]  = on_heal_local       (boolean-like 0/1)
#       rich_base[10] = on_cp_local         (boolean-like 0/1)
#   v2: rich_base[9]  = zone_effect_local   (signed scalar in [-1, +1])
#       rich_base[10] = on_cp_local         (boolean-like 0/1)
OBS_SCHEMA_VERSION: int = 2
OBS_SCHEMA_FAMILY: str = "signed_zone_scalar_v2"

# Rich-base column contract.
# These names document the exact order expected by the observation builder,
# obs_spec, and policy code. Changing order/meaning without coordinated updates
# is extremely dangerous because PPO can continue "working" while consuming the
# wrong semantics.
RICH_BASE_FEATURE_NAMES = (
    "hp_ratio",             # 0
    "x_norm",               # 1
    "y_norm",               # 2
    "is_red_team",          # 3
    "is_blue_team",         # 4
    "is_soldier",           # 5
    "is_archer",            # 6
    "atk_norm",             # 7
    "vision_norm",          # 8
    "zone_effect_local",    # 9  signed scalar in [-1, +1]
    "cp_local",             # 10 boolean-like 0/1
    "tick_norm",            # 11
    "red_score_norm",       # 12
    "blue_score_norm",      # 13
    "red_cp_points_norm",   # 14
    "blue_cp_points_norm",  # 15
    "red_kills_norm",       # 16
    "blue_kills_norm",      # 17
    "red_deaths_norm",      # 18
    "blue_deaths_norm",     # 19
    "pad_0",                # 20
    "pad_1",                # 21
    "pad_2",                # 22
)
if len(RICH_BASE_FEATURE_NAMES) != int(RICH_BASE_DIM):
    _config_issue(
        f"RICH_BASE_FEATURE_NAMES length mismatch: got {len(RICH_BASE_FEATURE_NAMES)} expected {int(RICH_BASE_DIM)}"
    )

RICH_BASE_ZONE_EFFECT_LOCAL_IDX = 9
RICH_BASE_CP_LOCAL_IDX = 10
OBS_ZONE_EFFECT_LOCAL_RANGE = (-1.0, 1.0)
OBS_CP_LOCAL_RANGE = (0.0, 1.0)

# Semantic grouping of rich-base indices used by tokenized architectures (e.g., Tron/Mirror).
# Each tuple/list defines which columns of rich_base belong to that semantic token.
# Changing these indices changes feature semantics without changing dimensionality—very dangerous.
# zone_context deliberately remains width-2 in v2, but its first element is now a
# SIGNED scalar local zone effect rather than a boolean heal-local flag.
SEMANTIC_RICH_BASE_INDICES = {
    "own_context":    (0, 1, 2, 5, 6, 7, 8),
    "world_context":  (11, 20, 21, 22),
    "zone_context":   (RICH_BASE_ZONE_EFFECT_LOCAL_IDX, RICH_BASE_CP_LOCAL_IDX),
    "team_context":   (3, 4, 12, 13, 14, 15),
    "combat_context": (16, 17, 18, 19),
}

# Canonical order in which semantic tokens are assembled/consumed.
# Must match downstream model expectations if used directly.
SEMANTIC_TOKEN_ORDER = (
    "own_context", "world_context", "zone_context", "team_context", "combat_context", "instinct_context"
)

# Number of discrete actions in policy output head.
# Env: FWS_NUM_ACTIONS
# Changing this invalidates policy heads/checkpoints and action decoding.
NUM_ACTIONS = _env_int("FWS_NUM_ACTIONS", 41)

# =============================================================================
# 🔄 POPULATION CONTROL (REINFORCEMENTS)
# =============================================================================
# These knobs control reinforcement/respawn behavior and are central to evolutionary dynamics,
# population pressure, and whether local policies propagate.

# Master respawn/reinforcement toggle.
# Env: FWS_RESPAWN
RESPAWN_ENABLED = _env_bool("FWS_RESPAWN", True)

# Minimum population floor per team (hysteresis/fill system may try to maintain this).
# Env: FWS_RESP_FLOOR_PER_TEAM
RESP_FLOOR_PER_TEAM      = _env_int("FWS_RESP_FLOOR_PER_TEAM", 110)

# Hard cap of respawns applied per tick.
# Env: FWS_RESP_MAX_PER_TICK
# Prevents large bursts from causing frame/tick spikes.
RESP_MAX_PER_TICK        = _env_int("FWS_RESP_MAX_PER_TICK", 5)

# Periodic reinforcement cycle length in ticks.
# Env: FWS_RESP_PERIOD_TICKS
RESP_PERIOD_TICKS        = _env_int("FWS_RESP_PERIOD_TICKS", 20000)

# Budget available per reinforcement period (implementation-defined accounting).
# Env: FWS_RESP_PERIOD_BUDGET
RESP_PERIOD_BUDGET       = _env_int("FWS_RESP_PERIOD_BUDGET", 40)

# Hysteresis cooldown in ticks to prevent oscillatory refill behavior.
# Env: FWS_RESP_HYST_COOLDOWN_TICKS
RESP_HYST_COOLDOWN_TICKS = _env_int("FWS_RESP_HYST_COOLDOWN_TICKS", 100)

# Spawn wall margin (avoid spawning too close to walls).
# Env: FWS_RESP_WALL_MARGIN
RESP_WALL_MARGIN         = _env_int("FWS_RESP_WALL_MARGIN", 2)

# Initial world spawn mode (fresh-start path in main.py).
# Env: FWS_SPAWN_MODE
# Supported values: "uniform", "symmetric"
SPAWN_MODE: str = _env_str("FWS_SPAWN_MODE", "uniform").strip().lower()
if SPAWN_MODE not in ("uniform", "symmetric"):
    _config_warn(f"Unknown SPAWN_MODE={SPAWN_MODE!r}; falling back to 'uniform'")
    SPAWN_MODE = "uniform"

# Initial spawn archer ratio (for starting populations / certain spawn paths).
# Env: FWS_SPAWN_ARCHER_RATIO
# Range typically [0,1]. 0.35 means ~35% archers.
SPAWN_ARCHER_RATIO       = _env_float("FWS_SPAWN_ARCHER_RATIO", 0.35)

# Probabilistic respawn chance per dead agent (or per death event, depending on runtime logic).
# Env: FWS_RESPAWN_PROB
# Lower values slow replacement pressure; higher values accelerate turnover.
RESPAWN_PROB_PER_DEAD        = _env_float("FWS_RESPAWN_PROB", 0.05)

# Number of spawn position attempts before giving up.
# Env: FWS_RESPAWN_TRIES
# Increase if spawn failures are common due to dense maps/walls.
RESPAWN_SPAWN_TRIES          = _env_int("FWS_RESPAWN_TRIES", 200)

# Standard deviation of mutation noise applied on respawned brains (when mutation is used).
# Env: FWS_MUT_STD
# Larger => more behavioral disruption/exploration, less stability/inheritance fidelity.
RESPAWN_MUTATION_STD         = _env_float("FWS_MUT_STD", 0.1)

# Probability of cloning path vs alternate respawn path (runtime-defined semantics).
# Env: FWS_CLONE_PROB
# Set to 1 for pure cloning+noise style inheritance if runtime supports it as documented.
RESPAWN_CLONE_PROB           = _env_float("FWS_CLONE_PROB", 1)

# Whether parent selection uses team elite criteria.
# Env: FWS_TEAM_ELITE
# This is a major evolutionary pressure knob.
RESPAWN_USE_TEAM_ELITE       = _env_bool("FWS_TEAM_ELITE", True)

# Whether optimizer state is reset on respawned agents.
# Env: FWS_RESET_OPT
# True usually prevents stale optimizer moments from carrying across inheritance.
RESPAWN_RESET_OPT_ON_RESPAWN = _env_bool("FWS_RESET_OPT", True)

# Spawn position jitter radius around chosen spawn point/parent point (runtime-dependent exact usage).
# Env: FWS_RESP_JITTER
RESPAWN_JITTER_RADIUS        = _env_int("FWS_RESP_JITTER", 1)

# Cooldown per respawned entity/slot/team before another respawn (runtime-defined scope).
# Env: FWS_RESPAWN_CD
RESPAWN_COOLDOWN_TICKS       = _env_int("FWS_RESPAWN_CD", 500)

# Batch respawn count per team when respawn batch path is triggered.
# Env: FWS_RESPAWN_BATCH
RESPAWN_BATCH_PER_TEAM       = _env_int("FWS_RESPAWN_BATCH", 1)

# Target archer share for respawned population (runtime may use as quota/ratio target).
# Env: FWS_RESPAWN_ARCHER_SHARE
RESPAWN_ARCHER_SHARE         = _env_float("FWS_RESPAWN_ARCHER_SHARE", 0.50)

# Bias for spawning toward interior regions vs edges.
# Env: FWS_RESPAWN_INTERIOR_BIAS
# Higher values can reduce edge/corner camping and improve contact probability.
RESPAWN_INTERIOR_BIAS        = _env_float("FWS_RESPAWN_INTERIOR_BIAS", 0.50)

# ----------------------------------------------------------------------
# Respawn evolution layer (optional; all switches default to safe/backward-compatible)
# ----------------------------------------------------------------------
# Tick-window anomaly ticket generator (new rare-mutation trigger).
# Env: FWS_RESP_RARE_TICK_WINDOW_ENABLE
RESPAWN_RARE_MUTATION_TICK_WINDOW_ENABLE: bool = _env_bool("FWS_RESP_RARE_TICK_WINDOW_ENABLE", True)

# Rare-mutation ticket cadence in ticks (global; at most one pending ticket).
# Env: FWS_RESP_RARE_TICK_WINDOW
RESPAWN_RARE_MUTATION_TICK_WINDOW_TICKS: int = _env_int("FWS_RESP_RARE_TICK_WINDOW", 7000)

# Apply slight physical drift to anomaly newborns (tick-window mode).
# Env: FWS_RESP_RARE_PHYS_ENABLE
RESPAWN_RARE_MUTATION_PHYSICAL_ENABLE: bool = _env_bool("FWS_RESP_RARE_PHYS_ENABLE", True)

# Physical drift magnitude (fractional std, e.g., 0.03 = 3% std per trait multiplier).
# Env: FWS_RESP_RARE_PHYS_STD
RESPAWN_RARE_MUTATION_PHYSICAL_DRIFT_STD_FRAC: float = _env_float("FWS_RESP_RARE_PHYS_STD", 0.09)

# Physical drift clamp (fractional absolute cap on multiplier delta).
# Env: FWS_RESP_RARE_PHYS_CLIP
RESPAWN_RARE_MUTATION_PHYSICAL_DRIFT_CLIP_FRAC: float = _env_float("FWS_RESP_RARE_PHYS_CLIP", 0.15)

# Apply heavy extra brain noise only on inherited/cloned anomaly path.
# Env: FWS_RESP_RARE_BRAIN_NOISE_ENABLE
RESPAWN_RARE_MUTATION_INHERITED_BRAIN_NOISE_ENABLE: bool = _env_bool("FWS_RESP_RARE_BRAIN_NOISE_ENABLE", True)

# Std for heavy anomaly inherited-brain noise (in addition to RESPAWN_MUTATION_STD).
# Env: FWS_RESP_RARE_BRAIN_NOISE_STD
RESPAWN_RARE_MUTATION_INHERITED_BRAIN_NOISE_STD: float = _env_float("FWS_RESP_RARE_BRAIN_NOISE_STD", 0.30)

# Parent selection policy for clone path.
# Env: FWS_RESP_PARENT_SELECT_MODE
# Supported: "random", "topk_weighted"
RESPAWN_PARENT_SELECTION_MODE: str = _env_str("FWS_RESP_PARENT_SELECT_MODE", "topk_weighted").strip().lower()
if RESPAWN_PARENT_SELECTION_MODE not in ("random", "topk_weighted"):
    _config_warn(f"Unknown RESPAWN_PARENT_SELECTION_MODE={RESPAWN_PARENT_SELECTION_MODE!r}; falling back to 'random'")
    RESPAWN_PARENT_SELECTION_MODE = "random"

# Fraction of parent candidates kept in the top-k pool for weighted sampling.
# Env: FWS_RESP_PARENT_TOPK_FRAC
RESPAWN_PARENT_SELECTION_TOPK_FRAC: float = _env_float("FWS_RESP_PARENT_TOPK_FRAC", 0.3)

# Exponent applied to parent score weights (higher => stronger bias inside top-k).
# Env: FWS_RESP_PARENT_SCORE_POWER
RESPAWN_PARENT_SELECTION_SCORE_POWER: float = _env_float("FWS_RESP_PARENT_SCORE_POWER", 0.75)

# Spawn location policy for respawn.
# Env: FWS_RESP_SPAWN_LOCATION_MODE
# Supported: "uniform", "near_parent"
RESPAWN_SPAWN_LOCATION_MODE: str = _env_str("FWS_RESP_SPAWN_LOCATION_MODE", "near_parent").strip().lower()
if RESPAWN_SPAWN_LOCATION_MODE not in ("uniform", "near_parent"):
    _config_warn(f"Unknown RESPAWN_SPAWN_LOCATION_MODE={RESPAWN_SPAWN_LOCATION_MODE!r}; falling back to 'uniform'")
    RESPAWN_SPAWN_LOCATION_MODE = "uniform"

# Radius for near-parent spawn search (Chebyshev neighborhood).
# Env: FWS_RESP_SPAWN_NEAR_PARENT_RADIUS
RESPAWN_SPAWN_NEAR_PARENT_RADIUS: int = _env_int("FWS_RESP_SPAWN_NEAR_PARENT_RADIUS", max(1, int(RESPAWN_JITTER_RADIUS)))

# =============================================================================
# 🏆 REWARD SHAPING (RL Feedback Loop)
# =============================================================================
# These are among the most behavior-defining knobs in the whole system.
# Small changes can radically alter emergent equilibrium.

# Team-level reward for a kill.
# Env: FWS_REW_KILL
# Larger => team combat aggression becomes more profitable.
TEAM_KILL_REWARD       = _env_float("FWS_REW_KILL",       0)

# Team-level reward for damage dealt (dense combat shaping).
# Env: FWS_REW_DMG_DEALT
# 0 disables this shaping channel.
TEAM_DMG_DEALT_REWARD  = _env_float("FWS_REW_DMG_DEALT",  0.00)

# Team-level penalty when an agent dies.
# Env: FWS_REW_DEATH
# More negative => risk aversion increases unless compensated by strong rewards.
TEAM_DEATH_PENALTY     = _env_float("FWS_REW_DEATH",     0)

# Team-level penalty for damage taken.
# Env: FWS_REW_DMG_TAKEN
# 0 disables this shaping channel.
TEAM_DMG_TAKEN_PENALTY = _env_float("FWS_REW_DMG_TAKEN",  0.00)

# PPO shaping (per-agent learning signals)
# ----------------------------------------
# These directly affect PPO optimization targets and can dominate learning if mis-scaled.

# Dense per-tick HP reward coefficient.
# Env: FWS_PPO_REW_HP_TICK
# If too high and positive, "stay alive safely" can dominate and produce camping behavior.
PPO_REWARD_HP_TICK         = _env_float("FWS_PPO_REW_HP_TICK", 0.00005)
# HP PPO reward mode selector (keeps legacy behavior by default).
# Env: FWS_PPO_HP_REWARD_MODE
# Supported values (runtime patch path):
# - "raw"            : legacy behavior, reward = HP * PPO_REWARD_HP_TICK
# - "threshold_ramp" : reward is gated below threshold, then scales smoothly to full reward
PPO_HP_REWARD_MODE         = _env_str("FWS_PPO_HP_REWARD_MODE", "threshold_ramp").strip().lower()

# HP percentage threshold used by threshold-ramp HP reward mode.
# Env: FWS_PPO_HP_REWARD_THRESHOLD
# Example: 0.60 => no HP PPO reward at <=60% HP, smooth ramp above it.
PPO_HP_REWARD_THRESHOLD    = _env_float("FWS_PPO_HP_REWARD_THRESHOLD", 0.5)

# Individual PPO reward for damage dealt (dense combat shaping, per-agent only).
# Env: FWS_PPO_REW_DMG_DEALT_AGENT
# 0 disables this shaping channel.
PPO_REWARD_DMG_DEALT_INDIVIDUAL = _env_float("FWS_PPO_REW_DMG_DEALT_AGENT", 0.4)

# Individual PPO penalty magnitude for damage taken (dense combat shaping, per-agent only).
# Env: FWS_PPO_PEN_DMG_TAKEN_AGENT
# Applied as a subtraction in reward code: reward -= damage_taken * this_value
PPO_PENALTY_DMG_TAKEN_INDIVIDUAL = _env_float("FWS_PPO_PEN_DMG_TAKEN_AGENT", 0.003)

# Individual kill reward for PPO agent signal.
# Env: FWS_PPO_REW_KILL_AGENT
# Larger => direct combat success becomes strongly reinforced.
PPO_REWARD_KILL_INDIVIDUAL = _env_float("FWS_PPO_REW_KILL_AGENT", 40.0)

# Death penalty for PPO agent signal.
# Env: FWS_PPO_REW_DEATH
# More negative => stronger risk aversion and survival bias.
PPO_REWARD_DEATH           = _env_float("FWS_PPO_REW_DEATH", -0.5)

# Reward for contested CP participation/control signal (implementation-dependent exact condition).
# Env: FWS_PPO_REW_CONTEST
# Increasing this pushes policies toward objective zones rather than isolated survival.
PPO_REWARD_CONTESTED_CP    = _env_float("FWS_PPO_REW_CONTEST", 2.5)

# =============================================================================
# 🧠 REINFORCEMENT LEARNING (PROXIMAL POLICY OPTIMIZATION)
# =============================================================================
# Core PPO runtime/training hyperparameters. These control optimization stability,
# sample efficiency, and compute cost.

# Master PPO enable flag.
# Env: FWS_PPO_ENABLED
# If False, runtime may run scripted/frozen policies depending on implementation.
PPO_ENABLED       = _env_bool("FWS_PPO_ENABLED", True)

# Reset/log PPO state behavior on startup/resume (runtime-defined exact semantics).
# Env: FWS_PPO_RESET_LOG
PPO_RESET_LOG     = _env_bool("FWS_PPO_RESET_LOG", False)

# PPO rollout window length in ticks (trajectory horizon before update).
# Env: FWS_PPO_TICKS
# Larger windows => more temporal context, more memory/latency before updates.
PPO_WINDOW_TICKS  = _env_int("FWS_PPO_TICKS", 256)

# Learning rate.
# Env: FWS_PPO_LR
# Typical PPO range often around 1e-4 to 3e-4 (task-dependent).
PPO_LR            = _env_float("FWS_PPO_LR", 3e-4)

# LR scheduler max time horizon (ticks/steps, scheduler-defined).
# Env: FWS_PPO_T_MAX
PPO_LR_T_MAX      = _env_int("FWS_PPO_T_MAX", 10_000_000)

# Minimum learning-rate floor for scheduler.
# Env: FWS_PPO_ETA_MIN
PPO_LR_ETA_MIN    = _env_float("FWS_PPO_ETA_MIN", 1e-6)

# PPO clip epsilon.
# Env: FWS_PPO_CLIP
# Larger => looser policy updates (more aggressive), smaller => more conservative.
PPO_CLIP          = _env_float("FWS_PPO_CLIP", 0.2)

# Alias for clip epsilon (compatibility/readability).
PPO_CLIP_EPS      = PPO_CLIP

# Entropy regularization coefficient.
# Env: FWS_PPO_ENTROPY
# Higher => more exploration, slower convergence; lower => more exploitation.
PPO_ENTROPY_COEF  = _env_float("FWS_PPO_ENTROPY", 0.05)

# Value loss coefficient.
# Env: FWS_PPO_VCOEF
# Controls critic loss weight relative to policy loss.
PPO_VALUE_COEF    = _env_float("FWS_PPO_VCOEF", 0.5)

# Number of PPO epochs per update.
# Env: FWS_PPO_EPOCHS
# More epochs improve sample reuse but can overfit stale data / reduce throughput.
PPO_EPOCHS        = _env_int("FWS_PPO_EPOCHS", 4)

# Number of minibatches per PPO update.
# Env: FWS_PPO_MINIB
# Effective minibatch size depends on rollout size.
PPO_MINIBATCHES   = _env_int("FWS_PPO_MINIB", 8)

# Gradient norm clipping threshold.
# Env: FWS_PPO_MAXGN
# Helps prevent unstable updates/spikes.
PPO_MAX_GRAD_NORM = _env_float("FWS_PPO_MAXGN", 0.5)

# Early-stop target KL (if runtime uses it).
# Env: FWS_PPO_TKL
# Lower => more conservative updates; higher => more aggressive updates.
PPO_TARGET_KL     = _env_float("FWS_PPO_TKL", 0.02)

# Discount factor gamma.
# Env: FWS_PPO_GAMMA
# Higher values weight long-term outcomes more strongly.
PPO_GAMMA         = _env_float("FWS_PPO_GAMMA", 0.995)

# GAE lambda.
# Env: FWS_PPO_LAMBDA
# Bias-variance tradeoff for advantage estimation.
PPO_LAMBDA        = _env_float("FWS_PPO_LAMBDA", 0.95)

# Update cadence in ticks (runtime-defined relation to PPO window scheduling).
# Env: FWS_PPO_UPDATE_TICKS
PPO_UPDATE_TICKS  = _env_int("FWS_PPO_UPDATE_TICKS", 5)

# Compatibility alias for entropy bonus.
# Env: FWS_PPO_ENTROPY_BONUS
# If set and FWS_PPO_ENTROPY is not explicitly set, this value backfills PPO_ENTROPY_COEF.
PPO_ENTROPY_BONUS = _env_float("FWS_PPO_ENTROPY_BONUS", PPO_ENTROPY_COEF)

# Compatibility precedence logic:
# - If user sets FWS_PPO_ENTROPY_BONUS but NOT FWS_PPO_ENTROPY, copy bonus into coef.
# - Preserves legacy env naming while allowing new explicit knob.
if _env_is_set("FWS_PPO_ENTROPY_BONUS") and not _env_is_set("FWS_PPO_ENTROPY"):
    PPO_ENTROPY_COEF = float(PPO_ENTROPY_BONUS)

# Whether each agent has its own brain parameters (no parameter sharing).
# Env: FWS_PER_AGENT_BRAINS
# True drastically increases compute/memory but allows divergent policies.
PER_AGENT_BRAINS        = _env_bool("FWS_PER_AGENT_BRAINS", True)

# Global mutation period (ticks) for periodic mutation events.
# Env: FWS_MUTATE_EVERY
# Larger => rarer perturbation; smaller => more frequent diversity injection.
MUTATION_PERIOD_TICKS   = _env_int("FWS_MUTATE_EVERY", 1000000000)

# Fraction of alive agents mutated during mutation event.
# Env: FWS_MUTATE_FRAC
# Small default supports rare diversity injection without resetting system dynamics.
MUTATION_FRACTION_ALIVE = _env_float("FWS_MUTATE_FRAC", 0.02)

# =============================================================================
# 🤖 BRAIN ARCHITECTURE (THE NEURAL ENGINES)
# =============================================================================
# Brain-type selection and assignment policy knobs. These choose which model class is used
# and how teams mix or split architectures.

# Primary brain kind selector for the experimental no-attention family.
# Env: FWS_BRAIN
BRAIN_KIND: str = _env_str("FWS_BRAIN", "whispering_abyss").strip().lower()

# Whether teams are explicitly assigned specific brain variants/policies.
# Env: FWS_TEAM_BRAIN_ASSIGNMENT
TEAM_BRAIN_ASSIGNMENT: bool = _env_bool("FWS_TEAM_BRAIN_ASSIGNMENT", True)

# Team brain assignment mode.
# Env: FWS_TEAM_BRAIN_MODE
# Supported by the new MLP family:
#   - "exclusive" / "split" / "team"
#   - "mix" / "hybrid" / "both"
TEAM_BRAIN_ASSIGNMENT_MODE: str = _env_str("FWS_TEAM_BRAIN_MODE", "mix").strip().lower()

# Strategy for mixing brain types across teams/slots.
# Env: FWS_TEAM_BRAIN_MIX_STRATEGY
# Supported:
#   - "alternate" / "roundrobin" / "rr"
#   - "random" / "prob" / "probabilistic"
TEAM_BRAIN_MIX_STRATEGY: str = _env_str("FWS_TEAM_BRAIN_MIX_STRATEGY", "random").strip().lower()

# Fixed architecture per team in exclusive mode.
# Env: FWS_TEAM_BRAIN_RED / FWS_TEAM_BRAIN_BLUE
TEAM_BRAIN_EXCLUSIVE_RED: str = _env_str("FWS_TEAM_BRAIN_RED", "whispering_abyss").strip().lower()
TEAM_BRAIN_EXCLUSIVE_BLUE: str = _env_str("FWS_TEAM_BRAIN_BLUE", "veil_of_echoes").strip().lower()

# Deterministic sequence used by alternate/round-robin mix mode.
# Env: FWS_TEAM_BRAIN_MIX_SEQUENCE
TEAM_BRAIN_MIX_SEQUENCE = tuple(
    s.strip().lower()
    for s in _env_str(
        "FWS_TEAM_BRAIN_MIX_SEQUENCE",
        "whispering_abyss,veil_of_echoes,cathedral_of_ash,dreamer_in_black_fog,obsidian_pulse",
    ).split(",")
    if s.strip()
)

# Weighted probabilities used by random/probabilistic mix mode.
# Env:
#   FWS_TEAM_BRAIN_P_WHISPERING_ABYSS
#   FWS_TEAM_BRAIN_P_VEIL_OF_ECHOES
#   FWS_TEAM_BRAIN_P_CATHEDRAL_OF_ASH
#   FWS_TEAM_BRAIN_P_DREAMER_IN_BLACK_FOG
#   FWS_TEAM_BRAIN_P_OBSIDIAN_PULSE
TEAM_BRAIN_MIX_P_WHISPERING_ABYSS: float = _env_float("FWS_TEAM_BRAIN_P_WHISPERING_ABYSS", 0.20)
TEAM_BRAIN_MIX_P_VEIL_OF_ECHOES: float = _env_float("FWS_TEAM_BRAIN_P_VEIL_OF_ECHOES", 0.20)
TEAM_BRAIN_MIX_P_CATHEDRAL_OF_ASH: float = _env_float("FWS_TEAM_BRAIN_P_CATHEDRAL_OF_ASH", 0.20)
TEAM_BRAIN_MIX_P_DREAMER_IN_BLACK_FOG: float = _env_float("FWS_TEAM_BRAIN_P_DREAMER_IN_BLACK_FOG", 0.20)
TEAM_BRAIN_MIX_P_OBSIDIAN_PULSE: float = _env_float("FWS_TEAM_BRAIN_P_OBSIDIAN_PULSE", 0.20)

# RNG seed for brain mixing assignment.
# Env: FWS_TEAM_BRAIN_MIX_SEED
TEAM_BRAIN_MIX_SEED: int = _env_int("FWS_TEAM_BRAIN_MIX_SEED", int(globals().get("RNG_SEED", 0)))

# =============================================================================
# 🌑 SIMPLE MLP BRAIN FAMILY (NO ATTENTION)
# =============================================================================
# Shared two-token contract for every variant:
#   1) 32 ray tokens -> embedded -> aggregated into 1 ray summary token
#   2) [rich_base | instinct] -> projected into 1 rich token
#   3) [ray_token, rich_token] -> flatten -> normalize
#   4) final input width = 2 * BRAIN_MLP_D_MODEL

# Shared embedding width for the two tokens.
# Env: FWS_BRAIN_MLP_DMODEL
BRAIN_MLP_D_MODEL = _env_int("FWS_BRAIN_MLP_DMODEL", 32)

# Final flattened MLP input width. Derived, never hard-coded elsewhere.
BRAIN_MLP_FINAL_INPUT_WIDTH = 2 * BRAIN_MLP_D_MODEL

# Shared activation and normalization families.
# Env: FWS_BRAIN_MLP_ACT / FWS_BRAIN_MLP_NORM
BRAIN_MLP_ACTIVATION: str = _env_str("FWS_BRAIN_MLP_ACT", "gelu").strip().lower()
BRAIN_MLP_NORM: str = _env_str("FWS_BRAIN_MLP_NORM", "layernorm").strip().lower()

# Ray token aggregation rule.
# Env: FWS_BRAIN_MLP_RAY_SUMMARY
# Current implementation supports "mean".
BRAIN_MLP_RAY_SUMMARY: str = _env_str("FWS_BRAIN_MLP_RAY_SUMMARY", "mean").strip().lower()

# PPO-friendly head initialization gains.
# Env: FWS_BRAIN_MLP_ACTOR_GAIN / FWS_BRAIN_MLP_CRITIC_GAIN
BRAIN_MLP_ACTOR_INIT_GAIN: float = _env_float("FWS_BRAIN_MLP_ACTOR_GAIN", 0.01)
BRAIN_MLP_CRITIC_INIT_GAIN: float = _env_float("FWS_BRAIN_MLP_CRITIC_GAIN", 1.0)

# Stable ordering used by config validation, UI, telemetry, and mix logic.
BRAIN_MLP_KIND_ORDER = (
    "whispering_abyss",
    "veil_of_echoes",
    "cathedral_of_ash",
    "dreamer_in_black_fog",
    "obsidian_pulse",
)

# Poetic labels shown in viewer, inspector, logs, telemetry, and mutation reports.
BRAIN_KIND_DISPLAY_NAMES = {
    "whispering_abyss": "Whispering Abyss",
    "veil_of_echoes": "Veil of Echoes",
    "cathedral_of_ash": "Cathedral of Ash",
    "dreamer_in_black_fog": "Dreamer in the Black Fog",
    "obsidian_pulse": "Obsidian Pulse",
}

# Tiny cell-safe labels for the renderer overlay.
BRAIN_KIND_SHORT_LABELS = {
    "whispering_abyss": "WA",
    "veil_of_echoes": "VE",
    "cathedral_of_ash": "CA",
    "dreamer_in_black_fog": "DB",
    "obsidian_pulse": "OP",
}

# Variant definitions. These are fully wired in the new brain module.
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
# 🖥️ UI, VIEWER & SCREEN RECORDING
# =============================================================================
# Viewer/rendering controls. For maximum training throughput, headless mode is usually preferred.

# UI enable flag.
# Env: FWS_UI
# True by default here (UI-on). Set False for headless speed / training throughput.
ENABLE_UI  = _env_bool("FWS_UI", True)

# Explicit inspector mode selector (no-output viewer mode).
# Values:
#   - "off" (default): normal behavior
#   - "ui_no_output": launch viewer/inspection mode without creating results/telemetry/checkpoints files
# Env: FWS_INSPECTOR_MODE
INSPECTOR_MODE: str = _env_str("FWS_INSPECTOR_MODE", "off").strip().lower()

# Backward-compatible explicit flag for no-output inspector UI mode.
# If True, behaves like INSPECTOR_MODE="ui_no_output".
# Env: FWS_INSPECTOR_UI_NO_OUTPUT
INSPECTOR_UI_NO_OUTPUT: bool = _env_bool("FWS_INSPECTOR_UI_NO_OUTPUT", False)

# How often (ticks/frames) viewer refreshes full state from sim.
# Env: FWS_VIEWER_STATE_REFRESH_EVERY
# Larger => less UI overhead but less smooth state updates.
VIEWER_STATE_REFRESH_EVERY = _env_int("FWS_VIEWER_STATE_REFRESH_EVERY", 3)

# How often viewer refreshes selected-agent details/picking info.
# Env: FWS_VIEWER_PICK_REFRESH_EVERY
VIEWER_PICK_REFRESH_EVERY  = _env_int("FWS_VIEWER_PICK_REFRESH_EVERY", 3)

# UI font name.
# Env: FWS_UI_FONT
# Must exist on system or runtime should handle fallback.
UI_FONT_NAME: str = _env_str("FWS_UI_FONT", "consolas")

# Whether to center the viewer window on launch.
# Env: FWS_VIEWER_CENTER_WINDOW
VIEWER_CENTER_WINDOW: bool = _env_bool("FWS_VIEWER_CENTER_WINDOW", True)

# Cell pixel size in renderer.
# Env: FWS_CELL_SIZE
# Larger => more visible detail, larger window, more rendering cost.
CELL_SIZE  = _env_int("FWS_CELL_SIZE", 5)

# HUD / layout sizing.
# Env: FWS_HUD_W
HUD_WIDTH  = _env_int("FWS_HUD_W", 380)
VIEWER_SIDE_PANEL_WIDTH = _env_int("FWS_VIEWER_SIDE_PANEL_WIDTH", 460)
VIEWER_BOTTOM_PANEL_HEIGHT = _env_int("FWS_VIEWER_BOTTOM_PANEL_HEIGHT", 148)
VIEWER_PANEL_GAP = _env_int("FWS_VIEWER_PANEL_GAP", 8)
VIEWER_MIN_WIDTH = _env_int("FWS_VIEWER_MIN_WIDTH", 1320)
VIEWER_MIN_HEIGHT = _env_int("FWS_VIEWER_MIN_HEIGHT", 820)
VIEWER_AUTO_FIT_WORLD = _env_bool("FWS_VIEWER_AUTO_FIT_WORLD", True)
VIEWER_HELP_FONT_SIZE = _env_int("FWS_VIEWER_HELP_FONT_SIZE", 12)
VIEWER_SHOW_SELECTED_AGENT_SCORES = _env_bool("FWS_VIEWER_SHOW_SELECTED_AGENT_SCORES", True)
VIEWER_ZONE_OVERLAY_DEFAULT = _env_bool("FWS_VIEWER_ZONE_OVERLAY_DEFAULT", True)
VIEWER_HUD_VERBOSE = _env_bool("FWS_VIEWER_HUD_VERBOSE", True)
VIEWER_HUD_SHOW_SCHEDULER = _env_bool("FWS_VIEWER_HUD_SHOW_SCHEDULER", True)
VIEWER_BASE_ZONE_EDIT_STEP = _env_float("FWS_VIEWER_BASE_ZONE_EDIT_STEP", 0.25)

# Target render FPS (UI mode throttling/render loop).
# Env: FWS_TARGET_FPS
TARGET_FPS = _env_int("FWS_TARGET_FPS", 60)

# Video recording toggle (if runtime supports frame capture).
# Env: FWS_RECORD_VIDEO
# Recording can heavily reduce performance and increase disk I/O.
RECORD_VIDEO: bool     = _env_bool("FWS_RECORD_VIDEO", False)

# Video output FPS.
# Env: FWS_VIDEO_FPS
VIDEO_FPS: int         = _env_int("FWS_VIDEO_FPS", 60)

# Video upscale multiplier.
# Env: FWS_VIDEO_SCALE
VIDEO_SCALE: int       = _env_int("FWS_VIDEO_SCALE", 4)

# Capture every N ticks for video.
# Env: FWS_VIDEO_EVERY_TICKS
# Larger => timelapse effect, lower storage and overhead.
VIDEO_EVERY_TICKS: int = _env_int("FWS_VIDEO_EVERY_TICKS", 1)

# UI color palette dictionary.
# These are rendering-only aesthetics and should not affect simulation behavior.
UI_COLORS = {
    "bg": (15, 17, 22), "hud_bg": (10, 12, 16), "side_bg": (14, 16, 20),
    "grid": (35, 37, 42), "border": (80, 85, 95), "wall": (100, 105, 115),
    "empty": (20, 22, 28),

    "red_soldier": (240, 50, 50), "red_archer":  (255, 120, 0), "red": (240, 50, 50),
    "blue_soldier": (30, 160, 255), "blue_archer":  (0, 220, 180), "blue": (30, 160, 255),

    "archer_glyph": (255, 245, 120), "marker": (255, 255, 255),
    "text": (240, 240, 245), "text_dim": (160, 165, 175),
    "green": (50, 220, 120), "warn": (255, 170, 0),
    "bar_bg": (30, 35, 40), "bar_fg": (50, 220, 120),
    "graph_red": (240, 50, 50, 180), "graph_blue": (30, 160, 255, 180),
    "graph_grid": (50, 50, 60), "pause_text": (255, 200, 50)
}

# =============================================================================
# 🛠️ PROFILE OVERRIDE INJECTION
# =============================================================================
# Profiles are macro presets applied *after* defaults/env parsing, but only for keys the user
# has NOT explicitly set via env. This preserves explicit user overrides while still providing
# convenient preset bundles.

def _apply_profile_overrides() -> None:
    """
    Apply preset overrides based on PROFILE.

    Precedence policy:
    - If PROFILE == "default": do nothing.
    - For non-default profiles, each preset row is applied only if the corresponding env var
      was NOT explicitly set. This is an important design choice:
        explicit env > profile preset > hard-coded default

    Validation:
    - Performs a divisibility check for TRON_D_MODEL / TRON_HEADS after applying overrides.
    """
    if PROFILE == "default":
        return

    # Each tuple is:
    #   (env_key, global_var_name, preset_value)
    #
    # env_key:
    #   The env var that, if explicitly set, blocks preset override.
    #
    # global_var_name:
    #   The actual Python constant in this module to modify.
    #
    # preset_value:
    #   The profile-specific value to apply.
    presets = {
        "debug": [
            # Debug profile prioritizes visibility + faster iteration, not training quality.
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
            # Fast training profile: headless + reduced-ish model complexity for throughput.
            ("FWS_UI", "ENABLE_UI", False),
            ("FWS_USE_VMAP", "USE_VMAP", True),
            ("FWS_BRAIN_MLP_DMODEL", "BRAIN_MLP_D_MODEL", 24),
        ],
        "train_quality": [
            # Quality profile: larger/deeper model for capacity, with increased compute cost.
            ("FWS_UI", "ENABLE_UI", False),
            ("FWS_USE_VMAP", "USE_VMAP", True),
            ("FWS_BRAIN_MLP_DMODEL", "BRAIN_MLP_D_MODEL", 40),
        ],
    }

    rows = presets.get(PROFILE)
    if not rows: return

    g = globals()

    # Apply preset rows iff the corresponding env var is absent.
    for env_key, var_name, value in rows:
        if not _env_is_set(env_key):
            g[var_name] = value

    if int(g.get("BRAIN_MLP_D_MODEL", 0)) <= 0:
        raise ValueError(
            f"BRAIN_MLP_D_MODEL must be > 0 (got {g.get('BRAIN_MLP_D_MODEL')})"
        )
    g["BRAIN_MLP_FINAL_INPUT_WIDTH"] = 2 * int(g["BRAIN_MLP_D_MODEL"])


def _validate_config_invariants() -> None:
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

    # Basic dimensions / capacities
    _positive_int("GRID_WIDTH", GRID_WIDTH)
    _positive_int("GRID_HEIGHT", GRID_HEIGHT)
    _positive_int("MAX_AGENTS", MAX_AGENTS)
    _non_negative_int("START_AGENTS_PER_TEAM", START_AGENTS_PER_TEAM)

    # Non-fatal by default: startup request can exceed capacity and get truncated.
    total_initial_requested = int(START_AGENTS_PER_TEAM) * 2
    if total_initial_requested > int(MAX_AGENTS):
        _config_issue(
            f"Initial spawn request START_AGENTS_PER_TEAM*2={total_initial_requested} "
            f"exceeds MAX_AGENTS={int(MAX_AGENTS)}; runtime spawn may truncate."
        )

    # Common probability/rate knobs
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
    _prob("MAP_WALL_STRAIGHT_PROB", MAP_WALL_STRAIGHT_PROB)
    _prob("MAP_WALL_GAP_PROB", MAP_WALL_GAP_PROB)
    _prob("TELEMETRY_MOVE_EVENTS_SAMPLE_RATE", TELEMETRY_MOVE_EVENTS_SAMPLE_RATE)

    _positive_int("RESPAWN_RARE_MUTATION_TICK_WINDOW_TICKS", RESPAWN_RARE_MUTATION_TICK_WINDOW_TICKS)
    _positive_int("RESPAWN_SPAWN_NEAR_PARENT_RADIUS", RESPAWN_SPAWN_NEAR_PARENT_RADIUS)
    if float(RESPAWN_PARENT_SELECTION_SCORE_POWER) < 0.0:
        _config_issue(f"RESPAWN_PARENT_SELECTION_SCORE_POWER must be >= 0 (got {RESPAWN_PARENT_SELECTION_SCORE_POWER})")
    if float(RESPAWN_RARE_MUTATION_INHERITED_BRAIN_NOISE_STD) < 0.0:
        _config_issue(f"RESPAWN_RARE_MUTATION_INHERITED_BRAIN_NOISE_STD must be >= 0 (got {RESPAWN_RARE_MUTATION_INHERITED_BRAIN_NOISE_STD})")

    _positive_int("BRAIN_MLP_D_MODEL", BRAIN_MLP_D_MODEL)
    _positive_int("BRAIN_MLP_FINAL_INPUT_WIDTH", BRAIN_MLP_FINAL_INPUT_WIDTH)
    if int(BRAIN_MLP_FINAL_INPUT_WIDTH) != (2 * int(BRAIN_MLP_D_MODEL)):
        _config_issue(
            f"BRAIN_MLP_FINAL_INPUT_WIDTH must equal 2 * BRAIN_MLP_D_MODEL "
            f"(got {BRAIN_MLP_FINAL_INPUT_WIDTH} vs {2 * int(BRAIN_MLP_D_MODEL)})"
        )

    if BRAIN_MLP_ACTIVATION not in {"gelu", "relu", "silu"}:
        _config_issue(
            f"BRAIN_MLP_ACTIVATION must be one of {{'gelu','relu','silu'}} "
            f"(got {BRAIN_MLP_ACTIVATION!r})"
        )

    if BRAIN_MLP_NORM not in {"layernorm", "none"}:
        _config_issue(
            f"BRAIN_MLP_NORM must be one of {{'layernorm','none'}} "
            f"(got {BRAIN_MLP_NORM!r})"
        )

    if BRAIN_MLP_RAY_SUMMARY not in {"mean"}:
        _config_issue(
            f"BRAIN_MLP_RAY_SUMMARY currently supports only 'mean' "
            f"(got {BRAIN_MLP_RAY_SUMMARY!r})"
        )

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
            _config_issue(
                f"TEAM_BRAIN_MIX_SEQUENCE contains unknown kinds: {bad!r}"
            )

    total_mix_prob = (
        float(TEAM_BRAIN_MIX_P_WHISPERING_ABYSS)
        + float(TEAM_BRAIN_MIX_P_VEIL_OF_ECHOES)
        + float(TEAM_BRAIN_MIX_P_CATHEDRAL_OF_ASH)
        + float(TEAM_BRAIN_MIX_P_DREAMER_IN_BLACK_FOG)
        + float(TEAM_BRAIN_MIX_P_OBSIDIAN_PULSE)
    )
    if total_mix_prob <= 0.0:
        _config_issue("At least one TEAM_BRAIN_MIX_P_* probability must be > 0")

    _prob("CATASTROPHE_GLOBAL_ATTENUATION_FACTOR", CATASTROPHE_GLOBAL_ATTENUATION_FACTOR)
    _prob("CATASTROPHE_REGIONAL_ATTENUATION_FACTOR", CATASTROPHE_REGIONAL_ATTENUATION_FACTOR)
    _prob("CATASTROPHE_POLARITY_SPLIT_MAGNITUDE_SCALE", CATASTROPHE_POLARITY_SPLIT_MAGNITUDE_SCALE)
    _prob("CATASTROPHE_POSITIVE_BAND_DORMANCY_FRACTION", CATASTROPHE_POSITIVE_BAND_DORMANCY_FRACTION)
    _prob("CATASTROPHE_DYNAMIC_BASE_HAZARD_PROB", CATASTROPHE_DYNAMIC_BASE_HAZARD_PROB)
    _prob("CATASTROPHE_DYNAMIC_MAX_HAZARD_PROB", CATASTROPHE_DYNAMIC_MAX_HAZARD_PROB)
    _positive_int("CATASTROPHE_DEFAULT_DURATION_TICKS", max(1, int(CATASTROPHE_DEFAULT_DURATION_TICKS)))
    _non_negative_int("CATASTROPHE_DYNAMIC_COOLDOWN_TICKS", CATASTROPHE_DYNAMIC_COOLDOWN_TICKS)
    _non_negative_int("CATASTROPHE_DYNAMIC_MIN_INTERVAL_TICKS", CATASTROPHE_DYNAMIC_MIN_INTERVAL_TICKS)
    _non_negative_int("CATASTROPHE_DYNAMIC_MAX_INTERVAL_TICKS", CATASTROPHE_DYNAMIC_MAX_INTERVAL_TICKS)
    _positive_int("VIEWER_SIDE_PANEL_WIDTH", VIEWER_SIDE_PANEL_WIDTH)
    _positive_int("VIEWER_BOTTOM_PANEL_HEIGHT", VIEWER_BOTTOM_PANEL_HEIGHT)
    _positive_int("VIEWER_MIN_WIDTH", VIEWER_MIN_WIDTH)
    _positive_int("VIEWER_MIN_HEIGHT", VIEWER_MIN_HEIGHT)
    if int(CATASTROPHE_DYNAMIC_MAX_INTERVAL_TICKS) > 0 and int(CATASTROPHE_DYNAMIC_MAX_INTERVAL_TICKS) < int(CATASTROPHE_DYNAMIC_MIN_INTERVAL_TICKS):
        _config_issue(
            f"CATASTROPHE_DYNAMIC_MAX_INTERVAL_TICKS must be >= CATASTROPHE_DYNAMIC_MIN_INTERVAL_TICKS "
            f"(got {CATASTROPHE_DYNAMIC_MAX_INTERVAL_TICKS} < {CATASTROPHE_DYNAMIC_MIN_INTERVAL_TICKS})"
        )
    if CATASTROPHE_DYNAMIC_BAND_AXIS_MODE not in {"random", "vertical", "horizontal"}:
        _config_issue(
            f"CATASTROPHE_DYNAMIC_BAND_AXIS_MODE must be one of {{'random','vertical','horizontal'}} "
            f"(got {CATASTROPHE_DYNAMIC_BAND_AXIS_MODE!r})"
        )
    if CATASTROPHE_MANUAL_POSITIVE_BAND_AXIS not in {"vertical", "horizontal"}:
        _config_issue(
            f"CATASTROPHE_MANUAL_POSITIVE_BAND_AXIS must be one of {{'vertical','horizontal'}} "
            f"(got {CATASTROPHE_MANUAL_POSITIVE_BAND_AXIS!r})"
        )

    # Profile name sanity (warn/strict, but do not mutate the value)
    if PROFILE not in {"default", "debug", "train_fast", "train_quality"}:
        _config_issue(
            f"Unrecognized PROFILE={PROFILE!r}. "
            "Known profiles: default/debug/train_fast/train_quality"
        )


# Apply profiles after all defaults/env values are initialized.
_apply_profile_overrides()
_validate_config_invariants()

def summary_str() -> str:
    """
    Return a compact one-line description of the run configuration.

    Intended use:
    - startup logs
    - checkpoint metadata
    - telemetry run headers
    - quick sanity checks when running multiple experiments

    Note:
    - This is intentionally concise and does NOT include all knobs.
    - For full reproducibility, log environment overrides and/or full config dumps elsewhere.
    """
    return (
        f"[Neural Siege: Custom] "
        f"dev={DEVICE.type} "
        f"grid={GRID_WIDTH}x{GRID_HEIGHT} "
        f"start={START_AGENTS_PER_TEAM}/team "
        f"obs={OBS_DIM} acts={NUM_ACTIONS} "
        f"AMP={'on' if AMP_ENABLED else 'off'}"
    )

