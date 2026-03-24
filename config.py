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
# BEGINNER OPERATING GUIDE
# =============================================================================
# Think of this file as the simulation's control panel.
# A newcomer usually wants answers to four questions first:
#   1) "Which knobs are safe to tweak right away?"
#   2) "Which knobs are deep contracts that can break checkpoints/models?"
#   3) "Which combinations of knobs correspond to actual runtime modes?"
#   4) "When I increase or decrease a value, what broad effect should I expect?"
#
# The safest mental model is:
#   - OUTPUT / UI / TELEMETRY / CHECKPOINT knobs are mostly operational knobs.
#   - WORLD SIZE / POPULATION / REWARD / PPO knobs are behavior/tuning knobs.
#   - OBSERVATION / ACTION / BRAIN-WIDTH knobs are contract knobs.
#
# A practical reading order for new users:
#   - read PROFILE first,
#   - then read fresh-vs-resume knobs,
#   - then world size and map knobs,
#   - then respawn/reward/PPO knobs,
#   - and only then touch observation or brain-architecture settings.
#
# Quick mode cheat sheet
# ----------------------
# The repository does not use a normal CLI flag parser for most runtime choices.
# The normal way to operate it is by setting FWS_* environment variables.
#
# Common modes in the current codebase:
#   - Fresh UI exploration:
#       FWS_UI=1
#       FWS_CHECKPOINT_PATH=""
#   - Fresh headless training / benchmarking:
#       FWS_UI=0
#       FWS_CHECKPOINT_PATH=""
#   - Resume an existing run:
#       FWS_CHECKPOINT_PATH=/path/to/checkpoint
#       optionally FWS_UI=1 or 0
#   - Inspector / view-without-side-effects mode:
#       FWS_INSPECTOR_MODE=ui_no_output
#       or FWS_INSPECTOR_UI_NO_OUTPUT=1
#
# How to interpret number knobs
# -----------------------------
# A few broad patterns repeat throughout the file:
#   - probabilities usually mean a float in [0, 1]
#   - counts and widths are usually positive integers
#   - many cadence knobs treat 0 as "off" or "unlimited", but not all of them
#   - "ratio" knobs usually scale with map size or a normalized score
#   - bigger values usually mean more capacity / more history / more compute
#   - smaller values usually mean lighter runs / more aggressive pruning / less detail
#
# Safe beginner changes
# ---------------------
# Usually safe:
#   - PROFILE
#   - ENABLE_UI
#   - RESULTS_DIR
#   - CHECKPOINT_EVERY_TICKS
#   - HEADLESS_PRINT_* and TELEMETRY_* cadence knobs
#   - GRID_WIDTH / GRID_HEIGHT / START_AGENTS_PER_TEAM / RANDOM_WALLS
#   - PPO rollout and reward coefficients
#
# Change with real care:
#   - OBS_SCHEMA
#   - RAY_TOKEN_COUNT
#   - RAY_FEAT_DIM
#   - RICH_BASE_DIM / INSTINCT_DIM / OBS_DIM (derived contracts)
#   - NUM_ACTIONS
#   - BRAIN_MLP_* widths/depths when loading old checkpoints
#   - brain-kind routing / team-brain assignment when comparing experiments
#
# Why that warning matters:
# these contract knobs affect not just one module, but model construction,
# observation slicing, checkpoint compatibility, PPO buffer shapes, and often
# telemetry interpretation too.
# =============================================================================


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
# This section decides what kind of run you are creating:
#   - a brand-new lineage,
#   - a continuation of an existing lineage,
#   - or a branch/fork of an existing checkpoint.
#
# New users often underestimate how important these knobs are.
# They do not change combat mechanics or PPO math, but they strongly affect
# where outputs go, whether CSV append is allowed, and whether resumed runs are
# treated as one continuous experiment or a new branch.

# PROFILE
# -------
# Valid values:
#   "default", "debug", "train_tiny", "train_fast", "train_quality"
#
# What it does:
#   Applies a bundle of preset overrides for values that were not explicitly set
#   via environment variables.
#
# When to set:
#   - use "debug" for small, quick, interactive runs
#   - use "train_tiny" when you need the lowest-memory self-centric MLP setup
#     and want every shipped brain variant to stay under ~50k params/agent
#   - use "train_fast" when throughput matters more than representation width
#   - use "train_quality" when you want somewhat higher model capacity
#   - use "default" when you want only the literal per-knob defaults
#
# What each mode actually changes in this file:
#   - "default":
#       no bundled overrides; every knob keeps its literal hard-coded default
#   - "debug":
#       smaller map, smaller population, fewer walls, UI on, lower FPS cap,
#       vmap off; this is the friendliest mode for local stepping and viewer use
#   - "train_tiny":
#       UI off, vmap on, aggressively narrowed brain widths/depths; intended for
#       low-VRAM runs where per-agent brain size must stay compact
#   - "train_fast":
#       UI off, vmap on, narrower brain towers/fusion widths; optimized for
#       cheaper/faster throughput rather than maximum model capacity
#   - "train_quality":
#       UI off, vmap on, wider brain towers/fusion widths; heavier but offers
#       more representational capacity than train_fast
#
# Important precedence detail:
# if you set a knob explicitly in the environment, that explicit value beats the
# profile. Example: FWS_PROFILE=debug and FWS_GRID_W=160 results in width=160,
# not the debug width of 80.
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
# This is one of the safest knobs to change because it affects artifact
# placement, not simulation semantics.
RESULTS_DIR: str = _env_str("FWS_RESULTS_DIR", "results").strip()

# CHECKPOINT_PATH
# ---------------
# Path to a checkpoint for resume.
# Leave empty to start fresh.
# If this is empty:
#   main.py builds a fresh world and a fresh output lineage.
# If this is non-empty:
#   main.py loads the checkpoint payload first, then decides whether outputs
#   continue in-place or branch into a new run directory based on the resume
#   knobs below.
CHECKPOINT_PATH: str = _env_str("FWS_CHECKPOINT_PATH", "").strip()

# RESUME_OUTPUT_CONTINUITY
# ------------------------
# If True, a resumed run appends back into the original run directory.
# Use True when you want one continuous lineage.
# Use False when you want resume to produce separate output trees.
# This matters mainly for:
#   - CSV append behavior,
#   - checkpoint folder continuity,
#   - telemetry numbering / lineage,
#   - and how easy it is to answer "is this still the same run?"
RESUME_OUTPUT_CONTINUITY: bool = _env_bool("FWS_RESUME_OUTPUT_CONTINUITY", True)

# RESUME_FORCE_NEW_RUN
# --------------------
# Stronger override than continuity: even if resuming from a checkpoint, create
# a fresh run folder.
# Use this when a resumed branch should be treated as a new experiment branch.
# In plain language:
#   continuity=True says "resume into the same lineage if possible"
#   force_new_run=True says "no, make a new lineage anyway"
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
# Runtime honesty:
# the provided orchestration path does not currently appear to schedule active
# wall-clock autosaves from this knob. The practical periodic-save path in the
# current repository is CHECKPOINT_EVERY_TICKS.
AUTOSAVE_EVERY_SEC: int = _env_int("FWS_AUTOSAVE_EVERY_SEC", 3600)


# =============================================================================
# CHECKPOINTING
# =============================================================================

# CHECKPOINT_EVERY_TICKS
# ----------------------
# Save a checkpoint every N ticks.
# 0 disables tick-based periodic checkpoints.
# Set lower values for safer long runs; set higher values for less I/O.
CHECKPOINT_EVERY_TICKS: int = _env_int("FWS_CHECKPOINT_EVERY_TICKS", 25_000)

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
HEADLESS_PRINT_EVERY_TICKS: int = _env_int("FWS_HEADLESS_PRINT_EVERY_TICKS", 1000)

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
HEADLESS_PRINT_GPU: bool = _env_bool("FWS_HEADLESS_PRINT_GPU", True)


# =============================================================================
# SCIENTIFIC RECORDING / TELEMETRY
# =============================================================================
# Telemetry is the experiment notebook of the runtime.
# When enabled, it writes structured artifacts that let you inspect what
# happened after the run ends instead of depending only on live prints.
#
# Beginner rule of thumb:
#   - leave telemetry on for any run you may want to compare later
#   - reduce cadence / event volume before disabling it entirely
#   - births/deaths/kills are usually cheap and useful
#   - movement and raw damage streams are the easiest way to create huge outputs

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
TELEMETRY_TICK_SUMMARY_EVERY: int = _env_int("FWS_TELEM_TICK_SUMMARY_EVERY", 300)

# Periodic sampling / flush cadences.
# Lower = more frequent measurement, better temporal resolution, more overhead.
# Higher = lighter logging, coarser history.
# Runtime honesty:
# these three cadence knobs are part of the documented config surface, but the
# current telemetry implementation more directly reads summary/validation/flush
# and specific move/PPO cadences than dedicated consumers for all three of these
# snapshot-style knobs. Treat them as forward-compatible telemetry controls
# unless you have verified a consumer in your branch.
TELEMETRY_TICK_METRICS_EVERY: int = _env_int("FWS_TELEM_TICK_EVERY", 300)
TELEMETRY_SNAPSHOT_EVERY: int = _env_int("FWS_TELEM_SNAPSHOT_EVERY", 1200)
TELEMETRY_REGISTRY_SNAPSHOT_EVERY: int = _env_int("FWS_TELEM_REG_EVERY", 600)
# TELEMETRY_VALIDATE_EVERY:
#   how often telemetry self-checks run.
# TELEMETRY_PERIODIC_FLUSH_EVERY:
#   how often buffered telemetry is forced to disk.
# Lower validation/flush values catch issues and persist data sooner.
# Higher values usually reduce overhead.
TELEMETRY_VALIDATE_EVERY: int = _env_int("FWS_TELEM_VALIDATE_EVERY", 1200)
TELEMETRY_PERIODIC_FLUSH_EVERY: int = _env_int("FWS_TELEM_FLUSH_EVERY", 2400)

# Buffer sizes.
# Increase for throughput and fewer writes.
# Decrease for lower memory pressure and smaller crash-loss windows.
# EVENT_CHUNK is the large event buffer.
# TICK_CHUNK is the lighter-weight tick-row buffer.
TELEMETRY_EVENT_CHUNK_SIZE: int = _env_int("FWS_TELEM_EVENT_CHUNK", 200_000)
TELEMETRY_TICK_CHUNK_SIZE: int = _env_int("FWS_TELEM_TICK_CHUNK", 20_000)

# Event-type toggles.
# Enable what you need to study.
# Disable high-volume streams when output size matters more.
# TELEMETRY_LOG_PPO is useful only when PPO is active.
# TELEMETRY_LOG_DAMAGE and TELEMETRY_LOG_MOVES are the two knobs most likely to
# materially increase artifact volume.
TELEMETRY_LOG_BIRTHS: bool = _env_bool("FWS_TELEM_BIRTHS", True)
TELEMETRY_LOG_DEATHS: bool = _env_bool("FWS_TELEM_DEATHS", True)
TELEMETRY_LOG_DAMAGE: bool = _env_bool("FWS_TELEM_DAMAGE", False)
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
TELEMETRY_PPO_RICH_LEVEL: str = _env_str("FWS_TELEM_PPO_RICH_LEVEL", "minibatch").strip().lower()

# TELEMETRY_PPO_RICH_FLUSH_EVERY
# ------------------------------
# Flush frequency for PPO-rich CSV rows.
# 1 is safest and most immediate.
# Larger values reduce write calls.
TELEMETRY_PPO_RICH_FLUSH_EVERY: int = _env_int("FWS_TELEM_PPO_RICH_FLUSH_EVERY", 1)

# TELEMETRY_APPEND_SCHEMA_STRICT
# ------------------------------
# Require matching telemetry CSV schema when appending.
# Leave this enabled unless you intentionally accept mixing unlike telemetry
# files into one lineage.
TELEMETRY_APPEND_SCHEMA_STRICT: bool = _env_bool("FWS_TELEM_APPEND_SCHEMA_STRICT", True)

# Headless live summary sidecar switches.
# These control a CSV summary surface that is separate from normal console
# prints. It is useful for remote/headless monitoring.
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
# MOVE_EVENTS_EVERY:
#   0 disables sampled per-agent move events while cheaper aggregates can still
#   remain available.
# MOVE_EVENTS_MAX_PER_TICK:
#   hard cap against event explosions on dense ticks.
# MOVE_EVENTS_SAMPLE_RATE:
#   fraction in [0, 1]; lower means lighter raw movement logs.
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
# If True, telemetry can stop the run instead of merely recording the anomaly.
TELEMETRY_ABORT_ON_ANOMALY: bool = _env_bool("FWS_TELEM_ABORT", False)

# End-of-run report artifact switches.
# These control optional reporting layers generated at shutdown.
TELEMETRY_REPORT_ENABLE: bool = _env_bool("FWS_TELEM_REPORT", True)
TELEMETRY_REPORT_EXCEL: bool = _env_bool("FWS_TELEM_EXCEL", False)
TELEMETRY_REPORT_PNG: bool = _env_bool("FWS_TELEM_PNG", True)


# =============================================================================
# HARDWARE ACCELERATION AND TORCH EXECUTION MODE
# =============================================================================
# These knobs decide where tensors live and how aggressively the runtime tries
# to accelerate inference/training.
#
# Simple starting advice:
#   - CPU + AMP off is the easiest setup for debugging
#   - CUDA + AMP on is usually the best throughput setup when available
#   - if bucketed inference behaves oddly, try USE_VMAP=0 before changing model
#     architecture or PPO settings

# USE_CUDA
# --------
# Becomes True only if the knob requests CUDA AND CUDA is actually available.
# Set False when forcing CPU runs for debugging or reproducibility checks.
# If CUDA is requested but unavailable, this resolves safely to CPU.
USE_CUDA: bool = _env_bool("FWS_CUDA", True) and torch.cuda.is_available()

# Canonical device used throughout the codebase.
# DEVICE and TORCH_DEVICE intentionally answer the same question so downstream
# modules have one stable placement answer.
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
# There is no direct dtype env knob here.
# The rule is:
#   CUDA + AMP enabled -> float16
#   otherwise          -> float32
# So turning AMP on while staying on CPU does not make CPU tensors half-precision.
TORCH_DTYPE = torch.float16 if (USE_CUDA and AMP_ENABLED) else torch.float32

# VMAP controls.
# These affect batched execution across many independent models.
# USE_VMAP:
#   master switch for the torch.func/vmap inference path.
# VMAP_MIN_BUCKET:
#   minimum bucket size before the code even tries vmap.
#   Lower values try acceleration more aggressively.
#   Higher values stay on the simpler loop path for small groups.
# VMAP_DEBUG:
#   emits debug prints about path choice and fallback.
USE_VMAP: bool = _env_bool("FWS_USE_VMAP", True)
VMAP_MIN_BUCKET: int = _env_int("FWS_VMAP_MIN_BUCKET", 8)
VMAP_DEBUG: bool = _env_bool("FWS_VMAP_DEBUG", False)

# VMAP_STACK_CACHE_MAX
# --------------------
# Max cached stacked-state entries for torch.func/vmap inference.
# <= 0 disables the cache.
# Increase when repeated bucket structures are common and memory is available.
# Decrease when memory pressure matters more than cache reuse.
VMAP_STACK_CACHE_MAX: int = _env_int("FWS_VMAP_STACK_CACHE_MAX", 256)


# =============================================================================
# WORLD SCALE, CAPACITY, AND RUNTIME LENGTH
# =============================================================================
# These are the fastest knobs for making the simulation feel "small and fast" or
# "large and heavy".
# Broad tradeoff:
#   bigger world + more agents = richer ecology + slower ticks
#   smaller world + fewer agents = faster iteration + less tactical diversity

# Grid size.
# Increase for larger, sparser, slower worlds.
# Decrease for faster, denser experiments.
# Remember that area scales as width * height.
# Doubling both width and height roughly quadruples battlefield area.
GRID_WIDTH: int = _env_int("FWS_GRID_W", 100)
GRID_HEIGHT: int = _env_int("FWS_GRID_H", 100)

# Starting population per team.
# Increase for richer population dynamics and heavier compute.
# Decrease for fast debugging.
# This is per-team, so total requested opening population is roughly
# 2 * START_AGENTS_PER_TEAM.
START_AGENTS_PER_TEAM: int = _env_int("FWS_START_PER_TEAM", 180)

# Total registry slot capacity.
# Must be large enough for the starting population plus respawn dynamics.
# If MAX_AGENTS is too low, the world simply does not have enough slots to hold
# the population pressure you asked for.
MAX_AGENTS: int = _env_int("FWS_MAX_AGENTS", 560)

# Run length cap.
# 0 means unlimited.
# Use 0 for open-ended runs; use a finite value for smoke tests, benchmarks, and
# controlled training horizons.
TICK_LIMIT: int = _env_int("FWS_TICK_LIMIT", 0)

# Throttle target ticks per second.
# 0 means unthrottled.
# Runtime honesty:
# no active use of TARGET_TPS appears in the current provided runtime loop.
# It is best read as a reserved/forward-compatible pacing knob.
TARGET_TPS: int = _env_int("FWS_TARGET_TPS", 0)

# Strict agent-registry schema size. Do not casually change.
AGENT_FEATURES: int = 10


# =============================================================================
# MAP TOPOLOGY, OBJECTIVES, AND HEALING LANDSCAPE
# =============================================================================
# This section shapes the battlefield itself:
#   - walls control pathing and chokepoints
#   - heal zones control sustain pockets
#   - capture points control positional scoring pressure

# Random wall count.
# Raise for more obstacle structure and tactical routing.
# Lower for more open battlefields.
RANDOM_WALLS: int = _env_int("FWS_RAND_WALLS", 12)

# Wall segment length bounds.
# WALL_SEG_MIN is the shortest sampled wall segment.
# WALL_SEG_MAX is the longest sampled wall segment.
# Lower lengths create fragmented cover.
# Higher lengths create longer corridors and harder partitions.
# Keep WALL_SEG_MAX >= WALL_SEG_MIN.
WALL_SEG_MIN: int = _env_int("FWS_WALL_SEG_MIN", 5)
WALL_SEG_MAX: int = _env_int("FWS_WALL_SEG_MAX", 40)

# Boundary margin for wall placement.
# Larger margins keep edges clearer.
# Smaller margins allow more edge-adjacent blockage.
WALL_AVOID_MARGIN: int = _env_int("FWS_WALL_MARGIN", 3)

# Straightness and gap probabilities for wall generation.
# Higher straight probability -> longer corridors.
# Higher gap probability -> more openings/passability.
MAP_WALL_STRAIGHT_PROB: float = _env_float("FWS_MAP_WALL_STRAIGHT_PROB", 0.65)
MAP_WALL_GAP_PROB: float = _env_float("FWS_MAP_WALL_GAP_PROB", 0.24)

# Heal-zone parameters.
# Raise count/size/rate when sustain should matter more.
# Lower them when combat lethality or territorial churn should dominate.
# HEAL_ZONE_COUNT:
#   how many heal rectangles are sampled.
# HEAL_ZONE_SIZE_RATIO:
#   rectangle side lengths as a fraction of map height/width.
# HEAL_RATE:
#   HP restored per tick while standing on an active heal cell.
# Higher HEAL_RATE or larger heal zones can noticeably slow population collapse.
HEAL_ZONE_COUNT: int = _env_int("FWS_HEAL_COUNT", 6)
HEAL_ZONE_SIZE_RATIO: float = _env_float("FWS_HEAL_SIZE_RATIO", 0.055)
HEAL_RATE: float = _env_float("FWS_HEAL_RATE", 0.0032)


# =============================================================================
# HEAL-ZONE CATASTROPHE SCHEDULER
# =============================================================================
# This subsystem exists to break overly stable heal-zone camping.
# Read it as "anti-stagnation policy for healing geography".
# If heal control becomes too dominant, catastrophe temporarily suppresses some
# heal zones so battles are forced back into motion.

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
# Larger cooldown = rarer catastrophes.
# Smaller cooldown = more frequent disruption.
CATASTROPHE_COOLDOWN_TICKS: int = max(0, _env_int("FWS_CATASTROPHE_COOLDOWN_TICKS", 5_000))

# Duration of one catastrophe event.
# Larger duration keeps heal suppression active longer.
# Smaller duration makes catastrophe feel like a brief shock.
CATASTROPHE_DURATION_TICKS: int = max(1, _env_int("FWS_CATASTROPHE_DURATION_TICKS", 1_200))

# Never suppress all heal zones; keep at least this many active.
# This is a safety knob against completely removing sustain from the map.
CATASTROPHE_MIN_ACTIVE_HEAL_ZONES: int = max(1, _env_int("FWS_CATASTROPHE_MIN_ACTIVE_HEAL_ZONES", 2))

# Minimum total zone count required before catastrophe logic is allowed.
# If the map has too few heal zones overall, catastrophe is skipped entirely.
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
# This is a fraction in [0, 1].
# Higher values mean dynamic mode waits for denser/sustained heal occupancy.
# Lower values make it easier to trigger.
CATASTROPHE_DYNAMIC_HEAL_OCCUPANCY_THRESHOLD: float = _env_float(
    "FWS_CATASTROPHE_DYNAMIC_HEAL_OCCUPANCY_THRESHOLD",
    0.20,
)

# Number of consecutive ticks the dynamic signal must persist.
# Larger values make dynamic triggering more conservative.
# Smaller values make it more reactive/noisy.
CATASTROPHE_DYNAMIC_SUSTAIN_TICKS: int = max(
    1,
    _env_int("FWS_CATASTROPHE_DYNAMIC_SUSTAIN_TICKS", 50),
)

# Fractional severity knobs.
# Raise them to suppress more healing support.
# Lower them to make catastrophes milder.
CATASTROPHE_SMALL_SUPPRESS_FRACTION: float = _env_float(
    "FWS_CATASTROPHE_SMALL_SUPPRESS_FRACTION",
    0.30,
)
CATASTROPHE_MEDIUM_SUPPRESS_FRACTION: float = _env_float(
    "FWS_CATASTROPHE_MEDIUM_SUPPRESS_FRACTION",
    0.60,
)
CATASTROPHE_CLUSTER_SURVIVOR_FRACTION: float = _env_float(
    "FWS_CATASTROPHE_CLUSTER_SURVIVOR_FRACTION",
    0.20,
)

# Log catastrophe trigger/clear events to console.
# Safe operational knob: changes observability, not mechanics.
CATASTROPHE_LOG_EVENTS: bool = _env_bool("FWS_CATASTROPHE_LOG_EVENTS", True)


# Capture points.
# Raise count/size/reward when territorial control should matter more.
# Lower reward to make CP mostly informational/spatial rather than score-driving.
# CP_COUNT:
#   how many capture zones exist.
# CP_SIZE_RATIO:
#   rectangle side lengths as a map fraction.
# CP_REWARD_PER_TICK:
#   score/reward contribution for occupying a control point.
#   Raising this makes map control more important relative to pure combat.
CP_COUNT: int = _env_int("FWS_CP_COUNT", 7)
CP_SIZE_RATIO: float = _env_float("FWS_CP_SIZE_RATIO", 0.07)
CP_REWARD_PER_TICK: float = _env_float("FWS_CP_REWARD", 0)


# =============================================================================
# COMBAT BIOLOGY, UNIT CLASSES, METABOLISM, AND SENSING
# =============================================================================
# This section defines "what a unit is" in the simulation:
#   - how much health it has,
#   - how hard it hits,
#   - how far it sees,
#   - how fast passive HP drain happens,
#   - and how much class asymmetry exists between soldiers and archers.

UNIT_SOLDIER_ID: int = 1
UNIT_ARCHER_ID: int = 2
UNIT_SOLDIER: int = UNIT_SOLDIER_ID
UNIT_ARCHER: int = UNIT_ARCHER_ID

# HP and damage scales.
# Increase HP for longer engagements.
# Increase ATK for higher lethality.
# MAX_HP:
#   normalization anchor used by several systems.
# SOLDIER_HP / ARCHER_HP:
#   class-specific maximum health values.
# BASE_ATK / SOLDIER_ATK / ARCHER_ATK:
#   damage magnitudes used in combat logic and observation normalization.
# Higher HP with unchanged attack slows battles.
# Higher attack with unchanged HP makes the world deadlier and usually noisier.
MAX_HP: float = _env_float("FWS_MAX_HP", 1.0)
SOLDIER_HP: float = _env_float("FWS_SOLDIER_HP", 1.0)
ARCHER_HP: float = _env_float("FWS_ARCHER_HP", 0.65)
BASE_ATK: float = _env_float("FWS_BASE_ATK", 0.20)
SOLDIER_ATK: float = _env_float("FWS_SOLDIER_ATK", 0.16)
ARCHER_ATK: float = _env_float("FWS_ARCHER_ATK", 0.11)
MAX_ATK: float = max(SOLDIER_ATK, ARCHER_ATK, BASE_ATK, 1e-6)

# Archer-specific mechanics.
# ARCHER_RANGE governs the longest legal archer attack range in the 41-action
# layout, capped by the action schema's 4-range design.
# ARCHER_LOS_BLOCKS_WALLS decides whether walls block ranged line-of-sight.
# Turning LOS blocking off makes archers tactically simpler and stronger across
# cluttered maps.
ARCHER_RANGE: int = _env_int("FWS_ARCHER_RANGE", 4)
ARCHER_LOS_BLOCKS_WALLS: bool = _env_bool("FWS_ARCHER_BLOCK_LOS", True)

# Metabolism drain.
# Raise it to make healing/objectives/combat more urgent.
# Lower it to allow more passive roaming or longer standoffs.
# META_SOLDIER_HP_PER_TICK and META_ARCHER_HP_PER_TICK are passive HP drains.
# Higher drain forces units to find value quickly or die off faster.
METABOLISM_ENABLED: bool = _env_bool("FWS_META_ON", True)
META_SOLDIER_HP_PER_TICK: float = _env_float("FWS_META_SOLDIER", 0.0026)
META_ARCHER_HP_PER_TICK: float = _env_float("FWS_META_ARCHER", 0.0020)

# Vision ranges.
# Raise to make agents more informed and long-range aware.
# Lower to make combat more local and uncertain.
# These values also determine the maximum raycast depth, so they affect both
# sensing reach and observation scale.
VISION_RANGE_SOLDIER: int = _env_int("FWS_VISION_SOLDIER", 7)
VISION_RANGE_ARCHER: int = _env_int("FWS_VISION_ARCHER", 14)

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
# In the live engine this radius governs the local-density / threat instinct
# features, not the 32 directional first-hit rays.
INSTINCT_RADIUS: int = _env_int("FWS_INSTINCT_RADIUS", 16)


# =============================================================================
# OBSERVATION LAYOUT CONTRACT
# =============================================================================
# These values are schema contracts. Changing them is not just a tuning choice.
# It changes the meaning and dimensionality of observations, which affects model
# architecture, checkpoint compatibility, PPO buffer shapes, and feature split
# logic in other modules.
# =============================================================================

# Number of ray tokens per observation.
# The live first-hit runtime still emits exactly 32 directional rays today.
# Keep the env surface only as a fail-fast contract guard until the runtime is
# generalized end to end.
# Important beginner distinction:
#   the action space uses 8 compass directions,
#   but the observation space uses 32 sensing directions.
_SUPPORTED_RAY_TOKEN_COUNT = 32
RAY_TOKEN_COUNT: int = _env_int("FWS_RAY_TOKENS", _SUPPORTED_RAY_TOKEN_COUNT)
if RAY_TOKEN_COUNT != _SUPPORTED_RAY_TOKEN_COUNT:
    _config_issue(
        f"RAY_TOKEN_COUNT currently supports only {_SUPPORTED_RAY_TOKEN_COUNT} first-hit rays "
        f"(got {RAY_TOKEN_COUNT}); using {_SUPPORTED_RAY_TOKEN_COUNT}"
    )
    RAY_TOKEN_COUNT = _SUPPORTED_RAY_TOKEN_COUNT

# Features carried by each ray token.
# Strict schema constant.
# The current 8-feature first-hit layout is:
#   onehot6(hit type) + normalized distance + normalized HP at the first hit.
# Changing this would require coordinated engine + obs-splitting + brain updates.
RAY_FEAT_DIM: int = 8

# Flattened ray block size.
# This is derived, not an independent tuning knob.
RAYS_FLAT_DIM: int = RAY_TOKEN_COUNT * RAY_FEAT_DIM

# Observation schema selection.
# This is part of the model/checkpoint contract, not a cosmetic runtime toggle.
# legacy_full_v1:
#   older wider scalar tail with more team/global channels and padding.
# self_centric_v1:
#   newer default schema focused on the agent's own state plus compact
#   self-centric PPO-related context.
OBS_SCHEMA_LEGACY_FULL_V1: str = "legacy_full_v1"
OBS_SCHEMA_SELF_CENTRIC_V1: str = "self_centric_v1"
OBS_SCHEMA_ALLOWED = (
    OBS_SCHEMA_LEGACY_FULL_V1,
    OBS_SCHEMA_SELF_CENTRIC_V1,
)
OBS_SCHEMA: str = _env_str("FWS_OBS_SCHEMA", OBS_SCHEMA_SELF_CENTRIC_V1).strip().lower()
if OBS_SCHEMA not in OBS_SCHEMA_ALLOWED:
    _config_warn(
        f"Unknown observation schema {OBS_SCHEMA!r}; using {OBS_SCHEMA_SELF_CENTRIC_V1!r}"
    )
    OBS_SCHEMA = OBS_SCHEMA_SELF_CENTRIC_V1

# Historical full-schema dimensions kept explicit for schema/checkpoint validation.
# legacy_full_v1 = 23 rich-base + 4 instinct = 27 scalar-tail features.
LEGACY_FULL_RICH_BASE_DIM: int = 23
LEGACY_FULL_INSTINCT_DIM: int = 4
LEGACY_FULL_RICH_TOTAL_DIM: int = LEGACY_FULL_RICH_BASE_DIM + LEGACY_FULL_INSTINCT_DIM
LEGACY_FULL_OBS_DIM: int = RAYS_FLAT_DIM + LEGACY_FULL_RICH_TOTAL_DIM

# New self-centric schema dimensions.
# self_centric_v1 = 16 rich-base + 4 instinct = 20 scalar-tail features.
SELF_CENTRIC_RICH_BASE_DIM: int = 16
SELF_CENTRIC_INSTINCT_DIM: int = 4
SELF_CENTRIC_RICH_TOTAL_DIM: int = SELF_CENTRIC_RICH_BASE_DIM + SELF_CENTRIC_INSTINCT_DIM
SELF_CENTRIC_OBS_DIM: int = RAYS_FLAT_DIM + SELF_CENTRIC_RICH_TOTAL_DIM

# Active rich non-ray feature dimensions.
# These resolve from OBS_SCHEMA above; do not treat them as independent knobs.
if OBS_SCHEMA == OBS_SCHEMA_SELF_CENTRIC_V1:
    RICH_BASE_DIM: int = SELF_CENTRIC_RICH_BASE_DIM
    INSTINCT_DIM: int = SELF_CENTRIC_INSTINCT_DIM
else:
    RICH_BASE_DIM: int = LEGACY_FULL_RICH_BASE_DIM
    INSTINCT_DIM: int = LEGACY_FULL_INSTINCT_DIM
RICH_TOTAL_DIM: int = RICH_BASE_DIM + INSTINCT_DIM

# Final observation size for the active schema.
OBS_DIM: int = RAYS_FLAT_DIM + RICH_TOTAL_DIM

# Semantic grouping used by observation splitting/token semantics.
# These indices must stay aligned with actual feature construction elsewhere.
# They are used when code asks for semantic slices like "own_context" or
# "combat_context". If feature ordering changes, these must change too.
if OBS_SCHEMA == OBS_SCHEMA_SELF_CENTRIC_V1:
    SEMANTIC_RICH_BASE_INDICES = {
        "own_context": (0, 1, 2, 3, 4, 5, 6),
        "world_context": (9,),
        "zone_context": (7, 8),
        "team_context": (),
        "combat_context": (10, 11, 12, 13, 14, 15),
    }
else:
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
# Current verified live schema is 41 actions:
#   0      = idle
#   1..8   = movement in 8 directions
#   9..40  = attacks in 8 direction blocks * 4 range columns
# The repository still contains traces of a 17-action legacy layout, but 41 is
# the active end-to-end contract.
NUM_ACTIONS: int = _env_int("FWS_NUM_ACTIONS", 41)


# =============================================================================
# POPULATION CONTROL, INITIAL SPAWNING, AND RESPAWN EVOLUTION
# =============================================================================
# This section governs how the world gets populated initially and how dead slots
# come back later.
#
# Beginner reading model:
#   - SPAWN_* affects only the opening layout
#   - RESP_* affects floor/period budgeting
#   - RESPAWN_* affects parent choice, mutation, child placement, lineage
#     behavior, and unit-class inheritance

# RESPAWN_ENABLED
# ---------------
# Master switch for reinforcement / repopulation.
# Keep True when studying long-lived population dynamics.
# Turn off when you want pure extinction-style combat.
# Runtime honesty:
# the persistent TickEngine path constructs RespawnController(RespawnCfg())
# directly, and RespawnCfg.enabled defaults to True. So this knob should be read
# as intended policy surface / compatibility knob, not an unquestionable hard
# off-switch, unless you also verify the controller construction path.
RESPAWN_ENABLED: bool = _env_bool("FWS_RESPAWN", True)

# Population floor per team.
# When a team drops below this level, respawn logic can help replenish it.
# Higher floor means the system fights extinction harder.
# Lower floor allows deeper collapses before intervention.
RESP_FLOOR_PER_TEAM: int = _env_int("FWS_RESP_FLOOR_PER_TEAM", 94)

# Hard cap on respawns per tick.
# Small values smooth population recovery.
# Large values allow faster rebounds but can create burstier density changes.
RESP_MAX_PER_TICK: int = _env_int("FWS_RESP_MAX_PER_TICK", 1)

# Periodic reinforcement window.
# RESP_PERIOD_TICKS:
#   how often the periodic budget logic is allowed to fire.
# RESP_PERIOD_BUDGET:
#   total number of respawns distributed when that window fires.
RESP_PERIOD_TICKS: int = _env_int("FWS_RESP_PERIOD_TICKS", 3_000)
RESP_PERIOD_BUDGET: int = _env_int("FWS_RESP_PERIOD_BUDGET", 28)

# Cooldown hysteresis to avoid rapid refill oscillation.
# Larger cooldown means the controller waits longer before reapplying floor
# pressure once a team recovers.
RESP_HYST_COOLDOWN_TICKS: int = _env_int("FWS_RESP_HYST_COOLDOWN_TICKS", 80)

# Wall margin for spawn placement.
# Larger values keep births farther from borders and some edge degeneracies.
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
# Probability in [0, 1].
# Higher values create more ranged-heavy openings.
SPAWN_ARCHER_RATIO: float = _env_float("FWS_SPAWN_ARCHER_RATIO", 0.40)

# Core respawn probabilities and placement search budget.
# RESPAWN_PROB_PER_DEAD:
#   compatibility-era knob; the new controller path documents this legacy field
#   as explicitly ignored.
# RESPAWN_SPAWN_TRIES:
#   max attempts to find a legal spawn cell.
# RESPAWN_MUTATION_STD:
#   Gaussian weight-noise scale for cloned brains.
# RESPAWN_CLONE_PROB:
#   probability of cloning from a parent instead of birthing a fresh brain.
# RESPAWN_USE_TEAM_ELITE / RESPAWN_RESET_OPT_ON_RESPAWN:
#   present on the config surface, but the current controller comments suggest
#   they are largely compatibility / future-facing rather than heavily active.
# RESPAWN_JITTER_RADIUS:
#   tiny local randomness in spawn placement.
# RESPAWN_COOLDOWN_TICKS:
#   per-slot/per-team respawn throttling interval.
# RESPAWN_BATCH_PER_TEAM:
#   batch size bias for one respawn step.
# RESPAWN_ARCHER_SHARE:
#   ranged-unit share during respawn/fresh-birth logic.
# RESPAWN_INTERIOR_BIAS:
#   bias toward interior placement rather than map edges.
RESPAWN_PROB_PER_DEAD: float = _env_float("FWS_RESPAWN_PROB", 0.10)
RESPAWN_SPAWN_TRIES: int = _env_int("FWS_RESPAWN_TRIES", 160)
RESPAWN_MUTATION_STD: float = _env_float("FWS_MUT_STD", 0.06)
RESPAWN_CLONE_PROB: float = _env_float("FWS_CLONE_PROB", 1)
RESPAWN_USE_TEAM_ELITE: bool = _env_bool("FWS_TEAM_ELITE", True)
RESPAWN_RESET_OPT_ON_RESPAWN: bool = _env_bool("FWS_RESET_OPT", True)
RESPAWN_JITTER_RADIUS: int = _env_int("FWS_RESP_JITTER", 1)
RESPAWN_COOLDOWN_TICKS: int = _env_int("FWS_RESPAWN_CD", 90)
RESPAWN_BATCH_PER_TEAM: int = _env_int("FWS_RESPAWN_BATCH", 4)
RESPAWN_ARCHER_SHARE: float = _env_float("FWS_RESPAWN_ARCHER_SHARE", 0.45)
RESPAWN_INTERIOR_BIAS: float = _env_float("FWS_RESPAWN_INTERIOR_BIAS", 0.18)

# Rare-mutation evolution layer.
# These knobs govern occasional stronger mutation events for evolutionary
# variety. Increase them when diversity pressure matters more. Lower them when
# stability and inheritance fidelity matter more.
# TICK_WINDOW_ENABLE / TICK_WINDOW_TICKS:
#   gate special mutations by age or recentness window logic.
# PHYSICAL_ENABLE / *_STD_FRAC / *_CLIP_FRAC:
#   allow stronger drift in physical attributes, bounded by clip fraction.
# INHERITED_BRAIN_NOISE_ENABLE / *_STD:
#   allow occasional stronger brain-parameter perturbation on inheritance.
RESPAWN_RARE_MUTATION_TICK_WINDOW_ENABLE: bool = _env_bool("FWS_RESP_RARE_TICK_WINDOW_ENABLE", True)
RESPAWN_RARE_MUTATION_TICK_WINDOW_TICKS: int = _env_int("FWS_RESP_RARE_TICK_WINDOW", 4000)
RESPAWN_RARE_MUTATION_PHYSICAL_ENABLE: bool = _env_bool("FWS_RESP_RARE_PHYS_ENABLE", True)
RESPAWN_RARE_MUTATION_PHYSICAL_DRIFT_STD_FRAC: float = _env_float("FWS_RESP_RARE_PHYS_STD", 0.12)
RESPAWN_RARE_MUTATION_PHYSICAL_DRIFT_CLIP_FRAC: float = _env_float("FWS_RESP_RARE_PHYS_CLIP", 0.18)
RESPAWN_RARE_MUTATION_INHERITED_BRAIN_NOISE_ENABLE: bool = _env_bool("FWS_RESP_RARE_BRAIN_NOISE_ENABLE", True)
RESPAWN_RARE_MUTATION_INHERITED_BRAIN_NOISE_STD: float = _env_float("FWS_RESP_RARE_BRAIN_NOISE_STD", 0.10)

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
# Probability-like fraction in [0, 1].
# Lower values make selection more elitist.
# Higher values preserve more diversity in the candidate pool.
RESPAWN_PARENT_SELECTION_TOPK_FRAC: float = _env_float("FWS_RESP_PARENT_TOPK_FRAC", 0.12)

# Exponent controlling how strongly weights favor higher scores within the pool.
# 0 means near-flat weighting across the pool.
# Larger values make high-scoring parents dominate more strongly.
RESPAWN_PARENT_SELECTION_SCORE_POWER: float = _env_float("FWS_RESP_PARENT_SCORE_POWER", 2.4)

# Doctrine of Birth / The Closed Cradle
# Ongoing births can be forced to remain bloodline-bound:
# - if True, a live same-team parent must exist for every birth
# - if False, legacy parentless fresh births remain possible
RESPAWN_REQUIRE_PARENT_FOR_BIRTH: bool = _env_bool("FWS_RESP_REQUIRE_PARENT_FOR_BIRTH", True)

# Active doctrine for scoring alive same-team parents when
# RESPAWN_PARENT_SELECTION_MODE="topk_weighted".
# Doctrine meanings in current respawn code:
#   "overall"         : blended overall score
#   "killer"          : favor parents with kills
#   "cp"              : favor control-point contribution
#   "health"          : favor currently healthy parents
#   "kill_health"     : blend kills and health
#   "health_cp"       : blend health and CP performance
#   "kill_cp"         : blend kills and CP performance
#   "trinity"         : blend kill + cp + health
#   "highest_spike"   : favor whichever single channel spikes highest
#   "personal_points" : favor a per-agent personal score channel
#   "random_per_birth": choose one doctrine from RESPAWN_BIRTH_RANDOM_DOCTRINE_POOL
RESPAWN_BIRTH_DOCTRINE_MODE: str = _env_str("FWS_RESP_BIRTH_DOCTRINE_MODE", "kill_cp").strip().lower()

# Pool sampled by random_per_birth.
# Comma-separated doctrine names. Do not include "random_per_birth" itself in
# the pool unless you want validation to reject recursive randomization.
RESPAWN_BIRTH_RANDOM_DOCTRINE_POOL = tuple(
    s.strip().lower()
    for s in _env_str("FWS_RESP_BIRTH_RANDOM_DOCTRINE_POOL", "overall").split(",")
    if s.strip()
)

# Explicit top-k parent count. Use 0 to retain legacy fraction-based sizing.
# If this is > 0, it overrides the fraction-based top-k size calculation.
RESPAWN_BIRTH_TOPK_SIZE: int = _env_int("FWS_RESP_BIRTH_TOPK_SIZE", 16)

# Behavior when a doctrine yields no positive candidate score.
# "uniform_candidates":
#   still allow birth, but pick uniformly from candidates.
# "abort_birth":
#   skip the birth when the doctrine cannot distinguish candidates positively.
RESPAWN_BIRTH_ZERO_SCORE_FALLBACK: str = _env_str(
    "FWS_RESP_BIRTH_ZERO_SCORE_FALLBACK",
    "uniform_candidates",
).strip().lower()

# Relative blend weights for multi-axis doctrines.
# These matter only for doctrines that combine multiple score channels.
# Setting one weight higher tells the scoring function to care more about that
# axis relative to the others.
RESPAWN_BIRTH_BLEND_WEIGHT_KILL: float = _env_float("FWS_RESP_BIRTH_BLEND_WEIGHT_KILL", 1.0)
RESPAWN_BIRTH_BLEND_WEIGHT_CP: float = _env_float("FWS_RESP_BIRTH_BLEND_WEIGHT_CP", 1.6)
RESPAWN_BIRTH_BLEND_WEIGHT_HEALTH: float = _env_float("FWS_RESP_BIRTH_BLEND_WEIGHT_HEALTH", 0.0)
RESPAWN_BIRTH_BLEND_WEIGHT_PERSONAL: float = _env_float("FWS_RESP_BIRTH_BLEND_WEIGHT_PERSONAL", 0.0)

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
# Larger radius loosens colony-style locality.
# Smaller radius keeps descendants clustered more tightly around parents.
RESPAWN_SPAWN_NEAR_PARENT_RADIUS: int = _env_int(
    "FWS_RESP_SPAWN_NEAR_PARENT_RADIUS",
    1,
)

# RESPAWN_CHILD_UNIT_MODE
# -----------------------
# Valid values:
#   "inherit_parent_on_clone" : clone births inherit parent unit; fresh births use spawn ratio
#   "inherit_parent"          : synonym for clone inheritance
#   "clone_inherits_unit"     : synonym for clone inheritance
#   "random"                  : every birth uses the global spawn ratio
# This is a true runtime knob consumed by engine.respawn and summarized by main.py.
RESPAWN_CHILD_UNIT_MODE: str = _env_str(
    "FWS_RESPAWN_CHILD_UNIT_MODE",
    "inherit_parent_on_clone",
).strip().lower()


# =============================================================================
# REWARD SHAPING
# =============================================================================
# These knobs influence team-level and PPO-agent-level rewards.
# Raise with care: reward scale strongly affects training dynamics.
# =============================================================================
# A practical beginner rule:
# reward magnitudes are relative signals, not isolated numbers.
# If one coefficient is 10x bigger than the others, training will usually
# prioritize that behavior whether or not you intended it.

# Team-level combat rewards/penalties.
# TEAM_KILL_REWARD:
#   team reward when enemies die.
# TEAM_DMG_DEALT_REWARD:
#   reward for raw damage contribution.
# TEAM_DEATH_PENALTY:
#   negative signal when your team dies.
# TEAM_DMG_TAKEN_PENALTY:
#   negative signal for taking damage.
TEAM_KILL_REWARD: float = _env_float("FWS_REW_KILL", 0.0)
TEAM_DMG_DEALT_REWARD: float = _env_float("FWS_REW_DMG_DEALT", 0.0)
TEAM_DEATH_PENALTY: float = _env_float("FWS_REW_DEATH", 0.0)
TEAM_DMG_TAKEN_PENALTY: float = _env_float("FWS_REW_DMG_TAKEN", 0.0)

# PPO dense HP reward.
# Dense means it can contribute frequently, not only on sparse events like kills.
PPO_REWARD_HP_TICK: float = _env_float("FWS_PPO_REW_HP_TICK", 0.0035)

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
PPO_HP_REWARD_THRESHOLD: float = _env_float("FWS_PPO_HP_REWARD_THRESHOLD", 0.70)

# Individual PPO reward terms.
# PPO_REWARD_DMG_DEALT_INDIVIDUAL:
#   per-agent positive reward for dealing damage.
# PPO_PENALTY_DMG_TAKEN_INDIVIDUAL:
#   per-agent penalty magnitude for taking damage.
# PPO_REWARD_KILL_INDIVIDUAL:
#   sparse individual reward for kills.
# PPO_REWARD_DEATH:
#   death-related PPO term; current runtime applies it as a same-team aggregate.
# PPO_REWARD_CONTESTED_CP:
#   reward for control-point contest/occupation dynamics.
# PPO_REWARD_HEALING_RECOVERY:
#   reward for recovering health through healing.
PPO_REWARD_DMG_DEALT_INDIVIDUAL: float = _env_float("FWS_PPO_REW_DMG_DEALT_AGENT", 0.20)
# Runtime subtracts this coefficient, so positive values create a true penalty.
PPO_PENALTY_DMG_TAKEN_INDIVIDUAL: float = _env_float("FWS_PPO_PEN_DMG_TAKEN_AGENT", 0.30)
PPO_REWARD_KILL_INDIVIDUAL: float = _env_float("FWS_PPO_REW_KILL_AGENT", 2.0)
# Despite the name, the current PPO path applies this as a same-team death aggregate.
PPO_REWARD_DEATH: float = _env_float("FWS_PPO_REW_DEATH", 0.0)
PPO_REWARD_CONTESTED_CP: float = _env_float("FWS_PPO_REW_CONTEST", 0.85)
PPO_REWARD_HEALING_RECOVERY: float = _env_float("FWS_PPO_REW_HEALING_RECOVERY", 2.5)


# =============================================================================
# REINFORCEMENT LEARNING (PPO)
# =============================================================================
# These are the actual PPO training knobs.
# If you are new to the project, change only one or two of these at a time.
# The most immediately influential ones are:
#   PPO_WINDOW_TICKS, PPO_LR, PPO_CLIP, PPO_ENTROPY_COEF, PPO_EPOCHS,
#   PPO_MINIBATCHES, PPO_GAMMA, PPO_LAMBDA.

# Master PPO switch.
PPO_ENABLED: bool = _env_bool("FWS_PPO_ENABLED", True)

# Reset/log PPO state on startup/resume.
# Runtime honesty:
# this knob exists on the config surface, but its effect depends on whether the
# downstream PPO/runtime path explicitly consumes it in your branch.
PPO_RESET_LOG: bool = _env_bool("FWS_PPO_RESET_LOG", False)

# Rollout horizon in ticks.
# Larger windows improve long-horizon information but delay updates.
# Smaller windows update more frequently but with shorter trajectories.
PPO_WINDOW_TICKS: int = _env_int("FWS_PPO_TICKS", 192)

# Optimizer learning rate.
# Higher values learn faster but destabilize more easily.
# Lower values learn more cautiously but can stall.
PPO_LR: float = _env_float("FWS_PPO_LR", 3e-4)

# LR scheduler horizon and floor.
# T_MAX controls the cosine-scheduler horizon.
# ETA_MIN is the minimum learning rate floor.
PPO_LR_T_MAX: int = _env_int("FWS_PPO_T_MAX", 10_000_000)
PPO_LR_ETA_MIN: float = _env_float("FWS_PPO_ETA_MIN", 1e-6)

# PPO clipping and coefficients.
# PPO_CLIP:
#   policy-ratio clipping range; larger = less conservative update.
# PPO_ENTROPY_COEF:
#   exploration encouragement; higher = more randomness pressure.
# PPO_VALUE_COEF:
#   weight of value loss relative to policy loss.
# PPO_EPOCHS:
#   how many optimization passes per rollout.
# PPO_MINIBATCHES:
#   how many chunks each rollout is split into.
# PPO_MAX_GRAD_NORM:
#   gradient clipping safety cap.
# PPO_TARGET_KL:
#   optional early-stop / monitoring target for update size.
# PPO_GAMMA:
#   long-horizon discount factor.
# PPO_LAMBDA:
#   GAE bias/variance tradeoff.
# PPO_UPDATE_TICKS:
#   how often the runtime attempts PPO updates.
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
# Runtime honesty:
# the repository architecture is fundamentally per-agent-brain-oriented; this
# knob expresses that doctrine more than it toggles between two equally mature
# implementations.
PER_AGENT_BRAINS: bool = _env_bool("FWS_PER_AGENT_BRAINS", True)

# Global periodic mutation event across alive agents.
# These are long-horizon evolutionary-pressure knobs.
# Very small periods or large fractions make the population drift more
# aggressively.
MUTATION_PERIOD_TICKS: int = _env_int("FWS_MUTATE_EVERY", 1_000_000_000)
MUTATION_FRACTION_ALIVE: float = _env_float("FWS_MUTATE_FRAC", 0.02)


# =============================================================================
# BRAIN FAMILY AND TEAM-ASSIGNMENT POLICY
# =============================================================================
# This section answers two separate questions:
#   1) "What architecture family is a brain?"
#   2) "How are brain kinds assigned across teams and births?"
#
# The repository supports three named MLP brain variants and several routing
# modes for deciding which variant a given agent receives.

# Legacy names are accepted only as migration aliases and normalized immediately.
_CANONICAL_BRAIN_KINDS = (
    "throne_of_ashen_dreams",
    "veil_of_the_hollow_crown",
    "black_grail_of_nightfire",
)
_TEAM_BRAIN_ASSIGNMENT_MODE_ALLOWED = frozenset({"exclusive", "split", "team", "mix", "hybrid", "both"})
_TEAM_BRAIN_MIX_STRATEGY_ALLOWED = frozenset({"alternate", "roundrobin", "rr", "random", "prob", "probabilistic"})
_RESPAWN_CHILD_UNIT_MODE_ALLOWED = frozenset({"inherit_parent_on_clone", "inherit_parent", "clone_inherits_unit", "random"})

BRAIN_MLP_KIND_ALIASES = {
    "whispering_abyss": "throne_of_ashen_dreams",
    "veil_of_echoes": "veil_of_the_hollow_crown",
    "cathedral_of_ash": "throne_of_ashen_dreams",
    "dreamer_in_black_fog": "black_grail_of_nightfire",
    "obsidian_pulse": "black_grail_of_nightfire",
}


def _normalize_brain_kind_name(value: str) -> str:
    norm = str(value).strip().lower()
    return str(BRAIN_MLP_KIND_ALIASES.get(norm, norm))


def _coerce_enum_choice(name: str, value: str, *, allowed: frozenset[str], default: str) -> str:
    norm = str(value).strip().lower()
    if norm in allowed:
        return norm
    _config_issue(f"{name} must be one of {sorted(allowed)} (got {value!r}); using {default!r}")
    return default


def _coerce_brain_kind(name: str, value: str, *, default: str) -> str:
    norm = _normalize_brain_kind_name(value)
    if norm in _CANONICAL_BRAIN_KINDS:
        return norm
    _config_issue(f"{name} must be one of {sorted(_CANONICAL_BRAIN_KINDS)} (got {value!r}); using {default!r}")
    return default


def _coerce_brain_kind_sequence(
    name: str,
    raw: str,
    *,
    default: tuple[str, ...],
) -> tuple[str, ...]:
    seq = []
    bad = []
    for item in str(raw).split(","):
        if not str(item).strip():
            continue
        norm = _normalize_brain_kind_name(str(item).strip().lower())
        if norm in _CANONICAL_BRAIN_KINDS:
            seq.append(norm)
        else:
            bad.append(str(item).strip())

    if bad:
        _config_issue(f"{name} contains unknown brain kinds {bad!r}; dropping them")

    if seq:
        return tuple(seq)

    _config_issue(f"{name} must contain at least one known brain kind; using {list(default)!r}")
    return tuple(default)


RESPAWN_CHILD_UNIT_MODE = _coerce_enum_choice(
    "RESPAWN_CHILD_UNIT_MODE",
    RESPAWN_CHILD_UNIT_MODE,
    allowed=_RESPAWN_CHILD_UNIT_MODE_ALLOWED,
    default="inherit_parent_on_clone",
)


# Default brain kind used when no team-specific override applies.
# This is the fallback architecture when team-specific logic is disabled or not
# applicable.
BRAIN_KIND: str = _coerce_brain_kind(
    "BRAIN_KIND",
    _env_str("FWS_BRAIN", "throne_of_ashen_dreams").strip().lower(),
    default="throne_of_ashen_dreams",
)

# TEAM_BRAIN_ASSIGNMENT
# ---------------------
# If True, team-aware brain assignment logic may override the single default
# brain kind. If False, the default BRAIN_KIND is used more uniformly.
# Turn this off when you want one architecture family everywhere.
# Leave it on when you want red/blue specialization or mixed-architecture
# populations.
TEAM_BRAIN_ASSIGNMENT: bool = _env_bool("FWS_TEAM_BRAIN_ASSIGNMENT", True)

# TEAM_BRAIN_ASSIGNMENT_MODE
# --------------------------
# Synonym groups recognized by runtime:
#   fixed-per-team:
#       "exclusive", "split", "team"
#   mixed-per-spawn:
#       "mix", "hybrid", "both"
# Fixed-per-team modes:
#   red and blue each keep their configured family identity.
# Mixed-per-spawn modes:
#   each spawn/respawn event can choose among multiple families.
TEAM_BRAIN_ASSIGNMENT_MODE: str = _coerce_enum_choice(
    "TEAM_BRAIN_ASSIGNMENT_MODE",
    _env_str("FWS_TEAM_BRAIN_MODE", "mix").strip().lower(),
    allowed=_TEAM_BRAIN_ASSIGNMENT_MODE_ALLOWED,
    default="mix",
)

# TEAM_BRAIN_MIX_STRATEGY
# -----------------------
# Synonym groups recognized by runtime:
#   deterministic cycling:
#       "alternate", "roundrobin", "rr"
#   weighted random draw:
#       "random", "prob", "probabilistic"
# Deterministic cycling is good for controlled comparisons and reproducibility.
# Weighted random is better when you want probabilistic population mixtures.
TEAM_BRAIN_MIX_STRATEGY: str = _coerce_enum_choice(
    "TEAM_BRAIN_MIX_STRATEGY",
    _env_str("FWS_TEAM_BRAIN_MIX_STRATEGY", "random").strip().lower(),
    allowed=_TEAM_BRAIN_MIX_STRATEGY_ALLOWED,
    default="random",
)

# Team-specific fixed architectures used in exclusive/split/team modes.
# In fixed-per-team mode, red gets TEAM_BRAIN_EXCLUSIVE_RED and blue gets
# TEAM_BRAIN_EXCLUSIVE_BLUE.
TEAM_BRAIN_EXCLUSIVE_RED: str = _coerce_brain_kind(
    "TEAM_BRAIN_EXCLUSIVE_RED",
    _env_str("FWS_TEAM_BRAIN_RED", "throne_of_ashen_dreams").strip().lower(),
    default="throne_of_ashen_dreams",
)
TEAM_BRAIN_EXCLUSIVE_BLUE: str = _coerce_brain_kind(
    "TEAM_BRAIN_EXCLUSIVE_BLUE",
    _env_str("FWS_TEAM_BRAIN_BLUE", "veil_of_the_hollow_crown").strip().lower(),
    default="veil_of_the_hollow_crown",
)

# Sequence used by alternate/round-robin mixing.
# This order matters only for deterministic cycling strategies.
TEAM_BRAIN_MIX_SEQUENCE = _coerce_brain_kind_sequence(
    "TEAM_BRAIN_MIX_SEQUENCE",
    _env_str(
        "FWS_TEAM_BRAIN_MIX_SEQUENCE",
        "throne_of_ashen_dreams,veil_of_the_hollow_crown,black_grail_of_nightfire",
    ),
    default=_CANONICAL_BRAIN_KINDS,
)

# Weighted probabilities used by random/probabilistic mixing.
# These are relative weights, not required-to-sum-to-1 probabilities.
# The runtime normalizes by total positive weight.
TEAM_BRAIN_MIX_P_THRONE_OF_ASHEN_DREAMS: float = _env_float(
    "FWS_TEAM_BRAIN_P_THRONE_OF_ASHEN_DREAMS",
    0.34,
)
TEAM_BRAIN_MIX_P_VEIL_OF_THE_HOLLOW_CROWN: float = _env_float(
    "FWS_TEAM_BRAIN_P_VEIL_OF_THE_HOLLOW_CROWN",
    0.33,
)
TEAM_BRAIN_MIX_P_BLACK_GRAIL_OF_NIGHTFIRE: float = _env_float(
    "FWS_TEAM_BRAIN_P_BLACK_GRAIL_OF_NIGHTFIRE",
    0.33,
)

# Separate seed for team-brain mixing logic.
# 0 means use non-deterministic SystemRandom behavior.
# Non-zero means deterministic random mixing, with different salts for red and
# blue so the teams do not mirror the same draw sequence.
TEAM_BRAIN_MIX_SEED: int = _env_int("FWS_TEAM_BRAIN_MIX_SEED", int(globals().get("RNG_SEED", 0)))


# =============================================================================
# SELF-CENTRIC MLP BRAIN SURFACE
# =============================================================================
# These knobs are architectural and checkpoint-sensitive.
# Width/depth controls are safe to tune.
# Observation-schema and action-width contracts are not.
# =============================================================================
# Read these as model-capacity and model-style knobs.
# Broad tradeoff:
#   wider/deeper = more expressive + slower/heavier
#   narrower/shallower = cheaper/faster + less capacity

# RAY_WIDTH:
#   hidden width used to encode per-ray features before pooling.
# SCALAR_WIDTH:
#   hidden width for the non-ray rich tail.
# FUSION_WIDTH:
#   width after ray/scalar streams are fused into the shared trunk.
BRAIN_MLP_RAY_WIDTH: int = _env_int("FWS_BRAIN_MLP_RAY_WIDTH", 16)
BRAIN_MLP_SCALAR_WIDTH: int = _env_int("FWS_BRAIN_MLP_SCALAR_WIDTH", 16)
BRAIN_MLP_FUSION_WIDTH: int = _env_int("FWS_BRAIN_MLP_FUSION_WIDTH", 20)

# Depth knobs count residual blocks or tower stages in their respective parts.
BRAIN_MLP_RAY_DEPTH: int = _env_int("FWS_BRAIN_MLP_RAY_DEPTH", 1)
BRAIN_MLP_SCALAR_DEPTH: int = _env_int("FWS_BRAIN_MLP_SCALAR_DEPTH", 2)
BRAIN_MLP_TRUNK_DEPTH: int = _env_int("FWS_BRAIN_MLP_TRUNK_DEPTH", 3)

# Contract-critical derived width used by the shared late-fusion interface.
# This is derived from ray/scalar widths and should stay in sync automatically.
BRAIN_MLP_FINAL_INPUT_WIDTH: int = BRAIN_MLP_RAY_WIDTH + BRAIN_MLP_SCALAR_WIDTH

# ACTIVATION:
#   "gelu", "relu", or "silu"
# NORM:
#   "layernorm" or "none"
# RAY_POOLING:
#   "mean"       = average across ray tokens
#   "mean_max"   = fuse mean and max summaries
#   "gated_mean" = learned soft-attention over ray tokens
BRAIN_MLP_ACTIVATION: str = _env_str("FWS_BRAIN_MLP_ACT", "gelu").strip().lower()
BRAIN_MLP_NORM: str = _env_str("FWS_BRAIN_MLP_NORM", "layernorm").strip().lower()
BRAIN_MLP_RAY_POOLING: str = _env_str("FWS_BRAIN_MLP_RAY_POOLING", "mean_max").strip().lower()

# BLOCK_EXPANSION:
#   hidden expansion inside residual MLP blocks.
# DROPOUT:
#   dropout probability in [0, 1].
# USE_RESIDUAL:
#   whether blocks add skip connections.
# LAYER_SCALE_INIT:
#   initial residual scaling factor; smaller values can make deep blocks gentler
#   at startup.
BRAIN_MLP_BLOCK_EXPANSION: float = _env_float("FWS_BRAIN_MLP_BLOCK_EXPANSION", 1.5)
BRAIN_MLP_DROPOUT: float = _env_float("FWS_BRAIN_MLP_DROPOUT", 0.0)
BRAIN_MLP_USE_RESIDUAL: bool = _env_bool("FWS_BRAIN_MLP_USE_RESIDUAL", True)
BRAIN_MLP_LAYER_SCALE_INIT: float = _env_float("FWS_BRAIN_MLP_LAYER_SCALE_INIT", 1.0)

# Reinjection knobs are most relevant for the scalar-reinjection variant.
# REINJECT_EVERY controls cadence of reinjection.
# REINJECT_SCALE controls how strongly reinjected scalar context is applied.
BRAIN_MLP_REINJECT_EVERY: int = _env_int("FWS_BRAIN_MLP_REINJECT_EVERY", 1)
BRAIN_MLP_REINJECT_SCALE: float = _env_float("FWS_BRAIN_MLP_REINJECT_SCALE", 1.0)

# Gate knobs are most relevant for the scalar-gated-ray variant.
# GATE_STYLE:
#   "sigmoid" gives bounded multiplicative scaling via a sigmoid transform.
#   "tanh" gives tanh-centered scaling around 1.0.
# GATE_STRENGTH controls how strongly the gate can amplify/suppress features.
BRAIN_MLP_GATE_HIDDEN_WIDTH: int = _env_int("FWS_BRAIN_MLP_GATE_HIDDEN_WIDTH", 16)
BRAIN_MLP_GATE_STYLE: str = _env_str("FWS_BRAIN_MLP_GATE_STYLE", "sigmoid").strip().lower()
BRAIN_MLP_GATE_STRENGTH: float = _env_float("FWS_BRAIN_MLP_GATE_STRENGTH", 1.0)

# These gains affect the final orthogonal initialization of actor and critic
# heads. Smaller actor gain usually means smaller initial logits; critic gain
# controls initial value-head scale.
BRAIN_MLP_ACTOR_INIT_GAIN: float = _env_float("FWS_BRAIN_MLP_ACTOR_GAIN", 0.01)
BRAIN_MLP_CRITIC_INIT_GAIN: float = _env_float("FWS_BRAIN_MLP_CRITIC_GAIN", 1.0)

BRAIN_MLP_KIND_ORDER = _CANONICAL_BRAIN_KINDS

BRAIN_KIND_DISPLAY_NAMES = {
    "throne_of_ashen_dreams": "Throne of Ashen Dreams",
    "veil_of_the_hollow_crown": "Veil of the Hollow Crown",
    "black_grail_of_nightfire": "Black Grail of Nightfire",
}

BRAIN_KIND_SHORT_LABELS = {
    "throne_of_ashen_dreams": "TAD",
    "veil_of_the_hollow_crown": "VHC",
    "black_grail_of_nightfire": "BGN",
}

# Human-readable architectural intent:
#   Throne of Ashen Dreams:
#       simplest dual-tower late-fusion baseline
#   Veil of the Hollow Crown:
#       scalar context gets reinjected repeatedly through the trunk
#   Black Grail of Nightfire:
#       scalar context gates/modulates ray interpretation
# These names are flavorful, but they map to concrete architectural differences.
BRAIN_MLP_VARIANTS = {
    "throne_of_ashen_dreams": {
        "family": "dual_tower_late_fusion",
        "role": "clean baseline",
    },
    "veil_of_the_hollow_crown": {
        "family": "dual_tower_scalar_reinjection",
        "role": "scalar persistence",
    },
    "black_grail_of_nightfire": {
        "family": "scalar_gated_ray_interpretation",
        "role": "context-conditioned perception",
    },
}


# =============================================================================
# UI, VIEWER, INSPECTOR MODE, AND SCREEN RECORDING
# =============================================================================
# These knobs choose between three operator experiences:
#   - headless runtime
#   - normal interactive viewer runtime
#   - inspector/no-output viewer runtime
#
# If throughput matters most, headless is the cheapest mode.
# If visual inspection matters most, UI mode is the right tool.
# If you want to inspect without contaminating experiment lineage, use
# inspector-no-output mode.

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
# If either this flag or an inspector alias is active, main.py suppresses normal
# output-system creation.
INSPECTOR_UI_NO_OUTPUT: bool = _env_bool("FWS_INSPECTOR_UI_NO_OUTPUT", False)

# Viewer refresh cadences.
# Lower values refresh more often but cost more UI work.
# STATE_REFRESH_EVERY governs how often broader cached state is refreshed.
# PICK_REFRESH_EVERY governs how often hovered/selected pick data is refreshed.
VIEWER_STATE_REFRESH_EVERY: int = _env_int("FWS_VIEWER_STATE_REFRESH_EVERY", 4)
VIEWER_PICK_REFRESH_EVERY: int = _env_int("FWS_VIEWER_PICK_REFRESH_EVERY", 4)

# Font used by the UI.
UI_FONT_NAME: str = _env_str("FWS_UI_FONT", "consolas")

# Center the window on startup.
VIEWER_CENTER_WINDOW: bool = _env_bool("FWS_VIEWER_CENTER_WINDOW", True)

# Require the pygame-ce distribution when the UI is used.
PYGAME_CE_STRICT_RUNTIME: bool = _env_bool("FWS_PYGAME_CE_STRICT_RUNTIME", True)

# Bound the number of rendered text surfaces cached by the viewer.
# Higher values use more memory but reduce repeated text rasterization.
VIEWER_TEXT_CACHE_MAX_SURFACES: int = _env_int("FWS_VIEWER_TEXT_CACHE_MAX_SURFACES", 2048)

# Relative export directory (inside run_dir) for saved agent brains.
VIEWER_BRAIN_EXPORT_DIRNAME: str = _env_str("FWS_VIEWER_BRAIN_EXPORT_DIRNAME", "exports/brains").strip()

# Renderer dimensions.
# Increase CELL_SIZE for a larger visual map.
# Increase HUD_WIDTH for more side-panel room.
# CELL_SIZE changes how large each grid cell appears on screen.
# HUD_WIDTH changes side-panel width, not simulation behavior.
CELL_SIZE: int = _env_int("FWS_CELL_SIZE", 4)
HUD_WIDTH: int = _env_int("FWS_HUD_W", 340)

# Target render FPS in UI mode.
# This affects viewer smoothness and UI pacing, not headless tick pacing.
TARGET_FPS: int = _env_int("FWS_TARGET_FPS", 45)

# Video recording settings.
# Enable only when you explicitly want captured output; recording costs storage
# and some runtime overhead.
# VIDEO_FPS:
#   frame rate of the saved video.
# VIDEO_SCALE:
#   output upscaling factor.
# VIDEO_EVERY_TICKS:
#   capture cadence; 1 means every tick, larger values skip ticks between
#   recorded frames.
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
# Actual preset intent:
#   debug:
#       smaller/cheaper interactive local run with UI on
#   train_tiny:
#       headless low-memory run that keeps every built-in MLP brain variant under
#       roughly 50k parameters per agent
#   train_fast:
#       headless throughput-oriented run with narrower model widths
#   train_quality:
#       headless run with higher model widths for more capacity
#
# Important subtlety:
# profiles only touch a subset of knobs. They are not full experiment bundles.
# You are still expected to set output, checkpoint, reward, PPO, and other
# policy knobs explicitly when needed.


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
            ("FWS_USE_VMAP", "USE_VMAP", True),
        ],
        "train_tiny": [
            ("FWS_UI", "ENABLE_UI", False),
            ("FWS_USE_VMAP", "USE_VMAP", True),
            ("FWS_BRAIN_MLP_RAY_WIDTH", "BRAIN_MLP_RAY_WIDTH", 48),
            ("FWS_BRAIN_MLP_SCALAR_WIDTH", "BRAIN_MLP_SCALAR_WIDTH", 32),
            ("FWS_BRAIN_MLP_FUSION_WIDTH", "BRAIN_MLP_FUSION_WIDTH", 64),
            ("FWS_BRAIN_MLP_RAY_DEPTH", "BRAIN_MLP_RAY_DEPTH", 1),
            ("FWS_BRAIN_MLP_SCALAR_DEPTH", "BRAIN_MLP_SCALAR_DEPTH", 1),
            ("FWS_BRAIN_MLP_TRUNK_DEPTH", "BRAIN_MLP_TRUNK_DEPTH", 1),
            ("FWS_BRAIN_MLP_GATE_HIDDEN_WIDTH", "BRAIN_MLP_GATE_HIDDEN_WIDTH", 16),
        ],
        "train_fast": [
            ("FWS_UI", "ENABLE_UI", False),
            ("FWS_USE_VMAP", "USE_VMAP", True),
            ("FWS_BRAIN_MLP_RAY_WIDTH", "BRAIN_MLP_RAY_WIDTH", 72),
            ("FWS_BRAIN_MLP_SCALAR_WIDTH", "BRAIN_MLP_SCALAR_WIDTH", 72),
            ("FWS_BRAIN_MLP_FUSION_WIDTH", "BRAIN_MLP_FUSION_WIDTH", 96),
        ],
        "train_quality": [
            ("FWS_UI", "ENABLE_UI", False),
            ("FWS_USE_VMAP", "USE_VMAP", True),
            ("FWS_BRAIN_MLP_RAY_WIDTH", "BRAIN_MLP_RAY_WIDTH", 128),
            ("FWS_BRAIN_MLP_SCALAR_WIDTH", "BRAIN_MLP_SCALAR_WIDTH", 128),
            ("FWS_BRAIN_MLP_FUSION_WIDTH", "BRAIN_MLP_FUSION_WIDTH", 160),
        ],
    }

    rows = presets.get(PROFILE)
    if not rows:
        return

    g = globals()
    for env_key, var_name, value in rows:
        if not _env_is_set(env_key):
            g[var_name] = value

    g["BRAIN_MLP_FINAL_INPUT_WIDTH"] = int(g["BRAIN_MLP_RAY_WIDTH"]) + int(g["BRAIN_MLP_SCALAR_WIDTH"])


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
    _prob("TEAM_BRAIN_MIX_P_THRONE_OF_ASHEN_DREAMS", TEAM_BRAIN_MIX_P_THRONE_OF_ASHEN_DREAMS)
    _prob("TEAM_BRAIN_MIX_P_VEIL_OF_THE_HOLLOW_CROWN", TEAM_BRAIN_MIX_P_VEIL_OF_THE_HOLLOW_CROWN)
    _prob("TEAM_BRAIN_MIX_P_BLACK_GRAIL_OF_NIGHTFIRE", TEAM_BRAIN_MIX_P_BLACK_GRAIL_OF_NIGHTFIRE)
    _prob("TELEMETRY_MOVE_EVENTS_SAMPLE_RATE", TELEMETRY_MOVE_EVENTS_SAMPLE_RATE)
    _prob("CATASTROPHE_DYNAMIC_HEAL_OCCUPANCY_THRESHOLD", CATASTROPHE_DYNAMIC_HEAL_OCCUPANCY_THRESHOLD)
    _prob("CATASTROPHE_SMALL_SUPPRESS_FRACTION", CATASTROPHE_SMALL_SUPPRESS_FRACTION)
    _prob("CATASTROPHE_MEDIUM_SUPPRESS_FRACTION", CATASTROPHE_MEDIUM_SUPPRESS_FRACTION)
    _prob("CATASTROPHE_CLUSTER_SURVIVOR_FRACTION", CATASTROPHE_CLUSTER_SURVIVOR_FRACTION)

    # Positive integer style checks.
    _positive_int("CHECKPOINT_KEEP_LAST_N", max(1, CHECKPOINT_KEEP_LAST_N))
    _positive_int("BRAIN_MLP_RAY_WIDTH", BRAIN_MLP_RAY_WIDTH)
    _positive_int("BRAIN_MLP_SCALAR_WIDTH", BRAIN_MLP_SCALAR_WIDTH)
    _positive_int("BRAIN_MLP_FUSION_WIDTH", BRAIN_MLP_FUSION_WIDTH)
    _positive_int("BRAIN_MLP_RAY_DEPTH", BRAIN_MLP_RAY_DEPTH)
    _positive_int("BRAIN_MLP_SCALAR_DEPTH", BRAIN_MLP_SCALAR_DEPTH)
    _positive_int("BRAIN_MLP_TRUNK_DEPTH", BRAIN_MLP_TRUNK_DEPTH)
    _positive_int("BRAIN_MLP_REINJECT_EVERY", BRAIN_MLP_REINJECT_EVERY)
    _positive_int("BRAIN_MLP_GATE_HIDDEN_WIDTH", BRAIN_MLP_GATE_HIDDEN_WIDTH)
    _positive_int("BRAIN_MLP_FINAL_INPUT_WIDTH", BRAIN_MLP_FINAL_INPUT_WIDTH)
    _positive_int("RESPAWN_RARE_MUTATION_TICK_WINDOW_TICKS", RESPAWN_RARE_MUTATION_TICK_WINDOW_TICKS)
    _positive_int("RESPAWN_SPAWN_NEAR_PARENT_RADIUS", RESPAWN_SPAWN_NEAR_PARENT_RADIUS)

    _non_negative_float("BRAIN_MLP_BLOCK_EXPANSION", BRAIN_MLP_BLOCK_EXPANSION)
    _non_negative_float("BRAIN_MLP_DROPOUT", BRAIN_MLP_DROPOUT)
    _non_negative_float("BRAIN_MLP_LAYER_SCALE_INIT", BRAIN_MLP_LAYER_SCALE_INIT)
    _non_negative_float("BRAIN_MLP_REINJECT_SCALE", BRAIN_MLP_REINJECT_SCALE)
    _non_negative_float("BRAIN_MLP_GATE_STRENGTH", BRAIN_MLP_GATE_STRENGTH)
    if float(BRAIN_MLP_BLOCK_EXPANSION) <= 0.0:
        _config_issue("BRAIN_MLP_BLOCK_EXPANSION must be > 0")
    if float(BRAIN_MLP_DROPOUT) > 1.0:
        _config_issue("BRAIN_MLP_DROPOUT must be <= 1")

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

    expected_fusion_input = int(BRAIN_MLP_RAY_WIDTH) + int(BRAIN_MLP_SCALAR_WIDTH)
    if int(BRAIN_MLP_FINAL_INPUT_WIDTH) != expected_fusion_input:
        _config_issue(
            "BRAIN_MLP_FINAL_INPUT_WIDTH must equal BRAIN_MLP_RAY_WIDTH + BRAIN_MLP_SCALAR_WIDTH "
            f"(got {BRAIN_MLP_FINAL_INPUT_WIDTH} vs {expected_fusion_input})"
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
    _one_of("PROFILE", PROFILE, {"default", "debug", "train_tiny", "train_fast", "train_quality"})
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
    _one_of("BRAIN_MLP_GATE_STYLE", BRAIN_MLP_GATE_STYLE, {"sigmoid", "tanh"})
    _one_of("BRAIN_MLP_RAY_POOLING", BRAIN_MLP_RAY_POOLING, {"mean", "mean_max", "gated_mean"})
    _one_of("TELEMETRY_PPO_RICH_LEVEL", TELEMETRY_PPO_RICH_LEVEL, {"update", "epoch", "minibatch"})
    _one_of("TELEMETRY_DAMAGE_MODE", TELEMETRY_DAMAGE_MODE, {"victim_sum", "per_hit"})
    _one_of("TELEMETRY_EVENTS_FORMAT", TELEMETRY_EVENTS_FORMAT, {"jsonl", "csv"})
    _one_of("TELEMETRY_TICKS_FORMAT", TELEMETRY_TICKS_FORMAT, {"csv", "jsonl", "parquet", "npz"})
    _one_of("TELEMETRY_SNAPSHOT_FORMAT", TELEMETRY_SNAPSHOT_FORMAT, {"npz", "pt", "pickle"})
    _one_of("PPO_HP_REWARD_MODE", PPO_HP_REWARD_MODE, {"raw", "threshold_ramp"})
    _one_of("TEAM_BRAIN_ASSIGNMENT_MODE", TEAM_BRAIN_ASSIGNMENT_MODE, set(_TEAM_BRAIN_ASSIGNMENT_MODE_ALLOWED))
    _one_of("TEAM_BRAIN_MIX_STRATEGY", TEAM_BRAIN_MIX_STRATEGY, set(_TEAM_BRAIN_MIX_STRATEGY_ALLOWED))
    _one_of("RESPAWN_CHILD_UNIT_MODE", RESPAWN_CHILD_UNIT_MODE, set(_RESPAWN_CHILD_UNIT_MODE_ALLOWED))

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
        float(TEAM_BRAIN_MIX_P_THRONE_OF_ASHEN_DREAMS)
        + float(TEAM_BRAIN_MIX_P_VEIL_OF_THE_HOLLOW_CROWN)
        + float(TEAM_BRAIN_MIX_P_BLACK_GRAIL_OF_NIGHTFIRE)
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
    if str(OBS_SCHEMA) not in OBS_SCHEMA_ALLOWED:
        _config_issue(f"OBS_SCHEMA must be one of {OBS_SCHEMA_ALLOWED!r} (got {OBS_SCHEMA!r})")

    expected_rays_flat = int(RAY_TOKEN_COUNT) * int(RAY_FEAT_DIM)
    if int(RAYS_FLAT_DIM) != expected_rays_flat:
        _config_issue(
            f"RAYS_FLAT_DIM must equal RAY_TOKEN_COUNT * RAY_FEAT_DIM "
            f"(got {RAYS_FLAT_DIM} vs {expected_rays_flat})"
        )
    if int(RAY_FEAT_DIM) != 8:
        _config_issue(f"RAY_FEAT_DIM must remain the 8-channel first-hit contract (got {RAY_FEAT_DIM})")

    expected_rich_total = int(RICH_BASE_DIM) + int(INSTINCT_DIM)
    if int(RICH_TOTAL_DIM) != expected_rich_total:
        _config_issue(
            f"RICH_TOTAL_DIM must equal RICH_BASE_DIM + INSTINCT_DIM "
            f"(got {RICH_TOTAL_DIM} vs {expected_rich_total})"
        )

    if str(OBS_SCHEMA) == OBS_SCHEMA_SELF_CENTRIC_V1:
        if int(RICH_BASE_DIM) != int(SELF_CENTRIC_RICH_BASE_DIM) or int(INSTINCT_DIM) != int(SELF_CENTRIC_INSTINCT_DIM):
            _config_issue(
                "self-centric schema dims must match SELF_CENTRIC_RICH_BASE_DIM/SELF_CENTRIC_INSTINCT_DIM"
            )
    else:
        if int(RICH_BASE_DIM) != int(LEGACY_FULL_RICH_BASE_DIM) or int(INSTINCT_DIM) != int(LEGACY_FULL_INSTINCT_DIM):
            _config_issue(
                "legacy schema dims must match LEGACY_FULL_RICH_BASE_DIM/LEGACY_FULL_INSTINCT_DIM"
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
