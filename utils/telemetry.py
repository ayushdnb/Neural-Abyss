"""Telemetry sinks for lineage, events, and periodic summaries."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import config


def _to_int(x: Any) -> int:
    """Convert a numeric-like value to ``int``."""
    try:
        return int(x)
    except Exception:
        return int(float(x))


def _atomic_write_text(path: Path, text: str) -> None:
    """Write text atomically with a small Windows retry budget."""
    import time

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")

    max_tries = 20 if os.name == "nt" else 3
    last_exc: Optional[BaseException] = None

    for attempt in range(max_tries):
        try:
            os.replace(tmp, path)
            return
        except OSError as e:
            last_exc = e
            winerr = getattr(e, "winerror", None)

            if os.name == "nt" and winerr in (5, 32):
                time.sleep(0.01 * (attempt + 1))
                continue
            raise

    try:
        path.write_text(text, encoding="utf-8")
        try:
            tmp.unlink()
        except Exception:
            pass
    except Exception:
        if last_exc is not None:
            raise last_exc
        raise


def _parse_validate_level(v: Any, default: int = 2) -> int:
    """Parse ``TELEMETRY_VALIDATE_LEVEL`` into ``0``, ``1``, or ``2``."""
    if isinstance(v, (int, float)):
        try:
            return int(v)
        except Exception:
            return int(default)

    if isinstance(v, str):
        s = v.strip().lower()
        m = {"off": 0, "0": 0, "basic": 1, "1": 1, "strict": 2, "2": 2}
        if s in m:
            return int(m[s])
        try:
            return int(s)
        except Exception:
            return int(default)

    return int(default)


def _parse_schema_version_int(x: Any, default: int = 2) -> int:
    """
    Parse telemetry schema version into an integer for event payloads.

    Accepts numeric values and strings like "2" or "v2". Falls back to `default`.
    """
    try:
        return int(x)
    except Exception:
        pass

    try:
        s = str(x).strip()
    except Exception:
        return int(default)

    if s[:1] in ("v", "V"):
        s = s[1:]

    try:
        return int(float(s))
    except Exception:
        return int(default)

class TelemetrySession:
    """Stateful telemetry recorder for one simulation run."""

    def __init__(self, run_dir: Path) -> None:
        """Initialize telemetry under ``run_dir/telemetry``."""
        self.enabled: bool = bool(getattr(config, "TELEMETRY_ENABLED", False))
        self.run_dir = Path(run_dir)

        self._event_ctx_tick: Optional[int] = None
        self._event_ctx_phase: Optional[str] = None
        self._event_ctx_tick_seq: int = 0
        self._event_ctx_phase_seq: int = 0

        # Schema versioning (important for downstream analysis compatibility)
        # NOTE: The original code sets schema_version as int here.
        self.schema_version_int: int = _parse_schema_version_int(
            getattr(config, "TELEMETRY_SCHEMA_VERSION", 2),
            default=2,
        )

        # Reuse existing knobs (do not invent a second config system).
        self.tag: str = str(getattr(config, "TELEMETRY_TAG", ""))
        self.event_chunk_size: int = int(getattr(config, "TELEMETRY_EVENT_CHUNK_SIZE", 50_000))
        self.flush_every: int = int(getattr(config, "TELEMETRY_PERIODIC_FLUSH_EVERY", 250))

        # Events format: this implementation supports JSONL only.
        self.events_format: str = str(getattr(config, "TELEMETRY_EVENTS_FORMAT", "jsonl"))
        self.events_gzip: bool = bool(getattr(config, "TELEMETRY_EVENTS_GZIP", False))  # reserved for future; no deps

        # Which event types to emit
        self.log_births: bool = bool(getattr(config, "TELEMETRY_LOG_BIRTHS", True))
        self.log_deaths: bool = bool(getattr(config, "TELEMETRY_LOG_DEATHS", True))
        self.log_damage: bool = bool(getattr(config, "TELEMETRY_LOG_DAMAGE", True))
        self.log_kills: bool = bool(getattr(config, "TELEMETRY_LOG_KILLS", True))
        self.damage_mode: str = str(getattr(config, "TELEMETRY_DAMAGE_MODE", "victim_sum"))

        # Movement telemetry (aggregates are cheap; per-agent events are optional)
        # NOTE: config already defines TELEMETRY_LOG_MOVES (FWS_TELEM_MOVES).
        # We interpret it as enabling movement telemetry in general:
        #   - always write cheap aggregates (move_summary.csv)
        #   - optionally emit sampled per-agent 'move' events (events_*.jsonl)
        self.log_moves: bool = bool(getattr(config, "TELEMETRY_LOG_MOVES", False))

        # Per-agent move event sampling knobs:
        #   - move_events_every: 0 disables per-agent move events (aggregates still work)
        #   - move_events_max_per_tick: hard cap to prevent event explosions
        #   - move_events_sample_rate: deterministic sampler (0..1)
        self.move_events_every: int = int(getattr(config, "TELEMETRY_MOVE_EVENTS_EVERY", 0))
        self.move_events_max_per_tick: int = int(getattr(config, "TELEMETRY_MOVE_EVENTS_MAX_PER_TICK", 256))
        self.move_events_sample_rate: float = float(getattr(config, "TELEMETRY_MOVE_EVENTS_SAMPLE_RATE", 1.0))

        # Per-slot movement totals (used to populate AgentLife movement columns)
        # Why per-slot?
        # - The engine is vectorized over registry slots.
        # - We can update per-slot counters with torch index_add_ (fast on GPU/CPU).
        # Correctness with slot reuse:
        # - When an agent dies, we "freeze" its final counters into the AgentLife record
        #   in record_deaths().
        # - When a new agent is born into a slot, we reset slot counters in record_birth().
        # NOTE: These are lazily allocated tensors (created only if enabled+log_moves).
        #       They are placed on the engine device (CPU/CUDA) so updates are cheap.
        self._move_slot_attempted = None
        self._move_slot_success = None
        self._move_slot_blocked_wall = None
        self._move_slot_blocked_occupied = None
        self._move_slot_conflict_lost = None
        self._move_slot_conflict_tie = None
        self._move_slot_cells_l1 = None
        self._move_slot_last_seen = None

        # Generic "counters" stream for lightweight extensions
        # Emits rows to counters.csv in long format: (tick, key, value).
        # This is intentionally schema-stable: adding new keys does not change the file schema.
        self.counters_every: int = int(getattr(config, "TELEMETRY_COUNTERS_EVERY", 0))

        # Validation knobs (detect internal inconsistencies)
        self.validate_level: int = _parse_validate_level(
            getattr(config, "TELEMETRY_VALIDATE_LEVEL", 2),
            default=2,
        )
        self.validate_every: int = int(getattr(config, "TELEMETRY_VALIDATE_EVERY", 1000))
        self.abort_on_anomaly: bool = bool(getattr(config, "TELEMETRY_ABORT_ON_ANOMALY", False))

        # Output layout
        self.telemetry_dir = self.run_dir / "telemetry"
        self.events_dir = self.telemetry_dir / "events"
        self.agent_life_path = self.telemetry_dir / "agent_life.csv"
        self.lineage_edges_path = self.telemetry_dir / "lineage_edges.csv"
        self.schema_manifest_path = self.telemetry_dir / "schema_manifest.json"

        # Optional sidecar outputs.
        self.run_meta_path = self.telemetry_dir / "run_meta.json"
        self.agent_static_path = self.telemetry_dir / "agent_static.csv"
        self.tick_summary_path = self.telemetry_dir / "tick_summary.csv"

        # Movement aggregates (cheap) + generic extension counters (long format)
        self.move_summary_path = self.telemetry_dir / "move_summary.csv"
        self.counters_path = self.telemetry_dir / "counters.csv"
        self.headless_summary_path = self.telemetry_dir / "telemetry_summary.csv"
        self.ppo_training_telemetry_path = self.telemetry_dir / "ppo_training_telemetry.csv"
        self.mutation_events_path = self.telemetry_dir / "mutation_events.csv"
        self.dead_agents_detailed_path = self.telemetry_dir / "dead_agents_log_detailed.csv"

        # Per-slot PPO reward accumulators
        # Why per-slot?
        # - The engine computes rewards on live registry slots.
        # - We want zero per-tick CPU sync overhead.
        # - On death, we freeze slot totals into the persistent AgentLife record.
        # - On birth, the slot is reset/seeded.
        self._reward_slot_total = None
        self._reward_slot_individual_total = None
        self._reward_slot_team_total = None
        self._reward_slot_hp = None
        self._reward_slot_kill_individual = None
        self._reward_slot_damage_dealt_individual = None
        self._reward_slot_damage_taken_penalty = None
        self._reward_slot_contested_cp_individual = None
        self._reward_slot_team_kill = None
        self._reward_slot_team_death = None
        self._reward_slot_team_cp = None
        self._reward_slot_healing_recovered = None

        # Create directories only when telemetry is enabled.
        if self.enabled:
            self.telemetry_dir.mkdir(parents=True, exist_ok=True)
            self.events_dir.mkdir(parents=True, exist_ok=True)

        # Preserve the string schema version for compatibility with existing metadata.
        self.schema_version_str: str = str(getattr(config, "TELEMETRY_SCHEMA_VERSION", "v2"))
        # Backward-compat alias used by existing metadata writers.
        self.schema_version: str = self.schema_version_str

        # - The method is write_run_meta(...)
        # - The flag is do_write_run_meta (boolean).
        self.do_write_run_meta: bool = bool(getattr(config, "TELEMETRY_WRITE_RUN_META", True))

        self.write_agent_static: bool = bool(getattr(config, "TELEMETRY_WRITE_AGENT_STATIC", False))
        self.tick_summary_every: int = int(getattr(config, "TELEMETRY_TICK_SUMMARY_EVERY", 0))

        # Additive headless live summary sidecar (CSV append, independent of prints)
        self.headless_live_summary_enabled: bool = bool(getattr(config, "TELEMETRY_HEADLESS_LIVE_SUMMARY", True))
        self.headless_summary_include_tps: bool = bool(getattr(config, "TELEMETRY_HEADLESS_SUMMARY_INCLUDE_TPS", True))
        self.headless_summary_include_gpu: bool = bool(getattr(config, "TELEMETRY_HEADLESS_SUMMARY_INCLUDE_GPU", False))
        self.headless_summary_include_tick_metrics: bool = bool(
            getattr(config, "TELEMETRY_HEADLESS_SUMMARY_INCLUDE_TICK_METRICS", True)
        )
        self.headless_summary_include_ppo: bool = bool(getattr(config, "TELEMETRY_HEADLESS_SUMMARY_INCLUDE_PPO", True))
        self.log_rare_mutations: bool = bool(getattr(config, "TELEMETRY_LOG_RARE_MUTATIONS", True))

        # Dedicated PPO training diagnostics CSV (append-safe, separate from headless summary sidecar)
        self.ppo_rich_csv_enabled: bool = bool(getattr(config, "TELEMETRY_PPO_RICH_CSV", True))
        self.ppo_rich_level: str = str(getattr(config, "TELEMETRY_PPO_RICH_LEVEL", "update")).strip().lower()
        self.ppo_rich_flush_every: int = max(1, int(getattr(config, "TELEMETRY_PPO_RICH_FLUSH_EVERY", 1)))
        self.telemetry_append_schema_strict: bool = bool(
            getattr(config, "TELEMETRY_APPEND_SCHEMA_STRICT", True)
        )

        # Context references (never mutates; just pointers for reading data)
        self._registry: Any = None
        self._stats: Any = None
        self._ppo_runtime: Any = None

        # Buffered side-streams (flushed on the same cadence as event chunks)
        self._move_agg_buf: List[Dict[str, Any]] = []
        self._counters_buf: List[Tuple[int, str, float]] = []
        self._ppo_rich_buf: List[Dict[str, Any]] = []
        # Telemetry-only fallback dedupe:
        # if the PPO runtime rich queue is empty but last_train_summary advances,
        # synthesize at most one row per update_seq.
        self._ppo_rich_last_fallback_update_seq: int = 0

        # Headless summary runtime state (additive)
        self._headless_wall_start: Optional[float] = None
        self._headless_wall_start_tick: Optional[int] = None
        self._headless_prev_summary_wall: Optional[float] = None
        self._headless_prev_summary_tick: Optional[int] = None
        self._headless_gpu_raw: Optional[str] = None
        self._headless_gpu_parsed: Dict[str, Any] = {}
        self._headless_prev_cp_red: Optional[float] = None
        self._headless_prev_cp_blue: Optional[float] = None
        self._headless_prev_dmg_dealt_red: Optional[float] = None
        self._headless_prev_dmg_dealt_blue: Optional[float] = None
        self._headless_prev_dmg_taken_red: Optional[float] = None
        self._headless_prev_dmg_taken_blue: Optional[float] = None
        self._headless_tick_metrics_window: Dict[str, float] = {}
        self._headless_tick_metrics_window_ticks: int = 0

        self._headless_tick_metric_keys = [
            "alive",
            "moved",
            "attacks",
            "deaths",
            "deaths_combat",
            "deaths_metabolism",
            "cp_red_tick",
            "cp_blue_tick",
            "move_attempted",
            "move_can_move",
            "move_blocked_wall",
            "move_blocked_occupied",
            "move_conflict_lost",
            "move_conflict_tie",
        ]

        # AgentStatic deduplication: track which agent_ids already written
        self._static_written: set[int] = set()
        if self.write_agent_static and self.agent_static_path.exists():
            # If agent_static.csv exists (e.g., resume), rehydrate `_static_written`
            # to avoid duplicating static rows.
            try:
                with self.agent_static_path.open("r", newline="", encoding="utf-8") as f:
                    r = csv.DictReader(f)
                    for row in r:
                        if "agent_id" in row and row["agent_id"] not in (None, ""):
                            # agent_id might be stored as "123" or "123.0".
                            self._static_written.add(int(float(row["agent_id"])))
            except Exception:
                # Fail-safe: if parsing fails, disable dedup state.
                # Worst-case: may re-emit some static rows; still usable for analysis.
                self._static_written = set()

        # Main AgentLife state
        # `_life` stores per-agent evolving stats and timestamps.
        # Key: agent_id (int)
        # Value: dict of fields (born_tick, death_tick, totals, etc.)
        self._life: Dict[int, Dict[str, Any]] = {}

        # Offspring count stored separately to keep `_life` simpler.
        self._offspring_count: Dict[int, int] = {}
        self._lineage_edge_count: Dict[int, int] = {}

        # Event buffering and chunking
        self._events_buf: List[Dict[str, Any]] = []
        self._chunk_idx: int = self._discover_next_chunk_idx()

        # Resume support: rehydrate agent life snapshot
        if self.enabled:
            self._rehydrate_agent_life()

        # Track last tick seen for graceful shutdown final summary.
        self._last_tick_seen: Optional[int] = None
        self._last_tick_summary_tick: Optional[int] = None

        # Ensure telemetry headers exist only for active telemetry sessions.
        if self.enabled and (not self.lineage_edges_path.exists()):
            self._append_csv_rows(
                self.lineage_edges_path,
                fieldnames=[
                    "tick",
                    "parent_id",
                    "child_id",
                    "parent_unit_type",
                    "child_unit_type",
                    "parent_team",
                    "spawn_reason",
                    "parent_generation",
                    "child_generation",
                    "mutation_flag",
                    "rare_mutation",
                    "mutation_delta_hp",
                    "mutation_delta_atk",
                    "mutation_delta_vis",
                ],
                rows=[],
                strict_schema=True,
            )

        # Create headers for agent_static and tick_summary if enabled and new.
        if self.enabled and self.write_agent_static and (not self.agent_static_path.exists()):
            self._append_csv_rows(
                self.agent_static_path,
                fieldnames=[
                    "agent_id", "team_id", "unit_type", "brain_type", "param_count",
                    "max_hp", "base_atk", "vision_range",
                    "spawn_tick", "parent_id", "spawn_reason",
                ],
                rows=[],
            )

        if self.enabled and self.tick_summary_every > 0 and (not self.tick_summary_path.exists()):
            self._append_csv_rows(
                self.tick_summary_path,
                fieldnames=[
                    "tick", "elapsed_s",
                    "red_alive", "blue_alive", "mean_hp_red", "mean_hp_blue",
                    "red_kills", "blue_kills", "red_deaths", "blue_deaths",
                    "red_dmg_dealt", "blue_dmg_dealt", "red_dmg_taken", "blue_dmg_taken",
                ],
                rows=[],
            )

        # Additive headless live summary CSV (superset sidecar; leaves tick_summary.csv unchanged)
        if (
            self.enabled
            and self.headless_live_summary_enabled
            and self.tick_summary_every > 0
            and (not self.headless_summary_path.exists())
        ):
            self._append_csv_rows(
                self.headless_summary_path,
                fieldnames=self._headless_summary_fieldnames(),
                rows=[],
            )

        # Dedicated rich PPO diagnostics CSV (append-safe; created only if enabled)
        if self.enabled and self.ppo_rich_csv_enabled and (not self.ppo_training_telemetry_path.exists()):
            self._append_csv_rows(
                self.ppo_training_telemetry_path,
                fieldnames=self._ppo_rich_fieldnames(),
                rows=[],
                strict_schema=self.telemetry_append_schema_strict,
            )

        # Rare mutation event CSV (append, low frequency)
        if self.enabled and self.log_rare_mutations and (not self.mutation_events_path.exists()):
            self._append_csv_rows(
                self.mutation_events_path,
                fieldnames=[
                    "tick",
                    "slot",
                    "agent_id",
                    "team_id",
                    "team_name",
                    "unit_id",
                    "parent_unit_id",
                    "child_generation",
                    "spawn_reason",
                    "spawn_hp",
                    "spawn_atk",
                    "spawn_vis",
                    "parent_agent_id",
                    "cloned",
                    "mutation_std",
                    "mutation_flag",
                    "rare_mutation",
                    "mutation_delta_hp",
                    "mutation_delta_atk",
                    "mutation_delta_vis",
                ],
                rows=[],
            )

    # Internal helpers

    def _discover_next_chunk_idx(self) -> int:
        """
        Determine the next events chunk index by scanning existing chunk files.

        Why this is needed
        ------------------
        If you resume a run and keep logging into the same run directory,
        you must not overwrite previous events_* files. Instead, you continue with
        the next index.

        Implementation
        --------------
        - Search for files matching: events_*.jsonl
        - Parse the numeric suffix and return max+1
        - If parsing fails for a file, ignore it.
        """
        max_idx = -1
        for p in self.events_dir.glob("events_*.jsonl"):
            try:
                stem = p.stem  # e.g., "events_000001"
                idx = int(stem.split("_", 1)[1])
                max_idx = max(max_idx, idx)
            except Exception:
                continue
        return max_idx + 1

    def _rehydrate_agent_life(self) -> None:
        """
        Load agent_life.csv into memory if it exists.

        Purpose
        -------
        If you resume from a checkpoint or restart the process and reuse the same run_dir,
        existing agents might already have recorded births/deaths/totals.
        This method repopulates `_life` and `_offspring_count`.

        Parsing strategy
        ---------------
        - Uses csv.DictReader (header-driven).
        - Attempts to parse known fields into appropriate types.
        - Gracefully ignores parse errors (fail-safe telemetry).
        """
        if not self.agent_life_path.exists():
            return

        try:
            with self.agent_life_path.open("r", newline="", encoding="utf-8") as f:
                r = csv.DictReader(f)
                for row in r:
                    if not row:
                        continue

                    aid_s = (row.get("agent_id") or "").strip()
                    if aid_s == "":
                        continue

                    aid = int(aid_s)
                    rec: Dict[str, Any] = {}

                    # Basic integer/string fields.
                    # - slot_id/team/unit_type are numeric but might appear as "12.0"
                    # - notes is free-form text
                    for k in (
                        "slot_id",
                        "team",
                        "unit_type",
                        "notes",
                        "spawn_reason",
                        "parent_unit_type",
                        "parent_team",
                        "parent_generation",
                        "generation",
                    ):
                        v = row.get(k, "")
                        if v != "":
                            try:
                                if k in ("notes", "spawn_reason"):
                                    rec[k] = str(v)
                                else:
                                    rec[k] = int(float(v))
                            except Exception:
                                rec[k] = str(v)

                    # Tick-like integer fields.
                    for k in ("born_tick", "death_tick", "parent_id", "kills_total"):
                        v = (row.get(k) or "").strip()
                        if v != "":
                            try:
                                rec[k] = int(float(v))
                            except Exception:
                                # If parse fails, skip silently (telemetry should not kill the sim).
                                pass

                    # Float totals for damage fields.
                    for k in ("damage_dealt_total", "damage_taken_total"):
                        v = (row.get(k) or "").strip()
                        if v != "":
                            try:
                                rec[k] = float(v)
                            except Exception:
                                pass

                    for k in ("mutation_flag", "rare_mutation"):
                        v = (row.get(k) or "").strip()
                        if v != "":
                            try:
                                rec[k] = bool(int(float(v)))
                            except Exception:
                                rec[k] = str(v).strip().lower() in ("1", "true", "yes", "y", "on")

                    for k in ("mutation_delta_hp", "mutation_delta_atk"):
                        v = (row.get(k) or "").strip()
                        if v != "":
                            try:
                                rec[k] = float(v)
                            except Exception:
                                pass

                    v = (row.get("mutation_delta_vis") or "").strip()
                    if v != "":
                        try:
                            rec["mutation_delta_vis"] = int(float(v))
                        except Exception:
                            pass

                    # Movement totals (optional columns; may not exist in older runs).
                    for k in (
                        "moves_attempted",
                        "moves_success",
                        "moves_blocked_wall",
                        "moves_blocked_occupied",
                        "moves_conflict_lost",
                        "moves_conflict_tie",
                        "cells_walked_total_l1",
                        "last_seen_tick",
                    ):
                        v = (row.get(k) or "").strip()
                        if v != "":
                            try:
                                rec[k] = int(float(v))
                            except Exception:
                                pass

                    # Reward totals / PPO category columns (optional in older runs).
                    for k in (
                        "reward_total",
                        "reward_individual_total",
                        "reward_team_total",
                        "reward_hp",
                        "reward_kill_individual",
                        "reward_damage_dealt_individual",
                        "reward_damage_taken_penalty",
                        "reward_contested_cp_individual",
                        "reward_team_kill",
                        "reward_team_death",
                        "reward_team_cp",
                        "reward_healing_recovered",
                    ):
                        v = (row.get(k) or "").strip()
                        if v != "":
                            try:
                                rec[k] = float(v)
                            except Exception:
                                pass

                    # Structured death metadata.
                    v = (row.get("death_cause") or "").strip()
                    if v != "":
                        rec["death_cause"] = str(v)

                    for k in ("killer_id", "killer_slot", "killer_team"):
                        v = (row.get(k) or "").strip()
                        if v != "":
                            try:
                                rec[k] = int(float(v))
                            except Exception:
                                pass

                    self._life[aid] = rec

                    # Offspring count stored separately.
                    oc = (row.get("offspring_count") or "").strip()
                    if oc != "":
                        try:
                            self._offspring_count[aid] = int(float(oc))
                        except Exception:
                            pass
        except Exception as e:
            self._anomaly(f"rehydrate agent_life failed: {e}")

    def _anomaly(self, msg: str) -> None:
        """Handle an internal telemetry anomaly."""
        if self.abort_on_anomaly:
            raise AssertionError(msg)
        return

    def _require_birth(self, agent_id: int, context: str) -> None:
        """Require that ``agent_id`` already exists in the life ledger."""
        if agent_id not in self._life:
            self._anomaly(f"{context}: missing birth for agent_id={agent_id}")

    def _append_csv_rows(
        self,
        path: Path,
        fieldnames: List[str],
        rows: List[Dict[str, Any]],
        strict_schema: bool = False,
    ) -> None:
        """
        Append rows to a CSV file, writing the header if the file does not exist.

        Why append?
        -----------
        - For lineage_edges.csv and tick_summary.csv, we want a growing log.
        - For agent_static.csv, we want to append new unique agent rows.

        Technical notes
        ---------------
        - newline="" is recommended for csv module to avoid blank lines on Windows.
        - DictWriter enforces consistent column order via fieldnames.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        exists = path.exists()
        write_header = (not exists) or (exists and path.stat().st_size == 0)

        if exists and (not write_header) and strict_schema:
            try:
                with path.open("r", newline="", encoding="utf-8") as rf:
                    r = csv.reader(rf)
                    existing_header = next(r, None)
                if existing_header is not None and list(existing_header) != list(fieldnames):
                    raise RuntimeError(
                        f"[telemetry] CSV header mismatch for append: {path.name} "
                        f"existing={existing_header} new={list(fieldnames)}"
                    )
            except RuntimeError:
                raise
            except Exception as e:
                raise RuntimeError(f"[telemetry] Failed to validate CSV header for {path.name}: {e}")

        with path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                w.writeheader()
            for r in rows:
                # Emit only declared fields; stable schema + tolerant to extra keys.
                # Missing fields become empty strings (analysis-friendly CSV).
                if set(r.keys()) != set(fieldnames):
                    r = {k: r.get(k, "") for k in fieldnames}
                w.writerow(r)

    def write_schema_manifest(self, manifest: Dict[str, Any]) -> None:
        """
        Persist a small telemetry schema/mechanics manifest.
        This is the append-in-place compatibility contract.
        """
        if not self.enabled:
            return

        out = dict(manifest or {})
        out.setdefault("schema_version", self.schema_version_str)
        _atomic_write_text(self.schema_manifest_path, json.dumps(out, indent=2, sort_keys=True))

    def validate_schema_manifest_compat(self, manifest: Dict[str, Any]) -> None:
        """
        Refuse append-in-place when the existing run uses a different telemetry contract.
        """
        if not self.enabled:
            return
        if not self.schema_manifest_path.exists():
            raise RuntimeError(
                f"[telemetry] missing schema manifest for append-in-place resume: {self.schema_manifest_path}"
            )

        existing = json.loads(self.schema_manifest_path.read_text(encoding="utf-8"))
        incoming = dict(manifest or {})

        keys = (
            "schema_version",
            "lineage_fields",
            "reward_fields",
            "death_causes",
            "mechanics",
        )
        mismatches = []
        for k in keys:
            if existing.get(k) != incoming.get(k):
                mismatches.append(k)
        if mismatches:
            existing_subset = {k: existing.get(k) for k in mismatches}
            incoming_subset = {k: incoming.get(k) for k in mismatches}
            raise RuntimeError(
                f"[telemetry] append-in-place manifest mismatch: keys={mismatches} "
                f"existing={existing_subset} "
                f"incoming={incoming_subset}"
            )

    def flush(self, reason: str = "manual") -> None:
        """
        Best-effort mid-run flush used by checkpointing and shutdown paths.
        """
        if not self.enabled:
            return

        self._flush_event_chunk()
        self._flush_agent_life_snapshot()
        self._flush_move_summary()
        self._flush_counters()
        self._drain_and_flush_ppo_rich_telemetry(
            observed_tick=self._last_tick_seen,
            force=True,
        )
        if self._last_tick_seen is not None and self.tick_summary_every > 0:
            self._write_tick_summary(tick=int(self._last_tick_seen))

    def _event_schema_version_int(self) -> int:
        """Return schema version as int for event payloads (robust to \"v2\" strings)."""
        if hasattr(self, "schema_version_int"):
            return int(getattr(self, "schema_version_int"))
        return _parse_schema_version_int(getattr(self, "schema_version", 2), default=2)

    # Event ordering context (optional, additive)
    def begin_tick_event_context(self, tick: int) -> None:
        """
        Begin/reset per-tick event ordering counters.

        Engine can call this at the start of a tick (or before a new tick's event-producing phase).
        Safe to call multiple times with the same tick (no reset in that case).
        """
        if not self.enabled:
            return

        t = int(tick)
        if self._event_ctx_tick != t:
            self._event_ctx_tick = t
            self._event_ctx_phase = None
            self._event_ctx_tick_seq = 0
            self._event_ctx_phase_seq = 0

    def set_event_phase(self, phase: str, tick: Optional[int] = None) -> None:
        """
        Set current event phase label (e.g., 'combat_damage', 'combat_kill_credit', 'move', 'respawn_birth').

        This resets phase-local sequence while preserving tick-global sequence.
        """
        if not self.enabled:
            return

        if tick is not None:
            self.begin_tick_event_context(int(tick))
        elif self._event_ctx_tick is None:
            # If engine forgot to set tick, keep graceful behavior (no exception).
            return

        phase_s = str(phase)
        if self._event_ctx_phase != phase_s:
            self._event_ctx_phase = phase_s
            self._event_ctx_phase_seq = 0

    def clear_event_context(self) -> None:
        """Optional cleanup hook; not required for correctness."""
        if not self.enabled:
            return
        self._event_ctx_phase = None

    def _emit_event(self, ev: Dict[str, Any]) -> None:
        """
        Buffer a single event dict and flush if the buffer reaches chunk size.

        Event buffering rationale
        -------------------------
        Writing to disk on every event can be expensive (especially on Windows),
        and it can degrade simulation performance.
        Buffering batches events into memory and flushes them as a chunk.

        Format policy
        -------------
        - JSONL only in this implementation.
        - If config requests something else, we degrade safely to JSONL and annotate notes.
        """
        if not self.enabled:
            return

        if self.events_format.lower() != "jsonl":
            # Degrade safely to jsonl; do not invent other writers.
            ev = dict(ev)
            ev["notes"] = (ev.get("notes", "") + " events_format_forced_jsonl").strip()

        # Additive event-order metadata (only if engine set tick/phase context).
        # This preserves backwards compatibility: downstream parsers can ignore these fields.
        if self._event_ctx_tick is not None:
            ev = dict(ev)  # avoid mutating caller-owned dict
            # Tick-global sequence across all event types/phases for this tick.
            ev.setdefault("tick_event_seq", int(self._event_ctx_tick_seq))
            self._event_ctx_tick_seq += 1

            # Optional phase label + sequence within phase.
            if self._event_ctx_phase is not None:
                ev.setdefault("phase", str(self._event_ctx_phase))
                ev.setdefault("phase_event_seq", int(self._event_ctx_phase_seq))
                self._event_ctx_phase_seq += 1

        self._events_buf.append(ev)
        if len(self._events_buf) >= self.event_chunk_size:
            self._flush_event_chunk()

    def _flush_event_chunk(self) -> None:
        """
        Flush buffered events into a new events_XXXXXX.jsonl file (atomic overwrite).

        Chunking scheme
        --------------
        Each chunk is its own file:
            telemetry/events/events_000000.jsonl
            telemetry/events/events_000001.jsonl
            ...

        Pros:
        - Avoid huge single-file logs.
        - Easy parallel processing later.
        - Resume-safe if chunk index is discovered from disk.

        Cons:
        - Many small files (still generally fine).
        """
        if not self._events_buf:
            return

        out = self.events_dir / f"events_{self._chunk_idx:06d}.jsonl"

        # JSON Lines: one JSON object per line.
        # This is robust because a parser can read line-by-line.
        lines = "\n".join(json.dumps(e, ensure_ascii=False) for e in self._events_buf) + "\n"
        _atomic_write_text(out, lines)

        self._events_buf.clear()
        self._chunk_idx += 1

    def _flush_agent_life_snapshot(self) -> None:
        """
        Write a full snapshot of `_life` to agent_life.csv (overwrite atomically).

        Why snapshot overwrite instead of append?
        -----------------------------------------
        AgentLife is a "current state" table: we want the latest known totals per agent.
        The simplest way is to write the whole table periodically.

        Atomic overwrite avoids partial snapshots.

        Field design
        ------------
        - Includes core life history (born_tick, death_tick, lifespan_ticks)
        - Includes totals (kills, damage)
        - Includes several "future extension" columns (moves_attempted, reward_total, etc.).
        """
        fieldnames = [
            "agent_id", "slot_id", "team", "unit_type",
            "born_tick", "death_tick", "lifespan_ticks", "parent_id", "offspring_count",
            "spawn_reason", "parent_unit_type", "parent_team", "parent_generation", "generation",
            "mutation_flag", "rare_mutation", "mutation_delta_hp", "mutation_delta_atk", "mutation_delta_vis",
            "kills_total", "damage_dealt_total", "damage_taken_total", "notes",

            "moves_attempted", "moves_success", "moves_blocked_wall", "moves_blocked_occupied",
            "moves_conflict_lost", "moves_conflict_tie",
            "cells_walked_total_l1", "last_seen_tick",

            "reward_total",
            "reward_individual_total",
            "reward_team_total",
            "reward_hp",
            "reward_kill_individual",
            "reward_damage_dealt_individual",
            "reward_damage_taken_penalty",
            "reward_contested_cp_individual",
            "reward_team_kill",
            "reward_team_death",
            "reward_team_cp",
            "reward_healing_recovered",

            "death_cause",
            "killer_id",
            "killer_slot",
            "killer_team",
        ]

        rows: List[Dict[str, Any]] = []
        try:
            import torch
        except Exception:
            torch = None

        # Movement totals integration (AgentLife snapshot)
        # We maintain per-slot movement counters (attempted/success/blocked/etc.)
        # via vectorized updates (see record_move_totals_by_slot()).
        # IMPORTANT PERFORMANCE NOTE:
        # - If the simulation runs on CUDA, reading per-agent counters via .item()
        #   inside the loop would cause *thousands of device synchronizations*.
        # - Therefore, we bulk-copy the counter tensors to CPU ONCE per snapshot,
        #   then do cheap Python list indexing while building CSV rows.
        move_lists = None
        if self.enabled and self.log_moves and getattr(self, "_move_slot_attempted", None) is not None and torch is not None:
            try:
                move_rows = torch.stack(
                    (
                        self._move_slot_attempted,
                        self._move_slot_success,
                        self._move_slot_blocked_wall,
                        self._move_slot_blocked_occupied,
                        self._move_slot_conflict_lost,
                        self._move_slot_conflict_tie,
                        self._move_slot_cells_l1,
                        self._move_slot_last_seen,
                    ),
                    dim=0,
                ).detach().to(torch.int64).cpu()
                move_lists = tuple(move_rows[i].tolist() for i in range(int(move_rows.size(0))))
            except Exception:
                move_lists = None

        reward_lists = None
        if self.enabled and getattr(self, "_reward_slot_total", None) is not None and torch is not None:
            try:
                reward_rows = torch.stack(
                    (
                        self._reward_slot_total,
                        self._reward_slot_individual_total,
                        self._reward_slot_team_total,
                        self._reward_slot_hp,
                        self._reward_slot_kill_individual,
                        self._reward_slot_damage_dealt_individual,
                        self._reward_slot_damage_taken_penalty,
                        self._reward_slot_contested_cp_individual,
                        self._reward_slot_team_kill,
                        self._reward_slot_team_death,
                        self._reward_slot_team_cp,
                        self._reward_slot_healing_recovered,
                    ),
                    dim=0,
                ).detach().to(torch.float32).cpu()
                reward_lists = tuple(reward_rows[i].tolist() for i in range(int(reward_rows.size(0))))
            except Exception:
                reward_lists = None

        for aid, rec in sorted(self._life.items(), key=lambda kv: kv[0]):
            r = dict(rec)
            r["agent_id"] = aid
            r["offspring_count"] = int(self._offspring_count.get(aid, 0))

            # lifespan_ticks is only computable if both born and death ticks exist.
            bt = r.get("born_tick")
            dt = r.get("death_tick")
            if bt is not None and dt is not None:
                r["lifespan_ticks"] = int(dt) - int(bt)
            else:
                r["lifespan_ticks"] = ""

            # Fill movement columns for currently-alive agents from per-slot counters.
            # For dead agents, values are frozen into the life record at record_deaths().
            if move_lists is not None and r.get("death_tick", None) is None:
                try:
                    sid = r.get("slot_id", None)
                    if sid is not None and sid != "":
                        s = int(sid)
                        a0, a1, a2, a3, a4, a5, a6, a7 = move_lists
                        if 0 <= s < len(a0):
                            r["moves_attempted"] = int(a0[s])
                            r["moves_success"] = int(a1[s])
                            r["moves_blocked_wall"] = int(a2[s])
                            r["moves_blocked_occupied"] = int(a3[s])
                            r["moves_conflict_lost"] = int(a4[s])
                            r["moves_conflict_tie"] = int(a5[s])
                            r["cells_walked_total_l1"] = int(a6[s])
                            lst = int(a7[s])
                            if lst >= 0:
                                r["last_seen_tick"] = lst
                except Exception:
                    pass

            # Fill reward columns for currently-alive agents from per-slot accumulators.
            # For dead agents, values are frozen into the life record at record_deaths().
            if reward_lists is not None and r.get("death_tick", None) is None:
                try:
                    sid = r.get("slot_id", None)
                    if sid is not None and sid != "":
                        s = int(sid)
                        (r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11) = reward_lists
                        if 0 <= s < len(r0):
                            r["reward_total"] = float(r0[s])
                            r["reward_individual_total"] = float(r1[s])
                            r["reward_team_total"] = float(r2[s])
                            r["reward_hp"] = float(r3[s])
                            r["reward_kill_individual"] = float(r4[s])
                            r["reward_damage_dealt_individual"] = float(r5[s])
                            r["reward_damage_taken_penalty"] = float(r6[s])
                            r["reward_contested_cp_individual"] = float(r7[s])
                            r["reward_team_kill"] = float(r8[s])
                            r["reward_team_death"] = float(r9[s])
                            r["reward_team_cp"] = float(r10[s])
                            r["reward_healing_recovered"] = float(r11[s])
                except Exception:
                    pass

            # Emit exactly the defined fields; missing fields become "" for CSV cleanliness.
            rows.append({k: r.get(k, "") for k in fieldnames})

        # CSV serialize using StringIO so we can atomic-write the complete content.
        import io
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

        _atomic_write_text(self.agent_life_path, buf.getvalue())

    def _flush_move_summary(self) -> None:
        """Flush buffered movement aggregate rows to move_summary.csv."""
        if not self.enabled or not self._move_agg_buf:
            return
        fieldnames = [
            "tick",
            "attempted",
            "can_move",
            "moved",
            "blocked_wall",
            "blocked_occupied",
            "conflict_lost",
            "conflict_tie",
        ]
        self._append_csv_rows(self.move_summary_path, fieldnames=fieldnames, rows=self._move_agg_buf)
        self._move_agg_buf.clear()

    def _flush_counters(self) -> None:
        """Flush buffered (tick, key, value) tuples to counters.csv."""
        if not self.enabled or not self._counters_buf:
            return
        rows: List[Dict[str, Any]] = [
            {"tick": int(t), "key": str(k), "value": float(v)} for (t, k, v) in self._counters_buf
        ]
        self._append_csv_rows(self.counters_path, fieldnames=["tick", "key", "value"], rows=rows)
        self._counters_buf.clear()

    # Movement telemetry (aggregates + optional per-agent events)

    def record_move_summary(
        self,
        tick: int,
        attempted: int,
        can_move: int,
        moved: int,
        blocked_wall: int,
        blocked_occupied: int,
        conflict_lost: int,
        conflict_tie: int,
    ) -> None:
        """
        Record cheap per-tick movement aggregates.

        Output:
          telemetry/move_summary.csv (append)
        """
        if not self.enabled or not self.log_moves:
            return

        self._move_agg_buf.append({
            "tick": int(tick),
            "attempted": int(attempted),
            "can_move": int(can_move),
            "moved": int(moved),
            "blocked_wall": int(blocked_wall),
            "blocked_occupied": int(blocked_occupied),
            "conflict_lost": int(conflict_lost),
            "conflict_tie": int(conflict_tie),
        })

        # Safety valve: if buffer grows very large (misconfig), flush immediately.
        if len(self._move_agg_buf) >= max(1, int(self.event_chunk_size)):
            self._flush_move_summary()

    def _ensure_move_slot_tensors(self, n_slots: int, device) -> None:
        """
        Lazily allocate per-slot movement counter tensors.

        This is called from:
        - record_birth() (to reset/seed a slot)
        - record_deaths() (to freeze a slot)
        - record_move_totals_by_slot() (to update counts)

        Implementation notes
        --------------------
        - We import torch lazily to avoid forcing torch import at module import time.
        - Tensors live on the same device as the simulation (CPU or CUDA).
        """
        if (not self.enabled) or (not self.log_moves):
            return

        try:
            import torch  # local import (torch is already a project dependency)
        except Exception:
            return

        n = int(n_slots)
        if n <= 0:
            return

        cur = getattr(self, "_move_slot_attempted", None)
        if cur is not None and int(cur.numel()) == n and str(cur.device) == str(device):
            return

        def _new_zeros() -> "torch.Tensor":
            return torch.zeros((n,), device=device, dtype=torch.int64)

        # If we already had counters, preserve them (best-effort) when resizing/moving.
        old_attempted = cur

        self._move_slot_attempted = _new_zeros()
        self._move_slot_success = _new_zeros()
        self._move_slot_blocked_wall = _new_zeros()
        self._move_slot_blocked_occupied = _new_zeros()
        self._move_slot_conflict_lost = _new_zeros()
        self._move_slot_conflict_tie = _new_zeros()
        self._move_slot_cells_l1 = _new_zeros()
        self._move_slot_last_seen = torch.full((n,), -1, device=device, dtype=torch.int64)

        if old_attempted is not None:
            try:
                m = min(int(old_attempted.numel()), n)
                if m > 0:
                    self._move_slot_attempted[:m] = old_attempted[:m].to(device=device)
            except Exception:
                pass

    def _ensure_reward_slot_tensors(self, n_slots: int, device) -> None:
        """
        Lazily allocate per-slot PPO reward accumulator tensors.

        Design goal:
        - Keep reward-category accumulation on the simulation device (CPU/CUDA).
        - Avoid per-tick GPU->CPU sync for all agents.
        - Freeze totals into AgentLife on death; bulk-copy only during snapshots.
        """
        if not self.enabled:
            return

        try:
            import torch
        except Exception:
            return

        n = int(n_slots)
        if n <= 0:
            return

        cur = getattr(self, "_reward_slot_total", None)
        if cur is not None and int(cur.numel()) == n and str(cur.device) == str(device):
            return

        names = (
            "_reward_slot_total",
            "_reward_slot_individual_total",
            "_reward_slot_team_total",
            "_reward_slot_hp",
            "_reward_slot_kill_individual",
            "_reward_slot_damage_dealt_individual",
            "_reward_slot_damage_taken_penalty",
            "_reward_slot_contested_cp_individual",
            "_reward_slot_team_kill",
            "_reward_slot_team_death",
            "_reward_slot_team_cp",
            "_reward_slot_healing_recovered",
        )

        old = {name: getattr(self, name, None) for name in names}

        def _new_zeros():
            return torch.zeros((n,), device=device, dtype=torch.float32)

        for name in names:
            setattr(self, name, _new_zeros())

        old_total = old.get("_reward_slot_total", None)
        if old_total is not None:
            try:
                m = min(int(old_total.numel()), n)
                if m > 0:
                    for name in names:
                        prev = old.get(name, None)
                        if prev is not None:
                            getattr(self, name)[:m] = prev[:m].to(device=device, dtype=torch.float32)
            except Exception:
                pass

    def record_move_totals_by_slot(self, tick: int, slot_ids, outcome_code) -> None:
        """
        Update per-slot movement totals for AgentLife snapshot.

        Inputs are tensors aligned per attempted move:
            slot_ids[i]      -> registry slot attempting a move
            outcome_code[i]  -> outcome code for that attempt

        Outcome codes (must match TickEngine):
            0 = success
            1 = blocked_wall
            2 = blocked_occupied
            3 = conflict_lost
            4 = conflict_tie
        """
        if (not self.enabled) or (not self.log_moves):
            return

        reg = getattr(self, "_registry", None)
        if reg is None or not hasattr(reg, "agent_data"):
            return

        try:
            import torch  # local import
        except Exception:
            return

        try:
            n_slots = int(reg.agent_data.shape[0])
        except Exception:
            return

        if n_slots <= 0:
            return

        # Prefer the device of slot_ids (engine device); fall back to registry tensor device.
        device = getattr(slot_ids, "device", getattr(reg.agent_data, "device", "cpu"))
        self._ensure_move_slot_tensors(n_slots=n_slots, device=device)

        if getattr(self, "_move_slot_attempted", None) is None:
            return

        try:
            s = slot_ids.to(dtype=torch.long, device=device)
            oc = outcome_code.to(dtype=torch.int16, device=device)
            if int(s.numel()) == 0:
                return

            ones = torch.ones((int(s.numel()),), device=device, dtype=torch.int64)

            # Always increment attempted.
            self._move_slot_attempted.index_add_(0, s, ones)

            # Update last_seen_tick for attempted movers.
            try:
                self._move_slot_last_seen.index_fill_(0, s, int(tick))
            except Exception:
                self._move_slot_last_seen[s] = int(tick)

            # Per-outcome increments.
            m0 = (oc == 0)
            if bool(m0.any()):
                ss = s[m0]
                oo = ones[m0]
                self._move_slot_success.index_add_(0, ss, oo)
                self._move_slot_cells_l1.index_add_(0, ss, oo)  # 1 cell per successful step

            m1 = (oc == 1)
            if bool(m1.any()):
                self._move_slot_blocked_wall.index_add_(0, s[m1], ones[m1])

            m2 = (oc == 2)
            if bool(m2.any()):
                self._move_slot_blocked_occupied.index_add_(0, s[m2], ones[m2])

            m3 = (oc == 3)
            if bool(m3.any()):
                self._move_slot_conflict_lost.index_add_(0, s[m3], ones[m3])

            m4 = (oc == 4)
            if bool(m4.any()):
                self._move_slot_conflict_tie.index_add_(0, s[m4], ones[m4])

        except Exception:
            return

    def record_move_events(
        self,
        tick: int,
        agent_ids: List[int],
        actions: List[int],
        from_x: List[int],
        from_y: List[int],
        to_x: List[int],
        to_y: List[int],
        outcome_code: List[int],
    ) -> None:
        """
        Record sampled per-agent move events into the main events stream.

        Event schema (type="move")
        -------------------------
          - tick, agent_id, action
          - from_x, from_y, to_x, to_y
          - outcome: one of:
              "success" | "blocked_wall" | "blocked_occupied" | "conflict_lost" | "conflict_tie"

        Important:
        ----------
        This method expects inputs already sampled/capped by the engine to keep overhead minimal.
        """
        if not self.enabled or not self.log_moves:
            return

        # If per-agent move events are disabled, treat as no-op (aggregates still work).
        if int(getattr(self, "move_events_every", 0)) <= 0:
            return

        # Defensive: align lengths to shortest.
        n = min(
            len(agent_ids),
            len(actions),
            len(from_x),
            len(from_y),
            len(to_x),
            len(to_y),
            len(outcome_code),
        )

        def _outcome_str(code: int) -> str:
            c = int(code)
            if c == 0:
                return "success"
            if c == 1:
                return "blocked_wall"
            if c == 2:
                return "blocked_occupied"
            if c == 3:
                return "conflict_lost"
            if c == 4:
                return "conflict_tie"
            return "unknown"

        for i in range(n):
            self._require_birth(int(agent_ids[i]), "move")
            self._emit_event({
                "schema_version": self._event_schema_version_int(),
                "tick": int(tick),
                "type": "move",
                "agent_id": int(agent_ids[i]),
                "action": int(actions[i]),
                "from_x": int(from_x[i]),
                "from_y": int(from_y[i]),
                "to_x": int(to_x[i]),
                "to_y": int(to_y[i]),
                "outcome": _outcome_str(outcome_code[i]),
            })

    # Generic counters stream (lightweight extension point)

    def emit_counters(self, tick: int, counters: Dict[str, float]) -> None:
        """
        Emit lightweight per-tick counters in long format to telemetry/counters.csv.

        Format:
          tick,key,value

        This is an intentionally schema-stable extension point:
        you can add new counter keys without changing the file schema.
        """
        if not self.enabled:
            return

        every = int(getattr(self, "counters_every", 0))
        if every <= 0 or (int(tick) % every) != 0:
            return

        for k, v in counters.items():
            self._counters_buf.append((int(tick), str(k), float(v)))

        if len(self._counters_buf) >= max(1, int(self.event_chunk_size)):
            self._flush_counters()

    def validate(self) -> None:
        """
        Run consistency checks on `_life` according to validation level.

        Levels
        ------
        0: off
        1: basic
        2: strict

        Current implementation note
        ---------------------------
        The method currently runs the same checks regardless of 1 vs 2,
        but the config architecture supports differentiating in the future.

        What we check
        -------------
        - death_tick >= born_tick
        - kills_total is not negative
        - damage totals are not negative

        Why negative totals matter
        --------------------------
        Totals represent cumulative sums of non-negative quantities.
        A negative value suggests overflow, wrong sign, or memory corruption.
        """
        if self.validate_level <= 0:
            return

        for aid, r in self._life.items():
            bt = r.get("born_tick")
            dt = r.get("death_tick")

            if bt is not None and dt is not None and int(dt) < int(bt):
                self._anomaly(f"agent_id={aid}: death_tick < born_tick ({dt} < {bt})")

            if r.get("kills_total", 0) < 0:
                self._anomaly(f"agent_id={aid}: negative kills_total")

            if r.get("damage_dealt_total", 0.0) < 0:
                self._anomaly(f"agent_id={aid}: negative damage_dealt_total")

            if r.get("damage_taken_total", 0.0) < 0:
                self._anomaly(f"agent_id={aid}: negative damage_taken_total")

        for parent_id, cnt in self._lineage_edge_count.items():
            if int(self._offspring_count.get(int(parent_id), 0)) != int(cnt):
                self._anomaly(
                    f"parent_id={parent_id}: offspring_count mismatch life={self._offspring_count.get(int(parent_id), 0)} edges={cnt}"
                )

    # Public API called from main/tick

    def attach_context(self, registry: Any, stats: Any, ppo_runtime: Any = None) -> None:
        """
        Attach references to simulation objects that telemetry can read.

        Why this method exists
        ----------------------
        Telemetry sometimes needs access to:
        - registry.agent_data tensor
        - registry.brains list (for brain_type and parameter counts)
        - stats (for tick summary metrics)

        By attaching these references, we avoid deep refactors of the engine:
        telemetry can be "plugged in" with minimal integration points.

        Important:
        ----------
        - Telemetry must never mutate these objects.
        - These are observational references only.
        """
        self._registry = registry
        self._stats = stats
        self._ppo_runtime = ppo_runtime

    def _ppo_rich_fieldnames(self) -> List[str]:
        """
        Stable schema for dedicated PPO training diagnostics CSV.
        This file is append-only and intentionally explicit for downstream analysis.
        """
        return [
            "schema_version",
            "observed_tick",
            "wall_time_unix",
            "wall_time_iso",
            "row_seq",
            "granularity_requested",
            "granularity_effective",
            "update_seq",
            "ppo_step",
            "trained_slots",
            "samples",
            "optimizer_steps",
            "epochs_run_total",
            "lr_before_mean",
            "lr_after_mean",
            "loss_pi_mean",
            "loss_v_mean",
            "loss_ent_mean",
            "loss_total_mean",
            "entropy_mean",
            "approx_kl_mean",
            "approx_kl_max",
            "clip_frac_mean",
            "clip_frac_max",
            "ratio_mean",
            "ratio_std",
            "ratio_min",
            "ratio_max",
            "adv_norm_mean",
            "adv_norm_std",
            "adv_norm_min",
            "adv_norm_max",
            "ret_mean",
            "ret_std",
            "ret_min",
            "ret_max",
            "rew_mean",
            "rew_std",
            "rew_min",
            "rew_max",
            "value_old_mean",
            "value_old_std",
            "value_old_min",
            "value_old_max",
            "value_pred_mean",
            "value_pred_std",
            "value_pred_min",
            "value_pred_max",
            "grad_norm_mean",
            "grad_norm_max",
            "explained_var_mean",
            "explained_var_min",
            "explained_var_max",
            "mask_valid_actions_mean",
            "mask_valid_actions_min",
            "mask_valid_actions_max",
            "clip_epsilon",
            "target_kl",
            "ppo_epochs_cfg",
            "ppo_minibatches_cfg",
            "max_grad_norm_cfg",
            "notes",
        ]

    def _drain_and_flush_ppo_rich_telemetry(self, *, observed_tick: Optional[int] = None, force: bool = False) -> None:
        """
        Pull pending PPO training diagnostic rows from PPO runtime and append them to
        telemetry/ppo_training_telemetry.csv in an append-safe way.
        """
        if (not self.enabled) or (not self.ppo_rich_csv_enabled):
            return

        p = getattr(self, "_ppo_runtime", None)
        if p is None:
            return

        drain_fn = getattr(p, "drain_rich_telemetry_rows", None)
        if not callable(drain_fn):
            return

        try:
            pending = drain_fn()
        except Exception as e:
            self._anomaly(f"ppo rich telemetry drain failed: {e}")
            return
        # Telemetry-only fallback path:
        # If PPO training summary advanced but no rich row was queued/drained, synthesize one row.
        # This keeps telemetry resilient without changing PPO optimizer/training semantics.
        if not pending:
            try:
                last_summary = getattr(p, "last_train_summary", None)
                if isinstance(last_summary, dict):
                    last_u = int(float(last_summary.get("update_seq", 0) or 0))
                    if last_u > int(self._ppo_rich_last_fallback_update_seq):
                        try:
                            row_seq_val = float(getattr(p, "_rich_telemetry_row_seq", 0))
                        except Exception:
                            row_seq_val = ""

                        pending = [{
                            "row_seq": row_seq_val,
                            "granularity_requested": str(getattr(self, "ppo_rich_level", "update")),
                            "granularity_effective": "update",
                            **dict(last_summary),
                            "notes": "telemetry_fallback_from_last_train_summary",
                        }]
                        self._ppo_rich_last_fallback_update_seq = last_u
            except Exception as e:
                self._anomaly(f"ppo rich telemetry fallback synth failed: {e}")

        if pending:
            try:
                import datetime as _dt
                import time as _time

                fields = self._ppo_rich_fieldnames()
                wall_unix = float(_time.time())
                wall_iso = _dt.datetime.utcfromtimestamp(wall_unix).isoformat(timespec="seconds") + "Z"
                obs_tick = (int(observed_tick) if observed_tick is not None else "")

                for row in pending:
                    if not isinstance(row, dict):
                        continue
                    out = dict(row)
                    out.setdefault("schema_version", self.schema_version)
                    out.setdefault("observed_tick", obs_tick)
                    out.setdefault("wall_time_unix", wall_unix)
                    out.setdefault("wall_time_iso", wall_iso)
                    self._ppo_rich_buf.append({k: out.get(k, "") for k in fields})
            except Exception as e:
                self._anomaly(f"ppo rich telemetry normalize failed: {e}")

        if self._ppo_rich_buf and (force or (len(self._ppo_rich_buf) >= int(self.ppo_rich_flush_every))):
            rows = list(self._ppo_rich_buf)
            fields = self._ppo_rich_fieldnames()

            try:
                self._append_csv_rows(
                    self.ppo_training_telemetry_path,
                    fieldnames=fields,
                    rows=rows,
                    strict_schema=self.telemetry_append_schema_strict,
                )
                self._ppo_rich_buf.clear()

            except Exception as e:
                # Minimal, localized recovery path:
                # On resume-in-place, ppo_training_telemetry.csv may exist with an older header.
                # Strict append then fails silently (because _anomaly is silent by default).
                # We auto-roll to a sibling file and continue logging instead of dropping rows.
                msg = str(e)
                recovered = False

                try:
                    if ("CSV header mismatch for append" in msg) and ("ppo_training_telemetry.csv" in msg):
                        alt_path = self.telemetry_dir / "ppo_training_telemetry_resume.csv"
                        primary_path = self.ppo_training_telemetry_path
                        target_path = self.telemetry_dir / "ppo_training_telemetry_resume.csv"
                        rewrote_header_only_primary = False

                        # If the primary PPO rich CSV is header-only (no data rows), it is safe to
                        # reinitialize it in-place with the current schema and keep the canonical filename.
                        try:
                            if primary_path.exists():
                                with primary_path.open("r", newline="", encoding="utf-8") as rf:
                                    r = csv.reader(rf)
                                    _ = next(r, None)  # existing header
                                    first_data = next(r, None)
                                if first_data is None:
                                    # Truncate so _append_csv_rows() can recreate header with current schema.
                                    with primary_path.open("w", newline="", encoding="utf-8"):
                                        pass
                                    target_path = primary_path
                                    rewrote_header_only_primary = True
                        except Exception:
                            # Fall back to sibling rollover path below.
                            pass

                        # True schema mismatch with existing data: roll to sibling and make it sticky
                        # so subsequent flushes don't repeatedly retry the mismatched primary file.
                        if target_path != primary_path:
                            self.ppo_training_telemetry_path = target_path
                        # Append to a fresh sibling file with current schema (safe; no corruption of old file).
                        self._append_csv_rows(
                            target_path,
                            fieldnames=fields,
                            rows=rows,
                            strict_schema=False,
                        )
                        self._ppo_rich_buf.clear()
                        recovered = True
                        tail = (
                            "rewrote header-only ppo_training_telemetry.csv in place and continued."
                            if rewrote_header_only_primary
                            else f"writing to {target_path.name} (current schema) instead."
                        )
                        # Tiny one-time style print is okay here because this is an actual data-loss prevention path.
                        try:
                            print(
                                "[telemetry] ppo rich CSV header mismatch on resume; "
                                + tail
                            )
                        except Exception:
                            pass
                except Exception as e2:
                    try:
                        print(f"[telemetry] ppo rich telemetry recovery failed: {type(e2).__name__}: {e2}")
                    except Exception:
                        pass

                if not recovered:
                    # Minimal surfacing while preserving existing anomaly policy.
                    self._anomaly(f"ppo rich telemetry flush failed: {e}")

    def _headless_summary_fieldnames(self) -> List[str]:
        fields = [
            "tick",
            "sim_elapsed_s",
            "wall_elapsed_s",
            "dt_since_last_summary_s",
            "tick_delta_since_last_summary",
            "tps_window",
            "tps_avg",
            "score_red",
            "score_blue",
            "cp_total_red",
            "cp_total_blue",
            "cp_delta_red",
            "cp_delta_blue",
            "alive_red",
            "alive_blue",
            "hp_mean_red",
            "hp_mean_blue",
            "kills_red",
            "kills_blue",
            "deaths_red",
            "deaths_blue",
            "dmg_dealt_red",
            "dmg_dealt_blue",
            "dmg_taken_red",
            "dmg_taken_blue",
            "dmg_dealt_delta_red",
            "dmg_dealt_delta_blue",
            "dmg_taken_delta_red",
            "dmg_taken_delta_blue",
            "tick_metrics_window_ticks",
        ]
        if self.headless_summary_include_gpu:
            fields.extend([
                "gpu_raw",
                "gpu_util_pct",
                "gpu_mem_used_mb",
                "gpu_mem_total_mb",
                "gpu_mem_pct",
                "gpu_power_w",
            ])
        if self.headless_summary_include_tick_metrics:
            fields.extend([f"tm_{k}_sum" for k in self._headless_tick_metric_keys])
        if self.headless_summary_include_ppo:
            fields.extend([
                "ppo_update_seq",
                "ppo_step",
                "ppo_trained_slots",
                "ppo_samples",
                "ppo_optimizer_steps",
                "ppo_epochs_run_total",
                "ppo_loss_pi_mean",
                "ppo_loss_v_mean",
                "ppo_loss_ent_mean",
                "ppo_entropy_mean",
                "ppo_approx_kl_max",
                "ppo_lr_before_mean",
                "ppo_lr_after_mean",
            ])
        return fields

    @staticmethod
    def _parse_gpu_summary_line(line: Optional[str]) -> Dict[str, Any]:
        if not line:
            return {}
        try:
            parts = [p.strip() for p in str(line).split(",")]
            if len(parts) < 4:
                return {"gpu_raw": str(line)}
            util = float(parts[0])
            mem_used = float(parts[1])
            mem_total = float(parts[2])
            power = float(parts[3])
            mem_pct = (100.0 * mem_used / mem_total) if mem_total > 0 else None
            return {
                "gpu_raw": str(line),
                "gpu_util_pct": util,
                "gpu_mem_used_mb": mem_used,
                "gpu_mem_total_mb": mem_total,
                "gpu_mem_pct": mem_pct,
                "gpu_power_w": power,
            }
        except Exception:
            return {"gpu_raw": str(line)}

    def on_headless_tick(
        self,
        tick: int,
        tick_metrics: Optional[Dict[str, Any]] = None,
        gpu_line: Optional[str] = None,
    ) -> None:
        """
        Additive headless-only hook:
        - accumulates per-tick metrics window
        - caches latest GPU probe (if provided)
        - writes telemetry_summary.csv every `tick_summary_every` ticks

        This is intentionally independent from console print verbosity.
        """
        if (not self.enabled) or (not self.headless_live_summary_enabled):
            return

        try:
            if gpu_line is not None:
                self._headless_gpu_raw = str(gpu_line)
                self._headless_gpu_parsed = self._parse_gpu_summary_line(gpu_line)

            if tick_metrics and self.headless_summary_include_tick_metrics:
                for k in self._headless_tick_metric_keys:
                    v = tick_metrics.get(k, None)
                    if v is None:
                        continue
                    try:
                        self._headless_tick_metrics_window[k] = float(self._headless_tick_metrics_window.get(k, 0.0)) + float(v)
                    except Exception:
                        continue
                self._headless_tick_metrics_window_ticks += 1

            if self.tick_summary_every > 0 and (int(tick) % int(self.tick_summary_every)) == 0:
                self._write_headless_summary(int(tick))
        except Exception as e:
            self._anomaly(f"headless summary hook failed: {e}")

    def _write_headless_summary(self, tick: int) -> None:
        if (not self.enabled) or (self._registry is None) or (self._stats is None):
            return
        try:
            import time
            from engine.agent_registry import COL_ALIVE, COL_TEAM, COL_HP

            now = float(time.perf_counter())
            if self._headless_wall_start is None:
                self._headless_wall_start = now
                self._headless_wall_start_tick = int(tick)

            wall_elapsed = ""
            if self._headless_wall_start is not None:
                wall_elapsed = float(now - self._headless_wall_start)

            dt_since_last = ""
            tick_delta = ""
            tps_window = ""
            if self._headless_prev_summary_wall is not None and self._headless_prev_summary_tick is not None:
                dt = float(now - self._headless_prev_summary_wall)
                dt_since_last = dt
                tick_delta_i = int(tick) - int(self._headless_prev_summary_tick)
                tick_delta = tick_delta_i
                if dt > 0 and tick_delta_i >= 0:
                    tps_window = float(tick_delta_i) / dt

            tps_avg = ""
            if (
                self.headless_summary_include_tps
                and self._headless_wall_start is not None
                and self._headless_wall_start_tick is not None
            ):
                dt0 = float(now - self._headless_wall_start)
                dticks0 = int(tick) - int(self._headless_wall_start_tick)
                if dt0 > 0 and dticks0 >= 0:
                    tps_avg = float(dticks0) / dt0

            data = self._registry.agent_data
            alive_mask = (data[:, COL_ALIVE] > 0.5)
            red_mask = alive_mask & (data[:, COL_TEAM] == 2.0)
            blue_mask = alive_mask & (data[:, COL_TEAM] == 3.0)
            red_n = int(red_mask.sum().item())
            blue_n = int(blue_mask.sum().item())
            mean_red = float(data[red_mask, COL_HP].mean().item()) if red_n > 0 else 0.0
            mean_blue = float(data[blue_mask, COL_HP].mean().item()) if blue_n > 0 else 0.0

            s = self._stats
            red_obj = getattr(s, "red", None)
            blue_obj = getattr(s, "blue", None)

            cp_red = float(getattr(red_obj, "cp_points", 0.0))
            cp_blue = float(getattr(blue_obj, "cp_points", 0.0))
            dmg_dealt_red = float(getattr(red_obj, "dmg_dealt", 0.0))
            dmg_dealt_blue = float(getattr(blue_obj, "dmg_dealt", 0.0))
            dmg_taken_red = float(getattr(red_obj, "dmg_taken", 0.0))
            dmg_taken_blue = float(getattr(blue_obj, "dmg_taken", 0.0))

            def _delta(cur: float, prev: Optional[float]):
                return (cur - float(prev)) if prev is not None else ""

            row: Dict[str, Any] = {
                "tick": int(tick),
                "sim_elapsed_s": float(getattr(s, "elapsed_seconds", 0.0)),
                "wall_elapsed_s": wall_elapsed,
                "dt_since_last_summary_s": (dt_since_last if self.headless_summary_include_tps else ""),
                "tick_delta_since_last_summary": (tick_delta if self.headless_summary_include_tps else ""),
                "tps_window": (tps_window if self.headless_summary_include_tps else ""),
                "tps_avg": (tps_avg if self.headless_summary_include_tps else ""),
                "score_red": float(getattr(red_obj, "score", 0.0)),
                "score_blue": float(getattr(blue_obj, "score", 0.0)),
                "cp_total_red": cp_red,
                "cp_total_blue": cp_blue,
                "cp_delta_red": _delta(cp_red, self._headless_prev_cp_red),
                "cp_delta_blue": _delta(cp_blue, self._headless_prev_cp_blue),
                "alive_red": red_n,
                "alive_blue": blue_n,
                "hp_mean_red": mean_red,
                "hp_mean_blue": mean_blue,
                "kills_red": float(getattr(red_obj, "kills", 0.0)),
                "kills_blue": float(getattr(blue_obj, "kills", 0.0)),
                "deaths_red": float(getattr(red_obj, "deaths", 0.0)),
                "deaths_blue": float(getattr(blue_obj, "deaths", 0.0)),
                "dmg_dealt_red": dmg_dealt_red,
                "dmg_dealt_blue": dmg_dealt_blue,
                "dmg_taken_red": dmg_taken_red,
                "dmg_taken_blue": dmg_taken_blue,
                "dmg_dealt_delta_red": _delta(dmg_dealt_red, self._headless_prev_dmg_dealt_red),
                "dmg_dealt_delta_blue": _delta(dmg_dealt_blue, self._headless_prev_dmg_dealt_blue),
                "dmg_taken_delta_red": _delta(dmg_taken_red, self._headless_prev_dmg_taken_red),
                "dmg_taken_delta_blue": _delta(dmg_taken_blue, self._headless_prev_dmg_taken_blue),
                "tick_metrics_window_ticks": int(self._headless_tick_metrics_window_ticks),
            }

            if self.headless_summary_include_gpu:
                row.update({
                    "gpu_raw": self._headless_gpu_parsed.get("gpu_raw", self._headless_gpu_raw or ""),
                    "gpu_util_pct": self._headless_gpu_parsed.get("gpu_util_pct", ""),
                    "gpu_mem_used_mb": self._headless_gpu_parsed.get("gpu_mem_used_mb", ""),
                    "gpu_mem_total_mb": self._headless_gpu_parsed.get("gpu_mem_total_mb", ""),
                    "gpu_mem_pct": self._headless_gpu_parsed.get("gpu_mem_pct", ""),
                    "gpu_power_w": self._headless_gpu_parsed.get("gpu_power_w", ""),
                })

            if self.headless_summary_include_tick_metrics:
                for k in self._headless_tick_metric_keys:
                    row[f"tm_{k}_sum"] = self._headless_tick_metrics_window.get(k, 0.0)

            if self.headless_summary_include_ppo:
                p = getattr(self, "_ppo_runtime", None)
                p_last = getattr(p, "last_train_summary", None) if p is not None else None
                row.update({
                    "ppo_update_seq": (p_last.get("update_seq", "") if isinstance(p_last, dict) else ""),
                    "ppo_step": (p_last.get("ppo_step", "") if isinstance(p_last, dict) else ""),
                    "ppo_trained_slots": (p_last.get("trained_slots", "") if isinstance(p_last, dict) else ""),
                    "ppo_samples": (p_last.get("samples", "") if isinstance(p_last, dict) else ""),
                    "ppo_optimizer_steps": (p_last.get("optimizer_steps", "") if isinstance(p_last, dict) else ""),
                    "ppo_epochs_run_total": (p_last.get("epochs_run_total", "") if isinstance(p_last, dict) else ""),
                    "ppo_loss_pi_mean": (p_last.get("loss_pi_mean", "") if isinstance(p_last, dict) else ""),
                    "ppo_loss_v_mean": (p_last.get("loss_v_mean", "") if isinstance(p_last, dict) else ""),
                    "ppo_loss_ent_mean": (p_last.get("loss_ent_mean", "") if isinstance(p_last, dict) else ""),
                    "ppo_entropy_mean": (p_last.get("entropy_mean", "") if isinstance(p_last, dict) else ""),
                    "ppo_approx_kl_max": (p_last.get("approx_kl_max", "") if isinstance(p_last, dict) else ""),
                    "ppo_lr_before_mean": (p_last.get("lr_before_mean", "") if isinstance(p_last, dict) else ""),
                    "ppo_lr_after_mean": (p_last.get("lr_after_mean", "") if isinstance(p_last, dict) else ""),
                })

            self._append_csv_rows(
                self.headless_summary_path,
                fieldnames=self._headless_summary_fieldnames(),
                rows=[row],
            )

            # Commit window baselines only after successful write.
            self._headless_prev_summary_wall = now
            self._headless_prev_summary_tick = int(tick)
            self._headless_prev_cp_red = cp_red
            self._headless_prev_cp_blue = cp_blue
            self._headless_prev_dmg_dealt_red = dmg_dealt_red
            self._headless_prev_dmg_dealt_blue = dmg_dealt_blue
            self._headless_prev_dmg_taken_red = dmg_taken_red
            self._headless_prev_dmg_taken_blue = dmg_taken_blue
            self._headless_tick_metrics_window.clear()
            self._headless_tick_metrics_window_ticks = 0

        except Exception as e:
            self._anomaly(f"headless summary write failed: {e}")

    def record_mutation_event(self, row: Dict[str, Any]) -> None:
        if (not self.enabled) or (not self.log_rare_mutations):
            return
        try:
            self._append_csv_rows(
                self.mutation_events_path,
                fieldnames=[
                    "tick",
                    "slot",
                    "agent_id",
                    "team_id",
                    "team_name",
                    "unit_id",
                    "parent_unit_id",
                    "child_generation",
                    "spawn_reason",
                    "spawn_hp",
                    "spawn_atk",
                    "spawn_vis",
                    "parent_agent_id",
                    "cloned",
                    "mutation_std",
                    "mutation_flag",
                    "rare_mutation",
                    "mutation_delta_hp",
                    "mutation_delta_atk",
                    "mutation_delta_vis",
                ],
                rows=[row],
            )
        except Exception as e:
            self._anomaly(f"mutation event write failed: {e}")

    def write_run_meta(self, meta: Dict[str, Any]) -> None:
        """
        Write run-level metadata to run_meta.json (atomic overwrite).

        Examples of meta
        ----------------
        - configuration summary
        - command line args
        - environment info (GPU name, torch version)
        - seeds
        - experiment name

        This data is crucial for reproducibility.

        Implementation details
        ----------------------
        - Schema version is inserted if missing.
        - Git commit is optionally added from env var GIT_COMMIT.
        """
        if not self.enabled or not self.do_write_run_meta:
            return

        out = dict(meta or {})
        out.setdefault("schema_version", self.schema_version)
        out.setdefault("git_commit", os.getenv("GIT_COMMIT", None))

        _atomic_write_text(self.run_meta_path, json.dumps(out, indent=2, sort_keys=True))

    def record_resume(self, tick: int, checkpoint_path: str) -> None:
        """
        Emit a 'resume' event when loading from a checkpoint.

        Why record resumes?
        -------------------
        In long runs, you may stop/restart/resume many times.
        Knowing exactly when and from where the run resumed is vital for analysis.
        """
        if not self.enabled:
            return
        self._emit_event({
            "tick": int(tick),
            "type": "resume",
            "checkpoint_path": str(checkpoint_path),
        })

    def bootstrap_from_registry(self, registry: Any, tick: int, note: str = "bootstrap") -> None:
        """
        Seed births for agents that already exist at startup.

        Use-case
        --------
        If the simulation begins with pre-spawned agents (e.g., initial population),
        telemetry needs a birth record for them to maintain consistency.

        Implementation detail
        ---------------------
        The registry is expected to have:
        - registry.agent_data tensor
        - optionally registry.agent_uids mapping slots->unique ids

        We find alive agents:
          alive_mask = data[:, COL_ALIVE] > 0.5
        and for each alive slot:
          - determine agent_id
          - record birth (allow_existing=True)
          - maybe write agent static attributes
        """
        if not self.enabled:
            return

        from engine.agent_registry import COL_ALIVE, COL_TEAM, COL_UNIT, COL_AGENT_ID
        data = registry.agent_data
        alive_mask = (data[:, COL_ALIVE] > 0.5)

        # nonzero(...) returns indices where alive_mask is True.
        alive_slots = alive_mask.nonzero(as_tuple=False).view(-1).tolist()

        for slot in alive_slots:
            # Two possible sources of agent id:
            # - registry.agent_uids (preferred stable mapping)
            # - agent_data column COL_AGENT_ID (fallback)
            if hasattr(registry, "agent_uids"):
                aid = int(registry.agent_uids[slot].item())
            else:
                aid = _to_int(data[slot, COL_AGENT_ID].item())

            if int(aid) < 0:
                self._anomaly(f"bootstrap: alive slot has invalid agent id slot={slot} aid={aid}")
                continue

            team = _to_int(data[slot, COL_TEAM].item())
            unit = _to_int(data[slot, COL_UNIT].item())
            generation = 0
            if hasattr(registry, "generations") and 0 <= int(slot) < len(registry.generations):
                generation = int(registry.generations[slot])
                if generation < 0:
                    self._anomaly(f"bootstrap: negative generation slot={slot} generation={generation}")
                    generation = 0

            self.record_birth(
                tick=tick,
                agent_id=aid,
                slot_id=int(slot),
                team=team,
                unit_type=unit,
                parent_id=None,
                notes=note,
                allow_existing=True,
                spawn_reason=str(note),
                generation=int(generation),
                mutation_flag=False,
                rare_mutation=False,
                mutation_delta_hp=0.0,
                mutation_delta_atk=0.0,
                mutation_delta_vis=0,
            )

            # Also write static attributes if enabled.
            self._maybe_write_agent_static(
                tick=int(tick),
                slot_id=int(slot),
                agent_id=int(aid),
                team_id=int(team),
                unit_type=int(unit),
                parent_id=None,
                spawn_reason=str(note),
            )

    def record_birth(
        self,
        tick: int,
        agent_id: int,
        slot_id: int,
        team: int,
        unit_type: int,
        parent_id: Optional[int],
        notes: str = "",
        allow_existing: bool = False,
        spawn_reason: str = "",
        parent_unit_type: Optional[int] = None,
        parent_team: Optional[int] = None,
        parent_generation: Optional[int] = None,
        generation: Optional[int] = None,
        mutation_flag: bool = False,
        rare_mutation: bool = False,
        mutation_delta_hp: Optional[float] = None,
        mutation_delta_atk: Optional[float] = None,
        mutation_delta_vis: Optional[int] = None,
    ) -> None:
        """
        Record an agent birth (creation).

        Parameters
        ----------
        tick:
            Current simulation tick when the agent appears.
        agent_id:
            Global unique id of the agent.
        slot_id:
            The registry slot index where this agent currently resides.
        team:
            Team identifier (commonly 2=red, 3=blue).
        unit_type:
            Unit type identifier (project-specific).
        parent_id:
            Parent agent id if this agent was spawned from another agent (lineage).
        notes:
            Free-form string for provenance (e.g., "respawn", "bootstrap").
        allow_existing:
            If True, update existing record rather than treating as a duplicate.
            Used during bootstrap/resume scenarios.

        Consistency rules
        -----------------
        - If agent_id already exists in `_life` and allow_existing is False -> anomaly.
        - When allow_existing is True:
            - preserve existing born_tick if already present
            - otherwise set born_tick to current tick
        """
        if not self.enabled:
            return

        if (agent_id in self._life) and not allow_existing:
            self._anomaly(f"birth: duplicate agent_id={agent_id}")
            return

        rec = self._life.get(agent_id, {})
        rec.update({
            "slot_id": int(slot_id),
            "team": int(team),
            "unit_type": int(unit_type),

            # born_tick handling:
            # - bootstrap should not overwrite old born_tick if present
            "born_tick": (rec.get("born_tick", int(tick)) if allow_existing else int(tick)),

            # keep any previous death_tick if rehydrated
            "death_tick": rec.get("death_tick", None),

            "parent_id": (int(parent_id) if parent_id is not None else None),
            "spawn_reason": str(spawn_reason or rec.get("spawn_reason", notes or "")),
            "parent_unit_type": (
                int(parent_unit_type) if parent_unit_type is not None else rec.get("parent_unit_type", None)
            ),
            "parent_team": (
                int(parent_team) if parent_team is not None else rec.get("parent_team", None)
            ),
            "parent_generation": (
                int(parent_generation) if parent_generation is not None else rec.get("parent_generation", None)
            ),
            "generation": (
                int(generation) if generation is not None else int(rec.get("generation", 0) or 0)
            ),
            "mutation_flag": bool(mutation_flag or rec.get("mutation_flag", False)),
            "rare_mutation": bool(rare_mutation or rec.get("rare_mutation", False)),
            "mutation_delta_hp": (
                float(mutation_delta_hp) if mutation_delta_hp is not None else float(rec.get("mutation_delta_hp", 0.0) or 0.0)
            ),
            "mutation_delta_atk": (
                float(mutation_delta_atk) if mutation_delta_atk is not None else float(rec.get("mutation_delta_atk", 0.0) or 0.0)
            ),
            "mutation_delta_vis": (
                int(mutation_delta_vis) if mutation_delta_vis is not None else int(rec.get("mutation_delta_vis", 0) or 0)
            ),

            # totals should persist if record exists
            "kills_total": int(rec.get("kills_total", 0)),
            "damage_dealt_total": float(rec.get("damage_dealt_total", 0.0)),
            "damage_taken_total": float(rec.get("damage_taken_total", 0.0)),

            # PPO reward-category totals
            "reward_total": float(rec.get("reward_total", 0.0)),
            "reward_individual_total": float(rec.get("reward_individual_total", 0.0)),
            "reward_team_total": float(rec.get("reward_team_total", 0.0)),
            "reward_hp": float(rec.get("reward_hp", 0.0)),
            "reward_kill_individual": float(rec.get("reward_kill_individual", 0.0)),
            "reward_damage_dealt_individual": float(rec.get("reward_damage_dealt_individual", 0.0)),
            "reward_damage_taken_penalty": float(rec.get("reward_damage_taken_penalty", 0.0)),
            "reward_contested_cp_individual": float(rec.get("reward_contested_cp_individual", 0.0)),
            "reward_team_kill": float(rec.get("reward_team_kill", 0.0)),
            "reward_team_death": float(rec.get("reward_team_death", 0.0)),
            "reward_team_cp": float(rec.get("reward_team_cp", 0.0)),
            "reward_healing_recovered": float(rec.get("reward_healing_recovered", 0.0)),

            # structured death forensics
            "death_cause": rec.get("death_cause", None),
            "killer_id": rec.get("killer_id", None),
            "killer_slot": rec.get("killer_slot", None),
            "killer_team": rec.get("killer_team", None),

            # notes: if notes provided, store it; else keep old
            "notes": str(notes or rec.get("notes", "")),
        })
        self._life[agent_id] = rec

        # Reset/seed per-slot movement counters for this slot.
        # - For new agents: reset to 0 (fresh life).
        # - For allow_existing bootstrap/resume: seed from existing AgentLife values when present.
        if self.log_moves:
            try:
                reg = getattr(self, "_registry", None)
                if reg is not None and hasattr(reg, "agent_data"):
                    device = getattr(reg.agent_data, "device", "cpu")
                    n_slots = int(reg.agent_data.shape[0])
                    self._ensure_move_slot_tensors(n_slots=n_slots, device=device)
                    sid = int(slot_id)
                    if 0 <= sid < int(self._move_slot_attempted.numel()):
                        if allow_existing:
                            self._move_slot_attempted[sid] = int(rec.get("moves_attempted", 0) or 0)
                            self._move_slot_success[sid] = int(rec.get("moves_success", 0) or 0)
                            self._move_slot_blocked_wall[sid] = int(rec.get("moves_blocked_wall", 0) or 0)
                            self._move_slot_blocked_occupied[sid] = int(rec.get("moves_blocked_occupied", 0) or 0)
                            self._move_slot_conflict_lost[sid] = int(rec.get("moves_conflict_lost", 0) or 0)
                            self._move_slot_conflict_tie[sid] = int(rec.get("moves_conflict_tie", 0) or 0)
                            self._move_slot_cells_l1[sid] = int(rec.get("cells_walked_total_l1", 0) or 0)
                            ls = rec.get("last_seen_tick", None)
                            self._move_slot_last_seen[sid] = int(ls) if ls is not None else -1
                        else:
                            self._move_slot_attempted[sid] = 0
                            self._move_slot_success[sid] = 0
                            self._move_slot_blocked_wall[sid] = 0
                            self._move_slot_blocked_occupied[sid] = 0
                            self._move_slot_conflict_lost[sid] = 0
                            self._move_slot_conflict_tie[sid] = 0
                            self._move_slot_cells_l1[sid] = 0
                            self._move_slot_last_seen[sid] = -1
            except Exception:
                pass

        # Reset/seed per-slot reward counters for this slot.
        try:
            reg = getattr(self, "_registry", None)
            if reg is not None and hasattr(reg, "agent_data"):
                device = getattr(reg.agent_data, "device", "cpu")
                n_slots = int(reg.agent_data.shape[0])
                self._ensure_reward_slot_tensors(n_slots=n_slots, device=device)
                sid = int(slot_id)

                cur = getattr(self, "_reward_slot_total", None)
                if cur is not None and 0 <= sid < int(cur.numel()):
                    if allow_existing:
                        self._reward_slot_total[sid] = float(rec.get("reward_total", 0.0) or 0.0)
                        self._reward_slot_individual_total[sid] = float(rec.get("reward_individual_total", 0.0) or 0.0)
                        self._reward_slot_team_total[sid] = float(rec.get("reward_team_total", 0.0) or 0.0)
                        self._reward_slot_hp[sid] = float(rec.get("reward_hp", 0.0) or 0.0)
                        self._reward_slot_kill_individual[sid] = float(rec.get("reward_kill_individual", 0.0) or 0.0)
                        self._reward_slot_damage_dealt_individual[sid] = float(rec.get("reward_damage_dealt_individual", 0.0) or 0.0)
                        self._reward_slot_damage_taken_penalty[sid] = float(rec.get("reward_damage_taken_penalty", 0.0) or 0.0)
                        self._reward_slot_contested_cp_individual[sid] = float(rec.get("reward_contested_cp_individual", 0.0) or 0.0)
                        self._reward_slot_team_kill[sid] = float(rec.get("reward_team_kill", 0.0) or 0.0)
                        self._reward_slot_team_death[sid] = float(rec.get("reward_team_death", 0.0) or 0.0)
                        self._reward_slot_team_cp[sid] = float(rec.get("reward_team_cp", 0.0) or 0.0)
                        self._reward_slot_healing_recovered[sid] = float(rec.get("reward_healing_recovered", 0.0) or 0.0)
                    else:
                        self._reward_slot_total[sid] = 0.0
                        self._reward_slot_individual_total[sid] = 0.0
                        self._reward_slot_team_total[sid] = 0.0
                        self._reward_slot_hp[sid] = 0.0
                        self._reward_slot_kill_individual[sid] = 0.0
                        self._reward_slot_damage_dealt_individual[sid] = 0.0
                        self._reward_slot_damage_taken_penalty[sid] = 0.0
                        self._reward_slot_contested_cp_individual[sid] = 0.0
                        self._reward_slot_team_kill[sid] = 0.0
                        self._reward_slot_team_death[sid] = 0.0
                        self._reward_slot_team_cp[sid] = 0.0
                        self._reward_slot_healing_recovered[sid] = 0.0
        except Exception:
            pass

        # If lineage info exists, record parent->child edge and update offspring counts.
        if parent_id is not None:
            if int(parent_id) == int(agent_id):
                self._anomaly(f"birth: self-parent edge for agent_id={agent_id}")
                return
            if int(parent_id) not in self._life:
                self._anomaly(f"birth: missing parent life record for parent_id={parent_id}, child_id={agent_id}")
                return
            self._offspring_count[int(parent_id)] = int(self._offspring_count.get(int(parent_id), 0)) + 1
            self._lineage_edge_count[int(parent_id)] = int(self._lineage_edge_count.get(int(parent_id), 0)) + 1
            self._append_csv_rows(
                self.lineage_edges_path,
                fieldnames=[
                    "tick",
                    "parent_id",
                    "child_id",
                    "parent_unit_type",
                    "child_unit_type",
                    "parent_team",
                    "spawn_reason",
                    "parent_generation",
                    "child_generation",
                    "mutation_flag",
                    "rare_mutation",
                    "mutation_delta_hp",
                    "mutation_delta_atk",
                    "mutation_delta_vis",
                ],
                rows=[{
                    "tick": int(tick),
                    "parent_id": int(parent_id),
                    "child_id": int(agent_id),
                    "parent_unit_type": (int(parent_unit_type) if parent_unit_type is not None else ""),
                    "child_unit_type": int(unit_type),
                    "parent_team": (int(parent_team) if parent_team is not None else ""),
                    "spawn_reason": str(spawn_reason or notes),
                    "parent_generation": (int(parent_generation) if parent_generation is not None else ""),
                    "child_generation": (int(generation) if generation is not None else ""),
                    "mutation_flag": int(bool(mutation_flag)),
                    "rare_mutation": int(bool(rare_mutation)),
                    "mutation_delta_hp": float(mutation_delta_hp or 0.0),
                    "mutation_delta_atk": float(mutation_delta_atk or 0.0),
                    "mutation_delta_vis": int(mutation_delta_vis or 0),
                }],
                strict_schema=True,
            )

        # Optionally emit a birth event.
        if self.log_births:
            self._emit_event({
                "schema_version": self._event_schema_version_int(),
                "tick": int(tick),
                "type": "birth",
                "agent_id": int(agent_id),
                "slot_id": int(slot_id),
                "team": int(team),
                "unit_type": int(unit_type),
                "parent_id": (int(parent_id) if parent_id is not None else None),
                "spawn_reason": str(spawn_reason or notes),
                "parent_unit_type": (int(parent_unit_type) if parent_unit_type is not None else None),
                "parent_team": (int(parent_team) if parent_team is not None else None),
                "parent_generation": (int(parent_generation) if parent_generation is not None else None),
                "generation": (int(generation) if generation is not None else None),
                "mutation_flag": bool(mutation_flag),
                "rare_mutation": bool(rare_mutation),
                "mutation_delta_hp": float(mutation_delta_hp or 0.0),
                "mutation_delta_atk": float(mutation_delta_atk or 0.0),
                "mutation_delta_vis": int(mutation_delta_vis or 0),
                "notes": str(notes),
            })

    def record_ppo_reward_components_by_slot(
        self,
        tick: int,
        slot_ids,
        reward_total,
        reward_hp,
        reward_kill_individual,
        reward_damage_dealt_individual,
        reward_damage_taken_penalty,
        reward_contested_cp_individual,
        reward_team_kill,
        reward_team_death,
        reward_team_cp,
        reward_healing_recovered,
    ) -> None:
        """
        Accumulate PPO reward categories by live registry slot.

        Important:
        - This method stays on-device (CPU/CUDA) and uses index_add_.
        - No per-agent Python loop in the hot path.
        - Persistent agent_id rows are materialized later via slot freeze/snapshot.
        """
        if not self.enabled:
            return

        reg = getattr(self, "_registry", None)
        if reg is None or not hasattr(reg, "agent_data"):
            return

        try:
            import torch
        except Exception:
            return

        device = reg.agent_data.device
        n_slots = int(reg.agent_data.shape[0])
        self._ensure_reward_slot_tensors(n_slots=n_slots, device=device)

        s = slot_ids
        if s is None:
            return

        if not isinstance(s, torch.Tensor):
            s = torch.as_tensor(s, device=device, dtype=torch.long)
        else:
            s = s.to(device=device, dtype=torch.long)

        s = s.view(-1)
        if s.numel() == 0:
            return

        def _vec(x):
            if isinstance(x, torch.Tensor):
                y = x.to(device=device, dtype=torch.float32)
            else:
                y = torch.as_tensor(x, device=device, dtype=torch.float32)
            return y.view(-1)

        rt = _vec(reward_total)
        rh = _vec(reward_hp)
        rk = _vec(reward_kill_individual)
        rdd = _vec(reward_damage_dealt_individual)
        rdt = _vec(reward_damage_taken_penalty)
        rcp = _vec(reward_contested_cp_individual)
        rtk = _vec(reward_team_kill)
        rtd = _vec(reward_team_death)
        rtcp = _vec(reward_team_cp)
        rhr = _vec(reward_healing_recovered)

        n = int(s.numel())
        tensors = (rt, rh, rk, rdd, rdt, rcp, rtk, rtd, rtcp, rhr)
        if any(int(t.numel()) != n for t in tensors):
            self._anomaly("record_ppo_reward_components_by_slot: length mismatch")
            return

        rind = rk + rdd + rdt + rcp + rhr
        rteam = rtk + rtd + rtcp

        self._reward_slot_total.index_add_(0, s, rt)
        self._reward_slot_individual_total.index_add_(0, s, rind)
        self._reward_slot_team_total.index_add_(0, s, rteam)
        self._reward_slot_hp.index_add_(0, s, rh)
        self._reward_slot_kill_individual.index_add_(0, s, rk)
        self._reward_slot_damage_dealt_individual.index_add_(0, s, rdd)
        self._reward_slot_damage_taken_penalty.index_add_(0, s, rdt)
        self._reward_slot_contested_cp_individual.index_add_(0, s, rcp)
        self._reward_slot_team_kill.index_add_(0, s, rtk)
        self._reward_slot_team_death.index_add_(0, s, rtd)
        self._reward_slot_team_cp.index_add_(0, s, rtcp)
        self._reward_slot_healing_recovered.index_add_(0, s, rhr)

    def ingest_spawn_meta(self, meta: List[Dict[str, Any]]) -> None:
        """
        Ingest spawn metadata from RespawnController.

        Expected meta schema
        --------------------
        The RespawnController likely emits dicts with keys like:
        - tick
        - slot
        - agent_id
        - team_id
        - unit_id
        - parent_agent_id (optional)

        For each spawn record:
        - record_birth(..., notes="respawn")
        - optionally write agent_static row
        """
        if not self.enabled or not meta:
            return

        for m in meta:
            required = ("tick", "slot", "agent_id", "team_id", "unit_id", "spawn_reason")
            missing = [k for k in required if k not in m]
            if missing:
                self._anomaly(f"ingest_spawn_meta: missing required keys {missing}")
                continue

            tick = _to_int(m.get("tick"))
            slot = _to_int(m.get("slot"))
            aid = _to_int(m.get("agent_id"))
            team = _to_int(m.get("team_id"))
            unit = _to_int(m.get("unit_id"))
            parent = m.get("parent_agent_id", None)
            parent_id = (_to_int(parent) if parent is not None else None)

            # Additive rare mutation CSV sidecar (observational only).
            if bool(m.get("rare_mutation", False)):
                team_name = "red" if int(team) == 2 else ("blue" if int(team) == 3 else str(team))
                self.record_mutation_event({
                    "tick": int(tick),
                    "slot": int(slot),
                    "agent_id": int(aid),
                    "team_id": int(team),
                    "team_name": str(team_name),
                    "unit_id": int(unit),
                    "parent_unit_id": (_to_int(m.get("parent_unit_id")) if m.get("parent_unit_id", None) is not None else ""),
                    "child_generation": _to_int(m.get("child_generation", 0)),
                    "spawn_reason": str(m.get("spawn_reason", "respawn")),
                    "spawn_hp": float(m.get("spawn_hp", "")) if m.get("spawn_hp", None) is not None else "",
                    "spawn_atk": float(m.get("spawn_atk", "")) if m.get("spawn_atk", None) is not None else "",
                    "spawn_vis": int(m.get("spawn_vis", 0)) if m.get("spawn_vis", None) is not None else "",
                    "parent_agent_id": (int(parent_id) if parent_id is not None else ""),
                    "cloned": bool(m.get("cloned", False)),
                    "mutation_std": float(m.get("mutation_std", 0.0)),
                    "mutation_flag": bool(m.get("mutation_flag", False)),
                    "rare_mutation": bool(m.get("rare_mutation", False)),
                    "mutation_delta_hp": float(m.get("mutation_delta_hp", 0.0)),
                    "mutation_delta_atk": float(m.get("mutation_delta_atk", 0.0)),
                    "mutation_delta_vis": int(m.get("mutation_delta_vis", 0)),
                })

            self.record_birth(
                tick=tick,
                agent_id=aid,
                slot_id=slot,
                team=team,
                unit_type=unit,
                parent_id=parent_id,
                notes="respawn",
                allow_existing=False,
                spawn_reason=str(m.get("spawn_reason", "respawn")),
                parent_unit_type=(
                    _to_int(m.get("parent_unit_id")) if m.get("parent_unit_id", None) is not None else None
                ),
                parent_team=(
                    _to_int(m.get("parent_team_id")) if m.get("parent_team_id", None) is not None else None
                ),
                parent_generation=(
                    _to_int(m.get("parent_generation")) if m.get("parent_generation", None) is not None else None
                ),
                generation=(
                    _to_int(m.get("child_generation")) if m.get("child_generation", None) is not None else None
                ),
                mutation_flag=bool(m.get("mutation_flag", False)),
                rare_mutation=bool(m.get("rare_mutation", False)),
                mutation_delta_hp=float(m.get("mutation_delta_hp", 0.0)),
                mutation_delta_atk=float(m.get("mutation_delta_atk", 0.0)),
                mutation_delta_vis=int(m.get("mutation_delta_vis", 0)),
            )

            self._maybe_write_agent_static(
                tick=int(tick),
                slot_id=int(slot),
                agent_id=int(aid),
                team_id=int(team),
                unit_type=int(unit),
                parent_id=parent_id,
                spawn_reason=str(m.get("spawn_reason", "respawn")),
            )

    def _maybe_write_agent_static(
        self,
        tick: int,
        slot_id: int,
        agent_id: int,
        team_id: int,
        unit_type: int,
        parent_id: Optional[int],
        spawn_reason: str,
    ) -> None:
        """
        Write a row to agent_static.csv for an agent (deduplicated).

        What "static" means here
        ------------------------
        Values that usually do not change over an agent's lifetime:
        - max_hp
        - base_atk
        - vision_range
        - brain type and parameter count

        Preconditions
        -------------
        - telemetry enabled
        - write_agent_static enabled
        - registry attached
        - agent_id not already written

        How brain info is retrieved
        ---------------------------
        registry.brains is expected to be a list aligned by slot.
        If present:
        - brain_type = class name
        - param_count = sum(p.numel()) across parameters

        Why param_count matters
        -----------------------
        In ML experiments, model size strongly influences compute cost and learning dynamics.
        Logging it enables later correlation analysis.
        """
        if (not self.enabled) or (not self.write_agent_static) or (self._registry is None):
            return

        if agent_id in self._static_written:
            return

        try:
            from engine.agent_registry import COL_HP_MAX, COL_VISION, COL_ATK
            data = self._registry.agent_data

            max_hp = float(data[slot_id, COL_HP_MAX].item())
            vision = float(data[slot_id, COL_VISION].item())
            atk = float(data[slot_id, COL_ATK].item())

            brain_type = None
            param_count = None
            brains = getattr(self._registry, "brains", None)

            if isinstance(brains, list) and 0 <= slot_id < len(brains) and brains[slot_id] is not None:
                b = brains[slot_id]
                brain_type = b.__class__.__name__
                try:
                    param_count = int(sum(int(p.numel()) for p in b.parameters()))
                except Exception:
                    param_count = None

            self._append_csv_rows(
                self.agent_static_path,
                fieldnames=[
                    "agent_id", "team_id", "unit_type", "brain_type", "param_count",
                    "max_hp", "base_atk", "vision_range",
                    "spawn_tick", "parent_id", "spawn_reason",
                ],
                rows=[{
                    "agent_id": int(agent_id),
                    "team_id": int(team_id),
                    "unit_type": int(unit_type),
                    "brain_type": brain_type,
                    "param_count": param_count,
                    "max_hp": max_hp,
                    "base_atk": atk,
                    "vision_range": vision,
                    "spawn_tick": int(tick),
                    "parent_id": (int(parent_id) if parent_id is not None else None),
                    "spawn_reason": str(spawn_reason),
                }],
            )

            self._static_written.add(int(agent_id))

        except Exception as e:
            self._anomaly(f"agent_static write failed: {e}")

    # Damage, kills, deaths (core event types)

    def record_damage_victim_sum(
        self,
        tick: int,
        victim_ids: List[int],
        victim_team: List[int],
        victim_unit: List[int],
        damage: List[float],
        hp_before: Optional[List[float]] = None,
        hp_after: Optional[List[float]] = None,
    ) -> None:
        """
        Record damage in 'victim_sum' mode.

        Definition
        ----------
        victim_sum means:
          For each victim, we record a single damage number = total damage received this tick.
          (Even if multiple attackers hit them.)

        Why this mode exists
        --------------------
        It is a compression strategy.
        Per-hit logging can explode event volume. victim_sum scales with number of victims,
        not number of hits.

        Effects
        -------
        - Updates victim's `damage_taken_total`.
        - Emits a 'damage' event if log_damage is enabled.
        - Optionally logs hp_before/hp_after if both provided.

        Notes on hp logging
        -------------------
        - hp_before/after are optional because not every engine stage has them available.
        - If only one is provided, this code logs neither (requires both).
        """
        if not self.enabled:
            return

        for i, vid in enumerate(victim_ids):
            self._require_birth(vid, "damage(victim_sum)")
            rec = self._life[vid]
            rec["damage_taken_total"] = float(rec.get("damage_taken_total", 0.0)) + float(damage[i])

            if self.log_damage:
                ev = {
                    "schema_version": self._event_schema_version_int(),
                    "tick": int(tick),
                    "type": "damage",
                    "mode": "victim_sum",
                    "victim_id": int(vid),
                    "victim_team": int(victim_team[i]) if i < len(victim_team) else None,
                    "victim_unit": int(victim_unit[i]) if i < len(victim_unit) else None,
                    "damage": float(damage[i]),
                }
                if hp_before is not None and hp_after is not None:
                    ev["hp_before"] = float(hp_before[i])
                    ev["hp_after"] = float(hp_after[i])

                self._emit_event(ev)

    def record_damage_attacker_sum(
        self,
        tick: int,
        attacker_ids: List[int],
        damage_dealt: List[float],
    ) -> None:
        """
        Record damage in 'attacker_sum' mode.

        Definition
        ----------
        attacker_sum means:
          For each attacker, we record a single number = total damage dealt this tick.

        Why it exists
        -------------
        This updates per-agent totals for analysis (e.g., "top damage dealers") while
        keeping event logs small.

        Important: no event is emitted here.
        -----------------------------------
        This is deliberate:
        - If you already log victim_sum or per_hit events, emitting attacker_sum events
          doubles event volume without adding much information.

        Effects
        -------
        - Updates attacker `damage_dealt_total`.
        """
        if not self.enabled:
            return

        for i, aid in enumerate(attacker_ids):
            self._require_birth(aid, "damage(attacker_sum)")
            rec = self._life[aid]
            rec["damage_dealt_total"] = float(rec.get("damage_dealt_total", 0.0)) + float(damage_dealt[i])

        # No event emission here; event volume is controlled elsewhere.

    def record_damage_per_hit(
        self,
        tick: int,
        attacker_ids: List[int],
        victim_ids: List[int],
        damage: List[float],
    ) -> None:
        """
        Record damage in 'per_hit' mode.

        Definition
        ----------
        per_hit mode logs *every* hit as its own event:
          (attacker_id, victim_id, damage)

        Pros
        ----
        - Maximum fidelity: can reconstruct micro-dynamics.
        - Enables causal graphs (who hit whom, how often).

        Cons
        ----
        - High volume: event count scales with number of hits.

        Important:
        ----------
        This method does NOT update totals. Totals are expected to be updated by
        victim_sum/attacker_sum methods in the engine, to avoid double-counting.
        """
        if not self.enabled or not self.log_damage:
            return

        for i in range(len(damage)):
            aid = int(attacker_ids[i])
            vid = int(victim_ids[i])
            self._require_birth(aid, "damage(per_hit attacker)")
            self._require_birth(vid, "damage(per_hit victim)")

            self._emit_event({
                "schema_version": self._event_schema_version_int(),
                "tick": int(tick),
                "type": "damage",
                "mode": "per_hit",
                "attacker_id": aid,
                "victim_id": vid,
                "damage": float(damage[i]),
            })

    def record_kills(self, tick: int, killer_ids: List[int], victim_ids: List[int]) -> None:
        """
        Record kills.

        Behavior
        --------
        - For each (killer_id, victim_id) pair:
            - ensure both have birth records
            - increment killer's kills_total
            - emit 'kill' event if enabled

        Note on correctness
        -------------------
        The engine must ensure killer_ids and victim_ids are aligned lists.
        This method uses `min(len(killer_ids), len(victim_ids))` to be defensive.
        """
        if not self.enabled:
            return

        n = min(len(killer_ids), len(victim_ids))
        for i in range(n):
            kid = int(killer_ids[i])
            vid = int(victim_ids[i])
            self._require_birth(kid, "kill(killer)")
            self._require_birth(vid, "kill(victim)")

            rec = self._life[kid]
            rec["kills_total"] = int(rec.get("kills_total", 0)) + 1

            if self.log_kills:
                self._emit_event({
                    "schema_version": self._event_schema_version_int(),
                    "tick": int(tick),
                    "type": "kill",
                    "killer_id": kid,
                    "victim_id": vid,
                })

    def record_deaths(
        self,
        tick: int,
        dead_ids: List[int],
        dead_team: List[int],
        dead_unit: List[int],
        dead_slots: List[int],
        notes: str = "",
        death_causes: Optional[List[Optional[str]]] = None,
        killer_ids: Optional[List[Optional[int]]] = None,
        killer_slots: Optional[List[Optional[int]]] = None,
        killer_teams: Optional[List[Optional[int]]] = None,
    ) -> None:
        """
        Record deaths with structured cause/killer metadata.

        Guardrail:
        - We keep legacy root dead_agents_log.csv untouched.
        - Rich death rows go to telemetry/dead_agents_log_detailed.csv.
        """
        if not self.enabled:
            return

        detailed_rows: List[Dict[str, Any]] = []

        for i, did in enumerate(dead_ids):
            self._require_birth(did, "death")
            rec = self._life[did]

            if rec.get("death_tick", None) is not None:
                self._anomaly(f"death: duplicate death for agent_id={did}")
                continue

            rec["death_tick"] = int(tick)

            sid = int(dead_slots[i]) if i < len(dead_slots) else None

            # Freeze final movement totals into the life record.
            if self.log_moves:
                try:
                    if sid is not None and getattr(self, "_move_slot_attempted", None) is not None:
                        if 0 <= sid < int(self._move_slot_attempted.numel()):
                            rec["moves_attempted"] = int(self._move_slot_attempted[sid].item())
                            rec["moves_success"] = int(self._move_slot_success[sid].item())
                            rec["moves_blocked_wall"] = int(self._move_slot_blocked_wall[sid].item())
                            rec["moves_blocked_occupied"] = int(self._move_slot_blocked_occupied[sid].item())
                            rec["moves_conflict_lost"] = int(self._move_slot_conflict_lost[sid].item())
                            rec["moves_conflict_tie"] = int(self._move_slot_conflict_tie[sid].item())
                            rec["cells_walked_total_l1"] = int(self._move_slot_cells_l1[sid].item())
                            lst = int(self._move_slot_last_seen[sid].item())
                            if lst >= 0:
                                rec["last_seen_tick"] = lst
                except Exception:
                    pass

            # Freeze final reward-category totals into the life record.
            try:
                if sid is not None and getattr(self, "_reward_slot_total", None) is not None:
                    if 0 <= sid < int(self._reward_slot_total.numel()):
                        rec["reward_total"] = float(self._reward_slot_total[sid].item())
                        rec["reward_individual_total"] = float(self._reward_slot_individual_total[sid].item())
                        rec["reward_team_total"] = float(self._reward_slot_team_total[sid].item())
                        rec["reward_hp"] = float(self._reward_slot_hp[sid].item())
                        rec["reward_kill_individual"] = float(self._reward_slot_kill_individual[sid].item())
                        rec["reward_damage_dealt_individual"] = float(self._reward_slot_damage_dealt_individual[sid].item())
                        rec["reward_damage_taken_penalty"] = float(self._reward_slot_damage_taken_penalty[sid].item())
                        rec["reward_contested_cp_individual"] = float(self._reward_slot_contested_cp_individual[sid].item())
                        rec["reward_team_kill"] = float(self._reward_slot_team_kill[sid].item())
                        rec["reward_team_death"] = float(self._reward_slot_team_death[sid].item())
                        rec["reward_team_cp"] = float(self._reward_slot_team_cp[sid].item())
                        rec["reward_healing_recovered"] = float(self._reward_slot_healing_recovered[sid].item())
            except Exception:
                pass

            cause_val = None
            if death_causes is not None and i < len(death_causes):
                v = death_causes[i]
                if v is not None and str(v).strip() != "":
                    cause_val = str(v).strip()
                    rec["death_cause"] = cause_val

            killer_id_val = None
            if killer_ids is not None and i < len(killer_ids):
                v = killer_ids[i]
                if v is not None:
                    killer_id_val = int(v)
                    rec["killer_id"] = killer_id_val

            killer_slot_val = None
            if killer_slots is not None and i < len(killer_slots):
                v = killer_slots[i]
                if v is not None:
                    killer_slot_val = int(v)
                    rec["killer_slot"] = killer_slot_val

            killer_team_val = None
            if killer_teams is not None and i < len(killer_teams):
                v = killer_teams[i]
                if v is not None:
                    killer_team_val = int(v)
                    rec["killer_team"] = killer_team_val

            detailed_rows.append({
                "tick": int(tick),
                "agent_id": int(did),
                "slot_id": sid if sid is not None else "",
                "team": int(dead_team[i]) if i < len(dead_team) else "",
                "unit_type": int(dead_unit[i]) if i < len(dead_unit) else "",
                "death_cause": cause_val or "",
                "killer_id": killer_id_val if killer_id_val is not None else "",
                "killer_slot": killer_slot_val if killer_slot_val is not None else "",
                "killer_team": killer_team_val if killer_team_val is not None else "",
                "notes": str(notes),
            })

            if self.log_deaths:
                self._emit_event({
                    "schema_version": self._event_schema_version_int(),
                    "tick": int(tick),
                    "type": "death",
                    "agent_id": int(did),
                    "slot_id": sid,
                    "team": int(dead_team[i]) if i < len(dead_team) else None,
                    "unit_type": int(dead_unit[i]) if i < len(dead_unit) else None,
                    "death_cause": cause_val,
                    "killer_id": killer_id_val,
                    "killer_slot": killer_slot_val,
                    "killer_team": killer_team_val,
                    "notes": str(notes),
                })

        if detailed_rows:
            self._append_csv_rows(
                self.dead_agents_detailed_path,
                fieldnames=[
                    "tick",
                    "agent_id",
                    "slot_id",
                    "team",
                    "unit_type",
                    "death_cause",
                    "killer_id",
                    "killer_slot",
                    "killer_team",
                    "notes",
                ],
                rows=detailed_rows,
                strict_schema=True,
            )

    # Tick lifecycle hooks

    def on_tick_end(self, tick: int) -> None:
        """
        Called once at the end of each simulation tick.

        Responsibilities
        ----------------
        1) Store last tick seen (for close()).
        2) Periodically validate telemetry state (detect anomalies early).
        3) Periodically flush:
           - agent_life snapshot
           - event chunk buffer
        4) Optionally write tick summary row.

        Performance considerations
        --------------------------
        Validation + flushing are expensive relative to pure simulation logic.
        That's why they run on schedules: every `validate_every` ticks, every `flush_every` ticks.
        """
        if not self.enabled:
            return

        self._last_tick_seen = int(tick)

        # Drain PPO training rows every tick so repeated resumes append into one diagnostics CSV.
        self._drain_and_flush_ppo_rich_telemetry(observed_tick=int(tick), force=False)

        if self.validate_every > 0 and (int(tick) % int(self.validate_every) == 0):
            self.validate()

        if self.flush_every > 0 and (int(tick) % int(self.flush_every) == 0):
            self._flush_agent_life_snapshot()
            self._flush_event_chunk()
            self._flush_move_summary()
            self._flush_counters()

        if self.tick_summary_every > 0 and (tick % int(self.tick_summary_every)) == 0:
            self._write_tick_summary(tick=int(tick))

    def _write_tick_summary(self, tick: int) -> None:
        """
        Compute and append a summary row to tick_summary.csv.

        Preconditions
        -------------
        - telemetry enabled
        - registry and stats attached (attach_context must be called)

        What is computed
        ----------------
        - Alive counts for red/blue
        - Mean HP for red/blue
        - Some accumulated stats from `self._stats` (kills, deaths, dmg dealt/taken)

        Why this is useful
        ------------------
        This gives you a low-frequency time series that is cheap to plot:
        - population dynamics
        - survivability trends
        - macro balance between teams
        """
        if (not self.enabled) or (self._registry is None) or (self._stats is None):
            return
        if self._last_tick_summary_tick == int(tick):
            return

        try:
            import torch  # local import; torch is core dep already used by project
            from engine.agent_registry import COL_ALIVE, COL_TEAM, COL_HP

            data = self._registry.agent_data

            alive_mask = (data[:, COL_ALIVE] > 0.5)
            red_mask = alive_mask & (data[:, COL_TEAM] == 2.0)
            blue_mask = alive_mask & (data[:, COL_TEAM] == 3.0)

            red_n = int(red_mask.sum().item())
            blue_n = int(blue_mask.sum().item())

            mean_red = 0.0
            if red_n > 0:
                mean_red = float(data[red_mask, COL_HP].mean().item())

            mean_blue = 0.0
            if blue_n > 0:
                mean_blue = float(data[blue_mask, COL_HP].mean().item())

            s = self._stats
            row = {
                "tick": int(tick),
                "elapsed_s": float(getattr(s, "elapsed_seconds", 0.0)),
                "red_alive": red_n,
                "blue_alive": blue_n,
                "mean_hp_red": mean_red,
                "mean_hp_blue": mean_blue,
                "red_kills": float(getattr(getattr(s, "red", None), "kills", 0.0)),
                "blue_kills": float(getattr(getattr(s, "blue", None), "kills", 0.0)),
                "red_deaths": float(getattr(getattr(s, "red", None), "deaths", 0.0)),
                "blue_deaths": float(getattr(getattr(s, "blue", None), "deaths", 0.0)),
                "red_dmg_dealt": float(getattr(getattr(s, "red", None), "dmg_dealt", 0.0)),
                "blue_dmg_dealt": float(getattr(getattr(s, "blue", None), "dmg_dealt", 0.0)),
                "red_dmg_taken": float(getattr(getattr(s, "red", None), "dmg_taken", 0.0)),
                "blue_dmg_taken": float(getattr(getattr(s, "blue", None), "dmg_taken", 0.0)),
            }

            # Write a single-row append with fieldnames derived from row keys.
            # This assumes the CSV header matches these keys (created in __init__).
            self._append_csv_rows(self.tick_summary_path, fieldnames=list(row.keys()), rows=[row])
            self._last_tick_summary_tick = int(tick)

        except Exception as e:
            self._anomaly(f"tick_summary write failed: {e}")

    # Shutdown

    def close(self) -> None:
        """
        Clean shutdown: flush buffers and write final snapshots.

        IMPORTANT RELIABILITY CHANGE
        ----------------------------
        Each flush step is best-effort and isolated: one failure (e.g. agent_life snapshot)
        must NOT prevent other flushes (e.g. move_summary, counters, events).
        """
        if not self.enabled:
            return

        errors: List[str] = []

        def _safe(step: str, fn) -> None:
            try:
                fn()
            except Exception as e:
                errors.append(f"{step}: {type(e).__name__}: {e}")

        _safe("events_chunk", self._flush_event_chunk)
        _safe("agent_life_snapshot", self._flush_agent_life_snapshot)
        _safe("move_summary", self._flush_move_summary)
        _safe("counters", self._flush_counters)
        _safe("ppo_rich_csv", lambda: self._drain_and_flush_ppo_rich_telemetry(observed_tick=self._last_tick_seen, force=True))

        if self._last_tick_seen is not None and self.tick_summary_every > 0:
            _safe("tick_summary_final", lambda: self._write_tick_summary(tick=self._last_tick_seen))

        if errors:
            print("[telemetry] Close encountered errors:\n  - " + "\n  - ".join(errors))
