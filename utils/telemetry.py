# Infinite_War_Simulation/utils/telemetry.py
"""
Telemetry module for recording agent life events, lineage, and damage.
Provides a session object that writes CSV snapshots and JSONL event chunks.
"""
from __future__ import annotations
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import config

def _to_int(x: Any) -> int:
    """
    Convert a value to int, handling floats that may be stored as float type.
    This is necessary because agent_data in the registry uses float dtype.
    """
    try:
        return int(x)
    except Exception:
        # Fallback: convert via float to handle strings/other numeric types
        return int(float(x))

def _atomic_write_text(path: Path, text: str) -> None:
    """
    Atomically write text to a file by writing to a temporary file first, then replacing the target.
    This prevents partial reads by other processes.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)

def _parse_validate_level(v: Any, default: int = 2) -> int:
    """
    Parse and validate the TELEMETRY_VALIDATE_LEVEL config value.
    Accepts:
      - int/float (direct conversion)
      - string values: "off"/"0", "basic"/"1", "strict"/"2"
    Returns an integer 0, 1, or 2.
    """
    # If it's already a number, try to convert to int
    if isinstance(v, (int, float)):
        try:
            return int(v)
        except Exception:
            return int(default)
    
    # If it's a string, handle common aliases
    if isinstance(v, str):
        s = v.strip().lower()
        m = {"off": 0, "0": 0, "basic": 1, "1": 1, "strict": 2, "2": 2}
        if s in m:
            return int(m[s])
        try:
            return int(s)
        except Exception:
            return int(default)
            
    # Fallback to default
    return int(default)

class TelemetrySession:
    """
    Minimal scientific telemetry:
      - AgentLife table (snapshot CSV)
      - LineageEdges table (append CSV)
      - Event logs (append-only jsonl, chunked)
      - Optional run metadata JSON, agent static CSV, tick summary CSV.
    """
    def __init__(self, run_dir: Path) -> None:
        # Read configuration flags; all settings default to safe values.
        self.enabled: bool = bool(getattr(config, "TELEMETRY_ENABLED", False))
        self.run_dir = Path(run_dir)

        # Schema version for events – helps downstream parsers handle different fields.
        self.schema_version: int = int(getattr(config, "TELEMETRY_SCHEMA_VERSION", 2))

        # Reuse existing knobs (do not invent a second telemetry config system).
        self.tag: str = str(getattr(config, "TELEMETRY_TAG", ""))
        self.event_chunk_size: int = int(getattr(config, "TELEMETRY_EVENT_CHUNK_SIZE", 50_000))
        self.flush_every: int = int(getattr(config, "TELEMETRY_PERIODIC_FLUSH_EVERY", 250))
        
        self.events_format: str = str(getattr(config, "TELEMETRY_EVENTS_FORMAT", "jsonl"))
        self.events_gzip: bool = bool(getattr(config, "TELEMETRY_EVENTS_GZIP", False)) # kept for future; no deps added

        self.log_births: bool = bool(getattr(config, "TELEMETRY_LOG_BIRTHS", True))
        self.log_deaths: bool = bool(getattr(config, "TELEMETRY_LOG_DEATHS", True))
        self.log_damage: bool = bool(getattr(config, "TELEMETRY_LOG_DAMAGE", True))
        self.log_kills: bool = bool(getattr(config, "TELEMETRY_LOG_KILLS", True))
        self.damage_mode: str = str(getattr(config, "TELEMETRY_DAMAGE_MODE", "victim_sum"))

        # Validation level: 0=off, 1=basic, 2=strict (default)
        self.validate_level: int = _parse_validate_level(
            getattr(config, "TELEMETRY_VALIDATE_LEVEL", 2), default=2
        )
        self.validate_every: int = int(getattr(config, "TELEMETRY_VALIDATE_EVERY", 1000))
        self.abort_on_anomaly: bool = bool(getattr(config, "TELEMETRY_ABORT_ON_ANOMALY", False))

        # Output layout (inside run_dir, minimal and trackable)
        self.telemetry_dir = self.run_dir / "telemetry"
        self.events_dir = self.telemetry_dir / "events"
        self.agent_life_path = self.telemetry_dir / "agent_life.csv"
        self.lineage_edges_path = self.telemetry_dir / "lineage_edges.csv"
        
        # New sidecar files added by patch
        self.run_meta_path = self.telemetry_dir / "run_meta.json"
        self.agent_static_path = self.telemetry_dir / "agent_static.csv"
        self.tick_summary_path = self.telemetry_dir / "tick_summary.csv"

        # Create directories if they don't exist
        self.telemetry_dir.mkdir(parents=True, exist_ok=True)
        self.events_dir.mkdir(parents=True, exist_ok=True)

        # Additive sidecars (all config-gated) – added by patch
        self.schema_version: str = str(getattr(config, "TELEMETRY_SCHEMA_VERSION", "v2"))
        
        # FIX: Renamed attribute to avoid conflict with write_run_meta method
        self.do_write_run_meta: bool = bool(getattr(config, "TELEMETRY_WRITE_RUN_META", True))
        
        self.write_agent_static: bool = bool(getattr(config, "TELEMETRY_WRITE_AGENT_STATIC", False))
        self.tick_summary_every: int = int(getattr(config, "TELEMETRY_TICK_SUMMARY_EVERY", 0))

        # Context refs (pure observation; never mutates) – added by patch
        self._registry: Any = None
        self._stats: Any = None

        # AgentStatic append-safety: track which agent_ids already written – added by patch
        self._static_written: set[int] = set()
        if self.write_agent_static and self.agent_static_path.exists():
            try:
                with self.agent_static_path.open("r", newline="", encoding="utf-8") as f:
                    r = csv.DictReader(f)
                    for row in r:
                        if "agent_id" in row and row["agent_id"] not in (None, ""):
                            self._static_written.add(int(float(row["agent_id"])))
            except Exception:
                # If parsing fails, degrade safely (may re-emit some rows; still analysis-usable).
                self._static_written = set()

        # AgentLife state keyed by agent_id
        self._life: Dict[int, Dict[str, Any]] = {}
        self._offspring_count: Dict[int, int] = {}

        # Event buffering (append-only, chunked)
        self._events_buf: List[Dict[str, Any]] = []
        self._chunk_idx: int = self._discover_next_chunk_idx()

        # If telemetry is enabled, attempt to reload previous agent life snapshot.
        if self.enabled:
            self._rehydrate_agent_life()
        
        self._last_tick_seen: Optional[int] = None

        # Ensure lineage edges header exists if file is new
        if not self.lineage_edges_path.exists():
            self._append_csv_rows(
                self.lineage_edges_path,
                fieldnames=["tick", "parent_id", "child_id"],
                rows=[],
            )
        
        # Create headers for agent_static and tick_summary if they are enabled and files are new – added by patch
        if self.write_agent_static and (not self.agent_static_path.exists()):
            self._append_csv_rows(
                self.agent_static_path,
                fieldnames=[
                    "agent_id","team_id","unit_type","brain_type","param_count",
                    "max_hp","base_atk","vision_range",
                    "spawn_tick","parent_id","spawn_reason",
                ],
                rows=[],
            )

        if self.tick_summary_every > 0 and (not self.tick_summary_path.exists()):
            self._append_csv_rows(
                self.tick_summary_path,
                fieldnames=[
                    "tick","elapsed_s",
                    "red_alive","blue_alive","mean_hp_red","mean_hp_blue",
                    "red_kills","blue_kills","red_deaths","blue_deaths",
                    "red_dmg_dealt","blue_dmg_dealt","red_dmg_taken","blue_dmg_taken",
                ],
                rows=[],
            )

    def _discover_next_chunk_idx(self) -> int:
        """
        Look for existing event chunks in the events directory and return the next available index.
        This allows resuming a run without overwriting.
        """
        max_idx = -1
        for p in self.events_dir.glob("events_*.jsonl"):
            try:
                stem = p.stem  # events_000001
                idx = int(stem.split("_", 1)[1])
                max_idx = max(max_idx, idx)
            except Exception:
                continue
        return max_idx + 1

    def _rehydrate_agent_life(self) -> None:
        """
        If an agent_life.csv snapshot exists from a previous run, load it into memory.
        This allows resuming telemetry with existing agents already accounted for.
        """
        if not self.agent_life_path.exists():
            return
        
        try:
            with self.agent_life_path.open("r", newline="", encoding="utf-8") as f:
                r = csv.DictReader(f)
                for row in r:
                    if not row: continue
                    aid_s = (row.get("agent_id") or "").strip()
                    if aid_s == "": continue
                    aid = int(aid_s)
                    
                    rec: Dict[str, Any] = {}
                    # Load basic integer/string fields
                    for k in ("slot_id", "team", "unit_type", "notes"):
                        v = row.get(k, "")
                        if v != "":
                            try:
                                rec[k] = int(float(v)) if k != "notes" else str(v)
                            except Exception:
                                rec[k] = str(v)
                    
                    # Load integer tick fields
                    for k in ("born_tick", "death_tick", "parent_id", "kills_total"):
                        v = (row.get(k) or "").strip()
                        if v != "":
                            try:
                                rec[k] = int(float(v))
                            except Exception:
                                pass
                    
                    # Load float damage fields
                    for k in ("damage_dealt_total", "damage_taken_total"):
                        v = (row.get(k) or "").strip()
                        if v != "":
                            try:
                                rec[k] = float(v)
                            except Exception:
                                pass
                    
                    self._life[aid] = rec

                    # Offspring count is stored separately
                    oc = (row.get("offspring_count") or "").strip()
                    if oc != "":
                        try:
                            self._offspring_count[aid] = int(float(oc))
                        except Exception:
                            pass
        except Exception as e:
            self._anomaly(f"rehydrate agent_life failed: {e}")

    def _anomaly(self, msg: str) -> None:
        """ Handle a detected anomaly: either abort or print a warning.
        """
        if self.abort_on_anomaly:
            raise AssertionError(msg)
        # FIX: Silence console output as requested.
        # print(f"[telemetry] ANOMALY: {msg}")
        pass

    def _require_birth(self, agent_id: int, context: str) -> None:
        """
        Ensure that an agent_id has a birth record; if not, report an anomaly.
        Used before recording events that require the agent to exist.
        """
        if agent_id not in self._life:
            self._anomaly(f"{context}: missing birth for agent_id={agent_id}")

    def _append_csv_rows(self, path: Path, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
        """
        Append rows to a CSV file, writing the header only if the file is new.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.exists()
        
        with path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                w.writeheader()
            for r in rows:
                w.writerow(r)

    def _emit_event(self, ev: Dict[str, Any]) -> None:
        """
        Buffer an event; flush the buffer when chunk size is reached.
        """
        if not self.enabled:
            return

        # Keep format minimal: jsonl only in this phase (no new deps for gzip).
        if self.events_format.lower() != "jsonl":
            # Do not invent other writers; degrade safely to jsonl.
            ev = dict(ev)
            ev["notes"] = (ev.get("notes", "") + " events_format_forced_jsonl").strip()

        self._events_buf.append(ev)
        if len(self._events_buf) >= self.event_chunk_size:
            self._flush_event_chunk()

    def _flush_event_chunk(self) -> None:
        """
        Write the current event buffer to a chunk file and clear the buffer.
        """
        if not self._events_buf:
            return
        
        out = self.events_dir / f"events_{self._chunk_idx:06d}.jsonl"
        lines = "\n".join(json.dumps(e, ensure_ascii=False) for e in self._events_buf) + "\n"
        _atomic_write_text(out, lines)
        
        self._events_buf.clear()
        self._chunk_idx += 1

    def _flush_agent_life_snapshot(self) -> None:
        """
        Write the current state of all agents to a CSV snapshot (overwrite).
        This includes both living and dead agents.
        """
        # Snapshot: overwrite atomically
        fieldnames = [
            "agent_id", "slot_id", "team", "unit_type",
            "born_tick", "death_tick", "lifespan_ticks", "parent_id", "offspring_count",
            "kills_total", "damage_dealt_total", "damage_taken_total", "notes",
            # Additional fields that may be populated by future extensions
            "moves_attempted", "moves_success", "moves_blocked_wall", "moves_blocked_occupied",
            "cells_walked_total_l1", "reward_total", "death_cause", "last_seen_tick",
        ]
        
        rows: List[Dict[str, Any]] = []
        for aid, rec in sorted(self._life.items(), key=lambda kv: kv[0]):
            r = dict(rec)
            r["agent_id"] = aid
            r["offspring_count"] = int(self._offspring_count.get(aid, 0))
            
            bt = r.get("born_tick")
            dt = r.get("death_tick")
            if bt is not None and dt is not None:
                r["lifespan_ticks"] = int(dt) - int(bt)
            else:
                r["lifespan_ticks"] = ""
            
            rows.append({k: r.get(k, "") for k in fieldnames})
        
        # CSV serialize using StringIO to avoid temporary file issues
        import io
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
        
        _atomic_write_text(self.agent_life_path, buf.getvalue())

    def validate(self) -> None:
        """
        Run consistency checks on the agent life records according to the validation level.
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

    # -------------------- Public API called from main/tick --------------------

    def attach_context(self, registry: Any, stats: Any) -> None:
        """Optional: lets telemetry compute low-frequency summaries/static attrs without engine refactors."""
        self._registry = registry
        self._stats = stats

    def write_run_meta(self, meta: Dict[str, Any]) -> None:
        """Write run metadata JSON file (atomically)."""
        # FIX: Use correct attribute name
        if not self.enabled or not self.do_write_run_meta:
            return
        
        out = dict(meta or {})
        out.setdefault("schema_version", self.schema_version)
        out.setdefault("git_commit", os.getenv("GIT_COMMIT", None))
        
        _atomic_write_text(self.run_meta_path, json.dumps(out, indent=2, sort_keys=True))

    def record_resume(self, tick: int, checkpoint_path: str) -> None:
        """Emit a resume event when loading from a checkpoint."""
        if not self.enabled:
            return
        self._emit_event({
            "tick": int(tick),
            "type": "resume",
            "checkpoint_path": str(checkpoint_path),
        })

    def bootstrap_from_registry(self, registry: Any, tick: int, note: str = "bootstrap") -> None:
        """
        Called once at startup to seed births for agents that already exist.
        This uses the agent registry to find all alive agents and records a birth for each,
        marking them as existing from the start.
        """
        if not self.enabled:
            return
        
        from engine.agent_registry import COL_ALIVE, COL_TEAM, COL_UNIT, COL_AGENT_ID
        data = registry.agent_data
        alive_mask = (data[:, COL_ALIVE] > 0.5)
        alive_slots = alive_mask.nonzero(as_tuple=False).view(-1).tolist()
        
        for slot in alive_slots:
            if hasattr(registry, "agent_uids"):
                aid = int(registry.agent_uids[slot].item())
            else:
                aid = _to_int(data[slot, COL_AGENT_ID].item())
            team = _to_int(data[slot, COL_TEAM].item())
            unit = _to_int(data[slot, COL_UNIT].item())
            
            self.record_birth(
                tick=tick,
                agent_id=aid,
                slot_id=int(slot),
                team=team,
                unit_type=unit,
                parent_id=None,
                notes=note,
                allow_existing=True,
            )
            
            # Also write static attributes if enabled – added by patch
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
    ) -> None:
        """
        Record the birth (creation) of an agent.
        If allow_existing is True, we may update an existing record (used during bootstrap).
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
            # If we allow existing, preserve the original born_tick; otherwise set it now.
            "born_tick": (rec.get("born_tick", int(tick)) if allow_existing else int(tick)),
            "death_tick": rec.get("death_tick", None),
            "parent_id": (int(parent_id) if parent_id is not None else None),
            "kills_total": int(rec.get("kills_total", 0)),
            "damage_dealt_total": float(rec.get("damage_dealt_total", 0.0)),
            "damage_taken_total": float(rec.get("damage_taken_total", 0.0)),
            "notes": str(notes or rec.get("notes", "")),
        })
        self._life[agent_id] = rec
        
        if parent_id is not None:
            self._offspring_count[int(parent_id)] = int(self._offspring_count.get(int(parent_id), 0)) + 1
            self._append_csv_rows(
                self.lineage_edges_path,
                fieldnames=["tick", "parent_id", "child_id"],
                rows=[{"tick": int(tick), "parent_id": int(parent_id), "child_id": int(agent_id)}],
            )
        
        if self.log_births:
            self._emit_event({
                "schema_version": int(getattr(self, "schema_version", 2)),
                "tick": int(tick),
                "type": "birth",
                "agent_id": int(agent_id),
                "slot_id": int(slot_id),
                "team": int(team),
                "unit_type": int(unit_type),
                "parent_id": (int(parent_id) if parent_id is not None else None),
                "notes": str(notes),
            })

    def ingest_spawn_meta(self, meta: List[Dict[str, Any]]) -> None:
        """
        Process spawn metadata from RespawnController and record births accordingly.
        """
        if not self.enabled or not meta:
            return

        for m in meta:
            # Schema provided by RespawnController.step meta_out.append(...)
            tick = _to_int(m.get("tick"))
            slot = _to_int(m.get("slot"))
            aid = _to_int(m.get("agent_id"))
            team = _to_int(m.get("team_id"))
            unit = _to_int(m.get("unit_id"))
            parent = m.get("parent_agent_id", None)
            parent_id = (_to_int(parent) if parent is not None else None)
            
            self.record_birth(
                tick=tick,
                agent_id=aid,
                slot_id=slot,
                team=team,
                unit_type=unit,
                parent_id=parent_id,
                notes="respawn",
                allow_existing=False,
            )

            # Also write static attributes if enabled – added by patch
            self._maybe_write_agent_static(
                tick=int(tick),
                slot_id=int(slot),
                agent_id=int(aid),
                team_id=int(team),
                unit_type=int(unit),
                parent_id=parent_id,
                spawn_reason="respawn",
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
        Write a row to agent_static.csv for a newly created agent, but only if:
        - telemetry enabled
        - agent_static writing enabled
        - registry is attached
        - agent_id not already written (deduplication)
        Retrieves static attributes from the registry and brain info.
        """
        if (not self.enabled) or (not self.write_agent_static) or (self._registry is None):
            return

        if agent_id in self._static_written:
            return
        
        try:
            # Local import to avoid circular dependencies at module level
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
                    "agent_id","team_id","unit_type","brain_type","param_count",
                    "max_hp","base_atk","vision_range",
                    "spawn_tick","parent_id","spawn_reason",
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
        Record damage in 'victim_sum' mode: each victim receives a total damage amount.
        The damage taken total is incremented, and a damage event is emitted.
        """
        if not self.enabled:
            return

        for i, vid in enumerate(victim_ids):
            self._require_birth(vid, "damage(victim_sum)")
            rec = self._life[vid]
            rec["damage_taken_total"] = float(rec.get("damage_taken_total", 0.0)) + float(damage[i])
            
            if self.log_damage:
                ev = {
                    "schema_version": int(getattr(self, "schema_version", 2)),
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
        Record damage in 'attacker_sum' mode: each attacker deals a total damage amount.
        Only the total damage dealt is updated; no event is emitted here (events come from victim_sum/per_hit).
        """
        if not self.enabled:
            return

        for i, aid in enumerate(attacker_ids):
            self._require_birth(aid, "damage(attacker_sum)")
            rec = self._life[aid]
            rec["damage_dealt_total"] = float(rec.get("damage_dealt_total", 0.0)) + float(damage_dealt[i])
        
        # No event emission here; event volume is controlled by victim_sum / per_hit mode.

    def record_damage_per_hit(
        self,
        tick: int,
        attacker_ids: List[int],
        victim_ids: List[int],
        damage: List[float],
    ) -> None:
        """
        Record damage in 'per_hit' mode: each hit is logged as a separate event.
        This does not update totals (those are handled by attacker_sum/victim_sum calls).
        """
        if not self.enabled or not self.log_damage:
            return

        for i in range(len(damage)):
            aid = int(attacker_ids[i])
            vid = int(victim_ids[i])
            self._require_birth(aid, "damage(per_hit attacker)")
            self._require_birth(vid, "damage(per_hit victim)")
            
            self._emit_event({
                "schema_version": int(getattr(self, "schema_version", 2)),
                "tick": int(tick),
                "type": "damage",
                "mode": "per_hit",
                "attacker_id": aid,
                "victim_id": vid,
                "damage": float(damage[i]),
            })

    def record_kills(self, tick: int, killer_ids: List[int], victim_ids: List[int]) -> None:
        """
        Record kills: increment killer's kill count and emit a kill event.
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
                    "schema_version": int(getattr(self, "schema_version", 2)),
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
    ) -> None:
        """
        Record deaths: set death_tick and emit a death event.
        """
        if not self.enabled:
            return

        for i, did in enumerate(dead_ids):
            self._require_birth(did, "death")
            rec = self._life[did]
            
            if rec.get("death_tick", None) is not None:
                self._anomaly(f"death: duplicate death for agent_id={did}")
                continue
            
            rec["death_tick"] = int(tick)
            
            if self.log_deaths:
                self._emit_event({
                    "schema_version": int(getattr(self, "schema_version", 2)),
                    "tick": int(tick),
                    "type": "death",
                    "agent_id": int(did),
                    "slot_id": int(dead_slots[i]) if i < len(dead_slots) else None,
                    "team": int(dead_team[i]) if i < len(dead_team) else None,
                    "unit_type": int(dead_unit[i]) if i < len(dead_unit) else None,
                    "notes": str(notes),
                })

    def on_tick_end(self, tick: int) -> None:
        """
        Called at the end of each simulation tick.
        Performs periodic validation and flushing, and optionally writes tick summary.
        """
        if not self.enabled:
            return
        
        self._last_tick_seen = int(tick)
        
        if self.validate_every > 0 and (int(tick) % int(self.validate_every) == 0):
            self.validate()

        if self.flush_every > 0 and (int(tick) % int(self.flush_every) == 0):
            # Periodic flush: snapshot AgentLife and also flush any pending events.
            self._flush_agent_life_snapshot()
            self._flush_event_chunk()
        
        # Write tick summary if enabled – added by patch
        if self.tick_summary_every > 0 and (tick % int(self.tick_summary_every)) == 0:
            self._write_tick_summary(tick=int(tick))

    def _write_tick_summary(self, tick: int) -> None:
        """
        Compute and append a summary row to tick_summary.csv.
        Requires registry and stats to be attached.
        """
        if (not self.enabled) or (self._registry is None) or (self._stats is None):
            return
        
        try:
            import torch # local import; core dep already used by project
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
            self._append_csv_rows(self.tick_summary_path, fieldnames=list(row.keys()), rows=[row])
        except Exception as e:
            self._anomaly(f"tick_summary write failed: {e}")

    def close(self) -> None:
        """
        Clean shutdown: flush all buffers and write final snapshots.
        """
        if not self.enabled:
            return
        try:
            self._flush_event_chunk()
            self._flush_agent_life_snapshot()
            if self._last_tick_seen is not None and self.tick_summary_every > 0:
                self._write_tick_summary(tick=self._last_tick_seen)
        except Exception as e:
            print(f"[telemetry] Close failed: {e}")