import csv
from pathlib import Path

import config
from tests._sim_helpers import make_test_engine
from utils.telemetry import TelemetrySession


def test_disabled_telemetry_session_creates_no_output_tree(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(config, "TELEMETRY_ENABLED", False)
    monkeypatch.setattr(config, "TELEMETRY_WRITE_AGENT_STATIC", True)
    monkeypatch.setattr(config, "TELEMETRY_TICK_SUMMARY_EVERY", 5)

    session = TelemetrySession(tmp_path)

    assert session.enabled is False
    assert not (tmp_path / "telemetry").exists()


def test_validate_schema_manifest_compat_rejects_append_drift(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(config, "TELEMETRY_ENABLED", True)

    session = TelemetrySession(tmp_path)
    session.write_schema_manifest(
        {
            "schema_version": "2",
            "lineage_fields": ["tick", "parent_id"],
            "reward_fields": ["reward_total"],
            "death_causes": ["combat"],
            "mechanics": {"archer_los_blocks_walls": True},
        }
    )

    try:
        session.validate_schema_manifest_compat(
            {
                "schema_version": "2",
                "lineage_fields": ["tick", "parent_id", "child_id"],
                "reward_fields": ["reward_total"],
                "death_causes": ["combat"],
                "mechanics": {"archer_los_blocks_walls": True},
            }
        )
    except RuntimeError as exc:
        assert "manifest mismatch" in str(exc)
    else:
        raise AssertionError("schema manifest drift should fail append-in-place validation")


def test_record_birth_and_death_write_life_and_detailed_logs(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(config, "TELEMETRY_ENABLED", True)

    session = TelemetrySession(tmp_path)
    session.record_birth(
        tick=4,
        agent_id=11,
        slot_id=3,
        team=2,
        unit_type=1,
        parent_id=None,
        spawn_reason="bootstrap",
        generation=2,
    )
    session.record_deaths(
        tick=9,
        dead_ids=[11],
        dead_team=[2],
        dead_unit=[1],
        dead_slots=[3],
        death_causes=["metabolism"],
        killer_ids=[None],
        killer_slots=[None],
        killer_teams=[None],
    )
    session.close()

    detailed = (tmp_path / "telemetry" / "dead_agents_log_detailed.csv").read_text(encoding="utf-8")
    with (tmp_path / "telemetry" / "agent_life.csv").open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert "metabolism" in detailed
    assert "11" in detailed
    assert rows[0]["agent_id"] == "11"
    assert rows[0]["death_tick"] == "9"


def test_tick_summary_is_not_duplicated_by_flush_and_close(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(config, "TELEMETRY_ENABLED", True)
    monkeypatch.setattr(config, "TELEMETRY_TICK_SUMMARY_EVERY", 1)

    engine, registry, _grid, stats = make_test_engine(monkeypatch, max_agents=4)
    stats.tick = 3

    session = TelemetrySession(tmp_path)
    session.attach_context(registry=registry, stats=stats, ppo_runtime=None)

    session.on_tick_end(3)
    session.flush(reason="checkpoint_save")
    session.close()

    with (tmp_path / "telemetry" / "tick_summary.csv").open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 1
    assert rows[0]["tick"] == "3"
