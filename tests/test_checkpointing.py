from __future__ import annotations

from pathlib import Path

import pytest
import torch

from engine.agent_registry import COL_ALIVE, COL_HP
from engine.tick import TickEngine
from simulation.stats import SimulationStats
from tests._sim_helpers import CPU, make_test_engine, make_zones, register_agent
from utils.checkpointing import CheckpointError, CheckpointManager


def test_checkpoint_save_load_and_apply_roundtrip(monkeypatch, tmp_path: Path) -> None:
    zones = make_zones(
        grid_h=7,
        grid_w=7,
        heal_cells=(((2, 2),),),
        cp_cells=(((4, 4),),),
    )
    zones.disable_zone_manually("heal_0")

    engine, registry, _grid, stats = make_test_engine(monkeypatch, zones=zones, max_agents=4)
    agent_id = register_agent(
        registry,
        engine.grid,
        0,
        team_is_red=True,
        x=3,
        y=3,
        hp=0.75,
        atk=0.4,
        generation=3,
    )

    stats.tick = 17
    stats.red.score = 2.5
    engine.agent_scores[agent_id] = 3.25
    engine.agent_kill_counts[agent_id] = 1.0

    manager = CheckpointManager(tmp_path / "run")
    out_dir = manager.save_atomic(
        engine=engine,
        registry=registry,
        stats=stats,
        viewer_state={"paused": True},
        notes="roundtrip",
    )

    ckpt = CheckpointManager.load(str(out_dir))
    grid2 = ckpt["world"]["grid"].to(CPU)
    zones2 = CheckpointManager.zones_from_checkpoint(ckpt["world"], device=CPU)
    stats2 = SimulationStats()
    registry2 = registry.__class__(grid2)
    engine2 = TickEngine(registry2, grid2, stats2, zones=zones2)

    CheckpointManager.apply_loaded_checkpoint(
        ckpt,
        engine=engine2,
        registry=registry2,
        stats=stats2,
        device=CPU,
    )

    assert (out_dir / "DONE").exists()
    assert (manager.ckpt_base / "latest.txt").read_text(encoding="utf-8").strip() == out_dir.name
    assert int(stats2.tick) == 17
    assert float(stats2.red.score) == 2.5
    assert int(registry2.agent_uids[0].item()) == agent_id
    assert registry2.generations[0] == 3
    assert float(registry2.agent_data[0, COL_ALIVE].item()) == 1.0
    assert float(registry2.agent_data[0, COL_HP].item()) == pytest.approx(0.75)
    assert engine2.agent_scores[agent_id] == pytest.approx(3.25)
    assert engine2.agent_kill_counts[agent_id] == pytest.approx(1.0)
    assert zones2 is not None
    assert zones2.manual_zone_enabled == {"heal_0": False}
    assert tuple(zones2.active_heal_zone_ids) == ()


def test_load_rejects_checkpoint_without_done_marker(tmp_path: Path) -> None:
    ckpt_dir = tmp_path / "broken"
    ckpt_dir.mkdir()
    torch.save({"checkpoint_version": 1}, ckpt_dir / "checkpoint.pt")

    with pytest.raises(CheckpointError, match="DONE marker"):
        CheckpointManager.load(str(ckpt_dir))


def test_apply_loaded_checkpoint_rejects_stale_next_agent_id(monkeypatch, tmp_path: Path) -> None:
    engine, registry, _grid, stats = make_test_engine(monkeypatch, max_agents=4)
    register_agent(registry, engine.grid, 0, team_is_red=True, x=2, y=2)
    stats.tick = 3

    manager = CheckpointManager(tmp_path / "run")
    out_dir = manager.save_atomic(engine=engine, registry=registry, stats=stats)
    ckpt = CheckpointManager.load(str(out_dir))
    ckpt["registry"]["next_agent_id"] = 0

    grid2 = ckpt["world"]["grid"].to(CPU)
    stats2 = SimulationStats()
    registry2 = registry.__class__(grid2)
    engine2 = TickEngine(registry2, grid2, stats2, zones=None)

    with pytest.raises(CheckpointError, match="next_agent_id is stale"):
        CheckpointManager.apply_loaded_checkpoint(
            ckpt,
            engine=engine2,
            registry=registry2,
            stats=stats2,
            device=CPU,
        )
