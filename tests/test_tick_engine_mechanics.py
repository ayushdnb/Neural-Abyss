from __future__ import annotations

import pytest
import torch

import config
from engine.agent_registry import COL_ALIVE, COL_HP, COL_X, COL_Y
from engine.tick import TickMetrics
from tests._sim_helpers import make_test_engine, make_zones, register_agent


def test_apply_deaths_records_root_dead_log_for_metabolism_regression(monkeypatch) -> None:
    engine, registry, _grid, stats = make_test_engine(monkeypatch, max_agents=2)
    agent_id = register_agent(registry, engine.grid, 0, team_is_red=True, x=2, y=2)

    red_deaths, blue_deaths = engine._apply_deaths(
        torch.tensor([0], dtype=torch.long),
        TickMetrics(),
        credit_kills=False,
        death_cause="metabolism",
    )

    assert (red_deaths, blue_deaths) == (1, 0)
    assert stats.drain_dead_log() == [
        {
            "tick": 0,
            "agent_id": agent_id,
            "team": "red",
            "x": 2,
            "y": 2,
            "killer_team": "",
        }
    ]


def test_run_tick_movement_conflict_highest_hp_wins(monkeypatch) -> None:
    engine, registry, _grid, _stats = make_test_engine(monkeypatch, max_agents=4)
    register_agent(registry, engine.grid, 0, team_is_red=True, x=2, y=3, hp=0.9)
    register_agent(registry, engine.grid, 1, team_is_red=False, x=4, y=3, hp=0.4)
    monkeypatch.setenv("FWS_DEBUG_FORCE_ACTIONS", "3,7")

    metrics = engine.run_tick()

    assert metrics["moved"] == 1
    assert metrics["move_conflict_lost"] == 1
    assert metrics["move_conflict_tie"] == 0
    assert float(registry.agent_data[0, COL_X].item()) == 3.0
    assert float(registry.agent_data[0, COL_Y].item()) == 3.0
    assert float(registry.agent_data[1, COL_X].item()) == 4.0
    assert float(registry.agent_data[1, COL_Y].item()) == 3.0


def test_run_tick_movement_conflict_tie_blocks_both_agents(monkeypatch) -> None:
    engine, registry, _grid, _stats = make_test_engine(monkeypatch, max_agents=4)
    register_agent(registry, engine.grid, 0, team_is_red=True, x=2, y=3, hp=0.8)
    register_agent(registry, engine.grid, 1, team_is_red=False, x=4, y=3, hp=0.8)
    monkeypatch.setenv("FWS_DEBUG_FORCE_ACTIONS", "3,7")

    metrics = engine.run_tick()

    assert metrics["moved"] == 0
    assert metrics["move_conflict_lost"] == 0
    assert metrics["move_conflict_tie"] == 2
    assert float(registry.agent_data[0, COL_X].item()) == 2.0
    assert float(registry.agent_data[1, COL_X].item()) == 4.0


def test_run_tick_heal_zone_recovers_hp(monkeypatch) -> None:
    zones = make_zones(grid_h=7, grid_w=7, heal_cells=(((2, 2),),))
    engine, registry, _grid, _stats = make_test_engine(monkeypatch, max_agents=2, zones=zones)
    register_agent(registry, engine.grid, 0, team_is_red=True, x=2, y=2, hp=0.5, hp_max=1.0)
    monkeypatch.setenv("FWS_DEBUG_FORCE_ACTIONS", "0")

    engine.run_tick()

    assert float(registry.agent_data[0, COL_HP].item()) == pytest.approx(0.6)
    assert float(engine.grid[1, 2, 2].item()) == pytest.approx(0.6)


def test_run_tick_capture_point_awards_team_and_agent_credit(monkeypatch) -> None:
    zones = make_zones(grid_h=7, grid_w=7, cp_cells=(((3, 3),),))
    engine, registry, _grid, stats = make_test_engine(monkeypatch, max_agents=4, zones=zones)
    red_id = register_agent(registry, engine.grid, 0, team_is_red=True, x=3, y=3)
    register_agent(registry, engine.grid, 1, team_is_red=False, x=5, y=5)
    monkeypatch.setenv("FWS_DEBUG_FORCE_ACTIONS", "0,0")

    metrics = engine.run_tick()

    assert float(stats.red.cp_points) == pytest.approx(1.0)
    assert float(stats.red.score) == pytest.approx(1.0)
    assert metrics["cp_red_tick"] == pytest.approx(1.0)
    assert engine.agent_cp_points[red_id] == pytest.approx(1.0)


def test_run_tick_combat_kill_updates_stats_and_registry(monkeypatch) -> None:
    engine, registry, _grid, stats = make_test_engine(monkeypatch, max_agents=4)
    register_agent(registry, engine.grid, 0, team_is_red=True, x=2, y=2, atk=1.0)
    register_agent(registry, engine.grid, 1, team_is_red=False, x=3, y=2, hp=0.5, hp_max=0.5)
    monkeypatch.setenv("FWS_DEBUG_FORCE_ACTIONS", "17,0")

    metrics = engine.run_tick()

    assert metrics["attacks"] == 1
    assert metrics["deaths_combat"] == 1
    assert float(registry.agent_data[1, COL_ALIVE].item()) == 0.0
    assert stats.red.kills == 1
    assert stats.blue.deaths == 1


def test_run_tick_matches_between_loop_and_vmap_inference(monkeypatch) -> None:
    def _run_once(use_vmap: bool):
        monkeypatch.setattr(config, "USE_VMAP", bool(use_vmap))
        monkeypatch.setattr(config, "VMAP_MIN_BUCKET", 2)
        torch.manual_seed(1234)
        engine, registry, _grid, stats = make_test_engine(monkeypatch, max_agents=8)

        placements = (
            (0, True, 1, 1),
            (1, True, 1, 3),
            (2, True, 1, 5),
            (3, False, 5, 1),
            (4, False, 5, 3),
            (5, False, 5, 5),
        )
        for slot, team_is_red, x, y in placements:
            register_agent(registry, engine.grid, slot, team_is_red=team_is_red, x=x, y=y)

        torch.manual_seed(9876)
        metrics = engine.run_tick()
        return metrics, registry.agent_data.detach().clone(), engine.grid.detach().clone(), stats

    loop_metrics, loop_data, loop_grid, loop_stats = _run_once(False)
    vmap_metrics, vmap_data, vmap_grid, vmap_stats = _run_once(True)

    assert vmap_metrics == pytest.approx(loop_metrics)
    assert torch.equal(vmap_data, loop_data)
    assert torch.equal(vmap_grid, loop_grid)
    assert float(vmap_stats.red.score) == pytest.approx(float(loop_stats.red.score))
    assert float(vmap_stats.blue.score) == pytest.approx(float(loop_stats.blue.score))
    assert int(vmap_stats.red.kills) == int(loop_stats.red.kills)
    assert int(vmap_stats.blue.kills) == int(loop_stats.blue.kills)
