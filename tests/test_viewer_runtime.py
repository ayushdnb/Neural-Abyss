from __future__ import annotations

from pathlib import Path

import pytest

import config
from tests._sim_helpers import make_test_engine, make_zones, register_agent

pygame = pytest.importorskip("pygame")
pytestmark = pytest.mark.filterwarnings(
    "ignore:The UI requires the pygame-ce distribution.*:RuntimeWarning"
)


@pytest.fixture()
def headless_pygame(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    try:
        pygame.quit()
    except Exception:
        pass
    yield
    try:
        pygame.quit()
    except Exception:
        pass


def test_viewer_zone_controls_and_legacy_checkpoint_state(headless_pygame, monkeypatch) -> None:
    from ui.viewer import Viewer

    monkeypatch.setattr(config, "VIEWER_CENTER_WINDOW", False, raising=False)
    monkeypatch.setattr(config, "PYGAME_CE_STRICT_RUNTIME", False, raising=False)
    monkeypatch.setattr(config, "CELL_SIZE", 8)

    zones = make_zones(
        grid_h=7,
        grid_w=7,
        heal_cells=(((2, 2), (2, 3)),),
        cp_cells=(((4, 4),),),
    )
    engine, registry, grid, stats = make_test_engine(monkeypatch, zones=zones, max_agents=4)

    viewer = Viewer(grid, cell_size=8)
    viewer.engine = engine
    viewer.registry = registry
    viewer.stats = stats

    assert viewer.select_heal_zone_at_cell(2, 2) is True
    assert viewer.get_selected_zone_status()["zone_id"] == "heal_0"

    assert viewer.manual_toggle_zone_at_cell(2, 2) is True
    zone_status = viewer.get_selected_zone_status()
    assert zone_status is not None
    assert zone_status["manual_override_label"] in {"forced ON", "forced OFF"}
    assert "manual override applied" in zone_status["status_message"]

    before_scheduler = bool(engine.catastrophe_controller.scheduler_enabled)
    viewer.toggle_catastrophe_scheduler()
    assert bool(engine.catastrophe_controller.scheduler_enabled) is (not before_scheduler)

    assert viewer.restore_all_zones_to_normal_effective_state() is True

    viewer.apply_checkpoint_state(
        {
            "paused": "true",
            "speed_mult": "4",
            "show_rays": "1",
            "threat_vision_mode": "1",
            "battle_view_enabled": "1",
            "show_brain_types": "1",
            "camera": {"offset_x": "3.5", "offset_y": 7, "zoom": "2.5"},
            "agent_scores": {"12": "1.25"},
            "marked": [1, "2", "bad", 1],
        }
    )

    state = viewer.capture_state()
    assert state["paused"] is True
    assert state["speed_multiplier"] == pytest.approx(4.0)
    assert state["show_rays"] is True
    assert state["threat_vision_mode"] is True
    assert state["battle_view_enabled"] is True
    assert state["show_brain_types"] is True
    assert state["camera"] == {"offset_x": 3.5, "offset_y": 6.0, "zoom": 2.5}
    assert state["agent_scores"] == {12: 1.25}
    assert state["marked"] == [1, 2]


def test_viewer_run_headless_smoke_handles_events_and_manual_checkpoint(
    headless_pygame,
    monkeypatch,
    tmp_path: Path,
) -> None:
    from ui.viewer import Viewer

    monkeypatch.setattr(config, "VIEWER_CENTER_WINDOW", False, raising=False)
    monkeypatch.setattr(config, "PYGAME_CE_STRICT_RUNTIME", False, raising=False)
    monkeypatch.setattr(config, "CELL_SIZE", 8)
    monkeypatch.setattr(config, "TARGET_FPS", 30)
    monkeypatch.setattr(config, "CHECKPOINT_EVERY_TICKS", 0)
    monkeypatch.setattr(config, "CHECKPOINT_KEEP_LAST_N", 1)
    monkeypatch.setattr(config, "CHECKPOINT_PIN_ON_MANUAL", True)

    zones = make_zones(
        grid_h=7,
        grid_w=7,
        heal_cells=(((2, 2),),),
        cp_cells=(((4, 4),),),
    )
    engine, registry, grid, stats = make_test_engine(monkeypatch, zones=zones, max_agents=4)
    register_agent(registry, grid, 0, team_is_red=True, x=3, y=3)
    engine.catastrophe_controller.set_scheduler_enabled(True)

    viewer = Viewer(grid, cell_size=8)
    for ev in [
        pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE),
        pygame.event.Event(pygame.KEYDOWN, key=pygame.K_PERIOD),
        pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r),
        pygame.event.Event(pygame.KEYDOWN, key=pygame.K_t),
        pygame.event.Event(pygame.KEYDOWN, key=pygame.K_b),
        pygame.event.Event(pygame.KEYDOWN, key=pygame.K_n),
        pygame.event.Event(pygame.KEYDOWN, key=pygame.K_y),
        pygame.event.Event(pygame.KEYDOWN, key=pygame.K_F9),
        pygame.event.Event(pygame.QUIT),
    ]:
        pygame.event.post(ev)

    run_dir = tmp_path / "viewer_run"
    viewer.run(
        engine,
        registry,
        stats,
        tick_limit=10,
        target_fps=30,
        run_dir=str(run_dir),
    )

    checkpoints_dir = run_dir / "checkpoints"
    checkpoint_dirs = sorted(p for p in checkpoints_dir.iterdir() if p.is_dir())

    assert int(stats.tick) == 1, "paused single-step flow should advance exactly one tick"
    assert viewer.paused is True
    assert viewer.show_rays is True
    assert viewer.threat_vision_mode is True
    assert viewer.battle_view_enabled is True
    assert viewer.show_brain_types is True
    assert bool(engine.catastrophe_controller.scheduler_enabled) is False
    assert checkpoints_dir.exists()
    assert (checkpoints_dir / "latest.txt").exists()
    assert len(checkpoint_dirs) == 1, "manual F9 save should create exactly one checkpoint in this smoke"
