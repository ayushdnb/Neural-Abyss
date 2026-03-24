from __future__ import annotations

import importlib

import pytest
import torch

import config as config_module
from agent import obs_spec
from engine.agent_registry import COL_ALIVE
from simulation.stats import SimulationStats
from tests._sim_helpers import CPU, make_test_engine, register_agent
from utils.checkpointing import CheckpointError, CheckpointManager


def _reload_obs_schema(monkeypatch, schema: str):
    monkeypatch.setenv("FWS_OBS_SCHEMA", schema)
    config = importlib.reload(config_module)
    obs_spec._IDX_CACHE.clear()
    return config


@pytest.fixture(autouse=True)
def _restore_config_after_test(monkeypatch):
    yield
    monkeypatch.delenv("FWS_OBS_SCHEMA", raising=False)
    importlib.reload(config_module)
    obs_spec._IDX_CACHE.clear()


def test_self_centric_schema_contract_exact_feature_order_and_dims(monkeypatch) -> None:
    config = _reload_obs_schema(monkeypatch, "self_centric_v1")

    assert config.RICH_BASE_DIM == 16
    assert config.INSTINCT_DIM == 4
    assert config.RICH_TOTAL_DIM == 20
    assert config.OBS_DIM == 32 * 8 + 20

    assert obs_spec.scalar_feature_names() == (
        "self_hp_ratio",
        "self_x_norm",
        "self_y_norm",
        "self_team_bit",
        "self_unit_bit",
        "self_attack_norm",
        "self_vision_norm",
        "on_heal_zone",
        "on_control_point",
        "tick_norm",
        "self_kill_ppo_points_norm",
        "self_damage_dealt_ppo_points_norm",
        "self_cp_ppo_points_norm",
        "self_damage_taken_penalty_mag_norm",
        "self_death_penalty_mag_norm",
        "self_kill_count_norm",
        "ally_archer_density",
        "ally_soldier_density",
        "noisy_enemy_density",
        "threat_ratio",
    )

    obs = torch.arange(2 * config.OBS_DIM, dtype=torch.float32).reshape(2, config.OBS_DIM)
    rays_flat, rich_base, instinct = obs_spec.split_obs_flat(obs)
    rays_raw, rich_vec = obs_spec.split_obs_for_mlp(obs)
    tokens = obs_spec.build_semantic_tokens(rich_base, instinct)

    assert rays_flat.shape == (2, config.RAYS_FLAT_DIM)
    assert rich_base.shape == (2, 16)
    assert instinct.shape == (2, 4)
    assert rays_raw.shape == (2, config.RAY_TOKEN_COUNT, config.RAY_FEAT_DIM)
    assert rich_vec.shape == (2, 20)
    assert tokens["team_context"].shape == (2, 0)


def test_self_centric_branch_ignores_global_team_scoreboard(monkeypatch) -> None:
    _reload_obs_schema(monkeypatch, "self_centric_v1")
    engine, registry, _grid, stats = make_test_engine(monkeypatch, max_agents=2)
    register_agent(registry, engine.grid, 0, team_is_red=True, x=2, y=2)
    register_agent(registry, engine.grid, 1, team_is_red=False, x=4, y=4)

    alive_idx = engine._recompute_alive_idx()
    pos_xy = registry.positions_xy(alive_idx)
    obs_a = engine._build_transformer_obs(alive_idx, pos_xy)

    stats.red.score = 9999.0
    stats.blue.score = 8888.0
    stats.red.cp_points = 7777.0
    stats.blue.cp_points = 6666.0
    stats.red.kills = 5555
    stats.blue.kills = 4444
    stats.red.deaths = 3333
    stats.blue.deaths = 2222

    obs_b = engine._build_transformer_obs(alive_idx, pos_xy)
    tail_a = obs_a[:, config_module.RAYS_FLAT_DIM:config_module.RAYS_FLAT_DIM + config_module.RICH_BASE_DIM]
    tail_b = obs_b[:, config_module.RAYS_FLAT_DIM:config_module.RAYS_FLAT_DIM + config_module.RICH_BASE_DIM]
    assert torch.equal(tail_a, tail_b)


def test_self_centric_normalization_is_bounded_and_finite(monkeypatch) -> None:
    _reload_obs_schema(monkeypatch, "self_centric_v1")
    engine, registry, _grid, stats = make_test_engine(monkeypatch, max_agents=1)
    register_agent(registry, engine.grid, 0, team_is_red=True, x=2, y=2)

    stats.tick = 10_000_000
    engine._obs_self_kill_ppo_points[0] = 1.0e9
    engine._obs_self_damage_dealt_ppo_points[0] = 1.0e9
    engine._obs_self_cp_ppo_points[0] = 1.0e9
    engine._obs_self_damage_taken_penalty_mag[0] = 1.0e9
    engine._obs_self_death_penalty_mag[0] = 1.0e9
    engine._obs_self_kill_count[0] = 1.0e9

    alive_idx = engine._recompute_alive_idx()
    obs = engine._build_transformer_obs(alive_idx, registry.positions_xy(alive_idx))
    tail = obs[:, config_module.RAYS_FLAT_DIM:]

    assert torch.isfinite(obs).all()
    assert torch.all(tail[:, :16] >= 0.0)
    assert torch.all(tail[:, :16] <= 1.0)


def test_self_history_accumulators_track_per_slot_reward_ownership(monkeypatch) -> None:
    _reload_obs_schema(monkeypatch, "self_centric_v1")
    monkeypatch.setattr(config_module, "PPO_REWARD_KILL_INDIVIDUAL", 2.0)
    monkeypatch.setattr(config_module, "PPO_REWARD_DMG_DEALT_INDIVIDUAL", 0.5)
    monkeypatch.setattr(config_module, "PPO_PENALTY_DMG_TAKEN_INDIVIDUAL", 0.25)
    monkeypatch.setattr(config_module, "PPO_REWARD_DEATH", -1.5)

    engine, registry, _grid, _stats = make_test_engine(monkeypatch, max_agents=4)
    register_agent(registry, engine.grid, 0, team_is_red=True, x=2, y=2, atk=1.0)
    register_agent(registry, engine.grid, 1, team_is_red=False, x=3, y=2, hp=0.5, hp_max=0.5)
    monkeypatch.setenv("FWS_DEBUG_FORCE_ACTIONS", "17,0")

    engine.run_tick()

    assert engine._obs_self_kill_ppo_points[0].item() == pytest.approx(2.0)
    assert engine._obs_self_kill_count[0].item() == pytest.approx(1.0)
    assert engine._obs_self_damage_dealt_ppo_points[0].item() == pytest.approx(0.5)
    assert engine._obs_self_damage_taken_penalty_mag[1].item() == pytest.approx(0.25)
    assert engine._obs_self_death_penalty_mag[1].item() == pytest.approx(1.5)
    assert engine._obs_self_kill_ppo_points[1].item() == pytest.approx(0.0)


def test_self_history_resets_when_dead_slot_respawns(monkeypatch) -> None:
    _reload_obs_schema(monkeypatch, "self_centric_v1")
    engine, registry, _grid, _stats = make_test_engine(
        monkeypatch,
        max_agents=2,
        respawn_enabled=True,
        metabolism_enabled=False,
    )
    register_agent(registry, engine.grid, 0, team_is_red=True, x=2, y=2)
    engine._obs_self_kill_ppo_points[0] = 3.0
    engine._obs_self_damage_dealt_ppo_points[0] = 4.0
    engine._obs_self_cp_ppo_points[0] = 5.0
    engine._obs_self_damage_taken_penalty_mag[0] = 6.0
    engine._obs_self_death_penalty_mag[0] = 7.0
    engine._obs_self_kill_count[0] = 8.0

    registry.agent_data[0, COL_ALIVE] = 0.0
    was_dead = torch.zeros((registry.capacity,), dtype=torch.bool, device=CPU)
    was_dead[0] = True
    registry.agent_data[0, COL_ALIVE] = 1.0

    engine._ppo_reset_on_respawn(was_dead)

    assert engine._obs_self_kill_ppo_points[0].item() == pytest.approx(0.0)
    assert engine._obs_self_damage_dealt_ppo_points[0].item() == pytest.approx(0.0)
    assert engine._obs_self_cp_ppo_points[0].item() == pytest.approx(0.0)
    assert engine._obs_self_damage_taken_penalty_mag[0].item() == pytest.approx(0.0)
    assert engine._obs_self_death_penalty_mag[0].item() == pytest.approx(0.0)
    assert engine._obs_self_kill_count[0].item() == pytest.approx(0.0)


def test_legacy_schema_path_is_preserved(monkeypatch) -> None:
    _reload_obs_schema(monkeypatch, "legacy_full_v1")
    engine, registry, _grid, stats = make_test_engine(monkeypatch, max_agents=1)
    register_agent(registry, engine.grid, 0, team_is_red=True, x=2, y=2)

    alive_idx = engine._recompute_alive_idx()
    obs_a = engine._build_transformer_obs(alive_idx, registry.positions_xy(alive_idx))
    stats.red.score = 123.0
    obs_b = engine._build_transformer_obs(alive_idx, registry.positions_xy(alive_idx))

    tail_a = obs_a[:, config_module.RAYS_FLAT_DIM:]
    tail_b = obs_b[:, config_module.RAYS_FLAT_DIM:]
    assert config_module.RICH_BASE_DIM == 23
    assert tail_a.shape[1] == 27
    assert tail_a[0, 12].item() != tail_b[0, 12].item()


def test_checkpoint_roundtrip_preserves_self_history_and_schema(monkeypatch, tmp_path) -> None:
    _reload_obs_schema(monkeypatch, "self_centric_v1")
    engine, registry, _grid, stats = make_test_engine(monkeypatch, max_agents=4)
    agent_id = register_agent(registry, engine.grid, 0, team_is_red=True, x=3, y=3)
    stats.tick = 17
    engine.agent_scores[agent_id] = 3.25
    engine._obs_self_kill_ppo_points[0] = 1.5
    engine._obs_self_damage_dealt_ppo_points[0] = 2.5
    engine._obs_self_cp_ppo_points[0] = 3.5
    engine._obs_self_damage_taken_penalty_mag[0] = 4.5
    engine._obs_self_death_penalty_mag[0] = 5.5
    engine._obs_self_kill_count[0] = 6.5

    manager = CheckpointManager(tmp_path / "run")
    out_dir = manager.save_atomic(engine=engine, registry=registry, stats=stats)
    ckpt = CheckpointManager.load(str(out_dir))

    assert ckpt["meta"]["obs_schema"] == "self_centric_v1"
    assert ckpt["meta"]["obs_dim"] == config_module.OBS_DIM

    grid2 = ckpt["world"]["grid"].to(CPU)
    stats2 = SimulationStats()
    registry2 = registry.__class__(grid2)
    engine2 = engine.__class__(registry2, grid2, stats2, zones=None)

    CheckpointManager.apply_loaded_checkpoint(
        ckpt,
        engine=engine2,
        registry=registry2,
        stats=stats2,
        device=CPU,
    )

    assert engine2._obs_self_kill_ppo_points[0].item() == pytest.approx(1.5)
    assert engine2._obs_self_damage_dealt_ppo_points[0].item() == pytest.approx(2.5)
    assert engine2._obs_self_cp_ppo_points[0].item() == pytest.approx(3.5)
    assert engine2._obs_self_damage_taken_penalty_mag[0].item() == pytest.approx(4.5)
    assert engine2._obs_self_death_penalty_mag[0].item() == pytest.approx(5.5)
    assert engine2._obs_self_kill_count[0].item() == pytest.approx(6.5)


def test_checkpoint_apply_rejects_observation_schema_mismatch(monkeypatch, tmp_path) -> None:
    _reload_obs_schema(monkeypatch, "legacy_full_v1")
    engine, registry, _grid, stats = make_test_engine(monkeypatch, max_agents=2)
    register_agent(registry, engine.grid, 0, team_is_red=True, x=2, y=2)

    manager = CheckpointManager(tmp_path / "run")
    out_dir = manager.save_atomic(engine=engine, registry=registry, stats=stats)
    ckpt = CheckpointManager.load(str(out_dir))

    _reload_obs_schema(monkeypatch, "self_centric_v1")
    grid2 = ckpt["world"]["grid"].to(CPU)
    stats2 = SimulationStats()
    registry2 = registry.__class__(grid2)
    engine2 = engine.__class__(registry2, grid2, stats2, zones=None)

    with pytest.raises(CheckpointError, match="observation schema mismatch"):
        CheckpointManager.apply_loaded_checkpoint(
            ckpt,
            engine=engine2,
            registry=registry2,
            stats=stats2,
            device=CPU,
        )
