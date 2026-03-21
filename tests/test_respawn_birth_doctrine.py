from __future__ import annotations

from types import SimpleNamespace

import torch

from engine.agent_registry import COL_ALIVE, COL_HP, COL_HP_MAX, NUM_COLS, TEAM_RED_ID
import engine.respawn as respawn


def _make_cfg(**overrides):
    base = dict(
        parent_selection_mode="topk_weighted",
        parent_selection_topk_frac=1.0,
        parent_selection_score_power=1.0,
        birth_doctrine_mode="overall",
        birth_random_doctrine_pool=("overall",),
        birth_topk_size=1,
        birth_zero_score_fallback="uniform_candidates",
        birth_blend_weight_kill=1.0,
        birth_blend_weight_cp=1.0,
        birth_blend_weight_health=1.0,
        birth_blend_weight_personal=1.0,
        require_parent_for_birth=True,
        clone_prob=1.0,
        spawn_tries=1,
        wall_margin=0,
        unit_soldier=1,
        unit_archer=2,
        spawn_archer_ratio=0.0,
        soldier_hp=1.0,
        soldier_atk=0.1,
        archer_hp=1.0,
        archer_atk=0.1,
        vision_soldier=1,
        vision_archer=1,
        mutation_std=0.0,
        spawn_location_mode="uniform",
        spawn_near_parent_radius=1,
        child_unit_mode="random",
        rare_mutation_tick_window_enable=False,
        rare_mutation_physical_enable=False,
        rare_mutation_inherited_brain_noise_enable=False,
        rare_mutation_inherited_brain_noise_std=0.0,
        rare_mutation_physical_drift_std_frac=0.0,
        rare_mutation_physical_drift_clip_frac=0.0,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _make_parent_state(hp_pairs):
    data = torch.zeros((len(hp_pairs), NUM_COLS), dtype=torch.float32)
    for i, (hp, hp_max) in enumerate(hp_pairs):
        data[i, COL_HP] = float(hp)
        data[i, COL_HP_MAX] = float(hp_max)
    parents = torch.tensor(list(range(len(hp_pairs))), dtype=torch.long)
    reg = SimpleNamespace(
        agent_uids=torch.tensor([100 + i for i in range(len(hp_pairs))], dtype=torch.int64)
    )
    return reg, data, parents


def test_pick_parent_slot_killer_uses_lifetime_kill_counts():
    cfg = _make_cfg(birth_doctrine_mode="killer", birth_topk_size=1)
    reg, data, parents = _make_parent_state([(1.0, 1.0), (1.0, 1.0)])
    engine = SimpleNamespace(
        agent_kill_counts={100: 0.0, 101: 5.0},
        agent_cp_points={},
        agent_scores={},
    )

    slot, doctrine, score = respawn._pick_parent_slot(parents, reg, data, cfg, engine=engine)

    assert slot == 1
    assert doctrine == "killer"
    assert score == 1.0


def test_pick_parent_slot_random_per_birth_singleton_pool_resolves_deterministically():
    cfg = _make_cfg(
        birth_doctrine_mode="random_per_birth",
        birth_random_doctrine_pool=("health",),
        birth_topk_size=1,
    )
    reg, data, parents = _make_parent_state([(0.25, 1.0), (1.0, 1.0)])
    engine = SimpleNamespace(agent_kill_counts={}, agent_cp_points={}, agent_scores={})

    slot, doctrine, score = respawn._pick_parent_slot(parents, reg, data, cfg, engine=engine)

    assert slot == 1
    assert doctrine == "health"
    assert score == 1.0


def test_pick_parent_slot_all_zero_scores_uses_uniform_candidates_fallback(monkeypatch):
    cfg = _make_cfg(
        birth_doctrine_mode="killer",
        birth_topk_size=2,
        birth_zero_score_fallback="uniform_candidates",
    )
    reg, data, parents = _make_parent_state([(1.0, 1.0), (1.0, 1.0)])
    engine = SimpleNamespace(agent_kill_counts={}, agent_cp_points={}, agent_scores={})

    monkeypatch.setattr(respawn.random, "randrange", lambda n: 1)

    slot, doctrine, score = respawn._pick_parent_slot(parents, reg, data, cfg, engine=engine)

    assert slot == 1
    assert doctrine == "killer"
    assert score == 0.0


class _DummyRegistry:
    def __init__(self):
        self.device = torch.device("cpu")
        self.agent_data = torch.zeros((3, NUM_COLS), dtype=torch.float32)
        self.brains = [None] * 3
        self.generations = [0] * 3


def test_respawn_some_closed_cradle_blocks_parentless_birth():
    reg = _DummyRegistry()
    grid = torch.zeros((3, 5, 5), dtype=torch.float32)
    grid[2].fill_(-1.0)

    cfg = _make_cfg(require_parent_for_birth=True)

    spawned = respawn._respawn_some(
        reg,
        grid,
        TEAM_RED_ID,
        1,
        cfg,
        tick=1,
        engine=None,
    )

    assert spawned == 0
    assert float(reg.agent_data[:, COL_ALIVE].sum().item()) == 0.0
