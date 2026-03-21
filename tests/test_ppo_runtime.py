from __future__ import annotations

import math

import pytest
import torch

import config
from rl.ppo_runtime import PerAgentPPORuntime
from tests._sim_helpers import CPU, configure_cpu_runtime, register_agent
from engine.agent_registry import AgentsRegistry
from engine.grid import make_grid


def _make_runtime(
    monkeypatch,
    *,
    agent_slots: int = 1,
    window_ticks: int = 2,
    epochs: int = 1,
    minibatches: int = 1,
) -> tuple[PerAgentPPORuntime, AgentsRegistry]:
    configure_cpu_runtime(monkeypatch, max_agents=max(4, agent_slots + 1), ppo_enabled=True)
    monkeypatch.setattr(config, "PPO_WINDOW_TICKS", int(window_ticks))
    monkeypatch.setattr(config, "PPO_EPOCHS", int(epochs))
    monkeypatch.setattr(config, "PPO_MINIBATCHES", int(minibatches))
    monkeypatch.setattr(config, "PPO_TARGET_KL", 0.0)
    monkeypatch.setattr(config, "TELEMETRY_LOG_PPO", True, raising=False)

    grid = make_grid(CPU)
    registry = AgentsRegistry(grid)
    for slot in range(int(agent_slots)):
        register_agent(
            registry,
            grid,
            slot,
            team_is_red=(slot % 2) == 0,
            x=2 + slot,
            y=2,
        )

    runtime = PerAgentPPORuntime(registry, CPU, config.OBS_DIM, config.NUM_ACTIONS)
    return runtime, registry


def _copy_brain_state(src: AgentsRegistry, dst: AgentsRegistry) -> None:
    for src_brain, dst_brain in zip(src.brains, dst.brains):
        if src_brain is None:
            assert dst_brain is None
            continue
        assert dst_brain is not None
        dst_brain.load_state_dict(src_brain.state_dict())


def _build_rollout_batch(
    runtime: PerAgentPPORuntime,
    registry: AgentsRegistry,
    *,
    step_idx: int,
) -> tuple[torch.Tensor, ...]:
    agent_ids = torch.tensor([0, 1], dtype=torch.long, device=CPU)
    base = torch.linspace(-1.0, 1.0, config.OBS_DIM, dtype=torch.float32, device=CPU)
    obs = torch.stack(
        [
            torch.roll(base, shifts=(step_idx + aid) % max(1, int(config.OBS_DIM)))
            + float(aid) * 0.05
            + float(step_idx) * 0.001
            for aid in agent_ids.tolist()
        ]
    )

    logits_parts = []
    value_parts = []
    for row_i, aid in enumerate(agent_ids.tolist()):
        model = registry.brains[int(aid)]
        assert model is not None
        with torch.no_grad():
            head, value = model(obs[row_i : row_i + 1])
        logits = head.logits if hasattr(head, "logits") else head
        logits_parts.append(logits.detach().to(torch.float32))
        value_parts.append(value.detach().reshape(-1).to(torch.float32))

    logits = torch.cat(logits_parts, dim=0)
    values = torch.cat(value_parts, dim=0)

    action_masks = torch.ones((2, config.NUM_ACTIONS), dtype=torch.bool, device=CPU)
    action_masks[0, 1::3] = False
    action_masks[1, 2::4] = False

    masked_logits = runtime._mask_logits(logits, action_masks)
    actions = masked_logits.argmax(dim=-1)
    rewards = torch.tensor(
        [
            ((step_idx % 7) - 3) / 7.0,
            (((step_idx + 2) % 9) - 4) / 9.0,
        ],
        dtype=torch.float32,
        device=CPU,
    )
    done = torch.tensor(
        [
            (step_idx % 11) == 5,
            (step_idx % 13) == 7,
        ],
        dtype=torch.bool,
        device=CPU,
    )
    return agent_ids, obs, logits, values, actions, rewards, done, action_masks


def _drive_runtime(
    runtime: PerAgentPPORuntime,
    registry: AgentsRegistry,
    *,
    start_step: int,
    steps: int,
) -> None:
    for step_idx in range(int(start_step), int(start_step + steps)):
        agent_ids, obs, logits, values, actions, rewards, done, action_masks = _build_rollout_batch(
            runtime,
            registry,
            step_idx=step_idx,
        )
        runtime.update_value_cache(agent_ids, values)
        runtime.finalize_pending_window_from_cache()
        runtime.record_step(
            agent_ids=agent_ids,
            obs=obs,
            logits=logits,
            values=values,
            actions=actions,
            rewards=rewards,
            done=done,
            action_masks=action_masks,
            bootstrap_values=None,
        )


def _finalize_pending_window(runtime: PerAgentPPORuntime, registry: AgentsRegistry, *, next_step: int) -> None:
    if not runtime.has_pending_window_bootstrap():
        return
    agent_ids, _obs, _logits, values, _actions, _rewards, _done, _action_masks = _build_rollout_batch(
        runtime,
        registry,
        step_idx=next_step,
    )
    runtime.update_value_cache(agent_ids, values)
    assert runtime.finalize_pending_window_from_cache() is True


def _assert_summary_is_finite(summary: dict[str, float]) -> None:
    for key, value in summary.items():
        assert math.isfinite(float(value)), f"expected finite PPO summary field {key}, got {value!r}"


def test_record_step_rejects_action_masks_with_no_legal_actions(monkeypatch) -> None:
    runtime, _registry = _make_runtime(monkeypatch)

    with pytest.raises(RuntimeError, match="no legal actions"):
        runtime.record_step(
            agent_ids=torch.tensor([0], dtype=torch.long),
            obs=torch.zeros((1, config.OBS_DIM), dtype=torch.float32),
            logits=torch.zeros((1, config.NUM_ACTIONS), dtype=torch.float32),
            values=torch.zeros((1,), dtype=torch.float32),
            actions=torch.zeros((1,), dtype=torch.long),
            rewards=torch.zeros((1,), dtype=torch.float32),
            done=torch.zeros((1,), dtype=torch.bool),
            action_masks=torch.zeros((1, config.NUM_ACTIONS), dtype=torch.bool),
        )


def test_pending_window_bootstrap_finalizes_from_value_cache(monkeypatch) -> None:
    runtime, _registry = _make_runtime(monkeypatch)
    agent_ids = torch.tensor([0], dtype=torch.long)
    obs = torch.zeros((1, config.OBS_DIM), dtype=torch.float32)
    logits = torch.zeros((1, config.NUM_ACTIONS), dtype=torch.float32)
    values = torch.zeros((1,), dtype=torch.float32)
    actions = torch.zeros((1,), dtype=torch.long)
    rewards = torch.tensor([0.5], dtype=torch.float32)
    done = torch.zeros((1,), dtype=torch.bool)
    action_masks = torch.ones((1, config.NUM_ACTIONS), dtype=torch.bool)

    runtime.record_step(agent_ids, obs, logits, values, actions, rewards, done, action_masks=action_masks)
    runtime.record_step(agent_ids, obs, logits, values, actions, rewards, done, action_masks=action_masks)

    assert runtime.has_pending_window_bootstrap() is True

    runtime.update_value_cache(agent_ids, torch.tensor([0.25], dtype=torch.float32))

    assert runtime.finalize_pending_window_from_cache() is True
    assert runtime.has_pending_window_bootstrap() is False
    assert 0 not in runtime._buf or len(runtime._buf[0].obs) == 0


def test_checkpoint_state_roundtrip_restores_rollout_buffers(monkeypatch) -> None:
    runtime, registry = _make_runtime(monkeypatch)
    agent_ids = torch.tensor([0], dtype=torch.long)
    obs = torch.ones((1, config.OBS_DIM), dtype=torch.float32)
    logits = torch.zeros((1, config.NUM_ACTIONS), dtype=torch.float32)
    values = torch.zeros((1,), dtype=torch.float32)
    actions = torch.zeros((1,), dtype=torch.long)
    rewards = torch.tensor([1.0], dtype=torch.float32)
    done = torch.zeros((1,), dtype=torch.bool)
    action_masks = torch.ones((1, config.NUM_ACTIONS), dtype=torch.bool)

    runtime.record_step(agent_ids, obs, logits, values, actions, rewards, done, action_masks=action_masks)
    state = runtime.get_checkpoint_state()

    restored = PerAgentPPORuntime(registry, CPU, config.OBS_DIM, config.NUM_ACTIONS)
    restored.load_checkpoint_state(state, registry=registry, device=CPU)

    assert restored._step == 1
    assert 0 in restored._buf
    assert len(restored._buf[0].obs) == 1
    assert restored._buf[0].act_mask[0].shape == (config.NUM_ACTIONS,)


def test_long_horizon_soak_produces_finite_training_summary(monkeypatch) -> None:
    runtime, registry = _make_runtime(
        monkeypatch,
        agent_slots=2,
        window_ticks=4,
        epochs=2,
        minibatches=2,
    )

    _drive_runtime(runtime, registry, start_step=0, steps=64)
    _finalize_pending_window(runtime, registry, next_step=64)

    assert runtime.has_pending_window_bootstrap() is False
    assert runtime.last_train_summary is not None
    assert int(runtime._train_update_seq) == 16
    assert float(runtime.last_train_summary["trained_slots"]) == 2.0
    assert float(runtime.last_train_summary["optimizer_steps"]) > 0.0
    _assert_summary_is_finite(runtime.last_train_summary)

    for aid in (0, 1):
        assert aid not in runtime._buf or len(runtime._buf[aid].obs) == 0
        brain = registry.brains[aid]
        assert brain is not None
        for param in brain.parameters():
            assert torch.isfinite(param).all(), f"non-finite PPO parameter detected for slot {aid}"


def test_checkpoint_resume_matches_continuous_training_over_pending_boundary(monkeypatch) -> None:
    original_rng = torch.random.get_rng_state()
    try:
        torch.manual_seed(12345)
        init_rng = torch.random.get_rng_state()

        torch.random.set_rng_state(init_rng.clone())
        runtime_cont, registry_cont = _make_runtime(
            monkeypatch,
            agent_slots=2,
            window_ticks=4,
            epochs=2,
            minibatches=2,
        )

        torch.random.set_rng_state(init_rng.clone())
        runtime_split, registry_split = _make_runtime(
            monkeypatch,
            agent_slots=2,
            window_ticks=4,
            epochs=2,
            minibatches=2,
        )

        rng_after_make = torch.random.get_rng_state().clone()

        torch.random.set_rng_state(rng_after_make.clone())
        _drive_runtime(runtime_cont, registry_cont, start_step=0, steps=32)
        _finalize_pending_window(runtime_cont, registry_cont, next_step=32)

        torch.random.set_rng_state(rng_after_make.clone())
        _drive_runtime(runtime_split, registry_split, start_step=0, steps=16)
        assert runtime_split.has_pending_window_bootstrap() is True
        split_state = runtime_split.get_checkpoint_state()
        rng_after_split = torch.random.get_rng_state().clone()

        runtime_resume, registry_resume = _make_runtime(
            monkeypatch,
            agent_slots=2,
            window_ticks=4,
            epochs=2,
            minibatches=2,
        )
        _copy_brain_state(registry_split, registry_resume)
        runtime_resume.load_checkpoint_state(split_state, registry=registry_resume, device=CPU)

        torch.random.set_rng_state(rng_after_split.clone())
        _drive_runtime(runtime_resume, registry_resume, start_step=16, steps=16)
        _finalize_pending_window(runtime_resume, registry_resume, next_step=32)

        assert runtime_resume.has_pending_window_bootstrap() is False
        assert runtime_resume._step == runtime_cont._step
        assert runtime_resume._train_update_seq == runtime_cont._train_update_seq
        assert runtime_resume.last_train_summary is not None
        assert runtime_cont.last_train_summary is not None
        _assert_summary_is_finite(runtime_resume.last_train_summary)
        _assert_summary_is_finite(runtime_cont.last_train_summary)

        for aid in (0, 1):
            cont_brain = registry_cont.brains[aid]
            resumed_brain = registry_resume.brains[aid]
            assert cont_brain is not None and resumed_brain is not None
            cont_state = cont_brain.state_dict()
            resumed_state = resumed_brain.state_dict()
            assert cont_state.keys() == resumed_state.keys()
            for key in cont_state:
                assert torch.allclose(
                    cont_state[key],
                    resumed_state[key],
                    atol=1e-6,
                    rtol=1e-6,
                ), f"PPO resume drift detected for slot {aid} param {key}"
    finally:
        torch.random.set_rng_state(original_rng)
