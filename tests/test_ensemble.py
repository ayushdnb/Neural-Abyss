from __future__ import annotations

import pytest
import torch

import config
from agent import ensemble
from agent.mlp_brain import create_mlp_brain


def test_ensemble_forward_handles_empty_bucket() -> None:
    dist, values = ensemble.ensemble_forward([], torch.empty((0, config.OBS_DIM), dtype=torch.float32))

    assert dist.logits.shape == (0, 0)
    assert values.shape == (0,)


@pytest.mark.skipif(
    ensemble.functional_call is None or ensemble.vmap is None or ensemble.stack_module_state is None,
    reason="torch.func is unavailable in this environment",
)
def test_vmap_path_matches_loop_and_refreshes_after_parameter_mutation() -> None:
    torch.manual_seed(7)
    models = [create_mlp_brain("black_grail_of_nightfire", config.OBS_DIM, config.NUM_ACTIONS) for _ in range(3)]
    obs = torch.randn((3, config.OBS_DIM), dtype=torch.float32)

    ensemble.clear_stacked_vmap_cache()

    loop_dist, loop_values = ensemble._ensemble_forward_loop(models, obs)
    vmap_dist, vmap_values = ensemble._ensemble_forward_vmap(models, obs)

    assert torch.allclose(vmap_dist.logits, loop_dist.logits, atol=1e-5, rtol=1e-5)
    assert torch.allclose(vmap_values, loop_values, atol=1e-5, rtol=1e-5)

    with torch.no_grad():
        next(models[0].parameters()).add_(0.25)

    loop_after, values_after = ensemble._ensemble_forward_loop(models, obs)
    vmap_after, vmap_values_after = ensemble._ensemble_forward_vmap(models, obs)

    assert torch.allclose(vmap_after.logits, loop_after.logits, atol=1e-5, rtol=1e-5)
    assert torch.allclose(vmap_values_after, values_after, atol=1e-5, rtol=1e-5)
