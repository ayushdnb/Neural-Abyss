from __future__ import annotations

import pytest
import torch

import config
from agent.mlp_brain import (
    brain_kind_from_module,
    create_mlp_brain,
    describe_brain_module,
)
from agent.obs_spec import build_semantic_tokens, split_obs_flat, split_obs_for_mlp


def test_split_obs_flat_and_mlp_preserve_contract() -> None:
    obs = torch.arange(2 * config.OBS_DIM, dtype=torch.float32).reshape(2, config.OBS_DIM)

    rays_flat, rich_base, instinct = split_obs_flat(obs)
    rays_raw, rich_vec = split_obs_for_mlp(obs)
    tokens = build_semantic_tokens(rich_base, instinct)

    assert rays_flat.shape == (2, config.RAYS_FLAT_DIM)
    assert rich_base.shape == (2, config.RICH_BASE_DIM)
    assert instinct.shape == (2, config.INSTINCT_DIM)
    assert rays_raw.shape == (2, config.RAY_TOKEN_COUNT, config.RAY_FEAT_DIM)
    assert rich_vec.shape == (2, config.RICH_TOTAL_DIM)
    assert tuple(tokens.keys()) == (
        "own_context",
        "world_context",
        "zone_context",
        "team_context",
        "combat_context",
        "instinct_context",
    )
    assert torch.equal(tokens["instinct_context"], instinct)


def test_split_obs_flat_rejects_bad_feature_dim() -> None:
    obs = torch.zeros((1, config.OBS_DIM - 1), dtype=torch.float32)

    with pytest.raises(RuntimeError, match="obs_dim mismatch"):
        split_obs_flat(obs)


def test_build_semantic_tokens_rejects_bad_instinct_shape() -> None:
    rich_base = torch.zeros((1, config.RICH_BASE_DIM), dtype=torch.float32)
    instinct = torch.zeros((2, config.INSTINCT_DIM), dtype=torch.float32)

    with pytest.raises(RuntimeError, match="instinct shape mismatch"):
        build_semantic_tokens(rich_base, instinct)


@pytest.mark.parametrize("kind", config.BRAIN_MLP_KIND_ORDER)
def test_create_mlp_brain_variants_produce_expected_shapes(kind: str) -> None:
    brain = create_mlp_brain(kind, config.OBS_DIM, config.NUM_ACTIONS)
    obs = torch.zeros((3, config.OBS_DIM), dtype=torch.float32)

    logits, values = brain(obs)

    assert logits.shape == (3, config.NUM_ACTIONS), f"{kind} logits shape drifted"
    assert values.shape == (3, 1), f"{kind} value head shape drifted"
    assert brain_kind_from_module(brain) == kind
    assert describe_brain_module(brain).strip()


def test_create_mlp_brain_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError, match="Unknown MLP brain kind"):
        create_mlp_brain("unknown", config.OBS_DIM, config.NUM_ACTIONS)
