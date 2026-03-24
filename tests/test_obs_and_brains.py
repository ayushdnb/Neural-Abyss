from __future__ import annotations

import importlib

import pytest
import torch

import config as config_module
from agent import obs_spec
from agent.mlp_brain import (
    brain_kind_from_module,
    create_mlp_brain,
    describe_brain_module,
    normalize_brain_kind,
)
from agent.obs_spec import build_semantic_tokens, split_obs_flat, split_obs_for_mlp


def test_split_obs_flat_and_mlp_preserve_contract() -> None:
    obs = torch.arange(2 * config_module.OBS_DIM, dtype=torch.float32).reshape(2, config_module.OBS_DIM)

    rays_flat, rich_base, instinct = split_obs_flat(obs)
    rays_raw, rich_vec = split_obs_for_mlp(obs)
    tokens = build_semantic_tokens(rich_base, instinct)

    assert config_module.OBS_SCHEMA == config_module.OBS_SCHEMA_SELF_CENTRIC_V1
    assert rays_flat.shape == (2, config_module.RAYS_FLAT_DIM)
    assert rich_base.shape == (2, config_module.RICH_BASE_DIM)
    assert instinct.shape == (2, config_module.INSTINCT_DIM)
    assert rays_raw.shape == (2, config_module.RAY_TOKEN_COUNT, config_module.RAY_FEAT_DIM)
    assert rich_vec.shape == (2, config_module.RICH_TOTAL_DIM)
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
    obs = torch.zeros((1, config_module.OBS_DIM - 1), dtype=torch.float32)

    with pytest.raises(RuntimeError, match="obs_dim mismatch"):
        split_obs_flat(obs)


def test_build_semantic_tokens_rejects_bad_instinct_shape() -> None:
    rich_base = torch.zeros((1, config_module.RICH_BASE_DIM), dtype=torch.float32)
    instinct = torch.zeros((2, config_module.INSTINCT_DIM), dtype=torch.float32)

    with pytest.raises(RuntimeError, match="instinct shape mismatch"):
        build_semantic_tokens(rich_base, instinct)


def test_validate_obs_contract_cache_rechecks_after_config_drift(monkeypatch) -> None:
    obs_spec.validate_obs_contract()
    monkeypatch.setattr(config_module, "RICH_BASE_DIM", int(config_module.RICH_BASE_DIM) + 1)

    with pytest.raises(RuntimeError, match="rich-base feature contract mismatch"):
        obs_spec.validate_obs_contract()


@pytest.mark.parametrize("kind", config_module.BRAIN_MLP_KIND_ORDER)
def test_create_mlp_brain_variants_produce_expected_shapes(kind: str) -> None:
    brain = create_mlp_brain(kind, config_module.OBS_DIM, config_module.NUM_ACTIONS)
    obs = torch.randn((3, config_module.OBS_DIM), dtype=torch.float32)

    logits, values = brain(obs)

    assert logits.shape == (3, config_module.NUM_ACTIONS), f"{kind} logits shape drifted"
    assert values.shape == (3, 1), f"{kind} value head shape drifted"
    assert brain_kind_from_module(brain) == normalize_brain_kind(kind)
    assert describe_brain_module(brain).strip()


def test_train_tiny_profile_keeps_all_brains_under_50k_params(monkeypatch) -> None:
    monkeypatch.setenv("FWS_PROFILE", "train_tiny")
    config = importlib.reload(config_module)

    for kind in config.BRAIN_MLP_KIND_ORDER:
        brain = create_mlp_brain(kind, config.OBS_DIM, config.NUM_ACTIONS)
        param_count = sum(p.numel() for p in brain.parameters())
        assert param_count <= 50_000, f"{kind} drifted above 50k params: {param_count}"


@pytest.mark.parametrize(
    ("legacy_kind", "canonical_kind"),
    [
        ("whispering_abyss", "throne_of_ashen_dreams"),
        ("veil_of_echoes", "veil_of_the_hollow_crown"),
        ("cathedral_of_ash", "throne_of_ashen_dreams"),
        ("dreamer_in_black_fog", "black_grail_of_nightfire"),
        ("obsidian_pulse", "black_grail_of_nightfire"),
    ],
)
def test_legacy_kind_aliases_normalize_and_instantiate(legacy_kind: str, canonical_kind: str) -> None:
    brain = create_mlp_brain(legacy_kind, config_module.OBS_DIM, config_module.NUM_ACTIONS)
    assert normalize_brain_kind(legacy_kind) == canonical_kind
    assert brain_kind_from_module(brain) == canonical_kind


def test_black_grail_scalar_context_changes_policy_for_same_rays() -> None:
    brain = create_mlp_brain("black_grail_of_nightfire", config_module.OBS_DIM, config_module.NUM_ACTIONS)

    obs = torch.zeros((2, config_module.OBS_DIM), dtype=torch.float32)
    ray_span = config_module.RAYS_FLAT_DIM
    obs[:, :ray_span] = 0.0
    obs[0, ray_span:] = 0.0
    obs[1, ray_span:] = 0.0
    obs[1, ray_span + 0] = 1.0
    obs[1, ray_span + 7] = 1.0
    obs[1, ray_span + 10] = 1.0

    logits, values = brain(obs)

    assert not torch.allclose(logits[0], logits[1])
    assert not torch.allclose(values[0], values[1])


def test_create_mlp_brain_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError, match="Unknown MLP brain kind"):
        create_mlp_brain("unknown", config_module.OBS_DIM, config_module.NUM_ACTIONS)


def test_legacy_schema_still_splits_but_is_not_the_default(monkeypatch) -> None:
    monkeypatch.setenv("FWS_OBS_SCHEMA", "legacy_full_v1")
    config = importlib.reload(config_module)

    obs = torch.zeros((1, config.OBS_DIM), dtype=torch.float32)
    rays_flat, rich_base, instinct = split_obs_flat(obs)

    assert config.OBS_SCHEMA == "legacy_full_v1"
    assert rich_base.shape[1] == 23
    assert instinct.shape[1] == 4
    assert rays_flat.shape[1] == config.RAYS_FLAT_DIM

    monkeypatch.delenv("FWS_OBS_SCHEMA", raising=False)
    importlib.reload(config_module)
