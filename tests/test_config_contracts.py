from __future__ import annotations

import importlib
import os

import pytest

import config as config_module


@pytest.fixture(autouse=True)
def _restore_config_after_test() -> None:
    yield
    importlib.reload(config_module)


def _reload_config(monkeypatch, **env):
    for key in list(os.environ):
        if key.startswith("FWS_"):
            monkeypatch.delenv(key, raising=False)

    for key, value in env.items():
        monkeypatch.setenv(key, str(value))

    return importlib.reload(config_module)


def test_invalid_env_values_warn_and_fallback(monkeypatch) -> None:
    config = _reload_config(monkeypatch, FWS_GRID_W="bad-width", FWS_UI="unclear")

    assert config.GRID_WIDTH == 100
    assert config.ENABLE_UI is True
    assert any("Invalid int env FWS_GRID_W" in msg for msg in config.config_warnings())
    assert any("Unknown boolean env FWS_UI" in msg for msg in config.config_warnings())


def test_invalid_ray_token_count_warns_and_falls_back_to_supported_runtime(monkeypatch) -> None:
    config = _reload_config(monkeypatch, FWS_RAY_TOKENS="64")

    assert config.RAY_TOKEN_COUNT == 32
    assert config.RAYS_FLAT_DIM == 256
    assert any("RAY_TOKEN_COUNT currently supports only 32 first-hit rays" in msg for msg in config.config_warnings())


def test_config_strict_rejects_invalid_profile(monkeypatch) -> None:
    with pytest.raises(ValueError, match="PROFILE"):
        _reload_config(monkeypatch, FWS_CONFIG_STRICT="1", FWS_PROFILE="broken")


def test_config_strict_rejects_unsupported_ray_token_count(monkeypatch) -> None:
    with pytest.raises(ValueError, match="RAY_TOKEN_COUNT"):
        _reload_config(monkeypatch, FWS_CONFIG_STRICT="1", FWS_RAY_TOKENS="64")


def test_profile_override_respects_explicit_env_precedence(monkeypatch) -> None:
    config = _reload_config(monkeypatch, FWS_PROFILE="debug", FWS_GRID_W="123")

    assert config.GRID_WIDTH == 123
    assert config.GRID_HEIGHT == 80
    assert config.START_AGENTS_PER_TEAM == 30


def test_debug_profile_keeps_grouped_vmap_enabled(monkeypatch) -> None:
    config = _reload_config(monkeypatch, FWS_PROFILE="debug")

    assert config.USE_VMAP is True


def test_train_tiny_profile_applies_compact_brain_preset(monkeypatch) -> None:
    config = _reload_config(monkeypatch, FWS_PROFILE="train_tiny")

    assert config.ENABLE_UI is False
    assert config.USE_VMAP is True
    assert config.BRAIN_MLP_RAY_WIDTH == 48
    assert config.BRAIN_MLP_SCALAR_WIDTH == 32
    assert config.BRAIN_MLP_FUSION_WIDTH == 64
    assert config.BRAIN_MLP_RAY_DEPTH == 1
    assert config.BRAIN_MLP_SCALAR_DEPTH == 1
    assert config.BRAIN_MLP_TRUNK_DEPTH == 1
    assert config.BRAIN_MLP_GATE_HIDDEN_WIDTH == 16
    assert config.BRAIN_MLP_FINAL_INPUT_WIDTH == 80


def test_dump_config_dict_includes_warning_snapshot(monkeypatch) -> None:
    config = _reload_config(monkeypatch, FWS_GRID_W="not-an-int")

    dumped = config.dump_config_dict()

    assert dumped["GRID_WIDTH"] == 100
    assert dumped["TORCH_DEVICE"] == str(config.TORCH_DEVICE)
    assert dumped["CONFIG_WARNINGS"] == list(config.config_warnings())


@pytest.mark.parametrize(
    ("env_key", "env_value", "attr_name", "expected"),
    [
        ("FWS_TEAM_BRAIN_MODE", "broken", "TEAM_BRAIN_ASSIGNMENT_MODE", "mix"),
        ("FWS_TEAM_BRAIN_MIX_STRATEGY", "broken", "TEAM_BRAIN_MIX_STRATEGY", "random"),
        ("FWS_RESPAWN_CHILD_UNIT_MODE", "broken", "RESPAWN_CHILD_UNIT_MODE", "inherit_parent_on_clone"),
        ("FWS_BRAIN", "broken", "BRAIN_KIND", "throne_of_ashen_dreams"),
        ("FWS_TEAM_BRAIN_RED", "broken", "TEAM_BRAIN_EXCLUSIVE_RED", "throne_of_ashen_dreams"),
        ("FWS_TEAM_BRAIN_BLUE", "broken", "TEAM_BRAIN_EXCLUSIVE_BLUE", "veil_of_the_hollow_crown"),
    ],
)
def test_invalid_runtime_knobs_warn_and_fallback(monkeypatch, env_key: str, env_value: str, attr_name: str, expected: str) -> None:
    config = _reload_config(monkeypatch, **{env_key: env_value})

    assert getattr(config, attr_name) == expected
    assert any(attr_name in msg for msg in config.config_warnings())


@pytest.mark.parametrize(
    ("env_key", "env_value", "match"),
    [
        ("FWS_TEAM_BRAIN_MODE", "broken", "TEAM_BRAIN_ASSIGNMENT_MODE"),
        ("FWS_TEAM_BRAIN_MIX_STRATEGY", "broken", "TEAM_BRAIN_MIX_STRATEGY"),
        ("FWS_RESPAWN_CHILD_UNIT_MODE", "broken", "RESPAWN_CHILD_UNIT_MODE"),
        ("FWS_BRAIN", "broken", "BRAIN_KIND"),
    ],
)
def test_config_strict_rejects_invalid_runtime_knobs(monkeypatch, env_key: str, env_value: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _reload_config(monkeypatch, FWS_CONFIG_STRICT="1", **{env_key: env_value})


def test_team_brain_mix_sequence_drops_unknown_aliases_but_keeps_known_order(monkeypatch) -> None:
    config = _reload_config(
        monkeypatch,
        FWS_TEAM_BRAIN_MIX_SEQUENCE="bad_kind, veil_of_echoes , black_grail_of_nightfire",
    )

    assert config.TEAM_BRAIN_MIX_SEQUENCE == (
        "veil_of_the_hollow_crown",
        "black_grail_of_nightfire",
    )
    assert any("TEAM_BRAIN_MIX_SEQUENCE contains unknown brain kinds" in msg for msg in config.config_warnings())


def test_team_brain_mix_sequence_falls_back_when_all_entries_are_unknown(monkeypatch) -> None:
    config = _reload_config(monkeypatch, FWS_TEAM_BRAIN_MIX_SEQUENCE="bad_kind, still_bad")

    assert config.TEAM_BRAIN_MIX_SEQUENCE == config.BRAIN_MLP_KIND_ORDER
    assert any("TEAM_BRAIN_MIX_SEQUENCE must contain at least one known brain kind" in msg for msg in config.config_warnings())
