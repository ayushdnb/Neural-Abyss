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

    assert config.GRID_WIDTH == 128
    assert config.ENABLE_UI is True
    assert any("Invalid int env FWS_GRID_W" in msg for msg in config.config_warnings())
    assert any("Unknown boolean env FWS_UI" in msg for msg in config.config_warnings())


def test_config_strict_rejects_invalid_profile(monkeypatch) -> None:
    with pytest.raises(ValueError, match="PROFILE"):
        _reload_config(monkeypatch, FWS_CONFIG_STRICT="1", FWS_PROFILE="broken")


def test_profile_override_respects_explicit_env_precedence(monkeypatch) -> None:
    config = _reload_config(monkeypatch, FWS_PROFILE="debug", FWS_GRID_W="123")

    assert config.GRID_WIDTH == 123
    assert config.GRID_HEIGHT == 80
    assert config.START_AGENTS_PER_TEAM == 30


def test_dump_config_dict_includes_warning_snapshot(monkeypatch) -> None:
    config = _reload_config(monkeypatch, FWS_GRID_W="not-an-int")

    dumped = config.dump_config_dict()

    assert dumped["GRID_WIDTH"] == 128
    assert dumped["TORCH_DEVICE"] == str(config.TORCH_DEVICE)
    assert dumped["CONFIG_WARNINGS"] == list(config.config_warnings())
