"""Observation slicing helpers for the agent input contract."""

from __future__ import annotations

from typing import Dict, Tuple, Optional, Any
import torch
import config

# Purpose of this module
# This file provides two closely related utilities that enforce a strict and
# stable observation schema:
#   1) split_obs_flat(obs)
#        Takes a flat observation tensor (B, OBS_DIM) and splits it into:
#          - rays_flat  : (B, RAYS_FLAT_DIM)
#          - rich_base  : (B, RICH_BASE_DIM)
#          - instinct   : (B, INSTINCT_DIM)
#   2) build_semantic_tokens(rich_base, instinct)
#        Builds a dictionary of semantic feature groups by selecting specific
#        index subsets from rich_base, plus an instinct token.
# Why strict schema enforcement matters:
# - Reinforcement learning policies are extremely sensitive to feature ordering.
# - Any silent mismatch (wrong indices, wrong slice boundaries) can ruin training
#   without causing an immediate exception.
# - This code chooses "fail loudly" behavior to prevent silent corruption.
# Performance note:
# - The semantic feature indices are cached as torch tensors per-device/schema to
#   avoid allocating a new index tensor on every forward pass / timestep.

LEGACY_FULL_V1 = getattr(config, "OBS_SCHEMA_LEGACY_FULL_V1", "legacy_full_v1")
SELF_CENTRIC_V1 = getattr(config, "OBS_SCHEMA_SELF_CENTRIC_V1", "self_centric_v1")

LEGACY_RICH_BASE_FEATURE_NAMES = (
    "self_hp_ratio",
    "self_x_norm",
    "self_y_norm",
    "self_red_team_bit",
    "self_blue_team_bit",
    "self_soldier_bit",
    "self_archer_bit",
    "self_attack_norm",
    "self_vision_norm",
    "on_heal_zone",
    "on_control_point",
    "tick_norm",
    "red_team_score_norm",
    "blue_team_score_norm",
    "red_team_cp_norm",
    "blue_team_cp_norm",
    "red_team_kills_norm",
    "blue_team_kills_norm",
    "red_team_deaths_norm",
    "blue_team_deaths_norm",
    "pad_0",
    "pad_1",
    "pad_2",
)

SELF_CENTRIC_BASE_FEATURE_NAMES = (
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
)

INSTINCT_FEATURE_NAMES = (
    "ally_archer_density",
    "ally_soldier_density",
    "noisy_enemy_density",
    "threat_ratio",
)


# Cache index tensors per (device, schema, name) to avoid per-step allocations.
# Key:
#   (torch.device, schema_name, token_name) -> LongTensor indices on that device
_IDX_CACHE: Dict[Tuple[torch.device, str, str], torch.Tensor] = {}
_VALIDATED_CONTRACT_TOKEN: Optional[Tuple[Any, ...]] = None


def obs_schema_name() -> str:
    """Return the active observation schema identifier."""
    return str(getattr(config, "OBS_SCHEMA", LEGACY_FULL_V1)).strip().lower()


def rich_base_feature_names() -> Tuple[str, ...]:
    """Return the ordered rich-base feature names for the active schema."""
    if obs_schema_name() == SELF_CENTRIC_V1:
        return SELF_CENTRIC_BASE_FEATURE_NAMES
    return LEGACY_RICH_BASE_FEATURE_NAMES


def instinct_feature_names() -> Tuple[str, ...]:
    """Return the ordered instinct feature names."""
    return INSTINCT_FEATURE_NAMES


def scalar_feature_names() -> Tuple[str, ...]:
    """Return the full ordered non-ray scalar feature names."""
    return rich_base_feature_names() + instinct_feature_names()


def _contract_validation_token() -> Tuple[Any, ...]:
    return (
        obs_schema_name(),
        int(config.RICH_BASE_DIM),
        int(config.INSTINCT_DIM),
        int(config.RICH_TOTAL_DIM),
        tuple(rich_base_feature_names()),
        tuple(instinct_feature_names()),
        tuple(
            (str(name), tuple(int(i) for i in indices))
            for name, indices in sorted(dict(getattr(config, "SEMANTIC_RICH_BASE_INDICES", {})).items())
        ),
    )


def validate_obs_contract() -> None:
    """Fail loudly if config-derived observation metadata drifted."""
    global _VALIDATED_CONTRACT_TOKEN

    token = _contract_validation_token()
    if token == _VALIDATED_CONTRACT_TOKEN:
        return

    base_names = rich_base_feature_names()
    inst_names = instinct_feature_names()
    schema = obs_schema_name()

    if len(base_names) != int(config.RICH_BASE_DIM):
        raise RuntimeError(
            f"rich-base feature contract mismatch: names={len(base_names)} config.RICH_BASE_DIM={int(config.RICH_BASE_DIM)}"
        )
    if len(inst_names) != int(config.INSTINCT_DIM):
        raise RuntimeError(
            f"instinct feature contract mismatch: names={len(inst_names)} config.INSTINCT_DIM={int(config.INSTINCT_DIM)}"
        )
    if len(scalar_feature_names()) != int(config.RICH_TOTAL_DIM):
        raise RuntimeError(
            f"scalar feature contract mismatch: names={len(scalar_feature_names())} config.RICH_TOTAL_DIM={int(config.RICH_TOTAL_DIM)}"
        )

    if schema == SELF_CENTRIC_V1:
        expected = SELF_CENTRIC_BASE_FEATURE_NAMES + INSTINCT_FEATURE_NAMES
        if scalar_feature_names() != expected:
            raise RuntimeError("self-centric scalar feature order drifted from the Phase-1 contract")
        if int(config.RICH_BASE_DIM) != 16 or int(config.INSTINCT_DIM) != 4:
            raise RuntimeError("self-centric schema dims must be rich_base=16 and instinct=4")
    elif schema == LEGACY_FULL_V1:
        expected = LEGACY_RICH_BASE_FEATURE_NAMES + INSTINCT_FEATURE_NAMES
        if scalar_feature_names() != expected:
            raise RuntimeError("legacy scalar feature order drifted from the historical contract")
    else:
        raise RuntimeError(f"unknown observation schema: {schema!r}")

    _IDX_CACHE.clear()
    _VALIDATED_CONTRACT_TOKEN = token


def _idx(name: str, device: torch.device) -> torch.Tensor:
    """
    Get cached index tensor for a semantic token by name.

    Args:
        name:
            String key identifying which semantic slice to select, e.g.:
              "own_context", "world_context", ...
            Must exist in config.SEMANTIC_RICH_BASE_INDICES.

        device:
            The torch device on which the returned index tensor must live.
            This must match rich_base.device to keep index_select valid.

    Returns:
        idx:
            1D LongTensor of indices on the requested device.

    """
    schema = obs_schema_name()
    key = (device, schema, name)
    t = _IDX_CACHE.get(key)
    if t is not None:
        return t

    validate_obs_contract()
    if name not in config.SEMANTIC_RICH_BASE_INDICES:
        raise KeyError(f"Unknown semantic token name: {name}")

    idx = torch.tensor(config.SEMANTIC_RICH_BASE_INDICES[name], dtype=torch.long, device=device)
    _IDX_CACHE[key] = idx
    return idx


def split_obs_flat(obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Split a flat observation into three components:
      rays_flat  : (B, RAYS_FLAT_DIM)
      rich_base  : (B, RICH_BASE_DIM)
      instinct   : (B, INSTINCT_DIM)

    """
    validate_obs_contract()

    if obs.dim() != 2:
        raise RuntimeError(f"obs must be rank-2 (B,F). got shape={tuple(obs.shape)}")

    B, F = int(obs.shape[0]), int(obs.shape[1])
    if F != int(config.OBS_DIM):
        raise RuntimeError(f"obs_dim mismatch: got F={F}, expected config.OBS_DIM={int(config.OBS_DIM)}")

    rays_dim = int(config.RAYS_FLAT_DIM)
    rich_total = int(config.RICH_TOTAL_DIM)
    if rays_dim + rich_total != F:
        raise RuntimeError(f"layout mismatch: rays_dim({rays_dim}) + rich_total({rich_total}) != F({F})")

    rays_flat = obs[:, :rays_dim]
    rich_tail = obs[:, rays_dim:]

    base_dim = int(config.RICH_BASE_DIM)
    inst_dim = int(config.INSTINCT_DIM)
    if base_dim + inst_dim != int(rich_tail.shape[1]):
        raise RuntimeError(
            f"rich_tail mismatch: got {int(rich_tail.shape[1])}, expected {base_dim}+{inst_dim}"
        )

    rich_base = rich_tail[:, :base_dim]
    instinct = rich_tail[:, base_dim:base_dim + inst_dim]
    return rays_flat, rich_base, instinct


def split_obs_for_mlp(obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Shared preprocessing entry point for the two-token MLP brain family.

    Returns:
        rays_raw:
            Shape (B, RAY_TOKEN_COUNT, RAY_FEAT_DIM)
            This preserves the existing ray-token interpretation exactly.

        rich_vec:
            Shape (B, RICH_BASE_DIM + INSTINCT_DIM)
            This is the full non-ray tail packed into one vector so the brain
            can project it into a single rich token.

    """
    rays_flat, rich_base, instinct = split_obs_flat(obs)

    B = int(obs.shape[0])
    num_rays = int(config.RAY_TOKEN_COUNT)
    ray_feat_dim = int(config.RAY_FEAT_DIM)
    expected_rays_flat = num_rays * ray_feat_dim

    if int(rays_flat.shape[1]) != expected_rays_flat:
        raise RuntimeError(
            f"rays_flat dim mismatch for MLP path: got {int(rays_flat.shape[1])}, "
            f"expected {expected_rays_flat} = {num_rays}*{ray_feat_dim}"
        )

    rays_raw = rays_flat.reshape(B, num_rays, ray_feat_dim)

    rich_vec = torch.cat([rich_base, instinct], dim=1)
    expected_rich = int(config.RICH_BASE_DIM) + int(config.INSTINCT_DIM)
    if int(rich_vec.shape[1]) != expected_rich:
        raise RuntimeError(
            f"rich_vec dim mismatch for MLP path: got {int(rich_vec.shape[1])}, "
            f"expected {expected_rich}"
        )

    return rays_raw, rich_vec


def build_semantic_tokens(
    rich_base: torch.Tensor,
    instinct: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Build semantic token tensors from rich_base and instinct components.

    """
    validate_obs_contract()

    if rich_base.dim() != 2:
        raise RuntimeError(f"rich_base must be (B,D). got {tuple(rich_base.shape)}")
    if instinct.dim() != 2:
        raise RuntimeError(f"instinct must be (B,4). got {tuple(instinct.shape)}")

    B = int(rich_base.shape[0])
    if int(rich_base.shape[1]) != int(config.RICH_BASE_DIM):
        raise RuntimeError(
            f"rich_base dim mismatch: got {int(rich_base.shape[1])}, expected {int(config.RICH_BASE_DIM)}"
        )

    if int(instinct.shape[0]) != B or int(instinct.shape[1]) != int(config.INSTINCT_DIM):
        raise RuntimeError(
            f"instinct shape mismatch: got {tuple(instinct.shape)}, expected ({B},{int(config.INSTINCT_DIM)})"
        )

    device = rich_base.device
    out: Dict[str, torch.Tensor] = {}
    for name in ("own_context", "world_context", "zone_context", "team_context", "combat_context"):
        idx = _idx(name, device)
        tok = torch.index_select(rich_base, dim=1, index=idx)
        out[name] = tok

    out["instinct_context"] = instinct
    return out
