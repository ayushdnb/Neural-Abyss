"""Observation slicing helpers for the agent input contract."""

from __future__ import annotations

from typing import Dict, Tuple
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
# - The semantic feature indices are cached as torch tensors per-device to avoid
#   allocating a new index tensor on every forward pass / timestep.


# Cache index tensors per (device, name) to avoid per-step allocations.
# Key:
#   (torch.device, token_name) -> LongTensor indices on that device
# This is important because:
# - torch.index_select expects an index tensor on the same device as the source.
# - Re-creating index tensors inside the simulation loop creates avoidable
#   allocation overhead and can fragment GPU memory over long runs.
_IDX_CACHE: Dict[Tuple[torch.device, str], torch.Tensor] = {}


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

    Implementation details:
    - We cache the resulting tensor because:
        * Converting Python lists -> torch tensors is a runtime allocation.
        * Doing this per-step in an RL simulation would be unnecessarily expensive.
    - The cache is keyed by (device, name) because:
        * The same indices must exist separately on CPU vs GPU.
        * The same logical token name maps to the same index list.
    """
    key = (device, name)
    t = _IDX_CACHE.get(key)
    if t is not None:
        return t

    # Validate semantic token name.
    # This prevents silent bugs where a typo would produce an empty or incorrect slice.
    if name not in config.SEMANTIC_RICH_BASE_INDICES:
        raise KeyError(f"Unknown semantic token name: {name}")

    # Convert the configured index list to a LongTensor on the correct device.
    # dtype=torch.long is required for indexing operations.
    idx = torch.tensor(config.SEMANTIC_RICH_BASE_INDICES[name], dtype=torch.long, device=device)

    # Save to cache for future calls.
    _IDX_CACHE[key] = idx
    return idx


def split_obs_flat(obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Split a flat observation into three components:
      rays_flat  : (B, RAYS_FLAT_DIM)
      rich_base  : (B, RICH_BASE_DIM)
      instinct   : (B, INSTINCT_DIM)

    This function performs strict shape and layout checking.

    Args:
        obs:
            A 2D tensor shaped (B, F), where:
              B = batch size
              F = feature dimension, expected to equal config.OBS_DIM

    Returns:
        rays_flat:
            obs[:, :RAYS_FLAT_DIM]
            Shape (B, RAYS_FLAT_DIM)

        rich_base:
            First portion of the "rich tail" after rays.
            Shape (B, RICH_BASE_DIM)

        instinct:
            Last portion of the "rich tail".
            Shape (B, INSTINCT_DIM)

    Layout definition (conceptual):
        obs = [ rays_flat | rich_base | instinct ]

    Where:
        RAYS_FLAT_DIM  = config.RAYS_FLAT_DIM
        RICH_TOTAL_DIM = config.RICH_TOTAL_DIM = RICH_BASE_DIM + INSTINCT_DIM

    Why the checks exist:
    - In RL, a single off-by-one slicing error can poison training.
    - These checks turn schema drift into an immediate runtime error.
    """
    # Enforce rank-2: (B, F). Any other rank indicates an upstream bug.
    if obs.dim() != 2:
        raise RuntimeError(f"obs must be rank-2 (B,F). got shape={tuple(obs.shape)}")

    # Read dimensions as Python ints for clarity and to avoid tensor->int surprises.
    B, F = int(obs.shape[0]), int(obs.shape[1])

    # OBS_DIM is the canonical configured total observation size.
    if F != int(config.OBS_DIM):
        raise RuntimeError(f"obs_dim mismatch: got F={F}, expected config.OBS_DIM={int(config.OBS_DIM)}")

    # Rays and rich tail sizes must sum to total feature count.
    rays_dim = int(config.RAYS_FLAT_DIM)
    rich_total = int(config.RICH_TOTAL_DIM)
    if rays_dim + rich_total != F:
        raise RuntimeError(f"layout mismatch: rays_dim({rays_dim}) + rich_total({rich_total}) != F({F})")

    # Split observation into rays and the remaining tail.
    # rays_flat: first segment
    # rich_tail: remainder segment
    rays_flat = obs[:, :rays_dim]
    rich_tail = obs[:, rays_dim:]

    # The rich tail itself is split into a base segment and the instinct segment.
    base_dim = int(config.RICH_BASE_DIM)
    inst_dim = int(config.INSTINCT_DIM)

    # Validate that the tail length matches base + instinct.
    if base_dim + inst_dim != int(rich_tail.shape[1]):
        raise RuntimeError(
            f"rich_tail mismatch: got {int(rich_tail.shape[1])}, expected {base_dim}+{inst_dim}"
        )

    # Extract the two segments.
    rich_base = rich_tail[:, :base_dim]
    instinct = rich_tail[:, base_dim:base_dim + inst_dim]
    return rays_flat, rich_base, instinct


def split_obs_for_mlp(obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Shared preprocessing entry point for the new two-token MLP brain family.

    Returns:
        rays_raw:
            Shape (B, RAY_TOKEN_COUNT, RAY_FEAT_DIM)
            This preserves the existing ray-token interpretation exactly.

        rich_vec:
            Shape (B, RICH_BASE_DIM + INSTINCT_DIM)
            This is the full non-ray tail packed into one vector so the brain
            can project it into a single rich token.

    Design intent:
    - Keep the observation schema authoritative in one place.
    - Do NOT duplicate hard-coded slicing logic inside each brain variant.
    - Do NOT change feature ordering or semantic meaning.
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

    # reshape is used instead of view so the helper is robust even if the input
    # tensor is non-contiguous.
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

    Inputs:
        rich_base:
            Shape (B, RICH_BASE_DIM).
            This is the dense feature block that contains many different feature groups.

        instinct:
            Shape (B, INSTINCT_DIM).
            This is a separate compact block meant to be treated as its own token.

    Output:
        A dictionary mapping token name -> tensor, containing:
          - "own_context"
          - "world_context"
          - "zone_context"
          - "team_context"
          - "combat_context"
          - "instinct_context"

        Each semantic tensor has shape (B, token_dim), where token_dim depends on how
        many indices are assigned to that token in config.SEMANTIC_RICH_BASE_INDICES.

        All outputs reside on the same device as rich_base (and therefore instinct).

    Conceptual purpose:
    - rich_base is a flat vector but it contains structured information.
    - config.SEMANTIC_RICH_BASE_INDICES defines which columns correspond to which
      semantic group.
    - This function materializes those groups explicitly by selecting columns.

    Implementation notes:
    - torch.index_select is used to select columns by index:
        tok = index_select(rich_base, dim=1, index=idx)
      where idx is a 1D LongTensor of column indices.

    Correctness notes:
    - This function enforces strict dimensional checks.
    - It also ensures batch dimension consistency across rich_base and instinct.
    """
    # Validate rank-2 input: (B, D)
    if rich_base.dim() != 2:
        raise RuntimeError(f"rich_base must be (B,D). got {tuple(rich_base.shape)}")
    if instinct.dim() != 2:
        raise RuntimeError(f"instinct must be (B,4). got {tuple(instinct.shape)}")

    B = int(rich_base.shape[0])

    # Enforce configured rich_base width.
    if int(rich_base.shape[1]) != int(config.RICH_BASE_DIM):
        raise RuntimeError(
            f"rich_base dim mismatch: got {int(rich_base.shape[1])}, expected {int(config.RICH_BASE_DIM)}"
        )

    # Enforce instinct alignment:
    # - same batch size as rich_base
    # - expected instinct feature width
    if int(instinct.shape[0]) != B or int(instinct.shape[1]) != int(config.INSTINCT_DIM):
        raise RuntimeError(
            f"instinct shape mismatch: got {tuple(instinct.shape)}, expected ({B},{int(config.INSTINCT_DIM)})"
        )

    device = rich_base.device
    out: Dict[str, torch.Tensor] = {}

    # Extract each semantic token from rich_base using configured indices.
    # Each token is a column selection, not a learned projection.
    # If a token name is not configured, _idx will raise KeyError.
    for name in ("own_context", "world_context", "zone_context", "team_context", "combat_context"):
        idx = _idx(name, device)

        # tok has shape (B, len(idx)).
        # This selects specified feature columns for all batch elements.
        tok = torch.index_select(rich_base, dim=1, index=idx)
        out[name] = tok

    # Add instinct token directly (no indexing required).
    out["instinct_context"] = instinct
    return out
