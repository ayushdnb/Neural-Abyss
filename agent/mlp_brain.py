"""MLP policy/value networks used by agent slots."""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import config
from . import obs_spec


def _activation_fn(x: torch.Tensor) -> torch.Tensor:
    """
    Select and apply the hidden activation function from configuration.

    Supported activation names:
    - "relu"
    - "silu"
    - anything else defaults to "gelu"

    Why this helper exists:
    - It centralizes activation choice so all custom residual/gated blocks
      obey one config knob.
    - It avoids repeating activation-selection logic throughout the file.

    Mathematical role:
    - Introduces non-linearity so stacked linear layers can represent richer
      functions than a single affine map.
    """
    act = str(getattr(config, "BRAIN_MLP_ACTIVATION", "gelu")).strip().lower()
    if act == "relu":
        return F.relu(x)
    if act == "silu":
        return F.silu(x)
    return F.gelu(x)


def _maybe_norm(dim: int) -> nn.Module:
    """
    Return the normalization module selected by configuration.

    Current behavior:
    - "none"      -> Identity (no normalization)
    - otherwise   -> LayerNorm(dim)

    Why normalization matters:
    - It stabilizes feature scale before downstream layers.
    - It reduces internal covariate drift.
    - It is especially useful when many agents or many rollouts produce
      diverse feature magnitudes.

    Why LayerNorm is a sensible default here:
    - It works naturally on per-sample feature vectors.
    - It does not depend on batch statistics.
    - It behaves well in PPO / actor-critic settings where batch geometry
      may vary.
    """
    norm = str(getattr(config, "BRAIN_MLP_NORM", "layernorm")).strip().lower()
    if norm == "none":
        return nn.Identity()
    return nn.LayerNorm(dim)


def brain_kind_display_name(kind: str) -> str:
    """
    Convert an internal brain kind key into a human-facing display name.

    Example:
    - "whispering_abyss" may map to a prettier UI label if config provides one.

    Why this exists:
    - Internal keys are often optimized for code stability.
    - UI/telemetry labels are often optimized for readability.
    - Keeping that mapping in config avoids hardcoding presentation policy.
    """
    mapping = dict(getattr(config, "BRAIN_KIND_DISPLAY_NAMES", {}))
    kind = str(kind).strip().lower()
    return str(mapping.get(kind, kind))


def brain_kind_short_label(kind: str) -> str:
    """
    Return a short label for compact displays, tags, overlays, or viewer UI.

    Fallback behavior:
    - If no explicit short label exists in config, use the first two letters
      uppercased.
    - If the input is empty, use "?" as a safe placeholder.

    This is useful when:
    - space is limited in the viewer
    - telemetry tags need concise identifiers
    - debugging many agents at once
    """
    mapping = dict(getattr(config, "BRAIN_KIND_SHORT_LABELS", {}))
    kind = str(kind).strip().lower()
    return str(mapping.get(kind, kind[:2].upper() if kind else "?"))


def brain_kind_from_module(module: Optional[nn.Module]) -> Optional[str]:
    """
    Extract the normalized brain kind string from a module, if present.

    Convention used in this file:
    - Each concrete brain class sets `brain_kind` as a lowercase identifier.

    Returns:
    - normalized string kind if found
    - None otherwise

    Why this helper matters:
    - It lets tooling inspect a brain instance without depending on exact class
      type checks.
    - It is friendly to wrappers, registries, UI code, and logging systems.
    """
    if module is None:
        return None
    kind = getattr(module, "brain_kind", None)
    if isinstance(kind, str) and kind:
        return kind.strip().lower()
    return None


def describe_brain_module(module: Optional[nn.Module]) -> str:
    """
    Produce a human-readable textual description of a brain module.

    Intended use:
    - logging
    - debugging
    - telemetry
    - sanity checks in experiments

    Example output shape idea:
    - "Whispering Abyss [64→96→96→41]"

    How this is assembled:
    - Determine the brain kind.
    - Look up the display label.
    - Infer input width and actor output width.
    - Walk through Linear layers to build a rough architecture signature.

    Important note:
    - This is descriptive tooling, not execution logic.
    - It is written defensively so failures in introspection do not break
      training. If inspection fails, it falls back to a simpler label.
    """
    kind = brain_kind_from_module(module)
    if not kind:
        return "<none>"

    display = brain_kind_display_name(kind)
    in_w = int(getattr(module, "final_input_width", 0))
    actor = getattr(module, "actor_head", None)
    out_w = int(getattr(actor, "out_features", 0)) if actor is not None else 0

    hidden_dims = []
    try:
        # We walk all submodules and record Linear widths to build a compact
        # signature string. This is only for display/debugging.
        for m in module.modules():
            if isinstance(m, nn.Linear):
                if int(m.in_features) == in_w:
                    hidden_dims.append(int(m.out_features))
                elif int(m.out_features) == out_w:
                    break
                else:
                    hidden_dims.append(int(m.out_features))
    except Exception:
        hidden_dims = []

    if hidden_dims:
        return f"{display} [{in_w}→" + "→".join(str(d) for d in hidden_dims) + f"→{out_w}]"
    return display


class _ResidualBlock(nn.Module):
    """
    Standard same-width residual MLP block.

    Structure:
        x
        └─> norm
            └─> linear(width→width)
                └─> activation
                    └─> linear(width→width)
                        └─> add skip connection with original x

    Formula:
        h = fc2( act( fc1( norm(x) ) ) )
        y = x + h

    Why residual connections help:
    - They preserve an identity path for gradient flow.
    - They allow deeper networks to learn refinements instead of relearning
      the whole representation from scratch.
    - In practice they often stabilize optimization.

    Why same-width:
    - Residual addition requires matching dimensionality between input and
      residual branch output.
    """

    def __init__(self, width: int) -> None:
        super().__init__()
        self.norm = _maybe_norm(width)
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the residual block.

        Dtype handling detail:
        - The normalization is performed in float() for numerical stability.
        - Outputs are converted to the target linear-layer dtype where needed.

        That pattern is often useful in mixed-precision or AMP-heavy pipelines,
        because some normalization operations are safer in full precision.
        """
        h = self.norm(x.float()).to(dtype=x.dtype)
        h = self.fc1(h.to(dtype=self.fc1.weight.dtype))
        h = _activation_fn(h)
        h = self.fc2(h.to(dtype=self.fc2.weight.dtype))
        return x + h


class _GatedBlock(nn.Module):
    """
    Gated residual block.

    Structure:
        x
        └─> norm
            ├─> value projection
            ├─> gate projection
            └─> combine as:
                activation(value) * sigmoid(gate)
                └─> output projection
                    └─> add residual x

    Formula:
        v = value(norm(x))
        g = gate(norm(x))
        h = act(v) * sigmoid(g)
        y = x + out(h)

    Intuition:
    - The value branch proposes content.
    - The gate branch modulates how much of that content should pass.
    - This can provide more expressive control than a plain residual block
      while remaining lighter and simpler than attention mechanisms.

    Why sigmoid on gate:
    - It smoothly constrains gate values to (0, 1).
    - It behaves like a soft feature selector.
    """

    def __init__(self, width: int) -> None:
        super().__init__()
        self.norm = _maybe_norm(width)
        self.value = nn.Linear(width, width)
        self.gate = nn.Linear(width, width)
        self.out = nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for gated residual transformation.
        """
        h = self.norm(x.float()).to(dtype=x.dtype)
        v = self.value(h.to(dtype=self.value.weight.dtype))
        g = self.gate(h.to(dtype=self.gate.weight.dtype))
        h = _activation_fn(v) * torch.sigmoid(g)
        h = self.out(h.to(dtype=self.out.weight.dtype))
        return x + h


class _BottleneckResidualBlock(nn.Module):
    """
    Bottleneck residual block.

    Structure:
        x
        └─> norm
            └─> down-project width→inner
                └─> activation
                    └─> up-project inner→width
                        └─> add residual x

    Formula:
        h = up( act( down( norm(x) ) ) )
        y = x + h

    Why use a bottleneck:
    - It reduces intermediate computation relative to a full width→width→width
      residual block.
    - It encourages the model to compress and re-expand information.
    - It often provides a good efficiency/expressivity tradeoff.

    Here:
    - `width` is the outer representation size.
    - `inner` is the compressed bottleneck size.
    """

    def __init__(self, width: int, inner: int) -> None:
        super().__init__()
        self.norm = _maybe_norm(width)
        self.down = nn.Linear(width, inner)
        self.up = nn.Linear(inner, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for bottleneck residual transformation.
        """
        h = self.norm(x.float()).to(dtype=x.dtype)
        h = self.down(h.to(dtype=self.down.weight.dtype))
        h = _activation_fn(h)
        h = self.up(h.to(dtype=self.up.weight.dtype))
        return x + h


class _BaseMLPBrain(nn.Module):
    """
    Shared base class for all MLP brain variants.

    Responsibilities:
    1. Validate the observation contract.
    2. Build the two-token shared input interface.
    3. Delegate trunk construction to subclass variant.
    4. Attach actor and critic heads.
    5. Initialize weights with PPO-friendly defaults.
    6. Execute forward pass and enforce output shape correctness.

    Core shared contract:
    - All concrete brains receive the SAME observation semantics.
    - All concrete brains produce the SAME output semantics.
    - Only the trunk architecture differs across variants.

    This is extremely important for fair comparison across model families:
    - same input preprocessing
    - same output heads
    - same overall interface
    - controlled architectural variation
    """

    brain_kind: str = "base"

    def __init__(self, obs_dim: int, act_dim: int) -> None:
        super().__init__()

        # Store public structural metadata.
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)

        # Observation schema parameters.
        # These define the expected meaning of the flattened observation vector.
        # Current expected structure:
        #   total_obs_dim = (num_rays * ray_feat_dim) + rich_total_dim
        #                 = (32 * 8) + (23 + 4)
        #                 = 256 + 27
        #                 = 283
        # This matches the observation spec used elsewhere in the project.
        self.num_rays = int(getattr(config, "RAY_TOKEN_COUNT", 32))
        self.ray_feat_dim = int(getattr(config, "RAY_FEAT_DIM", 8))
        self.rich_base_dim = int(getattr(config, "RICH_BASE_DIM", 23))
        self.instinct_dim = int(getattr(config, "INSTINCT_DIM", 4))
        self.rich_total_dim = self.rich_base_dim + self.instinct_dim

        expected_obs_dim = self.num_rays * self.ray_feat_dim + self.rich_total_dim
        cfg_obs_dim = int(getattr(config, "OBS_DIM", expected_obs_dim))

        # Hard validation against config.
        # Why fail fast here?
        # Because if observation layout changes silently but the network still
        # runs, training can degrade or collapse without obvious errors.
        if cfg_obs_dim != expected_obs_dim:
            raise RuntimeError(
                f"[{self.__class__.__name__}] OBS layout mismatch: expected {expected_obs_dim}, "
                f"config.OBS_DIM={cfg_obs_dim}"
            )

        # Hard validation against constructor input.
        # This protects the factory/runtime boundary. If some caller passes
        # an inconsistent obs_dim, the mismatch is surfaced immediately.
        if self.obs_dim != expected_obs_dim:
            raise RuntimeError(
                f"[{self.__class__.__name__}] obs_dim mismatch: ctor={self.obs_dim}, expected={expected_obs_dim}"
            )

        # Shared two-token input interface.
        # D is the learned token width (sometimes analogous to a model width).
        # Final flat input width is exactly 2D because we concatenate:
        #   [ray_summary_token, rich_token]
        self.d_model = int(getattr(config, "BRAIN_MLP_D_MODEL", 32))
        self.final_input_width = 2 * self.d_model

        # Optional config cross-check for systems that want the final width
        # explicitly exposed as a knob or validated elsewhere.
        if self.final_input_width != int(getattr(config, "BRAIN_MLP_FINAL_INPUT_WIDTH", self.final_input_width)):
            raise RuntimeError(
                f"[{self.__class__.__name__}] final input width mismatch with config"
            )

        # Shared two-token embedding pipeline.
        # Contract:
        #   rays_raw: (B, 32, 8)
        #       -> LayerNorm over feature dim
        #       -> Linear(8 -> D) per ray
        #       -> mean across the 32 rays
        #       -> ray token: (B, D)
        #   rich_vec: (B, 27)
        #       -> LayerNorm
        #       -> Linear(27 -> D)
        #       -> rich token: (B, D)
        #   concat(ray_token, rich_token)
        #       -> (B, 2D)
        #       -> final input normalization
        # Why mean summarize the rays?
        # - It converts variable structured ray channels into one fixed-size
        #   token without attention.
        # - It is cheap, stable, and batching-friendly.
        # - It preserves a shared semantic path across all MLP variants.
        self.ray_in_norm = nn.LayerNorm(self.ray_feat_dim)
        self.ray_in_proj = nn.Linear(self.ray_feat_dim, self.d_model)
        self.rich_in_norm = nn.LayerNorm(self.rich_total_dim)
        self.rich_in_proj = nn.Linear(self.rich_total_dim, self.d_model)
        self.input_norm = _maybe_norm(self.final_input_width)

        # Let the concrete subclass define only its own trunk shape.
        self.trunk, trunk_out = self._build_variant_trunk()

        # Actor head:
        # - outputs one logit per action
        # - logits are later used by categorical policy logic
        self.actor_head = nn.Linear(trunk_out, self.act_dim)

        # Critic head:
        # - outputs scalar state-value estimate V(s)
        self.critic_head = nn.Linear(trunk_out, 1)

        # Apply initialization after all layers exist.
        self._init_weights()

    def _build_variant_trunk(self) -> Tuple[nn.Module, int]:
        """
        Concrete subclasses must implement this.

        Returns:
        - trunk module
        - output width of that trunk

        Why return the width explicitly?
        - The actor/critic heads need to know the trunk's final feature size.
        - This keeps the contract clear and avoids fragile introspection.
        """
        raise NotImplementedError

    def _init_weights(self) -> None:
        """
        Initialize weights using orthogonal initialization.

        Initialization policy:
        - Hidden linear layers:
            orthogonal(gain=sqrt(2))
        - Actor head:
            orthogonal(gain=configurable, default 0.01)
        - Critic head:
            orthogonal(gain=configurable, default 1.0)
        - Biases:
            zeros

        Why orthogonal init is common in PPO / RL:
        - It often yields stable early optimization.
        - Small actor gain keeps initial policy logits modest, which reduces
          overly confident action distributions at startup.
        - Critic gain of 1.0 is a common default for value heads.
        """
        gain_hidden = math.sqrt(2.0)
        actor_gain = float(getattr(config, "BRAIN_MLP_ACTOR_INIT_GAIN", 0.01))
        critic_gain = float(getattr(config, "BRAIN_MLP_CRITIC_INIT_GAIN", 1.0))

        # Initialize all linear layers uniformly first.
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=gain_hidden)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Then specialize actor head gain.
        nn.init.orthogonal_(self.actor_head.weight, gain=actor_gain)
        if self.actor_head.bias is not None:
            nn.init.zeros_(self.actor_head.bias)

        # Then specialize critic head gain.
        nn.init.orthogonal_(self.critic_head.weight, gain=critic_gain)
        if self.critic_head.bias is not None:
            nn.init.zeros_(self.critic_head.bias)

    def _embed_rays(self, rays_raw: torch.Tensor) -> torch.Tensor:
        """
        Convert raw ray tensor into a single learned summary token.

        Input:
        - rays_raw shape: (B, RAY_TOKEN_COUNT, RAY_FEAT_DIM)

        Processing:
        1. LayerNorm on the per-ray feature dimension
        2. Linear projection from feature space -> d_model
        3. Summary reduction across ray index dimension

        Current supported summary:
        - "mean"

        Output:
        - shape: (B, d_model)

        Why mean is enforced here:
        - The code wants a shared, explicit, and controlled tokenization rule.
        - Unsupported modes raise immediately to avoid silent behavior drift.
        """
        x = self.ray_in_norm(rays_raw.float())
        x = self.ray_in_proj(x.to(dtype=self.ray_in_proj.weight.dtype))
        summary_mode = str(getattr(config, "BRAIN_MLP_RAY_SUMMARY", "mean")).strip().lower()
        if summary_mode != "mean":
            raise RuntimeError(
                f"[{self.__class__.__name__}] Unsupported BRAIN_MLP_RAY_SUMMARY={summary_mode!r}"
            )
        return x.mean(dim=1)

    def _embed_rich(self, rich_vec: torch.Tensor) -> torch.Tensor:
        """
        Convert the full non-ray rich feature vector into one learned token.

        Input:
        - rich_vec shape: (B, rich_total_dim)

        Output:
        - shape: (B, d_model)

        Intuition:
        - This treats all non-ray scalar/context information as one compact
          semantic packet.
        - It keeps the architecture simple and consistent across variants.
        """
        x = self.rich_in_norm(rich_vec.float())
        x = self.rich_in_proj(x.to(dtype=self.rich_in_proj.weight.dtype))
        return x

    def _build_flat_input(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Build the final shared flat input vector for the MLP trunk.

        Expected input:
        - obs shape: (B, F), where F == self.obs_dim

        Steps:
        1. Validate rank and feature width.
        2. Split flat observation into:
           - rays_raw
           - rich_vec
        3. Embed rays into one token.
        4. Embed rich features into one token.
        5. Concatenate the two tokens.
        6. Normalize final input.

        Output:
        - shape: (B, final_input_width) where final_input_width = 2 * d_model
        """
        if obs.dim() != 2:
            raise RuntimeError(
                f"[{self.__class__.__name__}] expected obs rank 2 (B,F), got {tuple(obs.shape)}"
            )
        if int(obs.shape[1]) != self.obs_dim:
            raise RuntimeError(
                f"[{self.__class__.__name__}] obs feature dim mismatch: got {int(obs.shape[1])}, expected {self.obs_dim}"
            )

        # Delegates the authoritative layout split to obs_spec so this module
        # does not duplicate observation parsing logic.
        rays_raw, rich_vec = obs_spec.split_obs_for_mlp(obs)

        # Build the two learned tokens.
        ray_tok = self._embed_rays(rays_raw)
        rich_tok = self._embed_rich(rich_vec)

        # Concatenate them into one flat vector.
        x = torch.cat([ray_tok, rich_tok], dim=-1)

        # Final normalization before the variant trunk.
        x = self.input_norm(x.float()).to(dtype=x.dtype)

        # Final shape assertion for safety.
        if int(x.shape[1]) != self.final_input_width:
            raise RuntimeError(
                f"[{self.__class__.__name__}] flat input mismatch: got {int(x.shape[1])}, expected {self.final_input_width}"
            )
        return x

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the actor-critic brain.

        Input:
        - obs: shape (B, obs_dim)

        Output:
        - logits: shape (B, act_dim)
        - value:  shape (B, 1)

        Detailed flow:
        1. Build shared flat two-token-derived input.
        2. Run it through the concrete MLP trunk.
        3. Feed trunk output into actor head and critic head.
        4. Validate output shapes.

        Why validate output shapes here:
        - Actor/critic shape errors can propagate very far before surfacing.
        - PPO training code usually assumes exact shapes.
        - Early hard failure saves debugging time.
        """
        x = self._build_flat_input(obs)
        h = self.trunk(x)

        # Actor head produces unnormalized action logits.
        logits = self.actor_head(h.to(dtype=self.actor_head.weight.dtype))

        # Critic head produces scalar value estimate.
        value = self.critic_head(h.to(dtype=self.critic_head.weight.dtype))

        B = int(obs.shape[0])

        # Shape assertions for the actor output.
        if logits.dim() != 2 or int(logits.shape[0]) != B or int(logits.shape[1]) != self.act_dim:
            raise RuntimeError(
                f"[{self.__class__.__name__}] bad logits shape: got {tuple(logits.shape)}, expected ({B},{self.act_dim})"
            )

        # Shape assertions for the critic output.
        if value.dim() != 2 or int(value.shape[0]) != B or int(value.shape[1]) != 1:
            raise RuntimeError(
                f"[{self.__class__.__name__}] bad value shape: got {tuple(value.shape)}, expected ({B},1)"
            )

        return logits, value


class WhisperingAbyssBrain(_BaseMLPBrain):
    """
    Variant 1: Whispering Abyss

    Shape:
        input(2D) -> 96 -> 96 -> heads

    Characteristics:
    - simplest compact two-layer GELU MLP trunk
    - low conceptual complexity
    - useful as a baseline member of the family

    Why this variant exists:
    - It provides a straightforward dense MLP reference architecture.
    - When comparing multiple variants, a plain trunk is valuable as a control.
    """

    brain_kind = "whispering_abyss"

    def _build_variant_trunk(self) -> Tuple[nn.Module, int]:
        h0 = nn.Linear(self.final_input_width, 96)
        h1 = nn.Linear(96, 96)
        trunk = nn.Sequential(
            h0,
            nn.GELU(),
            h1,
            nn.GELU(),
        )
        return trunk, 96


class VeilOfEchoesBrain(_BaseMLPBrain):
    """
    Variant 2: Veil of Echoes

    Shape:
        input(2D) -> 128 -> 96 -> 64 -> heads

    Characteristics:
    - progressively narrowing feedforward trunk
    - slightly deeper than Whispering Abyss
    - encourages staged compression of features

    Design intuition:
    - Early wider layer can capture mixed feature interactions.
    - Later narrowing may encourage abstraction/compression before the heads.
    """

    brain_kind = "veil_of_echoes"

    def _build_variant_trunk(self) -> Tuple[nn.Module, int]:
        trunk = nn.Sequential(
            nn.Linear(self.final_input_width, 128),
            nn.GELU(),
            nn.Linear(128, 96),
            nn.GELU(),
            nn.Linear(96, 64),
            nn.GELU(),
        )
        return trunk, 64


class CathedralOfAshBrain(_BaseMLPBrain):
    """
    Variant 3: Cathedral of Ash

    Shape:
        input(2D) -> 80 -> residual block -> residual block -> residual block -> heads

    Characteristics:
    - fixed-width residual stack
    - emphasizes iterative refinement at a stable hidden width

    Design intuition:
    - The entry projection moves the input into a working latent space.
    - Residual blocks repeatedly refine that latent state while preserving a
      strong identity path.
    """

    brain_kind = "cathedral_of_ash"

    def _build_variant_trunk(self) -> Tuple[nn.Module, int]:
        width = 80
        blocks = 3
        trunk = nn.Sequential(
            nn.Linear(self.final_input_width, width),
            nn.GELU(),
            *[_ResidualBlock(width) for _ in range(blocks)],
        )
        return trunk, width


class DreamerInBlackFogBrain(_BaseMLPBrain):
    """
    Variant 4: Dreamer in Black Fog

    Shape:
        input(2D) -> 80 -> gated block -> gated block -> heads

    Characteristics:
    - uses gated residual transformations
    - allows feature-wise modulation through learned gates

    Design intuition:
    - Not all transformed features should always pass equally.
    - Gating can improve selectivity without introducing attention machinery.
    """

    brain_kind = "dreamer_in_black_fog"

    def _build_variant_trunk(self) -> Tuple[nn.Module, int]:
        width = 80
        blocks = 2
        trunk = nn.Sequential(
            nn.Linear(self.final_input_width, width),
            nn.GELU(),
            *[_GatedBlock(width) for _ in range(blocks)],
        )
        return trunk, width


class ObsidianPulseBrain(_BaseMLPBrain):
    """
    Variant 5: Obsidian Pulse

    Shape:
        input(2D) -> 128 -> bottleneck residual -> bottleneck residual -> heads

    Characteristics:
    - wider outer representation
    - compressed inner bottleneck path
    - residual refinement with efficiency-conscious inner width

    Design intuition:
    - A larger outer width can preserve richer feature capacity.
    - A smaller bottleneck reduces internal compute while still forcing
      meaningful compression/re-expansion.
    """

    brain_kind = "obsidian_pulse"

    def _build_variant_trunk(self) -> Tuple[nn.Module, int]:
        outer = 128
        inner = 48
        trunk = nn.Sequential(
            nn.Linear(self.final_input_width, outer),
            nn.GELU(),
            _BottleneckResidualBlock(outer, inner),
            _BottleneckResidualBlock(outer, inner),
        )
        return trunk, outer


def create_mlp_brain(kind: str, obs_dim: int, act_dim: int) -> nn.Module:
    """
    Factory function for constructing a concrete MLP brain by string kind.

    Supported kinds:
    - "whispering_abyss"
    - "veil_of_echoes"
    - "cathedral_of_ash"
    - "dreamer_in_black_fog"
    - "obsidian_pulse"

    Why a factory is useful:
    - Centralizes mapping from config/runtime string -> concrete class.
    - Keeps calling code simple.
    - Makes logging/selection/registration cleaner.

    Raises:
    - ValueError if the requested kind is unknown.

    Important property:
    - Every returned module obeys the same forward interface:
          (obs) -> (logits, value)
    """
    k = str(kind).strip().lower()
    if k == "whispering_abyss":
        return WhisperingAbyssBrain(obs_dim, act_dim)
    if k == "veil_of_echoes":
        return VeilOfEchoesBrain(obs_dim, act_dim)
    if k == "cathedral_of_ash":
        return CathedralOfAshBrain(obs_dim, act_dim)
    if k == "dreamer_in_black_fog":
        return DreamerInBlackFogBrain(obs_dim, act_dim)
    if k == "obsidian_pulse":
        return ObsidianPulseBrain(obs_dim, act_dim)
    raise ValueError(f"Unknown MLP brain kind: {kind!r}")
