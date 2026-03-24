"""MLP policy/value networks used by agent slots."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import config
from . import obs_spec


CANONICAL_BRAIN_KINDS = (
    "throne_of_ashen_dreams",
    "veil_of_the_hollow_crown",
    "black_grail_of_nightfire",
)


def normalize_brain_kind(kind: str) -> str:
    """Normalize a configured or serialized brain kind into its canonical key."""
    norm = str(kind).strip().lower()
    alias_map = dict(getattr(config, "BRAIN_MLP_KIND_ALIASES", {}))
    return str(alias_map.get(norm, norm))


def _activation_fn(x: torch.Tensor) -> torch.Tensor:
    act = str(getattr(config, "BRAIN_MLP_ACTIVATION", "gelu")).strip().lower()
    if act == "relu":
        return F.relu(x)
    if act == "silu":
        return F.silu(x)
    return F.gelu(x)


def _maybe_norm(dim: int) -> nn.Module:
    norm = str(getattr(config, "BRAIN_MLP_NORM", "layernorm")).strip().lower()
    if norm == "none":
        return nn.Identity()
    return nn.LayerNorm(dim)


def _gate_scale(x: torch.Tensor) -> torch.Tensor:
    style = str(getattr(config, "BRAIN_MLP_GATE_STYLE", "sigmoid")).strip().lower()
    strength = float(getattr(config, "BRAIN_MLP_GATE_STRENGTH", 1.0))
    if style == "tanh":
        return 1.0 + strength * torch.tanh(x)
    return 1.0 + strength * (2.0 * torch.sigmoid(x) - 1.0)


def brain_kind_display_name(kind: str) -> str:
    mapping = dict(getattr(config, "BRAIN_KIND_DISPLAY_NAMES", {}))
    norm = normalize_brain_kind(kind)
    return str(mapping.get(norm, norm))


def brain_kind_short_label(kind: str) -> str:
    mapping = dict(getattr(config, "BRAIN_KIND_SHORT_LABELS", {}))
    norm = normalize_brain_kind(kind)
    return str(mapping.get(norm, norm[:3].upper() if norm else "?"))


def brain_kind_from_module(module: Optional[nn.Module]) -> Optional[str]:
    if module is None:
        return None
    kind = getattr(module, "brain_kind", None)
    if isinstance(kind, str) and kind:
        return normalize_brain_kind(kind)
    return None


def describe_brain_module(module: Optional[nn.Module]) -> str:
    kind = brain_kind_from_module(module)
    if not kind:
        return "<none>"

    display = brain_kind_display_name(kind)
    signature = str(getattr(module, "model_signature", "")).strip()
    if signature:
        return f"{display} [{signature}]"
    return display


class _ResidualBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        hidden = max(
            width,
            int(round(width * float(getattr(config, "BRAIN_MLP_BLOCK_EXPANSION", 2.0)))),
        )
        self.norm = _maybe_norm(width)
        self.fc1 = nn.Linear(width, hidden)
        self.fc2 = nn.Linear(hidden, width)
        drop_p = float(getattr(config, "BRAIN_MLP_DROPOUT", 0.0))
        self.dropout = nn.Dropout(drop_p) if drop_p > 0.0 else nn.Identity()
        layer_scale_init = float(getattr(config, "BRAIN_MLP_LAYER_SCALE_INIT", 1.0))
        self.layer_scale = (
            nn.Parameter(torch.full((width,), layer_scale_init))
            if layer_scale_init > 0.0
            else None
        )
        self.use_residual = bool(getattr(config, "BRAIN_MLP_USE_RESIDUAL", True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x.float()).to(dtype=x.dtype)
        h = self.fc1(h.to(dtype=self.fc1.weight.dtype))
        h = _activation_fn(h)
        h = self.dropout(h)
        h = self.fc2(h.to(dtype=self.fc2.weight.dtype))
        if self.layer_scale is not None:
            shape = (1,) * (h.dim() - 1) + (-1,)
            h = h * self.layer_scale.view(shape).to(dtype=h.dtype)
        if self.use_residual:
            return x + h
        return h


class _FeatureTower(nn.Module):
    def __init__(self, input_width: int, width: int, depth: int) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(input_width)
        self.input_proj = nn.Linear(input_width, width)
        self.blocks = nn.ModuleList([_ResidualBlock(width) for _ in range(depth)])
        self.output_norm = _maybe_norm(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x.float()).to(dtype=x.dtype)
        x = self.input_proj(x.to(dtype=self.input_proj.weight.dtype))
        x = _activation_fn(x)
        for block in self.blocks:
            x = block(x)
        x = self.output_norm(x.float()).to(dtype=x.dtype)
        return x


class _RayPool(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.mode = str(getattr(config, "BRAIN_MLP_RAY_POOLING", "mean_max")).strip().lower()
        self.score = nn.Linear(width, 1) if self.mode == "gated_mean" else None
        self.fuse = nn.Linear(2 * width, width) if self.mode == "mean_max" else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise RuntimeError(f"ray token tensor must be rank-3 (B,R,D), got {tuple(x.shape)}")

        if self.mode == "mean":
            return x.mean(dim=1)
        if self.mode == "gated_mean":
            assert self.score is not None
            scores = self.score(x.to(dtype=self.score.weight.dtype)).squeeze(-1)
            attn = torch.softmax(scores, dim=1).to(dtype=x.dtype)
            return torch.sum(x * attn.unsqueeze(-1), dim=1)
        if self.mode == "mean_max":
            mean_tok = x.mean(dim=1)
            max_tok = x.amax(dim=1)
            fused = torch.cat([mean_tok, max_tok], dim=-1)
            assert self.fuse is not None
            fused = self.fuse(fused.to(dtype=self.fuse.weight.dtype))
            return _activation_fn(fused)
        raise RuntimeError(f"Unsupported BRAIN_MLP_RAY_POOLING={self.mode!r}")


class _ContextGate(nn.Module):
    def __init__(self, width: int, context_width: int) -> None:
        super().__init__()
        hidden = int(getattr(config, "BRAIN_MLP_GATE_HIDDEN_WIDTH", max(width, context_width)))
        self.pre = nn.Linear(context_width, hidden)
        self.scale = nn.Linear(hidden, width)
        self.bias = nn.Linear(hidden, width)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        h = self.pre(context.to(dtype=self.pre.weight.dtype))
        h = _activation_fn(h)
        scale = _gate_scale(self.scale(h))
        bias = self.bias(h)
        while scale.dim() < x.dim():
            scale = scale.unsqueeze(1)
            bias = bias.unsqueeze(1)
        return x * scale.to(dtype=x.dtype) + bias.to(dtype=x.dtype)


class _BaseMLPBrain(nn.Module):
    brain_kind: str = "base"
    architecture_name: str = "base"

    def __init__(self, obs_dim: int, act_dim: int) -> None:
        super().__init__()
        obs_spec.validate_obs_contract()

        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.obs_schema = obs_spec.obs_schema_name()

        self.num_rays = int(getattr(config, "RAY_TOKEN_COUNT"))
        self.ray_feat_dim = int(getattr(config, "RAY_FEAT_DIM"))
        self.rich_total_dim = int(getattr(config, "RICH_TOTAL_DIM"))

        expected_obs_dim = self.num_rays * self.ray_feat_dim + self.rich_total_dim
        if self.obs_dim != expected_obs_dim:
            raise RuntimeError(
                f"[{self.__class__.__name__}] obs_dim mismatch: ctor={self.obs_dim}, expected={expected_obs_dim}"
            )
        if int(getattr(config, "OBS_DIM")) != expected_obs_dim:
            raise RuntimeError(
                f"[{self.__class__.__name__}] config.OBS_DIM drifted from the active observation contract"
            )

        self.ray_width = int(getattr(config, "BRAIN_MLP_RAY_WIDTH", 96))
        self.scalar_width = int(getattr(config, "BRAIN_MLP_SCALAR_WIDTH", 96))
        self.fusion_width = int(getattr(config, "BRAIN_MLP_FUSION_WIDTH", 128))
        self.ray_depth = int(getattr(config, "BRAIN_MLP_RAY_DEPTH", 1))
        self.scalar_depth = int(getattr(config, "BRAIN_MLP_SCALAR_DEPTH", 2))
        self.trunk_depth = int(getattr(config, "BRAIN_MLP_TRUNK_DEPTH", 3))
        self.final_input_width = self.ray_width + self.scalar_width

        cfg_fusion_input = int(getattr(config, "BRAIN_MLP_FINAL_INPUT_WIDTH", self.final_input_width))
        if cfg_fusion_input != self.final_input_width:
            raise RuntimeError(
                f"[{self.__class__.__name__}] final input width mismatch: brain={self.final_input_width}, config={cfg_fusion_input}"
            )

        self.ray_tower = _FeatureTower(self.ray_feat_dim, self.ray_width, self.ray_depth)
        self.scalar_tower = _FeatureTower(self.rich_total_dim, self.scalar_width, self.scalar_depth)
        self.ray_pool = _RayPool(self.ray_width)

        self._build_variant_layers()

        self.actor_head = nn.Linear(self.fusion_width, self.act_dim)
        self.critic_head = nn.Linear(self.fusion_width, 1)
        self.model_signature = (
            f"{self.architecture_name}|schema={self.obs_schema}|"
            f"ray={self.ray_width}x{self.ray_depth}|"
            f"scalar={self.scalar_width}x{self.scalar_depth}|"
            f"fusion={self.fusion_width}x{self.trunk_depth}"
        )
        self._init_weights()

    def _build_shared_fusion_stack(self) -> Tuple[nn.Linear, nn.ModuleList, nn.Module]:
        in_proj = nn.Linear(self.final_input_width, self.fusion_width)
        blocks = nn.ModuleList([_ResidualBlock(self.fusion_width) for _ in range(self.trunk_depth)])
        out_norm = _maybe_norm(self.fusion_width)
        return in_proj, blocks, out_norm

    def _run_shared_fusion_stack(
        self,
        x: torch.Tensor,
        in_proj: nn.Linear,
        blocks: nn.ModuleList,
        out_norm: nn.Module,
    ) -> torch.Tensor:
        x = in_proj(x.to(dtype=in_proj.weight.dtype))
        x = _activation_fn(x)
        for block in blocks:
            x = block(x)
        x = out_norm(x.float()).to(dtype=x.dtype)
        return x

    def _build_variant_layers(self) -> None:
        raise NotImplementedError

    def _condition_ray_tokens(self, ray_tokens: torch.Tensor, scalar_ctx: torch.Tensor) -> torch.Tensor:
        return ray_tokens

    def _fuse(self, ray_summary: torch.Tensor, scalar_ctx: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _init_weights(self) -> None:
        hidden_gain = math.sqrt(2.0)
        actor_gain = float(getattr(config, "BRAIN_MLP_ACTOR_INIT_GAIN", 0.01))
        critic_gain = float(getattr(config, "BRAIN_MLP_CRITIC_INIT_GAIN", 1.0))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=hidden_gain)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        for m in self.modules():
            if isinstance(m, _ContextGate):
                nn.init.zeros_(m.scale.weight)
                nn.init.zeros_(m.scale.bias)
                nn.init.zeros_(m.bias.weight)
                nn.init.zeros_(m.bias.bias)

        nn.init.orthogonal_(self.actor_head.weight, gain=actor_gain)
        nn.init.zeros_(self.actor_head.bias)
        nn.init.orthogonal_(self.critic_head.weight, gain=critic_gain)
        nn.init.zeros_(self.critic_head.bias)

    def _encode_features(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if obs.dim() != 2:
            raise RuntimeError(
                f"[{self.__class__.__name__}] expected obs rank 2 (B,F), got {tuple(obs.shape)}"
            )
        if int(obs.shape[1]) != self.obs_dim:
            raise RuntimeError(
                f"[{self.__class__.__name__}] obs feature dim mismatch: got {int(obs.shape[1])}, expected {self.obs_dim}"
            )

        rays_raw, rich_vec = obs_spec.split_obs_for_mlp(obs)
        scalar_ctx = self.scalar_tower(rich_vec)
        ray_tokens = self.ray_tower(rays_raw)
        ray_tokens = self._condition_ray_tokens(ray_tokens, scalar_ctx)
        ray_summary = self.ray_pool(ray_tokens)

        if tuple(ray_summary.shape) != (int(obs.shape[0]), self.ray_width):
            raise RuntimeError(
                f"[{self.__class__.__name__}] ray summary width drifted: got {tuple(ray_summary.shape)}"
            )
        if tuple(scalar_ctx.shape) != (int(obs.shape[0]), self.scalar_width):
            raise RuntimeError(
                f"[{self.__class__.__name__}] scalar tower width drifted: got {tuple(scalar_ctx.shape)}"
            )

        return ray_summary, scalar_ctx

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ray_summary, scalar_ctx = self._encode_features(obs)
        h = self._fuse(ray_summary, scalar_ctx)

        bsz = int(obs.shape[0])
        if tuple(h.shape) != (bsz, self.fusion_width):
            raise RuntimeError(
                f"[{self.__class__.__name__}] bad fusion shape: got {tuple(h.shape)}, expected ({bsz},{self.fusion_width})"
            )

        logits = self.actor_head(h.to(dtype=self.actor_head.weight.dtype))
        value = self.critic_head(h.to(dtype=self.critic_head.weight.dtype))

        if tuple(logits.shape) != (bsz, self.act_dim):
            raise RuntimeError(
                f"[{self.__class__.__name__}] bad logits shape: got {tuple(logits.shape)}, expected ({bsz},{self.act_dim})"
            )
        if tuple(value.shape) != (bsz, 1):
            raise RuntimeError(
                f"[{self.__class__.__name__}] bad value shape: got {tuple(value.shape)}, expected ({bsz},1)"
            )
        return logits, value


class ThroneOfAshenDreamsBrain(_BaseMLPBrain):
    brain_kind = "throne_of_ashen_dreams"
    architecture_name = "dual_tower_late_fusion"

    def _build_variant_layers(self) -> None:
        self.fusion_in, self.fusion_blocks, self.fusion_out_norm = self._build_shared_fusion_stack()

    def _fuse(self, ray_summary: torch.Tensor, scalar_ctx: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([ray_summary, scalar_ctx], dim=-1)
        return self._run_shared_fusion_stack(
            fused,
            self.fusion_in,
            self.fusion_blocks,
            self.fusion_out_norm,
        )


class VeilOfTheHollowCrownBrain(_BaseMLPBrain):
    brain_kind = "veil_of_the_hollow_crown"
    architecture_name = "dual_tower_scalar_reinjection"

    def _build_variant_layers(self) -> None:
        self.fusion_in, self.fusion_blocks, self.fusion_out_norm = self._build_shared_fusion_stack()
        self.reinject_every = max(1, int(getattr(config, "BRAIN_MLP_REINJECT_EVERY", 1)))
        self.reinject_scale = float(getattr(config, "BRAIN_MLP_REINJECT_SCALE", 1.0))
        self.scalar_reinject = nn.ModuleList(
            [nn.Linear(self.scalar_width, self.fusion_width) for _ in range(self.trunk_depth)]
        )

    def _fuse(self, ray_summary: torch.Tensor, scalar_ctx: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([ray_summary, scalar_ctx], dim=-1)
        h = self.fusion_in(fused.to(dtype=self.fusion_in.weight.dtype))
        h = _activation_fn(h)
        for i, block in enumerate(self.fusion_blocks):
            if i % self.reinject_every == 0:
                inject = self.scalar_reinject[i](scalar_ctx.to(dtype=self.scalar_reinject[i].weight.dtype))
                h = h + self.reinject_scale * inject.to(dtype=h.dtype)
            h = block(h)
        h = self.fusion_out_norm(h.float()).to(dtype=h.dtype)
        return h


class BlackGrailOfNightfireBrain(_BaseMLPBrain):
    brain_kind = "black_grail_of_nightfire"
    architecture_name = "scalar_gated_ray_interpretation"

    def _build_variant_layers(self) -> None:
        self.ray_gate = _ContextGate(self.ray_width, self.scalar_width)
        self.fusion_gate = _ContextGate(self.fusion_width, self.scalar_width)
        self.fusion_in, self.fusion_blocks, self.fusion_out_norm = self._build_shared_fusion_stack()

    def _condition_ray_tokens(self, ray_tokens: torch.Tensor, scalar_ctx: torch.Tensor) -> torch.Tensor:
        return self.ray_gate(ray_tokens, scalar_ctx)

    def _fuse(self, ray_summary: torch.Tensor, scalar_ctx: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([ray_summary, scalar_ctx], dim=-1)
        h = self.fusion_in(fused.to(dtype=self.fusion_in.weight.dtype))
        h = _activation_fn(h)
        h = self.fusion_gate(h, scalar_ctx)
        for block in self.fusion_blocks:
            h = block(h)
            h = self.fusion_gate(h, scalar_ctx)
        h = self.fusion_out_norm(h.float()).to(dtype=h.dtype)
        return h


def create_mlp_brain(kind: str, obs_dim: int, act_dim: int) -> nn.Module:
    norm = normalize_brain_kind(kind)
    if norm == "throne_of_ashen_dreams":
        return ThroneOfAshenDreamsBrain(obs_dim, act_dim)
    if norm == "veil_of_the_hollow_crown":
        return VeilOfTheHollowCrownBrain(obs_dim, act_dim)
    if norm == "black_grail_of_nightfire":
        return BlackGrailOfNightfireBrain(obs_dim, act_dim)
    raise ValueError(f"Unknown MLP brain kind: {kind!r}")
