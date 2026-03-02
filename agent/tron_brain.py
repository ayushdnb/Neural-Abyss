from __future__ import annotations
# ^ Defers evaluation of type annotations. This is valuable for:
#   - Forward references in type hints (types referenced before definition)
#   - Cleaner runtime behavior when type checking is performed externally
#   - Improved compatibility across Python versions and tooling

# This import ensures that type hints are evaluated as string literals,
# allowing forward references (e.g., 'TronBrain' in its own methods)
# and making the code compatible with older Python versions.

from typing import Dict, List, Tuple
# ^ Typing constructs used throughout:
#   - Dict[K, V]: mapping from keys to values
#   - List[T]: a list of items of type T
#   - Tuple[A, B]: a fixed-length tuple of two items

import math
# ^ Used for math.sqrt(2.0), which is a common gain factor for orthogonal initialization
#   when the network uses ReLU/GELU-like nonlinearities.

import torch
import torch.nn as nn
import torch.nn.functional as F
# ^ Core PyTorch imports:
#   - torch: tensors and general utilities
#   - torch.nn: neural network modules (layers, containers)
#   - torch.nn.functional: stateless neural network ops (activations, etc.)

import config
# ^ Project configuration module holding hyperparameters and constants.
#   Centralizing these values prevents subtle mismatches between:
#   - observation layout
#   - model architecture
#   - training runtime expectations


# ============================================================================
# Transformer block utilities used by TronBrain
# ============================================================================
# TronBrain is a transformer-style actor-critic network designed for
# reinforcement learning policies with discrete actions (logits output).
#
# The transformer core is composed of two reusable building blocks:
#   1) _SelfAttnBlock: self-attention + feed-forward, with residuals and LayerNorm
#   2) _CrossAttnBlock: cross-attention + feed-forward, with residuals and LayerNorm
#
# Definitions (mathematical):
#   Self-attention (single head) over a sequence X ∈ R^{B×S×D}:
#     Q = XW_Q, K = XW_K, V = XW_V
#     Attn(X) = softmax(QK^T / sqrt(d_k)) V
#
#   Cross-attention between a query sequence Qseq and a memory sequence M:
#     Q = Qseq W_Q, K = M W_K, V = M W_V
#     Attn(Qseq, M) = softmax(QK^T / sqrt(d_k)) V
#
# Multi-head attention:
#   - Performs attention in multiple subspaces (heads) in parallel.
#   - Each head uses d_k = D / n_heads.
#   - Outputs are concatenated and projected back to D.
#
# Residual connections:
#   - Add the sublayer output back to its input: x = x + sublayer(x)
#   - Improve gradient flow and training stability in deep networks.
#
# LayerNorm:
#   - Normalizes features across the last dimension D.
#   - Helps maintain stable activation scales.
# ============================================================================


class _SelfAttnBlock(nn.Module):
    """
    Pre-LN style transformer block: Self-Attention + Feed-Forward Network, with residual connections.

    Architecture (as implemented in this class):
        1) Self-attention: a = Attn(x, x, x)
        2) Residual + LayerNorm: x = LN(x + a)
        3) Feed-forward: f = FFN(x)
        4) Residual + LayerNorm: x = LN(x + f)

    Important note on terminology:
    - The docstring says "Pre-LN", but the code applies LayerNorm *after* the residual addition:
          x = norm1(x + a)
          x = norm2(x + f)
      This pattern is commonly described as "Post-LN" in transformer literature.
    - Regardless of naming, the implementation is unambiguous and correct as written.

    Shapes:
        Input:  x ∈ R^{B×S×D}
        Output: x ∈ R^{B×S×D}
    """

    def __init__(self, d_model: int, n_heads: int, ffn_mult: int = 4) -> None:
        """
        Initialize self-attention block.

        Args:
            d_model:
                Embedding dimension D for all tokens.
                Must be divisible by n_heads for MultiheadAttention.

            n_heads:
                Number of attention heads.

            ffn_mult:
                Multiplier for the feed-forward hidden dimension.
                FFN hidden dimension = ffn_mult * d_model.
                The default (4×) is standard in transformer architectures.
        """
        super().__init__()

        # Multi-head self-attention.
        # batch_first=True means inputs are (B, S, D) rather than (S, B, D).
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

        # LayerNorm applied after attention residual.
        self.norm1 = nn.LayerNorm(d_model)

        # Feed-forward network:
        #   Linear(D → 4D) → GELU → Linear(4D → D)
        #
        # GELU is a smooth non-linearity commonly used in modern transformers.
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.GELU(),
            nn.Linear(ffn_mult * d_model, d_model),
        )

        # LayerNorm applied after FFN residual.
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply self-attention + feed-forward with residuals and LayerNorm.

        Args:
            x:
                Input token sequence of shape (B, S, D).

        Returns:
            Output token sequence of shape (B, S, D).

        Mathematical interpretation:
            Let x be the current representation of the sequence.
            - Self-attention computes context-aware mixing across the S positions.
            - FFN applies a position-wise nonlinear transformation.
            - Residuals preserve information and help optimize deep stacks.
        """
        # Self-attention: Q=K=V=x.
        # need_weights=False saves memory/compute by not returning attention matrices.
        a, _ = self.attn(x, x, x, need_weights=False)

        # Residual connection then LayerNorm.
        x = self.norm1(x + a)

        # Feed-forward transformation.
        f = self.ffn(x)

        # Second residual then LayerNorm.
        x = self.norm2(x + f)
        return x


class _CrossAttnBlock(nn.Module):
    """
    Cross-Attention block: Query attends to Key-Value pairs, with FFN and residuals.

    Architecture (as implemented):
        1) Cross-attention: a = Attn(q, kv, kv)
        2) Residual + LayerNorm: q = LN(q + a)
        3) Feed-forward: f = FFN(q)
        4) Residual + LayerNorm: q = LN(q + f)

    Shapes:
        q  ∈ R^{B×Sq×D}   (query sequence)
        kv ∈ R^{B×Sk×D}   (memory/key-value sequence)
        output has shape R^{B×Sq×D}

    Purpose in TronBrain:
        - Decision/Semantic/Instinct/Memory tokens (queries) attend to ray tokens (kv).
        - This fuses perceptual information (rays) into the higher-level plan tokens.
    """

    def __init__(self, d_model: int, n_heads: int, ffn_mult: int = 4) -> None:
        """
        Initialize cross-attention block.

        Args:
            d_model:
                Embedding dimension D for query and memory tokens.
            n_heads:
                Number of attention heads.
            ffn_mult:
                FFN expansion multiplier (default 4).
        """
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.GELU(),
            nn.Linear(ffn_mult * d_model, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """
        Apply cross-attention where q attends to kv, followed by FFN.

        Args:
            q:
                Query tokens of shape (B, Sq, D).
                In TronBrain this is typically the plan token sequence.

            kv:
                Key/Value tokens of shape (B, Sk, D).
                In TronBrain this is the ray token sequence.

        Returns:
            Updated query tokens of shape (B, Sq, D).

        Mathematical interpretation:
            Each query token learns to "read" from the ray memory using attention weights:
                weights = softmax(QK^T / sqrt(d_k))
            producing a weighted sum of V that is added back to q via residual.
        """
        a, _ = self.attn(q, kv, kv, need_weights=False)
        q = self.norm1(q + a)
        f = self.ffn(q)
        q = self.norm2(q + f)
        return q


# ============================================================================
# TronBrain: Transformer-based actor-critic for per-agent control
# ============================================================================

class TronBrain(nn.Module):
    """
    TRON v1 Transformer-based brain for per-agent control.

    This model implements an actor-critic architecture suitable for PPO:
      - Actor produces action logits (for a discrete categorical policy).
      - Critic produces a scalar value estimate.

    Pipeline overview (4 stages):
      1) Ray encoder:
           Self-attention over ray tokens (perceptual input).
      2) Semantic encoder:
           Self-attention over "plan tokens":
             decision tokens + semantic tokens + instinct token + memory token
      3) Fusion:
           Cross-attention where plan tokens attend to ray tokens.
      4) Readout:
           Only decision tokens are used to produce logits and value.

    Forward contract (strict):
        forward(obs) -> (logits, value)
        logits: (B, act_dim)
        value:  (B, 1)

    Observation layout assumption (from config):
      [rays_flat (num_rays * ray_feat_dim) | rich_base | instinct]

    Notation:
      B  = batch size
      Nr = num_rays
      Fr = ray_feat_dim
      D  = d_model
      A  = act_dim
    """

    def __init__(self, obs_dim: int, act_dim: int) -> None:
        """
        Initialize TronBrain with observation and action dimensions.

        Args:
            obs_dim:
                Total observation dimension. This implementation requires it to match:
                  (num_rays * ray_feat_dim) + rich_base_dim + instinct_dim

            act_dim:
                Number of discrete actions.

        Raises:
            ValueError:
                If obs_dim does not match the expected layout or transformer config is invalid.

            RuntimeError:
                If SEMANTIC_RICH_BASE_INDICES is missing or semantic groups are misconfigured.
        """
        super().__init__()

        # Store dimensions as integers (avoid accidental tensor shapes).
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)

        # ------------------------------------------------------------------
        # Observation layout invariants
        # ------------------------------------------------------------------
        # These values define how the flat observation is interpreted.
        self.num_rays = int(getattr(config, "RAY_TOKEN_COUNT", 32))
        self.ray_feat_dim = int(getattr(config, "RAY_FEAT_DIM", 8))
        self.rich_base_dim = int(getattr(config, "RICH_BASE_DIM", 23))
        self.instinct_dim = int(getattr(config, "INSTINCT_DIM", 4))

        # Total rays flat dimension.
        self.rays_flat_dim = self.num_rays * self.ray_feat_dim

        # Total expected observation width.
        self.expected_obs_dim = self.rays_flat_dim + self.rich_base_dim + self.instinct_dim

        # Validate the caller-provided obs_dim matches the computed layout.
        if self.obs_dim != self.expected_obs_dim:
            raise ValueError(
                f"TronBrain obs_dim mismatch: got obs_dim={self.obs_dim}, "
                f"expected {self.expected_obs_dim} = ({self.num_rays}*{self.ray_feat_dim})"
                f"+{self.rich_base_dim}+{self.instinct_dim}."
            )
        cfg_obs_dim = int(getattr(config, "OBS_DIM", self.expected_obs_dim))
        if cfg_obs_dim != self.expected_obs_dim:
            raise ValueError(
                f"TronBrain config OBS layout mismatch: expected {self.expected_obs_dim} from "
                f"RAY_TOKEN_COUNT*RAY_FEAT_DIM + RICH_BASE_DIM + INSTINCT_DIM, but config.OBS_DIM={cfg_obs_dim}."
            )

        # ------------------------------------------------------------------
        # Semantic partition indices
        # ------------------------------------------------------------------
        # SEMANTIC_RICH_BASE_INDICES defines how rich_base is partitioned into semantic groups.
        idx_map = getattr(config, "SEMANTIC_RICH_BASE_INDICES", None)
        if idx_map is None:
            raise RuntimeError(
                "SEMANTIC_RICH_BASE_INDICES missing in config. "
                "Apply Phase 3 (semantic partitioning) before using TronBrain."
            )
        self._idx_map: Dict[str, List[int]] = dict(idx_map)

        # ------------------------------------------------------------------
        # Transformer hyperparameters (config-driven)
        # ------------------------------------------------------------------
        d_model = int(getattr(config, "TRON_D_MODEL", 64))
        n_heads = int(getattr(config, "TRON_HEADS", 4))
        ray_layers = int(getattr(config, "TRON_RAY_LAYERS", 4))
        sem_layers = int(getattr(config, "TRON_SEM_LAYERS", 2))
        fusion_layers = int(getattr(config, "TRON_FUSION_LAYERS", 2))
        mlp_hidden = int(getattr(config, "TRON_MLP_HIDDEN", 128))

        # Validate transformer configuration.
        if d_model <= 0 or n_heads <= 0:
            raise ValueError("Invalid TRON config: TRON_D_MODEL and TRON_HEADS must be positive.")
        if d_model % n_heads != 0:
            raise ValueError(
                f"Invalid TRON config: TRON_D_MODEL ({d_model}) must be divisible by TRON_HEADS ({n_heads})."
            )

        self.d_model = d_model
        self.n_heads = n_heads

        # ------------------------------------------------------------------
        # Ray input projection
        # ------------------------------------------------------------------
        # Raw ray features (Fr) are normalized and projected to model dimension D.
        self.ray_in_norm = nn.LayerNorm(self.ray_feat_dim)
        self.ray_in_proj = nn.Linear(self.ray_feat_dim, d_model)

        # Direction embedding acts like a learnable positional encoding for rays.
        # Shape (1, Nr, D), broadcast over batch.
        self.ray_dir_embed = nn.Parameter(torch.randn(1, self.num_rays, d_model) * 0.02)

        # ------------------------------------------------------------------
        # Semantic token embeddings
        # ------------------------------------------------------------------
        # Each semantic group corresponds to a distinct slice of rich_base.
        # Each slice may have a different dimension; all are projected to D.
        sem_keys = ["own_context", "world_context", "zone_context", "team_context", "combat_context"]
        self._sem_keys = sem_keys

        self.sem_in_norm = nn.ModuleDict()
        self.sem_in_proj = nn.ModuleDict()
        for k in sem_keys:
            idxs = self._idx_map.get(k, None)
            if idxs is None or len(idxs) == 0:
                raise RuntimeError(f"Missing/empty semantic index list for '{k}' in SEMANTIC_RICH_BASE_INDICES.")
            din = int(len(idxs))
            self.sem_in_norm[k] = nn.LayerNorm(din)
            self.sem_in_proj[k] = nn.Linear(din, d_model)

        # Instinct embedding: maps instinct_dim -> D.
        self.instinct_in_norm = nn.LayerNorm(self.instinct_dim)
        self.instinct_in_proj = nn.Linear(self.instinct_dim, d_model)

        # ------------------------------------------------------------------
        # Learned plan tokens
        # ------------------------------------------------------------------
        # Memory token: a single token representing persistent internal state.
        self.memory_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # Decision tokens: three learned tokens used for final readout.
        # These act as "designated slots" that the model can use to store decision-relevant state.
        self.decision_tokens = nn.Parameter(torch.randn(1, 3, d_model) * 0.02)

        # ------------------------------------------------------------------
        # Stage blocks
        # ------------------------------------------------------------------
        # Stage 1: Ray encoder (self-attention among rays).
        self.ray_encoder = nn.ModuleList([_SelfAttnBlock(d_model, n_heads) for _ in range(ray_layers)])

        # Stage 2: Semantic encoder (self-attention among plan tokens).
        self.sem_encoder = nn.ModuleList([_SelfAttnBlock(d_model, n_heads) for _ in range(sem_layers)])

        # Stage 3: Fusion (cross-attention, plan tokens attend to rays).
        self.fusion = nn.ModuleList([_CrossAttnBlock(d_model, n_heads) for _ in range(fusion_layers)])

        # ------------------------------------------------------------------
        # Readout heads (decision tokens only)
        # ------------------------------------------------------------------
        # Concatenate 3 decision tokens into (B, 3D), then apply MLP -> logits/value.
        self.read_fc0 = nn.Linear(3 * d_model, mlp_hidden)
        self.read_fc1 = nn.Linear(mlp_hidden, mlp_hidden)
        self.actor = nn.Linear(mlp_hidden, self.act_dim)
        self.critic = nn.Linear(mlp_hidden, 1)

        # Initialize weights.
        self._init_weights()

    def _init_weights(self) -> None:
        """
        Orthogonal initialization (PPO-friendly):
          - Hidden linear layers use gain = sqrt(2).
          - Actor head uses gain = 0.01 to keep initial logits small.
          - Critic head uses gain = 1.0.

        Mathematical background:
        - Orthogonal initialization constructs weight matrices whose columns/rows are orthonormal.
          For square matrices: W^T W ≈ I.
        - This helps preserve the norm of signals propagated through linear layers:
            ||Wx|| ≈ ||x||   (in expectation, depending on shape and gain)
        - With nonlinear activations, a gain is used to maintain variance after the nonlinearity.
          sqrt(2) is a common heuristic for ReLU-like activations and is often used with GELU.

        PPO-specific motivation:
        - Small actor gain biases the initial policy toward near-uniform distributions:
            If logits ≈ 0, softmax(logits) ≈ uniform
          which improves early exploration.
        """
        gain_hidden = math.sqrt(2.0)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=gain_hidden)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        if self.actor.bias is not None:
            nn.init.zeros_(self.actor.bias)

        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        if self.critic.bias is not None:
            nn.init.zeros_(self.critic.bias)

    # ------------------------------------------------------------------
    # Embedding helpers (stable + AMP-friendly)
    # ------------------------------------------------------------------
    # These helpers apply LayerNorm in float32 then cast back to projection dtype.
    # This pattern is commonly used for mixed precision training:
    #   - LayerNorm computations are more stable in fp32.
    #   - Linear projections can run in fp16/bf16 for performance.

    def _embed_rays(self, rays_raw: torch.Tensor) -> torch.Tensor:
        """
        Embed raw ray features to model dimension.

        Args:
            rays_raw:
                (B, num_rays, ray_feat_dim)

        Returns:
            (B, num_rays, d_model)
        """
        x = self.ray_in_norm(rays_raw.float())
        x = self.ray_in_proj(x.to(dtype=self.ray_in_proj.weight.dtype))
        return x

    def _embed_sem(self, x_raw: torch.Tensor, key: str) -> torch.Tensor:
        """
        Embed raw semantic features for a given group.

        Args:
            x_raw:
                (B, group_dim) raw semantic slice selected from rich_base.
            key:
                semantic group name.

        Returns:
            (B, d_model)
        """
        n = self.sem_in_norm[key]
        p = self.sem_in_proj[key]
        x = n(x_raw.float())
        x = p(x.to(dtype=p.weight.dtype))
        return x

    def _embed_instinct(self, inst_raw: torch.Tensor) -> torch.Tensor:
        """
        Embed raw instinct features to model dimension.

        Args:
            inst_raw:
                (B, instinct_dim)

        Returns:
            (B, d_model)
        """
        x = self.instinct_in_norm(inst_raw.float())
        x = self.instinct_in_proj(x.to(dtype=self.instinct_in_proj.weight.dtype))
        return x

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through TronBrain.

        Args:
            obs:
                Flat observation tensor of shape (B, expected_obs_dim).
                Layout: [rays_flat | rich_base | instinct]

        Returns:
            logits:
                (B, act_dim) policy logits for a discrete action distribution.
            value:
                (B, 1) scalar state value estimate.

        Implementation stages:
            1) Split flat observation into rays/rich_base/instinct.
            2) Encode ray tokens with self-attention.
            3) Build plan tokens (decision + semantic + instinct + memory) and encode them.
            4) Fuse plan tokens with ray tokens via cross-attention.
            5) Read out logits/value from decision tokens only.

        Strict invariants:
            - Input shape must match expected layout.
            - Output shapes must match PPO runtime expectations exactly.
        """
        # ------------------------------------------------------------------
        # Input shape checks (fail loudly on schema mismatch)
        # ------------------------------------------------------------------
        if obs.dim() != 2:
            raise RuntimeError(f"TronBrain.forward expects obs 2D [B,F], got shape={tuple(obs.shape)}")
        B, Fdim = int(obs.size(0)), int(obs.size(1))
        if Fdim != self.expected_obs_dim:
            raise RuntimeError(
                f"TronBrain.forward obs dim mismatch: got F={Fdim}, expected {self.expected_obs_dim}."
            )

        # ------------------------------------------------------------------
        # Split observation into components
        # ------------------------------------------------------------------
        rays_flat = obs[:, : self.rays_flat_dim]
        rich_base = obs[:, self.rays_flat_dim : self.rays_flat_dim + self.rich_base_dim]
        instinct = obs[:, self.rays_flat_dim + self.rich_base_dim : self.expected_obs_dim]

        # ------------------------------------------------------------------
        # Stage 1: Ray processing (self-attention among rays)
        # ------------------------------------------------------------------
        rays_raw = rays_flat.view(B, self.num_rays, self.ray_feat_dim)
        ray_tok = self._embed_rays(rays_raw)
        ray_tok = ray_tok + self.ray_dir_embed.to(dtype=ray_tok.dtype, device=ray_tok.device)

        for blk in self.ray_encoder:
            ray_tok = blk(ray_tok)

        # ------------------------------------------------------------------
        # Stage 2: Semantic token construction + plan token encoding
        # ------------------------------------------------------------------
        # Build semantic tokens by selecting configured index subsets from rich_base.
        # Each group yields a token of shape (B, 1, D).
        sem_tokens: List[torch.Tensor] = []
        for k in self._sem_keys:
            idxs = self._idx_map[k]
            xk = rich_base.index_select(dim=1, index=torch.tensor(idxs, device=rich_base.device, dtype=torch.long))
            sem_tokens.append(self._embed_sem(xk, k).unsqueeze(1))

        # Instinct token.
        inst_tok = self._embed_instinct(instinct).unsqueeze(1)

        # Expand learned tokens to batch dimension.
        dec = self.decision_tokens.expand(B, -1, -1).to(device=obs.device)
        mem = self.memory_token.expand(B, -1, -1).to(device=obs.device)

        # Concatenate plan token sequence:
        #   3 decision + 5 semantic + 1 instinct + 1 memory = 10 tokens
        sem = torch.cat(sem_tokens, dim=1)
        tok = torch.cat([dec, sem, inst_tok, mem], dim=1)

        # Self-attention over plan token sequence.
        for blk in self.sem_encoder:
            tok = blk(tok)

        # ------------------------------------------------------------------
        # Stage 3: Fusion (plan tokens attend to rays)
        # ------------------------------------------------------------------
        for blk in self.fusion:
            tok = blk(tok, ray_tok)

        # ------------------------------------------------------------------
        # Stage 4: Readout (decision tokens only)
        # ------------------------------------------------------------------
        # Decision tokens are the first 3 tokens in tok.
        dec_out = tok[:, :3, :].reshape(B, 3 * self.d_model)

        h = F.gelu(self.read_fc0(dec_out))
        h = F.gelu(self.read_fc1(h))

        logits = self.actor(h)
        value = self.critic(h)

        # ------------------------------------------------------------------
        # Output shape checks (strict PPO contract)
        # ------------------------------------------------------------------
        if logits.dim() != 2 or logits.size(0) != B or logits.size(1) != self.act_dim:
            raise RuntimeError(
                f"Bad logits shape from TronBrain: got {tuple(logits.shape)}, expected ({B},{self.act_dim})"
            )
        if value.dim() != 2 or value.size(0) != B or value.size(1) != 1:
            raise RuntimeError(
                f"Bad value shape from TronBrain: got {tuple(value.shape)}, expected ({B},1)"
            )

        return logits, value