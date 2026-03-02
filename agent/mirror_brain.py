from __future__ import annotations
# ^ Postpones evaluation of type annotations. This is useful when:
#   - Using forward references in type hints.
#   - Keeping runtime overhead small while still benefiting from static type checking.

# Import necessary modules from the standard library and PyTorch.
from typing import Dict, Tuple, List, Optional
import math                         # Used for sqrt(2) gain in orthogonal initialization

import torch
import torch.nn as nn
import torch.nn.functional as F

import config
from . import obs_spec


# ---------------------------------------------------------------------
# MirrorBrain: two-pass (propose -> reflect/edit) policy/value network.
#
# This module is designed for PPO-style actor-critic reinforcement learning.
# It produces:
#   - Policy logits for a discrete action space (used to sample actions)
#   - A scalar value estimate (used for advantage estimation and critic loss)
#
# High-level architecture:
#   PASS 1 (PROPOSE): build an initial action distribution and value estimate.
#   PASS 2 (REFLECT/EDIT): produce a small residual correction to the proposal.
#
# Hard invariants:
# - Input observation is a flat vector (B, OBS_DIM) using the same layout as TronBrain.
# - Output contract matches the PPO runtime:
#     logits: (B, NUM_ACTIONS)
#     value:  (B, 1)
#
# Notation used throughout:
#   B = batch size
#   N = number of ray tokens (e.g., 32)
#   F = ray feature dimension (e.g., 4)
#   D = model embedding dimension (d_model)
#   A = number of discrete actions (act_dim)
# ---------------------------------------------------------------------


class _SelfAttnBlock(nn.Module):
    """
    Pre-LN self-attention block (Transformer encoder style).

    "Pre-LN" (Pre-LayerNorm) means:
      - Apply LayerNorm before attention and before the feed-forward sublayer.
      - Then add residual connections.

    This is commonly used because it tends to improve training stability in deep transformers.

    Mathematical overview (single block):
      Given input X ∈ R^{B×S×D} (S = sequence length, D = embedding dim):

      1) X1 = LN(X)
      2) A  = SelfAttention(X1)          # multi-head attention over the sequence
      3) X  = X + A                     # residual
      4) X2 = LN(X)
      5) FF = MLP(X2)                   # position-wise feed-forward
      6) X  = X + FF                    # residual

    The attention mechanism:
      Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

    In self-attention:
      Q = K = V = X1 (the same sequence).
    """

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        # LayerNorm before self-attention (Pre-LN). Normalizes across the last dimension (D).
        self.norm1 = nn.LayerNorm(d_model)

        # Multi-head self-attention.
        # batch_first=True => input/output shape is (B, S, D) rather than (S, B, D).
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

        # LayerNorm before feed-forward sublayer.
        self.norm2 = nn.LayerNorm(d_model)

        # Feed-forward network:
        #   Linear(D → 4D) → GELU → Linear(4D → D)
        # This is the standard Transformer "MLP" expansion pattern.
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, S, D)

        Returns:
            x: (B, S, D), after attention + feed-forward with residual connections.

        Precision details:
        - LayerNorm is executed in float32 (fp32) for numerical stability.
        - Output is cast back to x.dtype (e.g., float16/bfloat16) to be AMP-friendly.
        """
        # LayerNorm in fp32 for stability, then cast back for AMP-friendly projections.
        x1 = self.norm1(x.float()).to(dtype=x.dtype)

        # Self-attention with queries, keys, values all equal to x1.
        # need_weights=False avoids returning attention weights (saves memory/compute).
        a, _ = self.attn(x1, x1, x1, need_weights=False)

        # Residual connection: preserve information flow and improve optimization.
        x = x + a

        # Second LayerNorm (also fp32 for stability).
        x2 = self.norm2(x.float()).to(dtype=x.dtype)

        # Feed-forward network with residual connection.
        x = x + self.ff(x2)
        return x


class _CrossAttnBlock(nn.Module):
    """
    Pre-LN cross-attention block: queries from x attend to memory m.

    Cross-attention differs from self-attention in that:
      - Queries come from one sequence (x).
      - Keys and values come from another sequence (m).

    Mathematical overview:
      Let X ∈ R^{B×Sx×D} and M ∈ R^{B×Sm×D}.

      1) Q = LN(X)
      2) K,V = LN(M)
      3) A = Attention(Q, K, V)
      4) X = X + A
      5) X2 = LN(X)
      6) X = X + MLP(X2)

    This is used here to:
      - Let "plan" tokens (queries) attend to "ray" tokens (memory),
        integrating local perceptual details into a higher-level plan representation.
    """

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        # LayerNorm for the query input (x).
        self.norm_q = nn.LayerNorm(d_model)

        # LayerNorm for the memory input (m).
        self.norm_m = nn.LayerNorm(d_model)

        # Multi-head attention for cross-attention.
        # Queries use x, keys/values use m.
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

        # LayerNorm after the attention residual.
        self.norm2 = nn.LayerNorm(d_model)

        # Feed-forward network.
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, Sx, D) query sequence
            m: (B, Sm, D) memory sequence (keys/values)

        Returns:
            x: (B, Sx, D) updated by attending to m.

        Precision details:
        - LayerNorm is done in fp32 for stability.
        - Cast back to original dtype for AMP-friendly attention projections.
        """
        # Apply LN to queries and memory (fp32 for stability, then cast back).
        q = self.norm_q(x.float()).to(dtype=x.dtype)
        kv = self.norm_m(m.float()).to(dtype=m.dtype)

        # Cross-attention: q attends to kv.
        a, _ = self.attn(q, kv, kv, need_weights=False)

        # Residual addition.
        x = x + a

        # LN + feed-forward + residual.
        x2 = self.norm2(x.float()).to(dtype=x.dtype)
        x = x + self.ff(x2)
        return x


class MirrorBrain(nn.Module):
    """
    MirrorBrain = Tron-family tokenization + propose logits/value + small reflection edit.

    This is an actor-critic network specialized for a flat observation layout that is
    tokenized into multiple groups:
      - Ray tokens (perceptual/line-of-sight style features)
      - Semantic tokens (structured slices of a rich feature vector)
      - Instinct token (small dense context)
      - Decision tokens (learned tokens representing internal decision-making slots)
      - Memory token (learned token to carry persistent state within the network)

    Why two passes?
    - PASS 1 produces a standard policy/value estimate, similar to a transformer-based policy.
    - PASS 2 produces a residual correction initialized near zero:
        logits_final = logits_proposal + delta_logits
        value_final  = value_proposal  + delta_value
      This design ensures initial behavior closely matches PASS 1, while allowing the model
      to learn a "reflection" process that adjusts decisions using internal uncertainty signals.

    PASS 1 (PROPOSE):
      - Encode rays (self-attn over ray tokens)
      - Build plan tokens: 3 decision + 5 semantic + 1 instinct + 1 memory = 10
      - Self-attn over plan tokens
      - Cross-attn fusion: plan tokens attend to rays
      - Readout from decision tokens -> logits_proposal, value_proposal

    PASS 2 (REFLECT + EDIT):
      - Build 1 reflection token REF from internal state only:
          mean(decision tokens), entropy/margin(logits_proposal), value_proposal
      - REF cross-attends to plan tokens and ray tokens (small)
      - Output delta_logits and delta_value (both initialized ~0)
      - logits_final = logits_proposal + delta_logits
        value_final  = value_proposal  + delta_value
    """

    def __init__(self, obs_dim: int, act_dim: int) -> None:
        super().__init__()
        # Store dimensions for validation and runtime checks.
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)

        # --- Observation layout MUST match obs_spec / TronBrain exactly ---
        #
        # The observation is a flat vector, but it conceptually contains:
        #   1) Ray features: num_rays × ray_feat_dim, flattened
        #   2) Rich base vector: rich_base_dim
        #   3) Instinct vector: instinct_dim
        #
        # This class enforces that the configured layout matches the expected layout,
        # to prevent silent bugs from changing feature ordering.
        self.num_rays = int(getattr(config, "RAY_TOKEN_COUNT", 32))
        self.ray_feat_dim = int(getattr(config, "RAY_FEAT_DIM", 8))
        self.rays_flat_dim = self.num_rays * self.ray_feat_dim

        self.rich_base_dim = int(getattr(config, "RICH_BASE_DIM", 64))
        self.rich_base_dim = int(getattr(config, "RICH_BASE_DIM", 23))
        self.instinct_dim = int(getattr(config, "INSTINCT_DIM", 4))
        # Expected observation dimension based on the parts above.
        expected_obs_dim = self.rays_flat_dim + self.rich_base_dim + self.instinct_dim

        # config.OBS_DIM is validated against the computed layout to ensure consistency.
        cfg_obs_dim = int(getattr(config, "OBS_DIM", expected_obs_dim))
        if cfg_obs_dim != expected_obs_dim:
            raise RuntimeError(
                f"[MirrorBrain] OBS layout mismatch: expected {expected_obs_dim} from "
                f"RAY_TOKEN_COUNT*RAY_FEAT_DIM + RICH_BASE_DIM + INSTINCT_DIM, but config.OBS_DIM={cfg_obs_dim}"
            )
        if self.obs_dim != expected_obs_dim:
            raise RuntimeError(
                f"[MirrorBrain] obs_dim mismatch: ctor obs_dim={self.obs_dim} but expected {expected_obs_dim} (must match Tron/obs_spec)."
            )

        # --- Model hyperparameters ---
        #
        # d_model: embedding dimension used by attention blocks.
        # n_heads: number of attention heads.
        #
        # These default to TronBrain settings when Mirror-specific values are absent,
        # ensuring comparable compute cost across brain variants.
        d_model = int(getattr(config, "MIRROR_D_MODEL", getattr(config, "TRON_D_MODEL", 64)))
        n_heads = int(getattr(config, "MIRROR_HEADS", getattr(config, "TRON_HEADS", 4)))

        # Depth controls:
        # - ray_layers: self-attention depth for ray tokens
        # - plan_layers: self-attention depth for plan tokens
        # - fusion_layers: cross-attention depth (plan attends to rays)
        ray_layers = int(getattr(config, "MIRROR_RAY_LAYERS", getattr(config, "TRON_RAY_LAYERS", 4)))
        plan_layers = int(getattr(config, "MIRROR_SEM_LAYERS", getattr(config, "TRON_SEM_LAYERS", 3)))
        fusion_layers = int(getattr(config, "MIRROR_FUSION_LAYERS", getattr(config, "TRON_FUSION_LAYERS", 2)))

        # Head MLP size for decision readout.
        mlp_hidden = int(getattr(config, "MIRROR_MLP_HIDDEN", getattr(config, "TRON_MLP_HIDDEN", 256)))

        self.d_model = d_model

        # Multi-head attention requires D to be divisible by number of heads.
        # Each head gets dimension d_k = D / n_heads.
        if d_model % n_heads != 0:
            raise RuntimeError(f"[MirrorBrain] d_model must be divisible by n_heads (d_model={d_model}, n_heads={n_heads}).")

        # -----------------------------------------------------------------
        # PASS 1: Embeddings
        # -----------------------------------------------------------------

        # Rays embedding:
        #   rays_raw: (B, num_rays, ray_feat_dim)
        #   -> LayerNorm over feature dim
        #   -> Linear projection to d_model
        self.ray_in_norm = nn.LayerNorm(self.ray_feat_dim)
        self.ray_in_proj = nn.Linear(self.ray_feat_dim, d_model)

        # Learnable ray direction embedding:
        # - Shape (1, num_rays, d_model) broadcasts across batch.
        # - Acts like a positional encoding, but specific to ray index/direction.
        # - Small init scale (0.02) is common to keep early activations controlled.
        self.ray_dir_embed = nn.Parameter(torch.randn(1, self.num_rays, d_model) * 0.02)

        # Semantic token projections:
        # A rich_base vector is partitioned into five semantic slices via indices.
        # Each slice is independently normalized and projected into d_model.
        self.sem_in_norm = nn.ModuleDict()
        self.sem_in_proj = nn.ModuleDict()

        # These keys must match what obs_spec.build_semantic_tokens produces.
        sem_keys = ["own_context", "world_context", "zone_context", "team_context", "combat_context"]

        # The mapping config.SEMANTIC_RICH_BASE_INDICES is expected to contain:
        #   key -> list of indices in rich_base corresponding to that semantic token.
        #
        # This enforces a stable, explicit schema for structured features.
        sem_idx_map: Dict[str, List[int]] = dict(getattr(config, "SEMANTIC_RICH_BASE_INDICES", {}))
        for k in sem_keys:
            idxs = sem_idx_map.get(k, None)
            if idxs is None or len(idxs) == 0:
                raise RuntimeError(f"[MirrorBrain] Missing/empty semantic index list for '{k}' in SEMANTIC_RICH_BASE_INDICES.")
            din = int(len(idxs))
            self.sem_in_norm[k] = nn.LayerNorm(din)
            self.sem_in_proj[k] = nn.Linear(din, d_model)

        # Instinct embedding:
        # Instinct features are a small dense vector that becomes a single token.
        self.instinct_in_norm = nn.LayerNorm(self.instinct_dim)
        self.instinct_in_proj = nn.Linear(self.instinct_dim, d_model)

        # Learnable plan tokens:
        # - memory_token: a single token initialized to zeros (neutral baseline)
        # - decision_tokens: 3 learned tokens representing decision "slots"
        #
        # These are parameters, not derived from observation directly.
        self.memory_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.decision_tokens = nn.Parameter(torch.randn(1, 3, d_model) * 0.02)

        # -----------------------------------------------------------------
        # PASS 1: Encoders / Fusion
        # -----------------------------------------------------------------

        # Self-attention encoder over rays.
        self.ray_encoder = nn.ModuleList([_SelfAttnBlock(d_model, n_heads) for _ in range(ray_layers)])

        # Self-attention encoder over plan tokens (decision + semantic + instinct + memory).
        self.plan_encoder = nn.ModuleList([_SelfAttnBlock(d_model, n_heads) for _ in range(plan_layers)])

        # Cross-attention fusion blocks: plan tokens attend to ray tokens.
        self.fusion = nn.ModuleList([_CrossAttnBlock(d_model, n_heads) for _ in range(fusion_layers)])

        # -----------------------------------------------------------------
        # PASS 1: Readout heads (decision tokens only)
        # -----------------------------------------------------------------
        #
        # Readout uses ONLY the first 3 plan tokens (decision tokens).
        # The decision tokens are concatenated into a single vector:
        #   (B, 3, D) -> reshape -> (B, 3D)
        #
        # Then a small MLP produces:
        #   - actor logits: (B, A)
        #   - critic value: (B, 1)
        self.read_fc0 = nn.Linear(3 * d_model, mlp_hidden)
        self.read_fc1 = nn.Linear(mlp_hidden, mlp_hidden)
        self.actor = nn.Linear(mlp_hidden, self.act_dim)
        self.critic = nn.Linear(mlp_hidden, 1)

        # -----------------------------------------------------------------
        # PASS 2: Reflection token and delta heads
        # -----------------------------------------------------------------
        #
        # Reflection token construction:
        # - uses internal summaries only (no additional observation information)
        # - includes:
        #     mean(decision tokens) : vector in R^D
        #     entropy(logits_prop) : scalar (uncertainty measure)
        #     margin(logits_prop)  : scalar (confidence gap between top-1 and top-2)
        #     value_prop           : scalar (critic estimate)
        #
        # Total input dim = D + 3.
        self.ref_in_norm = nn.LayerNorm(d_model + 3)
        self.ref_in_proj = nn.Linear(d_model + 3, d_model)

        # REF cross-attends to plan tokens and ray tokens (kept small for cost control).
        self.ref_attend_plan = _CrossAttnBlock(d_model, n_heads)
        self.ref_attend_rays = _CrossAttnBlock(d_model, n_heads)

        # Delta heads:
        # - produce residual corrections to logits and value
        # - these are initialized close to zero so early training behaves like PASS 1
        self.delta_fc0 = nn.Linear(d_model, mlp_hidden)
        self.delta_fc1 = nn.Linear(mlp_hidden, mlp_hidden)
        self.delta_actor = nn.Linear(mlp_hidden, self.act_dim)
        self.delta_critic = nn.Linear(mlp_hidden, 1)

        # Initialize weights.
        self._init_weights()

        # Note:
        # The comment indicates that separate delta zeroing was integrated into _init_weights.
        # As written, _init_weights performs zero-gain orthogonal initialization for delta heads.

    def _init_weights(self) -> None:
        """
        Orthogonal initialization (PPO-friendly). Keeps delta heads ~0 at start.

        What is orthogonal initialization?
        - A matrix W is orthogonal if W^T W = I (for square matrices).
        - For non-square matrices, "orthogonal init" constructs a matrix with
          orthonormal columns or rows, preserving norms in the corresponding subspace.

        Why it matters:
        - Preserving activation variance helps stabilize deep networks.
        - Orthogonal initialization can reduce vanishing/exploding gradients,
          especially when combined with residual connections and normalization.

        What is "gain"?
        - Initialization gain scales the initialized weight matrix.
        - For many nonlinearities, a gain of sqrt(2) is a common heuristic
          that keeps forward/backward variance in a reasonable range.
          This is historically associated with ReLU-like activations; GELU is often
          treated similarly in practice.

        This function applies:
        - gain_hidden = sqrt(2) to most Linear layers
        - actor head gain = 0.01 to start with small logits magnitude (near-uniform policy)
        - critic head gain = 1.0 as a standard scale for value output
        - delta actor/critic gain = 0.0 so residual corrections begin exactly at zero
        """
        gain_hidden = math.sqrt(2.0)

        # Apply orthogonal init to every Linear layer in this module tree.
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=gain_hidden)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Base actor:
        # Small gain -> logits near zero -> softmax(logits) near uniform distribution early.
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        if self.actor.bias is not None:
            nn.init.zeros_(self.actor.bias)

        # Base critic:
        # Standard gain to avoid overly tiny value outputs at initialization.
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        if self.critic.bias is not None:
            nn.init.zeros_(self.critic.bias)

        # Delta heads:
        # gain=0.0 means the initialized weights become exactly zero (given orthogonal_ scaling).
        # This ensures delta_logits and delta_value start at 0, making PASS 2 a true "edit" path.
        nn.init.orthogonal_(self.delta_actor.weight, gain=0.0)
        if self.delta_actor.bias is not None:
            nn.init.zeros_(self.delta_actor.bias)

        nn.init.orthogonal_(self.delta_critic.weight, gain=0.0)
        if self.delta_critic.bias is not None:
            nn.init.zeros_(self.delta_critic.bias)

    # -----------------------------------------------------------------
    # Embedding helpers (numerically stable + AMP-friendly)
    # -----------------------------------------------------------------

    def _embed_rays(self, rays_raw: torch.Tensor) -> torch.Tensor:
        """
        Embed raw ray features.

        Args:
            rays_raw: (B, num_rays, ray_feat_dim)

        Returns:
            (B, num_rays, d_model)

        Rationale:
        - LayerNorm is applied in float32 for stability when using mixed precision.
        - Projection is done in the layer's parameter dtype (fp16/bf16/fp32) for performance.
        """
        x = self.ray_in_norm(rays_raw.float())
        x = self.ray_in_proj(x.to(dtype=self.ray_in_proj.weight.dtype))
        return x

    def _embed_sem(self, x_raw: torch.Tensor, key: str) -> torch.Tensor:
        """
        Embed a raw semantic token slice.

        Args:
            x_raw: (B, din) where din depends on the semantic slice indices
            key: semantic key string

        Returns:
            (B, d_model)

        Notes:
        - This returns (B, d_model) and callers often add a token dimension via unsqueeze(1).
        """
        n = self.sem_in_norm[key]
        p = self.sem_in_proj[key]
        x = n(x_raw.float())
        x = p(x.to(dtype=p.weight.dtype))
        return x

    def _embed_instinct(self, inst_raw: torch.Tensor) -> torch.Tensor:
        """
        Embed raw instinct features.

        Args:
            inst_raw: (B, instinct_dim)

        Returns:
            (B, d_model)
        """
        x = self.instinct_in_norm(inst_raw.float())
        x = self.instinct_in_proj(x.to(dtype=self.instinct_in_proj.weight.dtype))
        return x

    @staticmethod
    def _logits_entropy_and_margin(logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute two scalar summaries of policy uncertainty from logits:
          1) Entropy of the implied categorical distribution
          2) Margin between the top-1 and top-2 logits

        These are computed without sampling and without environment inputs.

        Entropy definition:
          Let p = softmax(logits).
          Entropy H(p) = - Σ_a p(a) log p(a)

        Interpretation:
          - High entropy means the policy is uncertain / spread across many actions.
          - Low entropy means the policy is confident / peaked.

        Margin definition:
          margin = top1_logit - top2_logit
        Interpretation:
          - Large margin => the best action is significantly preferred (high confidence).
          - Small margin => top actions are close (ambiguity).

        Returns:
          ent:    (B,1)
          margin: (B,1)

        Numerical details:
        - Computation is performed in float32 for stability.
        """
        logits32 = logits.to(torch.float32)
        logp = F.log_softmax(logits32, dim=-1)
        p = logp.exp()
        ent = -(p * logp).sum(dim=-1, keepdim=True)

        top2 = torch.topk(logits32, k=2, dim=-1).values
        margin = (top2[:, 0:1] - top2[:, 1:2])
        return ent, margin

    def _build_ref_token(
        self,
        dec_tokens: torch.Tensor,          # (B,3,D)
        logits_proposal: torch.Tensor,     # (B,A)
        value_proposal: torch.Tensor,      # (B,1)
    ) -> torch.Tensor:
        """
        Build the reflection token REF as (B,1,D) from internal state only.

        Construction:
          - Compute mean of decision tokens: mean over the 3 decision slots
              dec_mean ∈ R^{B×D}
          - Compute entropy and margin from logits_proposal:
              ent ∈ R^{B×1}, margin ∈ R^{B×1}
          - Include value_proposal:
              v ∈ R^{B×1}
          - Concatenate to form a feature vector:
              feat = [dec_mean, ent, margin, v] ∈ R^{B×(D+3)}
          - Normalize and project to model space:
              REF = Linear(LN(feat)) ∈ R^{B×D}
          - Add a token dimension:
              REF.unsqueeze(1) ∈ R^{B×1×D}

        Why this is "internal-state only":
        - It does not directly use raw observation values.
        - It summarizes the internal representation (decision tokens) and the model's
          own uncertainty/confidence about the proposal (entropy/margin) plus its value estimate.
        """
        if dec_tokens.dim() != 3 or dec_tokens.size(1) != 3 or dec_tokens.size(2) != self.d_model:
            raise RuntimeError(f"[MirrorBrain] bad dec_tokens shape for REF: got {tuple(dec_tokens.shape)}, expected (B,3,{self.d_model})")

        dec_mean = dec_tokens.mean(dim=1).to(torch.float32)
        ent, margin = self._logits_entropy_and_margin(logits_proposal)
        v = value_proposal.to(torch.float32)
        if v.dim() != 2 or v.size(1) != 1:
            raise RuntimeError(f"[MirrorBrain] bad value_proposal shape for REF: got {tuple(v.shape)}, expected (B,1)")

        feat = torch.cat([dec_mean, ent, margin, v], dim=-1)
        x = self.ref_in_norm(feat)
        x = self.ref_in_proj(x.to(dtype=self.ref_in_proj.weight.dtype))
        return x.unsqueeze(1)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: produce (logits, value).

        Args:
            obs: Flat observation tensor (B, OBS_DIM).
                 The layout must match obs_spec / TronBrain.

        Returns:
            logits_final: (B, act_dim)
            value_final:  (B, 1)

        This method enforces shape invariants explicitly to prevent silent errors.
        """
        if obs.dim() != 2:
            raise RuntimeError(f"[MirrorBrain] expected obs to be 2D (B,OBS_DIM); got shape {tuple(obs.shape)}")
        B, D = obs.shape
        if D != self.obs_dim:
            raise RuntimeError(f"[MirrorBrain] obs feature dim mismatch: got {D}, expected {self.obs_dim}")

        # Split observation into (rays_flat, rich_base, instinct) using a shared function.
        # This is a critical invariant: it ensures identical feature semantics across brains.
        rays_flat, rich_base, instinct = obs_spec.split_obs_flat(obs)

        # ================================================================
        # PASS 1: PROPOSE
        # ================================================================

        # Rays -> token sequence
        rays_raw = rays_flat.view(B, self.num_rays, self.ray_feat_dim)
        ray_tok = self._embed_rays(rays_raw)

        # Add learnable direction embedding (a learned per-ray positional encoding).
        ray_tok = ray_tok + self.ray_dir_embed.to(device=obs.device, dtype=ray_tok.dtype)

        # Encode rays using stacked self-attention blocks.
        for blk in self.ray_encoder:
            ray_tok = blk(ray_tok)

        # Build raw semantic tokens from rich_base and instinct via obs_spec.
        # The comment indicates that this uses cached index tensors and does not change layout.
        sem_raw = obs_spec.build_semantic_tokens(rich_base, instinct)

        # Embed five semantic tokens, each becoming one token in model space.
        sem_keys = ["own_context", "world_context", "zone_context", "team_context", "combat_context"]
        sem_tokens = []
        for k in sem_keys:
            sem_tokens.append(self._embed_sem(sem_raw[k], k).unsqueeze(1))
        sem = torch.cat(sem_tokens, dim=1)  # (B,5,D)

        # Embed instinct context as one token.
        inst_tok = self._embed_instinct(sem_raw["instinct_context"]).unsqueeze(1)  # (B,1,D)

        # Expand learnable decision and memory tokens across the batch.
        # - expand does not allocate new memory for each batch element (broadcast view).
        # - .to(device=obs.device) ensures parameters reside on correct device at runtime.
        dec = self.decision_tokens.expand(B, -1, -1).to(device=obs.device)  # (B,3,D)
        mem = self.memory_token.expand(B, -1, -1).to(device=obs.device)     # (B,1,D)

        # Plan token sequence: (3 decision + 5 semantic + 1 instinct + 1 memory = 10 tokens)
        tok = torch.cat([dec, sem, inst_tok, mem], dim=1)  # (B,10,D)
        if tok.size(1) != 10:
            raise RuntimeError(f"[MirrorBrain] plan token length must be 10; got {int(tok.size(1))}")

        # Self-attention over plan tokens.
        for blk in self.plan_encoder:
            tok = blk(tok)

        # Cross-attention fusion: plan tokens attend to ray tokens.
        # This integrates perceptual features (rays) into the plan state.
        for blk in self.fusion:
            tok = blk(tok, ray_tok)

        # Readout: take decision tokens only (first 3 plan tokens).
        # Reshape (B,3,D) -> (B, 3D) by concatenation.
        dec_out = tok[:, :3, :].reshape(B, 3 * self.d_model)

        # MLP readout with GELU nonlinearity.
        h = F.gelu(self.read_fc0(dec_out))
        h = F.gelu(self.read_fc1(h))

        # Proposal policy logits and value.
        logits_prop = self.actor(h)   # (B,A)
        value_prop = self.critic(h)   # (B,1)

        # ================================================================
        # PASS 2: REFLECT + EDIT
        # ================================================================

        # Build reflection token from internal state only.
        ref = self._build_ref_token(tok[:, :3, :], logits_prop, value_prop).to(dtype=tok.dtype, device=tok.device)  # (B,1,D)

        # Reflection attends to plan tokens then rays.
        # This allows the reflection state to consider:
        #   - The plan representation (decision/semantic/instinct/memory tokens)
        #   - The perceptual ray representation
        ref = self.ref_attend_plan(ref, tok)
        ref = self.ref_attend_rays(ref, ray_tok)

        # Convert (B,1,D) -> (B,D) and compute residual deltas.
        ref_h = ref.squeeze(1)
        dh = F.gelu(self.delta_fc0(ref_h))
        dh = F.gelu(self.delta_fc1(dh))

        # Residual corrections (initialized ~0).
        delta_logits = self.delta_actor(dh)   # (B,A)
        delta_value = self.delta_critic(dh)   # (B,1)

        # Final outputs = proposal + residual edit.
        logits = logits_prop + delta_logits
        value = value_prop + delta_value

        # Final shape assertions.
        # These are strict invariants: PPO runtime and downstream code rely on exact shapes.
        if logits.dim() != 2 or logits.size(0) != B or logits.size(1) != self.act_dim:
            raise RuntimeError(f"[MirrorBrain] bad logits shape: got {tuple(logits.shape)}, expected ({B},{self.act_dim})")
        if value.dim() != 2 or value.size(0) != B or value.size(1) != 1:
            raise RuntimeError(f"[MirrorBrain] bad value shape: got {tuple(value.shape)}, expected ({B},1)")

        return logits, value