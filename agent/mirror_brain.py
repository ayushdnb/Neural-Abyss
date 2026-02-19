from __future__ import annotations
# Import necessary modules from the standard library and PyTorch.
from typing import Dict, Tuple, List, Optional
import math                         # Added for sqrt(2) gain in orthogonal init

import torch
import torch.nn as nn
import torch.nn.functional as F

import config
from . import obs_spec


# ---------------------------------------------------------------------
# MirrorBrain: two-pass (propose -> reflect/edit) policy/value network.
#
# Hard invariants:
# - Input obs is the SAME flat vector and split layout as TronBrain.
# - Output contract matches PPO runtime: (logits, value) with shapes:
#     logits: (B, NUM_ACTIONS), value: (B, 1)
# ---------------------------------------------------------------------


class _SelfAttnBlock(nn.Module):
    """Pre-LN self-attention block (Transformer encoder style)."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        # LayerNorm before self-attention (Pre-LN)
        self.norm1 = nn.LayerNorm(d_model)
        # Multi-head self-attention, batch_first=True for (B, seq, D)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        # LayerNorm before feed‑forward
        self.norm2 = nn.LayerNorm(d_model)
        # Feed‑forward network with GELU activation (typical for transformers)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # LN in fp32 for stability, then cast back for AMP-friendly projections
        x1 = self.norm1(x.float()).to(dtype=x.dtype)
        # Self-attention (queries, keys, values all = x1), no attention weights needed
        a, _ = self.attn(x1, x1, x1, need_weights=False)
        # Residual connection
        x = x + a
        # Second LN (fp32 for stability)
        x2 = self.norm2(x.float()).to(dtype=x.dtype)
        # Feed‑forward + residual
        x = x + self.ff(x2)
        return x


class _CrossAttnBlock(nn.Module):
    """Pre-LN cross-attention block: queries from x attend to memory m."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        # LayerNorm for the query input
        self.norm_q = nn.LayerNorm(d_model)
        # LayerNorm for the memory (key/value) input
        self.norm_m = nn.LayerNorm(d_model)
        # Cross-attention: queries = x, keys/values = m
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        # Second LN after residual
        self.norm2 = nn.LayerNorm(d_model)
        # Feed‑forward network (same as in self-attention block)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        # Apply LN to queries and memory (fp32 for stability, then cast back)
        q = self.norm_q(x.float()).to(dtype=x.dtype)
        kv = self.norm_m(m.float()).to(dtype=m.dtype)
        # Cross-attention: queries attend to memory
        a, _ = self.attn(q, kv, kv, need_weights=False)
        x = x + a
        # Second LN + feed‑forward
        x2 = self.norm2(x.float()).to(dtype=x.dtype)
        x = x + self.ff(x2)
        return x


class MirrorBrain(nn.Module):
    """
    MirrorBrain = Tron-family tokenization + propose logits/value + small reflection edit.

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
        # Store dimensions for later validation
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)

        # --- Observation layout MUST match obs_spec / TronBrain exactly ---
        self.num_rays = int(getattr(config, "RAY_TOKEN_COUNT", 32))
        self.ray_feat_dim = int(getattr(config, "RAY_FEAT_DIM", 4))
        self.rays_flat_dim = self.num_rays * self.ray_feat_dim

        self.rich_base_dim = int(getattr(config, "RICH_BASE_DIM", 64))
        self.instinct_dim = int(getattr(config, "INSTINCT_DIM", 4))

        # Expected observation dimension based on components
        expected_obs_dim = self.rays_flat_dim + self.rich_base_dim + self.instinct_dim
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

        # --- Model hyperparams (reuse Tron defaults for consistent compute budget) ---
        d_model = int(getattr(config, "MIRROR_D_MODEL", getattr(config, "TRON_D_MODEL", 64)))
        n_heads = int(getattr(config, "MIRROR_HEADS", getattr(config, "TRON_HEADS", 4)))

        # Depth controls
        ray_layers = int(getattr(config, "MIRROR_RAY_LAYERS", getattr(config, "TRON_RAY_LAYERS", 4)))
        plan_layers = int(getattr(config, "MIRROR_SEM_LAYERS", getattr(config, "TRON_SEM_LAYERS", 3)))
        fusion_layers = int(getattr(config, "MIRROR_FUSION_LAYERS", getattr(config, "TRON_FUSION_LAYERS", 2)))

        # Head MLP capacity
        mlp_hidden = int(getattr(config, "MIRROR_MLP_HIDDEN", getattr(config, "TRON_MLP_HIDDEN", 256)))

        self.d_model = d_model

        if d_model % n_heads != 0:
            raise RuntimeError(f"[MirrorBrain] d_model must be divisible by n_heads (d_model={d_model}, n_heads={n_heads}).")

        # --- PASS 1: Embeddings ---
        # Rays: (B, N, RAY_FEAT_DIM) -> (B, N, D)
        self.ray_in_norm = nn.LayerNorm(self.ray_feat_dim)          # Normalize ray features
        self.ray_in_proj = nn.Linear(self.ray_feat_dim, d_model)    # Project to model dimension

        # Optional learnable ray direction embedding (kept identical to Tron for parity)
        self.ray_dir_embed = nn.Parameter(torch.randn(1, self.num_rays, d_model) * 0.02)

        # Semantic partitions from rich_base (5 tokens)
        # We use ModuleDict to hold per‑semantic‑token LayerNorm and linear projections.
        self.sem_in_norm = nn.ModuleDict()
        self.sem_in_proj = nn.ModuleDict()

        # The five semantic token keys; indices for each are defined in config.SEMANTIC_RICH_BASE_INDICES.
        sem_keys = ["own_context", "world_context", "zone_context", "team_context", "combat_context"]
        sem_idx_map: Dict[str, List[int]] = dict(getattr(config, "SEMANTIC_RICH_BASE_INDICES", {}))
        for k in sem_keys:
            idxs = sem_idx_map.get(k, None)
            if idxs is None or len(idxs) == 0:
                raise RuntimeError(f"[MirrorBrain] Missing/empty semantic index list for '{k}' in SEMANTIC_RICH_BASE_INDICES.")
            din = int(len(idxs))                                   # Input dimension for this semantic token
            self.sem_in_norm[k] = nn.LayerNorm(din)                # Normalize the raw semantic slice
            self.sem_in_proj[k] = nn.Linear(din, d_model)          # Project to model dimension

        # Instinct embedding (INSTINCT_DIM -> D)
        self.instinct_in_norm = nn.LayerNorm(self.instinct_dim)
        self.instinct_in_proj = nn.Linear(self.instinct_dim, d_model)

        # Learnable plan tokens
        self.memory_token = nn.Parameter(torch.zeros(1, 1, d_model))          # Memory token (initialized zero)
        self.decision_tokens = nn.Parameter(torch.randn(1, 3, d_model) * 0.02)  # 3 decision tokens, small noise

        # --- PASS 1: Encoders / Fusion ---
        self.ray_encoder = nn.ModuleList([_SelfAttnBlock(d_model, n_heads) for _ in range(ray_layers)])
        self.plan_encoder = nn.ModuleList([_SelfAttnBlock(d_model, n_heads) for _ in range(plan_layers)])
        self.fusion = nn.ModuleList([_CrossAttnBlock(d_model, n_heads) for _ in range(fusion_layers)])

        # --- PASS 1: Readout (decision tokens only) ---
        self.read_fc0 = nn.Linear(3 * d_model, mlp_hidden)    # Concatenate 3 decision tokens → hidden
        self.read_fc1 = nn.Linear(mlp_hidden, mlp_hidden)     # Extra hidden layer
        self.actor = nn.Linear(mlp_hidden, self.act_dim)      # Logits head
        self.critic = nn.Linear(mlp_hidden, 1)                # Value head

        # --- PASS 2: Reflection token builder (internal-state only) ---
        # REF input: mean(decision tokens) + [entropy, top1_margin, value] → total size d_model + 3
        self.ref_in_norm = nn.LayerNorm(d_model + 3)
        self.ref_in_proj = nn.Linear(d_model + 3, d_model)

        # REF attends to plan tokens and ray tokens (keep this small: 2 cross-attn blocks)
        self.ref_attend_plan = _CrossAttnBlock(d_model, n_heads)
        self.ref_attend_rays = _CrossAttnBlock(d_model, n_heads)

        # Delta heads (residual edits) — init near zero for stable start (handled in _init_weights)
        self.delta_fc0 = nn.Linear(d_model, mlp_hidden)
        self.delta_fc1 = nn.Linear(mlp_hidden, mlp_hidden)
        self.delta_actor = nn.Linear(mlp_hidden, self.act_dim)
        self.delta_critic = nn.Linear(mlp_hidden, 1)

        # Initialize all weights (orthogonal init with proper gains, delta heads zero)
        self._init_weights()

        # Note: The separate delta zeroing originally after _init_weights is removed,
        # because it is now integrated into the patched _init_weights.

    def _init_weights(self) -> None:
        """Orthogonal init (PPO-friendly). Keeps delta heads ~0 at start for a true 'edit' model."""
        gain_hidden = math.sqrt(2.0)                     # Common gain for ReLU/GELU layers
        # Apply orthogonal init to every Linear layer in the whole module tree
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=gain_hidden)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Base actor: tiny init => near-uniform logits early (gain 0.01)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        if self.actor.bias is not None:
            nn.init.zeros_(self.actor.bias)

        # Base critic: standard scale (gain 1.0)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        if self.critic.bias is not None:
            nn.init.zeros_(self.critic.bias)

        # Delta heads should start near-zero (so PASS 2 truly "edits" PASS 1)
        nn.init.orthogonal_(self.delta_actor.weight, gain=0.0)   # gain 0 → zero initialization
        if self.delta_actor.bias is not None:
            nn.init.zeros_(self.delta_actor.bias)

        nn.init.orthogonal_(self.delta_critic.weight, gain=0.0)
        if self.delta_critic.bias is not None:
            nn.init.zeros_(self.delta_critic.bias)

    # ---- embedding helpers (stable + AMP-friendly) ----
    def _embed_rays(self, rays_raw: torch.Tensor) -> torch.Tensor:
        """Embed raw ray features (B, num_rays, ray_feat_dim) to (B, num_rays, d_model)."""
        x = self.ray_in_norm(rays_raw.float())                     # LN in fp32 for stability
        x = self.ray_in_proj(x.to(dtype=self.ray_in_proj.weight.dtype))  # Project in original dtype
        return x

    def _embed_sem(self, x_raw: torch.Tensor, key: str) -> torch.Tensor:
        """Embed a raw semantic token slice to (B, 1, d_model)."""
        n = self.sem_in_norm[key]
        p = self.sem_in_proj[key]
        x = n(x_raw.float())                                       # LN in fp32
        x = p(x.to(dtype=p.weight.dtype))                          # Project
        return x

    def _embed_instinct(self, inst_raw: torch.Tensor) -> torch.Tensor:
        """Embed raw instinct features to (B, 1, d_model)."""
        x = self.instinct_in_norm(inst_raw.float())                # LN in fp32
        x = self.instinct_in_proj(x.to(dtype=self.instinct_in_proj.weight.dtype))
        return x

    @staticmethod
    def _logits_entropy_and_margin(logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Cheap uncertainty summary from logits (no env inputs). Returns (entropy, margin) as (B,1) each."""
        logits32 = logits.to(torch.float32)                        # Work in fp32 for stability
        logp = F.log_softmax(logits32, dim=-1)                     # log probabilities
        p = logp.exp()                                              # probabilities
        ent = -(p * logp).sum(dim=-1, keepdim=True)                # entropy (B,1)

        # Top1 margin: (top1 - top2). Larger = more confident.
        top2 = torch.topk(logits32, k=2, dim=-1).values            # (B,2)
        margin = (top2[:, 0:1] - top2[:, 1:2])                      # (B,1)
        return ent, margin

    def _build_ref_token(
        self,
        dec_tokens: torch.Tensor,          # (B,3,D) in current dtype
        logits_proposal: torch.Tensor,     # (B,A)
        value_proposal: torch.Tensor,      # (B,1)
    ) -> torch.Tensor:
        """Build REF token as (B,1,D) from internal state only (fp32 LN, AMP-friendly projections)."""
        if dec_tokens.dim() != 3 or dec_tokens.size(1) != 3 or dec_tokens.size(2) != self.d_model:
            raise RuntimeError(f"[MirrorBrain] bad dec_tokens shape for REF: got {tuple(dec_tokens.shape)}, expected (B,3,{self.d_model})")

        dec_mean = dec_tokens.mean(dim=1).to(torch.float32)        # (B,D) in fp32
        ent, margin = self._logits_entropy_and_margin(logits_proposal)  # (B,1), (B,1)
        v = value_proposal.to(torch.float32)                       # (B,1) in fp32
        if v.dim() != 2 or v.size(1) != 1:
            raise RuntimeError(f"[MirrorBrain] bad value_proposal shape for REF: got {tuple(v.shape)}, expected (B,1)")

        feat = torch.cat([dec_mean, ent, margin, v], dim=-1)       # (B, D+3)
        x = self.ref_in_norm(feat)                                 # LN in fp32
        x = self.ref_in_proj(x.to(dtype=self.ref_in_proj.weight.dtype))  # Project in model dtype
        return x.unsqueeze(1)                                      # (B,1,D)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            obs: Flat observation tensor of shape (B, OBS_DIM). Layout must match obs_spec / TronBrain.

        Returns:
            (logits_final, value_final)
            logits_final: (B, act_dim)
            value_final:  (B, 1)
        """
        if obs.dim() != 2:
            raise RuntimeError(f"[MirrorBrain] expected obs to be 2D (B,OBS_DIM); got shape {tuple(obs.shape)}")
        B, D = obs.shape
        if D != self.obs_dim:
            raise RuntimeError(f"[MirrorBrain] obs feature dim mismatch: got {D}, expected {self.obs_dim}")

        # Split obs using the same function as TronBrain (ensures identical layout)
        rays_flat, rich_base, instinct = obs_spec.split_obs_flat(obs)

        # ----- PASS 1: PROPOSE -----
        # Rays -> tokens
        rays_raw = rays_flat.view(B, self.num_rays, self.ray_feat_dim)   # (B, num_rays, ray_feat_dim)
        ray_tok = self._embed_rays(rays_raw)                              # (B, num_rays, d_model)
        # Add learnable direction embedding
        ray_tok = ray_tok + self.ray_dir_embed.to(device=obs.device, dtype=ray_tok.dtype)
        # Self-attention over rays
        for blk in self.ray_encoder:
            ray_tok = blk(ray_tok)                                        # (B, num_rays, d_model)

        # Semantic partitions (uses cached index tensors; no layout changes)
        sem_raw = obs_spec.build_semantic_tokens(rich_base, instinct)    # dict of raw semantic tensors

        sem_keys = ["own_context", "world_context", "zone_context", "team_context", "combat_context"]
        sem_tokens = []
        for k in sem_keys:
            # Embed each semantic token and unsqueeze to (B,1,d_model)
            sem_tokens.append(self._embed_sem(sem_raw[k], k).unsqueeze(1))
        sem = torch.cat(sem_tokens, dim=1)                                # (B,5,d_model)

        # Instinct token (same tail, embedded)
        inst_tok = self._embed_instinct(sem_raw["instinct_context"]).unsqueeze(1)  # (B,1,d_model)

        # Learnable decision + memory tokens
        dec = self.decision_tokens.expand(B, -1, -1).to(device=obs.device)  # (B,3,d_model)
        mem = self.memory_token.expand(B, -1, -1).to(device=obs.device)     # (B,1,d_model)

        # Plan token sequence: 10 tokens total (must match Tron-family)
        tok = torch.cat([dec, sem, inst_tok, mem], dim=1)                   # (B,10,d_model)
        if tok.size(1) != 10:
            raise RuntimeError(f"[MirrorBrain] plan token length must be 10; got {int(tok.size(1))}")

        # Self-attention over plan tokens
        for blk in self.plan_encoder:
            tok = blk(tok)                                                  # (B,10,d_model)

        # Cross-attention: plan tokens attend to ray tokens
        for blk in self.fusion:
            tok = blk(tok, ray_tok)                                         # (B,10,d_model)

        # Readout from decision tokens only (first 3 tokens)
        dec_out = tok[:, :3, :].reshape(B, 3 * self.d_model)                # (B, 3*d_model)
        h = F.gelu(self.read_fc0(dec_out))
        h = F.gelu(self.read_fc1(h))
        logits_prop = self.actor(h)                                         # (B,A)
        value_prop = self.critic(h)                                         # (B,1)

        # ----- PASS 2: REFLECT + EDIT -----
        # Build REF from internal summaries only (no new env inputs)
        ref = self._build_ref_token(tok[:, :3, :], logits_prop, value_prop).to(dtype=tok.dtype, device=tok.device)  # (B,1,d_model)

        # REF attends to plan tokens and ray tokens (small, 2 blocks)
        ref = self.ref_attend_plan(ref, tok)                                # (B,1,d_model)
        ref = self.ref_attend_rays(ref, ray_tok)                            # (B,1,d_model)

        ref_h = ref.squeeze(1)                                              # (B,d_model)
        dh = F.gelu(self.delta_fc0(ref_h))
        dh = F.gelu(self.delta_fc1(dh))
        delta_logits = self.delta_actor(dh)                                 # (B,A) ~0 at init
        delta_value = self.delta_critic(dh)                                 # (B,1) ~0 at init

        logits = logits_prop + delta_logits
        value = value_prop + delta_value

        # ----- Final shape asserts (critical invariants) -----
        if logits.dim() != 2 or logits.size(0) != B or logits.size(1) != self.act_dim:
            raise RuntimeError(f"[MirrorBrain] bad logits shape: got {tuple(logits.shape)}, expected ({B},{self.act_dim})")
        if value.dim() != 2 or value.size(0) != B or value.size(1) != 1:
            raise RuntimeError(f"[MirrorBrain] bad value shape: got {tuple(value.shape)}, expected ({B},1)")

        return logits, value