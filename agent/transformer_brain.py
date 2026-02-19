from typing import Tuple
import math                # Added by patch: used for sqrt in orthogonal init gain
import torch
import torch.nn as nn
import torch.nn.functional as F
import config               # Global configuration parameters


class CrossAttentionBlock(nn.Module):
    """
    A single block of Cross-Attention followed by a Feed-Forward network.
    Includes residual connections and layer normalization.

    In cross-attention, the query comes from one sequence (ray tokens) and
    the key/value come from another sequence (the rich state token).
    This allows rays to attend to the agent's internal state.
    """
    def __init__(self, embed_dim: int, num_heads: int = 1):
        """
        Args:
            embed_dim: Embedding dimension for tokens (must match across Q/K/V).
            num_heads: Number of attention heads (default 1 for simplicity).
        """
        super().__init__()
        self.embed_dim = embed_dim
        # MultiheadAttention with batch_first=True expects inputs of shape (B, T, D)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        # First layer norm after attention residual
        self.norm1 = nn.LayerNorm(embed_dim)
        # Simple feed-forward network: expand, GELU, contract
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        # Second layer norm after FFN residual
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: Ray tokens (B, num_rays, embed_dim) – the sequence that asks for information.
            key_value: Rich token (B, 1, embed_dim) – the source of contextual info.
        Returns:
            Updated ray tokens after cross-attention and FFN.
        """
        # Cross-attention: query attends to key_value (same as value)
        attn_output, _ = self.attn(query, key_value, key_value, need_weights=False)
        # Residual connection + layer norm
        x = self.norm1(query + attn_output)
        # Feed-forward
        ffn_output = self.ffn(x)
        # Second residual + norm
        x = self.norm2(x + ffn_output)
        return x


class SelfAttentionBlock(nn.Module):
    """
    A single block of Self-Attention followed by a Feed-Forward network.
    Includes residual connections and layer normalization.
    Used to let ray tokens exchange information among themselves.
    """
    def __init__(self, embed_dim: int, num_heads: int = 1):
        super().__init__()
        self.embed_dim = embed_dim
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Ray tokens (B, num_rays, embed_dim)
        Returns:
            Self-attended ray tokens.
        """
        # Self-attention: Q, K, V all from the same sequence x
        attn_output, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + attn_output)
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        return x


# --- The Main Transformer Brain ---

class TransformerBrain(nn.Module):
    """
    A transformer-based brain that processes observations by treating raycasts
    as a sequence of tokens and enriching them with the agent's state via attention.

    The observation vector is split into:
        - ray features: 32 rays * 8 features each (total 256)
        - rich features: remaining dimensions (e.g., health, position, etc.)

    Ray features are embedded to `embed_dim`, positionally encoded, then passed through
    cross‑attention (with the rich state as context) and self‑attention. The pooled ray
    summary is concatenated with the rich token and fed to an MLP that produces action
    logits and a state value.
    """
    def __init__(self, obs_dim: int, act_dim: int, embed_dim: int = 32, mlp_hidden: int = 128):
        """
        Args:
            obs_dim: Total observation dimension (must be > num_rays * ray_feat_dim).
            act_dim: Number of discrete actions.
            embed_dim: Embedding size for tokens.
            mlp_hidden: Hidden size of the final MLP.
        """
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.embed_dim = int(embed_dim)

        # Fixed assumptions about observation structure
        self.num_rays = 32                # Number of raycast directions
        self.ray_feat_dim = 8              # Features per ray (distance, object type, etc.)
        # Remaining part of observation is the "rich" state
        self.rich_feat_dim = self.obs_dim - (self.num_rays * self.ray_feat_dim)

        if self.rich_feat_dim <= 0:
            raise ValueError(f"obs_dim ({obs_dim}) is not large enough for {self.num_rays} rays.")

        # ----- Embedding layers -----
        # Ray features: normalize first, then project to embed_dim
        self.ray_embed_norm = nn.LayerNorm(self.ray_feat_dim)
        self.ray_embed_proj = nn.Linear(self.ray_feat_dim, self.embed_dim)

        # Rich features: same two‑step process
        self.rich_embed_norm = nn.LayerNorm(self.rich_feat_dim)
        self.rich_embed_proj = nn.Linear(self.rich_feat_dim, self.embed_dim)

        # Learnable positional encoding for the 32 ray tokens
        self.positional_encoding = nn.Parameter(torch.randn(1, self.num_rays, self.embed_dim))

        # Attention blocks
        self.cross_attention = CrossAttentionBlock(self.embed_dim)   # rays attend to rich token
        self.self_attention = SelfAttentionBlock(self.embed_dim)     # rays attend among themselves

        # Final MLP: input is concatenated pooled rays (embed_dim) + rich token (embed_dim)
        mlp_input_dim = self.embed_dim * 2
        self.fc_in = nn.Linear(mlp_input_dim, mlp_hidden)
        self.fc1 = nn.Linear(mlp_hidden, mlp_hidden)
        # Actor head: produces logits for each action
        self.actor = nn.Linear(mlp_hidden, self.act_dim)
        # Critic head: produces scalar state value
        self.critic = nn.Linear(mlp_hidden, 1)

        self.init_weights()

    def init_weights(self):
        """
        Orthogonal initialisation (PPO‑friendly):
          - All linear layers (except heads) gain = sqrt(2) (common for ReLU/GELU).
          - Actor head gain very small (0.01) so initial logits are near‑uniform → high entropy.
          - Critic head gain = 1.0 (standard scale).
        Biases are initialised to zero.
        """
        gain_hidden = math.sqrt(2.0)      # Recommended for layers with GELU activation
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Orthogonal initialisation is stable and preserves gradient norm
                nn.init.orthogonal_(m.weight, gain=gain_hidden)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Override actor head: tiny gain ensures initial logits are small → uniform probs
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        if self.actor.bias is not None:
            nn.init.zeros_(self.actor.bias)

        # Override critic head: gain 1.0 is standard for value output
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        if self.critic.bias is not None:
            nn.init.zeros_(self.critic.bias)

    # --- JIT‑compatible helper methods for embedding ---
    # These are separate because TorchScript can struggle with generic functions
    # that have many branches. Each does one specific task.

    def _embed_rays(self, rays_raw: torch.Tensor) -> torch.Tensor:
        """
        Embed the raw ray features.
        Steps:
          1. Apply LayerNorm (works on float32, but input may be any dtype).
          2. Cast to the weight dtype of the projection layer.
          3. Project to embed_dim.
        """
        # Normalise the ray features (stable even if input is half precision)
        x_norm = self.ray_embed_norm(rays_raw.float())
        # Project using the correct dtype (avoids dtype mismatch errors in JIT)
        return self.ray_embed_proj(x_norm.to(dtype=self.ray_embed_proj.weight.dtype))

    def _embed_rich(self, rich_raw: torch.Tensor) -> torch.Tensor:
        """Embed the raw rich features, following the same pattern as _embed_rays."""
        x_norm = self.rich_embed_norm(rich_raw.float())
        return self.rich_embed_proj(x_norm.to(dtype=self.rich_embed_proj.weight.dtype))

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the transformer brain.

        Args:
            obs: Observation tensor of shape (B, obs_dim).

        Returns:
            logits: Action logits of shape (B, act_dim).
            value:   State value of shape (B, 1).
        """
        B = obs.shape[0]   # Batch size

        # ----- Split observation -----
        # First part: ray features (B, num_rays * ray_feat_dim) -> reshape to (B, num_rays, ray_feat_dim)
        rays_raw = obs[:, :self.num_rays * self.ray_feat_dim].view(B, self.num_rays, self.ray_feat_dim)
        # Remaining: rich features (B, rich_feat_dim)
        rich_raw = obs[:, self.num_rays * self.ray_feat_dim:]

        # ----- Embed to token space -----
        ray_tokens = self._embed_rays(rays_raw)                 # (B, num_rays, embed_dim)
        rich_token = self._embed_rich(rich_raw).unsqueeze(1)    # (B, 1, embed_dim)

        # Add positional encoding to ray tokens (broadcast over batch)
        ray_tokens = ray_tokens + self.positional_encoding

        # ----- Attention processing -----
        # Step 1: Cross‑attention: rays query the rich token to incorporate global context
        contextual_ray_tokens = self.cross_attention(query=ray_tokens, key_value=rich_token)
        # Step 2: Self‑attention among rays to let them exchange information
        processed_ray_tokens = self.self_attention(contextual_ray_tokens)

        # ----- Pool rays and combine with rich token for MLP -----
        # Average over the ray dimension to get a single vector summarising all rays
        pooled_ray_summary = processed_ray_tokens.mean(dim=1)   # (B, embed_dim)
        # Concatenate with the rich token (squeezed to remove the sequence dimension)
        mlp_input = torch.cat([pooled_ray_summary, rich_token.squeeze(1)], dim=-1)  # (B, embed_dim*2)

        # ----- Final MLP -----
        h = F.gelu(self.fc_in(mlp_input))
        h = F.gelu(self.fc1(h))

        logits = self.actor(h)   # Action logits (B, act_dim)
        value = self.critic(h)    # State value (B, 1)

        return logits, value

    def param_count(self) -> int:
        """Utility to count the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def scripted_transformer_brain(obs_dim: int, act_dim: int) -> torch.jit.ScriptModule:
    """
    Create a TorchScript version of the transformer brain.
    This is useful for deployment or for environments where Python overhead is undesirable.
    """
    model = TransformerBrain(obs_dim, act_dim)
    return torch.jit.script(model)