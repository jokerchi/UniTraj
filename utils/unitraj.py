"""
UniTraj: Universal Trajectory Foundation Model — Core Components.

This module implements the encoder-decoder architecture for trajectory modeling:
- Rotary Position Embedding (RoPE) for sequence-aware attention
- Transformer blocks with multi-head self-attention and feedforward layers
- Patch-based tokenization with configurable masking strategies
- Encoder: tokenizes trajectories, applies random mask/shuffle, encodes via Transformer
- Decoder: reconstructs full trajectory from unmasked features

References:
    UniTraj: Learning a Universal Trajectory Foundation Model from Billion-Scale
    Worldwide Traces (NeurIPS 2025)
"""

import math
import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from timm.layers import trunc_normal_
from timm.models.vision_transformer import Block

# ============================================================================
# Rotary Position Embedding (RoPE)
# ============================================================================

class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) for sequence-aware attention.

    Encodes position information through rotation in the embedding space,
    enabling the attention mechanism to capture relative positions naturally.

    Args:
        embedding_dim: Dimension of the head embedding (must be even).
        max_seq_len: Maximum sequence length to precompute.
    """

    def __init__(self, embedding_dim: int, max_seq_len: int = 512):
        super().__init__()
        self.embedding_dim = embedding_dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, embedding_dim, 2).float() / embedding_dim))
        positions = torch.arange(max_seq_len).float()
        sinusoid_input = torch.einsum("i , j -> i j", positions, inv_freq)
        self.register_buffer("sin", sinusoid_input.sin(), persistent=False)
        self.register_buffer("cos", sinusoid_input.cos(), persistent=False)

    def forward(self, seq_len: int):
        """Return sin/cos tensors for the given sequence length.

        Returns:
            Tuple[Tensor, Tensor]: sin and cos tensors, each of shape
            [1, 1, seq_len, embedding_dim//2].
        """
        sin = self.sin[:seq_len, :].unsqueeze(0).unsqueeze(0)
        cos = self.cos[:seq_len, :].unsqueeze(0).unsqueeze(0)
        return sin, cos


# ============================================================================
# Transformer Building Blocks
# ============================================================================

class FeedForward(nn.Module):
    """Standard Transformer feedforward network: LayerNorm -> Linear -> GELU -> Linear."""

    def __init__(self, embedding_dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Attention(nn.Module):
    """Multi-head self-attention with RoPE, using PyTorch's optimized SDPA backend.

    Args:
        embedding_dim: Input/output embedding dimension.
        num_heads: Number of attention heads.
        head_dim: Dimension per head (must be even for RoPE).
        dropout: Attention dropout rate.
        max_seq_len: Maximum sequence length for RoPE precomputation.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int = 8,
        head_dim: int = 64,
        dropout: float = 0.0,
        max_seq_len: int = 512,
    ):
        super().__init__()
        inner_dim = head_dim * num_heads
        project_out = not (num_heads == 1 and head_dim == embedding_dim)

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout_p = dropout

        self.norm = nn.LayerNorm(embedding_dim)
        self.to_qkv = nn.Linear(embedding_dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, embedding_dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

        assert head_dim % 2 == 0, "head_dim must be even for RoPE"
        self.rotary_emb = RotaryEmbedding(head_dim, max_seq_len=max_seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        x = self.norm(x)

        # Project to Q, K, V and split into heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.num_heads), qkv)

        # Apply Rotary Position Embedding
        sin, cos = self.rotary_emb(n)
        q1, q2 = q[..., :self.head_dim // 2], q[..., self.head_dim // 2:]
        k1, k2 = k[..., :self.head_dim // 2], k[..., self.head_dim // 2:]
        q = torch.cat([q1 * cos - q2 * sin, q2 * cos + q1 * sin], dim=-1)
        k = torch.cat([k1 * cos - k2 * sin, k2 * cos + k1 * sin], dim=-1)

        # Fused scaled dot-product attention (auto-selects optimal backend)
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout_p if self.training else 0.0,
        )

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    """Stack of Attention + FeedForward layers with residual connections.

    Args:
        embedding_dim: Model dimension throughout the transformer.
        depth: Number of transformer layers.
        num_heads: Attention heads per layer.
        head_dim: Dimension per attention head.
        feedforward_dim: Hidden dimension of the feedforward network.
        dropout: Dropout rate applied in attention and feedforward layers.
        max_seq_len: Maximum sequence length for RoPE.
    """

    def __init__(
        self,
        embedding_dim: int,
        depth: int,
        num_heads: int,
        head_dim: int,
        feedforward_dim: int,
        dropout: float = 0.0,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    Attention(
                        embedding_dim,
                        num_heads=num_heads,
                        head_dim=head_dim,
                        dropout=dropout,
                        max_seq_len=max_seq_len,
                    ),
                    FeedForward(embedding_dim, feedforward_dim, dropout=dropout),
                ])
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for attn_layer, ff_layer in self.layers:
            x = attn_layer(x) + x
            x = ff_layer(x) + x
        return x


# ============================================================================
# Indexing Utilities
# ============================================================================

def take_indices(sequence: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Reorder a sequence along the time dimension according to given indices.

    Args:
        sequence: [T, B, C] input tensor.
        indices: [T, B] index tensor specifying the new order.

    Returns:
        [T, B, C] reordered tensor.
    """
    return torch.gather(
        sequence, 0, repeat(indices, "t b -> t b c", c=sequence.shape[-1])
    )


def random_indices(size: int, device: torch.device = None):
    """Generate a random permutation of sequence indices and its inverse.

    Args:
        size: Sequence length.
        device: Target torch device.

    Returns:
        (forward_indices, backward_indices): forward shuffles the sequence;
        backward (argsort of forward) restores the original order.
    """
    forward_indices = torch.randperm(size, device=device)
    backward_indices = torch.argsort(forward_indices)
    return forward_indices, backward_indices


def specified_mask_indices(
    size: int, mask_indices: torch.Tensor, device: torch.device = None
):
    """Generate shuffle indices where specified positions are moved to the end.

    Unmasked (remaining) indices are randomly shuffled and placed first,
    followed by the explicitly masked indices. This allows the encoder to
    process only unmasked tokens while the decoder reconstructs the full sequence.

    Args:
        size: Total sequence length.
        mask_indices: 1-D tensor of indices to mask (move to the end).
        device: Target torch device.

    Returns:
        (forward_indices, backward_indices): forward places masked positions
        last; backward restores the original order.
    """
    # Deduplicate mask indices (duplicates would cause length mismatch)
    mask_indices = torch.unique(
        mask_indices.to(device) if device is not None else mask_indices
    )
    forward_indices = torch.arange(size, device=device)
    mask = ~torch.isin(forward_indices, mask_indices)
    remaining_indices = forward_indices[mask]
    remaining_indices = remaining_indices[
        torch.randperm(len(remaining_indices), device=device)
    ]
    forward_indices = torch.cat([remaining_indices, mask_indices])
    backward_indices = torch.argsort(forward_indices)
    return forward_indices, backward_indices


# ============================================================================
# Patch Shuffling & Masking
# ============================================================================

class PatchShuffle(nn.Module):
    """Randomly shuffle patches/tokens in the time dimension and drop a fraction.

    Implements the masking strategy from MAE (Masked Autoencoder): tokens are
    randomly permuted and a portion are discarded. When ``mask_indices`` is
    provided, those specific indices are deterministically moved to the end.

    Args:
        mask_ratio: Fraction of tokens to mask (drop).
    """

    def __init__(self, mask_ratio: float):
        super().__init__()
        self.mask_ratio = mask_ratio

    def forward(
        self, patches: torch.Tensor, mask_indices: torch.Tensor = None
    ):
        """Shuffle and mask patches.

        Args:
            patches: [L, B, C] tokenized trajectory patches.
            mask_indices: Optional [B, N] tensor of pre-selected mask indices.

        Returns:
            (shuffled_patches, forward_indices, backward_indices):
            shuffled_patches has shape [remain_L, B, C] (masked tokens removed).
            forward_indices and backward_indices are [L, B] tensors.
        """
        L, B, C = patches.shape
        remain_L = int(L * (1 - self.mask_ratio))

        if mask_indices is not None:
            indices = [
                specified_mask_indices(L, mask_indices[i], device=patches.device)
                for i in range(B)
            ]
            # Use unique count to correctly compute remain_L (duplicates are
            # deduplicated inside specified_mask_indices).
            remain_L = L - len(torch.unique(mask_indices[0]))
        else:
            indices = [random_indices(L, device=patches.device) for _ in range(B)]

        forward_indices = torch.stack([i[0] for i in indices], dim=-1).to(patches.device)
        backward_indices = torch.stack([i[1] for i in indices], dim=-1).to(patches.device)

        patches = take_indices(patches, forward_indices)
        patches = patches[:remain_L]

        return patches, forward_indices, backward_indices


# ============================================================================
# Encoder
# ============================================================================

class Encoder(nn.Module):
    """Trajectory Encoder: tokenize, fuse intervals, mask/shuffle, encode via Transformer.

    Pipeline:
        1. Conv1d tokenizer: [B, 2, L] -> [L/patch_size, B, embedding_dim]
        2. Add interval embeddings (time delta between points)
        3. PatchShuffle: randomly mask & permute tokens
        4. Prepend learnable CLS token for global aggregation
        5. Transformer encoder with RoPE attention

    Args:
        trajectory_length: Number of trajectory points (L).
        patch_size: Temporal patch size for tokenization (L // patch_size = num_tokens).
        embedding_dim: Model dimension throughout the encoder.
        num_layers: Number of Transformer layers.
        num_heads: Number of attention heads.
        mask_ratio: Fraction of tokens to mask during pre-training.
    """

    def __init__(
        self,
        trajectory_length: int,
        patch_size: int,
        embedding_dim: int,
        num_layers: int,
        num_heads: int,
        mask_ratio: float,
    ):
        super().__init__()
        self.num_tokens = trajectory_length // patch_size
        self.max_seq_len = 512

        # CLS token: a learnable global summary token (like ViT/BERT [CLS]).
        # Prepended to the sequence so the Transformer can aggregate
        # trajectory-wide information at this position.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        self.shuffle = PatchShuffle(mask_ratio)

        # Conv1d tokenizer: maps 2-channel (lon, lat) to embedding_dim,
        # kernel_size = stride = patch_size, so every patch_size points produce one token.
        self.tokenizer = nn.Conv1d(2, embedding_dim, patch_size, patch_size)
        self.transformer = Transformer(
            embedding_dim,
            depth=num_layers,
            num_heads=num_heads,
            head_dim=embedding_dim // num_heads,
            feedforward_dim=embedding_dim * 4,
            dropout=0.0,
            max_seq_len=self.max_seq_len,
        )
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self._init_weights()

    def _init_weights(self):
        trunc_normal_(self.cls_token, std=0.02)

    def forward(
        self,
        trajectory: torch.Tensor,
        interval_embedding: torch.Tensor,
        mask_indices: torch.Tensor = None,
    ):
        """Encode trajectories into latent features.

        Args:
            trajectory: [B, 2, L] raw trajectory (lon, lat after normalization).
            interval_embedding: [B, L, embedding_dim] time-interval embeddings.
            mask_indices: Optional [B, num_mask] pre-computed mask indices.

        Returns:
            (features, backward_indices):
            features: [num_unmasked_tokens+1, B, embedding_dim] (includes CLS).
            backward_indices: [num_tokens, B] for decoder reconstruction.
        """
        # Tokenize: [B, 2, L] -> [B, embedding_dim, num_tokens] -> [num_tokens, B, embedding_dim]
        tokens = self.tokenizer(trajectory)
        tokens = rearrange(tokens, "b c l -> l b c")

        # Fuse time-interval information
        interval_embedding = rearrange(interval_embedding, "b l c -> l b c")
        tokens = tokens + interval_embedding

        # Shuffle and mask
        tokens, forward_indices, backward_indices = self.shuffle(tokens, mask_indices)

        # Prepend CLS token
        tokens = torch.cat(
            [self.cls_token.expand(-1, tokens.shape[1], -1), tokens], dim=0
        )
        tokens = rearrange(tokens, "t b c -> b t c")

        # Transformer encoding
        features = self.transformer(tokens)
        features = self.layer_norm(features)
        features = rearrange(features, "b t c -> t b c")

        return features, backward_indices


# ============================================================================
# Decoder
# ============================================================================

class Decoder(nn.Module):
    """Trajectory Decoder: reconstructs full trajectory from unmasked encoder features.

    Pipeline:
        1. Pad features with learnable MASK tokens for masked positions
        2. Restore original token order via backward_indices
        3. Add time-interval embeddings (with learnable TIME token for CLS)
        4. Transformer decoder
        5. Linear head projects to [B, 2, L] trajectory

    Args:
        trajectory_length: Number of trajectory points (L).
        patch_size: Temporal patch size.
        embedding_dim: Model dimension throughout the decoder.
        num_layers: Number of Transformer layers.
        num_heads: Number of attention heads.
    """

    def __init__(
        self,
        trajectory_length: int,
        patch_size: int,
        embedding_dim: int,
        num_layers: int,
        num_heads: int,
    ):
        super().__init__()
        self.num_tokens = trajectory_length // patch_size
        self.max_seq_len = 512
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        self.time_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))

        self.transformer = Transformer(
            embedding_dim,
            depth=num_layers,
            num_heads=num_heads,
            head_dim=embedding_dim // num_heads,
            feedforward_dim=embedding_dim * 4,
            dropout=0.0,
            max_seq_len=self.max_seq_len,
        )

        # Linear projection: embedding_dim -> 2*patch_size (lon, lat for each point)
        self.head = nn.Linear(embedding_dim, 2 * patch_size)
        self.token_to_traj = Rearrange(
            "h b (c p) -> b c (h p)", p=patch_size, h=trajectory_length // patch_size
        )
        self._init_weights()

    def _init_weights(self):
        trunc_normal_(self.mask_token, std=0.02)
        trunc_normal_(self.time_token, std=0.02)

    def forward(
        self,
        features: torch.Tensor,
        backward_indices: torch.Tensor,
        interval_embedding: torch.Tensor,
    ):
        """Decode latent features back to trajectory space.

        Args:
            features: [num_unmasked+1, B, embedding_dim] encoder output.
            backward_indices: [num_tokens, B] indices to restore original order.
            interval_embedding: [B, L, embedding_dim] time-interval embeddings.

        Returns:
            (predicted_trajectory, mask):
            predicted_trajectory: [B, 2, L] reconstructed trajectory.
            mask: [B, 2, L] binary mask indicating masked regions.
        """
        T, B = features.shape[0], features.shape[1]

        # Offset backward_indices to account for CLS token (index 0)
        backward_indices = torch.cat(
            [
                torch.zeros(1, backward_indices.shape[1], dtype=backward_indices.dtype,
                            device=backward_indices.device),
                backward_indices + 1,
            ],
            dim=0,
        )  # [num_tokens+1, B]

        # Pad with learnable mask tokens for the masked positions
        num_masked = backward_indices.shape[0] - features.shape[0]
        features = torch.cat(
            [features, self.mask_token.expand(num_masked, B, -1)], dim=0
        )  # [num_tokens+1, B, embedding_dim]

        # Restore original token order
        features = take_indices(features, backward_indices)

        # Prepend time token for CLS position and fuse interval embeddings
        interval_embedding = torch.cat(
            [self.time_token.expand(features.shape[1], 1, -1), interval_embedding], dim=1
        )  # [B, num_tokens+1, embedding_dim]
        interval_embedding = rearrange(interval_embedding, "b t c -> t b c")
        features = features + interval_embedding

        # Transformer decoding
        features = rearrange(features, "t b c -> b t c")
        features = self.transformer(features)
        features = rearrange(features, "b t c -> t b c")

        # Remove CLS token, project to trajectory space
        features = features[1:]
        patches = self.head(features)  # [num_tokens, B, 2*patch_size]

        # Build mask: last (T-1) tokens are masked positions
        mask = torch.zeros_like(patches)
        mask[T - 1:] = 1
        mask = take_indices(mask, backward_indices[1:] - 1)

        traj = self.token_to_traj(patches)
        mask = self.token_to_traj(mask)
        return traj, mask


# ============================================================================
# UniTraj: Full Encoder-Decoder Model
# ============================================================================

class UniTraj(nn.Module):
    """UniTraj: Universal Trajectory Foundation Model.

    An encoder-decoder architecture pre-trained via masked autoencoding (MAE)
    on trajectory data. The encoder processes partially-observed trajectories
    and the decoder reconstructs the full sequence.

    Args:
        trajectory_length: Input trajectory length (number of points).
        patch_size: Temporal patch size for Conv1d tokenization (default 1).
        embedding_dim: Model dimension (default 128).
        encoder_layers: Number of Transformer layers in the encoder (default 8).
        encoder_heads: Number of attention heads in the encoder (default 4).
        decoder_layers: Number of Transformer layers in the decoder (default 4).
        decoder_heads: Number of attention heads in the decoder (default 2).
        mask_ratio: Fraction of tokens to mask during training (default 0.5).
    """

    def __init__(
        self,
        trajectory_length: int = 32,
        patch_size: int = 2,
        embedding_dim: int = 128,
        encoder_layers: int = 8,
        encoder_heads: int = 4,
        decoder_layers: int = 4,
        decoder_heads: int = 2,
        mask_ratio: float = 0.5,
    ):
        super().__init__()

        self.encoder = Encoder(
            trajectory_length, patch_size, embedding_dim,
            encoder_layers, encoder_heads, mask_ratio,
        )
        self.decoder = Decoder(
            trajectory_length, patch_size, embedding_dim,
            decoder_layers, decoder_heads,
        )
        self.interval_embedding = nn.Linear(1, embedding_dim)

    def forward(
        self,
        trajectory: torch.Tensor,
        intervals: torch.Tensor = None,
        mask_indices: torch.Tensor = None,
    ):
        """Forward pass: encode trajectory, reconstruct masked regions.

        Args:
            trajectory: [B, 2, L] normalized trajectory (lon, lat).
            intervals: [B, L] time intervals between consecutive points.
            mask_indices: Optional [B, num_mask] pre-computed mask indices.

        Returns:
            (predicted_trajectory, mask):
            predicted_trajectory: [B, 2, L] reconstructed trajectory.
            mask: [B, 2, L] binary mask (1 = masked region).
        """
        # Embed time intervals via linear projection, or use zeros if no intervals
        if intervals is not None:
            intervals = intervals.unsqueeze(-1)  # [B, L, 1]
            interval_embeddings = self.interval_embedding(intervals)  # [B, L, embedding_dim]
        else:
            intervals_pooled = torch.zeros(
                (trajectory.shape[0], self.encoder.num_tokens),
                device=trajectory.device,
            )
            interval_embeddings = self.interval_embedding(intervals_pooled.unsqueeze(-1))

        features, backward_indices = self.encoder(
            trajectory, interval_embeddings, mask_indices
        )
        predicted_trajectory, mask = self.decoder(
            features, backward_indices, interval_embeddings
        )
        return predicted_trajectory, mask
