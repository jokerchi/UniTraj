import torch
import math
import timm
import numpy as np
from torch import nn
from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block


# Rotary Embedding class
class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding, RoPE）
    """
    def __init__(self, embedding_dim, max_seq_len=512):
        super().__init__()
        self.embedding_dim = embedding_dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, embedding_dim, 2).float() / embedding_dim))
        positions = torch.arange(max_seq_len).float()
        sinusoid_input = torch.einsum("i , j -> i j", positions, inv_freq)
        self.register_buffer("sin", sinusoid_input.sin(), persistent=False)  # [max_seq_len, embedding_dim//2]
        self.register_buffer("cos", sinusoid_input.cos(), persistent=False)

    def forward(self, seq_len):

        sin = self.sin[:seq_len, :].unsqueeze(0).unsqueeze(0)  
        cos = self.cos[:seq_len, :].unsqueeze(0).unsqueeze(0)
        return sin, cos


class FeedForward(nn.Module):
    """
    Transformer feedforward module
    """
    def __init__(self, embedding_dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """
    Attention Module with RoPE
    """
    def __init__(self, embedding_dim, num_heads=8, head_dim=64, dropout=0.0, max_seq_len=512):
        super().__init__()
        inner_dim = head_dim * num_heads
        project_out = not (num_heads == 1 and head_dim == embedding_dim)

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.norm = nn.LayerNorm(embedding_dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(embedding_dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, embedding_dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

        assert head_dim % 2 == 0, "head_dim must be even for RoPE"
        self.rotary_emb = RotaryEmbedding(head_dim, max_seq_len=max_seq_len)

    def forward(self, x):
        b, n, _ = x.shape
        x = self.norm(x)

        
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # [b, n, inner_dim]
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.num_heads), qkv)

        
        sin, cos = self.rotary_emb(n)  # sin, cos: [1, 1, n, head_dim//2]

        
        q1, q2 = q[..., :self.head_dim // 2], q[..., self.head_dim // 2:]
        k1, k2 = k[..., :self.head_dim // 2], k[..., self.head_dim // 2:]

        # apply rope
        q_rotated = torch.cat([q1 * cos - q2 * sin, q2 * cos + q1 * sin], dim=-1)
        k_rotated = torch.cat([k1 * cos - k2 * sin, k2 * cos + k1 * sin], dim=-1)

        # attention scores
        attn_scores = torch.matmul(q_rotated, k_rotated.transpose(-1, -2)) * self.scale  # [b, num_heads, n, n]
        attn_probs = self.attend(attn_scores)
        attn_probs = self.dropout(attn_probs)

     
        out = torch.matmul(attn_probs, v)  # [b, num_heads, n, head_dim]
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    """
    Transformer Module
    """
    def __init__(
        self, embedding_dim, depth, num_heads, head_dim, feedforward_dim, dropout=0.0, max_seq_len=512
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(
                            embedding_dim,
                            num_heads=num_heads,
                            head_dim=head_dim,
                            dropout=dropout,
                            max_seq_len=max_seq_len,
                        ),
                        FeedForward(embedding_dim, feedforward_dim, dropout=dropout),
                    ]
                )
            )

    def forward(self, x):
        for attn_layer, ff_layer in self.layers:
            x = attn_layer(x) + x  
            x = ff_layer(x) + x    
        return x
    

# 按照给定索引重新排列序列（在时间维T上重排）
def take_indices(sequence, indices):
    """
    Args:
        sequence (Tensor): [T, B, C]。
        indices (Tensor):  [T, B]。

    Returns:
        Tensor: [T, B, C]。
    """
    return torch.gather(
        sequence, 0, repeat(indices, "t b -> t b c", c=sequence.shape[-1])
    )

# 生成一组“随机顺序索引”和“反向索引”，forward_indices打乱顺序，backward_indices还原顺序
def random_indices(size: int):
    """
    Args:
        size (int): length of sequence

    Returns:
        Tuple[np.ndarray, np.ndarray]: forward_indices and backward_indices。
    """
    forward_indices = np.arange(size)
    np.random.shuffle(forward_indices)
    backward_indices = np.argsort(forward_indices)
    return forward_indices, backward_indices

# 在“指定的遮蔽位置”基础上生成打乱索引，让被遮蔽的位置被放在序列末尾，forward_indices非遮蔽随机+遮蔽放末尾，backward_indices用于还原顺序
def specified_mask_indices(size: int, mask_indices):
    """
    Args:
        size (int): length of sequence
        mask_indices (List[int]): indices of masking

    Returns:
        Tuple[np.ndarray, np.ndarray]: forward_indices and backward_indices。
    """
    forward_indices = np.arange(size)
    mask = np.isin(forward_indices, mask_indices, invert=True)
    remaining_indices = forward_indices[mask]
    np.random.shuffle(remaining_indices)
    forward_indices = np.concatenate([remaining_indices, mask_indices])
    backward_indices = np.argsort(forward_indices)
    return forward_indices, backward_indices


# 在时间维度上随机打乱 patch/token，并按 mask_ratio 丢掉一部分，返回打乱后的序列和用于还原顺序的索引，
# patches打乱且截断后的token，forward_indices打乱索引，backward_indices还原索引
class PatchShuffle(nn.Module):
    """
    random shuffle the patch or tokenizer 
    """
    def __init__(self, mask_ratio):
        super().__init__()
        self.mask_ratio = mask_ratio

    def forward(self, patches: torch.Tensor, mask_indices=None):
        """
        Args:
            patches (Tensor): tokenizer of patches [T, B, C]
            mask_indices (List[List[int]]): optional

        Returns:
            Tuple[Tensor, Tensor, Tensor]: shuffled tokenizer and index
        """
        T, B, C = patches.shape
        remain_T = int(T * (1 - self.mask_ratio))

        if mask_indices is not None:
            indices = [specified_mask_indices(T, mask_indices[i]) for i in range(B)]
            remain_T = T - len(mask_indices[0])
        else:
            indices = [random_indices(T) for _ in range(B)]

        forward_indices = torch.as_tensor(
            np.stack([i[0] for i in indices], axis=-1), dtype=torch.long
        ).to(patches.device)
        backward_indices = torch.as_tensor(
            np.stack([i[1] for i in indices], axis=-1), dtype=torch.long
        ).to(patches.device)

        patches = take_indices(patches, forward_indices)
        patches = patches[:remain_T]

        return patches, forward_indices, backward_indices


# 把轨迹序列切成 token、融合时间间隔、随机遮蔽并加 CLS，然后用 Transformer 编码，输出特征和还原索引
# 
class Encoder(nn.Module):
    def __init__(
        self,
        trajectory_length=200,
        patch_size=2,
        embedding_dim=128,
        num_layers=8,
        num_heads=4,
        mask_ratio=0.5,
    ):
        super().__init__()
        self.num_tokens = trajectory_length // patch_size
        self.max_seq_len = 512
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        self.shuffle = PatchShuffle(mask_ratio)
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

    def forward(self, trajectory, interval_embedding, mask_indices=None):
        """
        Args:
            trajectory (Tensor):  [B, 2, L]
            interval_embedding (Tensor): [B, L', C]
            mask_indices (List[List[int]]): optional

        Returns:
            Tuple[Tensor, Tensor]: the embedding and index
        """
        # apply tokenizer to trajectory
        tokens = self.tokenizer(trajectory)  # [B, embedding_dim, num_tokens]
        tokens = rearrange(tokens, "b c l -> l b c")  # [num_tokens, B, embedding_dim]

        # add iterval embedding
        interval_embedding = rearrange(interval_embedding, "b l c -> l b c")  # [num_tokens, B, embedding_dim]
        tokens = tokens + interval_embedding

        # shuffle tokenizer and masking
        tokens, forward_indices, backward_indices = self.shuffle(tokens, mask_indices)

        # add first token for downstream tasks
        tokens = torch.cat(
            [self.cls_token.expand(-1, tokens.shape[1], -1), tokens], dim=0  # [num_tokens+1, B, embedding_dim]
        )
        tokens = rearrange(tokens, "t b c -> b t c")  # [B, num_tokens+1, embedding_dim]

        # Transformer 
        features = self.transformer(tokens)
        features = self.layer_norm(features)
        features = rearrange(features, "b t c -> t b c")  # [num_tokens+1, B, embedding_dim]

        return features, backward_indices


class Decoder(nn.Module):
    def __init__(
        self,
        trajectory_length=200,
        patch_size=2,
        embedding_dim=192,
        num_layers=4,
        num_heads=2,
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

        self.head = nn.Linear(embedding_dim, 2 * patch_size)
        self.token_to_traj = Rearrange(
            "h b (c p) -> b c (h p)", p=patch_size, h=trajectory_length // patch_size
        )

        self._init_weights()

    def _init_weights(self):
        trunc_normal_(self.mask_token, std=0.02)
        trunc_normal_(self.time_token, std=0.02)

    def forward(self, features, backward_indices, interval_embedding):
        """
        Args:
            features (Tensor): [T, B, C]
            backward_indices (Tensor): [T-1, B]
            interval_embedding (Tensor):  [B, L', C]

        Returns:
            Tuple[Tensor, Tensor]: reconstructed trajectory
        """
        T,B = features.shape[0],features.shape[1]

        backward_indices = torch.cat(
            [
                torch.zeros(1, backward_indices.shape[1], dtype=backward_indices.dtype).to(backward_indices.device),
                backward_indices + 1,
            ],
            dim=0,
        )  # [num_tokens+1, B]

        num_masked = backward_indices.shape[0] - features.shape[0]
        features = torch.cat(
            [
                features,
                self.mask_token.expand(num_masked, B, -1),
            ],
            dim=0,
        )  # [num_tokens+1, B, embedding_dim]

        features = take_indices(features, backward_indices)

        interval_embedding = torch.cat(
            [self.time_token.expand(features.shape[1], 1, -1), interval_embedding], dim=1
        )  # [B, num_tokens+1, embedding_dim]

        interval_embedding = rearrange(interval_embedding, "b t c -> t b c")  # [num_tokens+1, B, embedding_dim]
        features = features + interval_embedding

        features = rearrange(features, "t b c -> b t c")  # [B, num_tokens+1, embedding_dim]
        features = self.transformer(features)
        features = rearrange(features, "b t c -> t b c")  # [num_tokens+1, B, embedding_dim]
        features = features[1:]   

        patches = self.head(features)  # [num_tokens, B, 2 * patch_size]

        mask = torch.zeros_like(patches)
        mask[T - 1 :] = 1
        mask = take_indices(mask, backward_indices[1:] - 1)

        traj = self.token_to_traj(patches)
        mask = self.token_to_traj(mask)

        return traj, mask


class UniTraj(nn.Module):
    def __init__(
        self,
        trajectory_length=32,
        patch_size=2,
        embedding_dim=128,
        encoder_layers=8,
        encoder_heads=4,
        decoder_layers=4,
        decoder_heads=2,
        mask_ratio=0.5,
    ):
        super().__init__()

        self.encoder = Encoder(
            trajectory_length, patch_size, embedding_dim, encoder_layers, encoder_heads, mask_ratio
        )
        self.decoder = Decoder(
            trajectory_length, patch_size, embedding_dim, decoder_layers, decoder_heads
        )
        self.interval_embedding = nn.Linear(1, embedding_dim)

    def forward(self, trajectory, intervals=None, mask_indices=None):
        """
        Args:
            trajectory (Tensor):  [B, 2, L]
            interval_embedding (Tensor): [B, L', C]
            mask_indices (List[List[int]]): optional

        Returns:
            Tuple[Tensor, Tensor]: the reconstructed trajectory and mask index
        """

        if intervals is not None:
            intervals = intervals.unsqueeze(-1)  # [B, L, 1]
            interval_embeddings = self.interval_embedding(intervals)  # [B, L, embedding_dim]
            # intervals_pooled = F.avg_pool1d(intervals.float(), kernel_size=self.encoder.tokenizer.kernel_size[0], stride=self.encoder.tokenizer.stride[0]).squeeze(1)  # [B, num_tokens]
            # interval_embeddings = self.interval_embedding(intervals_pooled.unsqueeze(-1))  # [B, num_tokens, embedding_dim]
        else:
            intervals_pooled = torch.zeros((trajectory.shape[0], self.encoder.num_tokens), device=trajectory.device)
            interval_embeddings = self.interval_embedding(intervals_pooled.unsqueeze(-1))  # [B, num_tokens, embedding_dim]
        
        mask_indices = mask_indices.cpu().numpy() if mask_indices is not None else None
        features, backward_indices = self.encoder(trajectory, interval_embeddings, mask_indices)
        predicted_trajectory, mask = self.decoder(features, backward_indices, interval_embeddings)
        return predicted_trajectory, mask
