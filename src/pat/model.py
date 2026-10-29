"""Point Affinity Transformer (PAT).

The encoder compresses a point cloud into a fixed set of latent vectors. The
decoder predicts, for a pair of 3D query points, the probability that both
belong to the same neuron. Because the point embedding acts on each point
independently, query points do not have to be part of the encoded cloud.

Module and parameter names match the original research code, so its
checkpoints load with ``strict=True``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class PATConfig:
    """Architecture and preprocessing settings of a PAT model.

    Attributes:
        dim: Width of point embeddings and latents.
        num_latents: Number of learned latent vectors.
        depth: Number of latent self-attention blocks.
        heads: Attention heads in the latent self-attention blocks.
        dim_head: Width of each attention head.
        drop_path_rate: Stochastic depth rate used during training.
        num_points: Largest cloud size seen during training. Larger inputs
            are subsampled to this size before encoding.
        feature_dim: Number of extra per-point features after the xyz
            coordinates. Inputs then have ``3 + feature_dim`` columns.
    """

    dim: int = 1024
    num_latents: int = 512
    depth: int = 24
    heads: int = 8
    dim_head: int = 64
    drop_path_rate: float = 0.1
    num_points: int = 4288
    feature_dim: int = 0

    @property
    def in_dim(self) -> int:
        """Number of input columns: 3 coordinates plus `feature_dim` features."""
        return 3 + self.feature_dim

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PATConfig":
        fields = cls.__dataclass_fields__
        return cls(**{k: v for k, v in d.items() if k in fields})


class DropPath(nn.Module):
    """Stochastic depth: randomly drops the whole residual branch per sample."""

    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.p == 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        return x * x.new_empty(shape).bernoulli_(keep) / keep


class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim=None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if context_dim is not None else None

    def forward(self, x, **kwargs):
        x = self.norm(x)
        if self.norm_context is not None:
            kwargs["context"] = self.norm_context(kwargs["context"])
        return self.fn(x, **kwargs)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, drop_path_rate=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim),
        )
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

    def forward(self, x):
        return self.drop_path(self.net(x))


class Attention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, drop_path_rate=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = context_dim if context_dim is not None else query_dim
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

    def forward(self, x, context=None, mask=None):
        context = x if context is None else context
        q = self.to_q(x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        # (B, N, h*d) -> (B, h, N, d)
        B = x.shape[0]
        q, k, v = (t.reshape(B, t.shape[1], self.heads, -1).transpose(1, 2) for t in (q, k, v))

        attn_mask = None
        if mask is not None:
            attn_mask = mask.reshape(B, 1, 1, -1)  # True = attend

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(B, x.shape[1], -1)
        return self.drop_path(self.to_out(out))


class PointEmbed(nn.Module):
    """Embedding of each point independently.

    The xyz coordinates (first 3 columns) are Fourier-encoded. The encoding,
    the raw coordinates and any extra feature columns are concatenated and
    projected to ``dim``.
    """

    def __init__(self, hidden_dim=90, dim=128, feature_dim=0):
        super().__init__()
        assert hidden_dim % 6 == 0

        self.embedding_dim = hidden_dim
        self.feature_dim = feature_dim
        n = hidden_dim // 6
        e = torch.pow(2, torch.arange(n)).float() * math.pi
        z = torch.zeros(n)
        e = torch.stack([
            torch.cat([e, z, z]),
            torch.cat([z, e, z]),
            torch.cat([z, z, e]),
        ])
        self.register_buffer("basis", e)  # 3 x hidden_dim/2

        self.mlp = nn.Linear(self.embedding_dim + 3 + feature_dim, dim)

    def forward(self, x):
        # x: (..., 3 + feature_dim), coordinates first
        if x.shape[-1] != 3 + self.feature_dim:
            raise ValueError(
                f"Expected {3 + self.feature_dim} input columns (xyz + {self.feature_dim} features), "
                f"got {x.shape[-1]}"
            )
        proj = x[..., :3] @ self.basis
        pe = torch.cat([proj.sin(), proj.cos()], dim=-1)
        return self.mlp(torch.cat([pe, x], dim=-1))


class MLP(nn.Module):
    """Affinity head. Returns logits; apply a sigmoid to get probabilities."""

    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, 1),
        )

    def forward(self, x):
        return self.fc(x)


class PointAffinityTransformer(nn.Module):
    """Predicts same-neuron affinities between pairs of points of a point cloud.

    Typical use is to call [`encode`][pat.PointAffinityTransformer.encode] once per
    cloud and [`decode`][pat.PointAffinityTransformer.decode] for as many query
    pairs as needed. Coordinates are used as given; normalize them the same
    way for training and inference.

    Points and queries have ``config.in_dim = 3 + config.feature_dim`` columns:
    xyz followed by optional per-point features.
    """

    def __init__(self, config: PATConfig | None = None, **overrides):
        super().__init__()
        config = config or PATConfig()
        if overrides:
            config = PATConfig.from_dict({**config.to_dict(), **overrides})
        self.config = config
        dim = config.dim

        self.point_embed = PointEmbed(dim=dim, feature_dim=config.feature_dim)
        self.learnable_query = nn.Parameter(torch.randn(1, config.num_latents, dim))

        self.cross_attend_blocks = nn.ModuleList([
            PreNorm(dim, Attention(dim, dim, heads=1, dim_head=dim), context_dim=dim),
            PreNorm(dim, FeedForward(dim)),
        ])

        self.layers = nn.ModuleList([
            nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=config.heads, dim_head=config.dim_head,
                                       drop_path_rate=config.drop_path_rate)),
                PreNorm(dim, FeedForward(dim, drop_path_rate=config.drop_path_rate)),
            ])
            for _ in range(config.depth)
        ])

        self.decoder_cross_attn = PreNorm(
            dim * 2, Attention(dim * 2, dim, heads=1, dim_head=dim), context_dim=dim
        )
        self.mlp = MLP(dim * 2)

    @property
    def in_dim(self) -> int:
        return self.config.in_dim

    def encode(self, pc: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Encode a batch of point clouds into latents.

        Args:
            pc: Normalized points, shape ``(B, N, in_dim)``.
            mask: Optional boolean mask ``(B, N)``; ``False`` marks padding.

        Returns:
            Latents of shape ``(B, num_latents, dim)``.
        """
        cross_attn, cross_ff = self.cross_attend_blocks
        x = self.learnable_query.expand(pc.shape[0], -1, -1)
        x = cross_attn(x, context=self.point_embed(pc), mask=mask) + x
        x = cross_ff(x) + x
        for self_attn, self_ff in self.layers:
            x = self_attn(x) + x
            x = self_ff(x) + x
        return x

    def pair_features(
        self,
        latents: torch.Tensor,
        qa: torch.Tensor,
        qb: torch.Tensor,
        *,
        embedded: bool = False,
    ) -> torch.Tensor:
        """Decoder features for query pairs, shape ``(B, Q, 2 * dim)``.

        Args:
            latents: Output of `encode`, shape ``(B, M, dim)``.
            qa: First query of each pair: points ``(B, Q, in_dim)``,
                or point embeddings ``(B, Q, dim)`` if ``embedded=True``.
            qb: Second query of each pair, same format as ``qa``.
        """
        if not embedded:
            qa, qb = self.point_embed(qa), self.point_embed(qb)
        return self.decoder_cross_attn(torch.cat([qa, qb], dim=-1), context=latents)

    def decode(
        self,
        latents: torch.Tensor,
        qa: torch.Tensor,
        qb: torch.Tensor,
        *,
        embedded: bool = False,
        logits: bool = False,
    ) -> torch.Tensor:
        """Affinity for each query pair ``(qa[:, i], qb[:, i])``, shape ``(B, Q)``.

        Returns probabilities in ``[0, 1]``, or raw logits if ``logits=True``.
        """
        out = self.mlp(self.pair_features(latents, qa, qb, embedded=embedded)).squeeze(-1)
        return out if logits else out.sigmoid()

    def forward(self, pc, mask, qa, qb, *, logits: bool = False) -> torch.Tensor:
        return self.decode(self.encode(pc, mask), qa, qb, logits=logits)


def expand_features(model: PointAffinityTransformer, feature_dim: int) -> PointAffinityTransformer:
    """Copy of ``model`` that accepts ``feature_dim`` per-point features.

    The input weights of the new feature columns are zero, so the copy gives
    exactly the same outputs as ``model`` until it is trained. Use this to
    fine-tune a trained 3D model with additional features.
    """
    old = model.config.feature_dim
    if feature_dim < old:
        raise ValueError(f"Model already has {old} features, cannot reduce to {feature_dim}")
    new = PointAffinityTransformer(PATConfig.from_dict({**model.config.to_dict(), "feature_dim": feature_dim}))
    state = model.state_dict()
    w = state["point_embed.mlp.weight"]
    state["point_embed.mlp.weight"] = torch.cat([w, w.new_zeros(w.shape[0], feature_dim - old)], dim=1)
    new.load_state_dict(state)
    return new.to(w.device).train(model.training)
