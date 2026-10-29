"""Coordinate normalization, padding and subsampling.

Points are arrays of shape ``(N, 3 + F)``: xyz coordinates followed by ``F``
optional per-point features. Only the coordinates are normalized; features
are passed through unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

PAD_VALUE = -1.0
"""Coordinate (after normalization) used for padding points, as in training."""


def as_points(points, device=None, dim: int | None = None) -> torch.Tensor:
    """Convert an array-like of shape ``(N, 3 + F)`` to a float32 tensor.

    Args:
        points: Coordinates, optionally followed by feature columns.
        device: Target device.
        dim: Required number of columns, e.g. ``model.in_dim``.
    """
    t = torch.as_tensor(np.asarray(points) if not torch.is_tensor(points) else points)
    t = t.to(dtype=torch.float32, device=device)
    if t.ndim != 2 or t.shape[-1] < 3:
        raise ValueError(f"Expected points of shape (N, 3 + F), got {tuple(t.shape)}")
    if dim is not None and t.shape[-1] != dim:
        raise ValueError(
            f"Expected {dim} columns (xyz + {dim - 3} features), got {t.shape[-1]}"
        )
    return t


@dataclass
class Normalizer:
    """Maps physical coordinates (µm) to the model's normalized space.

    ``xyz_normalized = (xyz - center) / scale``; feature columns are unchanged.
    Fit it on the encoded cloud and apply the same instance to every query
    point, so queries and cloud share a coordinate frame.
    """

    center: torch.Tensor
    scale: float

    @classmethod
    def fit(cls, points, scale: float, center: bool = True) -> Normalizer:
        """Create a normalizer centered on the mean coordinate of ``points``.

        With ``center=False`` the points are only scaled.
        """
        points = as_points(points)
        c = points[:, :3].mean(0) if center else torch.zeros(3)
        return cls(center=c, scale=float(scale))

    def __call__(self, points) -> torch.Tensor:
        points = as_points(points, device=self.center.device)
        out = points.clone()
        out[:, :3] = (points[:, :3] - self.center) / self.scale
        return out

    def inverse(self, points: torch.Tensor) -> torch.Tensor:
        out = points.clone()
        out[..., :3] = points[..., :3] * self.scale + self.center.to(points.device)
        return out

    def to(self, device) -> Normalizer:
        return Normalizer(center=self.center.to(device), scale=self.scale)


def subsample(points: torch.Tensor, n: int, generator: torch.Generator | None = None):
    """Randomly pick ``n`` points without replacement.

    Returns ``(points[idx], idx)``. If there are at most ``n`` points, all are kept.
    """
    if points.shape[0] <= n:
        idx = torch.arange(points.shape[0], device=points.device)
    else:
        idx = torch.randperm(points.shape[0], generator=generator)[:n].sort().values
        idx = idx.to(points.device)
    return points[idx], idx


def pad_batch(clouds: list[torch.Tensor], size: int | None = None, pad_value: float = PAD_VALUE):
    """Stack clouds of different sizes into ``(B, size, 3 + F)`` plus a boolean mask.

    Padded points get coordinates ``pad_value`` and features ``0``. They are
    excluded from attention by the mask, so their values do not change the
    encoding.
    """
    size = size or max(c.shape[0] for c in clouds)
    device, dim = clouds[0].device, clouds[0].shape[-1]
    pc = torch.zeros(len(clouds), size, dim, device=device)
    pc[..., :3] = pad_value
    mask = torch.zeros(len(clouds), size, dtype=torch.bool, device=device)
    for i, c in enumerate(clouds):
        if c.shape[0] > size:
            raise ValueError(f"Cloud {i} has {c.shape[0]} points, more than size={size}")
        if c.shape[-1] != dim:
            raise ValueError(f"Cloud {i} has {c.shape[-1]} columns, expected {dim}")
        pc[i, : c.shape[0]] = c
        mask[i, : c.shape[0]] = True
    return pc, mask
