"""Coordinate normalization, padding and subsampling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

PAD_VALUE = -1.0
"""Coordinate (after normalization) used for padding points, as in training."""


def as_points(points, device=None) -> torch.Tensor:
    """Convert an array-like of shape ``(N, 3)`` to a float32 tensor."""
    t = torch.as_tensor(np.asarray(points) if not torch.is_tensor(points) else points)
    t = t.to(dtype=torch.float32, device=device)
    if t.ndim != 2 or t.shape[-1] != 3:
        raise ValueError(f"Expected points of shape (N, 3), got {tuple(t.shape)}")
    return t


@dataclass
class Normalizer:
    """Maps physical coordinates (µm) to the model's normalized space.

    ``normalized = (points - center) / scale``. Fit it on the encoded cloud and
    apply the same instance to every query point, so queries and cloud share a
    coordinate frame.
    """

    center: torch.Tensor
    scale: float

    @classmethod
    def fit(cls, points, scale: float, center: bool = True) -> "Normalizer":
        """Create a normalizer centered on the mean of ``points``.

        With ``center=False`` the points are only scaled.
        """
        points = as_points(points)
        c = points.mean(0) if center else torch.zeros(3)
        return cls(center=c, scale=float(scale))

    def __call__(self, points) -> torch.Tensor:
        points = as_points(points, device=self.center.device)
        return (points - self.center) / self.scale

    def inverse(self, points: torch.Tensor) -> torch.Tensor:
        return points * self.scale + self.center.to(points.device)

    def to(self, device) -> "Normalizer":
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
    """Stack clouds of different sizes into ``(B, size, 3)`` plus a boolean mask.

    Padded points are excluded from attention by the mask, so their value does
    not change the encoding.
    """
    size = size or max(c.shape[0] for c in clouds)
    device = clouds[0].device
    pc = torch.full((len(clouds), size, 3), pad_value, device=device)
    mask = torch.zeros(len(clouds), size, dtype=torch.bool, device=device)
    for i, c in enumerate(clouds):
        if c.shape[0] > size:
            raise ValueError(f"Cloud {i} has {c.shape[0]} points, more than size={size}")
        pc[i, : c.shape[0]] = c
        mask[i, : c.shape[0]] = True
    return pc, mask
