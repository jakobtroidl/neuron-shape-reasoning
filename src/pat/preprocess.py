"""Input validation, padding and subsampling.

Points are arrays of shape ``(N, 3 + F)``: xyz coordinates followed by ``F``
optional per-point features. ``pat`` never rescales or re-centers them;
normalize your data yourself, the same way for training and inference.
"""

from __future__ import annotations

import numpy as np
import torch

PAD_VALUE = -1.0
"""Coordinate value of padding points. Padding is masked out of the encoder."""


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
