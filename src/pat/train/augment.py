"""Point cloud augmentations used to synthesize training samples."""

from __future__ import annotations

import math

import torch


def jitter(points: torch.Tensor, amount: float) -> torch.Tensor:
    """Add uniform noise in ``[-amount, amount]`` to every coordinate."""
    return points + (torch.rand_like(points) * 2 - 1) * amount


def center(points: torch.Tensor) -> torch.Tensor:
    return points - points.mean(0, keepdim=True)


def random_offset(length: float) -> torch.Tensor:
    """A random 3D vector of the given length."""
    t = torch.rand(3) * 2 - 1
    return t / t.norm() * length


def random_rotation(degrees: float) -> torch.Tensor:
    """Rotation matrix about a random coordinate axis by an angle in ``[-degrees, degrees]``."""
    angle = math.radians((torch.rand(1).item() * 2 - 1) * degrees)
    c, s = math.cos(angle), math.sin(angle)
    axis = torch.randint(0, 3, (1,)).item()
    if axis == 0:
        m = [[1, 0, 0], [0, c, s], [0, -s, c]]
    elif axis == 1:
        m = [[c, 0, -s], [0, 1, 0], [s, 0, c]]
    else:
        m = [[c, s, 0], [-s, c, 0], [0, 0, 1]]
    return torch.tensor(m, dtype=torch.float32)
