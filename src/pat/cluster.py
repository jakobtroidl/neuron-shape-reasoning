"""Clustering of affinity matrices into neuron labels.

The label ``-1`` marks noise: DBSCAN outliers and, after
[`remove_small_clusters`][pat.cluster.remove_small_clusters], points of clusters
smaller than ``min_cluster_size``.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn import cluster as _skcluster

NOISE = -1


def to_numpy(x) -> np.ndarray:
    """Convert a tensor or array-like to a NumPy array."""
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def num_clusters(labels) -> int:
    """Number of clusters in ``labels`` numbered ``0..K-1``, ignoring noise (``-1``)."""
    labels = torch.as_tensor(labels)
    return int(labels.max().item()) + 1 if (labels >= 0).any() else 0


def agglomerative(distance, threshold: float = 0.8, linkage: str = "average") -> torch.Tensor:
    """Agglomerative clustering of a precomputed ``(N, N)`` distance matrix.

    Clusters are merged while their linkage distance is below ``threshold``.
    """
    d = to_numpy(distance)
    model = _skcluster.AgglomerativeClustering(
        n_clusters=None, metric="precomputed", distance_threshold=threshold, linkage=linkage
    )
    return torch.as_tensor(model.fit_predict(d), dtype=torch.int64)


def dbscan(distance, eps: float = 0.2, min_samples: int = 5) -> torch.Tensor:
    """DBSCAN on a precomputed distance matrix. Noise points get label ``-1``."""
    d = to_numpy(distance)
    labels = _skcluster.DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed").fit_predict(d)
    return torch.as_tensor(labels, dtype=torch.int64)


def remove_small_clusters(labels: torch.Tensor, min_size: int) -> torch.Tensor:
    """Mark clusters with fewer than ``min_size`` points as noise and relabel.

    The remaining clusters are numbered ``0..K-1``, largest first. Existing
    ``-1`` labels stay noise.
    """
    labels = torch.as_tensor(labels, dtype=torch.int64)
    out = torch.full_like(labels, NOISE)
    valid = labels >= 0
    if not valid.any():
        return out
    ids, inverse, counts = torch.unique(labels[valid], return_inverse=True, return_counts=True)
    keep = counts >= min_size
    order = torch.argsort(counts.masked_fill(~keep, -1), descending=True, stable=True)
    new_id = torch.full_like(ids, NOISE)
    new_id[order[: int(keep.sum())]] = torch.arange(int(keep.sum()))
    out[valid] = new_id[inverse]
    return out


def cluster_affinity(
    affinity,
    threshold: float = 0.8,
    min_cluster_size: int = 30,
    method: str = "agglomerative",
    symmetrize: bool = True,
) -> torch.Tensor:
    """Turn an ``(N, N)`` affinity matrix into one label per point.

    Args:
        affinity: Pairwise same-neuron probabilities.
        threshold: Distance threshold on ``1 - affinity``. For average linkage,
            two groups merge while their mean affinity exceeds ``1 - threshold``.
            For DBSCAN this is ``eps``.
        min_cluster_size: Smaller clusters become noise (``-1``). Use ``0`` to keep all.
        method: ``"agglomerative"`` or ``"dbscan"``.
        symmetrize: Average ``A`` and ``A.T`` first. The model is not exactly
            symmetric in its two inputs; the paper evaluation used ``False``.
    """
    a = to_numpy(affinity).astype(np.float32)
    if symmetrize:
        a = 0.5 * (a + a.T)
    dist = np.clip(1.0 - a, 0.0, None)
    np.fill_diagonal(dist, 0.0)
    if method == "agglomerative":
        labels = agglomerative(dist, threshold=threshold)
    elif method == "dbscan":
        labels = dbscan(dist, eps=threshold)
    else:
        raise ValueError(f"Unknown method {method!r}")
    return remove_small_clusters(labels, min_cluster_size) if min_cluster_size > 0 else labels
