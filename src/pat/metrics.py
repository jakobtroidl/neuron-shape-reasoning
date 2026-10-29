"""Segmentation metrics comparing predicted to ground-truth labels."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import adjusted_rand_score as _ari
from sklearn.metrics.cluster import contingency_matrix


def _labels(x) -> np.ndarray:
    return np.asarray(x.cpu() if hasattr(x, "cpu") else x).astype(np.int64).ravel()


def _entropy(counts: np.ndarray, n: int) -> float:
    p = counts[counts > 0] / n
    return float(-(p * np.log(p)).sum())


def voi(pred, gt) -> tuple[float, float, float]:
    """Variation of information (natural log).

    Returns:
        ``(voi, split, merge)`` with ``split = H(pred | gt)`` (over-segmentation)
        and ``merge = H(gt | pred)`` (under-segmentation).
        Noise (``-1``) is treated as one regular label.
    """
    pred, gt = _labels(pred), _labels(gt)
    n = pred.size
    joint = contingency_matrix(gt, pred, sparse=True).data
    h_joint = _entropy(joint, n)
    h_pred = _entropy(np.unique(pred, return_counts=True)[1], n)
    h_gt = _entropy(np.unique(gt, return_counts=True)[1], n)
    split, merge = h_joint - h_gt, h_joint - h_pred
    return split + merge, split, merge


def adjusted_rand_score(pred, gt) -> float:
    """Adjusted Rand index (1 = identical partitions)."""
    return float(_ari(_labels(gt), _labels(pred)))
