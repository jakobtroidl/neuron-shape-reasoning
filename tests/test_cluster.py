import torch

import pat
from pat.cluster import NOISE, cluster_affinity, remove_small_clusters


def block_affinity(sizes, noise=0.05, seed=0):
    g = torch.Generator().manual_seed(seed)
    labels = torch.cat([torch.full((s,), i) for i, s in enumerate(sizes)])
    A = (labels[:, None] == labels[None]).float() * 0.9 + 0.05
    A += (torch.rand(A.shape, generator=g) - 0.5) * noise
    return A.clamp(0, 1), labels


def test_remove_small_clusters():
    labels = torch.tensor([0, 0, 0, 1, 2, 2, 2, 2, -1])
    out = remove_small_clusters(labels, min_size=3)
    assert out.tolist() == [1, 1, 1, NOISE, 0, 0, 0, 0, NOISE]  # largest cluster first
    assert (remove_small_clusters(torch.tensor([0, 1]), 5) == NOISE).all()


def test_cluster_affinity_recovers_blocks():
    A, gt = block_affinity([50, 40, 5])
    pred = cluster_affinity(A, threshold=0.5, min_cluster_size=10)
    assert pred[:50].unique().numel() == 1 and pred[50:90].unique().numel() == 1
    assert pred[0] != pred[50]
    assert (pred[90:] == NOISE).all()
    vi, split, merge = pat.metrics.voi(pred[:90], gt[:90])
    assert abs(vi) < 1e-9
    assert pat.metrics.adjusted_rand_score(pred[:90], gt[:90]) == 1.0


def test_dbscan():
    A, _ = block_affinity([30, 30])
    pred = cluster_affinity(A, threshold=0.3, method="dbscan", min_cluster_size=0)
    assert pred.unique().numel() == 2
