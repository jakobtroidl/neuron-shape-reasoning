import torch

import pat


def test_segment_full_and_assign(tiny_model):
    pts = torch.randn(300, 3) * 20
    g = torch.Generator().manual_seed(0)
    full = pat.segment(tiny_model, pts, threshold=0.9, min_cluster_size=1, max_anchors=1000, generator=g)
    assert full.labels.shape == (300,) and full.affinity.shape == (300, 300)

    queries = torch.randn(500, 3) * 20
    res = pat.segment(tiny_model, pts, queries, threshold=0.9, min_cluster_size=1, max_anchors=64, generator=g)
    assert res.labels.shape == (500,) and res.anchors.numel() == 64 and res.affinity.shape == (64, 64)
    assert res.labels.min() >= -1 and res.labels.max() < max(res.num_neurons, 1)
