"""Segmentation evaluation on merged clouds, as in the paper."""

from __future__ import annotations

import torch

from .. import metrics
from ..cluster import cluster_affinity
from ..inference import EncodedCloud
from ..model import PointAffinityTransformer


@torch.no_grad()
def evaluate(
    model: PointAffinityTransformer,
    dataset,
    thresholds=(0.8,),
    *,
    min_cluster_size: int = 30,
    symmetrize: bool = False,
    limit: int | None = None,
    chunk_size: int = 16384,
    progress: bool = True,
) -> dict[float, dict[str, float]]:
    """Cluster every item of ``dataset`` and compare to its ground-truth labels.

    The affinity matrix is computed once per item over its non-padding points
    and clustered at every threshold. Dust counts as one ground-truth segment.

    Returns:
        ``{threshold: {metric: mean value}}`` with ``voi``, ``voi_split``,
        ``voi_merge``, ``ari``, and pairwise ``precision``, ``recall``, ``f1``.
    """
    device = next(model.parameters()).device
    n = len(dataset) if limit is None else min(limit, len(dataset))
    sums = {t: {} for t in thresholds}
    pair_counts = {t: torch.zeros(3) for t in thresholds}  # tp, fp, fn

    items = range(n)
    if progress:
        from tqdm import tqdm

        items = tqdm(items, desc="eval")
    for i in items:
        item = dataset[i]
        pc, mask, gt = item["pc"].to(device), item["mask"].to(device), item["labels"][item["mask"]]
        valid = pc[mask]
        enc = EncodedCloud(model, model.encode(pc[None], mask[None]), valid, mask.nonzero()[:, 0])
        A = enc.affinity(valid, chunk_size=chunk_size).cpu()
        gt_same = gt[:, None] == gt[None, :]
        if item.get("dust_label") is not None:
            gt_same &= (gt != int(item["dust_label"]))[:, None]
        for t in thresholds:
            pred = cluster_affinity(A, threshold=t, min_cluster_size=min_cluster_size, symmetrize=symmetrize)
            vi, split, merge = metrics.voi(pred, gt)
            row = {"voi": vi, "voi_split": split, "voi_merge": merge, "ari": metrics.adjusted_rand_score(pred, gt)}
            for k, v in row.items():
                sums[t][k] = sums[t].get(k, 0.0) + v
            pred_same = pred[:, None] == pred[None, :]
            pair_counts[t] += torch.tensor([
                (pred_same & gt_same).sum(), (pred_same & ~gt_same).sum(), (~pred_same & gt_same).sum()
            ], dtype=torch.float32)

    results = {}
    for t in thresholds:
        tp, fp, fn = pair_counts[t].tolist()
        p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        results[t] = {k: v / n for k, v in sums[t].items()}
        results[t].update(precision=p, recall=r, f1=2 * p * r / max(p + r, 1e-12))
    return results


def format_results(results: dict) -> str:
    keys = ["voi", "voi_split", "voi_merge", "ari", "precision", "recall", "f1"]
    lines = ["threshold " + " ".join(f"{k:>9}" for k in keys)]
    for t, row in results.items():
        lines.append(f"{t:>9.2f} " + " ".join(f"{row[k]:9.3f}" for k in keys))
    return "\n".join(lines)
