"""Datasets that synthesize merged-neuron point clouds and labeled query pairs."""

from __future__ import annotations

import glob
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset

from ..io import load_neuron, node_array
from ..preprocess import PAD_VALUE
from . import augment
from .config import DataConfig

QUERY_MODES = ("input", "input_valid", "skeleton", "jitter")


class SkeletonStore:
    """Loads skeleton nodes and dust nodes (in ``units``) by root id.

    Nodes are returned as ``(N, 3 + F)``: xyz followed by per-node features.
    Features are the SWC node-table columns named in ``features`` (e.g.
    ``"radius"``), followed by the columns returned by ``feature_fn``.

    Dust nodes are the terminal twigs shorter than ``dust_size``. With a
    ``cache_dir``, parsed skeletons are stored as ``.pt`` files so each SWC is
    parsed only once. The cache is keyed on the feature column names and the
    name of ``feature_fn``; clear it after changing what ``feature_fn`` computes.

    Args:
        swc_dir: Directory with ``{root_id}.swc`` files.
        units: Units to convert coordinates to.
        dust_size: Maximum twig length counted as dust, in ``units``.
        cache_dir: Optional cache directory.
        features: Node-table columns to use as features.
        feature_fn: Optional ``f(root_id, neuron) -> array (N, F)`` with one row
            per row of ``neuron.nodes`` (a ``navis.TreeNeuron``).
    """

    def __init__(self, swc_dir, units: str = "um", dust_size: float = 8.0, cache_dir=None,
                 features=(), feature_fn=None):
        self.swc_dir = Path(swc_dir)
        self.units = units
        self.dust_size = dust_size
        self.features = list(features)
        self.feature_fn = feature_fn
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def _cache_key(self) -> list[str]:
        return self.features + ([getattr(self.feature_fn, "__name__", "feature_fn")] if self.feature_fn else [])

    def load(self, root_id) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(nodes, dust)``, each of shape ``(N, 3 + F)``."""
        cached = self.cache_dir / f"{root_id}.pt" if self.cache_dir else None
        if cached is not None and cached.exists():
            d = torch.load(cached, weights_only=True)
            if d.get("features", []) == self._cache_key:
                return d["nodes"], d["dust"]

        neuron = load_neuron(self.swc_dir / f"{root_id}.swc", self.units)
        nodes = torch.from_numpy(node_array(neuron, self.features))
        if self.feature_fn is not None:
            extra = torch.as_tensor(self.feature_fn(root_id, neuron), dtype=torch.float32).reshape(len(nodes), -1)
            nodes = torch.cat([nodes, extra], dim=1)
        pruned = neuron.prune_twigs(self.dust_size)
        is_dust = torch.from_numpy(~neuron.nodes["node_id"].isin(pruned.nodes["node_id"]).to_numpy())
        dust = nodes[is_dust]

        if cached is not None:
            tmp = cached.with_suffix(f".{os.getpid()}.tmp")
            torch.save({"nodes": nodes, "dust": dust, "features": self._cache_key}, tmp)
            tmp.replace(cached)
        return nodes, dust


def read_root_ids(csv_path) -> list[int]:
    import pandas as pd

    return pd.read_csv(csv_path)["root_id"].tolist()


class MergedNeuronDataset(Dataset):
    """Synthetic merge errors: 1 to ``max_neurons_merged`` neurons plus dust.

    Item ``i`` always contains neuron ``root_ids[i]``; the partners, dust and
    augmentations are random.

    Points have ``3 + F`` columns: xyz followed by the store's ``F`` features.
    Augmentations (jitter, offsets, rotation, division by ``cfg.scale``) only
    touch xyz; padding points have zero features. Dust is sampled from the
    short twigs of other neurons, or from the whole skeleton of a neuron that
    has none. Each item is a dict with

    - ``pc`` ``(C, 3 + F)``: cloud divided by ``cfg.scale``, padded to the fixed size ``C``
    - ``mask`` ``(C,)``: ``False`` for padding
    - ``labels`` ``(C,)``: ``0`` padding, ``1..n`` neurons, ``n+1`` dust
    - ``dust_label``: the dust label ``n+1``, or ``-1`` without dust
    - ``root_ids`` ``(max_neurons_merged,)``: merged neurons, zero padded

    and in training mode additionally

    - ``qa``, ``qb`` ``(P, 3 + F)``: query pairs, divided by ``cfg.scale``
    - ``target`` ``(P,)``: ``1`` if both queries lie on the same neuron (not dust)

    Query points are drawn per pair from one of the ``cfg.query_mode`` sources:

    - ``input``: any point of the padded cloud, including padding (paper setting)
    - ``input_valid``: any non-padding point of the cloud
    - ``skeleton``: any node of the merged skeletons, also ones not in the cloud
    - ``jitter``: non-padding cloud points plus Gaussian noise (``query_jitter`` µm)
    """

    def __init__(self, store: SkeletonStore, root_ids: list, cfg: DataConfig, train: bool = True):
        unknown = set(cfg.query_mode) - set(QUERY_MODES)
        if unknown:
            raise ValueError(f"Unknown query modes {unknown}; choose from {QUERY_MODES}")
        if len(root_ids) < 2:
            raise ValueError("Need at least two neurons to synthesize merges")
        self.store = store
        self.root_ids = list(root_ids)
        self.cfg = cfg
        self.train = train
        self.n_dust = cfg.n_dust_neurons * cfg.n_dust_nodes_per_neuron
        self.cloud_size = cfg.samples_per_neuron * cfg.max_neurons_merged + self.n_dust
        self.num_pairs = cfg.num_pairs or self.cloud_size
        self._modes = list(cfg.query_mode)
        self._mode_weights = torch.tensor([float(cfg.query_mode[m]) for m in self._modes])

    def __len__(self):
        return len(self.root_ids)

    def _others(self, idx: int, n: int) -> list:
        choices = torch.randint(0, len(self.root_ids) - 1, (n,))
        choices = choices + (choices >= idx).long()  # skip idx itself
        return [self.root_ids[i] for i in choices.tolist()]

    def _place(self, points: torch.Tensor) -> torch.Tensor:
        """Jitter and center the coordinates, then move them by a random offset."""
        cfg = self.cfg
        out = points.clone()
        out[:, :3] = augment.center(augment.jitter(points[:, :3], cfg.jitter)) + augment.random_offset(cfg.translate)
        return out

    def _dust(self, idx: int, width: int) -> torch.Tensor:
        cfg = self.cfg
        out = []
        for rid in self._others(idx, cfg.n_dust_neurons):
            nodes, dust = self.store.load(rid)
            if dust.shape[0] == 0:  # no short twigs: sample from the whole skeleton
                dust = nodes
            dust = dust[torch.randint(0, dust.shape[0], (cfg.n_dust_nodes_per_neuron,)).sort().values]
            out.append(self._place(dust))
        return torch.cat(out) if out else torch.zeros(0, width)

    def __getitem__(self, idx):
        cfg = self.cfg
        n_partners = torch.randint(0, cfg.max_neurons_merged, (1,)).item()
        ids = [self.root_ids[idx]] + self._others(idx, n_partners)

        points, labels, skeletons = [], [], []
        for label, rid in enumerate(ids, start=1):
            nodes = self._place(self.store.load(rid)[0])
            sel = torch.randint(0, nodes.shape[0], (cfg.samples_per_neuron,)).sort().values
            points.append(nodes[sel])
            labels.append(torch.full((cfg.samples_per_neuron,), label))
            skeletons.append((nodes, label))

        width = points[0].shape[1]
        dust_label = len(ids) + 1
        if self.n_dust > 0:
            points.append(self._dust(idx, width))
            labels.append(torch.full((self.n_dust,), dust_label))

        n = sum(p.shape[0] for p in points)
        pc = torch.zeros(self.cloud_size, width)
        pc[:, :3] = PAD_VALUE * cfg.scale  # becomes PAD_VALUE after scaling
        pc[:n] = torch.cat(points)
        point_labels = torch.zeros(self.cloud_size, dtype=torch.int64)
        point_labels[:n] = torch.cat(labels)
        mask = point_labels > 0

        rot = augment.random_rotation(cfg.rotate_degrees)

        def transform(x):
            x = x.clone()
            x[:, :3] = x[:, :3] @ rot.T / cfg.scale
            return x

        pc = transform(pc)

        root_ids = torch.zeros(cfg.max_neurons_merged, dtype=torch.int64)
        root_ids[: len(ids)] = torch.tensor(ids, dtype=torch.int64)
        item = {"pc": pc, "mask": mask, "labels": point_labels, "root_ids": root_ids,
                "dust_label": dust_label if self.n_dust > 0 else -1}
        if not self.train:
            return item

        skel = [(transform(nodes), torch.full((nodes.shape[0],), lab)) for nodes, lab in skeletons]
        qa, qb, target = self._sample_pairs(pc, point_labels, mask, skel, dust_label)
        item.update(qa=qa, qb=qb, target=target)
        return item

    def _pool(self, mode, pc, labels, mask, skel):
        if mode == "input":
            return pc, labels
        if mode == "input_valid":
            return pc[mask], labels[mask]
        if mode == "skeleton":
            return torch.cat([s for s, _ in skel]), torch.cat([lab for _, lab in skel])
        # jitter: noise on the coordinates only
        valid = pc[mask].clone()
        valid[:, :3] += torch.randn_like(valid[:, :3]) * (self.cfg.query_jitter / self.cfg.scale)
        return valid, labels[mask]

    def _sample_pairs(self, pc, labels, mask, skel, dust_label):
        P = self.num_pairs
        mode_of_pair = torch.multinomial(self._mode_weights, P, replacement=True)
        qa, qb = torch.empty(P, pc.shape[1]), torch.empty(P, pc.shape[1])
        la, lb = torch.empty(P, dtype=torch.int64), torch.empty(P, dtype=torch.int64)
        for m, mode in enumerate(self._modes):
            sel = mode_of_pair == m
            k = int(sel.sum())
            if k == 0:
                continue
            pts, lab = self._pool(mode, pc, labels, mask, skel)
            ia, ib = torch.randint(0, pts.shape[0], (k,)), torch.randint(0, pts.shape[0], (k,))
            qa[sel], qb[sel], la[sel], lb[sel] = pts[ia], pts[ib], lab[ia], lab[ib]
        target = (la == lb) & (la > 0) & (la != dust_label)
        return qa, qb, target.float()


class TensorDirDataset(Dataset):
    """Pre-generated evaluation items (``pc_{i}.pt``, ``labels_{i}.pt``, ``mask_{i}.pt``).

    This is the ``reproducable_items`` format of the research code. Clouds are
    already normalized, and the highest label is dust.
    """

    def __init__(self, path):
        self.path = Path(path)
        self.n = len(glob.glob(str(self.path / "pc_*.pt")))

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        load = lambda name: torch.load(self.path / f"{name}_{i}.pt", weights_only=True)  # noqa: E731
        labels = load("labels").long()
        return {"pc": load("pc").float(), "mask": load("mask").bool(), "labels": labels,
                "dust_label": int(labels.max())}
