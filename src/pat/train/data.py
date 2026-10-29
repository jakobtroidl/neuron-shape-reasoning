"""Datasets that synthesize merged-neuron point clouds and labeled query pairs."""

from __future__ import annotations

import glob
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset

from ..io import load_neuron
from . import augment
from .config import DataConfig

QUERY_MODES = ("input", "input_valid", "skeleton", "jitter")


class SkeletonStore:
    """Loads skeleton nodes and dust nodes (in ``units``) by root id.

    Dust nodes are the terminal twigs shorter than ``dust_size``. With a
    ``cache_dir``, parsed skeletons are stored as ``.pt`` files so each SWC is
    parsed only once.
    """

    def __init__(self, swc_dir, units: str = "um", dust_size: float = 8.0, cache_dir=None):
        self.swc_dir = Path(swc_dir)
        self.units = units
        self.dust_size = dust_size
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load(self, root_id) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(nodes, dust)``, each of shape ``(N, 3)``."""
        cached = self.cache_dir / f"{root_id}.pt" if self.cache_dir else None
        if cached is not None and cached.exists():
            d = torch.load(cached, weights_only=True)
            return d["nodes"], d["dust"]

        neuron = load_neuron(self.swc_dir / f"{root_id}.swc", self.units)
        nodes = torch.tensor(neuron.nodes[["x", "y", "z"]].values, dtype=torch.float32)
        pruned = neuron.prune_twigs(self.dust_size)
        is_dust = ~neuron.nodes["node_id"].isin(pruned.nodes["node_id"])
        dust = torch.tensor(neuron.nodes.loc[is_dust, ["x", "y", "z"]].values, dtype=torch.float32)

        if cached is not None:
            tmp = cached.with_suffix(f".{os.getpid()}.tmp")
            torch.save({"nodes": nodes, "dust": dust}, tmp)
            tmp.replace(cached)
        return nodes, dust


def read_root_ids(csv_path) -> list[int]:
    import pandas as pd

    return pd.read_csv(csv_path)["root_id"].tolist()


class MergedNeuronDataset(Dataset):
    """Synthetic merge errors: 1 to ``max_neurons_merged`` neurons plus dust.

    Item ``i`` always contains neuron ``root_ids[i]``; the partners, dust and
    augmentations are random.

    Each item is a dict with

    - ``pc`` ``(C, 3)``: normalized cloud, padded to the fixed size ``C``
    - ``mask`` ``(C,)``: ``False`` for padding
    - ``labels`` ``(C,)``: ``0`` padding, ``1..n`` neurons, ``n+1`` dust
    - ``root_ids`` ``(max_neurons_merged,)``: merged neurons, zero padded

    and in training mode additionally

    - ``qa``, ``qb`` ``(P, 3)``: normalized query pairs
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

    def _dust(self, idx: int) -> torch.Tensor:
        cfg = self.cfg
        out = []
        for rid in self._others(idx, cfg.n_dust_neurons):
            dust = self.store.load(rid)[1]
            k = cfg.n_dust_nodes_per_neuron
            if dust.shape[0] > 0:
                dust = dust[torch.randint(0, dust.shape[0], (k,)).sort().values]
            else:
                dust = torch.randn(k, 3) * cfg.scale
            out.append(augment.center(augment.jitter(dust, cfg.jitter)) + augment.random_offset(cfg.translate))
        return torch.cat(out) if out else torch.zeros(0, 3)

    def __getitem__(self, idx):
        cfg = self.cfg
        n_partners = torch.randint(0, cfg.max_neurons_merged, (1,)).item()
        ids = [self.root_ids[idx]] + self._others(idx, n_partners)

        points, labels, skeletons = [], [], []
        for label, rid in enumerate(ids, start=1):
            nodes = self.store.load(rid)[0]
            nodes = augment.center(augment.jitter(nodes, cfg.jitter)) + augment.random_offset(cfg.translate)
            sel = torch.randint(0, nodes.shape[0], (cfg.samples_per_neuron,)).sort().values
            points.append(nodes[sel])
            labels.append(torch.full((cfg.samples_per_neuron,), label))
            skeletons.append((nodes, label))

        dust_label = len(ids) + 1
        if self.n_dust > 0:
            points.append(self._dust(idx))
            labels.append(torch.full((self.n_dust,), dust_label))

        n = sum(p.shape[0] for p in points)
        pc = torch.full((self.cloud_size, 3), -cfg.scale)
        pc[:n] = torch.cat(points)
        point_labels = torch.zeros(self.cloud_size, dtype=torch.int64)
        point_labels[:n] = torch.cat(labels)
        mask = point_labels > 0

        rot = augment.random_rotation(cfg.rotate_degrees)
        pc = pc @ rot.T / cfg.scale

        root_ids = torch.zeros(cfg.max_neurons_merged, dtype=torch.int64)
        root_ids[: len(ids)] = torch.tensor(ids, dtype=torch.int64)
        item = {"pc": pc, "mask": mask, "labels": point_labels, "root_ids": root_ids,
                "dust_label": dust_label if self.n_dust > 0 else -1}
        if not self.train:
            return item

        skel = [(nodes @ rot.T / cfg.scale, torch.full((nodes.shape[0],), lab)) for nodes, lab in skeletons]
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
        # jitter
        valid = pc[mask]
        return valid + torch.randn_like(valid) * (self.cfg.query_jitter / self.cfg.scale), labels[mask]

    def _sample_pairs(self, pc, labels, mask, skel, dust_label):
        P = self.num_pairs
        mode_of_pair = torch.multinomial(self._mode_weights, P, replacement=True)
        qa, qb = torch.empty(P, 3), torch.empty(P, 3)
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
