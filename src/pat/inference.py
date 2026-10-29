"""High-level inference: encode a cloud once, then query affinities anywhere."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .cluster import NOISE, cluster_affinity
from .model import PointAffinityTransformer
from .preprocess import Normalizer, as_points, subsample

DEFAULT_CHUNK_SIZE = 16384


@dataclass
class EncodedCloud:
    """A point cloud encoded by PAT. Query it with arbitrary points.

    Create it with [`pat.encode`][pat.encode]. Queries are given like the encoded
    cloud: ``(Q, 3 + F)`` with coordinates in the same physical units, followed
    by the same ``F`` feature columns. Their coordinates are normalized with the
    cloud's `normalizer`.

    Attributes:
        model: The model that produced the latents.
        latents: Latent vectors, shape ``(1, num_latents, dim)``.
        normalizer: Maps physical coordinates into the model frame.
        points: The (possibly subsampled) points that were encoded, physical units,
            including feature columns.
        index: Indices of `points` in the cloud passed to `encode`.
    """

    model: PointAffinityTransformer
    latents: torch.Tensor
    normalizer: Normalizer
    points: torch.Tensor
    index: torch.Tensor

    @property
    def device(self) -> torch.device:
        return self.latents.device

    def _embed(self, queries) -> torch.Tensor:
        q = self.normalizer(as_points(queries, device=self.device, dim=self.model.in_dim))
        return self.model.point_embed(q)

    @torch.no_grad()
    def pair_affinity(self, qa, qb=None, *, chunk_size: int = DEFAULT_CHUNK_SIZE) -> torch.Tensor:
        """Affinity of explicit point pairs.

        Args:
            qa: First points ``(P, 3 + F)``, or pairs ``(P, 2, 3 + F)`` if ``qb`` is None.
            qb: Second points ``(P, 3 + F)``.

        Returns:
            Affinities of shape ``(P,)``.
        """
        if qb is None:
            pairs = torch.as_tensor(qa, dtype=torch.float32)
            qa, qb = pairs[:, 0], pairs[:, 1]
        ea, eb = self._embed(qa), self._embed(qb)
        if ea.shape != eb.shape:
            raise ValueError("qa and qb must contain the same number of points")
        out = [
            self.model.decode(self.latents, a[None], b[None], embedded=True)[0]
            for a, b in zip(ea.split(chunk_size), eb.split(chunk_size))
        ]
        return torch.cat(out)

    @torch.no_grad()
    def affinity(self, queries=None, others=None, *, chunk_size: int = DEFAULT_CHUNK_SIZE) -> torch.Tensor:
        """Affinity matrix between two sets of query points.

        Args:
            queries: Points ``(Q, 3 + F)``. Defaults to the encoded points.
            others: Points ``(K, 3 + F)``. Defaults to ``queries``.

        Returns:
            ``A[i, j]`` is the affinity of ``(queries[i], others[j])``, shape ``(Q, K)``.
        """
        queries = self.points if queries is None else queries
        ea = self._embed(queries)
        eb = ea if others is None else self._embed(others)
        Q, K = ea.shape[0], eb.shape[0]

        out = torch.empty(Q * K, device=self.device)
        for start in range(0, Q * K, chunk_size):
            flat = torch.arange(start, min(start + chunk_size, Q * K), device=self.device)
            i, j = flat // K, flat % K
            out[start : start + flat.numel()] = self.model.decode(
                self.latents, ea[i][None], eb[j][None], embedded=True
            )[0]
        return out.view(Q, K)


@torch.no_grad()
def encode(
    model: PointAffinityTransformer,
    points,
    *,
    scale: float | None = None,
    center: bool = True,
    max_points: int | None = None,
    generator: torch.Generator | None = None,
) -> EncodedCloud:
    """Encode one point cloud.

    Args:
        model: A PAT model, e.g. from [`load_pretrained`][pat.load_pretrained].
        points: Coordinates ``(N, 3)`` in µm, followed by the model's
            ``F = model.config.feature_dim`` feature columns if it has any.
        scale: Global scale factor; defaults to ``model.config.scale``.
        center: Subtract the cloud's centroid before scaling.
        max_points: Clouds larger than this are randomly subsampled before
            encoding. Defaults to ``model.config.num_points``.
        generator: Random generator for subsampling.
    """
    device = next(model.parameters()).device
    points = as_points(points, dim=model.in_dim)
    max_points = max_points or model.config.num_points
    encoded, index = subsample(points, max_points, generator=generator)

    normalizer = Normalizer.fit(encoded, scale or model.config.scale, center=center).to(device)
    latents = model.encode(normalizer(encoded)[None])
    return EncodedCloud(model, latents, normalizer, encoded.to(device), index.to(device))


@dataclass
class Segmentation:
    """Result of [`pat.segment`][pat.segment].

    Attributes:
        labels: Neuron label per query point, ``-1`` for noise. Shape ``(Q,)``.
        queries: The query points, physical units, including feature columns.
        anchors: Indices of the queries that were clustered directly.
        affinity: Affinity matrix between the anchors.
    """

    labels: torch.Tensor
    queries: torch.Tensor
    anchors: torch.Tensor
    affinity: torch.Tensor

    @property
    def num_neurons(self) -> int:
        return int(self.labels.max().item()) + 1 if (self.labels >= 0).any() else 0


@torch.no_grad()
def segment(
    model: PointAffinityTransformer,
    points,
    queries=None,
    *,
    threshold: float = 0.8,
    min_cluster_size: int = 30,
    method: str = "agglomerative",
    max_anchors: int = 4096,
    symmetrize: bool = True,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    generator: torch.Generator | None = None,
    **encode_kwargs,
) -> Segmentation:
    """Split a point cloud into neurons.

    The cloud ``points`` is encoded once. Labels are computed for ``queries``,
    which can be any points (default: ``points`` itself).

    If there are more than ``max_anchors`` queries, a random subset (the anchors)
    is clustered on its full affinity matrix, and each remaining query joins
    the cluster with the highest mean affinity to it, provided that mean is
    above ``1 - threshold``. Otherwise it becomes noise. This keeps the cost
    linear in the number of queries.

    Args:
        model: A PAT model.
        points: Cloud to encode, ``(N, 3 + F)``: coordinates in µm, then features.
        queries: Points to label, ``(Q, 3 + F)``. Defaults to ``points``.
        threshold: Distance threshold, see [`cluster_affinity`][pat.cluster.cluster_affinity].
        min_cluster_size: Minimum anchors per neuron.
        method: Clustering method for the anchors.
        max_anchors: Maximum number of queries clustered directly.
        symmetrize: Symmetrize the anchor affinity matrix.
        chunk_size: Pairs per decoder pass; lower it if you run out of memory.
        generator: Random generator for subsampling.
        **encode_kwargs: Passed to [`encode`][pat.encode].
    """
    enc = encode(model, points, generator=generator, **encode_kwargs)
    queries = as_points(points if queries is None else queries, device=enc.device, dim=model.in_dim)
    Q = queries.shape[0]

    _, anchors = subsample(queries, max_anchors, generator=generator)
    A = enc.affinity(queries[anchors], chunk_size=chunk_size)
    anchor_labels = cluster_affinity(
        A, threshold=threshold, min_cluster_size=min_cluster_size, method=method, symmetrize=symmetrize
    ).to(enc.device)

    labels = torch.full((Q,), NOISE, dtype=torch.int64, device=enc.device)
    labels[anchors] = anchor_labels
    rest = torch.ones(Q, dtype=torch.bool, device=enc.device)
    rest[anchors] = False
    n_clusters = int(anchor_labels.max().item()) + 1 if (anchor_labels >= 0).any() else 0

    if rest.any() and n_clusters > 0:
        valid = anchor_labels >= 0
        onehot = torch.nn.functional.one_hot(anchor_labels[valid], n_clusters).float()
        rest_idx = rest.nonzero().squeeze(1)
        for chunk in rest_idx.split(max(1, chunk_size // max(1, anchors.numel()))):
            a = enc.affinity(queries[chunk], queries[anchors][valid], chunk_size=chunk_size)
            mean_aff = (a @ onehot) / onehot.sum(0)
            best, arg = mean_aff.max(dim=1)
            labels[chunk] = torch.where(best > 1.0 - threshold, arg, torch.full_like(arg, NOISE))

    return Segmentation(labels=labels.cpu(), queries=queries.cpu(), anchors=anchors.cpu(), affinity=A.cpu())
