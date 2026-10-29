# Querying affinities

[`pat.encode`][pat.encode] runs the encoder once and returns an
[`EncodedCloud`][pat.EncodedCloud]. It can then be queried with any points, in the same
physical units as the cloud, without encoding again.

```python
enc = pat.encode(model, cloud_um)

A = enc.affinity()                    # (n, n) between the encoded points
A = enc.affinity(queries)             # (Q, Q) between arbitrary query points
A = enc.affinity(queries, others)     # (Q, K) rectangular
a = enc.pair_affinity(pa, pb)         # (P,)   explicit pairs, pa and pb of shape (P, 3)
a = enc.pair_affinity(pairs)          # (P,)   pairs of shape (P, 2, 3)
```

Everything runs in chunks of `chunk_size` pairs (default 16384). Lower it if the GPU
runs out of memory; raise it for speed.

## Labeling points that were not encoded

`pat.segment(model, cloud, queries)` labels `queries` against the encoding of `cloud`.
Typical uses:

- **Full-resolution labels:** encode a subsample and label every skeleton node or
  mesh vertex (this is the default when the cloud has more than `num_points` points).
- **Labeling a region:** compute labels only for the points you care about.

```python
seg = pat.segment(model, cloud_um, queries=all_nodes_um, max_anchors=4096)
```

## Low-level use

Work directly with the module for batched or custom pipelines. Normalize
coordinates yourself in that case:

```python
from pat import Normalizer
from pat.preprocess import pad_batch

norm = [Normalizer.fit(c, scale=model.config.scale) for c in clouds]
pc, mask = pad_batch([n(c) for n, c in zip(norm, clouds)])
latents = model.encode(pc, mask)                 # (B, M, D)
aff = model.decode(latents, qa, qb)              # qa, qb: (B, Q, 3), normalized
```

!!! warning "Out-of-distribution queries"
    The paper models only saw query points that were points of the input cloud.
    Queries off the skeletons, or far from the cloud, have no training signal. To
    support them, train with `query_mode: {skeleton: 1, jitter: 1}` (see [Training](training.md)).
