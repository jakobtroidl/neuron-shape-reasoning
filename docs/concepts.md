# Concepts

## Model

```
points (N, 3) ──PointEmbed──► (N, D) ──cross-attention──► latents (M, D) ──24× self-attention──► latents (M, D)
                                                                                                   │
query pair (a, b) ──PointEmbed──► [e(a) ‖ e(b)] (2D) ──cross-attention to latents──► MLP ──► affinity(a, b)
```

- **Encoder** ([`PointAffinityTransformer.encode`][pat.PointAffinityTransformer.encode]):
  `M` learned queries cross-attend to the Fourier-embedded points, followed by
  `depth` self-attention blocks. The result is a fixed-size description of the
  whole cloud, independent of `N`.
- **Decoder** ([`PointAffinityTransformer.decode`][pat.PointAffinityTransformer.decode]):
  the embeddings of two query points are concatenated and cross-attend to the
  latents. A small MLP outputs the probability that both points lie on the same neuron.

`PointEmbed` acts on each point independently, so a query point does not have to
be one of the encoded points. See [Querying affinities](queries.md).

Model sizes are named `ae_d{D}_m{M}` ([`pat.PRESETS`][pat.PRESETS]). The paper models use
`ae_d1024_m512` with depth 24, about 380M parameters.

## Preprocessing

[`pat.encode`][pat.encode] applies the same transformation that training used:

1. **Units:** coordinates are expected in **µm**. [`pat.io.read_swc`][pat.io.read_swc]
   converts SWC files that declare their units.
2. **Normalization:** the cloud is centered on its centroid and divided by the global
   `scale` stored in the model config (659.88367 for the paper models). A
   [`Normalizer`][pat.Normalizer] fitted on the cloud is applied to every query
   point too, so queries and cloud share one coordinate frame.
3. **Subsampling:** clouds larger than `model.config.num_points` (4288 for the paper
   models) are randomly subsampled before encoding. Queries are not subsampled.

Padding is only needed to batch clouds of different sizes
([`pad_batch`][pat.preprocess.pad_batch]). Padded points are masked out of
attention and do not change the encoding.

!!! warning "Training distribution"
    The paper models were trained on 1 to 4 neurons, each centered and moved by a
    random offset of 20 to 70 µm, with 1024 nodes per neuron plus small "dust"
    fragments. They work best on inputs that look similar: merges of a few neurons,
    not whole brain volumes. Two properties matter in practice:

    - **Position.** Every training neuron sat exactly `translate` µm from the origin, and
      the models picked this up. On Hemibrain test merges, re-centering a cloud that
      was already in the training frame lowered ARI from 0.89–0.99 to 0.46–0.88 in
      three of four cases. Use `center=False` when your cloud is constructed like a
      training sample; keep the default `center=True` for data in brain coordinates.
    - **Balance.** Training always used the same number of points per neuron. If one
      neuron has many more nodes than the others, uniform subsampling over-represents
      it and accuracy drops.

    Fine-tuning with your own data distribution (see [Training](training.md)) is the
    reliable fix for both.

## Clustering

[`pat.cluster.cluster_affinity`][pat.cluster.cluster_affinity] converts an `(N, N)` affinity
matrix `A` into labels:

1. Optionally symmetrize: `A ← (A + Aᵀ) / 2`.
2. Run average-linkage agglomerative clustering on distances `1 - A`, merging groups
   while their mean distance is below `threshold`. DBSCAN is available as well.
3. Clusters with fewer than `min_cluster_size` points become noise (`-1`). The
   remaining clusters are numbered `0..K-1`, largest first.

[`pat.segment`][pat.segment] runs these steps on at most `max_anchors` query points.
Each remaining query joins the anchor cluster with the highest mean affinity to it,
provided that mean exceeds `1 - threshold`; otherwise it becomes noise. The cost
therefore grows linearly with the number of queries instead of quadratically.

### Choosing the threshold

Higher thresholds merge more aggressively (fewer, larger neurons). Lower
thresholds split more. The paper uses `0.8`. The split and merge terms of
[`pat.metrics.voi`][pat.metrics.voi] show which error dominates on your data, and
`pat eval --thresholds 0.5,0.6,0.7,0.8,0.9` sweeps several values at once.
