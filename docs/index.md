# Point Affinity Transformer

`pat` splits point clouds of merged neurons into individual neurons. It is the
reference implementation of
[Global Neuron Shape Reasoning with Point Affinity Transformers](https://www.biorxiv.org/content/10.1101/2024.11.24.625067v1)
(Troidl, Knittel, Li, Zhan, Pfister\*, Turaga\*).

A Point Affinity Transformer (PAT) encodes a whole point cloud into a set of
latent vectors. It then predicts, for any pair of points, the probability
that both belong to the same neuron. Clustering these affinities yields one
label per point.

```python
import pat

model = pat.load_pretrained("my_model", device="cuda")  # trained with pat train
seg = pat.segment(model, points)             # points: (N, 3), normalized like the training data
seg.labels                                   # one neuron id per point, -1 = noise
```

Query points do not have to be part of the encoded cloud:

```python
enc = pat.encode(model, points)               # encode once
A = enc.affinity(queries, others)             # (Q, K) affinities between any points
```

- **[Installation](installation.md)**: pip and CUDA notes
- **[API reference](api.md)**: all public functions and classes
- **[Citation](citation.md)**: how to cite the paper
