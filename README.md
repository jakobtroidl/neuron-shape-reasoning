[![Paper](https://img.shields.io/badge/paper-arxiv-blue.svg?colorB=4AC8F4)](https://www.biorxiv.org/content/10.1101/2024.11.24.625067v1)
[![Data](https://img.shields.io/badge/data-gdrive-red.svg?colorB=f25100)](https://drive.google.com/drive/folders/1vgPSYsqDJyhv1s9aD09GgXqAeGm0Rb1V?usp=sharing)
[![Models](https://img.shields.io/badge/models-gdrive-purple.svg?colorB=C46CFD)](https://drive.google.com/drive/folders/1vgPSYsqDJyhv1s9aD09GgXqAeGm0Rb1V?usp=sharing)
[![PyPI](https://img.shields.io/pypi/v/point-affinity-transformer.svg)](https://pypi.org/project/point-affinity-transformer/)

## Point Affinity Transformers


### Installation
```bash
pip install point-affinity-transformer
```
The package is imported as `pat`. Install a PyTorch build matching your CUDA version first ([pytorch.org](https://pytorch.org/get-started/locally/)).

### Getting Started

```python
import pat
from pat.io import read_swcs

points, gt = read_swcs(["a.swc", "b.swc"])          # (N, 3) coordinates in µm
points = points / 659.88                            # your normalization: the same as for the training data
model = pat.load_pretrained("my_model", device="cuda")  # directory from pat train or pat.save_pretrained

seg = pat.segment(model, points, threshold=0.8)     # one label per point, -1 = noise
print(seg.num_neurons, pat.metrics.voi(seg.labels, gt))

enc = pat.encode(model, points)                     # encode once ...
A = enc.affinity(queries, others)                   # ... then query affinities between any points
```

### Point Clouds and Features

Point clouds of shape `(B, N, 3)` with `D` extra features are passed to the model as `(B, N, 3 + D)`. Training only needs a neuron label per point.

```python
import torch
import torch.nn.functional as F
import pat

B, N, D = 8, 1024, 4                           # random toy data: replace with your own
xyz = torch.rand(B, N, 3) * 2 - 1              # coordinates, already normalized to roughly [-1, 1]
features = torch.randn(B, N, D)                # D features per point
labels = torch.randint(0, 3, (B, N))           # neuron id of every point (training targets)
pc = torch.cat([xyz, features], dim=-1)        # (B, N, 3 + D)

model = pat.create_model("ae_d256_m64", depth=4, feature_dim=D)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Training: sample random point pairs, target = 1 if both lie on the same neuron
def gather(x, idx):                            # x: (B, N, C), idx: (B, Q) -> (B, Q, C)
    return x.gather(1, idx[..., None].expand(-1, -1, x.shape[-1]))

model.train()
for step in range(100):
    i, j = torch.randint(0, N, (2, B, 2048))   # 2048 query pairs per cloud
    target = (labels.gather(1, i) == labels.gather(1, j)).float()
    logits = model(pc, None, gather(pc, i), gather(pc, j), logits=True)  # mask=None: no padding
    loss = F.binary_cross_entropy_with_logits(logits, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Inference
model.eval()
with torch.no_grad():
    latents = model.encode(pc)                             # (B, num_latents, dim), once per cloud
    affinity = model.decode(latents, pc[:, :5], pc[:, 5:10])  # (B, 5): any query pairs, with features

seg = pat.segment(model, pc[0])                # cluster one cloud: labels (N,), -1 = noise
pat.save_pretrained(model, "my_model")         # reload with pat.load_pretrained("my_model")
```

Coordinates should lie within roughly [-1, 1]. Clouds of different sizes can be padded with `pat.preprocess.pad_batch`, which returns the `mask` to pass instead of
`None`. A trained 3D model can be extended to `D` features with `pat.expand_features(model, D)` and then fine-tuned
the same way. To train from SWC files, run `pat train config.yaml` (examples in [`research/configs`](research/configs)),
optionally with `data.features: [radius]` or, from Python, `pat.train.train(cfg, feature_fn=...)`.

### Model Sizes

`pat.create_model(name)` accepts the presets `ae_d{dim}_m{latents}` for every combination below. Parameter counts are
for the default `depth=24`; the number of latents changes them by at most 1.4%.

| `dim` | Parameters | Latents (`m`) |
|---|---|---|
| 64 | 4.5M | 64, 256, 512, 1024 |
| 128 | 12M | 64, 256, 512, 1024 |
| 256 | 33M | 64, 256, 512, 1024 |
| 512 | 108M | 64, 256, 512, 1024 |
| 1024 | 380M | 64, 256, 512, 1024 |
| 2048 | 1.4B | 64, 256, 512, 1024 |

The paper models use `ae_d1024_m512` (the default). Other settings, such as `depth`, can be passed as keyword
arguments, e.g. `pat.create_model("ae_d256_m64", depth=4)`.

See the [quickstart notebook](examples/quickstart.ipynb) for an end-to-end example, and the documentation
(`pip install -e ".[docs]" && mkdocs serve`) for the API reference.

### Citation
```bibtex
@techreport{troidlgnsr2024,
  title = {Global Neuron Shape Reasoning with Point Affinity Transformers},
  author = {Troidl, Jakob and Knittel, Johannes and Li, Wanhua and Zhan, Fengnang and Pfister*, Hanspeter and Turaga*, Srinivas},
  journal = {bioRxiv},
  year = {2024},
  publisher = {Cold Spring Harbor Laboratory},
  keywords = {preprint}
}
```

### Acknowledgements
We acknowledge NSF grants CRCNS-2309041, NCS-FO-2124179, and NIH grant R01HD104969. We also thank the HHMI Janelia Visiting Scientist Program and the Harvard Data Science Initiative Postdoctoral Fellowship for their support. The code is partially based on [3DShape2VecSet](https://arxiv.org/abs/2301.11445) by Zhang et. al.

