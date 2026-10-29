# Installation

```bash
pip install point-affinity-transformer
```

The package installs as `point-affinity-transformer` and is imported as `pat`.
The core depends only on PyTorch, NumPy, scikit-learn and safetensors. Optional
features come as extras:

| Extra | Adds | Needed for |
|---|---|---|
| `swc` | navis | reading SWC files (`pat.io`) |
| `pretrained` | gdown | downloading the paper checkpoints |
| `train` | navis, pandas, pyyaml, tqdm | `pat train`, `pat eval`, `pat.train` |
| `wandb` | wandb | logging training runs to Weights & Biases |
| `docs` | mkdocs-material, mkdocstrings, mkdocs-jupyter | building this documentation |

```bash
pip install "point-affinity-transformer[swc,pretrained]"   # inference on SWC files
pip install "point-affinity-transformer[train]"            # training
```

## PyTorch and CUDA

Install a PyTorch build that matches your CUDA version first, following
[pytorch.org](https://pytorch.org/get-started/locally/), for example:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

PyTorch 2.1 or newer is required. The paper experiments used PyTorch 2.1 with CUDA 12.1.

## From source

```bash
git clone https://github.com/jakobtroidl/neuron-shape-reasoning
cd neuron-shape-reasoning
pip install -e ".[swc,pretrained,train,dev]"
pytest
```

!!! note "Name clash"
    An unrelated PyPI package called `pat` also installs a top-level `pat` module.
    Do not install both into the same environment.
