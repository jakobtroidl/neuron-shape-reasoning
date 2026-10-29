# Installation

```bash
pip install point-affinity-transformer
```

The package installs as `point-affinity-transformer` and is imported as `pat`. It
includes inference, SWC reading, training and evaluation.

## PyTorch and CUDA

Install a PyTorch build that matches your CUDA version first, following
[pytorch.org](https://pytorch.org/get-started/locally/), for example:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

PyTorch 2.1 or newer is required. The paper experiments used PyTorch 2.1 with CUDA 12.1.

!!! note "Name clash"
    An unrelated PyPI package called `pat` also installs a top-level `pat` module.
    Do not install both into the same environment.
