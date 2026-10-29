[![Paper](https://img.shields.io/badge/paper-arxiv-blue.svg?colorB=4AC8F4)](https://www.biorxiv.org/content/10.1101/2024.11.24.625067v1)
[![Data](https://img.shields.io/badge/data-gdrive-red.svg?colorB=f25100)](https://drive.google.com/drive/folders/1vgPSYsqDJyhv1s9aD09GgXqAeGm0Rb1V?usp=sharing)
[![Models](https://img.shields.io/badge/models-gdrive-purple.svg?colorB=C46CFD)](https://drive.google.com/drive/folders/1vgPSYsqDJyhv1s9aD09GgXqAeGm0Rb1V?usp=sharing)

## Global Neuron Shape Reasoning with Point Affinity Transformers
This repository contains the official implementation of the paper "[Global Neuron Shape Reasoning with Point Affinity Transformers](https://www.biorxiv.org/content/10.1101/2024.11.24.625067v1)" by 
[Jakob Troidl](https://jakobtroidl.github.io/), 
[Johannes Knittel](https://www.knittel.ai/), 
[Wanhua Li](https://li-wanhua.github.io/), 
[Fangneng Zhan](https://fnzhan.com/), 
[Hanspeter Pfister*](https://vcg.seas.harvard.edu/people), 
and [Srinivas Turaga*](https://www.janelia.org/people/srinivas-turaga) (*equal advising). 

https://github.com/user-attachments/assets/f28ef445-87d7-4427-a7ce-0ae0ac0bf7e6


### Installation
```bash
pip install point-affinity-transformer                        # core: torch, numpy, scikit-learn, safetensors
pip install "point-affinity-transformer[swc,pretrained]"      # + SWC reading and paper checkpoints
pip install "point-affinity-transformer[train]"               # + training and evaluation
```
The package is imported as `pat`. Install a PyTorch build matching your CUDA version first ([pytorch.org](https://pytorch.org/get-started/locally/)).

### Getting Started

```python
import pat
from pat.io import read_swcs

points, gt = read_swcs(["a.swc", "b.swc"])          # (N, 3) coordinates in µm
model = pat.load_pretrained("hemibrain", device="cuda")

seg = pat.segment(model, points, threshold=0.8)     # one label per point, -1 = noise
print(seg.num_neurons, pat.metrics.voi(seg.labels, gt))

enc = pat.encode(model, points)                     # encode once ...
A = enc.affinity(queries, others)                   # ... then query affinities between any points
```

See the [quickstart notebook](examples/quickstart.ipynb) for an end-to-end example, and the documentation
(`pip install -e ".[docs]" && mkdocs serve`) for concepts, training and the API reference.

### Data and Checkpoints
Download instructions for the FlyWire, Hemibrain and MANC data are [here](research/data/README.md).
`pat.load_pretrained("flywire" | "hemibrain" | "manc")` downloads and caches the paper checkpoints automatically.

### Training and Evaluation

```bash
pat train research/configs/flywire.yaml                                              # single GPU
torchrun --standalone --nproc_per_node 8 -m pat train research/configs/flywire.yaml  # multi-GPU
pat eval research/configs/hemibrain.yaml --model hemibrain --thresholds 0.7,0.8,0.9
```

Configs for all three datasets are in [`research/configs`](research/configs). Override entries with
`--set key=value`, e.g. `--set data.query_mode="{input_valid: 1, skeleton: 1}"` to train on query points beyond the input cloud.

### Repository Structure
```
src/pat/          pip package: model, inference, clustering, metrics, training (pat.train)
docs/             documentation (MkDocs)
examples/         quickstart notebook
tests/            pytest suite
research/
  configs/        paper training configs for FlyWire, Hemibrain, MANC
  lsf/            cluster launch scripts
  legacy/         original research code (incl. contrastive DeepSet embeddings)
  notebooks/      analysis notebooks
```

### Contrastive Neuron Embeddings

Training and test embeddings are available in the `embeddings` folder [here](https://drive.google.com/drive/folders/1vgPSYsqDJyhv1s9aD09GgXqAeGm0Rb1V?usp=sharing). Check out [this notebook](research/notebooks/confusion_matrix.ipynb) to generate neuron-type classification confusion matrices. The contrastive DeepSet code has not been ported to the package yet; it lives in [`research/legacy`](research/legacy) and runs from that directory, e.g. `cd research/legacy && python train_contrastive.py ...` with the arguments listed in its argparse help.

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

### Contact
Please open an issue or contact Jakob Troidl (troidlj@janelia.hhmi.org) for any questions or feedback.

### Known Issues
In all datasets the `train` folder contains all data files (train + test). The actual train and test split is defined by the neuron id CSV (`data.neuron_ids` / `data.eval_neuron_ids` in the configs).

