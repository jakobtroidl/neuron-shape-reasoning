# Research code

Code and settings behind the paper. The reusable model, inference and training code lives in the
`pat` package (`src/pat`); this folder holds everything specific to the paper experiments.

| Path | Contents |
|---|---|
| `configs/` | `pat train` / `pat eval` configs reproducing the paper runs on FlyWire, Hemibrain and MANC |
| `lsf/` | `bsub` wrappers for the Janelia cluster, e.g. `research/lsf/train.sh research/configs/flywire.yaml` |
| `data/README.md` | data and checkpoint download instructions |
| `notebooks/` | analysis notebooks (neuron-type confusion matrices) |
| `legacy/` | the original research code: `train_affinity.py`, `eval_affinity.py`, contrastive DeepSet training and evaluation, shell scripts of all paper runs and ablations |

The legacy scripts import `src.*` and must be run from `research/legacy/`. They need the
full research environment (`pip install -r research/requirements.txt`, plus `torch_geometric`).
Paths below are relative to the repository root as before the restructuring; adjust them accordingly.

### Paper checkpoints

| Name | Training data | Preset | Scale | Translate (µm) |
|---|---|---|---|---|
| `flywire` | FlyWire v783, optic lobe | `ae_d1024_m512`, depth 24 | 659.88367 | 20 |
| `hemibrain` | Hemibrain v1.2 | `ae_d1024_m512`, depth 24 | 659.88367 | 60 |
| `manc` | MANC v1.0 | `ae_d1024_m512`, depth 24 | 659.88367 | 70 |

The checkpoints are listed in [data/README.md](data/README.md). They are research-code training checkpoints
(about 4.5 GB each, including optimizer state). Convert one into a `pat` model directory (1.5 GB) with

```bash
pat convert hemibrain_final.pth ckpt/hemibrain/model
```

or let `pat.load_pretrained("hemibrain")` download (needs `gdown`) and convert it into `~/.cache/pat/hemibrain`.
Paper evaluation on the fixed test items:

```bash
pat eval --model ckpt/hemibrain/model --items data/hemibrain_v1.2/affinity/reproducable_items --thresholds 0.8
```

On Hemibrain this gives VOI 0.200, ARI 0.923 and pairwise F1 0.933 at threshold 0.8.

### Original commands

Training a Point Affinity Transformer Model on the FlyWire dataset:

```bash
python train_affinity.py \
    --data_path ./data/flywire_full_v783/train \
    --neuron_id_path ./data/flywire_full_v783/affinity/ol_family_balanced/affinity_train.csv \
    --fam_to_id_mapping ./data/flywire_full_v783/types/visual_neurons_family_to_id.json \
    --output_dir ./ckpt/flywire_affinity_train \
    --log_dir ./logs/flywire_affinity_train \
    --point_cloud_size 1024 \
    --data_global_scale_factor 659.88367 \
    --lr 1e-4 \
```

Testing a Pretrained Affinity Model on the FlyWire dataset:

```bash
python eval_affinity.py \
    --pth ./path/to/flywire_final.pth \
    --data_path ./data/flywire_full_v783/train \
    --neuron_id_path .data/flywire_full_v783/affinity/opticlobe_family/affinity_test_paper.csv \
    --output_dir ./ckpt/flywire_affinity_eval \
    --fam_to_id_mapping ./data/flywire_full_v783/types/visual_neurons_family_to_id.json \
    --point_cloud_size 1024 \
    --batch_size 1 \
    --data_global_scale_factor 659.88367 \
    --thresholds 0.8 \
    --store_tensors \
    --qual_results
```

### Contrastive Neuron Embeddings

Training and test embeddings are available in the `embeddings` folder [here](https://drive.google.com/drive/folders/1vgPSYsqDJyhv1s9aD09GgXqAeGm0Rb1V?usp=sharing). Check out [this notebook](notebooks/confusion_matrix.ipynb) to generate neuron-type classification confusion matrices. If you want to train a deepset to produce contrastive neuron embeddings on the FlyWire dataset:

```bash
python train_contrastive.py \
    --model ae_d1024_m512 \
    --pth ./data/ckpt/flywire_affinity_final.pth \
    --data_path ./data/flywire_full_v783/train \
    --neuron_id_path ./data/flywire_full_v783/affinity/ol_family_balanced/affinity_train_metric_balanced.csv \
    --output_dir ./ckpt/flywire_deepset_train_normed \
    --fam_to_id_mapping ./data/flywire_full_v783/types/visual_neurons_family_to_id.json \
    --point_cloud_size 1024 \
    --batch_size 650 \
    --data_global_scale_factor 659.88367 \
    --depth 24 \
    --norm_emb
```

Test pretrained Deepset model on the FlyWire dataset:

```bash
python eval_contrastive.py \
    --model ae_d1024_m512 \
    --encoder_pth ./data/ckpt/flywire_affinity_final.pth \
    --deep_set_pth ./data/ckpt/flywire_deepset_final.pth \
    --data_path ./data/flywire_full_v783/train \
    --neuron_id_path ./data/flywire_full_v783/affinity/ol_family_balanced/affinity_test.csv \
    --output_dir ./ckpt/flywire_deepset_eval \
    --train_emb_path ./ckpt/flywire_deepset_train_normed/emb_ep_XXXX/ \
    --fam_to_id_mapping ./data/flywire_full_v783/types/visual_neurons_family_to_id.json \
    --point_cloud_size 1024 \
    --batch_size 650 \
    --data_global_scale_factor 659.88367 \
    --depth 24 \
    --norm_emb
```



