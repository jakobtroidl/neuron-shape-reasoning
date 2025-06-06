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
```
conda create --name gnsr python=3.9
conda activate gnsr
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
pip install torch_geometric # more details here https://pytorch-geometric.readthedocs.io/en/2.5.2/notes/installation.html
pip install -r requirements.txt
```


<details>
  <summary>Troubleshooting & Versions!</summary>

  All code was tested using PyTorch version 2.1.0 and Cuda version 12.1. <br>
  ```
  pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
  ```
</details>

### Data Access
View download instructions for data and model checkpoints [here](https://github.com/jakobtroidl/neuron-shape-reasoning/blob/970d25cfb496f8bbbba62ba68eb087601db2f6b6/data/README.md).

### Getting Started

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

Training and test embeddings are available in the `embeddings` folder [here](https://drive.google.com/drive/folders/1vgPSYsqDJyhv1s9aD09GgXqAeGm0Rb1V?usp=sharing). Check out [this notebook](https://github.com/jakobtroidl/neuron-shape-reasoning/blob/main/notebooks/confusion_matrix.ipynb) to generate neuron-type classification confusion matrices. If you want to train a deepset to produce contrastive neuron embeddings on the FlyWire dataset:

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
Please open an issue or contact Jakob Troidl (jtroidl@g.harvard.edu) for any questions or feedback.

### Known Issues
In all datasets the `train` folder contains all data files (train + test). The actual train and test split is defined in the file given through the `--neuron_id_path` argument.

