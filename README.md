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

### Overview
TODO


### Installation
```
conda create --name gnsr python=3.9
conda activate gnsr
pip install torch torchvision torchaudio
pip install -r requirements.txt
```


<details>
  <summary>Troubleshooting & Versions!</summary>

  All code was tested using PyTorch version 2.1.0 and Cuda version 12.1. <br>
  ```
  pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
  ```
</details>


### Getting Started

View download instructions for data and model checkpoints [here](https://github.com/jakobtroidl/neuron-shape-reasoning/blob/303676557368178a0d3ee6ed1794532272634729/data/README.md). Training a Point Affinity Transformer Model on the FlyWire dataset:

```bash
python train_affinity.py \
    --data_path ./data/flywire_full_v783/train \
    --types_path ./data/flywire_full_v783/affinity/ol_family_balanced/affinity_train.csv \
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
    --types_path .data/flywire_full_v783/affinity/opticlobe_family/affinity_test_paper.csv \
    --output_dir ./ckpt/flywire_affinity_eval \
    --fam_to_id_mapping ./data/flywire_full_v783/types/visual_neurons_family_to_id.json \
    --point_cloud_size 1024 \
    --batch_size 1 \
    --data_global_scale_factor 659.88367 \
    --thresholds 0.8 \
    --store_tensors \
    --qual_results
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

