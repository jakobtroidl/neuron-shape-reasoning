
**Disclaimer: Code Still under construction**

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
and [Srinivas Turaga*](https://www.janelia.org/people/srinivas-turaga). 


TODO add video here

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
Training a Point Affinity Transformer Model on the FlyWire dataset:
```bash
python train_affinity.py \
    --model ae_d1024_m512 \
    --data_path ./data/flywire_full_v783/train \
    --types_path ./data/flywire_full_v783/affinity/ol_family_balanced/affinity_train.csv \
    --output_dir ./ckpt/flywire_affinity_train \
    --log_dir ./logs/flywire_affinity_train \
    --fam_to_id_mapping ./data/flywire_full_v783/types/visual_neurons_family_to_id.json \
    --point_cloud_size 1024 \
    --data_global_scale_factor 659.88367 \
    --lr 1e-4 \
```
