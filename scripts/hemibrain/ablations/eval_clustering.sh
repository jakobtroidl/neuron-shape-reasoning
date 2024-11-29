#!/bin/bash

thresholds=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)

for t in "${thresholds[@]}"
do
    bsub -J "AE Eval (Topology)" -n 8 -gpu "num=1" -q gpu_h100 -o "logs/paper_hemibrain_ours_${t}.log" python eval_topology.py \
        --model ae_d1024_m512 \
        --pth /nrs/turaga/jakob/implicit-neurons/ckpt/ae/paper_hemibrain_ours_train_v1/ckpt/checkpoint-946.pth \
        --data_path /nrs/turaga/jakob/implicit-neurons/hemibrain_v1.2/swc \
        --neuron_id_path /nrs/turaga/jakob/implicit-neurons/hemibrain_v1.2/affinity/test.csv \
        --output_dir "/nrs/turaga/jakob/implicit-neurons/ckpt/ae/paper_hemibrain_ours_${t}$" \
        --fam_to_id_mapping /nrs/turaga/jakob/implicit-neurons/hemibrain_v1.2/types/family_to_id.json \
        --num_workers 6 \
        --point_cloud_size 1024 \
        --batch_size 1 \
        --data_global_scale_factor 659.88367 \
        --thresholds "${t}" \
        --translate_augmentation 60
done
