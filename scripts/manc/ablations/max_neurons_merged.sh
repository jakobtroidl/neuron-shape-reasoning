#!/bin/bash

max_neurons_merged=(1 2 3 4 5 6 7)

for n in "${max_neurons_merged[@]}"
do
    bsub -J "AE Eval (Topology)" -n 8 -gpu "num=1" -q gpu_h100 -o "logs/paper_manc_n_neurons_ablation_max_${n}.log" python eval_topology.py \
        --model ae_d1024_m512 \
        --pth /nrs/turaga/jakob/implicit-neurons/ckpt/ae/paper_manc_ours_train_v1/ckpt/checkpoint-946.pth \
        --data_path /nrs/turaga/jakob/implicit-neurons/manc_v1.0/swc \
        --types_path /nrs/turaga/jakob/implicit-neurons/manc_v1.0/affinity/test.csv \
        --output_dir "/nrs/turaga/jakob/implicit-neurons/ckpt/ae/paper_manc_n_neurons_ablation_max_${n}" \
        --fam_to_id_mapping /nrs/turaga/jakob/implicit-neurons/manc_v1.0/types/family_to_id.json \
        --num_workers 6 \
        --point_cloud_size 1024 \
        --batch_size 1 \
        --data_global_scale_factor 659.88367 \
        --thresholds 0.8 \
        --max_neurons_merged "${n}" \
        --translate_augmentation 70
done