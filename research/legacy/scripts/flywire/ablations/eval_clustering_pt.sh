#!/bin/bash

thresholds=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)

for t in "${thresholds[@]}"
do
    bsub -J "AE Eval (PT)" -n 8 -gpu "num=1" -q gpu_h100 -o "logs/paper_flywire_pt_eval_clustering_${t}.log" python -m benchmarks.transformer.eval \
        --model pt_b4_n16_dim64 \
        --pth /nrs/turaga/jakob/implicit-neurons/ckpt/ae/paper_flywire_pt_train/ckpt/checkpoint-766.pth \
        --data_path /nrs/turaga/jakob/implicit-neurons/flywire_full_v783/train \
        --neuron_id_path /nrs/turaga/jakob/implicit-neurons/flywire_full_v783/affinity/opticlobe_family/affinity_test_paper.csv \
        --output_dir "/nrs/turaga/jakob/implicit-neurons/ckpt/ae/paper_flywire_pt_eval_clustering_${t}" \
        --fam_to_id_mapping /nrs/turaga/jakob/implicit-neurons/flywire_full_v783/types/visual_neurons_family_to_id.json \
        --num_workers 6 \
        --point_cloud_size 1024 \
        --batch_size 1 \
        --data_global_scale_factor 659.88367 \
        --thresholds "${t}"
done