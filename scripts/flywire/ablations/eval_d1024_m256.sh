bsub -J "AE Eval (Topology)" -n 8 -gpu "num=1" -q gpu_h100 -o logs/mapped_labels_paper_flywire_ablation_d1024_m256_eval.log python eval_topology.py \
    --model ae_d1024_m256 \
    --pth /nrs/turaga/jakob/implicit-neurons/ckpt/ae/paper_flywire_ablation_d1024_m256/ckpt/checkpoint-946.pth \
    --data_path /nrs/turaga/jakob/implicit-neurons/flywire_full_v783/train \
    --types_path /nrs/turaga/jakob/implicit-neurons/flywire_full_v783/affinity/opticlobe_family/affinity_test_paper.csv \
    --output_dir /nrs/turaga/jakob/implicit-neurons/ckpt/ae/mapped_labels_paper_flywire_ablation_d1024_m256_eval \
    --fam_to_id_mapping /nrs/turaga/jakob/implicit-neurons/flywire_full_v783/types/visual_neurons_family_to_id.json \
    --num_workers 6 \
    --point_cloud_size 1024 \
    --batch_size 1 \
    --data_global_scale_factor 659.88367 \
    --thresholds 0.8 \
    # --store_tensors \
    # --qual_results