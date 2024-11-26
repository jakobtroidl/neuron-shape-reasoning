bsub -J "AE Train (Topology)" -n 96 -gpu "num=8" -q gpu_h100 -o logs/paper_flywire_ablation_d512_m512.log python main_ae_topology.py \
    --accum_iter 2 \
    --model ae_d512_m512 \
    --data_path /nrs/turaga/jakob/implicit-neurons/flywire_full_v783/train \
    --types_path /nrs/turaga/jakob/implicit-neurons/flywire_full_v783/affinity/opticlobe_family/affinity_train.csv \
    --output_dir /nrs/turaga/jakob/implicit-neurons/ckpt/ae/paper_flywire_ablation_d512_m512 \
    --log_dir /nrs/turaga/jakob/implicit-neurons/logs/ae/paper_flywire_ablation_d512_m512 \
    --fam_to_id_mapping /nrs/turaga/jakob/implicit-neurons/flywire_full_v783/types/visual_neurons_family_to_id.json \
    --num_workers 86 \
    --point_cloud_size 1024 \
    --batch_size 300 \
    --epochs 1000 \
    --warmup_epochs 5 \
    --data_global_scale_factor 659.88367 \
    --lr 1e-4 \
    --depth 24 \
    --distributed \