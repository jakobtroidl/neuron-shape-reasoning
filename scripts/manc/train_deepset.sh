bsub -J "AE Train (DeepSet)" -n 12 -gpu "num=1" -q gpu_h100 -o logs/deepset_train_v3_not_normed.log python main_contrastive.py \
    --model ae_d1024_m512 \
    --pth /nrs/turaga/jakob/implicit-neurons/ckpt/ae/affinity_benchmark_family_v1/ckpt/checkpoint-632.pth \
    --data_path /nrs/turaga/jakob/implicit-neurons/flywire_full_v783/train \
    --neuron_id_path /nrs/turaga/jakob/implicit-neurons/flywire_full_v783/affinity/ol_family_balanced/affinity_train_metric_balanced.csv \
    --output_dir /nrs/turaga/jakob/implicit-neurons/ckpt/ae/deepset_train_v3_not_normed \
    --num_workers 1 \
    --point_cloud_size 1024 \
    --batch_size 650 \
    --data_global_scale_factor 659.88367 \
    --depth 24 \
    # --norm_emb