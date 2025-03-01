bsub -J "AE Train (Topology)" -n 96 -gpu "num=8" -q gpu_h100 -o logs/paper_manc_ours_train_v1.log python main_ae_topology.py \
    --accum_iter 2 \
    --model ae_d1024_m512 \
    --data_path /nrs/turaga/jakob/implicit-neurons/manc_v1.0/swc \
    --neuron_id_path /nrs/turaga/jakob/implicit-neurons/manc_v1.0/affinity/train.csv \
    --output_dir /nrs/turaga/jakob/implicit-neurons/ckpt/ae/paper_manc_ours_train_v1 \
    --log_dir /nrs/turaga/jakob/implicit-neurons/logs/ae/paper_manc_ours_train_v1 \
    --fam_to_id_mapping /nrs/turaga/jakob/implicit-neurons/manc_v1.0/types/family_to_id.json \
    --num_workers 86 \
    --point_cloud_size 1024 \
    --batch_size 230 \
    --epochs 1000 \
    --warmup_epochs 5 \
    --translate_augmentation 70 \
    --data_global_scale_factor 659.88367 \
    --lr 1e-4 \
    --depth 24 \
    --distributed \
