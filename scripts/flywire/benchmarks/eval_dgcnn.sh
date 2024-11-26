bsub -J "AE Eval (DGCNN)" -n 12 -gpu "num=1" -q gpu_h100 -o logs/paper_flywire_dgcnn_test.log python -m benchmarks.dgcnn.eval \
    --dgcnn_pth /nrs/turaga/jakob/implicit-neurons/ckpt/ae/paper_flywire_dgcnn_train/ckpt/checkpoint-90.pth \
    --data_path /nrs/turaga/jakob/implicit-neurons/flywire_full_v783/train \
    --types_path /nrs/turaga/jakob/implicit-neurons/flywire_full_v783/affinity/ol_family_balanced/affinity_test.csv \
    --output_dir /nrs/turaga/jakob/implicit-neurons/ckpt/ae/paper_flywire_dgcnn_test \
    --train_emb_path /nrs/turaga/jakob/implicit-neurons/ckpt/ae/paper_flywire_dgcnn_train/emb_ep_90/ \
    --fam_to_id_mapping /nrs/turaga/jakob/implicit-neurons/flywire_full_v783/types/visual_neurons_family_to_id.json \
    --num_workers 1 \
    --point_cloud_size 1024 \
    --batch_size 500 \
    --data_global_scale_factor 659.88367 \
    # --k 5 \