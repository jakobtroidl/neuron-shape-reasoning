bsub -J "AE Train (DGCNN)" -n 12 -gpu "num=1" -q gpu_h100 -o logs/paper_hemibrain_dgcnn_train.log python -m benchmarks.dgcnn.train \
    --data_path /nrs/turaga/jakob/implicit-neurons/hemibrain_v1.2/swc \
    --types_path /nrs/turaga/jakob/implicit-neurons/hemibrain_v1.2/metric/train.csv \
    --output_dir /nrs/turaga/jakob/implicit-neurons/ckpt/ae/paper_hemibrain_dgcnn_train \
    --fam_to_id_mapping /nrs/turaga/jakob/implicit-neurons/hemibrain_v1.2/types/family_to_id.json \
    --num_workers 1 \
    --point_cloud_size 1024 \
    --batch_size 200 \
    --data_global_scale_factor 659.88367 \
    --translate_augmentation 60 \