bsub -J "AE Eval (DeepSet)" -n 12 -gpu "num=1" -q gpu_h100 -o logs/paper_hemibrain_deepset_test.log python eval_contrastive.py \
    --model ae_d1024_m512 \
    --encoder_pth /nrs/turaga/jakob/implicit-neurons/ckpt/ae/paper_hemibrain_ours_train_v1/ckpt/checkpoint-946.pth \
    --deep_set_pth /nrs/turaga/jakob/implicit-neurons/ckpt/ae/paper_hemibrain_deepset_train_normed/ckpt/checkpoint-90.pth \
    --data_path /nrs/turaga/jakob/implicit-neurons/hemibrain_v1.2/swc \
    --types_path /nrs/turaga/jakob/implicit-neurons/hemibrain_v1.2/metric/test.csv \
    --output_dir /nrs/turaga/jakob/implicit-neurons/ckpt/ae/paper_hemibrain_deepset_test \
    --train_emb_path /nrs/turaga/jakob/implicit-neurons/ckpt/ae/paper_hemibrain_deepset_train_normed/emb_ep_90/ \
    --fam_to_id_mapping /nrs/turaga/jakob/implicit-neurons/hemibrain_v1.2/types/family_to_id.json \
    --translate_augmentation 60 \
    --num_workers 1 \
    --point_cloud_size 1024 \
    --batch_size 650 \
    --data_global_scale_factor 659.88367 \
    --depth 24 \
    --norm_emb