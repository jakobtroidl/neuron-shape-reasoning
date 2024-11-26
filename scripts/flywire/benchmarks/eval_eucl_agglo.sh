bsub -J "AE Eval Euclidean" -n 4 -gpu "num=1" -q gpu_l4 -o logs/bold_color_paper_flywire_eucl_agglo_test.log python -m benchmarks.affinity.euclidean  \
    --data_path /nrs/turaga/jakob/implicit-neurons/flywire_full_v783/train \
    --types_path /nrs/turaga/jakob/implicit-neurons/flywire_full_v783/affinity/opticlobe_family/affinity_test_paper.csv \
    --output_dir /nrs/turaga/jakob/implicit-neurons/ckpt/ae/bold_color_paper_flywire_eucl_agglo_test \
    --fam_to_id_mapping /nrs/turaga/jakob/implicit-neurons/flywire_full_v783/types/visual_neurons_family_to_id.json \
    --point_cloud_size 1024 \
    --data_global_scale_factor 659.88367 \
    --clustering agglomerative \
    --threshold 0.3 \
    --store_tensors \
    --qual_results \