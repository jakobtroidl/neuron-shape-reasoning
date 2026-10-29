#!/bin/bash
# Submit a multi-GPU training run on the Janelia LSF cluster.
# Usage: research/lsf/train.sh research/configs/flywire.yaml [--set key=value ...]
CONFIG=$1; shift
NAME=$(basename "$CONFIG" .yaml)
mkdir -p logs
bsub -J "PAT train ($NAME)" -n 96 -gpu "num=8" -q gpu_h100 -o "logs/train_${NAME}.log" \
    torchrun --standalone --nproc_per_node 8 -m pat train "$CONFIG" "$@"
