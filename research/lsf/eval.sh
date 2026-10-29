#!/bin/bash
# Submit an evaluation run on the Janelia LSF cluster.
# Usage: research/lsf/eval.sh research/configs/flywire.yaml MODEL [extra pat eval args ...]
CONFIG=$1; MODEL=$2; shift 2
NAME=$(basename "$CONFIG" .yaml)
mkdir -p logs
bsub -J "PAT eval ($NAME)" -n 8 -gpu "num=1" -q gpu_h100 -o "logs/eval_${NAME}.log" \
    pat eval "$CONFIG" --model "$MODEL" --thresholds 0.5,0.6,0.7,0.8,0.9 "$@"
