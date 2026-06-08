#!/bin/bash

set -e
set -x

USE_SEED=${USE_SEED:-0}
TRAIN_SEED=${TRAIN_SEED:-0}
SEED_ARGS=()

if [ "$USE_SEED" = "1" ]; then
    export PYTHONHASHSEED="$TRAIN_SEED"
    export CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG:-:4096:8}
    SEED_ARGS=(--seed "$TRAIN_SEED" --deterministic)
fi

CUDA_VISIBLE_DEVICES=0 python train.py \
    --dataset_name 'scars' \
    --batch_size 128 \
    --grad_from_block 11 \
    --epochs 200 \
    --num_workers 8 \
    --use_ssb_splits \
    --sup_weight 0.35 \
    --weight_decay 5e-5 \
    --transform 'imagenet' \
    --lr 0.1 \
    --eval_funcs 'v2' \
    --warmup_teacher_temp 0.07 \
    --teacher_temp 0.04 \
    --warmup_teacher_temp_epochs 30 \
    --memax_weight 1 \
    --exp_name scars_simgcd \
    --dataset_dir 'stanford_car' \
    --mask_dir 'stanford_car' \
    --pretrain_path 'pretrain_weight/dino_vitbase16_pretrain.pth' \
    "${SEED_ARGS[@]}"
