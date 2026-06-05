#!/bin/bash

set -e
set -x

CUDA_VISIBLE_DEVICES=0 python train.py \
    --dataset_name 'cub' \
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
    --warmup_teacher_temp_epochs 20 \
    --memax_weight 2 \
    --exp_name cub_gt_bbox \
    --dataset_dir '/lustre1/g/stat_han/datasets/cub/CUB_200_2011' \
    --mask_dir '/home/mkyahx/MOS/userhome/cs/mkyahx/TokenCut/datasets/CUB/masks' \
    --pretrain_path 'pretrain_weight/dino_vitbase16_pretrain.pth'

