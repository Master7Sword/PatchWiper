#!/bin/bash

DEVICE_IDS="0,1" # "0" for single GPU training
DATASET="PRWD"  # options: PRWD, CLWD, ILAW

python ../train_WLN.py \
  --epochs 100 \
  --lr 0.001 \
  --train_batch_size 64 \
  --val_batch_size 64 \
  --device_ids $DEVICE_IDS \
  --dataset $DATASET \
  --dataset_dir "/data1/mozihao/Data/Dewatermarking/PRWD"  \
  --checkpoint_dir "../checkpoint" \
  --lambda_iou 0.25 \
  --lambda_primary 0.01 \
  --gamma 0.5 \
  --num_workers 8 \
  --schedule 65
