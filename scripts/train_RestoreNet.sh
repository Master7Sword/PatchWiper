#!/bin/bash

DEVICE_IDS="0,1,2,3" # e.g. "0" for single GPU, "0,1" for multi-GPU
DATASET="PRWD"  # options: PRWD, CLWD, ILAW

python3 ../train_RestoreNet.py \
  --epochs 100 \
  --lr 0.0003 \
  --train_batch_size 12 \
  --val_batch_size 32 \
  --device_ids $DEVICE_IDS \
  --dataset $DATASET \
  --dataset_dir "/data1/mozihao/Data/Dewatermarking/pixabay/pixabay_256" \
  --checkpoint_dir "../checkpoint" \
  --wln_ckpt "../checkpoint/WLN(PRWD).pth" \
  --num_workers 16 \
  --schedule 50 \

