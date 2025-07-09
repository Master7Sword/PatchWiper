#!/bin/bash

DEVICE_IDS="0,1"  # e.g. "0" for single GPU, "0,1" for multi-GPU
DATASET="PRWD" # options: PRWD, CLWD, ILAW

python3 ../test.py \
  --dataset $DATASET \
  --dataset_dir "/data1/mozihao/Data/Dewatermarking/pixabay/pixabay_256" \
  --batch_size 32 \
  --device_ids $DEVICE_IDS \
  --ckpt_path "../checkpoint/PatchWiper(PRWD).pth" \
  --num_workers 8