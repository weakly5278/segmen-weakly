#!/usr/bin/env bash
set -e

python train.py \
  --train_csv data/train.csv \
  --val_csv data/val.csv \
  --batch_size 4 \
  --epochs 100 \
  --image_size 1024 \
  --output_dir outputs \
  --checkpoint_dir checkpoints
