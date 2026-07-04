#!/usr/bin/env bash
set -e

python test.py \
  --test_csv data/test.csv \
  --checkpoint checkpoints/best_lmcf_net.pth \
  --batch_size 4 \
  --image_size 1024 \
  --output_dir outputs \
  --save_outputs
