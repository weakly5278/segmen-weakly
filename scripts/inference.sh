#!/usr/bin/env bash
set -e
python inference.py --csv data/test.csv --checkpoint checkpoints/best_lmcf_net.pth --output_dir outputs/masks
