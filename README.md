# LMCF-Net

Official PyTorch implementation of **LMCF-Net: Learnable Multi-scale Discrepancy and Cross-attention Fusion Network** for weakly supervised lesion localization in longitudinal mammograms.

LMCF-Net uses paired prior and current mammograms with image-level labels during training. Pixel-level masks are not required for optimization and are used only for localization evaluation when available.

## Repository structure

```text
LMCF-Net/
├── main.py                    # training and testing entry point
├── inference.py               # mask generation from trained checkpoints
├── config.py                  # default experiment configuration
├── datasets/                  # paired mammogram dataset and transforms
├── models/                    # encoder, LMDB, TCFM, and prediction heads
├── losses/                    # dynamic weak localization objective
├── postprocessing/            # BLM refinement
├── trainer/                   # training, evaluation, metrics
├── utils/                     # logging, checkpoints, visualization
├── scripts/                   # command-line examples
└── data/                      # CSV split files
```

## Input format

Each split is defined by a CSV file with the following columns:

```csv
prior_path,current_path,label,mask_path,patient_id,view
/path/to/prior.png,/path/to/current.png,1,/path/to/mask.png,P001,LCC
```

Required columns:

- `prior_path`: path to the prior mammogram
- `current_path`: path to the current mammogram
- `label`: image-level label (`0` for normal, `1` for cancer)

Optional columns:

- `mask_path`: binary lesion annotation used only for evaluation
- `patient_id`: case identifier used for saved outputs
- `view`: mammographic view identifier

## Installation

```bash
pip install -r requirements.txt
```

## Training

```bash
python main.py \
  --mode train \
  --train_csv data/train.csv \
  --val_csv data/val.csv \
  --batch_size 4 \
  --epochs 100 \
  --image_size 1024
```

The best validation checkpoint is saved to:

```text
checkpoints/best_lmcf_net.pth
```

## Testing

```bash
python main.py \
  --mode test \
  --test_csv data/test.csv \
  --checkpoint checkpoints/best_lmcf_net.pth \
  --output_dir outputs
```

## Inference

```bash
python inference.py \
  --csv data/test.csv \
  --checkpoint checkpoints/best_lmcf_net.pth \
  --output_dir outputs/masks
```

The inference script saves both continuous score maps and refined Binary Localization Masks (BLMs).

## Model overview

LMCF-Net contains four main components:

1. **Shared hierarchical transformer encoder** for prior-current feature extraction.
2. **Learnable Multi-scale Discrepancy Block (LMDB)** for stage-wise temporal discrepancy modeling.
3. **Temporal Cross-Attention Fusion Module (TCFM)** for semantic-guided multi-scale discrepancy fusion.
4. **Dynamic weak localization objective and BLM refinement** for spatially coherent localization.
