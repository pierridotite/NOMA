# Python Static Baseline

PyTorch implementation of the static model baseline for MNIST Test-Time Adaptation (TTA) experiments.

## Overview

Reference baseline demonstrating performance degradation with distribution shift:
- **Clean phase (t < 30000)**: Original MNIST data → high accuracy (~98%)
- **Drift phase (t ≥ 30000)**: Degraded MNIST data → accuracy drop to ~60%

## Quick Start

```bash
cd /workspaces/NOMA/demo_TTA_MNIST/python_baseline

pip install -r requirements.txt
python static_model_baseline.py
```

## Model Architecture

Simple 2-layer MLP:
```
Input (784) → Dense(256) + ReLU + Dropout(0.2)
           → Dense(128) + ReLU + Dropout(0.2)
           → Dense(10) → Output
```

## Output Files

- `output/static_baseline_accuracy.png` - Rolling accuracy plot with t0 marker
- `output/static_baseline_metrics.csv` - Per-sample predictions and accuracy
- `output/static_baseline_summary.json` - Summary statistics

## Results Summary

| Metric | Value |
|--------|-------|
| Clean Phase Accuracy | 98.5% |
| Drift Phase Accuracy | 60.0% |
| Accuracy Drop | 38.5% |
| Total Samples | 60,000 |

## Usage Options

```bash
python static_model_baseline.py \
  --stream-dir ../sequencer/mnist_stream \
  --epochs 10 \
  --batch-size 64 \
  --lr 1e-3 \
  --window-size 500 \
  --seed 42
```

## Scripts

- `static_model_baseline.py` - Main training and evaluation script

## Dependencies

- torch, torchvision
- numpy, matplotlib
- safetensors
- pathlib, csv
