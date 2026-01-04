# NOMA Static Baseline

Static model baseline for MNIST Test-Time Adaptation (TTA) experiments, fully implemented in NOMA.

## Overview

Demonstrates performance degradation when a frozen model encounters distribution shift:
- **Clean phase (t < 30000)**: Original MNIST data → high accuracy (~95%)
- **Drift phase (t ≥ 30000)**: Degraded MNIST data → significant accuracy drop (~57%)

## Quick Start

```bash
cd /workspaces/NOMA/demo_TTA_MNIST

# Run complete pipeline
./noma_baseline/run_pipeline.sh
```

Or individual steps:

```bash
# Train
../target/release/noma run noma_baseline/train_mnist.noma

# Evaluate on stream
../target/release/noma run noma_baseline/eval_stream.noma

# Post-process & visualize
python noma_baseline/postprocess_results.py
```

## Model Architecture

Simple 2-layer MLP:
```
Input (784) → Dense(128) + ReLU → Dense(10) → Softmax → Output
```

## Output Files

- `output/model_weights.safetensors` - Trained weights
- `output/eval_results.safetensors` - Streaming evaluation predictions
- `output/noma_baseline_accuracy.png` - Rolling accuracy plot with t0 marker
- `output/noma_baseline_metrics.csv` - Per-sample predictions and accuracy
- `output/noma_baseline_summary.json` - Summary statistics

## Results Summary

| Metric | Value |
|--------|-------|
| Clean Phase Accuracy | 95.5% |
| Drift Phase Accuracy | 57.5% |
| Accuracy Drop | 38.0% |
| Total Samples | 60,000 |

## NOMA Features Used

- `softmax()` - Probability normalization
- `argmax()` - Class prediction
- `load_safetensors_named` - Checkpoint loading
- `matmul()` - Linear layer computation
- `relu()` - Activation function
- `save_safetensors` - Result export
- `EpochLoop` - Mini-batch training
- Broadcasting & tensor operations - Batch processing

## Scripts

- `train_mnist.noma` - Training script
- `eval_stream.noma` - Streaming evaluation script  
- `prepare_data.py` - Data preparation
- `prepare_stream.py` - Stream generation
- `postprocess_results.py` - Metrics & visualization
- `run_pipeline.sh` - Full pipeline automation
