# NOMA Test-Time Adaptation (TTA)

Test-Time Adaptation implementation in NOMA for handling distribution shift on MNIST streaming data.

## Overview

This experiment demonstrates **online adaptation** of a frozen model during deployment:

1. **Training Phase**: Train a 2-layer MLP on clean MNIST (60k samples)
2. **Deployment Phase**: Evaluate on a streaming dataset where:
   - First 30k samples: Clean MNIST (high accuracy expected)
   - Last 30k samples: Degraded MNIST with perturbations (accuracy drops)
3. **TTA Mechanism**: At drift detection (t=30000), inject a LoRA adapter and update it online

## Key Features

- **Frozen Backbone**: Original model weights are never modified during TTA
- **LoRA Injection**: Low-rank adapter injected on the last layer with neutral initialization
- **Online Learning**: Adapter parameters updated in real-time on streaming data
- **Backbone Integrity**: Checksums verify backbone remains unchanged
- **Control Experiment**: LoRA without updates proves gains come from learning

## Architecture

```
Input (784) ─────────────────────────────────────────────────────►
              │
              ▼
         ┌─────────────────┐
         │ W1 [784 x 128]  │  ◄── FROZEN
         │     + ReLU      │
         └────────┬────────┘
                  │
                  ▼
         ┌─────────────────┐
         │ W2 [128 x 10]   │  ◄── FROZEN
         └────────┬────────┘
                  │                ┌───────────────────────────────┐
                  │                │ LoRA Adapter (after t0)       │
                  │                │                               │
                  │                │ A [128 x 4] @ B [4 x 10]     │
                  │                │ ◄── LEARNABLE (online updates)│
                  ├────────────────┼───────────────────────────────►
                  ▼                                                 │
         ┌─────────────────────────────────────────────────────────┤
         │            Softmax(W2 @ h + LoRA @ h + b2)              │
         └─────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                              Output (10)
```

## Quick Start

```bash
cd /workspaces/NOMA/demo_TTA_MNIST

# Run complete pipeline (training + all evaluations)
./noma_TTA/run_pipeline.sh

# Or run individual steps:
# 1. Prepare data
python noma_TTA/prepare_data.py
python noma_TTA/prepare_stream.py

# 2. Train backbone
../target/release/noma run noma_TTA/train_mnist.noma

# 3. Run evaluations
../target/release/noma run noma_TTA/eval_stream_baseline.noma
../target/release/noma run noma_TTA/eval_stream_control.noma
../target/release/noma run noma_TTA/eval_stream_tta.noma

# 4. Post-process and compare
python noma_TTA/postprocess_results.py
```

## Experiments

Three evaluation runs on the **same stream** for fair comparison:

| Experiment | LoRA Injected | Online Updates | Expected Behavior |
|------------|---------------|----------------|-------------------|
| **Baseline** | No | No | Accuracy drops after drift, no recovery |
| **Control** | Yes (zero init) | No | Same as baseline (LoRA = 0) |
| **TTA** | Yes (zero init) | Yes | Accuracy recovers after drift |

## Expected Results

| Metric | Baseline | Control | TTA |
|--------|----------|---------|-----|
| Overall Accuracy | 76.58% | 76.58% | **89.21%** |
| Clean Phase Accuracy | 95.42% | 95.42% | 95.42% |
| Drift Phase Accuracy | 57.73% | 57.73% | **83.00%** |
| Accuracy Drop | 37.69% | 37.69% | 12.42% |
| **TTA Recovery** | - | - | **+25.27%** |

## Output Files

```
noma_TTA/output/
├── backbone_weights.safetensors      # Trained frozen backbone
├── eval_baseline_results.safetensors # Baseline predictions
├── eval_control_results.safetensors  # Control predictions  
├── eval_tta_results.safetensors      # TTA predictions + LoRA weights
├── tta_comparison.png                # Accuracy comparison plot
├── tta_metrics.csv                   # Per-sample metrics
└── tta_summary.json                  # Summary statistics
```

## LoRA Implementation Details

### Neutral Injection

LoRA is initialized to produce zero output at injection time:
- `A [128, 4]`: Small random initialization (scale 0.01)
- `B [4, 10]`: Zero initialization
- Result: `A @ B = 0`, so model output is unchanged

### Online Updates

During drift phase, only LoRA parameters are marked as `learn` and updated:
```noma
learn lora_A = rand_tensor(128.0, 4.0) * 0.01;  // Learnable
learn lora_B = rand_tensor(4.0, 10.0) * 0.0;    // Learnable

// W1, W2, b1, b2 are loaded (not learn) -> frozen
```

### Supervised Signal

For this demonstration, we use supervised loss (labels available in stream).
This models a "feedback online" scenario where the system receives occasional ground truth.

For fully unsupervised TTA, alternatives include:
- Entropy minimization
- Pseudo-labeling
- Confidence-based adaptation

## Verification

### Backbone Integrity Check

The TTA script computes checksums before and after adaptation:
```
Backbone checksum verification:
  Initial: 1234.567890
  Final:   1234.567890
  Difference (should be 0): 0.000000
```

### Control Experiment

The control run proves that performance gains come from learning, not architecture:
- LoRA is injected but never updated
- Since `A @ B = 0` at initialization and no updates occur, output = baseline
- Any improvement in TTA vs control is due to online learning

## NOMA Features Used

- `load_safetensors_named`: Load frozen checkpoint
- `save_safetensors`: Export predictions and adapted weights
- `learn`: Mark parameters for gradient tracking (LoRA only)
- `epoch batch`: Mini-batch online training
- `minimize`: Gradient update (only affects `learn` parameters)
- `matmul`, `relu`, `softmax`: Neural network layers
- Tensor operations: Broadcasting, reductions

## Comparison with Baseline

To compare with the static baseline:

```bash
# Run both pipelines
./noma_baseline/run_pipeline.sh
./noma_TTA/run_pipeline.sh

# Compare outputs
python compare_stream_plot.py
```

## References

- **LoRA**: [Hu et al., 2021 - LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- **Test-Time Adaptation**: [Wang et al., 2020 - Tent: Fully Test-Time Adaptation by Entropy Minimization](https://arxiv.org/abs/2006.10726)
