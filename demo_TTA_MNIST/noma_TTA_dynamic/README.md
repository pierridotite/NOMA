# Dynamic TTA with NOMA

This demo showcases NOMA's unique capability for Test-Time Adaptation: 
dynamic allocation and growth of adaptation parameters at runtime using `alloc` and `realloc`.

## Difference with Standard TTA

### Standard TTA (static)

- LoRA adapter allocated with fixed size before deployment
- Size chosen based on assumptions about drift severity
- Too small: cannot adapt to severe drift
- Too large: wastes memory, risk of overfitting

### Dynamic TTA with NOMA

- Adapter allocated only when drift is detected (`alloc`)
- Capacity grows based on adaptation performance (`realloc`)
- Optimizer state preserved across reallocation
- Memory freed when drift subsides (`free`)

## NOMA Features Used

| Feature | Usage |
|---------|-------|
| `alloc` | Create adapter when drift detected |
| `realloc` | Grow adapter capacity while preserving weights and optimizer state |
| `learn` | Mark adapter parameters as trainable |
| `free` | Release adapter when no longer needed |

## Protocol

1. **Clean phase (t < 30000)**: Pure backbone inference, no adapter
2. **Drift detection (t = 30000)**: Allocate minimal LoRA (rank=2)
3. **Adaptive growth**: If loss > threshold, realloc to larger rank
4. **Stabilization**: Optionally free adapter when confidence recovers

## Files

```
noma_TTA_dynamic/
├── train_backbone.noma      # Train initial model
├── eval_dynamic.noma        # Dynamic TTA with alloc/realloc
├── eval_static.noma         # Static TTA for comparison
├── eval_baseline.noma       # No adaptation baseline
├── prepare_data.py          # Data preparation
├── postprocess_results.py   # Analysis
├── run_pipeline.sh          # Full pipeline
├── data/                    # Generated data
└── output/                  # Results
```

## Usage

```bash
./run_pipeline.sh
```

Or step by step:

```bash
python prepare_data.py
../../target/release/noma run train_backbone.noma
../../target/release/noma run eval_baseline.noma
../../target/release/noma run eval_static.noma
../../target/release/noma run eval_dynamic.noma
python postprocess_results.py
```

## Key Code Example

```noma
// Allocate minimal adapter when drift detected
alloc lora_A = [128, 2];
alloc lora_B = [2, 10];

// If adaptation insufficient, grow capacity
// Preserves existing weights AND optimizer state
realloc lora_A = [128, 4];
realloc lora_B = [4, 10];
```

This is impossible in standard frameworks without rebuilding the model and losing optimizer momentum.
