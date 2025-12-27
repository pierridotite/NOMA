# Self-Growing XOR Benchmark

A reproducible benchmark comparing NOMA against standard implementations for the XOR classification task with dynamic network growth.

## Overview

This benchmark evaluates the performance of a self-growing neural network that:

1. Starts with 2 hidden neurons
2. Trains on XOR until a fixed step (growth trigger)
3. Expands to 16 hidden neurons
4. Continues training until convergence (loss < 0.002)

All implementations share identical initial weights and hyperparameters to ensure fair comparison.

## Implementations

| Name | Description | Gradient Computation |
|------|-------------|---------------------|
| `noma_compiled` | NOMA compiled to native binary | Automatic (compile-time) |
| `noma_interpreted` | NOMA via interpreter | Automatic (runtime) |
| `numpy_manual` | NumPy with manual gradients | Manual backprop |
| `torch_eager` | PyTorch eager mode | Automatic (runtime) |
| `torch_compile` | PyTorch with torch.compile | Automatic (JIT) |
| `cpp_manual` | C++ with manual gradients | Manual backprop |

## Requirements

### System

- Linux (tested on Ubuntu 24.04)
- Rust toolchain (for NOMA)
- Python 3.10+
- g++ with C++17 support

### Python Dependencies

```bash
pip install numpy matplotlib
pip install torch  # for PyTorch baselines
```

## Usage

### Quick Start

Run the complete benchmark suite:

```bash
./scripts/run_all.sh
```

This will:
1. Generate shared initial weights
2. Compile NOMA and C++ binaries
3. Run all implementations (N=10 times each)
4. Aggregate results
5. Generate plots and summary

### Output Files

All outputs are written to `out/`:

```
out/
  init/
    init_weights.json     # Shared initial weights
  run_001/
    noma_compiled/
      loss.csv            # step,loss,accuracy,hidden,phase
      timings.json        # Detailed timing metrics
      stdout.txt          # Raw output
    ...
  summary.json            # Aggregated results (median, p25, p75)
  comparison.png          # Bar charts
  loss.png                # Training curves
  loc.json                # Lines of code count
```

### Individual Scripts

```bash
# Clean outputs (preserves init/)
./scripts/clean_out.sh

# Count lines of code
./scripts/count_loc.sh
```

## Methodology

### Hyperparameters (config.json)

| Parameter | Value |
|-----------|-------|
| Learning rate (phase 1) | 0.05 |
| Learning rate (phase 2) | 0.12 |
| Growth trigger step | 200 |
| Initial hidden neurons | 2 |
| Final hidden neurons | 16 |
| Convergence threshold | 0.002 |
| Benchmark runs | 10 |
| Warmup steps (timing) | 50 |

### Weight Initialization

All implementations load weights from `init_weights.json`:

- `W1`: Input to hidden weights (2x2 initial, extended to 2x16 at growth)
- `b1`: Hidden biases (2 initial, extended to 16)
- `W2`: Hidden to output weights (2x1 initial, extended to 16x1)
- `b2`: Output bias (1)

### Timing Metrics

Each implementation reports:

- `cold_start_ms`: Initial setup time
- `compile_overhead_ms`: JIT/compile time (if applicable)
- `train_warmup_ms`: First 50 steps (excluded from steady-state)
- `train_steady_ms`: Steps 50 to growth
- `steady_step_us_median`: Median step time (microseconds)
- `steady_step_us_p95`: 95th percentile step time
- `growth_event_ms`: Time for network expansion
- `total_ms`: Total execution time

### Normalized Output Format

All implementations produce `loss.csv`:

```csv
step,loss,accuracy,hidden,phase
0,0.693147,0.5000,2,warmup
50,0.421300,0.7500,2,steady
...
200,0.012345,1.0000,16,growth
...
```

## Results

Run the benchmark to generate `comparison.png` and `summary.json`.

### Execution Time (N=10 runs, median)

| Implementation | Time (ms) | Speedup vs NumPy | Final Loss |
|----------------|-----------|------------------|------------|
| noma_compiled | 0.70 | **42.3x** | 0.001978 |
| cpp_manual | 0.80 | 37.0x | 0.001977 |
| numpy_manual | 29.71 | 1.0x | 0.001977 |
| noma_interpreted | 96.44 | 0.3x | 0.001987 |

### Compile Times

| Language | Compile Time (ms) |
|----------|-------------------|
| NOMA | 212 |
| C++ | 1424 |

### Performance Comparison

![Comparison](assets/comparison.png)

### Training Convergence

![Loss Curves](assets/loss.png) 

**Note**: C++ and NumPy curves overlap perfectly (identical algorithms and initial weights). C++ is shown with a dashed line to make it visible.

## File Structure

```
demo_self_growing_xor/
  config.json              # Shared hyperparameters
  noma/
    xor.noma               # NOMA implementation
  baselines/
    python_numpy_manual/
      xor.py               # NumPy baseline
    python_torch/
      xor_eager.py         # PyTorch eager
      xor_compile.py       # PyTorch compile
    cpp_manual/
      xor.cpp              # C++ baseline
      build.sh
  python/
    init_gen.py            # Weight initialization
    benchmark.py           # Benchmark orchestrator
    plot_loss.py           # Loss curve visualization
    summarize.py           # Results aggregation
  scripts/
    run_all.sh             # Master script
    clean_out.sh           # Clean outputs
    count_loc.sh           # LOC counter
  out/                     # Generated outputs
```

## Notes

- XOR is a toy problem used for methodology validation, not real-world performance claims.
- All implementations use float64 precision.
- Adam optimizer with default parameters (beta1=0.9, beta2=0.999, epsilon=1e-8).
- Growth is triggered at a fixed step, not by loss threshold (for reproducibility).

## License

MIT
