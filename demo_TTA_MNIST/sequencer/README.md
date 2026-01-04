# MNIST Stream Generator

Generates reproducible MNIST data streams with controlled distribution shift for Test-Time Adaptation experiments.

## Overview

Creates sequential MNIST stream with:
- **Clean phase (t < 30000)**: Original MNIST data
- **Drift phase (t ≥ 30000)**: Progressively degraded images
- **Output**: Safetensors chunks + metadata for NOMA/Python pipelines

## Quick Start

```bash
python generate_stream.py --output-dir ./mnist_stream
```

## Scripts

- `generate_stream.py` - Main stream generator
- `verify_stream.py` - Validates stream integrity
- `visualize_stream.py` - Generates diagnostic plots

## Output Format

Exported as `mnist_stream_full.safetensors` containing:
- `x` - [60000, 784] images (float32, normalized [0, 1])
- `y` - [60000] true labels (int32, 0-9)
- `t` - [60000] time indices
- `phase` - [60000] phase labels (0=clean, 1=drift)
- `intensity` - [60000] perturbation intensity [0, 0.8]

## Perturbation Types

- Gaussian blur: increasing blur radius
- Brightness: random adjustments
- Contrast: random modulation
- Noise: Gaussian and impulse noise

## Configuration

Default parameters:
- Total samples: 60,000
- t0 (shift point): 30,000
- Transition duration: ~10,000 samples
- Intensity range: [0, 0.8]

## Dependencies

- torch, torchvision
- safetensors, numpy
- matplotlib, scipy
