#!/usr/bin/env python3
"""
Prepare Streaming Data for NOMA Evaluation

This script concatenates all stream chunks into a single safetensors file
that NOMA can process with batch loops.

The output format:
- x: [N, 784], float32, normalized images (flattened from [1,28,28])
- y: [N], float32, labels (0-9)
- t: [N], float32, time indices
- phase: [N], float32, 0=clean, 1=drift
- intensity: [N], float32, perturbation intensity

This enables NOMA to iterate over the entire stream using batch processing.
"""

import numpy as np
import json
import argparse
from pathlib import Path
from datetime import datetime
import sys

try:
    from safetensors.numpy import load_file, save_file
except ImportError:
    print("Error: safetensors not found.")
    print("Please install: pip install safetensors")
    sys.exit(1)


def prepare_stream_for_noma(
    stream_dir: str = "./sequencer/mnist_stream",
    output_dir: str = "./noma_baseline/data"
):
    """
    Concatenate all stream chunks into a single file for NOMA.
    
    Args:
        stream_dir: Directory containing the stream chunks and manifest
        output_dir: Output directory for the combined file
    """
    stream_path = Path(stream_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load manifest
    manifest_path = stream_path / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    n_chunks = manifest['stream']['n_chunks']
    chunk_files = manifest['stream']['chunk_files']
    t0 = manifest['distribution_shift']['t0']
    n_samples = manifest['dataset']['n_samples']
    image_shape = manifest['format']['image_shape']
    
    print(f"Loading stream: {n_samples} samples in {n_chunks} chunks")
    print(f"Image shape in stream: {image_shape}")
    print(f"Distribution shift at t0 = {t0}")
    
    # Pre-allocate arrays
    # Flatten images from [1, 28, 28] to [784]
    flat_size = 784
    all_x = np.zeros((n_samples, flat_size), dtype=np.float32)
    all_y = np.zeros(n_samples, dtype=np.float32)
    all_t = np.zeros(n_samples, dtype=np.float32)
    all_phase = np.zeros(n_samples, dtype=np.float32)
    all_intensity = np.zeros(n_samples, dtype=np.float32)
    
    # Also create one-hot encoded labels for compatibility with training
    all_y_onehot = np.zeros((n_samples, 10), dtype=np.float32)
    
    current_idx = 0
    
    for chunk_idx, chunk_file in enumerate(chunk_files):
        chunk_path = stream_path / chunk_file
        chunk = load_file(str(chunk_path))
        
        # Get data from chunk
        x = chunk['x']  # Shape: [chunk_size, 1, 28, 28]
        y = chunk['y']  # Shape: [chunk_size]
        t = chunk['t']  # Shape: [chunk_size]
        phase = chunk['phase']  # Shape: [chunk_size]
        intensity = chunk['intensity']  # Shape: [chunk_size]
        
        chunk_size = len(y)
        
        # Flatten images and copy to pre-allocated arrays
        x_flat = x.reshape(chunk_size, -1)  # [chunk_size, 784]
        
        all_x[current_idx:current_idx + chunk_size] = x_flat.astype(np.float32)
        all_y[current_idx:current_idx + chunk_size] = y.astype(np.float32)
        all_t[current_idx:current_idx + chunk_size] = t.astype(np.float32)
        all_phase[current_idx:current_idx + chunk_size] = phase.astype(np.float32)
        all_intensity[current_idx:current_idx + chunk_size] = intensity.astype(np.float32)
        
        # Create one-hot encoding
        for i in range(chunk_size):
            all_y_onehot[current_idx + i, int(y[i])] = 1.0
        
        current_idx += chunk_size
        
        if (chunk_idx + 1) % 10 == 0:
            print(f"  Processed {chunk_idx + 1}/{n_chunks} chunks...")
    
    print(f"Total samples processed: {current_idx}")
    
    # Save combined stream
    output_file = output_path / "mnist_stream_full.safetensors"
    save_file({
        "x": all_x,
        "y": all_y,
        "y_onehot": all_y_onehot,
        "t": all_t,
        "phase": all_phase,
        "intensity": all_intensity
    }, str(output_file))
    print(f"Saved full stream to: {output_file}")
    
    # Also save metadata
    meta = {
        "source": "mnist_stream",
        "n_samples": n_samples,
        "t0": t0,
        "image_shape": [784],
        "original_image_shape": image_shape,
        "perturbation_type": manifest['distribution_shift']['perturbation_type'],
        "max_intensity": manifest['distribution_shift']['schedule']['max_intensity'],
        "created": datetime.now().isoformat()
    }
    
    meta_file = output_path / "stream_metadata.json"
    with open(meta_file, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata to: {meta_file}")
    
    # Print statistics
    clean_mask = all_phase == 0
    drift_mask = all_phase == 1
    
    print(f"\nStream statistics:")
    print(f"  Clean samples: {clean_mask.sum()} (t < {t0})")
    print(f"  Drift samples: {drift_mask.sum()} (t >= {t0})")
    print(f"  X range: [{all_x.min():.3f}, {all_x.max():.3f}]")
    print(f"  Labels distribution: {np.bincount(all_y.astype(int))}")
    
    return all_x, all_y, all_t, all_phase


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare streaming data for NOMA evaluation"
    )
    parser.add_argument("--stream-dir", type=str,
                        default="./sequencer/mnist_stream",
                        help="Directory containing stream chunks")
    parser.add_argument("--output-dir", type=str,
                        default="./noma_baseline/data",
                        help="Output directory for combined file")
    
    args = parser.parse_args()
    
    prepare_stream_for_noma(
        stream_dir=args.stream_dir,
        output_dir=args.output_dir
    )
    
    print("\nStream preparation complete!")
