#!/usr/bin/env python3
"""
Prepare Streaming Data for NOMA TTA Evaluation

This script:
1. Concatenates all stream chunks into a single safetensors file
2. Splits the stream into clean (t < t0) and drift (t >= t0) portions
   for separate processing in NOMA TTA

Output files:
- mnist_stream_full.safetensors: Complete stream for baseline/control
- stream_clean.safetensors: Clean phase (t < 30000) 
- stream_drift.safetensors: Drift phase (t >= 30000) for TTA adaptation
- stream_metadata.json: Metadata about the stream
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


def prepare_stream_for_noma_tta(
    stream_dir: str = "./sequencer/mnist_stream",
    output_dir: str = "./noma_TTA/data",
    t0: int = 30000
):
    """
    Prepare streaming data for NOMA TTA experiments.
    
    Creates three output files:
    - Full stream (for baseline/control evaluation)
    - Clean phase (t < t0, for clean-phase-only evaluation)
    - Drift phase (t >= t0, for TTA adaptation)
    
    Args:
        stream_dir: Directory containing the stream chunks and manifest
        output_dir: Output directory for the combined files
        t0: Distribution shift point
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
    n_samples = manifest['dataset']['n_samples']
    image_shape = manifest['format']['image_shape']
    
    print(f"Loading stream: {n_samples} samples in {n_chunks} chunks")
    print(f"Distribution shift at t0 = {t0}")
    
    # Pre-allocate arrays
    flat_size = 784
    all_x = np.zeros((n_samples, flat_size), dtype=np.float32)
    all_y = np.zeros(n_samples, dtype=np.float32)
    all_y_onehot = np.zeros((n_samples, 10), dtype=np.float32)
    all_t = np.zeros(n_samples, dtype=np.float32)
    all_phase = np.zeros(n_samples, dtype=np.float32)
    all_intensity = np.zeros(n_samples, dtype=np.float32)
    
    current_idx = 0
    
    for chunk_idx, chunk_file in enumerate(chunk_files):
        chunk_path = stream_path / chunk_file
        chunk = load_file(str(chunk_path))
        
        x = chunk['x']
        y = chunk['y']
        t = chunk['t']
        phase = chunk['phase']
        intensity = chunk['intensity']
        
        chunk_size = len(y)
        x_flat = x.reshape(chunk_size, -1)
        
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
    
    # =========================================================================
    # Save Full Stream
    # =========================================================================
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
    
    # =========================================================================
    # Split into Clean and Drift Phases for TTA
    # =========================================================================
    clean_mask = all_t < t0
    drift_mask = all_t >= t0
    
    n_clean = clean_mask.sum()
    n_drift = drift_mask.sum()
    
    print(f"\nSplitting stream:")
    print(f"  Clean samples (t < {t0}): {n_clean}")
    print(f"  Drift samples (t >= {t0}): {n_drift}")
    
    # Save clean phase
    clean_file = output_path / "stream_clean.safetensors"
    save_file({
        "x": all_x[clean_mask],
        "y": all_y[clean_mask],
        "y_onehot": all_y_onehot[clean_mask],
        "t": all_t[clean_mask],
        "phase": all_phase[clean_mask],
        "intensity": all_intensity[clean_mask]
    }, str(clean_file))
    print(f"Saved clean phase to: {clean_file}")
    
    # Save drift phase
    drift_file = output_path / "stream_drift.safetensors"
    save_file({
        "x": all_x[drift_mask],
        "y": all_y[drift_mask],
        "y_onehot": all_y_onehot[drift_mask],
        "t": all_t[drift_mask],
        "phase": all_phase[drift_mask],
        "intensity": all_intensity[drift_mask]
    }, str(drift_file))
    print(f"Saved drift phase to: {drift_file}")
    
    # =========================================================================
    # Save Metadata
    # =========================================================================
    meta = {
        "source": "mnist_stream",
        "n_samples": n_samples,
        "n_clean": int(n_clean),
        "n_drift": int(n_drift),
        "t0": t0,
        "image_shape": [784],
        "original_image_shape": image_shape,
        "perturbation_type": manifest['distribution_shift']['perturbation_type'],
        "max_intensity": manifest['distribution_shift']['schedule']['max_intensity'],
        "created": datetime.now().isoformat(),
        "files": {
            "full_stream": "mnist_stream_full.safetensors",
            "clean_phase": "stream_clean.safetensors",
            "drift_phase": "stream_drift.safetensors"
        }
    }
    
    meta_file = output_path / "stream_metadata.json"
    with open(meta_file, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata to: {meta_file}")
    
    # Print statistics
    print(f"\nStream statistics:")
    print(f"  X range: [{all_x.min():.3f}, {all_x.max():.3f}]")
    print(f"  Clean intensity range: [{all_intensity[clean_mask].min():.3f}, {all_intensity[clean_mask].max():.3f}]")
    print(f"  Drift intensity range: [{all_intensity[drift_mask].min():.3f}, {all_intensity[drift_mask].max():.3f}]")
    
    return all_x, all_y, all_t, all_phase


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare streaming data for NOMA TTA evaluation"
    )
    parser.add_argument("--stream-dir", type=str,
                        default="./sequencer/mnist_stream",
                        help="Directory containing stream chunks")
    parser.add_argument("--output-dir", type=str,
                        default="./noma_TTA/data",
                        help="Output directory for combined files")
    parser.add_argument("--t0", type=int, default=30000,
                        help="Distribution shift point")
    
    args = parser.parse_args()
    
    prepare_stream_for_noma_tta(
        stream_dir=args.stream_dir,
        output_dir=args.output_dir,
        t0=args.t0
    )
    
    print("\nStream preparation complete!")
