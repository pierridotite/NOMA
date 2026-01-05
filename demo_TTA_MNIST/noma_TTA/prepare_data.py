#!/usr/bin/env python3
"""
Prepare Clean MNIST Data for NOMA TTA Training

This script exports clean MNIST training data to safetensors format.
Identical to noma_baseline/prepare_data.py but outputs to noma_TTA/data/.
"""

import numpy as np
import json
import argparse
from pathlib import Path
from datetime import datetime
import sys

try:
    from torchvision import datasets
    from safetensors.numpy import save_file
except ImportError:
    print("Error: Required packages not found.")
    print("Please install: pip install torch torchvision safetensors")
    sys.exit(1)


def prepare_mnist_for_noma(
    data_root: str = "./sequencer/mnist_data",
    output_dir: str = "./noma_TTA/data",
    use_train: bool = True,
    max_samples: int = None,
    seed: int = 42
):
    """
    Prepare MNIST data for NOMA training.
    
    Args:
        data_root: Root directory for MNIST data
        output_dir: Output directory for safetensors files
        use_train: Use training set (True) or test set (False)
        max_samples: Maximum number of samples (None = all)
        seed: Random seed for shuffling
    """
    np.random.seed(seed)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading MNIST {'train' if use_train else 'test'} set...")
    dataset = datasets.MNIST(
        root=data_root,
        train=use_train,
        download=True
    )
    
    n_samples = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    print(f"Preparing {n_samples} samples...")
    
    # Extract and normalize data
    images = []
    labels = []
    
    for i in range(n_samples):
        img, label = dataset[i]
        # Convert PIL to numpy, normalize to [0, 1], flatten to [784]
        img_np = np.array(img, dtype=np.float32) / 255.0
        img_flat = img_np.flatten()
        images.append(img_flat)
        labels.append(label)
    
    X = np.stack(images, axis=0).astype(np.float32)  # [N, 784]
    y = np.array(labels, dtype=np.float32)            # [N]
    
    # Create one-hot encoding
    y_onehot = np.zeros((n_samples, 10), dtype=np.float32)
    y_onehot[np.arange(n_samples), labels] = 1.0
    
    print(f"X shape: {X.shape}, range: [{X.min():.3f}, {X.max():.3f}]")
    print(f"y shape: {y.shape}, classes: {np.unique(y)}")
    print(f"y_onehot shape: {y_onehot.shape}")
    
    # Save to safetensors
    data_file = output_path / "mnist_train.safetensors"
    save_file({
        "x": X,
        "y": y,
        "y_onehot": y_onehot
    }, str(data_file))
    print(f"Saved training data to: {data_file}")
    
    # Save metadata
    meta = {
        "source": "mnist",
        "split": "train" if use_train else "test",
        "n_samples": n_samples,
        "image_shape": [784],
        "n_classes": 10,
        "created": datetime.now().isoformat()
    }
    
    meta_file = output_path / "metadata.json"
    with open(meta_file, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata to: {meta_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare MNIST data for NOMA TTA training"
    )
    parser.add_argument("--data-root", type=str,
                        default="./sequencer/mnist_data",
                        help="Root directory for MNIST data")
    parser.add_argument("--output-dir", type=str,
                        default="./noma_TTA/data",
                        help="Output directory for safetensors files")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum number of samples (default: all 60k)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    prepare_mnist_for_noma(
        data_root=args.data_root,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        seed=args.seed
    )
    
    print("\nData preparation complete!")
