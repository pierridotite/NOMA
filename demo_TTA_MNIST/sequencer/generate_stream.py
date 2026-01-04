#!/usr/bin/env python3
"""
MNIST Stream Generator with Distribution Shift

This script generates a sequential, reproducible MNIST stream with a controlled
distribution shift at a specified time point t0. The stream is exported as
chunked safetensors files for efficient streaming consumption.

Key features:
- Deterministic ordering with fixed seed
- Temporal index t for each sample
- Clean data before t0, degraded data after t0
- Progressive perturbation intensity following a schedule
- Export to safetensors chunks with JSON manifest
"""

import numpy as np
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys

try:
    from torchvision import datasets
    from safetensors.numpy import save_file
except ImportError:
    print("Error: Required packages not found.")
    print("Please install: pip install torch torchvision safetensors")
    sys.exit(1)


class PerturbationSchedule:
    """Defines how perturbation intensity evolves over time."""
    
    def __init__(self, t0: int, max_intensity: float, ramp_duration: int):
        """
        Args:
            t0: Time point where perturbation starts
            max_intensity: Maximum perturbation intensity (0.0 to 1.0)
            ramp_duration: Number of samples over which to ramp from 0 to max
        """
        self.t0 = t0
        self.max_intensity = max_intensity
        self.ramp_duration = ramp_duration
    
    def get_intensity(self, t: int) -> float:
        """Get perturbation intensity at time t."""
        if t < self.t0:
            return 0.0
        
        elapsed = t - self.t0
        if elapsed >= self.ramp_duration:
            return self.max_intensity
        
        # Linear ramp
        return self.max_intensity * (elapsed / self.ramp_duration)
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "t0": self.t0,
            "max_intensity": self.max_intensity,
            "ramp_duration": self.ramp_duration,
            "type": "linear_ramp"
        }


class ImagePerturbation:
    """Applies controlled perturbations to images."""
    
    def __init__(self, seed: int, perturbation_type: str = "gaussian_noise"):
        """
        Args:
            seed: Random seed for reproducible perturbations
            perturbation_type: Type of perturbation to apply
        """
        self.seed = seed
        self.perturbation_type = perturbation_type
        self.rng = np.random.RandomState(seed)
    
    def apply(self, image: np.ndarray, intensity: float) -> np.ndarray:
        """
        Apply perturbation with given intensity.
        
        Args:
            image: Input image array, values in [0, 1]
            intensity: Perturbation intensity in [0, 1]
        
        Returns:
            Perturbed image, clipped to [0, 1]
        """
        if intensity == 0.0:
            return image
        
        perturbed = image.copy()
        
        if self.perturbation_type == "gaussian_noise":
            # Add Gaussian noise with std proportional to intensity
            noise = self.rng.randn(*image.shape) * intensity * 0.5
            perturbed = image + noise
        
        elif self.perturbation_type == "brightness":
            # Decrease brightness progressively
            perturbed = image * (1.0 - intensity * 0.7)
        
        elif self.perturbation_type == "contrast":
            # Reduce contrast toward gray (0.5)
            perturbed = image * (1.0 - intensity) + 0.5 * intensity
        
        elif self.perturbation_type == "combined":
            # Combination of noise and brightness
            noise = self.rng.randn(*image.shape) * intensity * 0.3
            brightness_factor = 1.0 - intensity * 0.5
            perturbed = (image * brightness_factor) + noise
        
        else:
            raise ValueError(f"Unknown perturbation type: {self.perturbation_type}")
        
        # Clip to valid range
        return np.clip(perturbed, 0.0, 1.0)


class MNISTStreamGenerator:
    """Generates a sequential MNIST stream with distribution shift."""
    
    def __init__(
        self,
        seed: int = 42,
        use_train: bool = True,
        flatten: bool = False,
        shuffle: bool = False,
        normalize: bool = True,
        data_root: str = "./mnist_data"
    ):
        """
        Initialize the stream generator.
        
        Args:
            seed: Global random seed for reproducibility
            use_train: Use training set (60k) or test set (10k)
            flatten: Output shape [784] if True, else [1, 28, 28]
            shuffle: Shuffle the dataset order
            normalize: Normalize images to [0, 1]
            data_root: Root directory for MNIST data
        """
        self.seed = seed
        self.use_train = use_train
        self.flatten = flatten
        self.shuffle = shuffle
        self.normalize = normalize
        self.data_root = data_root
        
        # Set global seed
        np.random.seed(seed)
        
        # Load MNIST
        print(f"Loading MNIST {'train' if use_train else 'test'} set...")
        self.dataset = datasets.MNIST(
            root=data_root,
            train=use_train,
            download=True
        )
        
        self.n_samples = len(self.dataset)
        print(f"Loaded {self.n_samples} samples")
        
        # Create index ordering
        self.indices = np.arange(self.n_samples)
        if shuffle:
            rng = np.random.RandomState(seed)
            rng.shuffle(self.indices)
            print(f"Shuffled with seed {seed}")
        
        # Output shape
        if flatten:
            self.output_shape = (784,)
        else:
            self.output_shape = (1, 28, 28)
        
        print(f"Output shape: {self.output_shape}")
    
    def get_sample(self, t: int) -> Tuple[np.ndarray, int]:
        """
        Get sample at time index t.
        
        Args:
            t: Time index (0 to n_samples-1)
        
        Returns:
            image: Image array in specified format
            label: Integer label (0-9)
        """
        if t >= self.n_samples:
            raise IndexError(f"Time index {t} out of range [0, {self.n_samples})")
        
        idx = self.indices[t]
        image, label = self.dataset[idx]
        
        # Convert to numpy
        image = np.array(image, dtype=np.float32)
        
        # Normalize to [0, 1]
        if self.normalize:
            image = image / 255.0
        
        # Reshape
        if self.flatten:
            image = image.reshape(784)
        else:
            image = image.reshape(1, 28, 28)
        
        return image, int(label)
    
    def generate_stream(
        self,
        output_dir: str,
        t0: int,
        chunk_size: int = 1000,
        perturbation_type: str = "gaussian_noise",
        max_intensity: float = 0.8,
        ramp_duration: int = 5000
    ):
        """
        Generate complete stream with distribution shift.
        
        Args:
            output_dir: Output directory for chunks and manifest
            t0: Time point where distribution shift begins
            chunk_size: Number of samples per chunk file
            perturbation_type: Type of perturbation to apply
            max_intensity: Maximum perturbation intensity
            ramp_duration: Duration of intensity ramp
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n=== Generating MNIST Stream ===")
        print(f"Output directory: {output_path}")
        print(f"Total samples: {self.n_samples}")
        print(f"Chunk size: {chunk_size}")
        print(f"Distribution shift at t0={t0}")
        print(f"Perturbation: {perturbation_type}")
        print(f"Max intensity: {max_intensity}, Ramp: {ramp_duration}")
        
        # Initialize perturbation
        schedule = PerturbationSchedule(t0, max_intensity, ramp_duration)
        perturbation = ImagePerturbation(self.seed + 1, perturbation_type)
        
        # Generate chunks
        n_chunks = (self.n_samples + chunk_size - 1) // chunk_size
        chunk_files = []
        
        for chunk_idx in range(n_chunks):
            start_t = chunk_idx * chunk_size
            end_t = min(start_t + chunk_size, self.n_samples)
            actual_size = end_t - start_t
            
            # Allocate arrays for this chunk
            if self.flatten:
                x_chunk = np.zeros((actual_size, 784), dtype=np.float32)
            else:
                x_chunk = np.zeros((actual_size, 1, 28, 28), dtype=np.float32)
            
            y_chunk = np.zeros(actual_size, dtype=np.int64)
            t_chunk = np.arange(start_t, end_t, dtype=np.int64)
            phase_chunk = np.zeros(actual_size, dtype=np.int64)  # 0=clean, 1=drift
            intensity_chunk = np.zeros(actual_size, dtype=np.float32)
            
            # Generate samples
            for i, t in enumerate(range(start_t, end_t)):
                image, label = self.get_sample(t)
                
                # Apply perturbation
                intensity = schedule.get_intensity(t)
                if intensity > 0:
                    image = perturbation.apply(image, intensity)
                    phase_chunk[i] = 1
                
                x_chunk[i] = image
                y_chunk[i] = label
                intensity_chunk[i] = intensity
            
            # Save chunk
            chunk_filename = f"chunk_{chunk_idx:04d}.safetensors"
            chunk_path = output_path / chunk_filename
            
            save_file({
                "x": x_chunk,
                "y": y_chunk,
                "t": t_chunk,
                "phase": phase_chunk,
                "intensity": intensity_chunk
            }, str(chunk_path))
            
            chunk_files.append(chunk_filename)
            
            print(f"  Chunk {chunk_idx+1}/{n_chunks}: t={start_t}-{end_t-1} -> {chunk_filename}")
        
        # Create manifest
        manifest = {
            "version": "1.0",
            "description": "MNIST stream with distribution shift",
            "generation_date": str(np.datetime64('now')),
            "seed": self.seed,
            "dataset": {
                "name": "MNIST",
                "split": "train" if self.use_train else "test",
                "n_samples": int(self.n_samples),
                "shuffled": self.shuffle,
                "shuffle_seed": self.seed if self.shuffle else None
            },
            "format": {
                "image_shape": list(self.output_shape),
                "flattened": self.flatten,
                "normalized": self.normalize,
                "dtype": "float32",
                "label_dtype": "int64"
            },
            "stream": {
                "chunk_size": chunk_size,
                "n_chunks": len(chunk_files),
                "chunk_files": chunk_files
            },
            "distribution_shift": {
                "t0": t0,
                "perturbation_type": perturbation_type,
                "schedule": schedule.to_dict(),
                "perturbation_seed": self.seed + 1
            },
            "metadata": {
                "t_range": [0, self.n_samples - 1],
                "clean_samples": t0,
                "drift_samples": self.n_samples - t0
            }
        }
        
        manifest_path = output_path / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"\n✓ Stream generated successfully")
        print(f"  Manifest: {manifest_path}")
        print(f"  Chunks: {len(chunk_files)} files")
        print(f"  Clean samples: {t0}")
        print(f"  Drift samples: {self.n_samples - t0}")
        
        return manifest


def main():
    parser = argparse.ArgumentParser(
        description="Generate MNIST stream with distribution shift"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./mnist_stream",
        help="Output directory for stream files"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--use-test",
        action="store_true",
        help="Use test set instead of train set"
    )
    
    parser.add_argument(
        "--flatten",
        action="store_true",
        help="Flatten images to [784] instead of [1, 28, 28]"
    )
    
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle dataset order"
    )
    
    parser.add_argument(
        "--t0",
        type=int,
        default=30000,
        help="Time point where distribution shift begins"
    )
    
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Number of samples per chunk"
    )
    
    parser.add_argument(
        "--perturbation",
        type=str,
        default="gaussian_noise",
        choices=["gaussian_noise", "brightness", "contrast", "combined"],
        help="Type of perturbation to apply"
    )
    
    parser.add_argument(
        "--max-intensity",
        type=float,
        default=0.8,
        help="Maximum perturbation intensity (0.0 to 1.0)"
    )
    
    parser.add_argument(
        "--ramp-duration",
        type=int,
        default=5000,
        help="Number of samples for intensity ramp"
    )
    
    parser.add_argument(
        "--data-root",
        type=str,
        default="./mnist_data",
        help="Root directory for MNIST data"
    )
    
    args = parser.parse_args()
    
    # Create generator
    generator = MNISTStreamGenerator(
        seed=args.seed,
        use_train=not args.use_test,
        flatten=args.flatten,
        shuffle=args.shuffle,
        normalize=True,
        data_root=args.data_root
    )
    
    # Generate stream
    generator.generate_stream(
        output_dir=args.output_dir,
        t0=args.t0,
        chunk_size=args.chunk_size,
        perturbation_type=args.perturbation,
        max_intensity=args.max_intensity,
        ramp_duration=args.ramp_duration
    )


if __name__ == "__main__":
    main()
