#!/usr/bin/env python3
"""
MNIST Stream Visualizer

This script provides visualization tools for analyzing a generated MNIST stream,
including sample images, intensity evolution, and distribution statistics.
"""

import numpy as np
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import sys

try:
    from safetensors.numpy import load_file
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
except ImportError:
    print("Error: Required packages not found.")
    print("Please install: pip install safetensors matplotlib")
    sys.exit(1)


class StreamVisualizer:
    """Visualizes a generated MNIST stream."""
    
    def __init__(self, stream_dir: str):
        """
        Initialize visualizer with stream directory.
        
        Args:
            stream_dir: Directory containing manifest.json and chunk files
        """
        self.stream_dir = Path(stream_dir)
        
        # Load manifest
        manifest_path = self.stream_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        
        with open(manifest_path, 'r') as f:
            self.manifest = json.load(f)
        
        print(f"Loaded stream: {self.manifest['dataset']['n_samples']} samples")
        print(f"  t0={self.manifest['distribution_shift']['t0']}")
        print(f"  Perturbation: {self.manifest['distribution_shift']['perturbation_type']}")
    
    def load_chunk(self, chunk_idx: int) -> Dict[str, np.ndarray]:
        """Load a specific chunk by index."""
        chunk_file = self.manifest['stream']['chunk_files'][chunk_idx]
        chunk_path = self.stream_dir / chunk_file
        return load_file(str(chunk_path))
    
    def get_sample_at_time(self, t: int) -> tuple:
        """
        Get a specific sample by time index.
        
        Returns:
            (image, label, intensity, phase)
        """
        chunk_size = self.manifest['stream']['chunk_size']
        chunk_idx = t // chunk_size
        local_idx = t % chunk_size
        
        chunk = self.load_chunk(chunk_idx)
        
        image = chunk['x'][local_idx]
        label = chunk['y'][local_idx]
        intensity = chunk['intensity'][local_idx]
        phase = chunk['phase'][local_idx]
        
        return image, label, intensity, phase
    
    def plot_sample_grid(
        self,
        times: List[int],
        output_file: Optional[str] = None,
        title: Optional[str] = None
    ):
        """
        Plot a grid of samples at specified time points.
        
        Args:
            times: List of time indices to visualize
            output_file: If provided, save to this file
            title: Optional title for the plot
        """
        n_samples = len(times)
        ncols = min(n_samples, 5)
        nrows = (n_samples + ncols - 1) // ncols
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(2.5*ncols, 3*nrows))
        if n_samples == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for i, t in enumerate(times):
            image, label, intensity, phase = self.get_sample_at_time(t)
            
            # Reshape if needed
            if len(image.shape) == 3:  # [1, 28, 28]
                image = image[0]
            elif len(image.shape) == 1:  # [784]
                image = image.reshape(28, 28)
            
            ax = axes[i]
            ax.imshow(image, cmap='gray', vmin=0, vmax=1)
            ax.axis('off')
            
            phase_str = "clean" if phase == 0 else "drift"
            ax.set_title(f"t={t}, label={label}\n{phase_str}, I={intensity:.3f}",
                        fontsize=9)
        
        # Hide unused subplots
        for i in range(n_samples, len(axes)):
            axes[i].axis('off')
        
        if title:
            fig.suptitle(title, fontsize=14, y=0.98)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"Saved to {output_file}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_intensity_evolution(
        self,
        output_file: Optional[str] = None,
        sample_rate: int = 10
    ):
        """
        Plot the evolution of perturbation intensity over time.
        
        Args:
            output_file: If provided, save to this file
            sample_rate: Sample every N-th chunk for efficiency
        """
        t0 = self.manifest['distribution_shift']['t0']
        n_chunks = self.manifest['stream']['n_chunks']
        
        # Collect intensity data
        times = []
        intensities = []
        
        for chunk_idx in range(0, n_chunks, sample_rate):
            chunk = self.load_chunk(chunk_idx)
            times.extend(chunk['t'])
            intensities.extend(chunk['intensity'])
        
        times = np.array(times)
        intensities = np.array(intensities)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 4))
        
        ax.plot(times, intensities, linewidth=0.5, alpha=0.7)
        ax.axvline(t0, color='red', linestyle='--', linewidth=2, 
                   label=f't0={t0}', alpha=0.7)
        
        ax.set_xlabel('Time Index (t)', fontsize=12)
        ax.set_ylabel('Perturbation Intensity', fontsize=12)
        ax.set_title('Distribution Shift: Perturbation Intensity Over Time', 
                     fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"Saved to {output_file}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_distribution_comparison(
        self,
        n_samples_per_phase: int = 1000,
        output_file: Optional[str] = None
    ):
        """
        Compare image statistics between clean and drift phases.
        
        Args:
            n_samples_per_phase: Number of samples to analyze per phase
            output_file: If provided, save to this file
        """
        t0 = self.manifest['distribution_shift']['t0']
        chunk_size = self.manifest['stream']['chunk_size']
        
        # Sample from clean phase (before t0)
        clean_start_t = max(0, t0 - n_samples_per_phase)
        clean_start_chunk = clean_start_t // chunk_size
        
        # Sample from drift phase (well after t0)
        drift_start_t = t0 + min(5000, 
                                (self.manifest['dataset']['n_samples'] - t0) // 4)
        drift_start_chunk = drift_start_t // chunk_size
        
        # Collect samples
        clean_images = []
        drift_images = []
        
        # Clean phase
        for t in range(clean_start_t, min(clean_start_t + n_samples_per_phase, t0)):
            img, _, _, _ = self.get_sample_at_time(t)
            if len(img.shape) == 3:
                img = img[0]
            elif len(img.shape) == 1:
                img = img.reshape(28, 28)
            clean_images.append(img.flatten())
        
        # Drift phase
        max_t = self.manifest['dataset']['n_samples']
        for t in range(drift_start_t, 
                      min(drift_start_t + n_samples_per_phase, max_t)):
            img, _, _, _ = self.get_sample_at_time(t)
            if len(img.shape) == 3:
                img = img[0]
            elif len(img.shape) == 1:
                img = img.reshape(28, 28)
            drift_images.append(img.flatten())
        
        clean_images = np.array(clean_images)
        drift_images = np.array(drift_images)
        
        # Create comparison plot
        fig = plt.figure(figsize=(14, 10))
        gs = gridspec.GridSpec(3, 2, hspace=0.3, wspace=0.3)
        
        # Pixel intensity distributions
        ax1 = fig.add_subplot(gs[0, :])
        ax1.hist(clean_images.flatten(), bins=50, alpha=0.5, label='Clean', 
                density=True, color='blue')
        ax1.hist(drift_images.flatten(), bins=50, alpha=0.5, label='Drift', 
                density=True, color='red')
        ax1.set_xlabel('Pixel Intensity')
        ax1.set_ylabel('Density')
        ax1.set_title('Pixel Intensity Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Mean images
        ax2 = fig.add_subplot(gs[1, 0])
        clean_mean = clean_images.mean(axis=0).reshape(28, 28)
        im2 = ax2.imshow(clean_mean, cmap='gray', vmin=0, vmax=1)
        ax2.set_title(f'Clean Phase Mean Image\n(n={len(clean_images)})')
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2, fraction=0.046)
        
        ax3 = fig.add_subplot(gs[1, 1])
        drift_mean = drift_images.mean(axis=0).reshape(28, 28)
        im3 = ax3.imshow(drift_mean, cmap='gray', vmin=0, vmax=1)
        ax3.set_title(f'Drift Phase Mean Image\n(n={len(drift_images)})')
        ax3.axis('off')
        plt.colorbar(im3, ax=ax3, fraction=0.046)
        
        # Std images
        ax4 = fig.add_subplot(gs[2, 0])
        clean_std = clean_images.std(axis=0).reshape(28, 28)
        im4 = ax4.imshow(clean_std, cmap='viridis')
        ax4.set_title('Clean Phase Std Dev')
        ax4.axis('off')
        plt.colorbar(im4, ax=ax4, fraction=0.046)
        
        ax5 = fig.add_subplot(gs[2, 1])
        drift_std = drift_images.std(axis=0).reshape(28, 28)
        im5 = ax5.imshow(drift_std, cmap='viridis')
        ax5.set_title('Drift Phase Std Dev')
        ax5.axis('off')
        plt.colorbar(im5, ax=ax5, fraction=0.046)
        
        fig.suptitle('Distribution Comparison: Clean vs Drift', 
                     fontsize=16, y=0.995)
        
        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"Saved to {output_file}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_transition_sequence(
        self,
        n_before: int = 5,
        n_after: int = 10,
        output_file: Optional[str] = None
    ):
        """
        Plot a sequence of samples around the transition point t0.
        
        Args:
            n_before: Number of samples before t0
            n_after: Number of samples after t0
            output_file: If provided, save to this file
        """
        t0 = self.manifest['distribution_shift']['t0']
        
        # Create time points around t0
        times = []
        
        # Before t0
        for i in range(n_before, 0, -1):
            t = max(0, t0 - i * 100)
            times.append(t)
        
        # At and after t0
        for i in range(n_after + 1):
            t = min(t0 + i * 500, self.manifest['dataset']['n_samples'] - 1)
            times.append(t)
        
        self.plot_sample_grid(
            times,
            output_file=output_file,
            title=f"Transition Sequence (t0={t0})"
        )
    
    def create_full_report(self, output_dir: Optional[str] = None):
        """
        Create a complete visualization report.
        
        Args:
            output_dir: Directory to save all plots
        """
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        else:
            output_path = self.stream_dir / "visualizations"
            output_path.mkdir(exist_ok=True)
        
        print(f"\nGenerating visualization report...")
        print(f"Output directory: {output_path}")
        
        # 1. Intensity evolution
        print("\n1. Plotting intensity evolution...")
        self.plot_intensity_evolution(
            output_file=str(output_path / "intensity_evolution.png")
        )
        
        # 2. Transition sequence
        print("\n2. Plotting transition sequence...")
        self.plot_transition_sequence(
            output_file=str(output_path / "transition_sequence.png")
        )
        
        # 3. Distribution comparison
        print("\n3. Plotting distribution comparison...")
        self.plot_distribution_comparison(
            output_file=str(output_path / "distribution_comparison.png")
        )
        
        # 4. Sample grids at different phases
        print("\n4. Plotting sample grids...")
        
        t0 = self.manifest['distribution_shift']['t0']
        n_total = self.manifest['dataset']['n_samples']
        
        # Early clean phase
        early_times = [i * 500 for i in range(10)]
        self.plot_sample_grid(
            early_times,
            output_file=str(output_path / "samples_early_clean.png"),
            title="Early Clean Phase Samples"
        )
        
        # Late drift phase
        late_start = t0 + (n_total - t0) * 2 // 3
        late_times = [late_start + i * 200 for i in range(10)]
        late_times = [min(t, n_total - 1) for t in late_times]
        self.plot_sample_grid(
            late_times,
            output_file=str(output_path / "samples_late_drift.png"),
            title="Late Drift Phase Samples"
        )
        
        print(f"\n✓ Report generated in {output_path}")
        print(f"  - intensity_evolution.png")
        print(f"  - transition_sequence.png")
        print(f"  - distribution_comparison.png")
        print(f"  - samples_early_clean.png")
        print(f"  - samples_late_drift.png")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize MNIST stream"
    )
    
    parser.add_argument(
        "stream_dir",
        type=str,
        help="Directory containing the stream (with manifest.json)"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        default="report",
        choices=["report", "intensity", "transition", "comparison", "samples"],
        help="Visualization mode"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for plots (default: stream_dir/visualizations)"
    )
    
    parser.add_argument(
        "--times",
        type=int,
        nargs="+",
        help="Specific time indices to visualize (for samples mode)"
    )
    
    args = parser.parse_args()
    
    visualizer = StreamVisualizer(args.stream_dir)
    
    if args.mode == "report":
        visualizer.create_full_report(args.output_dir)
    
    elif args.mode == "intensity":
        output_file = None
        if args.output_dir:
            output_file = str(Path(args.output_dir) / "intensity_evolution.png")
        visualizer.plot_intensity_evolution(output_file)
    
    elif args.mode == "transition":
        output_file = None
        if args.output_dir:
            output_file = str(Path(args.output_dir) / "transition_sequence.png")
        visualizer.plot_transition_sequence(output_file=output_file)
    
    elif args.mode == "comparison":
        output_file = None
        if args.output_dir:
            output_file = str(Path(args.output_dir) / "distribution_comparison.png")
        visualizer.plot_distribution_comparison(output_file=output_file)
    
    elif args.mode == "samples":
        if not args.times:
            # Default: show some samples
            t0 = visualizer.manifest['distribution_shift']['t0']
            args.times = [0, t0 - 1000, t0, t0 + 1000, t0 + 5000]
        
        output_file = None
        if args.output_dir:
            output_file = str(Path(args.output_dir) / "samples.png")
        visualizer.plot_sample_grid(args.times, output_file=output_file)


if __name__ == "__main__":
    main()
