#!/usr/bin/env python3
"""
MNIST Stream Verifier

This script verifies that a generated MNIST stream is correctly formatted,
continuous, and exhibits the expected distribution shift at t0.
"""

import numpy as np
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import sys

try:
    from safetensors.numpy import load_file
except ImportError:
    print("Error: safetensors not found.")
    print("Please install: pip install safetensors")
    sys.exit(1)


class StreamVerifier:
    """Verifies integrity and properties of a generated stream."""
    
    def __init__(self, stream_dir: str):
        """
        Initialize verifier with stream directory.
        
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
        
        print(f"Loaded manifest from {manifest_path}")
        print(f"  Dataset: {self.manifest['dataset']['name']} "
              f"({self.manifest['dataset']['split']})")
        print(f"  Samples: {self.manifest['dataset']['n_samples']}")
        print(f"  Chunks: {self.manifest['stream']['n_chunks']}")
        print(f"  t0: {self.manifest['distribution_shift']['t0']}")
    
    def load_chunk(self, chunk_idx: int) -> Dict[str, np.ndarray]:
        """Load a specific chunk by index."""
        chunk_file = self.manifest['stream']['chunk_files'][chunk_idx]
        chunk_path = self.stream_dir / chunk_file
        
        if not chunk_path.exists():
            raise FileNotFoundError(f"Chunk not found: {chunk_path}")
        
        return load_file(str(chunk_path))
    
    def verify_chunk_format(self, chunk_idx: int) -> bool:
        """Verify that a chunk has the correct format and types."""
        print(f"\n--- Verifying Chunk {chunk_idx} Format ---")
        
        chunk = self.load_chunk(chunk_idx)
        
        # Check required keys
        required_keys = ['x', 'y', 't', 'phase', 'intensity']
        for key in required_keys:
            if key not in chunk:
                print(f"  ✗ Missing required key: {key}")
                return False
        
        print(f"  ✓ All required keys present: {list(chunk.keys())}")
        
        # Check shapes
        n_samples = len(chunk['y'])
        expected_x_shape = tuple([n_samples] + self.manifest['format']['image_shape'])
        
        if chunk['x'].shape != expected_x_shape:
            print(f"  ✗ x shape mismatch: got {chunk['x'].shape}, "
                  f"expected {expected_x_shape}")
            return False
        
        print(f"  ✓ x shape: {chunk['x'].shape}")
        
        if chunk['y'].shape != (n_samples,):
            print(f"  ✗ y shape mismatch: got {chunk['y'].shape}, "
                  f"expected ({n_samples},)")
            return False
        
        print(f"  ✓ y shape: {chunk['y'].shape}")
        
        # Check dtypes
        if chunk['x'].dtype != np.float32:
            print(f"  ✗ x dtype: got {chunk['x'].dtype}, expected float32")
            return False
        
        if chunk['y'].dtype != np.int64:
            print(f"  ✗ y dtype: got {chunk['y'].dtype}, expected int64")
            return False
        
        print(f"  ✓ Dtypes correct (x: {chunk['x'].dtype}, y: {chunk['y'].dtype})")
        
        # Check value ranges
        x_min, x_max = chunk['x'].min(), chunk['x'].max()
        if x_min < -0.1 or x_max > 1.1:
            print(f"  ⚠ x values outside [0,1]: min={x_min:.3f}, max={x_max:.3f}")
        else:
            print(f"  ✓ x values in valid range: [{x_min:.3f}, {x_max:.3f}]")
        
        y_min, y_max = chunk['y'].min(), chunk['y'].max()
        if y_min < 0 or y_max > 9:
            print(f"  ✗ y values outside [0,9]: min={y_min}, max={y_max}")
            return False
        
        print(f"  ✓ y values in valid range: [{y_min}, {y_max}]")
        
        return True
    
    def verify_continuity(self) -> bool:
        """Verify that time indices are continuous across chunks."""
        print(f"\n--- Verifying Stream Continuity ---")
        
        n_chunks = self.manifest['stream']['n_chunks']
        chunk_size = self.manifest['stream']['chunk_size']
        
        expected_t = 0
        
        for chunk_idx in range(n_chunks):
            chunk = self.load_chunk(chunk_idx)
            t_values = chunk['t']
            
            # Check first t value
            if t_values[0] != expected_t:
                print(f"  ✗ Chunk {chunk_idx}: first t={t_values[0]}, "
                      f"expected {expected_t}")
                return False
            
            # Check sequential within chunk
            if not np.all(np.diff(t_values) == 1):
                print(f"  ✗ Chunk {chunk_idx}: t values not sequential")
                return False
            
            expected_t = t_values[-1] + 1
            
            if chunk_idx % 10 == 0:
                print(f"  ✓ Chunk {chunk_idx}: t=[{t_values[0]}, {t_values[-1]}]")
        
        print(f"  ✓ All {n_chunks} chunks are continuous")
        return True
    
    def verify_distribution_shift(self) -> bool:
        """Verify that distribution shift occurs at t0."""
        print(f"\n--- Verifying Distribution Shift ---")
        
        t0 = self.manifest['distribution_shift']['t0']
        chunk_size = self.manifest['stream']['chunk_size']
        
        # Find chunks around t0
        t0_chunk_idx = t0 // chunk_size
        
        # Check clean phase (before t0)
        if t0_chunk_idx > 0:
            chunk_before = self.load_chunk(max(0, t0_chunk_idx - 1))
            clean_intensities = chunk_before['intensity']
            clean_phases = chunk_before['phase']
            
            if np.any(clean_intensities > 0):
                print(f"  ✗ Found non-zero intensities before t0: "
                      f"max={clean_intensities.max():.4f}")
                return False
            
            if np.any(clean_phases != 0):
                print(f"  ✗ Found drift phase before t0")
                return False
            
            print(f"  ✓ Chunk {t0_chunk_idx-1} is clean (intensity=0)")
        
        # Check transition chunk (contains t0)
        chunk_transition = self.load_chunk(t0_chunk_idx)
        t_values = chunk_transition['t']
        intensities = chunk_transition['intensity']
        phases = chunk_transition['phase']
        
        # Find where t0 occurs in this chunk
        t0_pos = np.where(t_values >= t0)[0]
        
        if len(t0_pos) > 0:
            t0_local = t0_pos[0]
            
            # Before t0 in this chunk should be clean
            if t0_local > 0:
                if np.any(intensities[:t0_local] > 0):
                    print(f"  ✗ Found perturbation before t0 in transition chunk")
                    return False
            
            # At and after t0 should have intensity >= 0
            if not np.all(intensities[t0_local:] >= 0):
                print(f"  ✗ Negative intensities after t0")
                return False
            
            print(f"  ✓ Transition at chunk {t0_chunk_idx}, local position {t0_local}")
            print(f"    Before t0: intensity={intensities[max(0,t0_local-1)]:.4f}")
            print(f"    At t0: intensity={intensities[t0_local]:.4f}")
            print(f"    After t0: intensity={intensities[min(len(intensities)-1, t0_local+10)]:.4f}")
        
        # Check drift phase (after t0)
        if t0_chunk_idx < self.manifest['stream']['n_chunks'] - 1:
            chunk_after_idx = min(t0_chunk_idx + 2, 
                                  self.manifest['stream']['n_chunks'] - 1)
            chunk_after = self.load_chunk(chunk_after_idx)
            drift_intensities = chunk_after['intensity']
            drift_phases = chunk_after['phase']
            
            if np.all(drift_intensities == 0):
                print(f"  ✗ No perturbation found in chunk {chunk_after_idx} "
                      f"(after t0)")
                return False
            
            if not np.all(drift_phases == 1):
                print(f"  ⚠ Not all samples marked as drift in chunk {chunk_after_idx}")
            
            print(f"  ✓ Chunk {chunk_after_idx} has drift (mean intensity="
                  f"{drift_intensities.mean():.4f})")
        
        return True
    
    def compute_statistics(self) -> Dict:
        """Compute statistics about the stream."""
        print(f"\n--- Computing Stream Statistics ---")
        
        n_chunks = self.manifest['stream']['n_chunks']
        t0 = self.manifest['distribution_shift']['t0']
        
        # Sample chunks to compute stats
        sample_indices = [0, n_chunks // 4, n_chunks // 2, 
                         3 * n_chunks // 4, n_chunks - 1]
        
        stats = {
            'chunks': []
        }
        
        for chunk_idx in sample_indices:
            chunk = self.load_chunk(chunk_idx)
            
            chunk_stats = {
                'chunk_idx': chunk_idx,
                't_start': int(chunk['t'][0]),
                't_end': int(chunk['t'][-1]),
                'n_samples': len(chunk['y']),
                'x_mean': float(chunk['x'].mean()),
                'x_std': float(chunk['x'].std()),
                'x_min': float(chunk['x'].min()),
                'x_max': float(chunk['x'].max()),
                'intensity_mean': float(chunk['intensity'].mean()),
                'intensity_max': float(chunk['intensity'].max()),
                'phase_clean': int(np.sum(chunk['phase'] == 0)),
                'phase_drift': int(np.sum(chunk['phase'] == 1)),
                'label_distribution': {
                    str(i): int(np.sum(chunk['y'] == i)) 
                    for i in range(10)
                }
            }
            
            stats['chunks'].append(chunk_stats)
            
            print(f"\n  Chunk {chunk_idx} (t={chunk_stats['t_start']}-"
                  f"{chunk_stats['t_end']}):")
            print(f"    x: mean={chunk_stats['x_mean']:.3f}, "
                  f"std={chunk_stats['x_std']:.3f}")
            print(f"    intensity: mean={chunk_stats['intensity_mean']:.4f}, "
                  f"max={chunk_stats['intensity_max']:.4f}")
            print(f"    phase: clean={chunk_stats['phase_clean']}, "
                  f"drift={chunk_stats['phase_drift']}")
        
        return stats
    
    def run_full_verification(self) -> bool:
        """Run all verification checks."""
        print("\n" + "="*60)
        print("MNIST STREAM VERIFICATION")
        print("="*60)
        
        all_passed = True
        
        # Verify format of first and last chunks
        if not self.verify_chunk_format(0):
            all_passed = False
        
        if not self.verify_chunk_format(self.manifest['stream']['n_chunks'] - 1):
            all_passed = False
        
        # Verify continuity
        if not self.verify_continuity():
            all_passed = False
        
        # Verify distribution shift
        if not self.verify_distribution_shift():
            all_passed = False
        
        # Compute statistics
        stats = self.compute_statistics()
        
        print("\n" + "="*60)
        if all_passed:
            print("✓ ALL VERIFICATION CHECKS PASSED")
        else:
            print("✗ SOME VERIFICATION CHECKS FAILED")
        print("="*60)
        
        return all_passed


def main():
    parser = argparse.ArgumentParser(
        description="Verify MNIST stream integrity"
    )
    
    parser.add_argument(
        "stream_dir",
        type=str,
        help="Directory containing the stream (with manifest.json)"
    )
    
    args = parser.parse_args()
    
    verifier = StreamVerifier(args.stream_dir)
    success = verifier.run_full_verification()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
