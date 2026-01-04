#!/usr/bin/env python3
"""
Post-Processing for NOMA Baseline Evaluation Results

This script:
1. Loads NOMA evaluation results (predictions, labels, timing)
2. Computes rolling accuracy over time
3. Generates visualization with t0 marker
4. Exports metrics to CSV
5. Compares with Python baseline if available
"""

import numpy as np
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import csv
from datetime import datetime
import sys

try:
    from safetensors.numpy import load_file
except ImportError:
    print("Error: safetensors not found.")
    print("Please install: pip install safetensors")
    sys.exit(1)


def load_noma_results(results_path: str) -> dict:
    """Load NOMA evaluation results from safetensors."""
    data = load_file(results_path)
    
    # Get predicted classes from prediction probabilities (argmax)
    pred_probs = data['pred_probs'].astype(np.float32)
    pred = np.argmax(pred_probs, axis=1).astype(np.int32)
    
    return {
        'pred': pred,
        'y_true': data['y_true'].astype(np.int32),
        't': data['t'].astype(np.int32),
        'phase': data['phase'],  # Keep as string
        'intensity': data['intensity'].astype(np.float32)
    }


def compute_rolling_accuracy(pred: np.ndarray, y_true: np.ndarray, 
                              window_size: int = 500) -> np.ndarray:
    """
    Compute rolling accuracy over time.
    
    Args:
        pred: Predicted labels
        y_true: True labels
        window_size: Size of rolling window
    
    Returns:
        Rolling accuracy array (same length as inputs)
    """
    correct = (pred == y_true).astype(np.float32)
    
    # Compute rolling mean
    rolling_acc = np.zeros(len(correct))
    buffer = []
    
    for i, c in enumerate(correct):
        buffer.append(c)
        if len(buffer) > window_size:
            buffer.pop(0)
        rolling_acc[i] = 100.0 * np.mean(buffer)
    
    return rolling_acc


def generate_results(results_path: str, output_dir: str, 
                     window_size: int = 500, t0: int = 30000):
    """
    Generate all results: metrics, plot, comparison.
    
    Args:
        results_path: Path to NOMA results safetensors
        output_dir: Output directory for files
        window_size: Rolling window size
        t0: Distribution shift point
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading NOMA results from: {results_path}")
    results = load_noma_results(results_path)
    
    pred = results['pred']
    y_true = results['y_true']
    t = results['t']
    phase = results['phase']
    intensity = results['intensity']
    
    n_samples = len(pred)
    print(f"Loaded {n_samples} samples")
    
    # Compute correctness and rolling accuracy
    correct = (pred == y_true).astype(np.int32)
    rolling_acc = compute_rolling_accuracy(pred, y_true, window_size)
    
    # Compute phase-wise statistics
    clean_mask = phase == 0
    drift_mask = phase == 1
    
    clean_correct = correct[clean_mask]
    drift_correct = correct[drift_mask]
    
    clean_acc = 100.0 * clean_correct.mean() if len(clean_correct) > 0 else 0
    drift_acc = 100.0 * drift_correct.mean() if len(drift_correct) > 0 else 0
    overall_acc = 100.0 * correct.mean()
    
    # Compute accuracy drop (last 1000 samples of each phase)
    clean_end_acc = 100.0 * clean_correct[-1000:].mean() if len(clean_correct) >= 1000 else clean_acc
    drift_end_acc = 100.0 * drift_correct[-1000:].mean() if len(drift_correct) >= 1000 else drift_acc
    accuracy_drop = clean_end_acc - drift_end_acc
    
    print(f"\n{'='*60}")
    print("NOMA Baseline Evaluation Summary")
    print(f"{'='*60}")
    print(f"Total samples: {n_samples}")
    print(f"Window size: {window_size}")
    print(f"Distribution shift at t0 = {t0}")
    print(f"")
    print(f"Overall accuracy: {overall_acc:.2f}%")
    print(f"Clean phase accuracy: {clean_acc:.2f}%")
    print(f"Drift phase accuracy: {drift_acc:.2f}%")
    print(f"Accuracy drop: {accuracy_drop:.2f}%")
    print(f"{'='*60}\n")
    
    # =========================================================================
    # Generate Visualization
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                    gridspec_kw={'height_ratios': [3, 1]})
    
    # Main plot: Rolling accuracy
    ax1.plot(t, rolling_acc, 'b-', linewidth=1.0, alpha=0.8, label='Rolling Accuracy')
    
    # Add t0 marker
    ax1.axvline(x=t0, color='red', linestyle='--', linewidth=2, label=f't0 = {t0}')
    
    # Shade regions
    ax1.axvspan(0, t0, alpha=0.1, color='green', label='Clean Phase')
    ax1.axvspan(t0, max(t), alpha=0.1, color='red', label='Drift Phase')
    
    # Labels and formatting
    ax1.set_ylabel('Rolling Accuracy (%)', fontsize=12)
    ax1.set_ylim(0, 105)
    ax1.set_xlim(0, max(t))
    ax1.legend(loc='lower left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f'NOMA Static Baseline - Streaming Evaluation\n'
                  f'Window={window_size}, '
                  f'Clean Acc={clean_acc:.1f}%, '
                  f'Drift Acc={drift_acc:.1f}%',
                  fontsize=14)
    
    # Add horizontal lines at key accuracy levels
    ax1.axhline(y=clean_acc, color='green', linestyle=':', alpha=0.5, linewidth=1)
    ax1.axhline(y=drift_acc, color='red', linestyle=':', alpha=0.5, linewidth=1)
    
    # Secondary plot: Perturbation intensity
    ax2.fill_between(t, intensity, alpha=0.5, color='orange', label='Perturbation Intensity')
    ax2.axvline(x=t0, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('Time (t)', fontsize=12)
    ax2.set_ylabel('Intensity', fontsize=12)
    ax2.set_ylim(0, 1.0)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left', fontsize=10)
    
    plt.tight_layout()
    
    plot_path = output_path / 'noma_baseline_accuracy.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {plot_path}")
    
    # =========================================================================
    # Export Metrics CSV
    # =========================================================================
    csv_path = output_path / 'noma_baseline_metrics.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['t', 'correct', 'predicted', 'actual', 'phase', 
                        'intensity', 'rolling_accuracy'])
        
        phase_names = {0: 'clean', 1: 'drift'}
        for i in range(n_samples):
            writer.writerow([
                int(t[i]),
                int(correct[i]),
                int(pred[i]),
                int(y_true[i]),
                phase_names.get(int(phase[i]), 'unknown'),
                float(intensity[i]),
                float(rolling_acc[i])
            ])
    print(f"Metrics exported to: {csv_path}")
    
    # =========================================================================
    # Export Summary JSON
    # =========================================================================
    summary = {
        'experiment': 'noma_static_baseline',
        'timestamp': datetime.now().isoformat(),
        'evaluation': {
            't0': int(t0),
            'n_samples': n_samples,
            'window_size': window_size,
            'overall_accuracy': float(overall_acc),
            'clean_accuracy': float(clean_acc),
            'drift_accuracy': float(drift_acc),
            'accuracy_drop': float(accuracy_drop)
        }
    }
    
    summary_path = output_path / 'noma_baseline_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary exported to: {summary_path}")
    
    return summary


def compare_with_python_baseline(noma_summary_path: str, python_summary_path: str):
    """
    Compare NOMA and Python baseline results.
    """
    with open(noma_summary_path, 'r') as f:
        noma = json.load(f)
    
    with open(python_summary_path, 'r') as f:
        python = json.load(f)
    
    print(f"\n{'='*60}")
    print("Comparison: NOMA vs Python Baseline")
    print(f"{'='*60}")
    print(f"{'Metric':<25} {'NOMA':>12} {'Python':>12} {'Diff':>12}")
    print(f"{'-'*60}")
    
    noma_eval = noma['evaluation']
    python_eval = python['evaluation']
    
    metrics = [
        ('Clean Accuracy', 'clean_accuracy'),
        ('Drift Accuracy', 'drift_accuracy'),
        ('Accuracy Drop', 'accuracy_drop'),
        ('Overall Accuracy', 'overall_accuracy')
    ]
    
    for name, key in metrics:
        noma_val = noma_eval.get(key, 0)
        python_val = python_eval.get(key, 0)
        diff = noma_val - python_val
        print(f"{name:<25} {noma_val:>11.2f}% {python_val:>11.2f}% {diff:>+11.2f}%")
    
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Post-process NOMA baseline evaluation results"
    )
    parser.add_argument("--results", type=str,
                        default="./noma_baseline/output/eval_results.safetensors",
                        help="Path to NOMA evaluation results")
    parser.add_argument("--output-dir", type=str,
                        default="./noma_baseline/output",
                        help="Output directory for generated files")
    parser.add_argument("--window-size", type=int, default=500,
                        help="Rolling window size for accuracy")
    parser.add_argument("--t0", type=int, default=30000,
                        help="Distribution shift point")
    parser.add_argument("--compare-python", type=str, default=None,
                        help="Path to Python baseline summary for comparison")
    
    args = parser.parse_args()
    
    summary = generate_results(
        results_path=args.results,
        output_dir=args.output_dir,
        window_size=args.window_size,
        t0=args.t0
    )
    
    # Compare with Python baseline if available
    if args.compare_python and Path(args.compare_python).exists():
        noma_summary = Path(args.output_dir) / 'noma_baseline_summary.json'
        compare_with_python_baseline(str(noma_summary), args.compare_python)
    elif Path("./baseline/output/static_baseline_summary.json").exists():
        noma_summary = Path(args.output_dir) / 'noma_baseline_summary.json'
        compare_with_python_baseline(str(noma_summary), "./baseline/output/static_baseline_summary.json")
    
    print("\nPost-processing complete!")
