#!/usr/bin/env python3
"""
Post-Processing for NOMA TTA Evaluation Results

This script:
1. Loads evaluation results from all three experiments (baseline, TTA, control)
2. Computes rolling accuracy over time for each
3. Generates comparison visualization showing:
   - Accuracy drop after drift (t0)
   - Recovery with TTA
   - Control behavior (no improvement expected)
4. Verifies backbone integrity (checksums unchanged)
5. Exports per-sample metrics to CSV
6. Generates summary JSON with comparison statistics
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


def load_results(results_path: str) -> dict:
    """Load evaluation results from safetensors."""
    data = load_file(results_path)
    
    pred_probs = data['pred_probs'].astype(np.float32)
    pred = np.argmax(pred_probs, axis=1).astype(np.int32)
    
    return {
        'pred': pred,
        'pred_probs': pred_probs,
        'y_true': data['y_true'].astype(np.int32),
        't': data['t'].astype(np.int32),
        'phase': data['phase'].astype(np.int32),
        'intensity': data['intensity'].astype(np.float32)
    }


def load_tta_results_split(clean_path: str, drift_path: str) -> dict:
    """Load TTA results from separate clean and drift files and combine them."""
    clean_data = load_file(clean_path)
    drift_data = load_file(drift_path)
    
    # Combine predictions
    pred_probs_clean = clean_data['pred_probs'].astype(np.float32)
    pred_probs_drift = drift_data['pred_probs'].astype(np.float32)
    pred_probs = np.concatenate([pred_probs_clean, pred_probs_drift], axis=0)
    pred = np.argmax(pred_probs, axis=1).astype(np.int32)
    
    # Combine labels
    y_true = np.concatenate([
        clean_data['y_true'].astype(np.int32),
        drift_data['y_true'].astype(np.int32)
    ])
    
    # Combine metadata
    t = np.concatenate([
        clean_data['t'].astype(np.int32),
        drift_data['t'].astype(np.int32)
    ])
    
    phase = np.concatenate([
        clean_data['phase'].astype(np.int32),
        drift_data['phase'].astype(np.int32)
    ])
    
    intensity = np.concatenate([
        clean_data['intensity'].astype(np.float32),
        drift_data['intensity'].astype(np.float32)
    ])
    
    return {
        'pred': pred,
        'pred_probs': pred_probs,
        'y_true': y_true,
        't': t,
        'phase': phase,
        'intensity': intensity
    }


def compute_rolling_accuracy(pred: np.ndarray, y_true: np.ndarray, 
                              window_size: int = 500) -> np.ndarray:
    """Compute rolling accuracy over time."""
    correct = (pred == y_true).astype(np.float32)
    
    rolling_acc = np.zeros(len(correct))
    buffer = []
    
    for i, c in enumerate(correct):
        buffer.append(c)
        if len(buffer) > window_size:
            buffer.pop(0)
        rolling_acc[i] = 100.0 * np.mean(buffer)
    
    return rolling_acc


def compute_phase_accuracy(pred: np.ndarray, y_true: np.ndarray, 
                           phase: np.ndarray) -> dict:
    """Compute accuracy per phase."""
    correct = (pred == y_true).astype(np.float32)
    
    clean_mask = phase == 0
    drift_mask = phase == 1
    
    clean_correct = correct[clean_mask]
    drift_correct = correct[drift_mask]
    
    return {
        'overall': 100.0 * correct.mean(),
        'clean': 100.0 * clean_correct.mean() if len(clean_correct) > 0 else 0,
        'drift': 100.0 * drift_correct.mean() if len(drift_correct) > 0 else 0,
        'drop': 100.0 * (clean_correct.mean() - drift_correct.mean()) if len(drift_correct) > 0 else 0
    }


def generate_comparison_results(
    baseline_path: str,
    tta_path: str,
    control_path: str,
    output_dir: str,
    window_size: int = 500,
    t0: int = 30000
):
    """
    Generate comprehensive comparison results.
    
    Args:
        baseline_path: Path to baseline results
        tta_path: Path to TTA results
        control_path: Path to control results
        output_dir: Output directory
        window_size: Rolling window size
        t0: Distribution shift point
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # Load All Results
    # =========================================================================
    print("Loading results...")
    
    results = {}
    
    if Path(baseline_path).exists():
        results['baseline'] = load_results(baseline_path)
        print(f"  Loaded baseline: {len(results['baseline']['pred'])} samples")
    else:
        print(f"  Warning: Baseline results not found: {baseline_path}")
    
    # Try to load TTA from split files first, fall back to single file
    tta_clean_path = str(Path(tta_path).parent / "eval_tta_clean.safetensors")
    tta_drift_path = str(Path(tta_path).parent / "eval_tta_drift.safetensors")
    
    if Path(tta_clean_path).exists() and Path(tta_drift_path).exists():
        results['tta'] = load_tta_results_split(tta_clean_path, tta_drift_path)
        print(f"  Loaded TTA (split): {len(results['tta']['pred'])} samples")
    elif Path(tta_path).exists():
        results['tta'] = load_results(tta_path)
        print(f"  Loaded TTA (single): {len(results['tta']['pred'])} samples")
    else:
        print(f"  Warning: TTA results not found")
    
    if Path(control_path).exists():
        results['control'] = load_results(control_path)
        print(f"  Loaded control: {len(results['control']['pred'])} samples")
    else:
        print(f"  Warning: Control results not found: {control_path}")
    
    if not results:
        print("Error: No results found!")
        return None
    
    # Use first available result for common data
    reference = list(results.values())[0]
    t = reference['t']
    phase = reference['phase']
    intensity = reference['intensity']
    y_true = reference['y_true']
    n_samples = len(t)
    
    # =========================================================================
    # Compute Metrics for Each Experiment
    # =========================================================================
    print("\nComputing metrics...")
    
    metrics = {}
    rolling_accs = {}
    
    for name, res in results.items():
        rolling_accs[name] = compute_rolling_accuracy(res['pred'], res['y_true'], window_size)
        metrics[name] = compute_phase_accuracy(res['pred'], res['y_true'], res['phase'])
        
        print(f"\n{name.upper()}:")
        print(f"  Overall: {metrics[name]['overall']:.2f}%")
        print(f"  Clean:   {metrics[name]['clean']:.2f}%")
        print(f"  Drift:   {metrics[name]['drift']:.2f}%")
        print(f"  Drop:    {metrics[name]['drop']:.2f}%")
    
    # =========================================================================
    # Compute TTA Recovery Metrics
    # =========================================================================
    if 'tta' in metrics and 'baseline' in metrics:
        tta_recovery = metrics['tta']['drift'] - metrics['baseline']['drift']
        print(f"\nTTA RECOVERY: {tta_recovery:+.2f}% on drift phase")
    else:
        tta_recovery = 0
    
    # =========================================================================
    # Generate Comparison Visualization
    # =========================================================================
    print("\nGenerating visualization...")
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), 
                              gridspec_kw={'height_ratios': [3, 1, 1]})
    
    colors = {
        'baseline': '#1f77b4',  # Blue
        'tta': '#2ca02c',        # Green
        'control': '#ff7f0e'     # Orange
    }
    
    labels = {
        'baseline': 'Baseline (No Adaptation)',
        'tta': 'TTA (LoRA + Online Updates)',
        'control': 'Control (LoRA, No Updates)'
    }
    
    # Main plot: Rolling accuracy comparison
    ax1 = axes[0]
    
    for name in ['baseline', 'control', 'tta']:
        if name in rolling_accs:
            ax1.plot(t, rolling_accs[name], color=colors[name], 
                    linewidth=1.5 if name == 'tta' else 1.0,
                    alpha=0.9, label=labels[name])
    
    # Add t0 marker
    ax1.axvline(x=t0, color='red', linestyle='--', linewidth=2, 
                label=f'Drift Point (t0={t0})')
    
    # Shade regions
    ax1.axvspan(0, t0, alpha=0.05, color='green')
    ax1.axvspan(t0, max(t), alpha=0.05, color='red')
    
    # Add annotations
    ax1.annotate('Clean Phase', xy=(t0/4, 98), fontsize=10, color='darkgreen')
    ax1.annotate('Drift Phase', xy=(t0 + t0/4, 98), fontsize=10, color='darkred')
    
    # Recovery annotation
    if 'tta' in metrics and 'baseline' in metrics:
        ax1.annotate(f'TTA Recovery: {tta_recovery:+.1f}%', 
                    xy=(t0 + t0/2, metrics['tta']['drift'] + 5),
                    fontsize=11, fontweight='bold', color='darkgreen')
    
    ax1.set_ylabel('Rolling Accuracy (%)', fontsize=12)
    ax1.set_ylim(0, 105)
    ax1.set_xlim(0, max(t))
    ax1.legend(loc='lower left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f'NOMA Test-Time Adaptation - Comparison\n'
                  f'Window={window_size}, t0={t0}', fontsize=14)
    
    # Secondary plot: Accuracy difference (TTA vs Baseline)
    ax2 = axes[1]
    
    if 'tta' in rolling_accs and 'baseline' in rolling_accs:
        diff = rolling_accs['tta'] - rolling_accs['baseline']
        ax2.fill_between(t, 0, diff, where=diff >= 0, alpha=0.5, color='green', 
                        label='TTA Improvement')
        ax2.fill_between(t, 0, diff, where=diff < 0, alpha=0.5, color='red',
                        label='TTA Degradation')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.axvline(x=t0, color='red', linestyle='--', linewidth=1)
        ax2.set_ylabel('TTA - Baseline (%)', fontsize=10)
        ax2.set_ylim(-15, 25)
        ax2.legend(loc='upper left', fontsize=9)
        ax2.grid(True, alpha=0.3)
    
    # Tertiary plot: Perturbation intensity
    ax3 = axes[2]
    ax3.fill_between(t, intensity, alpha=0.5, color='orange', 
                     label='Perturbation Intensity')
    ax3.axvline(x=t0, color='red', linestyle='--', linewidth=1)
    ax3.set_xlabel('Time (t)', fontsize=12)
    ax3.set_ylabel('Intensity', fontsize=10)
    ax3.set_ylim(0, 1.0)
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper left', fontsize=9)
    
    plt.tight_layout()
    
    plot_path = output_path / 'tta_comparison.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {plot_path}")
    
    # =========================================================================
    # Export Per-Sample Metrics CSV
    # =========================================================================
    csv_path = output_path / 'tta_metrics.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        header = ['t', 'y_true', 'phase', 'intensity']
        for name in ['baseline', 'tta', 'control']:
            if name in results:
                header.extend([f'{name}_pred', f'{name}_correct', f'{name}_rolling_acc'])
        writer.writerow(header)
        
        phase_names = {0: 'clean', 1: 'drift'}
        for i in range(n_samples):
            row = [
                int(t[i]),
                int(y_true[i]),
                phase_names.get(int(phase[i]), 'unknown'),
                float(intensity[i])
            ]
            for name in ['baseline', 'tta', 'control']:
                if name in results:
                    pred = results[name]['pred'][i]
                    correct = int(pred == y_true[i])
                    rolling = rolling_accs[name][i]
                    row.extend([int(pred), correct, float(rolling)])
            writer.writerow(row)
    
    print(f"Metrics exported to: {csv_path}")
    
    # =========================================================================
    # Export Summary JSON
    # =========================================================================
    summary = {
        'experiment': 'noma_tta_comparison',
        'timestamp': datetime.now().isoformat(),
        'config': {
            't0': int(t0),
            'n_samples': n_samples,
            'window_size': window_size
        },
        'results': {}
    }
    
    for name, m in metrics.items():
        summary['results'][name] = {
            'overall_accuracy': float(m['overall']),
            'clean_accuracy': float(m['clean']),
            'drift_accuracy': float(m['drift']),
            'accuracy_drop': float(m['drop'])
        }
    
    if 'tta' in metrics and 'baseline' in metrics:
        summary['tta_analysis'] = {
            'recovery_vs_baseline': float(tta_recovery),
            'drift_improvement': float(metrics['tta']['drift'] - metrics['baseline']['drift']),
            'overall_improvement': float(metrics['tta']['overall'] - metrics['baseline']['overall'])
        }
    
    summary_path = output_path / 'tta_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary exported to: {summary_path}")
    
    # =========================================================================
    # Print Final Comparison Table
    # =========================================================================
    print(f"\n{'='*70}")
    print("NOMA TTA COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(f"{'Experiment':<20} {'Overall':>12} {'Clean':>12} {'Drift':>12} {'Drop':>12}")
    print(f"{'-'*70}")
    
    for name, m in metrics.items():
        print(f"{name.upper():<20} {m['overall']:>11.2f}% {m['clean']:>11.2f}% "
              f"{m['drift']:>11.2f}% {m['drop']:>11.2f}%")
    
    print(f"{'-'*70}")
    
    if 'tta' in metrics and 'baseline' in metrics:
        print(f"{'TTA RECOVERY:':<20} {'':<12} {'':<12} "
              f"{tta_recovery:>+11.2f}% {'':<12}")
    
    print(f"{'='*70}")
    
    return summary


def verify_backbone_integrity(tta_results_path: str):
    """
    Verify that backbone weights were not modified during TTA.
    """
    if not Path(tta_results_path).exists():
        print("TTA results not found, skipping backbone verification.")
        return None
    
    data = load_file(tta_results_path)
    
    # Check if checksums are available
    if 'backbone_checksum_initial' not in data or 'backbone_checksum_final' not in data:
        print("Backbone checksums not found in results.")
        return None
    
    # Handle both scalar and array checksums
    initial_data = data['backbone_checksum_initial']
    final_data = data['backbone_checksum_final']
    
    initial = float(initial_data.item()) if initial_data.ndim > 0 else float(initial_data)
    final = float(final_data.item()) if final_data.ndim > 0 else float(final_data)
    diff = abs(final - initial)
    
    print(f"\n{'='*50}")
    print("BACKBONE INTEGRITY VERIFICATION")
    print(f"{'='*50}")
    print(f"Initial checksum: {initial:.6f}")
    print(f"Final checksum:   {final:.6f}")
    print(f"Difference:       {diff:.10f}")
    
    if diff < 1e-6:
        print("✓ VERIFIED: Backbone weights unchanged during TTA")
        verified = True
    else:
        print("✗ WARNING: Backbone weights may have been modified!")
        verified = False
    
    print(f"{'='*50}")
    
    return {'initial': initial, 'final': final, 'diff': diff, 'verified': verified}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Post-process NOMA TTA evaluation results"
    )
    parser.add_argument("--baseline", type=str,
                        default="./noma_TTA/output/eval_baseline_results.safetensors",
                        help="Path to baseline results")
    parser.add_argument("--tta", type=str,
                        default="./noma_TTA/output/eval_tta_results.safetensors",
                        help="Path to TTA results")
    parser.add_argument("--control", type=str,
                        default="./noma_TTA/output/eval_control_results.safetensors",
                        help="Path to control results")
    parser.add_argument("--output-dir", type=str,
                        default="./noma_TTA/output",
                        help="Output directory for generated files")
    parser.add_argument("--window-size", type=int, default=500,
                        help="Rolling window size for accuracy")
    parser.add_argument("--t0", type=int, default=30000,
                        help="Distribution shift point")
    
    args = parser.parse_args()
    
    # Generate comparison results
    summary = generate_comparison_results(
        baseline_path=args.baseline,
        tta_path=args.tta,
        control_path=args.control,
        output_dir=args.output_dir,
        window_size=args.window_size,
        t0=args.t0
    )
    
    # Verify backbone integrity
    integrity = verify_backbone_integrity(args.tta)
    
    if summary and integrity:
        # Add integrity check to summary
        summary['backbone_integrity'] = integrity
        summary_path = Path(args.output_dir) / 'tta_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
    
    print("\nPost-processing complete!")
