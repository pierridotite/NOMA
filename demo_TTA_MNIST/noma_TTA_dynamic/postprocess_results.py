#!/usr/bin/env python3
"""Postprocess and compare TTA results."""

import numpy as np
import struct
import json
from pathlib import Path
import matplotlib.pyplot as plt

def load_safetensors(filepath):
    with open(filepath, 'rb') as f:
        header_size = struct.unpack('<Q', f.read(8))[0]
        header_json = f.read(header_size).decode('utf-8')
        header = json.loads(header_json)
        data_start = 8 + header_size
        
        result = {}
        for name, info in header.items():
            shape = info['shape']
            start, end = info['data_offsets']
            f.seek(data_start + start)
            arr = np.frombuffer(f.read(end - start), dtype=np.float64)
            result[name] = arr.reshape(shape) if shape else arr
        
        return result

def compute_accuracy(pred_probs, y_true):
    pred_labels = np.argmax(pred_probs, axis=1)
    return np.mean(pred_labels == y_true.astype(int)) * 100

def compute_rolling_accuracy(pred_probs, y_true, window_size=500):
    """Compute rolling accuracy over time."""
    pred_labels = np.argmax(pred_probs, axis=1)
    correct = (pred_labels == y_true.astype(int)).astype(float)
    
    rolling_acc = np.zeros(len(correct))
    buffer = []
    
    for i, c in enumerate(correct):
        buffer.append(c)
        if len(buffer) > window_size:
            buffer.pop(0)
        rolling_acc[i] = 100.0 * np.mean(buffer)
    
    return rolling_acc

def load_and_combine(clean_path, drift_path):
    clean = load_safetensors(clean_path)
    drift = load_safetensors(drift_path)
    
    return {
        'pred_probs': np.vstack([clean['pred_probs'], drift['pred_probs']]),
        'y_true': np.concatenate([clean['y_true'], drift['y_true']]),
        't': np.concatenate([clean['t'], drift['t']]),
        'phase': np.concatenate([clean['phase'], drift['phase']]),
        'intensity': np.concatenate([clean['intensity'], drift['intensity']])
    }

def main():
    print("=" * 60)
    print("Dynamic TTA - Results Analysis")
    print("=" * 60)
    
    output_dir = Path("output")
    
    # Load results
    print("\nLoading results...")
    
    baseline = load_safetensors(output_dir / "eval_baseline.safetensors")
    static = load_and_combine(
        output_dir / "eval_static_clean.safetensors",
        output_dir / "eval_static_drift.safetensors"
    )
    
    # Streaming TTA (single file - uses streaming_adapt)
    streaming_path = output_dir / "eval_streaming.safetensors"
    if streaming_path.exists():
        streaming = load_safetensors(streaming_path)
    else:
        streaming = None
    
    # Compute metrics
    results = {}
    
    datasets = [("Baseline", baseline), ("Static TTA", static)]
    if streaming is not None:
        datasets.append(("Streaming TTA", streaming))
    
    for name, data in datasets:
        overall = compute_accuracy(data['pred_probs'], data['y_true'])
        
        clean_mask = data['phase'] == 0
        drift_mask = data['phase'] > 0
        
        clean_acc = compute_accuracy(
            data['pred_probs'][clean_mask], 
            data['y_true'][clean_mask]
        )
        drift_acc = compute_accuracy(
            data['pred_probs'][drift_mask], 
            data['y_true'][drift_mask]
        )
        
        results[name] = {
            'overall': overall,
            'clean': clean_acc,
            'drift': drift_acc,
            'drop': clean_acc - drift_acc
        }
    
    # Print comparison
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"{'Method':<15} {'Overall':>10} {'Clean':>10} {'Drift':>10} {'Drop':>10}")
    print("-" * 60)
    
    for name, r in results.items():
        print(f"{name:<15} {r['overall']:>9.2f}% {r['clean']:>9.2f}% {r['drift']:>9.2f}% {r['drop']:>9.2f}%")
    
    print("-" * 60)
    
    # Recovery metrics
    baseline_drift = results["Baseline"]["drift"]
    static_recovery = results["Static TTA"]["drift"] - baseline_drift
    
    print(f"\nRecovery on drift phase:")
    print(f"  Static TTA:    +{static_recovery:.2f}%")
    
    if "Streaming TTA" in results:
        streaming_recovery = results["Streaming TTA"]["drift"] - baseline_drift
        print(f"  Streaming TTA: +{streaming_recovery:.2f}%")
        print(f"  Difference:    {streaming_recovery - static_recovery:+.2f}%")
    else:
        streaming_recovery = 0
    
    # Save summary
    summary = {
        'results': results,
        'recovery': {
            'static': static_recovery,
            'streaming': streaming_recovery if "Streaming TTA" in results else None
        }
    }
    
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to {output_dir / 'summary.json'}")
    
    # Generate comparison plot
    print("\nGenerating visualization...")
    generate_plot(baseline, static, streaming, results, output_dir)
    
    print("\nDone.")


def generate_plot(baseline, static, streaming, results, output_dir):
    """Generate comparison visualization."""
    
    t0 = 30000
    window_size = 500
    
    # Compute rolling accuracies
    rolling_baseline = compute_rolling_accuracy(baseline['pred_probs'], baseline['y_true'], window_size)
    rolling_static = compute_rolling_accuracy(static['pred_probs'], static['y_true'], window_size)
    
    t = baseline['t']
    intensity = baseline['intensity']
    
    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), 
                              gridspec_kw={'height_ratios': [3, 1, 1]})
    
    colors = {
        'baseline': '#1f77b4',  # Blue
        'static': '#ff7f0e',    # Orange
        'streaming': '#2ca02c'  # Green
    }
    
    # Main plot: Rolling accuracy comparison
    ax1 = axes[0]
    
    ax1.plot(t, rolling_baseline, color=colors['baseline'], 
             linewidth=1.0, alpha=0.9, label='Baseline (No Adaptation)')
    ax1.plot(t, rolling_static, color=colors['static'], 
             linewidth=1.2, alpha=0.9, label='Static TTA (epoch-based)')
    
    if streaming is not None:
        rolling_streaming = compute_rolling_accuracy(streaming['pred_probs'], streaming['y_true'], window_size)
        ax1.plot(t, rolling_streaming, color=colors['streaming'], 
                 linewidth=1.5, alpha=0.9, label='Streaming TTA (streaming_adapt)')
    
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
    static_recovery = results['Static TTA']['drift'] - results['Baseline']['drift']
    ax1.annotate(f'Static TTA: +{static_recovery:.1f}%', 
                xy=(t0 + t0/4, results['Static TTA']['drift'] + 8),
                fontsize=11, fontweight='bold', color='darkorange')
    
    if 'Streaming TTA' in results:
        streaming_recovery = results['Streaming TTA']['drift'] - results['Baseline']['drift']
        ax1.annotate(f'Streaming TTA: +{streaming_recovery:.1f}%', 
                    xy=(t0 + t0/4, results['Streaming TTA']['drift'] + 3),
                    fontsize=11, fontweight='bold', color='darkgreen')
    
    ax1.set_ylabel('Rolling Accuracy (%)', fontsize=12)
    ax1.set_ylim(0, 105)
    ax1.set_xlim(0, max(t))
    ax1.legend(loc='lower left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f'NOMA Streaming TTA - Comparison (streaming_adapt)\n'
                  f'Window={window_size}, t0={t0}', fontsize=14)
    
    # Secondary plot: Accuracy difference vs Baseline
    ax2 = axes[1]
    
    diff_static = rolling_static - rolling_baseline
    ax2.plot(t, diff_static, color=colors['static'], linewidth=1.0, 
             alpha=0.7, label='Static TTA - Baseline')
    
    if streaming is not None:
        diff_streaming = rolling_streaming - rolling_baseline
        ax2.plot(t, diff_streaming, color=colors['streaming'], linewidth=1.2, 
                 alpha=0.9, label='Streaming TTA - Baseline')
    
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.axvline(x=t0, color='red', linestyle='--', linewidth=1)
    ax2.set_ylabel('vs Baseline (%)', fontsize=10)
    ax2.set_ylim(-10, 50)
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Tertiary plot: Perturbation intensity
    ax3 = axes[2]
    ax3.fill_between(t, intensity, alpha=0.5, color='orange', 
                     label='Perturbation Intensity')
    ax3.axvline(x=t0, color='red', linestyle='--', linewidth=1)
    ax3.set_xlabel('Time (t)', fontsize=12)
    ax3.set_ylabel('Intensity', fontsize=10)
    ax3.set_xlim(0, max(t))
    ax3.legend(loc='upper left', fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = output_dir / "tta_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved plot: {plot_path}")

if __name__ == "__main__":
    main()
