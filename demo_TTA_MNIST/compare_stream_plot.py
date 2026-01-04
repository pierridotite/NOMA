#!/usr/bin/env python3
"""
Generate a stream-accuracy comparison plot between the Python DCP baseline
and the NOMA baseline using the precomputed stream metrics CSVs.
"""
from pathlib import Path
import csv
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
# Python baseline metrics live under python_baseline/output
PYTHON_METRICS = ROOT / "python_baseline" / "output" / "static_baseline_metrics.csv"
NOMA_METRICS = ROOT / "noma_baseline" / "output" / "noma_baseline_metrics.csv"
OUT_DIR = ROOT / "output"
OUT_PNG = OUT_DIR / "stream_accuracy_comparison.png"
OUT_CSV = OUT_DIR / "stream_accuracy_comparison_samples.csv"


def load_metrics(path: Path):
    """Load t, rolling_accuracy, and phase columns from a metrics CSV."""
    ts, accs, phases = [], [], []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts.append(int(row["t"]))
            accs.append(float(row["rolling_accuracy"]))
            phases.append(row["phase"])
    return ts, accs, phases


def find_phase_changes(phases):
    """Return indices where the phase changes."""
    markers = []
    if not phases:
        return markers
    prev = phases[0]
    for idx, p in enumerate(phases[1:], start=1):
        if p != prev:
            markers.append((idx, p))
            prev = p
    return markers


def write_sample_csv(ts_py, acc_py, ts_noma, acc_noma):
    """Write a compact CSV with aligned samples for quick inspection."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sample = min(len(ts_py), len(ts_noma))
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "python_acc", "noma_acc"])
        for i in range(sample):
            writer.writerow([ts_py[i], acc_py[i], acc_noma[i]])


def main():
    ts_py, acc_py, phases_py = load_metrics(PYTHON_METRICS)
    ts_noma, acc_noma, phases_noma = load_metrics(NOMA_METRICS)

    # Align lengths in case one stream is shorter
    n = min(len(ts_py), len(ts_noma))
    ts_py, acc_py = ts_py[:n], acc_py[:n]
    ts_noma, acc_noma = ts_noma[:n], acc_noma[:n]
    phases = phases_py[:n]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_sample_csv(ts_py, acc_py, ts_noma, acc_noma)

    # Find t0 (where phase changes from "clean" to "drift")
    t0 = None
    for idx, phase in enumerate(phases):
        if phase == "drift":
            t0 = ts_py[idx]
            break
    if t0 is None:
        t0 = n // 2  # fallback to midpoint

    # Load intensity from NOMA metrics for bottom subplot
    with open(NOMA_METRICS) as f:
        reader = list(list(line) for line in __import__('csv').reader(f))[1:]  # skip header
        intensity = [float(row[5]) for row in reader[:n]]  # column 5 is intensity

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                     gridspec_kw={'height_ratios': [3, 1]})

    # Top plot: Rolling accuracy comparison
    ax1.plot(ts_py, acc_py, label="DCP Python baseline", color="#2b6cb0", linewidth=1.2, alpha=0.8)
    ax1.plot(ts_noma, acc_noma, label="NOMA baseline", color="#d97706", linewidth=1.2, alpha=0.8)

    # Add t0 marker
    ax1.axvline(x=t0, color="red", linestyle="--", linewidth=2, label=f"t0 = {t0}")

    # Shade regions (clean and drift)
    ax1.axvspan(0, t0, alpha=0.1, color="green", label="Clean Phase")
    ax1.axvspan(t0, max(ts_py), alpha=0.1, color="red", label="Drift Phase")

    ax1.set_ylabel("Rolling accuracy (%)", fontsize=12)
    ax1.set_ylim(0, 105)
    ax1.set_xlim(0, max(ts_py))
    ax1.legend(loc="lower left", fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_title("DCP Python vs NOMA - Streaming Evaluation\nMNIST with Distribution Shift",
                  fontsize=14)

    # Bottom plot: Perturbation intensity
    ax2.fill_between(ts_py, intensity, alpha=0.5, color="orange", label="Perturbation Intensity")
    ax2.axvline(x=t0, color="red", linestyle="--", linewidth=2)
    ax2.set_xlabel("Time (t)", fontsize=12)
    ax2.set_ylabel("Intensity", fontsize=12)
    ax2.set_ylim(0, 1.0)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper left", fontsize=10)

    plt.tight_layout()
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {OUT_PNG}")
    print(f"Saved aligned samples to {OUT_CSV}")


if __name__ == "__main__":
    main()
