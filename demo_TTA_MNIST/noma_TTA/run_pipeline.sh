#!/bin/bash
# NOMA TTA - Full Pipeline
# This script runs the complete TTA experiment:
# 1. Train backbone on clean MNIST
# 2. Run baseline evaluation (no adaptation)
# 3. Run TTA evaluation (LoRA + online updates)
# 4. Run control evaluation (LoRA without updates)
# 5. Post-process and compare results

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NOMA_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$NOMA_ROOT"

echo "========================================"
echo " NOMA TTA - Full Pipeline"
echo "========================================"
echo "Working directory: $(pwd)"

# Create output directory
mkdir -p noma_TTA/output
mkdir -p noma_TTA/data

# Step 1: Prepare training data
echo ""
echo "[Step 1/7] Preparing training data..."
python noma_TTA/prepare_data.py

# Step 2: Prepare streaming data
echo ""
echo "[Step 2/7] Preparing streaming data..."
python noma_TTA/prepare_stream.py

# Step 3: Train NOMA backbone
echo ""
echo "[Step 3/7] Training backbone on clean MNIST..."
../target/release/noma run noma_TTA/train_mnist.noma

# Step 4: Baseline evaluation (no adaptation)
echo ""
echo "[Step 4/7] Running baseline evaluation (no adaptation)..."
../target/release/noma run noma_TTA/eval_stream_baseline.noma

# Step 5: Control evaluation (LoRA without updates)
echo ""
echo "[Step 5/7] Running control evaluation (LoRA, no updates)..."
../target/release/noma run noma_TTA/eval_stream_control.noma

# Step 6: TTA evaluation (LoRA with online updates)
echo ""
echo "[Step 6/7] Running TTA evaluation (LoRA + online updates)..."
../target/release/noma run noma_TTA/eval_stream_tta.noma

# Step 7: Post-process and compare results
echo ""
echo "[Step 7/7] Post-processing and comparing results..."
python noma_TTA/postprocess_results.py

echo ""
echo "========================================"
echo " Pipeline Complete!"
echo "========================================"
echo ""
echo "Output files:"
echo "  - noma_TTA/output/backbone_weights.safetensors      (trained backbone)"
echo "  - noma_TTA/output/eval_baseline_results.safetensors (baseline predictions)"
echo "  - noma_TTA/output/eval_control_results.safetensors  (control predictions)"
echo "  - noma_TTA/output/eval_tta_results.safetensors      (TTA predictions)"
echo "  - noma_TTA/output/tta_comparison.png                (comparison plot)"
echo "  - noma_TTA/output/tta_metrics.csv                   (per-sample metrics)"
echo "  - noma_TTA/output/tta_summary.json                  (summary statistics)"
echo ""
echo "Expected results:"
echo "  - Baseline: High clean accuracy, significant drop after drift"
echo "  - Control:  Same as baseline (LoRA is zero, no updates)"
echo "  - TTA:      Recovery after drift due to online LoRA adaptation"
