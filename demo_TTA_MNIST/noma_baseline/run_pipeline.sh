#!/bin/bash
# NOMA Static Baseline - Full Pipeline
# This script runs the complete training and evaluation pipeline

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NOMA_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$NOMA_ROOT"

echo "========================================"
echo " NOMA Static Baseline - Full Pipeline"
echo "========================================"

# Step 1: Prepare training data
echo ""
echo "[Step 1/5] Preparing training data..."
python noma_baseline/prepare_data.py --max-samples 1000

# Step 2: Prepare streaming data
echo ""
echo "[Step 2/5] Preparing streaming data..."
python noma_baseline/prepare_stream.py

# Step 3: Train NOMA model
echo ""
echo "[Step 3/5] Training NOMA model..."
mkdir -p noma_baseline/output
../target/release/noma run noma_baseline/train_mnist.noma

# Step 4: Evaluate on stream
echo ""
echo "[Step 4/5] Evaluating on stream..."
../target/release/noma run noma_baseline/eval_stream.noma

# Step 5: Post-process results
echo ""
echo "[Step 5/5] Post-processing results..."
python noma_baseline/postprocess_results.py

echo ""
echo "========================================"
echo " Pipeline Complete!"
echo "========================================"
echo ""
echo "Output files:"
echo "  - noma_baseline/output/model_weights.safetensors"
echo "  - noma_baseline/output/eval_results.safetensors"
echo "  - noma_baseline/output/noma_baseline_accuracy.png"
echo "  - noma_baseline/output/noma_baseline_metrics.csv"
echo "  - noma_baseline/output/noma_baseline_summary.json"
