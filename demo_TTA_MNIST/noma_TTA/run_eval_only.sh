#!/bin/bash
# NOMA TTA - Quick Evaluation Only
# Runs only the evaluation steps (assumes backbone is already trained)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NOMA_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$NOMA_ROOT"

echo "========================================"
echo " NOMA TTA - Evaluation Only"
echo "========================================"

# Check if backbone exists
if [ ! -f "noma_TTA/output/backbone_weights.safetensors" ]; then
    echo "Error: Backbone weights not found!"
    echo "Please run the full pipeline first: ./noma_TTA/run_pipeline.sh"
    exit 1
fi

# Step 1: Baseline evaluation
echo ""
echo "[Step 1/4] Running baseline evaluation..."
../target/release/noma run noma_TTA/eval_stream_baseline.noma

# Step 2: Control evaluation
echo ""
echo "[Step 2/4] Running control evaluation..."
../target/release/noma run noma_TTA/eval_stream_control.noma

# Step 3: TTA evaluation
echo ""
echo "[Step 3/4] Running TTA evaluation..."
../target/release/noma run noma_TTA/eval_stream_tta.noma

# Step 4: Post-process
echo ""
echo "[Step 4/4] Post-processing results..."
python noma_TTA/postprocess_results.py

echo ""
echo "========================================"
echo " Evaluation Complete!"
echo "========================================"
