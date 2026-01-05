#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=================================================="
echo "Dynamic TTA Pipeline"
echo "=================================================="

# Prepare data
echo ""
echo "[1/5] Preparing data..."
python3 prepare_data.py

# Train backbone (skip if already exists)
echo ""
echo "[2/5] Training backbone..."
if [ -f "output/backbone_weights.safetensors" ]; then
    echo "Backbone weights already exist, skipping training."
else
    ../../target/release/noma run train_backbone.noma
fi

# Baseline evaluation
echo ""
echo "[3/5] Running baseline (no TTA)..."
../../target/release/noma run eval_baseline.noma

# Static TTA
echo ""
echo "[4/5] Running static TTA (fixed rank=4)..."
../../target/release/noma run eval_static.noma

# Dynamic TTA
echo ""
echo "[5/5] Running dynamic TTA (alloc/realloc)..."
../../target/release/noma run eval_dynamic.noma

# Postprocess
echo ""
echo "Analyzing results..."
python3 postprocess_results.py

echo ""
echo "Pipeline complete."
