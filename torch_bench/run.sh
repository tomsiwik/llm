#!/bin/bash
# RunPod benchmark runner — run inside the pod
# Usage: bash torch_bench/run.sh [--quick]
set -e

WORKSPACE=${WORKSPACE:-/workspace}
REPO_DIR="$WORKSPACE/llm"
LOG="$WORKSPACE/bench.log"
RESULTS="$WORKSPACE/results_7b.json"

echo "=== 7B CL Benchmark Setup ==="
echo "Log: $LOG"
echo "Results: $RESULTS"

# Install deps
pip install -q -r "$REPO_DIR/torch_bench/requirements.txt"

# Run benchmark (nohup so it survives SSH disconnect)
cd "$REPO_DIR"

STEPS=500
if [[ "$1" == "--quick" ]]; then
    STEPS=50
    echo "Quick mode: $STEPS steps/domain"
fi

echo "Starting benchmark at $(date)"
echo "Steps per domain: $STEPS"

nohup python -u torch_bench/bench.py \
    --compare \
    --steps "$STEPS" \
    --seed 42 \
    --output "$RESULTS" \
    > "$LOG" 2>&1 &

echo "PID: $!"
echo "$!" > "$WORKSPACE/bench.pid"
echo "Benchmark running in background. Monitor with:"
echo "  tail -f $LOG"
echo "  python torch_bench/status.py"
