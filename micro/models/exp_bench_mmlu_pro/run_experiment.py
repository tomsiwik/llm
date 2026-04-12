#!/usr/bin/env python3
"""
BENCH: MMLU-Pro Baseline + Pierre Adapted
Google target: 69.4%

Runs MMLU-Pro via lm-eval-harness against mlx_lm.server.
Two phases: (1) base model, (2) base + best adapter merged.
"""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"
MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"
PORT = 8321
IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"

# Smoke: use a tiny task for validation; full: mmlu_pro
TASK = "mmlu_pro" if not IS_SMOKE else "mmlu_pro"
LIMIT = 20 if IS_SMOKE else None  # None = full dataset


def log(msg):
    print(msg, flush=True)


def start_mlx_server(model_path, adapter_path=None, port=PORT):
    """Start mlx_lm.server in background. Returns Popen."""
    cmd = [sys.executable, "-m", "mlx_lm.server",
           "--model", model_path, "--port", str(port)]
    if adapter_path:
        cmd += ["--adapter-path", str(adapter_path)]
    log(f"  Starting MLX server: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # Wait for server to be ready
    import urllib.request
    for _ in range(60):
        time.sleep(2)
        try:
            urllib.request.urlopen(f"http://localhost:{port}/v1/models", timeout=2)
            log(f"  MLX server ready on port {port}")
            return proc
        except Exception:
            continue
    raise RuntimeError("MLX server failed to start in 120s")


def stop_server(proc):
    """Kill server process group."""
    if proc and proc.poll() is None:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=10)
        log("  MLX server stopped")


def run_lm_eval(task, port=PORT, limit=None, output_dir="results"):
    """Run lm-eval-harness against local server. Returns parsed results."""
    out_path = EXPERIMENT_DIR / output_dir
    out_path.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "local-chat-completions",
        "--model_args", f"model=gemma-4-e4b,base_url=http://localhost:{port}/v1,tokenized_requests=False",
        "--tasks", task,
        "--batch_size", "1",
        "--output_path", str(out_path),
        "--log_samples",
    ]
    if limit:
        cmd += ["--limit", str(limit)]

    log(f"  Running: {' '.join(cmd)}")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=7 * 3600)
    elapsed = time.time() - t0

    log(f"  lm-eval completed in {elapsed:.0f}s (exit={result.returncode})")
    if result.returncode != 0:
        log(f"  STDERR: {result.stderr[-500:]}")
        return {"error": result.stderr[-500:], "elapsed_s": elapsed}

    # Parse results from output
    log(f"  STDOUT (last 1000 chars): {result.stdout[-1000:]}")

    # Try to find results JSON
    results_files = list(out_path.rglob("results_*.json"))
    if results_files:
        with open(results_files[-1]) as f:
            parsed = json.load(f)
        return {"parsed": parsed, "elapsed_s": elapsed}
    return {"stdout": result.stdout[-2000:], "elapsed_s": elapsed}


def main():
    log("=" * 70)
    log("BENCH: MMLU-Pro — Base E4B vs Pierre Adapted")
    log(f"SMOKE_TEST={IS_SMOKE}, LIMIT={LIMIT}")
    log("=" * 70)

    results = {"experiment": "exp_bench_mmlu_pro", "smoke": IS_SMOKE}

    # Phase 1: Base model
    log("\n[Phase 1] Base Gemma 4 E4B")
    server = start_mlx_server(MODEL_ID)
    try:
        base_results = run_lm_eval(TASK, limit=LIMIT, output_dir="results_base")
        results["base"] = base_results
    finally:
        stop_server(server)

    # Phase 2: Pierre adapted (TODO: merge best adapter and test)
    # For now, just record base results
    log("\n[Phase 2] Pierre Adapted — skipped (adapter merge needed)")
    results["adapted"] = {"status": "pending", "note": "Merge best adapter and re-run"}

    # Summary
    log("\n" + "=" * 70)
    log("RESULTS")
    log("=" * 70)
    if "parsed" in results.get("base", {}):
        log(f"  Base results: {json.dumps(results['base']['parsed'].get('results', {}), indent=2)[:500]}")
    else:
        log(f"  Base: {results.get('base', {}).get('error', 'unknown')}")

    RESULTS_FILE.write_text(json.dumps(results, indent=2, default=str))
    log(f"\nSaved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
