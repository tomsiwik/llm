#!/usr/bin/env python3
"""
BENCH: LiveCodeBench v6 Baseline + Pierre Adapted
Google target: 52.0%

Runs LiveCodeBench v6 code generation locally. No submission anywhere.
Uses mlx_lm.server as OpenAI-compatible backend.
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
LCB_DIR = EXPERIMENT_DIR.parent / "reference_implementations" / "LiveCodeBench"
PORT = 8321
IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"


def log(msg):
    print(msg, flush=True)


def start_mlx_server(model_path, port=PORT):
    cmd = [sys.executable, "-m", "mlx_lm.server",
           "--model", model_path, "--port", str(port)]
    log(f"  Starting MLX server: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    import urllib.request
    for _ in range(60):
        time.sleep(2)
        try:
            urllib.request.urlopen(f"http://localhost:{port}/v1/models", timeout=2)
            log(f"  MLX server ready on port {port}")
            return proc
        except Exception:
            continue
    raise RuntimeError("MLX server failed to start")


def stop_server(proc):
    if proc and proc.poll() is None:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=10)
        log("  MLX server stopped")


def run_livecodebench(port=PORT):
    """Run LiveCodeBench v6 code generation against local server."""
    env = os.environ.copy()
    env["OPENAI_API_KEY"] = "not-needed"
    env["OPENAI_BASE_URL"] = f"http://localhost:{port}/v1"

    cmd = [
        sys.executable, "-m", "lcb_runner.runner.main",
        "--model", "gemma-4-e4b",
        "--scenario", "codegeneration",
        "--release_version", "release_v6",
        "--evaluate",
        "--n", "1" if IS_SMOKE else "10",
        "--temperature", "0.2",
    ]

    log(f"  Running: {' '.join(cmd)}")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=10 * 3600, env=env, cwd=str(LCB_DIR))
    elapsed = time.time() - t0

    log(f"  LiveCodeBench completed in {elapsed:.0f}s (exit={result.returncode})")
    log(f"  STDOUT: {result.stdout[-1000:]}")
    if result.returncode != 0:
        log(f"  STDERR: {result.stderr[-500:]}")

    return {
        "stdout": result.stdout[-2000:],
        "stderr": result.stderr[-500:] if result.returncode != 0 else "",
        "elapsed_s": elapsed,
        "exit_code": result.returncode,
    }


def main():
    log("=" * 70)
    log("BENCH: LiveCodeBench v6 — Base E4B")
    log(f"SMOKE={IS_SMOKE}")
    log("=" * 70)

    results = {"experiment": "exp_bench_livecodebench_v6", "smoke": IS_SMOKE}

    # Install LiveCodeBench if needed
    log("\n[Setup] Installing LiveCodeBench...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", str(LCB_DIR)],
                   capture_output=True, timeout=300)

    # Phase 1: Base model
    log("\n[Phase 1] Base Gemma 4 E4B")
    server = start_mlx_server(MODEL_ID)
    try:
        base_results = run_livecodebench()
        results["base"] = base_results
    finally:
        stop_server(server)

    RESULTS_FILE.write_text(json.dumps(results, indent=2, default=str))
    log(f"\nSaved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
