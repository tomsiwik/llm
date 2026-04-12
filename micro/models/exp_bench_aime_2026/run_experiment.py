#!/usr/bin/env python3
"""
BENCH: AIME 2026 Baseline + Pierre Adapted
Google target: 42.5%

Runs AIME 2026 (30 math competition problems) via MathArena harness.
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
MATHARENA_DIR = EXPERIMENT_DIR.parent / "reference_implementations" / "matharena"
PORT = 8321
IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"
N_SAMPLES = 1 if IS_SMOKE else 4  # seeds per problem


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


def run_matharena(port=PORT, n_samples=N_SAMPLES):
    """Run MathArena AIME 2026 against local server."""
    # MathArena uses OpenAI-compatible API
    env = os.environ.copy()
    env["OPENAI_API_KEY"] = "not-needed"
    env["OPENAI_BASE_URL"] = f"http://localhost:{port}/v1"

    # Create a temporary model config for local serving
    config = {
        "model": "gemma-4-e4b",
        "api": "openai",
        "base_url": f"http://localhost:{port}/v1",
        "api_key": "not-needed",
        "temperature": 0.7,
        "max_tokens": 4096,
    }
    config_path = EXPERIMENT_DIR / "model_config.json"
    config_path.write_text(json.dumps(config, indent=2))

    # Run MathArena
    cmd = [
        sys.executable, str(MATHARENA_DIR / "scripts" / "run.py"),
        "--comp", "aime/aime_2026",
        "--n", str(n_samples),
        "--output_dir", str(EXPERIMENT_DIR / "matharena_output"),
    ]

    log(f"  Running: {' '.join(cmd)}")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=3 * 3600, env=env, cwd=str(MATHARENA_DIR))
    elapsed = time.time() - t0

    log(f"  MathArena completed in {elapsed:.0f}s (exit={result.returncode})")
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
    log("BENCH: AIME 2026 — Base E4B vs Pierre Adapted")
    log(f"SMOKE={IS_SMOKE}, N_SAMPLES={N_SAMPLES}")
    log("=" * 70)

    results = {"experiment": "exp_bench_aime_2026", "smoke": IS_SMOKE}

    # Install matharena if needed
    log("\n[Setup] Installing MathArena...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", str(MATHARENA_DIR)],
                   capture_output=True, timeout=300)

    # Phase 1: Base model
    log("\n[Phase 1] Base Gemma 4 E4B")
    server = start_mlx_server(MODEL_ID)
    try:
        base_results = run_matharena()
        results["base"] = base_results
    finally:
        stop_server(server)

    RESULTS_FILE.write_text(json.dumps(results, indent=2, default=str))
    log(f"\nSaved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
