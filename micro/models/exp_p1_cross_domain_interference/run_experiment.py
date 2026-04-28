#!/usr/bin/env python3
"""
Cross-domain interference matrix: run each adapter on all 3 benchmarks.

Kill criteria:
  K2067: Off-domain adapter degrades base ≤3pp (math on MedQA, medical on GSM8K, etc.)
  K2068: On-domain adapter improves base ≥10pp (reconfirms single-domain baselines)
"""

import gc
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

mx.set_memory_limit(mx.device_info()["memory_size"] - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"
REPO_ROOT = EXPERIMENT_DIR.parent.parent.parent
MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"

ADAPTER_PATHS = {
    "math": REPO_ROOT / "adapters" / "math",
    "python": REPO_ROOT / "adapters" / "python",
    "medical": REPO_ROOT / "adapters" / "medical",
}

IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"
N_EVAL = 5 if IS_SMOKE else 50
SEED = 42


def log_memory(label=""):
    active = mx.get_active_memory() / 1e9
    cache = mx.get_cache_memory() / 1e9
    print(f"[MEM {label}] active={active:.2f}GB cache={cache:.2f}GB", flush=True)


def cleanup(*objects):
    for obj in objects:
        del obj
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()


def load_model(adapter_path=None):
    from mlx_lm import load
    if adapter_path:
        model, tokenizer = load(MODEL_ID, adapter_path=str(adapter_path))
    else:
        model, tokenizer = load(MODEL_ID)
    mx.eval(model.parameters())
    return model, tokenizer


def eval_gsm8k(model, tokenizer, n_eval=50) -> float:
    from datasets import load_dataset
    from mlx_lm import generate

    ds = load_dataset("openai/gsm8k", "main", split="test")
    ds = ds.shuffle(seed=SEED).select(range(min(n_eval, len(ds))))

    correct = 0
    for ex in ds:
        prompt = f"Solve the following math problem step by step.\n\n{ex['question']}\n\nAnswer:"
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        response = generate(model, tokenizer, prompt=formatted, max_tokens=1024, verbose=False)

        gt_match = re.search(r"####\s*([\d,\-\.]+)", ex["answer"])
        if not gt_match:
            continue
        gt_ans = gt_match.group(1).replace(",", "").strip()

        pred_match = re.search(r"####\s*([\d,\-\.]+)", response)
        if pred_match:
            if pred_match.group(1).replace(",", "").strip() == gt_ans:
                correct += 1
        else:
            nums = re.findall(r"\b\d+\.?\d*\b", response.replace(",", ""))
            if nums and nums[-1] == gt_ans:
                correct += 1

    return correct / len(ds) * 100


def eval_humaneval(model, tokenizer, n_eval=50) -> float:
    from datasets import load_dataset
    from mlx_lm import generate

    ds = load_dataset("openai_humaneval", split="test")
    ds = ds.select(range(min(n_eval, len(ds))))

    passed = 0
    for ex in ds:
        messages = [{"role": "user", "content": f"Complete the following Python function:\n\n```python\n{ex['prompt']}\n```\n\nRespond with only the function body, no markdown."}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        response = generate(model, tokenizer, prompt=formatted, max_tokens=512, verbose=False)

        code_match = re.search(r"```python\n(.*?)```", response, re.DOTALL)
        completion = code_match.group(1) if code_match else response
        full_code = ex["prompt"] + completion + "\n\n" + ex["test"] + f"\n\ncheck({ex['entry_point']})\n"

        try:
            result = subprocess.run(
                [sys.executable, "-c", full_code],
                timeout=10, capture_output=True, text=True,
            )
            if result.returncode == 0:
                passed += 1
        except (subprocess.TimeoutExpired, Exception):
            pass

    return passed / len(ds) * 100


def eval_medqa(model, tokenizer, n_eval=50) -> float:
    from datasets import load_dataset
    from mlx_lm import generate

    ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
    ds = ds.shuffle(seed=SEED).select(range(min(n_eval, len(ds))))

    correct = 0
    for ex in ds:
        opts = ex["options"]
        question = (
            f"{ex['question']}\n"
            f"(A) {opts['A']}\n(B) {opts['B']}\n(C) {opts['C']}\n(D) {opts['D']}"
        )
        prompt = f"Answer this medical multiple choice question. Respond with only the letter (A/B/C/D).\n\n{question}"
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        response = generate(model, tokenizer, prompt=formatted, max_tokens=20, verbose=False)

        gt = ex["answer_idx"]
        pred = response.strip().upper()
        pred_letter = None
        for letter in ["A", "B", "C", "D"]:
            if pred.startswith(letter):
                pred_letter = letter
                break
        if pred_letter is None:
            m = re.search(r"\b([ABCD])\b", pred)
            if m:
                pred_letter = m.group(1)
        if pred_letter == gt:
            correct += 1

    return correct / len(ds) * 100


BENCHMARKS = {
    "gsm8k": eval_gsm8k,
    "humaneval": eval_humaneval,
    "medqa": eval_medqa,
}

ON_DOMAIN = {
    "math": "gsm8k",
    "python": "humaneval",
    "medical": "medqa",
}


def main():
    t_start = time.time()
    log_memory("start")
    print(f"Cross-domain interference matrix (SMOKE={IS_SMOKE}, N_EVAL={N_EVAL})", flush=True)

    for name, path in ADAPTER_PATHS.items():
        if not (path / "adapters.safetensors").exists():
            print(f"FATAL: Missing {path / 'adapters.safetensors'}", flush=True)
            sys.exit(1)

    # ── Phase 1: Base model on all benchmarks ─────────────
    print("\n=== Phase 1: Base model baselines ===", flush=True)
    model, tokenizer = load_model()
    base = {}
    for bench_name, eval_fn in BENCHMARKS.items():
        acc = eval_fn(model, tokenizer, n_eval=N_EVAL)
        base[bench_name] = round(acc, 1)
        print(f"  base → {bench_name}: {acc:.1f}%", flush=True)
    cleanup(model, tokenizer)

    # ── Phase 2: Each adapter on all benchmarks (3x3) ─────
    print("\n=== Phase 2: 3x3 interference matrix ===", flush=True)
    matrix = {}

    for adapter_name in ["math", "python", "medical"]:
        print(f"\n  Loading {adapter_name} adapter...", flush=True)
        model, tokenizer = load_model(adapter_path=ADAPTER_PATHS[adapter_name])
        matrix[adapter_name] = {}

        for bench_name, eval_fn in BENCHMARKS.items():
            acc = eval_fn(model, tokenizer, n_eval=N_EVAL)
            matrix[adapter_name][bench_name] = round(acc, 1)
            delta = acc - base[bench_name]
            is_on = ON_DOMAIN[adapter_name] == bench_name
            tag = "ON-DOMAIN" if is_on else "off-domain"
            print(f"    {adapter_name} → {bench_name}: {acc:.1f}% ({delta:+.1f}pp) [{tag}]", flush=True)

        cleanup(model, tokenizer)

    # ── Kill criteria ─────────────────────────────────────
    print("\n=== Kill Criteria ===", flush=True)

    # K2067: Off-domain degradation ≤3pp
    off_domain_deltas = []
    for adapter_name in ["math", "python", "medical"]:
        for bench_name in BENCHMARKS:
            if ON_DOMAIN[adapter_name] == bench_name:
                continue
            delta = matrix[adapter_name][bench_name] - base[bench_name]
            off_domain_deltas.append({
                "adapter": adapter_name,
                "benchmark": bench_name,
                "delta_pp": round(delta, 1),
            })

    worst_off = min(d["delta_pp"] for d in off_domain_deltas)
    k2067_pass = all(d["delta_pp"] >= -3 for d in off_domain_deltas)

    # K2068: On-domain improvement ≥10pp
    on_domain_deltas = []
    for adapter_name in ["math", "python", "medical"]:
        bench_name = ON_DOMAIN[adapter_name]
        delta = matrix[adapter_name][bench_name] - base[bench_name]
        on_domain_deltas.append({
            "adapter": adapter_name,
            "benchmark": bench_name,
            "delta_pp": round(delta, 1),
        })

    k2068_pass = all(d["delta_pp"] >= 10 for d in on_domain_deltas)

    all_pass = k2067_pass and k2068_pass
    verdict = "SUPPORTED" if all_pass else "KILLED"
    if IS_SMOKE:
        verdict = "PROVISIONAL"

    results = {
        "is_smoke": IS_SMOKE,
        "n_eval": N_EVAL,
        "base": base,
        "matrix": matrix,
        "kill_criteria": {
            "K2067_off_domain_degradation": {
                "pass": k2067_pass,
                "worst_delta_pp": worst_off,
                "details": off_domain_deltas,
            },
            "K2068_on_domain_improvement": {
                "pass": k2068_pass,
                "details": on_domain_deltas,
            },
        },
        "verdict": verdict,
        "all_pass": all_pass,
        "total_time_s": round(time.time() - t_start, 1),
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2))

    print(f"\nK2067 Off-domain ≤3pp degradation: {'PASS' if k2067_pass else 'FAIL'} (worst: {worst_off:+.1f}pp)", flush=True)
    for d in off_domain_deltas:
        print(f"  {d['adapter']} → {d['benchmark']}: {d['delta_pp']:+.1f}pp", flush=True)
    print(f"K2068 On-domain ≥10pp improvement: {'PASS' if k2068_pass else 'FAIL'}", flush=True)
    for d in on_domain_deltas:
        print(f"  {d['adapter']} → {d['benchmark']}: {d['delta_pp']:+.1f}pp", flush=True)
    print(f"\nVERDICT: {verdict}", flush=True)
    print(f"Total time: {results['total_time_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
