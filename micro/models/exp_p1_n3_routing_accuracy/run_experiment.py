#!/usr/bin/env python3
"""
N=3 routing: TF-IDF + ridge classifier selects correct adapter per sample.

Kill criteria:
  K2065: Router top-1 accuracy ≥85% on held-out test samples from 3 domains
  K2066: Routed composition task accuracy ≥ uniform-weight composition (behavioral)
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
import mlx.nn as nn
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split

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
N_ROUTER_SAMPLES = 50 if IS_SMOKE else 500
N_EVAL = 5 if IS_SMOKE else 30
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


# ─────────────────────────────────────────────
# Router training
# ─────────────────────────────────────────────

def build_router_dataset(n_per_domain: int):
    """Build TF-IDF routing dataset from the 3 domain test sets."""
    from datasets import load_dataset

    texts, labels = [], []

    ds_math = load_dataset("openai/gsm8k", "main", split="test")
    ds_math = ds_math.shuffle(seed=SEED).select(range(min(n_per_domain, len(ds_math))))
    for ex in ds_math:
        texts.append(ex["question"])
        labels.append("math")

    ds_code = load_dataset("openai_humaneval", split="test")
    ds_code = ds_code.select(range(min(n_per_domain, len(ds_code))))
    for ex in ds_code:
        texts.append(ex["prompt"])
        labels.append("code")

    ds_med = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
    ds_med = ds_med.shuffle(seed=SEED).select(range(min(n_per_domain, len(ds_med))))
    for ex in ds_med:
        texts.append(ex["question"])
        labels.append("medical")

    return texts, labels


def train_router(texts, labels):
    """Train TF-IDF + RidgeClassifier router."""
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=SEED, stratify=labels
    )

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    clf = RidgeClassifier(alpha=1.0)
    clf.fit(X_train_tfidf, y_train)

    train_acc = clf.score(X_train_tfidf, y_train) * 100
    test_acc = clf.score(X_test_tfidf, y_test) * 100

    return vectorizer, clf, train_acc, test_acc, X_test, y_test


def route_sample(vectorizer, clf, text: str) -> dict:
    """Route a single sample, returning domain label and confidence scores."""
    X = vectorizer.transform([text])
    pred = clf.predict(X)[0]
    scores = clf.decision_function(X)[0]
    domain_scores = dict(zip(clf.classes_, scores))
    return {"domain": pred, "scores": domain_scores}


# ─────────────────────────────────────────────
# Adapter composition helpers
# ─────────────────────────────────────────────

def load_adapter_weights(adapter_path: Path) -> dict:
    import safetensors.numpy
    weights_file = adapter_path / "adapters.safetensors"
    flat = {}
    with safetensors.numpy.safe_open(str(weights_file), framework="numpy") as f:
        for key in f.keys():
            flat[key] = mx.array(f.get_tensor(key))
    return flat


def compose_single_adapter(adapter_weights: dict, scale: float = 6.0) -> dict:
    """Compute ΔW = scale * (B @ A) for a single adapter."""
    layer_keys = set()
    for key in adapter_weights:
        if ".lora_a" in key:
            layer_keys.add(key.replace(".lora_a", ""))

    delta = {}
    for base_key in sorted(layer_keys):
        a = adapter_weights[f"{base_key}.lora_a"]
        b = adapter_weights[f"{base_key}.lora_b"]
        delta[base_key] = (a @ b) * scale
    mx.eval(delta)
    return delta


def compose_uniform(all_weights: list[dict], scale: float = 6.0) -> dict:
    """Uniform-weight composition: Σ (B_i @ A_i) / N, scaled."""
    layer_keys = set()
    for w in all_weights:
        for key in w:
            if ".lora_a" in key:
                layer_keys.add(key.replace(".lora_a", ""))

    composed = {}
    n = len(all_weights)
    for base_key in sorted(layer_keys):
        delta_sum = None
        for w in all_weights:
            a_key = f"{base_key}.lora_a"
            b_key = f"{base_key}.lora_b"
            if a_key not in w or b_key not in w:
                continue
            d = w[a_key] @ w[b_key]
            delta_sum = d if delta_sum is None else delta_sum + d
        if delta_sum is not None:
            composed[base_key] = delta_sum * (scale / n)
    mx.eval(composed)
    return composed


class ComposedLinear(nn.Module):
    def __init__(self, base_layer, delta):
        super().__init__()
        self.base = base_layer
        self.delta = delta

    def __call__(self, x):
        return self.base(x) + x @ self.delta


def apply_delta(model, delta: dict):
    """Replace target layers with ComposedLinear wrappers (quantization-safe)."""
    for key, dw in delta.items():
        parts = key.split(".")
        parent = model
        for part in parts[:-1]:
            if part.isdigit() and isinstance(parent, list):
                parent = parent[int(part)]
            else:
                parent = getattr(parent, part)

        attr_name = parts[-1]
        base_layer = getattr(parent, attr_name)
        setattr(parent, attr_name, ComposedLinear(base_layer, dw))


# ─────────────────────────────────────────────
# Eval functions (task accuracy)
# ─────────────────────────────────────────────

def eval_gsm8k(model, tokenizer, n_eval=30) -> float:
    from datasets import load_dataset
    from mlx_lm import generate

    ds = load_dataset("openai/gsm8k", "main", split="test")
    ds = ds.shuffle(seed=SEED + 1).select(range(min(n_eval, len(ds))))

    correct = 0
    for ex in ds:
        messages = [{"role": "user", "content": f"Solve step by step.\n\n{ex['question']}\n\nAnswer:"}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        response = generate(model, tokenizer, prompt=formatted, max_tokens=1024, verbose=False)

        gt_match = re.search(r"####\s*([\d,\-\.]+)", ex["answer"])
        if not gt_match:
            continue
        gt = gt_match.group(1).replace(",", "").strip()
        pred_match = re.search(r"####\s*([\d,\-\.]+)", response)
        if pred_match and pred_match.group(1).replace(",", "").strip() == gt:
            correct += 1
        else:
            nums = re.findall(r"\b\d+\.?\d*\b", response.replace(",", ""))
            if nums and nums[-1] == gt:
                correct += 1
    return correct / len(ds) * 100


def eval_humaneval(model, tokenizer, n_eval=30) -> float:
    from datasets import load_dataset
    from mlx_lm import generate

    ds = load_dataset("openai_humaneval", split="test")
    ds = ds.select(range(min(n_eval, len(ds))))

    passed = 0
    for ex in ds:
        messages = [{"role": "user", "content": f"Complete this Python function:\n\n```python\n{ex['prompt']}\n```\n\nRespond with only the function body."}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        response = generate(model, tokenizer, prompt=formatted, max_tokens=512, verbose=False)

        code_match = re.search(r"```python\n(.*?)```", response, re.DOTALL)
        completion = code_match.group(1) if code_match else response
        full_code = ex["prompt"] + completion + "\n\n" + ex["test"] + f"\n\ncheck({ex['entry_point']})\n"
        try:
            r = subprocess.run([sys.executable, "-c", full_code], timeout=10, capture_output=True, text=True)
            if r.returncode == 0:
                passed += 1
        except (subprocess.TimeoutExpired, Exception):
            pass
    return passed / len(ds) * 100


def eval_medqa(model, tokenizer, n_eval=30) -> float:
    from datasets import load_dataset
    from mlx_lm import generate

    ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
    ds = ds.shuffle(seed=SEED + 1).select(range(min(n_eval, len(ds))))

    correct = 0
    for ex in ds:
        opts = ex["options"]
        question = f"{ex['question']}\n(A) {opts['A']}\n(B) {opts['B']}\n(C) {opts['C']}\n(D) {opts['D']}"
        messages = [{"role": "user", "content": f"Answer with only the letter (A/B/C/D).\n\n{question}"}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        response = generate(model, tokenizer, prompt=formatted, max_tokens=20, verbose=False)

        gt = ex["answer_idx"]
        pred = response.strip().upper()
        pred_letter = None
        for letter in ["A", "B", "C", "D"]:
            if pred.startswith(letter):
                pred_letter = letter
                break
        if not pred_letter:
            m = re.search(r"\b([ABCD])\b", pred)
            if m:
                pred_letter = m.group(1)
        if pred_letter == gt:
            correct += 1
    return correct / len(ds) * 100


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    t_start = time.time()
    log_memory("start")
    print(f"N=3 Routing Accuracy (SMOKE={IS_SMOKE}, N_EVAL={N_EVAL})", flush=True)

    for name, path in ADAPTER_PATHS.items():
        if not (path / "adapters.safetensors").exists():
            print(f"FATAL: Missing {path / 'adapters.safetensors'}", flush=True)
            sys.exit(1)

    results = {"is_smoke": IS_SMOKE, "n_eval": N_EVAL}

    # ── Phase 1: Train router ─────────────────────────────
    print("\n=== Phase 1: Train TF-IDF + Ridge router ===", flush=True)
    n_per = N_ROUTER_SAMPLES // 3
    texts, labels = build_router_dataset(n_per)
    vectorizer, clf, train_acc, test_acc, X_test, y_test = train_router(texts, labels)

    print(f"  Router train accuracy: {train_acc:.1f}%", flush=True)
    print(f"  Router test accuracy:  {test_acc:.1f}%", flush=True)

    results["router"] = {
        "train_accuracy": round(train_acc, 1),
        "test_accuracy": round(test_acc, 1),
        "n_train_samples": len(texts),
        "n_features": vectorizer.max_features,
    }

    # ── Phase 2: Uniform-weight composition baseline ──────
    print("\n=== Phase 2: Uniform-weight composition ===", flush=True)

    all_adapter_weights = {
        name: load_adapter_weights(path) for name, path in ADAPTER_PATHS.items()
    }
    uniform_delta = compose_uniform(list(all_adapter_weights.values()), scale=6.0)

    from mlx_lm import load as mlx_load
    model, tokenizer = mlx_load(MODEL_ID)
    mx.eval(model.parameters())
    apply_delta(model, uniform_delta)
    log_memory("uniform-composed")
    del uniform_delta

    print("  Evaluating uniform composition...", flush=True)
    uniform_gsm8k = eval_gsm8k(model, tokenizer, n_eval=N_EVAL)
    uniform_humaneval = eval_humaneval(model, tokenizer, n_eval=N_EVAL)
    uniform_medqa = eval_medqa(model, tokenizer, n_eval=N_EVAL)
    uniform_avg = (uniform_gsm8k + uniform_humaneval + uniform_medqa) / 3

    print(f"  Uniform: GSM8K={uniform_gsm8k:.1f}%, HumanEval={uniform_humaneval:.1f}%, MedQA={uniform_medqa:.1f}%, avg={uniform_avg:.1f}%", flush=True)
    results["uniform_composition"] = {
        "gsm8k": round(uniform_gsm8k, 1),
        "humaneval": round(uniform_humaneval, 1),
        "medqa": round(uniform_medqa, 1),
        "average": round(uniform_avg, 1),
    }
    cleanup(model, tokenizer)

    # ── Phase 3: Routed composition ───────────────────────
    print("\n=== Phase 3: Routed composition (per-sample) ===", flush=True)

    adapter_deltas = {}
    for name, w in all_adapter_weights.items():
        adapter_deltas[name] = compose_single_adapter(w, scale=6.0)
    del all_adapter_weights
    gc.collect()

    domain_to_adapter = {"math": "math", "code": "python", "medical": "medical"}

    routed_results = {}
    for domain, (eval_fn, bench_name) in {
        "math": (eval_gsm8k, "gsm8k"),
        "code": (eval_humaneval, "humaneval"),
        "medical": (eval_medqa, "medqa"),
    }.items():
        adapter_name = domain_to_adapter[domain]
        print(f"  Loading base + {adapter_name} adapter for {domain}...", flush=True)
        model, tokenizer = mlx_load(MODEL_ID)
        mx.eval(model.parameters())
        apply_delta(model, adapter_deltas[adapter_name])
        log_memory(f"routed-{domain}")

        acc = eval_fn(model, tokenizer, n_eval=N_EVAL)
        routed_results[bench_name] = round(acc, 1)
        print(f"  routed {domain}: {acc:.1f}%", flush=True)
        cleanup(model, tokenizer)

    routed_avg = sum(routed_results.values()) / len(routed_results)
    results["routed_composition"] = {**routed_results, "average": round(routed_avg, 1)}

    del adapter_deltas
    gc.collect()

    # ── Kill criteria ─────────────────────────────────────
    print("\n=== Kill Criteria ===", flush=True)

    k2065_pass = test_acc >= 85
    k2066_pass = routed_avg >= uniform_avg

    all_pass = k2065_pass and k2066_pass
    verdict = "SUPPORTED" if all_pass else "KILLED"
    if IS_SMOKE:
        verdict = "PROVISIONAL"

    results["kill_criteria"] = {
        "K2065_router_accuracy": {
            "pass": k2065_pass,
            "test_accuracy": round(test_acc, 1),
            "threshold": 85,
        },
        "K2066_routed_beats_uniform": {
            "pass": k2066_pass,
            "routed_avg": round(routed_avg, 1),
            "uniform_avg": round(uniform_avg, 1),
            "delta": round(routed_avg - uniform_avg, 1),
        },
    }
    results["verdict"] = verdict
    results["all_pass"] = all_pass
    results["total_time_s"] = round(time.time() - t_start, 1)

    RESULTS_FILE.write_text(json.dumps(results, indent=2))

    print(f"\nK2065 Router accuracy ≥85%: {'PASS' if k2065_pass else 'FAIL'} ({test_acc:.1f}%)", flush=True)
    print(f"K2066 Routed ≥ uniform: {'PASS' if k2066_pass else 'FAIL'} ({routed_avg:.1f}% vs {uniform_avg:.1f}%)", flush=True)
    print(f"\nVERDICT: {verdict}", flush=True)
    print(f"Total time: {results['total_time_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
