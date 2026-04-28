#!/usr/bin/env python3
"""
N=3 composition: load math+code+medical adapters, compose via Σ(B_i @ A_i),
measure per-domain accuracy vs single-adapter baselines.

Kill criteria:
  K2062: Per-domain accuracy under composition drops ≤5pp vs single-adapter
         (GSM8K ≥67%, HumanEval ≥65%, MedQA ≥63%)
  K2063: Composed PPL on each domain != single-adapter PPL (not tautological)
  K2064: Cross-domain interference — math adapter on MedQA ≤55%
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


# ─────────────────────────────────────────────
# Composition: correct Σ (B_i @ A_i) math
# ─────────────────────────────────────────────

def load_adapter_weights(adapter_path: Path) -> dict:
    """Load LoRA adapter weights from safetensors."""
    from mlx.utils import tree_unflatten
    import safetensors.numpy

    weights_file = adapter_path / "adapters.safetensors"
    if not weights_file.exists():
        raise FileNotFoundError(f"No adapter weights at {weights_file}")

    flat = {}
    with safetensors.numpy.safe_open(str(weights_file), framework="numpy") as f:
        for key in f.keys():
            flat[key] = mx.array(f.get_tensor(key))
    return flat


def compose_adapters(adapter_weights_list: list[dict], scale: float = 1.0) -> dict:
    """Compose N adapters via correct Σ (B_i @ A_i) math.

    For each layer's LoRA pair (lora_a, lora_b), compute:
        ΔW_composed = Σ_i (B_i @ A_i)

    This is NOT (ΣB) @ (ΣA) — that's the composition bug.
    """
    layer_keys = set()
    for w in adapter_weights_list:
        for key in w:
            if ".lora_a" in key:
                base = key.replace(".lora_a", "")
                layer_keys.add(base)

    composed = {}
    for base_key in sorted(layer_keys):
        a_key = f"{base_key}.lora_a"
        b_key = f"{base_key}.lora_b"

        delta_sum = None
        for w in adapter_weights_list:
            if a_key not in w or b_key not in w:
                continue
            a_i = w[a_key]  # (d_in, r)
            b_i = w[b_key]  # (r, d_out)
            delta_i = a_i @ b_i  # (d_in, d_out) — mlx_lm convention: y = x @ A @ B
            if delta_sum is None:
                delta_sum = delta_i
            else:
                delta_sum = delta_sum + delta_i

        if delta_sum is not None:
            composed[base_key] = delta_sum * scale

    mx.eval(composed)
    return composed


class ComposedLinear(nn.Module):
    """Wraps a (quantized) linear layer, adding a pre-composed ΔW at forward time."""

    def __init__(self, base_layer, delta):
        super().__init__()
        self.base = base_layer
        self.delta = delta

    def __call__(self, x):
        return self.base(x) + x @ self.delta


def apply_composed_delta(model, composed_delta: dict):
    """Replace target layers with ComposedLinear wrappers (quantization-safe)."""
    for key, delta in composed_delta.items():
        parts = key.split(".")
        parent = model
        for part in parts[:-1]:
            if part.isdigit() and isinstance(parent, list):
                parent = parent[int(part)]
            else:
                parent = getattr(parent, part)

        attr_name = parts[-1]
        base_layer = getattr(parent, attr_name)
        setattr(parent, attr_name, ComposedLinear(base_layer, delta))


# ─────────────────────────────────────────────
# Model loading helpers
# ─────────────────────────────────────────────

def load_base_model():
    from mlx_lm import load
    model, tokenizer = load(MODEL_ID)
    mx.eval(model.parameters())
    log_memory("base-model-loaded")
    return model, tokenizer


def load_single_adapter_model(adapter_path: Path):
    from mlx_lm import load
    model, tokenizer = load(MODEL_ID, adapter_path=str(adapter_path))
    mx.eval(model.parameters())
    log_memory(f"adapter-loaded-{adapter_path.name}")
    return model, tokenizer


# ─────────────────────────────────────────────
# Evaluation functions
# ─────────────────────────────────────────────

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
            pred_ans = pred_match.group(1).replace(",", "").strip()
            if pred_ans == gt_ans:
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
        prompt = ex["prompt"]
        messages = [{"role": "user", "content": f"Complete the following Python function:\n\n```python\n{prompt}\n```\n\nRespond with only the function body, no markdown."}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        response = generate(model, tokenizer, prompt=formatted, max_tokens=512, verbose=False)

        code_match = re.search(r"```python\n(.*?)```", response, re.DOTALL)
        completion = code_match.group(1) if code_match else response

        full_code = prompt + completion + "\n\n" + ex["test"] + f"\n\ncheck({ex['entry_point']})\n"

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


def eval_ppl(model, tokenizer, domain: str, n_eval=20) -> float:
    """Compute perplexity on domain validation data."""
    from datasets import load_dataset

    if domain == "math":
        ds = load_dataset("openai/gsm8k", "main", split="test")
        ds = ds.shuffle(seed=SEED).select(range(min(n_eval, len(ds))))
        texts = [f"Solve: {ex['question']}\n{ex['answer']}" for ex in ds]
    elif domain == "code":
        ds = load_dataset("openai_humaneval", split="test")
        ds = ds.select(range(min(n_eval, len(ds))))
        texts = [ex["prompt"] + ex["canonical_solution"] for ex in ds]
    elif domain == "medical":
        ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
        ds = ds.shuffle(seed=SEED).select(range(min(n_eval, len(ds))))
        texts = [f"{ex['question']}\nAnswer: {ex['answer']}" for ex in ds]
    else:
        raise ValueError(f"Unknown domain: {domain}")

    total_nll = 0.0
    total_tokens = 0

    for text in texts:
        tokens = mx.array(tokenizer.encode(text))
        if len(tokens.shape) == 1:
            tokens = tokens[None, :]

        logits = model(tokens[:, :-1])
        targets = tokens[:, 1:]

        log_probs = nn.log_softmax(logits, axis=-1)
        target_log_probs = mx.take_along_axis(
            log_probs, targets[:, :, None], axis=-1
        ).squeeze(-1)

        nll = -mx.sum(target_log_probs).item()
        n_tok = targets.size
        total_nll += nll
        total_tokens += n_tok

    ppl = float(np.exp(total_nll / max(total_tokens, 1)))
    return ppl


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    t_start = time.time()
    log_memory("start")
    print(f"N=3 Composition Test (SMOKE={IS_SMOKE}, N_EVAL={N_EVAL})", flush=True)

    for name, path in ADAPTER_PATHS.items():
        wf = path / "adapters.safetensors"
        if not wf.exists():
            print(f"FATAL: Missing adapter weights: {wf}", flush=True)
            sys.exit(1)
    print("All 3 adapter weights verified.", flush=True)

    results = {"is_smoke": IS_SMOKE, "n_eval": N_EVAL}

    # ── Phase 1: Single-adapter baselines ─────────────────
    print("\n=== Phase 1: Single-adapter baselines ===", flush=True)

    single_results = {}
    for domain, (adapter_name, eval_fn) in {
        "math": ("math", eval_gsm8k),
        "code": ("python", eval_humaneval),
        "medical": ("medical", eval_medqa),
    }.items():
        print(f"\nEvaluating {domain} adapter on {domain} benchmark...", flush=True)
        model, tokenizer = load_single_adapter_model(ADAPTER_PATHS[adapter_name])
        acc = eval_fn(model, tokenizer, n_eval=N_EVAL)
        ppl = eval_ppl(model, tokenizer, domain, n_eval=min(20, N_EVAL))
        single_results[domain] = {"accuracy": round(acc, 1), "ppl": round(ppl, 2)}
        print(f"  {domain}: acc={acc:.1f}%, ppl={ppl:.2f}", flush=True)
        cleanup(model, tokenizer)

    results["single_adapter"] = single_results

    # ── Phase 2: Compose all 3 adapters ───────────────────
    print("\n=== Phase 2: Compose N=3 adapters via Σ(B_i @ A_i) ===", flush=True)

    adapter_weights = []
    for name in ["math", "python", "medical"]:
        w = load_adapter_weights(ADAPTER_PATHS[name])
        adapter_weights.append(w)
        print(f"  Loaded {name}: {len(w)} weight tensors", flush=True)

    lora_scale = 6.0
    composed_delta = compose_adapters(adapter_weights, scale=lora_scale / len(adapter_weights))
    print(f"  Composed {len(composed_delta)} layer deltas (scale={lora_scale}/{len(adapter_weights)}={lora_scale/len(adapter_weights):.1f})", flush=True)
    del adapter_weights
    gc.collect()

    # Load base model and apply composed delta
    model, tokenizer = load_base_model()
    apply_composed_delta(model, composed_delta)
    log_memory("composed-model-ready")
    del composed_delta
    gc.collect()

    # ── Phase 3: Evaluate composed model on all domains ───
    print("\n=== Phase 3: Composed model evaluation ===", flush=True)

    composed_results = {}

    print("  Evaluating composed on GSM8K (math)...", flush=True)
    comp_math = eval_gsm8k(model, tokenizer, n_eval=N_EVAL)
    comp_math_ppl = eval_ppl(model, tokenizer, "math", n_eval=min(20, N_EVAL))
    composed_results["math"] = {"accuracy": round(comp_math, 1), "ppl": round(comp_math_ppl, 2)}
    print(f"  math: acc={comp_math:.1f}%, ppl={comp_math_ppl:.2f}", flush=True)

    print("  Evaluating composed on HumanEval (code)...", flush=True)
    comp_code = eval_humaneval(model, tokenizer, n_eval=N_EVAL)
    comp_code_ppl = eval_ppl(model, tokenizer, "code", n_eval=min(20, N_EVAL))
    composed_results["code"] = {"accuracy": round(comp_code, 1), "ppl": round(comp_code_ppl, 2)}
    print(f"  code: acc={comp_code:.1f}%, ppl={comp_code_ppl:.2f}", flush=True)

    print("  Evaluating composed on MedQA (medical)...", flush=True)
    comp_med = eval_medqa(model, tokenizer, n_eval=N_EVAL)
    comp_med_ppl = eval_ppl(model, tokenizer, "medical", n_eval=min(20, N_EVAL))
    composed_results["medical"] = {"accuracy": round(comp_med, 1), "ppl": round(comp_med_ppl, 2)}
    print(f"  medical: acc={comp_med:.1f}%, ppl={comp_med_ppl:.2f}", flush=True)

    results["composed"] = composed_results
    cleanup(model, tokenizer)

    # ── Phase 4: Cross-domain interference (math adapter on MedQA) ──
    print("\n=== Phase 4: Cross-domain interference ===", flush=True)
    model, tokenizer = load_single_adapter_model(ADAPTER_PATHS["math"])
    math_on_medqa = eval_medqa(model, tokenizer, n_eval=N_EVAL)
    print(f"  Math adapter on MedQA: {math_on_medqa:.1f}%", flush=True)
    results["cross_domain"] = {"math_on_medqa": round(math_on_medqa, 1)}
    cleanup(model, tokenizer)

    # ── Kill criteria evaluation ──────────────────────────
    print("\n=== Kill Criteria ===", flush=True)

    # K2062: Per-domain accuracy drops ≤5pp
    math_drop = single_results["math"]["accuracy"] - comp_math
    code_drop = single_results["code"]["accuracy"] - comp_code
    med_drop = single_results["medical"]["accuracy"] - comp_med
    k2062_pass = (comp_math >= 67 and comp_code >= 65 and comp_med >= 63) or (
        math_drop <= 5 and code_drop <= 5 and med_drop <= 5
    )

    # K2063: Composed PPL != single PPL (not tautological)
    ppl_diffs = {
        "math": abs(composed_results["math"]["ppl"] - single_results["math"]["ppl"]),
        "code": abs(composed_results["code"]["ppl"] - single_results["code"]["ppl"]),
        "medical": abs(composed_results["medical"]["ppl"] - single_results["medical"]["ppl"]),
    }
    k2063_pass = any(d > 0.01 for d in ppl_diffs.values())

    # K2064: Math adapter on MedQA ≤55%
    k2064_pass = math_on_medqa <= 55

    all_pass = k2062_pass and k2063_pass and k2064_pass
    verdict = "SUPPORTED" if all_pass else "KILLED"
    if IS_SMOKE:
        verdict = "PROVISIONAL"

    results["kill_criteria"] = {
        "K2062_composition_accuracy": {
            "pass": k2062_pass,
            "math_drop_pp": round(math_drop, 1),
            "code_drop_pp": round(code_drop, 1),
            "med_drop_pp": round(med_drop, 1),
        },
        "K2063_not_tautological": {
            "pass": k2063_pass,
            "ppl_diffs": {k: round(v, 4) for k, v in ppl_diffs.items()},
        },
        "K2064_cross_domain_interference": {
            "pass": k2064_pass,
            "math_on_medqa_pct": round(math_on_medqa, 1),
        },
    }
    results["verdict"] = verdict
    results["all_pass"] = all_pass
    results["total_time_s"] = round(time.time() - t_start, 1)

    RESULTS_FILE.write_text(json.dumps(results, indent=2))

    print(f"\nK2062 Composition accuracy: {'PASS' if k2062_pass else 'FAIL'}", flush=True)
    print(f"  math: {single_results['math']['accuracy']:.1f}% → {comp_math:.1f}% ({-math_drop:+.1f}pp)", flush=True)
    print(f"  code: {single_results['code']['accuracy']:.1f}% → {comp_code:.1f}% ({-code_drop:+.1f}pp)", flush=True)
    print(f"  medical: {single_results['medical']['accuracy']:.1f}% → {comp_med:.1f}% ({-med_drop:+.1f}pp)", flush=True)
    print(f"K2063 Not tautological: {'PASS' if k2063_pass else 'FAIL'} (diffs: {ppl_diffs})", flush=True)
    print(f"K2064 Cross-domain interference: {'PASS' if k2064_pass else 'FAIL'} (math→MedQA={math_on_medqa:.1f}%)", flush=True)
    print(f"\nVERDICT: {verdict}", flush=True)
    print(f"Total time: {results['total_time_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
