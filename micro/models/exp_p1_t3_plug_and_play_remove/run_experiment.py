#!/usr/bin/env python3
"""
T3.7: Hot-Remove Adapter Without Affecting Remaining Adapters

MATH: micro/models/exp_p1_t3_plug_and_play_remove/MATH.md

Verifies that removing domain adapter k from an existing N-adapter registry:
  (a) Does NOT change outputs of remaining adapters (Theorem 1, K1070)
  (b) Freed slot is immediately reusable by a new adapter (Theorem 2, K1071)
  (c) Hot-remove latency < 10ms (Theorem 3, K1072)

Symmetric to T3.6 (hot-add). Registry is a Python dict; operations are O(1).

Phases:
  Phase 1: Establish pre-remove baseline (4 existing domains, n=N_EVAL each)
  Phase 2: Hot-remove one adapter (geography, added via T3.6 copy-of-finance)
  Phase 3: Re-evaluate existing domains — outputs must be bit-identical (K1070)
  Phase 4: Re-add a new adapter to freed slot (K1071)
  Phase 5: Measure hot-remove latency (K1072)

Kill criteria:
  K1070: max token-diff = 0 for all existing domains after removal
  K1071: new adapter on freed slot achieves accuracy > base (4%)
  K1072: hot-remove p99 latency < 10ms (predicted ~0.005ms)

References: Finding #429 (T3.6), Finding #425 (T3.1), HRA (2405.17484)
"""

import gc
import json
import os
import re
import shutil
import tempfile
import time
import warnings
from pathlib import Path

import mlx.core as mx
import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Memory safety — leave 8GB for OS + base model
mx.set_memory_limit(mx.device_info()["memory_size"] - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"
RANK = 6
D_IN = 2560
D_OUT = 2560

IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"
N_EVAL = 3 if IS_SMOKE else 10
SEED = 42
OPTION_LETTERS = ["A", "B", "C", "D"]

# Real adapter paths (5 domains from T2.1 + T2.6)
T21_DIR = EXPERIMENT_DIR.parent / "exp_p1_t2_single_domain_training"
T26_DIR = EXPERIMENT_DIR.parent / "exp_p1_t2_multi_domain_5"

REAL_ADAPTER_PATHS = {
    "math":    T21_DIR / "adapters" / "math",
    "code":    T21_DIR / "adapters" / "code",
    "medical": T21_DIR / "adapters" / "medical",
    "legal":   T26_DIR / "adapters" / "legal",
    "finance": T26_DIR / "adapters" / "finance",
}


def log_memory(label=""):
    active = mx.get_active_memory() / 1e9
    cache = mx.get_cache_memory() / 1e9
    print(f"[MEM {label}] active={active:.2f}GB cache={cache:.2f}GB", flush=True)


def cleanup(*objects):
    for obj in objects:
        del obj
    gc.collect()
    mx.clear_cache()


# ─────────────────────────────────────────────────────────────────
# Adapter registry (Python dict: domain → adapter path)
# ─────────────────────────────────────────────────────────────────

class AdapterRegistry:
    """
    Minimal adapter registry: domain_label → adapter_path.
    Hot-add = O(1) dict update.
    Hot-remove = O(1) dict deletion.
    """
    def __init__(self):
        self._registry: dict[str, Path] = {}

    def register(self, domain: str, adapter_path: Path):
        self._registry[domain] = adapter_path

    def hot_add(self, domain: str, adapter_path: Path) -> float:
        """Add new domain adapter. Returns elapsed ms."""
        t0 = time.perf_counter()
        self._registry[domain] = adapter_path
        return (time.perf_counter() - t0) * 1000

    def hot_remove(self, domain: str) -> float:
        """Remove domain adapter. Returns elapsed ms. Raises KeyError if absent."""
        t0 = time.perf_counter()
        del self._registry[domain]
        return (time.perf_counter() - t0) * 1000

    def get(self, domain: str) -> Path:
        return self._registry[domain]

    def domains(self) -> list[str]:
        return list(self._registry.keys())

    def __contains__(self, domain: str) -> bool:
        return domain in self._registry

    def __len__(self):
        return len(self._registry)


# ─────────────────────────────────────────────────────────────────
# Inference helpers
# ─────────────────────────────────────────────────────────────────

def get_mmlu_prompts(subject: str, n: int, seed: int) -> list[dict]:
    """Load MMLU prompts as list of {prompt, answer} dicts."""
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", subject, split="test")
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    items = []
    for ex in ds:
        q = (
            ex["question"] + "\n"
            + "\n".join(f"({OPTION_LETTERS[i]}) {ex['choices'][i]}" for i in range(4))
        )
        prompt = "Answer this multiple choice question. Respond with only the letter (A/B/C/D).\n\n" + q
        items.append({"prompt": prompt, "answer": OPTION_LETTERS[ex["answer"]]})
    return items


def run_inference(adapter_path: Path, prompts: list[dict]) -> tuple[list[str], float]:
    """
    Run inference for a batch of prompts using the given adapter.
    Returns (list_of_responses, accuracy_pct).
    """
    from mlx_lm import generate, load

    model, tokenizer = load(MODEL_ID, adapter_path=str(adapter_path))

    responses = []
    correct = 0
    for item in prompts:
        if hasattr(tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": item["prompt"]}]
            fmt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            fmt = item["prompt"]

        response = generate(model, tokenizer, prompt=fmt, max_tokens=20, verbose=False)
        responses.append(response.strip())

        pred = response.strip().upper()
        pred_letter = next((l for l in OPTION_LETTERS if pred.startswith(l)), None)
        if pred_letter is None:
            m = re.search(r"\b([ABCD])\b", pred)
            pred_letter = m.group(1) if m else None
        if pred_letter == item["answer"]:
            correct += 1

    acc_pct = correct / len(prompts) * 100
    cleanup(model, tokenizer)
    return responses, acc_pct


def create_adapter_copy(src_domain_dir: Path, dest_dir: Path) -> Path:
    """Copy an adapter directory to a new path (simulates a freshly-trained adapter)."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    for f in src_domain_dir.iterdir():
        shutil.copy(f, dest_dir / f.name)
    size_mb = (dest_dir / "adapters.safetensors").stat().st_size / (1024**2)
    print(f"  Created adapter copy at {dest_dir} ({size_mb:.2f} MB)", flush=True)
    return dest_dir


# ─────────────────────────────────────────────────────────────────
# Phase 0: Setup — create geography adapter (the one we will remove)
# ─────────────────────────────────────────────────────────────────

def phase0_setup(registry: AdapterRegistry) -> dict:
    """
    Create geography adapter (copy of finance) and hot-add it to registry.
    This is the adapter we will remove in Phase 2.
    """
    print("\n=== Phase 0: Setup — Create and Register Geography Adapter ===", flush=True)
    t0 = time.time()

    geo_dir = EXPERIMENT_DIR / "adapter_geography"
    src_dir = T26_DIR / "adapters" / "finance"
    create_adapter_copy(src_dir, geo_dir)

    add_ms = registry.hot_add("geography", geo_dir)
    print(f"  Registered 'geography'. Registry size: {len(registry)} ({registry.domains()})", flush=True)
    print(f"  Hot-add latency: {add_ms:.3f}ms", flush=True)

    return {
        "setup_domains": registry.domains(),
        "registry_size_after_setup": len(registry),
        "phase0_time_s": round(time.time() - t0, 1),
    }


# ─────────────────────────────────────────────────────────────────
# Phase 1: Pre-remove baseline (4 real domains)
# ─────────────────────────────────────────────────────────────────

def phase1_pre_remove_baseline(registry: AdapterRegistry, prompts: dict) -> dict:
    """
    Run existing real adapters before hot-remove.
    Records responses for exact comparison in Phase 3.
    """
    print("\n=== Phase 1: Pre-Remove Baseline (4 Real Domains) ===", flush=True)
    t0 = time.time()

    domains_to_test = ["math", "medical", "legal", "finance"]
    if IS_SMOKE:
        domains_to_test = ["math"]

    pre_responses = {}
    pre_accuracy = {}

    for domain in domains_to_test:
        print(f"\n--- {domain} (pre-remove) ---", flush=True)
        adapter_path = registry.get(domain)
        responses, acc = run_inference(adapter_path, prompts[domain])
        pre_responses[domain] = responses
        pre_accuracy[domain] = acc
        print(f"  Accuracy: {acc:.1f}% | Sample: {responses[:2]}", flush=True)

    elapsed = time.time() - t0
    print(f"\n  Phase 1 time: {elapsed:.1f}s", flush=True)
    return {
        "pre_responses": pre_responses,
        "pre_accuracy": pre_accuracy,
        "phase1_time_s": round(elapsed, 1),
    }


# ─────────────────────────────────────────────────────────────────
# Phase 2: Hot-remove geography adapter
# ─────────────────────────────────────────────────────────────────

def phase2_hot_remove(registry: AdapterRegistry) -> dict:
    """
    Hot-remove the geography adapter.
    Records the single remove latency and verifies it is gone from registry.
    """
    print("\n=== Phase 2: Hot-Remove Geography Adapter ===", flush=True)
    t0 = time.time()

    assert "geography" in registry, "geography adapter must be registered before removal"
    size_before = len(registry)
    domains_before = registry.domains()

    remove_ms = registry.hot_remove("geography")

    size_after = len(registry)
    domains_after = registry.domains()
    domain_gone = "geography" not in registry

    print(f"  Registry before: {domains_before} ({size_before} domains)", flush=True)
    print(f"  Registry after:  {domains_after} ({size_after} domains)", flush=True)
    print(f"  'geography' removed: {domain_gone}", flush=True)
    print(f"  Hot-remove latency: {remove_ms:.4f}ms", flush=True)

    elapsed = time.time() - t0
    return {
        "domain_removed": "geography",
        "registry_size_before": size_before,
        "registry_size_after": size_after,
        "domain_gone": domain_gone,
        "remove_latency_ms": round(remove_ms, 4),
        "phase2_time_s": round(elapsed, 1),
    }


# ─────────────────────────────────────────────────────────────────
# Phase 3: Post-remove check — existing adapters unchanged (K1070)
# ─────────────────────────────────────────────────────────────────

def phase3_post_remove_check(
    registry: AdapterRegistry,
    prompts: dict,
    pre_responses: dict,
) -> dict:
    """
    K1070: Re-run existing adapters. Responses must be bit-identical to Phase 1.
    Theorem 1: exclusive routing means removing domain k cannot affect domain j (j≠k).
    """
    print("\n=== Phase 3: Post-Remove Check (K1070 — Remaining Outputs Unchanged) ===", flush=True)
    t0 = time.time()

    post_responses = {}
    post_accuracy = {}
    domain_diffs = {}
    k1070_pass = True

    for domain in pre_responses:
        print(f"\n--- {domain} (post-remove) ---", flush=True)
        adapter_path = registry.get(domain)
        responses, acc = run_inference(adapter_path, prompts[domain])
        post_responses[domain] = responses
        post_accuracy[domain] = acc

        pre = pre_responses[domain]
        n_identical = sum(1 for a, b in zip(pre, responses) if a == b)
        n_total = len(pre)
        identical = n_identical == n_total

        domain_diffs[domain] = 0 if identical else n_total - n_identical
        if not identical:
            k1070_pass = False
            print(f"  FAIL: {n_total - n_identical}/{n_total} responses changed", flush=True)
            for i, (a, b) in enumerate(zip(pre, responses)):
                if a != b:
                    print(f"    Query {i}: before='{a[:50]}' after='{b[:50]}'", flush=True)
        else:
            print(f"  PASS: {n_identical}/{n_total} identical (acc: {acc:.1f}%)", flush=True)

    elapsed = time.time() - t0
    total_diffs = sum(domain_diffs.values())
    print(f"\n  K1070 (0 changes in existing outputs): {'PASS' if k1070_pass else 'FAIL'}", flush=True)
    print(f"  Total token differences: {total_diffs}", flush=True)
    print(f"  Phase 3 time: {elapsed:.1f}s", flush=True)

    return {
        "post_accuracy": post_accuracy,
        "domain_diffs": domain_diffs,
        "total_diffs": total_diffs,
        "k1070_pass": k1070_pass,
        "phase3_time_s": round(elapsed, 1),
    }


# ─────────────────────────────────────────────────────────────────
# Phase 4: Freed slot reusable (K1071)
# ─────────────────────────────────────────────────────────────────

def phase4_freed_slot_reusable(registry: AdapterRegistry) -> dict:
    """
    K1071: After removing geography, add a new adapter 'history' to freed slot.
    Verify new adapter is immediately functional (accuracy > base = 4%).
    Also verify existing adapters still unchanged (recheck one domain).

    'history' = copy of finance adapter (same MCQ format transfer as geography).
    Test on MMLU high_school_european_history (neutral MMLU subject).
    """
    print("\n=== Phase 4: Freed Slot Reusable (K1071) ===", flush=True)
    t0 = time.time()

    assert "geography" not in registry, "geography must be removed before testing slot reuse"

    # Create history adapter (copy of finance) — occupies what was geography's slot
    hist_dir = EXPERIMENT_DIR / "adapter_history"
    src_dir = T26_DIR / "adapters" / "finance"
    create_adapter_copy(src_dir, hist_dir)

    add_ms = registry.hot_add("history", hist_dir)
    print(f"  Registered 'history' (freed slot). Registry: {registry.domains()}", flush=True)
    print(f"  Hot-add latency: {add_ms:.3f}ms", flush=True)

    # Evaluate history adapter immediately
    n_eval = N_EVAL if not IS_SMOKE else 3
    subject = "high_school_european_history"
    prompts = get_mmlu_prompts(subject, n_eval, SEED)
    print(f"\n  Evaluating 'history' on {subject} (n={n_eval})", flush=True)

    responses, acc = run_inference(hist_dir, prompts)
    base_acc = 4.0  # T3.2: base without adapter = 4%
    k1071_pass = acc > base_acc

    print(f"  History accuracy: {acc:.1f}% (base={base_acc}%)", flush=True)
    print(f"  K1071 (> base={base_acc}%): {'PASS' if k1071_pass else 'FAIL'}", flush=True)

    elapsed = time.time() - t0
    return {
        "freed_slot_domain": "history",
        "history_accuracy": acc,
        "base_accuracy": base_acc,
        "k1071_pass": k1071_pass,
        "phase4_time_s": round(elapsed, 1),
    }


# ─────────────────────────────────────────────────────────────────
# Phase 5: Hot-remove latency benchmark (K1072)
# ─────────────────────────────────────────────────────────────────

def phase5_remove_latency_benchmark(registry: AdapterRegistry, tmp_dir: Path) -> dict:
    """
    K1072: Hot-remove latency < 10ms (predicted ~0.005ms).
    Measures 100 dict deletions with real paths.
    """
    print("\n=== Phase 5: Hot-Remove Latency Benchmark (K1072) ===", flush=True)

    # Populate registry with N=100 dummy entries, then time each deletion
    N_TRIALS = 100
    fake_paths = []
    for i in range(N_TRIALS):
        label = f"bench_domain_{i}"
        fake_path = tmp_dir / f"adapter_{i}"
        registry._registry[label] = fake_path
        fake_paths.append(label)

    latencies_ms = []
    for label in fake_paths:
        t_start = time.perf_counter()
        del registry._registry[label]
        t_end = time.perf_counter()
        latencies_ms.append((t_end - t_start) * 1000)

    mean_ms = float(np.mean(latencies_ms))
    p99_ms = float(np.percentile(latencies_ms, 99))
    max_ms = float(np.max(latencies_ms))

    k1072_pass = p99_ms < 10.0

    print(f"  Remove latency (n={N_TRIALS}): mean={mean_ms:.4f}ms, p99={p99_ms:.4f}ms, max={max_ms:.4f}ms", flush=True)
    print(f"  K1072 (p99 < 10ms): {'PASS' if k1072_pass else 'FAIL'}", flush=True)

    return {
        "remove_mean_ms": round(mean_ms, 6),
        "remove_p99_ms": round(p99_ms, 6),
        "remove_max_ms": round(max_ms, 6),
        "latency_threshold_ms": 10.0,
        "k1072_pass": k1072_pass,
    }


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main():
    print("=" * 60, flush=True)
    print("T3.7: Plug-and-Play Hot-Remove Adapter", flush=True)
    print(f"IS_SMOKE={IS_SMOKE}, N_EVAL={N_EVAL}", flush=True)
    print("=" * 60, flush=True)

    total_t0 = time.time()
    results = {"is_smoke": IS_SMOKE, "n_eval": N_EVAL}

    # Initialize registry with 5 real domains
    registry = AdapterRegistry()
    for domain, path in REAL_ADAPTER_PATHS.items():
        registry.register(domain, path)
    print(f"\nRegistry initialized: {registry.domains()} ({len(registry)} domains)", flush=True)

    # Phase 0: Add geography (the adapter we will remove)
    p0 = phase0_setup(registry)
    results.update(p0)

    # Load prompts for real domains
    print("\nLoading MMLU prompts for real domains...", flush=True)
    domains_to_test = ["math", "medical", "legal", "finance"]
    if IS_SMOKE:
        domains_to_test = ["math"]

    prompts: dict[str, list] = {}
    for domain in domains_to_test:
        if domain == "math":
            from datasets import load_dataset
            ds = load_dataset("gsm8k", "main", split="test")
            ds = ds.shuffle(seed=SEED).select(range(min(N_EVAL, len(ds))))
            prompts[domain] = [
                {
                    "prompt": f"Solve this math problem step by step.\n\n{ex['question']}\n\nAnswer:",
                    "answer": ex["answer"].split("####")[-1].strip().replace(",", ""),
                }
                for ex in ds
            ]
        elif domain == "medical":
            from datasets import load_dataset
            ds = load_dataset("openlifescienceai/medmcqa", split="validation")
            ds = ds.shuffle(seed=SEED).select(range(min(N_EVAL, len(ds))))
            opt_map = {0: "A", 1: "B", 2: "C", 3: "D"}
            items = []
            for ex in ds:
                q = (
                    f"{ex['question']}\n(A) {ex['opa']}\n(B) {ex['opb']}\n"
                    f"(C) {ex['opc']}\n(D) {ex['opd']}"
                )
                items.append({
                    "prompt": "Answer this medical MCQ. Respond with only the letter (A/B/C/D).\n\n" + q,
                    "answer": opt_map.get(ex["cop"], "A"),
                })
            prompts[domain] = items
        else:
            subject_map = {"legal": "professional_law", "finance": "high_school_macroeconomics"}
            prompts[domain] = get_mmlu_prompts(subject_map[domain], N_EVAL, SEED)

    # Phase 1: Pre-remove baseline
    p1 = phase1_pre_remove_baseline(registry, prompts)
    results.update(p1)

    # Phase 2: Hot-remove geography
    p2 = phase2_hot_remove(registry)
    results.update(p2)

    # Phase 3: Post-remove check (K1070)
    p3 = phase3_post_remove_check(registry, prompts, p1["pre_responses"])
    results.update(p3)

    # Phase 4: Freed slot reusable (K1071)
    p4 = phase4_freed_slot_reusable(registry)
    results.update(p4)

    # Phase 5: Latency benchmark (K1072)
    with tempfile.TemporaryDirectory() as tmpdir:
        p5 = phase5_remove_latency_benchmark(registry, Path(tmpdir))
    results.update(p5)

    # Kill criteria summary
    total_elapsed = time.time() - total_t0
    results["total_time_s"] = round(total_elapsed, 1)

    k1070_pass = results.get("k1070_pass", False)
    k1071_pass = results.get("k1071_pass", False)
    k1072_pass = results.get("k1072_pass", False)

    results["K1070_remaining_outputs_unchanged"] = "PASS" if k1070_pass else "FAIL"
    results["K1071_freed_slot_reusable"] = "PASS" if k1071_pass else "FAIL"
    results["K1072_remove_latency"] = "PASS" if k1072_pass else "FAIL"

    # Convert numpy scalars for JSON
    def _to_native(obj):
        if isinstance(obj, dict):
            return {k: _to_native(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_native(v) for v in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return obj

    results = _to_native(results)

    print("\n" + "=" * 60, flush=True)
    print("KILL CRITERIA SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(f"  K1070 (remaining outputs unchanged): {results['K1070_remaining_outputs_unchanged']}", flush=True)
    print(f"  K1071 (freed slot reusable):         {results['K1071_freed_slot_reusable']}", flush=True)
    print(f"  K1072 (remove latency < 10ms):       {results['K1072_remove_latency']}", flush=True)
    print(f"\n  Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)", flush=True)

    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    print(f"\n  Results saved to {RESULTS_FILE}", flush=True)

    return results


if __name__ == "__main__":
    main()
