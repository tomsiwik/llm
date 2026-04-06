#!/usr/bin/env python3
"""Pierre Pro: THE BIG QUESTION — does MMLU degrade under composition on fp16/4-bit base?

This is the most important experiment in the project.

On BitNet-2B (ternary): MMLU degrades -5 to -6pp under composition (#263).
Root cause hypothesis: ternary flat spectrum (#272).

If degradation < 2pp on Qwen3-4B → ternary was the problem → Pierre Pro is viable.
If degradation > 4pp → composition itself is the problem → need different math.

Kill criteria:
  K814: MMLU degradation > 8pp (worse than ternary)
  K815: Single-adapter MMLU < base model MMLU
"""

import gc
import json
import math
import os
import time
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

device_info = mx.device_info()
mx.set_memory_limit(device_info["memory_size"] - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

from mlx_lm import load
from pierre import compose_adapters

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

BASE_DIR = EXPERIMENT_DIR.parent / "pro_base_validation"
INIT_DIR = EXPERIMENT_DIR.parent / "pro_grassmannian_init"
ADAPTER_DIR = EXPERIMENT_DIR.parent / "pro_sft_5_adapters" / "adapters"

LORA_RANK = 16
LORA_SCALE = 20.0
MAX_SEQ = 512
SEED = 42
DOMAINS = ["medical", "code", "math", "legal", "finance"]

TARGET_KEYS = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
    "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
]


class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.bool_,)): return bool(o)
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return super().default(o)

def log(m): print(m, flush=True)
def cleanup(*o):
    for x in o: del x
    gc.collect(); mx.clear_cache(); mx.reset_peak_memory()


# ── LoRA attachment (same as pro_sft_5_adapters) ─────────────────────────

class LoRALinear(nn.Module):
    def __init__(self, base_module, rank=16, scale=20.0, a_init=None):
        super().__init__()
        self.base = base_module
        in_f = base_module.in_features if hasattr(base_module, 'in_features') else base_module.weight.shape[-1]
        out_f = base_module.out_features if hasattr(base_module, 'out_features') else base_module.weight.shape[0]
        self.lora_a = a_init if a_init is not None else mx.random.normal(shape=(in_f, rank)) * (1.0/math.sqrt(in_f))
        self.lora_b = mx.zeros((rank, out_f))
        self.scale = scale
        self.base.freeze()
        self.freeze(keys=["base", "lora_a"], strict=False)

    def __call__(self, x):
        base_out = self.base(x)
        return base_out + ((x @ self.lora_a) @ self.lora_b * self.scale).astype(base_out.dtype)


def attach_adapter(model, skeleton, adapter_b, domain_idx, scale):
    count = 0
    for li, layer in enumerate(model.model.layers):
        updates = []
        for key in TARGET_KEYS:
            bk = f"model.layers.{li}.{key}.lora_b"
            ak = f"layer_{li}_{key}_domain_{domain_idx}"
            if bk not in adapter_b or ak not in skeleton: continue
            m = layer
            for part in key.split("."): m = getattr(m, part, None)
            if m is None: continue
            A = mx.array(skeleton[ak]).astype(mx.bfloat16)
            lora = LoRALinear(m, rank=LORA_RANK, scale=scale, a_init=A)
            # Load trained B weights
            lora.lora_b = adapter_b[bk].astype(mx.bfloat16)
            updates.append((key, lora)); count += 1
        if updates: layer.update_modules(tree_unflatten(updates))
    mx.eval(model.parameters())
    return count


# ── MMLU evaluation ──────────────────────────────────────────────────────

# Larger MMLU subset for statistical significance
MMLU_QUESTIONS = {
    "abstract_algebra": [
        ("Find the degree of the extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q.", "A) 4\nB) 2\nC) 6\nD) 3", "A"),
        ("Statement 1: Every free abelian group is torsion free. Statement 2: Every finitely generated torsion-free abelian group is a free abelian group.", "A) True, True\nB) False, False\nC) True, False\nD) False, True", "A"),
        ("Find all c in Z_3 such that Z_3[x]/(x^2 + c) is a field.", "A) 0\nB) 1\nC) 2\nD) 3", "B"),
    ],
    "anatomy": [
        ("Which of the following is NOT a function of the liver?", "A) Glycogen storage\nB) Production of insulin\nC) Bile production\nD) Detoxification", "B"),
        ("The longest bone in the human body is the:", "A) Humerus\nB) Tibia\nC) Femur\nD) Fibula", "C"),
        ("Which cranial nerve controls the muscles of mastication?", "A) Facial\nB) Trigeminal\nC) Vagus\nD) Glossopharyngeal", "B"),
    ],
    "computer_science": [
        ("Which of the following is true about quicksort time complexity?", "A) O(n) average\nB) O(n log n) average\nC) O(n^2) average\nD) O(log n) average", "B"),
        ("In a binary search tree, what is worst-case search time?", "A) O(1)\nB) O(log n)\nC) O(n)\nD) O(n log n)", "C"),
        ("Which data structure uses FIFO?", "A) Stack\nB) Queue\nC) Tree\nD) Graph", "B"),
    ],
    "high_school_mathematics": [
        ("If f(x) = 2x + 3, what is f(f(1))?", "A) 7\nB) 13\nC) 11\nD) 9", "B"),
        ("What is the derivative of x^3 + 2x?", "A) 3x^2 + 2\nB) x^2 + 2\nC) 3x + 2\nD) 3x^2", "A"),
        ("What is the sum of the first 10 positive integers?", "A) 45\nB) 55\nC) 50\nD) 65", "B"),
    ],
    "clinical_knowledge": [
        ("Which of the following is the most common cause of dementia?", "A) Vascular dementia\nB) Lewy body dementia\nC) Alzheimer's disease\nD) Frontotemporal dementia", "C"),
        ("Normal range for fasting blood glucose is:", "A) 40-70 mg/dL\nB) 70-100 mg/dL\nC) 100-126 mg/dL\nD) 126-200 mg/dL", "B"),
    ],
    "jurisprudence": [
        ("Which legal philosopher is associated with the 'command theory' of law?", "A) H.L.A. Hart\nB) John Austin\nC) Ronald Dworkin\nD) Lon Fuller", "B"),
    ],
}


def eval_mmlu(model, tokenizer):
    total_correct = 0
    total_questions = 0
    per_subject = {}

    for subject, questions in MMLU_QUESTIONS.items():
        correct = 0
        for question, choices, answer in questions:
            prompt = f"Question: {question}\n{choices}\nAnswer: The correct answer is"
            tokens = tokenizer.encode(prompt)[:MAX_SEQ]
            x = mx.array(tokens)[None, :]
            logits = model(x)
            mx.eval(logits)

            last_logits = logits[0, -1]
            answer_tokens = {}
            for letter in ["A", "B", "C", "D"]:
                toks = tokenizer.encode(f" {letter}")
                answer_tokens[letter] = toks[0] if toks else tokenizer.encode(letter)[0]

            answer_logits = {k: last_logits[v].item() for k, v in answer_tokens.items()}
            predicted = max(answer_logits, key=answer_logits.get)

            if predicted == answer:
                correct += 1
            total_questions += 1
            del logits, x

        total_correct += correct
        per_subject[subject] = {"correct": correct, "total": len(questions),
                                "accuracy": round(correct/len(questions), 3)}

    return total_correct, total_questions, per_subject


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    log("Pierre Pro: MMLU Composition Test (THE BIG QUESTION)")
    log("=" * 60)
    mx.random.seed(SEED)

    # Load config
    base_data = json.loads((BASE_DIR / "results.json").read_text()) if (BASE_DIR / "results.json").exists() else {}
    model_id = base_data.get("model_id", "mlx-community/Qwen2.5-3B-Instruct-4bit")
    base_mmlu = base_data.get("mmlu", {}).get("accuracy", 0)

    skeleton_path = INIT_DIR / "grassmannian_skeleton_n5.npz"
    skeleton = dict(np.load(str(skeleton_path)))

    results = {"model_id": model_id, "base_mmlu_from_phase1": base_mmlu}

    # Phase 1: Re-measure base MMLU (for consistency)
    log("\n=== Phase 1: Base model MMLU ===")
    model, tokenizer = load(model_id)
    base_correct, base_total, base_subjects = eval_mmlu(model, tokenizer)
    base_acc = base_correct / base_total if base_total else 0
    log(f"  Base MMLU: {base_acc:.1%} ({base_correct}/{base_total})")
    results["base_mmlu"] = round(base_acc, 4)
    cleanup(model, tokenizer)

    # Phase 2: Single-adapter MMLU (each domain separately)
    log("\n=== Phase 2: Single-adapter MMLU ===")
    single_mmlu = {}
    for di, d in enumerate(DOMAINS):
        adapter_path = ADAPTER_DIR / d / "adapter.npz"
        if not adapter_path.exists():
            log(f"  SKIP {d}: no adapter")
            continue

        model, tokenizer = load(model_id)
        adapter_b = dict(mx.load(str(adapter_path)))
        n = attach_adapter(model, skeleton, adapter_b, di, LORA_SCALE)
        correct, total, subjects = eval_mmlu(model, tokenizer)
        acc = correct / total if total else 0
        single_mmlu[d] = round(acc, 4)
        deg = (acc - base_acc) * 100
        log(f"  {d}: {acc:.1%} ({deg:+.1f}pp vs base, {n} modules)")
        cleanup(model, tokenizer, adapter_b)

    results["single_adapter_mmlu"] = single_mmlu

    # Phase 3: Composed N=5 MMLU (THE KEY MEASUREMENT)
    log("\n=== Phase 3: Composed N=5 MMLU (THE BIG QUESTION) ===")
    all_adapters = []
    for d in DOMAINS:
        adapter_path = ADAPTER_DIR / d / "adapter.npz"
        if adapter_path.exists():
            all_adapters.append(dict(mx.load(str(adapter_path))))

    if len(all_adapters) >= 2:
        composed = compose_adapters(all_adapters)
        model, tokenizer = load(model_id)
        n = attach_adapter(model, skeleton, composed, 0, LORA_SCALE)
        comp_correct, comp_total, comp_subjects = eval_mmlu(model, tokenizer)
        comp_acc = comp_correct / comp_total if comp_total else 0
        comp_deg = (comp_acc - base_acc) * 100

        log(f"  Composed MMLU: {comp_acc:.1%} ({comp_deg:+.1f}pp vs base)")
        log(f"  Base: {base_acc:.1%} | Composed: {comp_acc:.1%} | Degradation: {comp_deg:+.1f}pp")

        results["composed_mmlu"] = round(comp_acc, 4)
        results["composition_degradation_pp"] = round(comp_deg, 2)
        results["composed_per_subject"] = comp_subjects
        cleanup(model, tokenizer, composed)
    else:
        log("  ERROR: fewer than 2 adapters available")
        results["composed_mmlu"] = 0
        results["composition_degradation_pp"] = -100

    # Phase 4: Compare against BitNet results
    log("\n=== Phase 4: Comparison ===")
    bitnet_degradation = -5.5  # from Finding #263
    pro_degradation = results.get("composition_degradation_pp", -100)

    log(f"  BitNet-2B composition degradation: {bitnet_degradation:+.1f}pp")
    log(f"  Qwen3 composition degradation:     {pro_degradation:+.1f}pp")

    if abs(pro_degradation) < abs(bitnet_degradation):
        log(f"  → TERNARY WAS THE BOTTLENECK. Pro degrades {abs(pro_degradation):.1f}pp vs BitNet's {abs(bitnet_degradation):.1f}pp.")
    else:
        log(f"  → COMPOSITION IS THE BOTTLENECK. Both degrade similarly.")

    results["bitnet_degradation_pp"] = bitnet_degradation
    results["ternary_was_bottleneck"] = abs(pro_degradation) < abs(bitnet_degradation) * 0.5
    results["total_time_s"] = round(time.time() - t0, 1)

    # Kill criteria
    k814 = abs(pro_degradation) <= 8.0
    k815 = all(v >= base_acc - 0.02 for v in single_mmlu.values()) if single_mmlu else False

    results["kill_criteria"] = {
        "K814": {"pass": k814, "value": round(pro_degradation, 2), "threshold": 8.0},
        "K815": {"pass": k815, "detail": f"worst single: {min(single_mmlu.values()) if single_mmlu else 'N/A'} vs base: {base_acc:.4f}"},
    }
    results["all_pass"] = k814 and k815

    log(f"\n{'='*60}")
    log(f"THE ANSWER: Composition degradation on Qwen3 = {pro_degradation:+.1f}pp")
    log(f"  (BitNet was {bitnet_degradation:+.1f}pp)")
    for k, v in results["kill_criteria"].items():
        log(f"  {k}: {'PASS' if v['pass'] else 'FAIL'} — {v}")
    log(f"\n{'ALL PASS' if results['all_pass'] else 'KILLED'}")

    RESULTS_FILE.write_text(json.dumps(results, indent=2, cls=NumpyEncoder))

if __name__ == "__main__":
    main()
