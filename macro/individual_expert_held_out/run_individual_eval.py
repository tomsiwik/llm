#!/usr/bin/env python3
"""Individual expert held-out eval: test each adapter ALONE on MMLU.

Diagnoses whether MMLU -3.67pp regression comes from:
(a) Distillation quality issue (individuals also regress) → fix training
(b) Composition interference (individuals neutral, composed negative) → fix composition

Tests top-20 adapters individually on 20 MMLU subjects.
No training needed — inference only.
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ADAPTER_DIR = Path("/workspace/llm/adapters")
BASE_MODEL = os.environ.get("BASE_MODEL", "/workspace/models/Qwen2.5-7B")
RESULTS_DIR = Path("/workspace/llm/results/individual_expert_held_out")
SEED = 42
N_EXPERTS_TO_TEST = 20

# Same held-out MMLU subjects as small_n_eval
HELD_OUT_SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics",
    "clinical_knowledge", "college_biology", "college_chemistry",
    "college_computer_science", "college_mathematics", "college_medicine",
    "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics",
    "formal_logic", "global_facts", "high_school_biology",
    "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography",
    "high_school_government_and_politics", "high_school_macroeconomics",
    "high_school_mathematics", "high_school_microeconomics",
    "high_school_physics", "high_school_psychology", "high_school_statistics",
    "high_school_us_history", "high_school_world_history", "human_aging",
    "human_sexuality", "international_law", "jurisprudence",
    "logical_fallacies", "machine_learning", "management", "marketing",
    "medical_genetics", "miscellaneous", "moral_disputes", "moral_scenarios",
    "nutrition", "philosophy", "prehistory", "professional_accounting",
    "professional_law", "professional_medicine", "professional_psychology",
    "public_relations", "security_studies", "sociology", "us_foreign_policy",
    "virology", "world_religions",
]


def load_base_model():
    """Load quantized base model."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    model.eval()
    return model, tokenizer


def format_mmlu_prompt(example):
    """Format MMLU example as multiple choice prompt."""
    question = example["question"]
    choices = example["choices"]
    prompt = f"{question}\n"
    for i, choice in enumerate(choices):
        letter = "ABCD"[i]
        prompt += f"{letter}. {choice}\n"
    prompt += "Answer:"
    return prompt


def evaluate_mmlu_logprob(model, tokenizer, subjects, max_per_subject=100):
    """Evaluate model on MMLU subjects using log-prob scoring."""
    from datasets import load_dataset

    results = {}
    total_correct = 0
    total_count = 0

    # Precompute choice token IDs
    choice_tokens = {}
    for letter in "ABCD":
        ids = tokenizer.encode(letter, add_special_tokens=False)
        choice_tokens[letter] = ids[0]
        ids_space = tokenizer.encode(f" {letter}", add_special_tokens=False)
        if ids_space:
            choice_tokens[f" {letter}"] = ids_space[-1]

    for subject in subjects:
        try:
            ds = load_dataset("cais/mmlu", subject, split="test", trust_remote_code=True)
        except Exception as e:
            print(f"  Skip {subject}: {e}")
            continue

        if len(ds) > max_per_subject:
            ds = ds.select(range(max_per_subject))

        correct = 0
        count = 0

        for ex in ds:
            prompt = format_mmlu_prompt(ex)
            inputs = tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=512
            ).to(model.device)

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits[0, -1]
                log_probs = torch.log_softmax(logits, dim=-1)

            scores = {}
            for letter in "ABCD":
                tid = choice_tokens[letter]
                tid_space = choice_tokens.get(f" {letter}", tid)
                scores[letter] = max(log_probs[tid].item(), log_probs[tid_space].item())

            pred = max(scores, key=scores.get)
            gold = "ABCD"[ex["answer"]]
            correct += int(pred == gold)
            count += 1

        acc = correct / max(1, count)
        results[subject] = {"correct": correct, "total": count, "accuracy": round(acc, 4)}
        total_correct += correct
        total_count += count

    overall_acc = total_correct / max(1, total_count) if total_count > 0 else 0.0
    return {
        "per_subject": results,
        "overall": {"correct": total_correct, "total": total_count, "accuracy": round(overall_acc, 4)},
    }


def get_adapter_list():
    """Get available adapters sorted by quality (fallback: alphabetical)."""
    benchmark_path = Path("/workspace/llm/results/pilot50_benchmark.json")
    if benchmark_path.exists():
        with open(benchmark_path) as f:
            data = json.load(f)
        ranked = []
        for name, info in data.get("per_adapter", {}).items():
            ppl_improvement = info.get("ppl_improvement_pct", 0)
            ranked.append((name, ppl_improvement))
        ranked.sort(key=lambda x: x[1], reverse=True)
        return [name for name, _ in ranked]

    return sorted(
        d.name for d in ADAPTER_DIR.iterdir()
        if d.is_dir() and (d / "adapter_config.json").exists()
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-per-subject", type=int, default=50)
    parser.add_argument("--subjects", type=int, default=20, help="Number of MMLU subjects")
    parser.add_argument("--n-experts", type=int, default=N_EXPERTS_TO_TEST, help="Number of experts to test individually")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(SEED)

    # Select subjects
    subjects = list(HELD_OUT_SUBJECTS)
    rng.shuffle(subjects)
    subjects = subjects[:args.subjects]
    print(f"Evaluating {len(subjects)} MMLU subjects")

    adapters = get_adapter_list()[:args.n_experts]
    print(f"Testing {len(adapters)} adapters individually")

    # Phase 1: Base model eval
    print("\n=== Phase 1: Base model ===")
    t0 = time.time()
    base_model, tokenizer = load_base_model()
    base_results = evaluate_mmlu_logprob(base_model, tokenizer, subjects, args.max_per_subject)
    base_acc = base_results["overall"]["accuracy"]
    print(f"Base accuracy: {base_acc:.4f} ({time.time() - t0:.0f}s)")

    all_results = {
        "base": base_results,
        "individual_experts": {},
        "config": {
            "subjects": subjects,
            "max_per_subject": args.max_per_subject,
            "n_experts": len(adapters),
            "seed": SEED,
            "base_model": BASE_MODEL,
        },
    }

    # Phase 2: Test each adapter individually
    from peft import PeftModel

    deltas = []
    # Free base model — each adapter eval reloads fresh (PeftModel modifies in-place)
    del base_model
    torch.cuda.empty_cache()

    for i, adapter_name in enumerate(adapters):
        print(f"\n=== Phase 2: Expert {i+1}/{len(adapters)}: {adapter_name} ===")
        t1 = time.time()

        # Reload fresh base each time (PeftModel.from_pretrained modifies model in-place)
        fresh_model, _ = load_base_model()
        adapter_path = str(ADAPTER_DIR / adapter_name)
        try:
            peft_model = PeftModel.from_pretrained(fresh_model, adapter_path, adapter_name=adapter_name)
            peft_model.set_adapter(adapter_name)
            peft_model.eval()
        except Exception as e:
            print(f"  Failed to load {adapter_name}: {e}")
            all_results["individual_experts"][adapter_name] = {"error": str(e)}
            del fresh_model
            torch.cuda.empty_cache()
            continue

        expert_results = evaluate_mmlu_logprob(peft_model, tokenizer, subjects, args.max_per_subject)
        expert_acc = expert_results["overall"]["accuracy"]
        delta_pp = (expert_acc - base_acc) * 100

        print(f"  {adapter_name}: {expert_acc:.4f} ({delta_pp:+.2f}pp) [{time.time() - t1:.0f}s]")

        all_results["individual_experts"][adapter_name] = {
            "eval_results": expert_results,
            "accuracy": expert_acc,
            "delta_vs_base_pp": round(delta_pp, 4),
        }
        deltas.append(delta_pp)

        # Cleanup adapter
        del peft_model, fresh_model
        torch.cuda.empty_cache()

    # Phase 3: Analysis
    print("\n=== Results Summary ===")
    print(f"Base: {base_acc:.4f}")
    positive = 0
    negative = 0
    for name, info in all_results["individual_experts"].items():
        if "error" in info:
            continue
        d = info["delta_vs_base_pp"]
        marker = "+" if d > 0 else "-" if d < 0 else "="
        print(f"  {name}: {info['accuracy']:.4f} ({d:+.2f}pp) [{marker}]")
        if d > 0:
            positive += 1
        elif d < 0:
            negative += 1

    avg_delta = np.mean(deltas) if deltas else 0.0
    median_delta = np.median(deltas) if deltas else 0.0
    std_delta = np.std(deltas) if deltas else 0.0

    print(f"\n  Average delta: {avg_delta:+.2f}pp")
    print(f"  Median delta:  {median_delta:+.2f}pp")
    print(f"  Std delta:     {std_delta:.2f}pp")
    print(f"  Positive/Negative: {positive}/{negative}")

    # Diagnosis
    if avg_delta < -1.0:
        diagnosis = "DISTILLATION_QUALITY"
        detail = "Individual experts regress on MMLU → distillation is memorizing, not generalizing"
    elif avg_delta > 1.0:
        diagnosis = "COMPOSITION_INTERFERENCE"
        detail = "Individual experts help but composed hurts → composition creates interference"
    else:
        diagnosis = "NEUTRAL_INDIVIDUALS"
        detail = "Individual experts are neutral → dilution from irrelevant experts in composition"

    print(f"\n  Diagnosis: {diagnosis}")
    print(f"  Detail: {detail}")

    all_results["analysis"] = {
        "avg_delta_pp": round(avg_delta, 4),
        "median_delta_pp": round(median_delta, 4),
        "std_delta_pp": round(std_delta, 4),
        "n_positive": positive,
        "n_negative": negative,
        "n_tested": len(deltas),
        "diagnosis": diagnosis,
        "detail": detail,
    }

    # Save
    out_path = RESULTS_DIR / "individual_expert_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
