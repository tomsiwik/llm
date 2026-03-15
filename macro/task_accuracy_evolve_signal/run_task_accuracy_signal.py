#!/usr/bin/env python3
"""Task accuracy as Evolve quality signal: can 10 held-out questions reliably rank adapters?

Motivation: Answer-only PPL was killed at macro (r=-0.63 cross-domain). The Evolve phase
clone-and-compete needs a cheap, reliable quality signal. This tests whether a tiny
held-out benchmark (10 questions per domain) produces stable adapter rankings.

Kill criteria:
  K1: 10-question ranking Kendall tau < 0.7 vs 100-question gold standard
  K2: per-domain evaluation cost exceeds 60s/adapter
  K3: accuracy ranking disagrees with PPL ranking AND gold-standard ranking
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
RESULTS_DIR = Path("/workspace/llm/results/task_accuracy_evolve_signal")
SEED = 42
N_SUBSETS = 5  # random 10-question draws per domain
SUBSET_SIZE = 10
GOLD_SIZE = 100
N_EXPERTS = 15  # test 15 adapters


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


def evaluate_mmlu_accuracy(model, tokenizer, dataset, choice_tokens):
    """Evaluate accuracy on a dataset split."""
    correct = 0
    for ex in dataset:
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

    return correct / max(1, len(dataset))


def compute_answer_only_ppl(model, tokenizer, dataset):
    """Compute answer-only perplexity on dataset (for PPL comparison)."""
    total_nll = 0.0
    count = 0
    for ex in dataset:
        prompt = format_mmlu_prompt(ex)
        gold_letter = "ABCD"[ex["answer"]]
        full_text = prompt + " " + gold_letter
        inputs = tokenizer(
            full_text, return_tensors="pt", truncation=True, max_length=512
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs)
            # NLL of the last token (the answer)
            logits = outputs.logits[0, -2]  # logit that predicts the answer token
            log_probs = torch.log_softmax(logits, dim=-1)
            answer_ids = tokenizer.encode(gold_letter, add_special_tokens=False)
            if answer_ids:
                total_nll -= log_probs[answer_ids[0]].item()
                count += 1

    return np.exp(total_nll / max(1, count))


def kendall_tau(ranking_a, ranking_b):
    """Compute Kendall tau rank correlation."""
    from scipy.stats import kendalltau
    tau, p_value = kendalltau(ranking_a, ranking_b)
    return tau, p_value


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
    parser.add_argument("--n-experts", type=int, default=N_EXPERTS)
    parser.add_argument("--n-subsets", type=int, default=N_SUBSETS)
    parser.add_argument("--subset-size", type=int, default=SUBSET_SIZE)
    parser.add_argument("--gold-size", type=int, default=GOLD_SIZE)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(SEED)

    # Select 10 diverse MMLU subjects (spanning knowledge domains)
    subjects = [
        "abstract_algebra", "anatomy", "college_computer_science",
        "college_physics", "econometrics", "high_school_biology",
        "high_school_us_history", "machine_learning",
        "professional_medicine", "world_religions",
    ]

    from datasets import load_dataset

    print(f"Loading {len(subjects)} MMLU subjects...")
    subject_datasets = {}
    for subj in subjects:
        try:
            ds = load_dataset("cais/mmlu", subj, split="test", trust_remote_code=True)
            if len(ds) >= args.gold_size:
                subject_datasets[subj] = ds
                print(f"  {subj}: {len(ds)} examples")
            else:
                print(f"  Skip {subj}: only {len(ds)} examples (need {args.gold_size})")
        except Exception as e:
            print(f"  Skip {subj}: {e}")

    print(f"\nUsable subjects: {len(subject_datasets)}")
    if len(subject_datasets) < 3:
        raise RuntimeError("Need at least 3 subjects with enough data")

    # Generate subset indices for each subject
    subset_indices = {}
    for subj, ds in subject_datasets.items():
        gold_indices = list(range(min(args.gold_size, len(ds))))
        subsets = []
        for _ in range(args.n_subsets):
            sub = rng.choice(gold_indices, size=min(args.subset_size, len(gold_indices)), replace=False)
            subsets.append(sub.tolist())
        subset_indices[subj] = {"gold": gold_indices, "subsets": subsets}

    adapters = get_adapter_list()[:args.n_experts]
    print(f"Testing {len(adapters)} adapters")

    # Precompute choice tokens
    _, tokenizer = load_base_model()
    choice_tokens = {}
    for letter in "ABCD":
        ids = tokenizer.encode(letter, add_special_tokens=False)
        choice_tokens[letter] = ids[0]
        ids_space = tokenizer.encode(f" {letter}", add_special_tokens=False)
        if ids_space:
            choice_tokens[f" {letter}"] = ids_space[-1]

    # Evaluate each adapter on gold and subsets
    all_results = {
        "config": {
            "subjects": list(subject_datasets.keys()),
            "n_experts": len(adapters),
            "n_subsets": args.n_subsets,
            "subset_size": args.subset_size,
            "gold_size": args.gold_size,
            "seed": SEED,
        },
        "adapters": {},
        "rankings": {},
        "kill_criteria": {},
    }

    from peft import PeftModel

    for adapter_idx, adapter_name in enumerate(adapters):
        print(f"\n=== Adapter {adapter_idx+1}/{len(adapters)}: {adapter_name} ===")
        t0 = time.time()

        # Load model with adapter
        base_model, _ = load_base_model()
        adapter_path = str(ADAPTER_DIR / adapter_name)
        try:
            peft_model = PeftModel.from_pretrained(
                base_model, adapter_path, adapter_name=adapter_name
            )
            peft_model.set_adapter(adapter_name)
            peft_model.eval()
        except Exception as e:
            print(f"  Failed: {e}")
            all_results["adapters"][adapter_name] = {"error": str(e)}
            del base_model
            torch.cuda.empty_cache()
            continue

        adapter_results = {"per_subject": {}}

        for subj, ds in subject_datasets.items():
            # Gold standard (100 questions)
            gold_ds = ds.select(subset_indices[subj]["gold"])
            gold_acc = evaluate_mmlu_accuracy(peft_model, tokenizer, gold_ds, choice_tokens)
            gold_ppl = compute_answer_only_ppl(peft_model, tokenizer, gold_ds)

            # Subsets (10 questions each, N_SUBSETS draws)
            subset_accs = []
            for sub_idx in subset_indices[subj]["subsets"]:
                sub_ds = ds.select(sub_idx)
                sub_acc = evaluate_mmlu_accuracy(peft_model, tokenizer, sub_ds, choice_tokens)
                subset_accs.append(sub_acc)

            adapter_results["per_subject"][subj] = {
                "gold_accuracy": round(gold_acc, 4),
                "gold_ppl": round(gold_ppl, 4),
                "subset_accuracies": [round(a, 4) for a in subset_accs],
                "subset_mean": round(np.mean(subset_accs), 4),
                "subset_std": round(np.std(subset_accs), 4),
            }
            print(f"  {subj}: gold={gold_acc:.3f}, subset_mean={np.mean(subset_accs):.3f}±{np.std(subset_accs):.3f}")

        elapsed = time.time() - t0
        adapter_results["elapsed_s"] = round(elapsed, 1)
        all_results["adapters"][adapter_name] = adapter_results

        del peft_model, base_model
        torch.cuda.empty_cache()

    # Phase 3: Compute rankings and correlations
    print("\n=== Ranking Analysis ===")

    from scipy.stats import kendalltau

    adapter_names = [n for n in adapters if "error" not in all_results["adapters"].get(n, {})]
    taus_by_subject = {}
    overall_taus = []

    for subj in subject_datasets:
        # Gold ranking
        gold_scores = []
        subset_scores = []
        ppl_scores = []

        for name in adapter_names:
            info = all_results["adapters"][name]["per_subject"].get(subj, {})
            if not info:
                continue
            gold_scores.append(info["gold_accuracy"])
            subset_scores.append(info["subset_mean"])
            ppl_scores.append(-info["gold_ppl"])  # negate: lower PPL = better

        if len(gold_scores) < 3:
            continue

        gold_rank = np.argsort(np.argsort(-np.array(gold_scores)))
        subset_rank = np.argsort(np.argsort(-np.array(subset_scores)))
        ppl_rank = np.argsort(np.argsort(-np.array(ppl_scores)))

        tau_subset, p_subset = kendalltau(gold_rank, subset_rank)
        tau_ppl, p_ppl = kendalltau(gold_rank, ppl_rank)

        taus_by_subject[subj] = {
            "tau_subset_vs_gold": round(tau_subset, 4),
            "p_subset": round(p_subset, 4),
            "tau_ppl_vs_gold": round(tau_ppl, 4),
            "p_ppl": round(p_ppl, 4),
        }
        overall_taus.append(tau_subset)
        print(f"  {subj}: tau(10q,100q)={tau_subset:.3f} (p={p_subset:.3f}), tau(PPL,100q)={tau_ppl:.3f}")

    all_results["rankings"] = taus_by_subject

    # Kill criteria assessment
    mean_tau = np.mean(overall_taus) if overall_taus else 0.0
    subjects_below_threshold = sum(1 for t in overall_taus if t < 0.7)

    # K1: 10-question ranking Kendall tau < 0.7
    k1_fail = mean_tau < 0.7
    # K2: per-domain eval cost > 60s
    eval_times = [all_results["adapters"][n].get("elapsed_s", 0) for n in adapter_names]
    mean_time_per_adapter = np.mean(eval_times) if eval_times else 0
    k2_fail = mean_time_per_adapter > 60 * len(subject_datasets)  # total, not per-domain

    all_results["kill_criteria"] = {
        "K1_mean_tau": round(mean_tau, 4),
        "K1_threshold": 0.7,
        "K1_fail": bool(k1_fail),
        "K1_subjects_below_0.7": subjects_below_threshold,
        "K2_mean_time_per_adapter_s": round(mean_time_per_adapter, 1),
        "K2_threshold_s": 60 * len(subject_datasets),
        "K2_fail": bool(k2_fail),
    }

    print(f"\n=== Kill Criteria ===")
    print(f"K1: mean tau = {mean_tau:.3f} (threshold 0.7) -> {'KILL' if k1_fail else 'PASS'}")
    print(f"K2: mean time = {mean_time_per_adapter:.0f}s/adapter -> {'KILL' if k2_fail else 'PASS'}")

    # Save results
    results_path = RESULTS_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
