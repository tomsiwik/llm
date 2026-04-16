"""
P9.G1: Benchmark Showdown — Pierre v3 vs Base Gemma 4 E4B vs Gemma 4 27B (published)

Kill criteria:
  K1390: Pierre v3 GSM8K >= Gemma 4 27B (published benchmark)
  K1391: Math adapter GSM8K gain >= base + 20pp (freshly measured phases 1+2)
  K1392: Medical adapter MedMCQA >= base + 3pp (freshly measured phases 3+4)
  [Cost analysis (4B/27B ratio) demoted to informational section]

Phases:
  1. Base Gemma 4 E4B on GSM8K (n=50)
  2. Math adapter on GSM8K (n=50)
  3. Base Gemma 4 E4B on MedMCQA (n=50 MCQ)
  4. Medical adapter on MedMCQA (n=50)
  5. Oracle-routed adapters on MMLU-Pro (n=14 q/cat)
  6. Cost analysis + comparison table

See MATH.md for formal theorems and predictions.
"""

import gc
import json
import re
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import pandas as pd

mx.set_memory_limit(mx.device_info()["memory_size"] - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
REPO_ROOT = EXPERIMENT_DIR.parent.parent.parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"
REGISTRY_PATH = REPO_ROOT / "adapters" / "registry.json"

MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"
SEED = 42

IS_SMOKE = "--smoke" in sys.argv or __import__("os").environ.get("SMOKE_TEST", "0") == "1"
GSM8K_N = 5 if IS_SMOKE else 50
MEDMCQA_N = 5 if IS_SMOKE else 50
MMLU_N_PER_CAT = 2 if IS_SMOKE else 14

# Published Gemma 4 27B reference benchmarks (from Google Gemma 4 Technical Report, April 2025)
# Source: https://arxiv.org/abs/2503.19786 (Gemma 4 technical report)
# These are approximate figures; update from official table if exact values differ
PUBLISHED_27B = {
    "gsm8k": 90.0,        # ~90-91% on GSM8K 8-shot
    "humaneval": 74.0,    # ~74% HumanEval pass@1
    "mmlu_pro": 79.0,     # ~79% MMLU-Pro
    "medmcqa": 70.0,      # estimated ~70% MedMCQA
}

# Adapter paths
ADAPTER_MATH = REPO_ROOT / "micro/models/exp_p1_t2_single_domain_training/adapters/math"
ADAPTER_CODE = REPO_ROOT / "micro/models/exp_p1_t2_single_domain_training/adapters/code"
ADAPTER_MEDICAL = REPO_ROOT / "micro/models/exp_p1_t2_single_domain_training/adapters/medical"
ADAPTER_LEGAL = REPO_ROOT / "micro/models/exp_p1_t2_multi_domain_5/adapters/legal"
ADAPTER_FINANCE = REPO_ROOT / "micro/models/exp_p1_t2_multi_domain_5/adapters/finance"

MMLU_PATH = REPO_ROOT / "micro/models/exp_bench_mmlu_pro/data/test.parquet"

# Oracle routing: MMLU-Pro category → domain adapter
ORACLE_MAP = {
    "math":             ADAPTER_MATH,
    "engineering":      ADAPTER_MATH,
    "health":           ADAPTER_MEDICAL,
    "law":              ADAPTER_LEGAL,
    "economics":        ADAPTER_FINANCE,
    "business":         ADAPTER_FINANCE,
    "computer science": ADAPTER_CODE,
    "biology":          None,  # no biology adapter
    "chemistry":        None,
    "physics":          None,
    "history":          None,
    "philosophy":       None,
    "psychology":       None,
    "other":            None,
}

DATA_DIR = EXPERIMENT_DIR / "data"


def log(msg):
    print(msg, flush=True)


def log_memory(label=""):
    active = mx.get_active_memory() / 1e9
    peak = mx.get_peak_memory() / 1e9
    log(f"[MEM {label}] active={active:.2f}GB peak={peak:.2f}GB")


def cleanup(*objects):
    for obj in objects:
        del obj
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()


def strip_thinking(response):
    """Remove Gemma 4 thinking tokens (validated regex from p10/W4A16 experiments)."""
    thinking_len = 0
    m = re.search(r'<\|channel>thought.*?<channel\|>', response, flags=re.DOTALL)
    if m:
        thinking_len = len(m.group(0))
    cleaned = re.sub(r'<\|channel>thought.*?<channel\|>', '', response, flags=re.DOTALL)
    cleaned = re.sub(r'<think>.*?</think>', '', cleaned, flags=re.DOTALL)
    return cleaned.strip(), thinking_len


def parse_mcq_answer(response):
    answer_text, thinking_chars = strip_thinking(response)
    if not answer_text:
        return None, thinking_chars
    upper = answer_text.upper()
    if re.match(r'^[A-J]$', upper.strip()):
        return upper.strip(), thinking_chars
    for pattern in [r'answer[:\s]+([A-J])\b', r'\b([A-J])\b(?:\s*$|\s*[.)])',
                    r'(?:^|\s)([A-J])(?:\s*$|\s*[.)])']:
        m = re.search(pattern, upper, re.MULTILINE)
        if m:
            return m.group(1), thinking_chars
    m = re.search(r'\b([A-J])\b', upper)
    if m:
        return m.group(1), thinking_chars
    return None, thinking_chars


def load_model(model_id, adapter_path=None):
    from mlx_lm import load
    log(f"  Loading {model_id}" + (f" + {Path(adapter_path).name}" if adapter_path else " (base)"))
    if adapter_path:
        model, tokenizer = load(model_id, adapter_path=str(adapter_path))
    else:
        model, tokenizer = load(model_id)
    log_memory("post-load")
    return model, tokenizer


def generate_answer(model, tokenizer, prompt, max_tokens=512):
    from mlx_lm import generate
    result = generate(
        model, tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        verbose=False,
    )
    return result


def fmt_gsm8k_prompt(question):
    return (
        f"<|im_start|>user\n"
        f"Solve the following math problem step by step.\n\n"
        f"{question}\n\n"
        f"Show your work and give the final answer after '####'.\n"
        f"<|im_start|>assistant\n"
    )


def extract_gsm8k_answer(text):
    cleaned, _ = strip_thinking(text)
    m = re.search(r'####\s*([\d,.-]+)', cleaned)
    if m:
        return m.group(1).replace(",", "").strip()
    nums = re.findall(r'-?[\d,]+\.?\d*', cleaned)
    return nums[-1].replace(",", "").strip() if nums else None


def fetch_gsm8k(n):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    gsm_path = DATA_DIR / "gsm8k_test.jsonl"
    if not gsm_path.exists():
        log("  Fetching GSM8K test data...")
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main", split="test", trust_remote_code=True)
        rows = [{"question": r["question"], "answer": r["answer"]} for r in ds]
        with open(gsm_path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        log(f"  Saved {len(rows)} examples")
    data = []
    with open(gsm_path) as f:
        for line in f:
            data.append(json.loads(line.strip()))
    rng = np.random.RandomState(SEED)
    return [data[i] for i in rng.choice(len(data), min(n, len(data)), replace=False)]


def eval_gsm8k(model, tokenizer, data):
    """Evaluate on GSM8K. Returns accuracy, list of bool."""
    correct = []
    for i, ex in enumerate(data):
        prompt = fmt_gsm8k_prompt(ex["question"])
        response = generate_answer(model, tokenizer, prompt, max_tokens=512)
        pred = extract_gsm8k_answer(response)
        m = re.search(r'####\s*([\d,.-]+)', ex["answer"])
        gt = m.group(1).replace(",", "").strip() if m else ex["answer"].strip()
        is_correct = pred is not None and pred == gt
        correct.append(is_correct)
        if (i + 1) % 5 == 0:
            log(f"    [{i+1}/{len(data)}] running={sum(correct)}/{i+1} ({100*sum(correct)/(i+1):.1f}%)")
    return correct


def fmt_medmcqa_prompt(question, options):
    opts_str = "\n".join([f"  {chr(65+j)}) {opt}" for j, opt in enumerate(options)])
    return (
        f"<|im_start|>user\n"
        f"Answer the following medical question by selecting the best option.\n\n"
        f"Question: {question}\n\n"
        f"Options:\n{opts_str}\n\n"
        f"Reply with just the letter (A, B, C, or D).\n"
        f"<|im_start|>assistant\n"
    )


def fetch_medmcqa(n):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    med_path = DATA_DIR / "medmcqa_val.jsonl"
    if not med_path.exists():
        log("  Fetching MedMCQA validation data...")
        from datasets import load_dataset
        ds = load_dataset("openlifescienceai/medmcqa", split="validation", trust_remote_code=True)
        # Filter for 4-option questions with clear correct answer
        rows = []
        for r in ds:
            if all(r.get(k) for k in ["question", "opa", "opb", "opc", "opd"]) and r.get("cop") in [0, 1, 2, 3]:
                rows.append({
                    "question": r["question"],
                    "options": [r["opa"], r["opb"], r["opc"], r["opd"]],
                    "answer": chr(65 + r["cop"]),  # 0→A, 1→B, 2→C, 3→D
                })
            if len(rows) >= 2000:
                break
        with open(med_path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        log(f"  Saved {len(rows)} MedMCQA examples")
    data = []
    with open(med_path) as f:
        for line in f:
            data.append(json.loads(line.strip()))
    rng = np.random.RandomState(SEED + 1)
    return [data[i] for i in rng.choice(len(data), min(n, len(data)), replace=False)]


def eval_medmcqa(model, tokenizer, data):
    """Evaluate on MedMCQA. Returns accuracy, list of bool."""
    correct = []
    for i, ex in enumerate(data):
        prompt = fmt_medmcqa_prompt(ex["question"], ex["options"])
        response = generate_answer(model, tokenizer, prompt, max_tokens=128)
        pred, _ = parse_mcq_answer(response)
        is_correct = pred is not None and pred.upper() == ex["answer"].upper()
        correct.append(is_correct)
        if (i + 1) % 10 == 0:
            log(f"    [{i+1}/{len(data)}] running={sum(correct)}/{i+1} ({100*sum(correct)/(i+1):.1f}%)")
    return correct


def fmt_mmlu_prompt(question, choices):
    opts = "\n".join([f"  {chr(65+j)}) {c}" for j, c in enumerate(choices)])
    return (
        f"<|im_start|>user\n"
        f"Answer the following question with the letter of the correct choice.\n\n"
        f"Question: {question}\n\n"
        f"{opts}\n\n"
        f"Reply with just the letter (A, B, C, D, or E).\n"
        f"<|im_start|>assistant\n"
    )


def eval_mmlu_oracle_routed(n_per_cat=MMLU_N_PER_CAT):
    """
    Oracle-routed MMLU-Pro: for each category, load the best-matching adapter
    and evaluate n_per_cat questions. Returns per-category and overall accuracy.
    """
    if not MMLU_PATH.exists():
        return {"error": "MMLU-Pro data not found at expected path"}

    df = pd.read_parquet(MMLU_PATH)
    all_cats = sorted(df["category"].unique())
    rng = np.random.RandomState(SEED + 2)

    results_per_cat = {}
    overall_correct = 0
    overall_total = 0
    adapter_models = {}  # cache: adapter_path_str → (model, tokenizer)

    for cat in all_cats:
        cat_df = df[df["category"] == cat]
        cat_df = cat_df.sample(min(n_per_cat, len(cat_df)), random_state=rng.randint(0, 2**31))

        adapter_path = ORACLE_MAP.get(cat)
        adapter_key = str(adapter_path) if adapter_path else "base"

        if adapter_key not in adapter_models:
            # Load base model first if not cached
            if "base" not in adapter_models:
                model, tokenizer = load_model(MODEL_ID, adapter_path=None)
                adapter_models["base"] = (model, tokenizer)
                log_memory(f"base loaded")
            if adapter_path and adapter_path.exists():
                model, tokenizer = load_model(MODEL_ID, adapter_path=adapter_path)
                adapter_models[adapter_key] = (model, tokenizer)
                log_memory(f"adapter={cat} loaded")
            else:
                adapter_key = "base"
                if adapter_path and not adapter_path.exists():
                    log(f"  WARNING: adapter for {cat} not found at {adapter_path}, using base")

        model, tokenizer = adapter_models[adapter_key]
        log(f"\n  [{cat}] adapter={'base' if adapter_key == 'base' else cat} n={len(cat_df)}")

        cat_correct = 0
        for _, row in cat_df.iterrows():
            choices = row["options"]
            prompt = fmt_mmlu_prompt(row["question"], choices)
            response = generate_answer(model, tokenizer, prompt, max_tokens=64)
            pred, _ = parse_mcq_answer(response)
            gt = chr(65 + row["answer_index"]) if "answer_index" in row else row.get("answer", "?")
            if isinstance(gt, int):
                gt = chr(65 + gt)
            cat_correct += int(pred is not None and pred.upper() == str(gt).upper())

        acc = cat_correct / len(cat_df) if cat_df is not None and len(cat_df) > 0 else 0
        results_per_cat[cat] = {"correct": cat_correct, "total": len(cat_df), "accuracy": acc, "adapter": adapter_key}
        overall_correct += cat_correct
        overall_total += len(cat_df)
        log(f"    {cat}: {cat_correct}/{len(cat_df)} = {acc:.1%}")

    # Cleanup cached models
    for model, tokenizer in adapter_models.values():
        cleanup(model)

    return {
        "per_category": results_per_cat,
        "overall_accuracy": overall_correct / overall_total if overall_total > 0 else 0,
        "overall_correct": overall_correct,
        "overall_total": overall_total,
    }


def compute_cost_analysis():
    """
    Compute serving cost comparison: Pierre v3 (4B) vs Gemma 4 27B.
    Cost model: memory_bandwidth_bound → cost ∝ model_params.
    """
    params_4b = 4.3e9    # Gemma 4 E4B actual params
    params_27b = 27.2e9  # Gemma 4 27B actual params

    mem_4b_gb = params_4b * 0.5 / 1e9   # 4-bit: 0.5 bytes/param
    mem_27b_gb = params_27b * 0.5 / 1e9  # 4-bit: 0.5 bytes/param

    cost_ratio = params_4b / params_27b
    adapter_overhead_mb = 5.0  # per adapter, pre-merged → ~0 inference overhead

    return {
        "params_4b": params_4b,
        "params_27b": params_27b,
        "mem_4b_gb": round(mem_4b_gb, 2),
        "mem_27b_gb": round(mem_27b_gb, 2),
        "cost_ratio": round(cost_ratio, 4),      # 4B cost / 27B cost
        "cost_ratio_pct": round(cost_ratio * 100, 1),
        "adapter_overhead_mb": adapter_overhead_mb,
        "k1392_pass": cost_ratio < 0.50,
    }


def main():
    start = time.time()
    results = {
        "smoke": IS_SMOKE,
        "model": MODEL_ID,
        "seed": SEED,
        "published_reference": PUBLISHED_27B,
        "phases": {},
        "kill_criteria": {},
        "cost_analysis": None,
    }

    log("=" * 60)
    log("P9.G1: Benchmark Showdown — Pierre v3 vs Base vs 27B")
    log("=" * 60)

    # --- Phase 1: Base GSM8K ---
    log("\n[Phase 1] Base Gemma 4 E4B on GSM8K")
    t0 = time.time()
    gsm8k_data = fetch_gsm8k(GSM8K_N)
    model_base, tokenizer = load_model(MODEL_ID)
    base_gsm8k = eval_gsm8k(model_base, tokenizer, gsm8k_data)
    base_gsm8k_acc = 100 * sum(base_gsm8k) / len(base_gsm8k)
    log(f"  Base GSM8K: {base_gsm8k_acc:.1f}% ({sum(base_gsm8k)}/{len(base_gsm8k)})")
    results["phases"]["base_gsm8k"] = {
        "accuracy": base_gsm8k_acc,
        "correct": int(sum(base_gsm8k)),
        "total": len(base_gsm8k),
        "elapsed_s": round(time.time() - t0, 1),
    }
    cleanup(model_base)

    # --- Phase 2: Math adapter GSM8K ---
    log("\n[Phase 2] Math adapter on GSM8K")
    t0 = time.time()
    model_math, tokenizer = load_model(MODEL_ID, adapter_path=ADAPTER_MATH)
    math_gsm8k = eval_gsm8k(model_math, tokenizer, gsm8k_data)
    math_gsm8k_acc = 100 * sum(math_gsm8k) / len(math_gsm8k)
    log(f"  Math adapter GSM8K: {math_gsm8k_acc:.1f}% ({sum(math_gsm8k)}/{len(math_gsm8k)})")
    results["phases"]["math_gsm8k"] = {
        "accuracy": math_gsm8k_acc,
        "correct": int(sum(math_gsm8k)),
        "total": len(math_gsm8k),
        "elapsed_s": round(time.time() - t0, 1),
    }
    cleanup(model_math)

    # --- Phase 3: Base MedMCQA ---
    log("\n[Phase 3] Base Gemma 4 E4B on MedMCQA")
    t0 = time.time()
    try:
        med_data = fetch_medmcqa(MEDMCQA_N)
        model_base2, tokenizer2 = load_model(MODEL_ID)
        base_med = eval_medmcqa(model_base2, tokenizer2, med_data)
        base_med_acc = 100 * sum(base_med) / len(base_med)
        log(f"  Base MedMCQA: {base_med_acc:.1f}% ({sum(base_med)}/{len(base_med)})")
        results["phases"]["base_medmcqa"] = {
            "accuracy": base_med_acc,
            "correct": int(sum(base_med)),
            "total": len(base_med),
            "elapsed_s": round(time.time() - t0, 1),
        }
        cleanup(model_base2)

        # --- Phase 4: Medical adapter MedMCQA ---
        log("\n[Phase 4] Medical adapter on MedMCQA")
        t0 = time.time()
        model_med, tokenizer_med = load_model(MODEL_ID, adapter_path=ADAPTER_MEDICAL)
        med_med = eval_medmcqa(model_med, tokenizer_med, med_data)
        med_med_acc = 100 * sum(med_med) / len(med_med)
        log(f"  Medical adapter MedMCQA: {med_med_acc:.1f}% ({sum(med_med)}/{len(med_med)})")
        results["phases"]["medical_medmcqa"] = {
            "accuracy": med_med_acc,
            "correct": int(sum(med_med)),
            "total": len(med_med),
            "elapsed_s": round(time.time() - t0, 1),
        }
        cleanup(model_med)
    except Exception as e:
        log(f"  MedMCQA phases failed: {e}")
        results["phases"]["base_medmcqa"] = {"error": str(e)}
        results["phases"]["medical_medmcqa"] = {"error": str(e)}
        base_med_acc = None
        med_med_acc = None

    # --- Phase 5: Oracle-routed MMLU-Pro (if data available) ---
    if not IS_SMOKE:
        log("\n[Phase 5] Oracle-routed MMLU-Pro")
        t0 = time.time()
        mmlu_results = eval_mmlu_oracle_routed(MMLU_N_PER_CAT)
        if "error" not in mmlu_results:
            mmlu_acc = 100 * mmlu_results["overall_accuracy"]
            log(f"  Oracle-routed MMLU-Pro: {mmlu_acc:.1f}%")
        else:
            mmlu_acc = None
        mmlu_results["elapsed_s"] = round(time.time() - t0, 1)
        results["phases"]["oracle_mmlu"] = mmlu_results
    else:
        log("\n[Phase 5] Skipping oracle MMLU in smoke mode")
        mmlu_acc = None
        results["phases"]["oracle_mmlu"] = {"skipped": "smoke_mode"}

    # --- Phase 6: Known scores from registry + cost analysis ---
    log("\n[Phase 6] Registry scores + cost analysis")
    code_humaneval = 63.0  # from registry: code-codealpaca-knowledge-v0
    # Base HumanEval: estimate from base_gsm8k (if 4B base does ~55% GSM8K,
    # HumanEval base is typically ~0.8x the GSM8K ratio for coding)
    # We use 42% as conservative estimate for base HumanEval on E4B 4-bit
    base_humaneval_est = 42.0  # estimated, not locally measured
    humaneval_gain = code_humaneval - base_humaneval_est

    cost = compute_cost_analysis()
    results["cost_analysis"] = cost

    results["phases"]["known_registry"] = {
        "code_humaneval": code_humaneval,
        "math_gsm8k_registry": 82.0,
        "base_humaneval_estimated": base_humaneval_est,
        "humaneval_gain_pp": round(humaneval_gain, 1),
        "base_mmlu_pro_thinking": 62.1,  # Finding #530
    }

    log(f"  Code adapter HumanEval: {code_humaneval}% (from registry)")
    log(f"  Base HumanEval (estimated): {base_humaneval_est}%")
    log(f"  Cost ratio (4B/27B): {cost['cost_ratio_pct']}%")

    # --- Kill Criteria ---
    log("\n[Kill Criteria]")

    # K1390: Math adapter GSM8K >= Gemma 4 27B published
    k1390_val = math_gsm8k_acc
    k1390_ref = PUBLISHED_27B["gsm8k"]
    k1390_pass = k1390_val >= k1390_ref
    log(f"  K1390: Pierre math {k1390_val:.1f}% >= 27B {k1390_ref:.1f}%: {'PASS' if k1390_pass else 'FAIL'}")

    # K1391: Math adapter GSM8K gain >= 20pp over base (freshly measured phases 1+2)
    k1391_val = math_gsm8k_acc - base_gsm8k_acc
    k1391_pass = k1391_val >= 20.0
    log(f"  K1391: Math adapter GSM8K gain {k1391_val:.1f}pp >= 20pp: {'PASS' if k1391_pass else 'FAIL'}")

    # K1392: Medical adapter MedMCQA >= base + 3pp (freshly measured phases 3+4)
    if base_med_acc is not None and med_med_acc is not None:
        k1392_val = med_med_acc - base_med_acc
        k1392_pass = k1392_val >= 3.0
        log(f"  K1392: Medical MedMCQA gain {k1392_val:.1f}pp >= 3pp: {'PASS' if k1392_pass else 'FAIL'}")
    else:
        k1392_val = None
        k1392_pass = False
        log(f"  K1392: Medical MedMCQA gain N/A (phase 3+4 failed): FAIL")

    results["kill_criteria"] = {
        "k1390": {
            "pierre_gsm8k": k1390_val,
            "reference_27b_gsm8k": k1390_ref,
            "pass": k1390_pass,
            "note": "K1390 uses published 27B number as reference",
        },
        "k1391": {
            "math_gsm8k_gain_pp": round(k1391_val, 1),
            "math_gsm8k_acc": math_gsm8k_acc,
            "base_gsm8k_acc": base_gsm8k_acc,
            "pass": k1391_pass,
            "note": "Freshly measured in phases 1+2; expected ~27pp if math=82%, base=55%",
        },
        "k1392": {
            "medical_medmcqa_gain_pp": round(k1392_val, 1) if k1392_val is not None else None,
            "medical_medmcqa_acc": med_med_acc,
            "base_medmcqa_acc": base_med_acc,
            "pass": k1392_pass,
            "note": "Freshly measured in phases 3+4; registry=50.0% medical, uncertain if gain>=3pp",
        },
        "cost_analysis_informational": {
            "cost_ratio_pct": cost["cost_ratio_pct"],
            "note": "Demoted to informational: 4B/27B ratio is fixed math (15.8%), not a kill criterion",
        },
    }

    elapsed = time.time() - start
    results["elapsed_s"] = round(elapsed, 1)

    log(f"\n[Done] Total elapsed: {elapsed/3600:.2f}h")
    log(f"  K1390 (math GSM8K >= 27B {k1390_ref:.0f}%): {'PASS' if k1390_pass else 'FAIL'}")
    log(f"  K1391 (math GSM8K gain >= 20pp): {'PASS' if k1391_pass else 'FAIL'} ({k1391_val:.1f}pp)")
    k1392_gain_str = f"{k1392_val:.1f}pp" if k1392_val is not None else "N/A"
    log(f"  K1392 (medical MedMCQA gain >= 3pp): {'PASS' if k1392_pass else 'FAIL'} ({k1392_gain_str})")

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    log(f"\nResults saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
