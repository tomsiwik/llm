#!/usr/bin/env python3
"""Leave-One-Out Expert Contribution Ranking at Macro Scale.

For each of N=50 pilot adapters, compose N-1 adapters and compare composed model
PPL against the full N composition. The expert whose removal IMPROVES quality
most is the worst contributor. The expert whose removal HURTS quality most is
the most valuable.

This is a label-free diagnostic: no task labels needed, just next-token
prediction on a generic calibration set.

Kill criteria:
- K1: all experts contribute equally (leave-one-out PPL variance < 0.1%) — ranking is noise
- K2: ranking takes >4hrs for N=50 (impractical for iterative use)
- K3: leave-one-out ranking is unstable across calibration sets (Kendall tau < 0.5 between two disjoint sets)

Supports SMOKE_TEST=1 for <60s validation.
"""

import gc
import json
import os
import random
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

IS_SMOKE = os.environ.get("SMOKE_TEST") == "1"
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

REPO_ROOT = Path("/workspace/llm")
ADAPTER_DIR = REPO_ROOT / "adapters"
DATA_DIR = REPO_ROOT / "data" / "distillation"
RESULTS_DIR = REPO_ROOT / "results" / "leave_one_out_expert_ranking"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BASE_MODEL = "Qwen/Qwen2.5-7B"
HF_CACHE = "/workspace/hf_cache"

MAX_SEQ_LEN = 512 if not IS_SMOKE else 128
# Two disjoint calibration sets for stability test
CALIB_SAMPLES_PER_SET = 20 if not IS_SMOKE else 3
N_EXPERTS = 50 if not IS_SMOKE else 5
SEED = 42


def log(msg):
    ts = time.strftime("%H:%M:%S", time.gmtime())
    print(f"[{ts}] {msg}", flush=True)


def discover_adapters():
    adapters = []
    if not ADAPTER_DIR.exists():
        log(f"ERROR: adapter dir {ADAPTER_DIR} not found")
        sys.exit(1)
    for d in sorted(ADAPTER_DIR.iterdir()):
        if d.is_dir() and (d / "adapter_model.safetensors").exists():
            adapters.append(d.name)
    return adapters


def compose_adapters_on_cpu(adapter_names):
    """Compose adapters on CPU via weight averaging, save as single adapter."""
    from safetensors.torch import load_file, save_file
    import torch

    composed = {}
    for name in adapter_names:
        path = ADAPTER_DIR / name / "adapter_model.safetensors"
        tensors = load_file(str(path), device="cpu")
        for key, val in tensors.items():
            if key in composed:
                composed[key] = composed[key] + val.float()
            else:
                composed[key] = val.float()
        del tensors
        gc.collect()

    tmpdir = tempfile.mkdtemp(prefix="composed_")
    out_path = os.path.join(tmpdir, "adapter_model.safetensors")
    save_file({k: v.to(torch.bfloat16) for k, v in composed.items()}, out_path)

    cfg_src = ADAPTER_DIR / adapter_names[0] / "adapter_config.json"
    if cfg_src.exists():
        shutil.copy(str(cfg_src), os.path.join(tmpdir, "adapter_config.json"))

    del composed
    gc.collect()
    return tmpdir


def load_calibration_data(adapters, n_per_set):
    """Load calibration texts from adapter domains, split into two disjoint sets."""
    all_texts = []
    for domain in adapters:
        domain_dir = DATA_DIR / domain
        if not domain_dir.exists():
            continue
        for f in sorted(domain_dir.glob("*.jsonl")):
            with open(f) as fh:
                for line in fh:
                    try:
                        obj = json.loads(line)
                        # Extract text from messages format
                        if "messages" in obj:
                            text = " ".join(
                                m.get("content", "") for m in obj["messages"]
                                if m.get("role") in ("user", "assistant")
                            )
                        else:
                            text = obj.get("text", obj.get("content", obj.get("output", "")))
                        if text and len(text) > 100:
                            all_texts.append(text[:MAX_SEQ_LEN * 4])
                    except Exception:
                        continue
            if len(all_texts) >= n_per_set * 4:
                break
        if len(all_texts) >= n_per_set * 4:
            break

    random.shuffle(all_texts)
    set_a = all_texts[:n_per_set]
    set_b = all_texts[n_per_set:n_per_set * 2]
    return set_a, set_b


def compute_ppl(model, tokenizer, texts):
    """Compute perplexity over texts."""
    import torch
    import math

    total_loss = 0.0
    total_tokens = 0

    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                          max_length=MAX_SEQ_LEN).to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
            shift_logits = outputs.logits[:, :-1, :].contiguous()
            shift_labels = inputs["input_ids"][:, 1:].contiguous()
            loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)),
                          shift_labels.view(-1))
            total_loss += loss.item()
            total_tokens += shift_labels.numel()
        del inputs, outputs
        torch.cuda.empty_cache()

    return math.exp(total_loss / total_tokens) if total_tokens > 0 else float("inf")


def main():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    t0 = time.time()
    log("=" * 60)
    log("LEAVE-ONE-OUT EXPERT CONTRIBUTION RANKING")
    log(f"  N experts: {N_EXPERTS}")
    log(f"  Calibration samples/set: {CALIB_SAMPLES_PER_SET}")
    log(f"  Smoke test: {IS_SMOKE}")
    log("=" * 60)

    # Discover adapters
    all_adapters = discover_adapters()
    log(f"Found {len(all_adapters)} adapters")
    selected = all_adapters[:N_EXPERTS]
    log(f"Using first {len(selected)} adapters")

    # Load calibration data (two disjoint sets)
    log("Loading calibration data...")
    calib_a, calib_b = load_calibration_data(selected, CALIB_SAMPLES_PER_SET)
    log(f"Calibration set A: {len(calib_a)} texts, set B: {len(calib_b)} texts")
    if len(calib_a) < 3 or len(calib_b) < 3:
        log("ERROR: insufficient calibration data")
        sys.exit(1)

    # Load tokenizer
    log("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, cache_dir=HF_CACHE, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Step 1: Compose ALL N adapters and measure reference PPL
    log(f"\nComposing all {len(selected)} adapters (reference)...")
    ref_dir = compose_adapters_on_cpu(selected)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, cache_dir=HF_CACHE, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True
    )
    model = PeftModel.from_pretrained(model, ref_dir)
    model.eval()

    ref_ppl_a = compute_ppl(model, tokenizer, calib_a)
    ref_ppl_b = compute_ppl(model, tokenizer, calib_b)
    log(f"Reference PPL: set_A={ref_ppl_a:.4f}, set_B={ref_ppl_b:.4f}")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    shutil.rmtree(ref_dir)

    # Step 2: Leave-one-out for each expert
    rankings_a = {}  # expert -> PPL change on set A
    rankings_b = {}  # expert -> PPL change on set B

    for i, removed in enumerate(selected):
        log(f"\n[{i+1}/{len(selected)}] Removing: {removed}")
        remaining = [a for a in selected if a != removed]
        loo_dir = compose_adapters_on_cpu(remaining)

        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, cache_dir=HF_CACHE, torch_dtype=torch.bfloat16,
            device_map="auto", trust_remote_code=True
        )
        model = PeftModel.from_pretrained(model, loo_dir)
        model.eval()

        ppl_a = compute_ppl(model, tokenizer, calib_a)
        ppl_b = compute_ppl(model, tokenizer, calib_b)

        # Positive = removal HURTS (expert is valuable)
        # Negative = removal HELPS (expert is harmful)
        delta_a = (ppl_a - ref_ppl_a) / ref_ppl_a * 100
        delta_b = (ppl_b - ref_ppl_b) / ref_ppl_b * 100
        rankings_a[removed] = delta_a
        rankings_b[removed] = delta_b

        log(f"  PPL_A={ppl_a:.4f} (delta={delta_a:+.4f}%), "
            f"PPL_B={ppl_b:.4f} (delta={delta_b:+.4f}%)")

        del model
        gc.collect()
        torch.cuda.empty_cache()
        shutil.rmtree(loo_dir)

    elapsed = time.time() - t0

    # Step 3: Analyze
    log(f"\n{'='*60}")
    log("ANALYSIS")
    log(f"{'='*60}")

    deltas_a = np.array([rankings_a[e] for e in selected])
    deltas_b = np.array([rankings_b[e] for e in selected])

    # K1: variance check
    ppl_variance = float(np.std(deltas_a))
    k1_pass = ppl_variance >= 0.1  # need sufficient variance for ranking
    log(f"\nK1 (PPL delta std): {ppl_variance:.4f}% (threshold >=0.1%)")
    log(f"  {'SURVIVES' if k1_pass else 'KILLED'} — "
        f"{'enough variance for meaningful ranking' if k1_pass else 'experts contribute equally, ranking is noise'}")

    # K2: time check
    k2_pass = elapsed <= 4 * 3600  # 4 hours
    log(f"\nK2 (time): {elapsed:.0f}s = {elapsed/3600:.2f}hrs (threshold <=4hrs)")
    log(f"  {'SURVIVES' if k2_pass else 'KILLED'}")

    # K3: stability across calibration sets
    from scipy.stats import kendalltau
    tau, p = kendalltau(deltas_a, deltas_b)
    k3_pass = tau >= 0.5
    log(f"\nK3 (ranking stability): Kendall tau={tau:.3f} (p={p:.4f}), threshold >=0.5")
    log(f"  {'SURVIVES' if k3_pass else 'KILLED'}")

    # Expert ranking (by set A, validated by set B)
    rank_order = sorted(selected, key=lambda e: rankings_a[e])
    log(f"\nExpert ranking (worst → best contribution):")
    log(f"  {'Expert':<30} {'Delta_A':>10} {'Delta_B':>10}")
    for e in rank_order:
        log(f"  {e:<30} {rankings_a[e]:>+10.4f}% {rankings_b[e]:>+10.4f}%")

    n_harmful = sum(1 for d in deltas_a if d < 0)
    n_helpful = sum(1 for d in deltas_a if d > 0)
    log(f"\n  Harmful (removal helps): {n_harmful}/{len(selected)}")
    log(f"  Helpful (removal hurts): {n_helpful}/{len(selected)}")

    overall = k1_pass and k2_pass and k3_pass
    log(f"\nOVERALL: {'SURVIVES' if overall else 'KILLED'}")

    # Save results
    results = {
        "config": {
            "n_experts": len(selected),
            "calib_samples_per_set": CALIB_SAMPLES_PER_SET,
            "max_seq_len": MAX_SEQ_LEN,
            "smoke_test": IS_SMOKE,
        },
        "reference_ppl": {"set_a": ref_ppl_a, "set_b": ref_ppl_b},
        "rankings": {
            e: {"delta_ppl_a_pct": rankings_a[e], "delta_ppl_b_pct": rankings_b[e]}
            for e in selected
        },
        "rank_order_worst_to_best": rank_order,
        "n_harmful": n_harmful,
        "n_helpful": n_helpful,
        "kill_criteria": {
            "K1_ppl_delta_std": ppl_variance,
            "K1_threshold": 0.1,
            "K1_pass": k1_pass,
            "K2_elapsed_s": elapsed,
            "K2_threshold_s": 4 * 3600,
            "K2_pass": k2_pass,
            "K3_kendall_tau": float(tau),
            "K3_kendall_p": float(p),
            "K3_threshold": 0.5,
            "K3_pass": k3_pass,
            "overall": "SURVIVES" if overall else "KILLED",
        },
        "elapsed_s": elapsed,
    }

    out_path = RESULTS_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
