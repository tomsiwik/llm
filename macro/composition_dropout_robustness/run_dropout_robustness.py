#!/usr/bin/env python3
"""Expert Dropout Composition Robustness at Macro Scale.

Tests whether composed model quality is robust to random expert subsets.
Composes random 80% subsets (40 of 50) and measures PPL variance across
20 bootstrap samples. If variance < 2%, individual expert quality is
noise-dominated and the composition is inherently robust.

Kill criteria:
- K1: quality variance across 20 random subsets > 5% (composition is fragile)
- K2: best 80% subset is >10% better than all-50 (pruning is critical, not optional)
- K3: worst 80% subset is >15% worse than all-50 (random pruning is dangerous)

Supports SMOKE_TEST=1 for <60s validation.
"""

import gc
import json
import math
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
RESULTS_DIR = REPO_ROOT / "results" / "composition_dropout_robustness"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BASE_MODEL = "Qwen/Qwen2.5-7B"
HF_CACHE = "/workspace/hf_cache"

MAX_SEQ_LEN = 512 if not IS_SMOKE else 128
CALIB_SAMPLES = 30 if not IS_SMOKE else 3
N_EXPERTS = 50 if not IS_SMOKE else 6
DROPOUT_FRAC = 0.8  # keep 80% of experts
N_BOOTSTRAP = 20 if not IS_SMOKE else 3
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


def load_calibration_data(adapters, n_samples):
    """Load generic calibration texts from adapter domains."""
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
            if len(all_texts) >= n_samples * 2:
                break
        if len(all_texts) >= n_samples * 2:
            break

    random.shuffle(all_texts)
    return all_texts[:n_samples]


def compute_ppl(model, tokenizer, texts):
    """Compute perplexity over texts."""
    import torch

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
    n_keep = max(2, int(N_EXPERTS * DROPOUT_FRAC))

    log("=" * 60)
    log("EXPERT DROPOUT COMPOSITION ROBUSTNESS")
    log(f"  N experts: {N_EXPERTS}")
    log(f"  Dropout: keep {n_keep}/{N_EXPERTS} ({DROPOUT_FRAC*100:.0f}%)")
    log(f"  Bootstrap samples: {N_BOOTSTRAP}")
    log(f"  Calibration samples: {CALIB_SAMPLES}")
    log(f"  Smoke test: {IS_SMOKE}")
    log("=" * 60)

    # Discover adapters
    all_adapters = discover_adapters()
    log(f"Found {len(all_adapters)} adapters")
    selected = all_adapters[:N_EXPERTS]
    log(f"Using first {len(selected)} adapters")

    # Load calibration data
    log("Loading calibration data...")
    calib = load_calibration_data(selected, CALIB_SAMPLES)
    log(f"Calibration: {len(calib)} texts")
    if len(calib) < 3:
        log("ERROR: insufficient calibration data")
        sys.exit(1)

    # Load tokenizer
    log("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, cache_dir=HF_CACHE, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Step 1: Reference — compose ALL N adapters
    log(f"\nComposing all {len(selected)} adapters (reference)...")
    ref_dir = compose_adapters_on_cpu(selected)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, cache_dir=HF_CACHE, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True
    )
    model = PeftModel.from_pretrained(model, ref_dir)
    model.eval()
    ref_ppl = compute_ppl(model, tokenizer, calib)
    log(f"Reference PPL (all {len(selected)}): {ref_ppl:.4f}")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    shutil.rmtree(ref_dir)

    # Step 2: Also measure base model PPL
    log("\nMeasuring base model PPL...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, cache_dir=HF_CACHE, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True
    )
    base_model.eval()
    base_ppl = compute_ppl(base_model, tokenizer, calib)
    log(f"Base PPL: {base_ppl:.4f}")
    del base_model
    gc.collect()
    torch.cuda.empty_cache()

    # Step 3: Bootstrap — random subsets
    bootstrap_results = []
    rng = random.Random(SEED)

    for b in range(N_BOOTSTRAP):
        subset = sorted(rng.sample(selected, n_keep))
        log(f"\n[Bootstrap {b+1}/{N_BOOTSTRAP}] {n_keep} experts")

        sub_dir = compose_adapters_on_cpu(subset)
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, cache_dir=HF_CACHE, torch_dtype=torch.bfloat16,
            device_map="auto", trust_remote_code=True
        )
        model = PeftModel.from_pretrained(model, sub_dir)
        model.eval()
        ppl = compute_ppl(model, tokenizer, calib)

        delta_from_ref = (ppl - ref_ppl) / ref_ppl * 100
        delta_from_base = (ppl - base_ppl) / base_ppl * 100

        log(f"  PPL={ppl:.4f} (vs ref: {delta_from_ref:+.3f}%, vs base: {delta_from_base:+.3f}%)")

        bootstrap_results.append({
            "bootstrap_id": b,
            "n_experts": n_keep,
            "subset": subset,
            "ppl": ppl,
            "delta_from_ref_pct": delta_from_ref,
            "delta_from_base_pct": delta_from_base,
        })

        del model
        gc.collect()
        torch.cuda.empty_cache()
        shutil.rmtree(sub_dir)

    elapsed = time.time() - t0

    # Step 4: Analysis
    log(f"\n{'='*60}")
    log("ANALYSIS")
    log(f"{'='*60}")

    ppls = np.array([r["ppl"] for r in bootstrap_results])
    deltas = np.array([r["delta_from_ref_pct"] for r in bootstrap_results])

    mean_ppl = float(np.mean(ppls))
    std_ppl = float(np.std(ppls))
    cv_ppl = std_ppl / mean_ppl * 100 if mean_ppl > 0 else 0
    best_delta = float(np.min(deltas))
    worst_delta = float(np.max(deltas))

    log(f"\nPPL distribution across {N_BOOTSTRAP} subsets:")
    log(f"  Mean: {mean_ppl:.4f}")
    log(f"  Std:  {std_ppl:.4f}")
    log(f"  CV:   {cv_ppl:.3f}%")
    log(f"  Best delta from ref: {best_delta:+.3f}%")
    log(f"  Worst delta from ref: {worst_delta:+.3f}%")
    log(f"  Reference (all {N_EXPERTS}): {ref_ppl:.4f}")
    log(f"  Base model: {base_ppl:.4f}")

    # Kill criteria
    k1_pass = cv_ppl <= 5.0
    k2_pass = abs(best_delta) <= 10.0
    k3_pass = worst_delta <= 15.0

    log(f"\nKILL CRITERIA:")
    log(f"  K1 (variance CV): {cv_ppl:.3f}% (threshold <=5%): "
        f"{'SURVIVES' if k1_pass else 'KILLED — composition is fragile'}")
    log(f"  K2 (best subset vs all): {best_delta:+.3f}% (threshold <=-10% means pruning critical): "
        f"{'SURVIVES' if k2_pass else 'KILLED — pruning is critical'}")
    log(f"  K3 (worst subset vs all): {worst_delta:+.3f}% (threshold <=15%): "
        f"{'SURVIVES' if k3_pass else 'KILLED — random dropout is dangerous'}")

    overall = k1_pass and k2_pass and k3_pass
    log(f"\n  OVERALL: {'SURVIVES — composition is robust to 20% random expert dropout' if overall else 'KILLED'}")

    # Save results
    results = {
        "config": {
            "n_experts": N_EXPERTS,
            "n_keep": n_keep,
            "dropout_frac": DROPOUT_FRAC,
            "n_bootstrap": N_BOOTSTRAP,
            "calib_samples": CALIB_SAMPLES,
            "smoke_test": IS_SMOKE,
        },
        "reference_ppl": ref_ppl,
        "base_ppl": base_ppl,
        "bootstrap_summary": {
            "mean_ppl": mean_ppl,
            "std_ppl": std_ppl,
            "cv_pct": cv_ppl,
            "best_delta_pct": best_delta,
            "worst_delta_pct": worst_delta,
        },
        "kill_criteria": {
            "K1_cv_pct": cv_ppl,
            "K1_threshold": 5.0,
            "K1_pass": k1_pass,
            "K2_best_delta_pct": best_delta,
            "K2_threshold": 10.0,
            "K2_pass": k2_pass,
            "K3_worst_delta_pct": worst_delta,
            "K3_threshold": 15.0,
            "K3_pass": k3_pass,
            "overall": "SURVIVES" if overall else "KILLED",
        },
        "bootstrap_details": bootstrap_results,
        "elapsed_s": elapsed,
    }

    out_path = RESULTS_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
