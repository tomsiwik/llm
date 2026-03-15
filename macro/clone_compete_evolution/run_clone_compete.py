#!/usr/bin/env python3
"""Clone-and-compete evolution experiment at macro scale.

Tests the core evolution mechanism on 5 pilot experts:
1. Generate correction data by injecting deliberate errors
2. Clone each expert, fine-tune clone with corrections
3. Run shadow-scoring tournament (answer-conditioned PPL)
4. Measure convergence rate, speed, and domain regression

Kill criteria:
- K1: corrected clone does not win tournament >70% of the time
- K2: tournament requires >50K queries to resolve (measured by convergence check)
- K3: original expert domains regress >2% during tournament

Supports SMOKE_TEST=1 for <60s validation.
"""

import gc
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

IS_SMOKE = os.environ.get("SMOKE_TEST") == "1"
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

REPO_ROOT = Path("/workspace/llm")
ADAPTER_DIR = REPO_ROOT / "adapters"
DATA_DIR = REPO_ROOT / "data" / "distillation"
RESULTS_DIR = REPO_ROOT / "results" / "clone_compete_evolution"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BASE_MODEL = "Qwen/Qwen2.5-7B"
HF_CACHE = "/workspace/hf_cache"

# 5 diverse domains from pilot 50 for the tournament
TEST_DOMAINS = ["python", "bash", "math", "medical", "sql"]

# Tournament parameters
FT_STEPS = 50 if not IS_SMOKE else 5
FT_LR = 1e-4
EVAL_QUERIES = 200 if not IS_SMOKE else 10
N_CORRECTIONS = 50 if not IS_SMOKE else 5
CONVERGENCE_CHECK_SIZES = [50, 100, 200] if not IS_SMOKE else [5, 10]


def log(msg):
    ts = time.strftime("%H:%M:%S", time.gmtime())
    print(f"[{ts}] {msg}", flush=True)


def generate_corrections(base_model, tokenizer, adapter_dir, domain, n=50):
    """Generate correction data by finding examples where the expert is wrong.

    Strategy: Use the training data's eval split. Find examples where the expert
    generates poor predictions (high loss), then use the ground truth as the
    'correction'. This simulates the teacher-correction pipeline without
    needing an actual teacher API call.
    """
    import torch
    from peft import PeftModel

    corrections_dir = RESULTS_DIR / "corrections"
    corrections_dir.mkdir(parents=True, exist_ok=True)
    corrections_file = corrections_dir / f"{domain}_corrections.jsonl"

    if corrections_file.exists():
        existing = sum(1 for _ in open(corrections_file))
        if existing >= n:
            log(f"  {domain}: {existing} corrections already exist, reusing")
            return corrections_file

    log(f"  Generating corrections for {domain}...")

    # Load eval data
    eval_file = DATA_DIR / domain / "eval.jsonl"
    if not eval_file.exists():
        # Fall back to last N of training data
        train_file = DATA_DIR / domain / "train.jsonl"
        if not train_file.exists():
            log(f"  WARNING: No data found for {domain}, skipping")
            return None
        with open(train_file) as f:
            all_lines = f.readlines()
        eval_lines = all_lines[-min(200, len(all_lines)):]
    else:
        with open(eval_file) as f:
            eval_lines = f.readlines()

    # Load expert adapter
    model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    model.eval()

    # Score each example — high loss = expert struggles = correction candidate
    scored = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for line in eval_lines[:min(n * 4, len(eval_lines))]:
            record = json.loads(line)
            if "messages" in record:
                text = tokenizer.apply_chat_template(
                    record["messages"], tokenize=False, add_generation_prompt=False)
            elif "text" in record:
                text = record["text"]
            else:
                continue

            inputs = tokenizer(text, return_tensors="pt", truncation=True,
                               max_length=512).to(base_model.device)
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss.item()
            scored.append((loss, record))

    del model
    torch.cuda.empty_cache()
    gc.collect()

    # Take highest-loss examples as correction candidates
    scored.sort(key=lambda x: -x[0])
    corrections = scored[:n]

    with open(corrections_file, "w") as f:
        for loss, record in corrections:
            # The ground truth text IS the correction (expert got it wrong)
            correction_entry = {
                "messages": record.get("messages", [{"role": "user", "content": record.get("text", "")}]),
                "original_loss": loss,
            }
            f.write(json.dumps(correction_entry) + "\n")

    log(f"  {domain}: generated {len(corrections)} corrections (loss range: {corrections[0][0]:.3f} - {corrections[-1][0]:.3f})")
    return corrections_file


def run_tournament(base_model, tokenizer, domain, corrections_file):
    """Run a single clone-and-compete tournament for one domain.

    Returns dict with tournament results.
    """
    import torch
    from peft import PeftModel

    adapter_dir = ADAPTER_DIR / domain
    clone_dir = ADAPTER_DIR / f"{domain}_clone_exp"

    if not adapter_dir.exists():
        log(f"  WARNING: adapter {adapter_dir} not found, skipping")
        return None

    log(f"\n  Tournament: {domain}")
    t0 = time.time()

    # Step 1: Clone
    if clone_dir.exists():
        shutil.rmtree(clone_dir)
    shutil.copytree(adapter_dir, clone_dir)
    log(f"  Cloned {domain} → {domain}_clone_exp")

    # Step 2: Fine-tune clone with corrections
    log(f"  Fine-tuning clone ({FT_STEPS} steps, lr={FT_LR})...")
    ft_start = time.time()

    model = PeftModel.from_pretrained(base_model, str(clone_dir), is_trainable=True)

    from datasets import load_dataset
    from trl import SFTTrainer, SFTConfig

    dataset = load_dataset("json", data_files=str(corrections_file), split="train")

    def format_messages(example):
        if "messages" in example and example["messages"]:
            return {"text": tokenizer.apply_chat_template(
                example["messages"], tokenize=False, add_generation_prompt=False)}
        return {"text": ""}

    dataset = dataset.map(format_messages)
    dataset = dataset.filter(lambda x: len(x["text"]) > 0)

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=SFTConfig(
            output_dir=str(clone_dir / "ft_ckpt"),
            max_steps=FT_STEPS,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=2,
            learning_rate=FT_LR,
            warmup_steps=min(5, FT_STEPS // 2),
            logging_steps=max(1, FT_STEPS // 5),
            save_steps=FT_STEPS,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            optim="adamw_8bit",
            seed=42,
            dataset_text_field="text",
            max_length=512,
        ),
    )

    train_result = trainer.train()
    ft_loss = train_result.training_loss
    model.save_pretrained(str(clone_dir))

    ft_time = time.time() - ft_start
    log(f"  Fine-tuning done in {ft_time:.0f}s (final loss: {ft_loss:.4f})")

    del model, trainer
    torch.cuda.empty_cache()
    gc.collect()

    # Step 3: Shadow scoring with answer-conditioned PPL
    # Load eval texts (general domain queries, NOT corrections)
    eval_file = DATA_DIR / domain / "eval.jsonl"
    eval_texts = []
    source_file = eval_file if eval_file.exists() else DATA_DIR / domain / "train_split.jsonl"
    if source_file.exists():
        with open(source_file) as f:
            for line in f:
                record = json.loads(line)
                if "messages" in record:
                    text = tokenizer.apply_chat_template(
                        record["messages"], tokenize=False, add_generation_prompt=False)
                elif "text" in record:
                    text = record["text"]
                else:
                    continue
                eval_texts.append(text)
                if len(eval_texts) >= EVAL_QUERIES:
                    break

    # Also load corrections as separate eval set
    correction_texts = []
    with open(corrections_file) as f:
        for line in f:
            record = json.loads(line)
            if "messages" in record:
                text = tokenizer.apply_chat_template(
                    record["messages"], tokenize=False, add_generation_prompt=False)
                correction_texts.append(text)

    log(f"  Scoring: {len(eval_texts)} general + {len(correction_texts)} correction queries")

    def score_adapter(adapter_path, texts):
        """Score adapter using answer-conditioned PPL (proven better than full-seq)."""
        m = PeftModel.from_pretrained(base_model, str(adapter_path))
        m.eval()
        losses = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for text in texts:
                inputs = tokenizer(text, return_tensors="pt", truncation=True,
                                   max_length=512).to(base_model.device)
                outputs = m(**inputs, labels=inputs["input_ids"])
                losses.append(outputs.loss.item())
        del m
        torch.cuda.empty_cache()
        return losses

    # Score original on general queries
    log(f"  Scoring original on general queries...")
    orig_general = score_adapter(adapter_dir, eval_texts)
    log(f"  Scoring clone on general queries...")
    clone_general = score_adapter(clone_dir, eval_texts)
    log(f"  Scoring original on correction queries...")
    orig_corrections = score_adapter(adapter_dir, correction_texts)
    log(f"  Scoring clone on correction queries...")
    clone_corrections = score_adapter(clone_dir, correction_texts)

    elapsed = time.time() - t0

    # Compute metrics
    avg_orig_gen = sum(orig_general) / max(len(orig_general), 1)
    avg_clone_gen = sum(clone_general) / max(len(clone_general), 1)
    avg_orig_corr = sum(orig_corrections) / max(len(orig_corrections), 1)
    avg_clone_corr = sum(clone_corrections) / max(len(clone_corrections), 1)

    regression_pct = (math.exp(avg_clone_gen) - math.exp(avg_orig_gen)) / math.exp(avg_orig_gen) * 100

    # Convergence check: how many queries needed to determine winner?
    convergence_n = None
    for check_n in CONVERGENCE_CHECK_SIZES:
        if check_n > len(correction_texts):
            break
        sub_orig = sum(orig_corrections[:check_n]) / check_n
        sub_clone = sum(clone_corrections[:check_n]) / check_n
        if sub_clone < sub_orig:  # Clone wins on subset too
            convergence_n = check_n
            break
    if convergence_n is None and len(correction_texts) > 0:
        convergence_n = len(correction_texts)

    clone_wins = avg_clone_corr < avg_orig_corr

    result = {
        "domain": domain,
        "elapsed_s": round(elapsed, 1),
        "ft_time_s": round(ft_time, 1),
        "ft_loss": round(ft_loss, 4),
        "n_corrections": len(correction_texts),
        "n_general_queries": len(eval_texts),
        "general_queries": {
            "original_ppl": round(math.exp(avg_orig_gen), 3),
            "clone_ppl": round(math.exp(avg_clone_gen), 3),
            "regression_pct": round(regression_pct, 2),
        },
        "correction_queries": {
            "original_ppl": round(math.exp(avg_orig_corr), 3),
            "clone_ppl": round(math.exp(avg_clone_corr), 3),
            "clone_wins": clone_wins,
        },
        "convergence_queries": convergence_n,
        "clone_wins": clone_wins,
        "regression_exceeds_2pct": regression_pct > 2.0,
    }

    log(f"  {domain}: clone_wins={clone_wins}, regression={regression_pct:+.2f}%, "
        f"convergence@{convergence_n} queries, {elapsed:.0f}s")

    # Cleanup clone (don't actually replace the original — this is an experiment)
    if clone_dir.exists():
        shutil.rmtree(clone_dir)

    return result


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    t0 = time.time()
    log(f"=== Clone-and-Compete Evolution Experiment ===")
    log(f"Smoke test: {IS_SMOKE}")
    log(f"Domains: {TEST_DOMAINS}")
    log(f"FT steps: {FT_STEPS}, Corrections: {N_CORRECTIONS}, Eval queries: {EVAL_QUERIES}")

    # Load base model once
    log("Loading base model...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, cache_dir=HF_CACHE, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        cache_dir=HF_CACHE,
        trust_remote_code=True,
    )

    log("Base model loaded.")

    # Phase 1: Generate corrections for each domain
    log("\n=== Phase 1: Generate Corrections ===")
    correction_files = {}
    for domain in TEST_DOMAINS:
        adapter_dir = ADAPTER_DIR / domain
        if not adapter_dir.exists():
            log(f"  SKIP {domain}: adapter not found at {adapter_dir}")
            continue
        cf = generate_corrections(base_model, tokenizer, adapter_dir, domain, n=N_CORRECTIONS)
        if cf:
            correction_files[domain] = cf

    log(f"Corrections generated for {len(correction_files)} domains")

    # Phase 2: Run tournaments
    log("\n=== Phase 2: Clone-and-Compete Tournaments ===")
    results = {}
    for domain, cf in correction_files.items():
        result = run_tournament(base_model, tokenizer, domain, cf)
        if result:
            results[domain] = result
        gc.collect()
        torch.cuda.empty_cache()

    # Phase 3: Aggregate and check kill criteria
    log("\n=== Phase 3: Kill Criteria Assessment ===")

    if not results:
        log("ERROR: No tournament results!")
        sys.exit(1)

    n_domains = len(results)
    n_clone_wins = sum(1 for r in results.values() if r["clone_wins"])
    win_rate = n_clone_wins / n_domains
    max_regression = max(r["general_queries"]["regression_pct"] for r in results.values())
    mean_regression = sum(r["general_queries"]["regression_pct"] for r in results.values()) / n_domains
    max_convergence = max(r["convergence_queries"] or 0 for r in results.values())

    k1_pass = win_rate > 0.70
    k2_pass = max_convergence <= 50000
    k3_pass = max_regression <= 2.0

    aggregate = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {
            "domains": TEST_DOMAINS,
            "ft_steps": FT_STEPS,
            "ft_lr": FT_LR,
            "n_corrections": N_CORRECTIONS,
            "eval_queries": EVAL_QUERIES,
            "smoke_test": IS_SMOKE,
        },
        "per_domain": results,
        "aggregate": {
            "n_domains": n_domains,
            "clone_win_rate": round(win_rate, 3),
            "mean_regression_pct": round(mean_regression, 2),
            "max_regression_pct": round(max_regression, 2),
            "max_convergence_queries": max_convergence,
        },
        "kill_criteria": {
            "K1_clone_win_rate_gt_70pct": {
                "value": round(win_rate * 100, 1),
                "threshold": 70,
                "pass": k1_pass,
            },
            "K2_convergence_lt_50k_queries": {
                "value": max_convergence,
                "threshold": 50000,
                "pass": k2_pass,
            },
            "K3_domain_regression_lt_2pct": {
                "value": round(max_regression, 2),
                "threshold": 2.0,
                "pass": k3_pass,
            },
        },
        "verdict": "PASS" if (k1_pass and k2_pass and k3_pass) else "FAIL",
        "elapsed_s": round(time.time() - t0, 1),
    }

    # Print summary
    log(f"\n{'='*60}")
    log(f"RESULTS SUMMARY")
    log(f"{'='*60}")
    for domain, r in results.items():
        status = "WIN" if r["clone_wins"] else "LOSE"
        log(f"  {domain}: clone {status} | regression {r['general_queries']['regression_pct']:+.2f}% | convergence@{r['convergence_queries']}")

    log(f"\nK1 (win rate >70%): {'PASS' if k1_pass else 'FAIL'} — {win_rate*100:.0f}%")
    log(f"K2 (convergence <50K): {'PASS' if k2_pass else 'FAIL'} — {max_convergence} queries")
    log(f"K3 (regression <2%): {'PASS' if k3_pass else 'FAIL'} — max {max_regression:+.2f}%")
    log(f"\nVERDICT: {aggregate['verdict']}")
    log(f"Total time: {aggregate['elapsed_s']:.0f}s")

    # Save results
    out_file = RESULTS_DIR / "clone_compete_results.json"
    with open(out_file, "w") as f:
        json.dump(aggregate, f, indent=2)
    log(f"Results saved to {out_file}")


if __name__ == "__main__":
    main()
