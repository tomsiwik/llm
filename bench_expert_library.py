"""Expert Library Scaling Benchmark: Experiments 4-5.

Exp 4: Scale to 5 languages — does the library scale? Do subspaces stay orthogonal?
Exp 5: Beat Qwen2.5-Coder-1.5B — can 0.5B + N experts match a 3x larger model?

Usage:
    # Quick mode (50 steps, 5 languages)
    PYTHONUNBUFFERED=1 uv run --with mlx,mlx-lm,datasets \
      python bench_expert_library.py --quick

    # Full mode (200 steps, 5 languages)
    PYTHONUNBUFFERED=1 uv run --with mlx,mlx-lm,datasets \
      python bench_expert_library.py --full

    # Full + 1.5B baseline comparison
    PYTHONUNBUFFERED=1 uv run --with mlx,mlx-lm,datasets \
      python bench_expert_library.py --full --baseline=1.5B
"""

import argparse
import time
import math
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from tribe.llm import load_backbone, patch_with_standard_lora, patch_with_library, \
    freeze_base, compute_perplexity
from tribe.lora_standard import collect_standard_lora_layers
from tribe.lora_library import collect_library_layers
from tribe.routing_calibration import (
    extract_calibration_features, calibrate_routing_keys,
    evaluate_routing_accuracy,
)

# ── Config ──────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen2.5-Coder-0.5B"
MODEL_1_5B = "Qwen/Qwen2.5-Coder-1.5B"
RANK = 16
SCALE = 16.0
TARGETS = ("q_proj", "v_proj")
SEQ_LEN = 256
BATCH_SIZE = 4

LANGUAGES = ["python", "javascript", "go", "rust", "ruby"]

# Dataset mappings — streaming from HuggingFace
DATASET_MAP = {
    "python": ("Nan-Do/code-search-net-python", "code", None),
    "javascript": ("Nan-Do/code-search-net-javascript", "code", None),
    "go": ("Nan-Do/code-search-net-go", "code", None),
    "rust": ("bigcode/the-stack-dedup", "content", "rust"),
    "ruby": ("Nan-Do/code-search-net-ruby", "code", None),
}


def parse_args():
    p = argparse.ArgumentParser(description="Expert Library Scaling Benchmark")
    p.add_argument("--quick", action="store_true", help="Quick mode (50 steps)")
    p.add_argument("--full", action="store_true", help="Full mode (200 steps)")
    p.add_argument("--steps", type=int, default=None, help="Override training steps")
    p.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    p.add_argument("--baseline", type=str, default=None,
                   help="Compare against larger model (e.g., '1.5B')")
    p.add_argument("--rank", type=int, default=16, help="LoRA rank")
    p.add_argument("--langs", type=str, default=None,
                   help="Comma-separated languages (default: all 5)")
    return p.parse_args()


# ── Data Loading ────────────────────────────────────────────

def load_code_domain(lang, tokenizer, n_train=300, n_eval=30):
    """Load code data for a language."""
    from datasets import load_dataset

    ds_name, text_field, subset = DATASET_MAP[lang]
    print(f"    Loading {lang} data (streaming)...", end=" ", flush=True)

    try:
        if subset:
            ds = load_dataset(ds_name, data_dir=subset, split="train", streaming=True)
        else:
            ds = load_dataset(ds_name, split="train", streaming=True)

        texts = []
        needed = n_train + n_eval
        for ex in ds:
            content = ex.get(text_field, "")
            if len(content.strip()) > 100:
                texts.append(content[:2048])
            if len(texts) >= needed:
                break

        if len(texts) < needed:
            print(f"WARNING: only got {len(texts)} samples", end=" ")
    except Exception as e:
        print(f"FALLBACK ({e})", end=" ")
        # Generate synthetic data as fallback
        texts = _generate_synthetic(lang, n_train + n_eval)

    rng = np.random.RandomState(42)
    rng.shuffle(texts)
    eval_texts = texts[:n_eval]
    train_texts = texts[n_eval:n_eval + n_train]

    all_tokens = []
    for text in train_texts:
        tokens = tokenizer.encode(text)
        all_tokens.extend(tokens)

    train_seqs = []
    for i in range(0, len(all_tokens) - SEQ_LEN, SEQ_LEN):
        seq = all_tokens[i:i + SEQ_LEN]
        train_seqs.append(mx.array(seq))

    print(f"{len(train_seqs)} seqs, {len(eval_texts)} eval")
    return train_seqs, eval_texts


def _generate_synthetic(lang, n_samples):
    """Generate minimal synthetic code samples as fallback."""
    templates = {
        "rust": [
            "fn main() {\n    let x: i32 = 42;\n    println!(\"{}\", x);\n}\n",
            "struct Point {\n    x: f64,\n    y: f64,\n}\nimpl Point {\n    fn new(x: f64, y: f64) -> Self {\n        Point { x, y }\n    }\n}\n",
            "use std::collections::HashMap;\nfn count_words(text: &str) -> HashMap<&str, usize> {\n    let mut counts = HashMap::new();\n    for word in text.split_whitespace() {\n        *counts.entry(word).or_insert(0) += 1;\n    }\n    counts\n}\n",
        ],
        "go": [
            "package main\nimport \"fmt\"\nfunc main() {\n    fmt.Println(\"hello\")\n}\n",
            "func fibonacci(n int) int {\n    if n <= 1 {\n        return n\n    }\n    return fibonacci(n-1) + fibonacci(n-2)\n}\n",
        ],
        "ruby": [
            "class Animal\n  attr_accessor :name\n  def initialize(name)\n    @name = name\n  end\n  def speak\n    \"...\"\n  end\nend\n",
            "def fibonacci(n)\n  return n if n <= 1\n  fibonacci(n - 1) + fibonacci(n - 2)\nend\n",
        ],
    }
    t = templates.get(lang, templates["go"])
    return [t[i % len(t)] * 3 for i in range(n_samples)]


# ── LoRA Training ───────────────────────────────────────────

def train_adapter(lang, train_seqs, steps, lr, rank=16):
    """Train a LoRA adapter on a single domain."""
    model, tokenizer = load_backbone(MODEL_NAME)
    patch_with_standard_lora(model, rank=rank, scale=SCALE, targets=TARGETS)
    freeze_base(model)

    n_params = sum(l.d_in * l.rank + l.rank * l.d_out
                   for _, l in collect_standard_lora_layers(model))
    print(f"    LoRA params: {n_params:,} (rank={rank})")

    optimizer = optim.Adam(learning_rate=lr)

    def loss_fn(model, tokens):
        logits = model(tokens[:, :-1])
        targets = tokens[:, 1:]
        return nn.losses.cross_entropy(logits, targets, reduction='mean')

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    n_seqs = len(train_seqs)

    for step in range(steps):
        idx = np.random.randint(0, n_seqs, size=min(BATCH_SIZE, n_seqs))
        batch = mx.stack([train_seqs[i] for i in idx])

        loss, grads = loss_and_grad(model, batch)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        if (step + 1) % max(steps // 4, 1) == 0 or step == 0:
            ppl = math.exp(loss.item()) if loss.item() < 20 else float('inf')
            print(f"      step {step+1:4d}/{steps}: loss={loss.item():.3f}, ppl={ppl:.1f}")

    # Extract weights
    weights = {}
    for name, layer in collect_standard_lora_layers(model):
        weights[name] = (mx.array(layer.lora_A), mx.array(layer.lora_B))

    del model, optimizer
    return weights


# ── Subspace Analysis ───────────────────────────────────────

def analyze_subspaces(all_lora, languages):
    """Measure pairwise cosine similarity between A matrices."""
    print(f"\n  Subspace Overlap Matrix (cosine of A matrices):")
    n = len(languages)

    # Use first layer for analysis
    first_name = list(all_lora[languages[0]].keys())[0]

    # Compute pairwise cosines
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            A_i, _ = all_lora[languages[i]][first_name]
            A_j, _ = all_lora[languages[j]][first_name]
            a_flat = A_i.reshape(-1)
            b_flat = A_j.reshape(-1)
            mx.eval(a_flat, b_flat)
            cos = (mx.sum(a_flat * b_flat) / (mx.sqrt(mx.sum(a_flat**2)) *
                   mx.sqrt(mx.sum(b_flat**2)) + 1e-8)).item()
            matrix[i, j] = cos

    # Print matrix
    header = "  " + " " * 12 + "  ".join(f"{l:>8s}" for l in languages)
    print(header)
    for i, lang in enumerate(languages):
        row = f"  {lang:>10s}  " + "  ".join(f"{matrix[i,j]:8.4f}" for j in range(n))
        print(row)

    # Mean off-diagonal overlap
    off_diag = []
    for i in range(n):
        for j in range(n):
            if i != j:
                off_diag.append(abs(matrix[i, j]))
    print(f"\n  Mean |off-diagonal| cosine: {np.mean(off_diag):.4f}")
    print(f"  Max  |off-diagonal| cosine: {np.max(off_diag):.4f}")

    return matrix


# ── Main Benchmark ──────────────────────────────────────────

def main():
    args = parse_args()
    global RANK
    RANK = args.rank
    train_steps = args.steps or (50 if args.quick else 200)
    n_train = 150 if args.quick else 300
    n_eval = 15 if args.quick else 30

    languages = args.langs.split(",") if args.langs else LANGUAGES

    t0 = time.time()

    # ── Phase 0: Load data ──────────────────────────────────
    print("=" * 72)
    print(f"  Expert Library Benchmark: {len(languages)} languages")
    print(f"  Model: {MODEL_NAME}, Rank: {RANK}, Steps: {train_steps}")
    print("=" * 72)

    _, tokenizer = load_backbone(MODEL_NAME)
    domain_data = {}
    for lang in languages:
        train_seqs, eval_texts = load_code_domain(lang, tokenizer, n_train, n_eval)
        domain_data[lang] = (train_seqs, eval_texts)

    # ── Phase 1: Baseline perplexity ────────────────────────
    print(f"\n{'=' * 72}")
    print(f"  PHASE 1: Baseline perplexity (no LoRA)")
    print(f"{'=' * 72}")

    model_base, _ = load_backbone(MODEL_NAME)
    base_ppls = {}
    for lang in languages:
        _, eval_texts = domain_data[lang]
        ppl = compute_perplexity(model_base, tokenizer, eval_texts)
        base_ppls[lang] = ppl
        print(f"    {lang}: {ppl:.2f}")
    del model_base

    # ── Phase 2: Train per-language adapters ─────────────────
    print(f"\n{'=' * 72}")
    print(f"  PHASE 2: Train {len(languages)} adapters ({train_steps} steps each)")
    print(f"{'=' * 72}")

    all_lora = {}
    individual_ppls = {}  # lang → {eval_lang: ppl}

    for lang in languages:
        print(f"\n  Training {lang} adapter...")
        train_seqs, _ = domain_data[lang]
        lora_w = train_adapter(lang, train_seqs, train_steps, args.lr, rank=RANK)
        all_lora[lang] = lora_w

        # Eval individual adapter on all languages
        model_ind, _ = load_backbone(MODEL_NAME)
        patch_with_standard_lora(model_ind, rank=RANK, scale=SCALE, targets=TARGETS)
        freeze_base(model_ind)
        for name, layer in collect_standard_lora_layers(model_ind):
            if name in lora_w:
                A, B = lora_w[name]
                layer.lora_A = A
                layer.lora_B = B
        mx.eval(model_ind.parameters())

        individual_ppls[lang] = {}
        for eval_lang in languages:
            _, eval_texts = domain_data[eval_lang]
            ppl = compute_perplexity(model_ind, tokenizer, eval_texts)
            individual_ppls[lang][eval_lang] = ppl
        own_ppl = individual_ppls[lang][lang]
        print(f"    {lang} adapter → {lang}: {own_ppl:.2f}")
        del model_ind

    # ── Phase 3: Subspace analysis ──────────────────────────
    print(f"\n{'=' * 72}")
    print(f"  PHASE 3: Subspace analysis")
    print(f"{'=' * 72}")

    overlap_matrix = analyze_subspaces(all_lora, languages)

    # ── Phase 4: Expert library evaluation ──────────────────
    print(f"\n{'=' * 72}")
    print(f"  PHASE 4: Expert library (all {len(languages)} experts)")
    print(f"{'=' * 72}")

    lora_scale = SCALE / RANK
    lora_list = [all_lora[lang] for lang in languages]

    library_ppls = {}
    for top_k in [1, 2, len(languages)]:
        k_label = f"top_{top_k}" if top_k < len(languages) else "full"
        print(f"\n  Library top_k={top_k} ({k_label}):")

        model_lib, _ = load_backbone(MODEL_NAME)
        patch_with_library(
            model_lib, lora_list,
            labels=languages, top_k=top_k, scale=lora_scale, targets=TARGETS,
        )
        mx.eval(model_lib.parameters())

        ppls = {}
        for eval_lang in languages:
            _, eval_texts = domain_data[eval_lang]
            ppl = compute_perplexity(model_lib, tokenizer, eval_texts)
            ppls[eval_lang] = ppl
            print(f"    {eval_lang}: {ppl:.2f}")

        # Routing stats
        print(f"    Routing distribution (first layer):")
        lib_layers = collect_library_layers(model_lib)
        if lib_layers:
            for eval_lang in languages[:3]:
                _, eval_texts = domain_data[eval_lang]
                loads = []
                for text in eval_texts[:5]:
                    tokens = tokenizer.encode(text)
                    if len(tokens) > SEQ_LEN:
                        tokens = tokens[:SEQ_LEN]
                    input_ids = mx.array([tokens])
                    x = model_lib.model.embed_tokens(input_ids)
                    mx.eval(x)
                    stats = lib_layers[0][1].routing_stats(x.reshape(-1, x.shape[-1]))
                    loads.append(stats["expert_load"])
                if loads:
                    mean_load = np.mean(loads, axis=0)
                    load_str = ", ".join(f"{languages[i]}={mean_load[i]:.1%}"
                                         for i in range(len(mean_load)))
                    print(f"      {eval_lang}: {load_str}")

        library_ppls[k_label] = ppls
        del model_lib

    # ── Phase 4b: Calibrated routing ────────────────────────
    print(f"\n{'=' * 72}")
    print(f"  PHASE 4b: Calibrated routing ({len(languages)}-way contrastive)")
    print(f"{'=' * 72}")

    cal_steps = 25 if args.quick else 50

    # Split eval data: odd-indexed for calibration, even-indexed for hold-out
    cal_texts = {}   # calibration features
    ho_texts = {}    # hold-out accuracy evaluation
    for lang in languages:
        _, eval_texts = domain_data[lang]
        cal_texts[lang] = [t for i, t in enumerate(eval_texts) if i % 2 == 1]
        ho_texts[lang] = [t for i, t in enumerate(eval_texts) if i % 2 == 0]

    # Build library model (top_k=1) for calibrated routing
    lora_scale = SCALE / RANK
    lora_list = [all_lora[lang] for lang in languages]
    model_cal, _ = load_backbone(MODEL_NAME)
    patch_with_library(
        model_cal, lora_list,
        labels=languages, top_k=1, scale=lora_scale, targets=TARGETS,
    )
    mx.eval(model_cal.parameters())

    # Extract features from base model (no LoRA influence)
    model_feat, _ = load_backbone(MODEL_NAME)
    print(f"\n  Extracting calibration features...")
    cal_features = extract_calibration_features(model_feat, tokenizer, cal_texts)
    print(f"  Extracting hold-out features...")
    ho_features = extract_calibration_features(model_feat, tokenizer, ho_texts)
    del model_feat

    # Initialize routing keys (SVD warm-start from A matrices)
    for _, lib in collect_library_layers(model_cal):
        lib.initialize_routing_keys(d_key=8, init_from_A=True)
    mx.eval(model_cal.parameters())

    # Pre-calibration accuracy (SVD init baseline)
    print(f"\n  Pre-calibration routing accuracy (SVD init):")
    pre_result = evaluate_routing_accuracy(model_cal, ho_features)
    pre_acc = pre_result["mean_accuracy"]

    # Calibrate
    print(f"\n  Calibrating routing keys ({cal_steps} steps)...")
    calibrate_routing_keys(model_cal, cal_features, steps=cal_steps,
                           lr=1e-3, temperature=0.1, verbose=True)

    # Post-calibration accuracy on hold-out
    print(f"\n  Post-calibration routing accuracy (hold-out):")
    post_result = evaluate_routing_accuracy(model_cal, ho_features)
    post_acc = post_result["mean_accuracy"]

    # Calibrated library PPL (on ALL eval texts, same as Phase 4)
    print(f"\n  Calibrated library PPL:")
    cal_ppls = {}
    for eval_lang in languages:
        _, eval_texts = domain_data[eval_lang]
        ppl = compute_perplexity(model_cal, tokenizer, eval_texts)
        cal_ppls[eval_lang] = ppl
        print(f"    {eval_lang}: {ppl:.2f}")
    library_ppls["calibrated"] = cal_ppls

    # Summary
    print(f"\n  Routing accuracy: {pre_acc:.1%} (SVD init) → {post_acc:.1%} (calibrated)")
    top1_ppls = library_ppls.get("top_1", {})
    if top1_ppls:
        for lang in languages:
            a_ppl = top1_ppls.get(lang, float('inf'))
            c_ppl = cal_ppls.get(lang, float('inf'))
            delta = c_ppl - a_ppl
            print(f"    {lang}: A-matrix={a_ppl:.2f} → calibrated={c_ppl:.2f} "
                  f"({'+'if delta>=0 else ''}{delta:.2f})")

    del model_cal

    # ── Phase 5: 1.5B Baseline (optional) ───────────────────
    baseline_ppls = None
    if args.baseline == "1.5B":
        print(f"\n{'=' * 72}")
        print(f"  PHASE 5: 1.5B Baseline ({MODEL_1_5B})")
        print(f"{'=' * 72}")

        try:
            model_big, tokenizer_big = load_backbone(MODEL_1_5B)
            baseline_ppls = {}
            for lang in languages:
                _, eval_texts = domain_data[lang]
                ppl = compute_perplexity(model_big, tokenizer_big, eval_texts)
                baseline_ppls[lang] = ppl
                print(f"    {lang}: {ppl:.2f}")
            del model_big
        except Exception as e:
            print(f"    Failed to load 1.5B model: {e}")
            print(f"    Skipping 1.5B comparison.")

    # ── Results Table ───────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'=' * 72}")
    print(f"  RESULTS ({elapsed:.0f}s total)")
    print(f"{'=' * 72}")

    # Param counts
    n_base = 494_032_000  # Qwen2.5-Coder-0.5B approximate
    n_lora_per = sum(A.shape[0] * A.shape[1] + B.shape[0] * B.shape[1]
                     for A, B in list(all_lora[languages[0]].values()))
    n_experts = len(languages)
    n_total_experts = n_lora_per * n_experts
    print(f"\n  Params: base={n_base/1e6:.1f}M, per-expert={n_lora_per/1e6:.2f}M, "
          f"total experts={n_total_experts/1e6:.2f}M ({n_experts}x)")
    print(f"  Active per token (top-1): base + 1 expert = "
          f"{(n_base + n_lora_per)/1e6:.1f}M")

    # Per-language comparison table
    header = f"  {'Method':<20s}"
    for lang in languages:
        header += f" | {lang:>8s}"
    header += f" | {'Mean':>8s}"
    print(f"\n{header}")
    print(f"  {'-'*20}" + "-+-" + "-+-".join("-" * 8 for _ in languages) + f"-+-{'-'*8}")

    def print_row(label, ppls_dict):
        row = f"  {label:<20s}"
        vals = []
        for lang in languages:
            v = ppls_dict.get(lang, float('inf'))
            row += f" | {v:8.2f}"
            vals.append(v)
        row += f" | {np.mean(vals):8.2f}"
        print(row)

    print_row("base (no LoRA)", base_ppls)
    for lang in languages:
        print_row(f"lora_{lang}", individual_ppls[lang])
    for k_label in library_ppls:
        print_row(f"library_{k_label}", library_ppls[k_label])
    if baseline_ppls:
        print_row("qwen_1.5B", baseline_ppls)

    # ── Verdict ─────────────────────────────────────────────
    print(f"\n  Verdict:")

    # Check 1: Each language improves over base
    best_lib = library_ppls.get("top_1", library_ppls.get("full", {}))
    improvements = 0
    for lang in languages:
        lib_ppl = best_lib.get(lang, float('inf'))
        base_ppl = base_ppls.get(lang, float('inf'))
        if lib_ppl < base_ppl:
            improvements += 1
    print(f"    Languages improved over base: {improvements}/{len(languages)} "
          f"({'PASS' if improvements == len(languages) else 'PARTIAL'})")

    # Check 2: No language regresses vs individual adapter
    regressions = 0
    for lang in languages:
        lib_ppl = best_lib.get(lang, float('inf'))
        ind_ppl = individual_ppls[lang].get(lang, float('inf'))
        if lib_ppl > ind_ppl * 1.1:  # 10% tolerance
            regressions += 1
            print(f"    WARNING: {lang} regressed: library={lib_ppl:.2f} vs "
                  f"individual={ind_ppl:.2f}")
    if regressions == 0:
        print(f"    No regressions vs individual adapters: PASS")

    # Check 3: Beat 1.5B (if available)
    if baseline_ppls:
        lib_mean = np.mean([best_lib.get(l, float('inf')) for l in languages])
        big_mean = np.mean([baseline_ppls.get(l, float('inf')) for l in languages])
        print(f"    0.5B+experts mean: {lib_mean:.2f}, 1.5B mean: {big_mean:.2f}")
        print(f"    Beat 1.5B: {'PASS' if lib_mean <= big_mean else 'FAIL'}")

    # Check 4: N-way routing accuracy
    print(f"    {len(languages)}-way routing accuracy: {post_acc:.1%} "
          f"({'PASS' if post_acc > 0.80 else 'FAIL'}, target >80%)")

    # Check 5: Calibrated vs individual adapters
    cal_ppls_verdict = library_ppls.get("calibrated", {})
    if cal_ppls_verdict:
        cal_regressions = 0
        for lang in languages:
            c_ppl = cal_ppls_verdict.get(lang, float('inf'))
            ind_ppl = individual_ppls[lang].get(lang, float('inf'))
            if c_ppl > ind_ppl * 1.1:
                cal_regressions += 1
                print(f"    WARNING: calibrated {lang} regressed: "
                      f"{c_ppl:.2f} vs individual {ind_ppl:.2f}")
        if cal_regressions == 0:
            print(f"    Calibrated vs individual adapters: PASS (all within 10%)")
        else:
            print(f"    Calibrated vs individual adapters: "
                  f"{cal_regressions}/{len(languages)} regressions")

    print(f"\n  Done in {elapsed:.0f}s")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
