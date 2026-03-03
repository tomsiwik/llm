"""Toy LLM Benchmark: Self-Routing LoRA Atoms for Continual Learning.

SmolLM-135M (frozen) + self-routing LoRA atoms trained sequentially on 3 domains:
  1. Wikipedia (general English prose)
  2. Python code
  3. Math word problems

Compares: static (no lifecycle) vs lifecycle (freeze) vs ghosts (parasitic atoms).

Usage:
    PYTHONUNBUFFERED=1 uv run --with mlx,mlx-lm,datasets python bench_llm_toy.py --quick
    PYTHONUNBUFFERED=1 uv run --with mlx,mlx-lm,datasets python bench_llm_toy.py --compare
    PYTHONUNBUFFERED=1 uv run --with mlx,mlx-lm,datasets python bench_llm_toy.py --ghosts=2
"""

import argparse
import time
import math
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from tribe.llm import (load_backbone, patch_with_atoms, freeze_base,
                        compute_perplexity, print_atom_summary)
from tribe.lora_atom import (collect_atom_layers, mask_frozen_gradients,
                              freeze_top_atoms, apply_graduated_protection,
                              spawn_ghosts_from_frozen,
                              ghost_lifecycle_step, ghost_regularization_loss,
                              lifecycle_summary)

# ── Config ──────────────────────────────────────────────────
MODEL_NAME = "HuggingFaceTB/SmolLM-135M"
N_ATOMS = 32
SEQ_LEN = 128
BATCH_SIZE = 8
SCALE = 1.0
DOMAINS = ["wiki", "code", "math"]


def parse_args():
    p = argparse.ArgumentParser(description="Toy LLM benchmark for self-routing LoRA atoms")
    p.add_argument("--quick", action="store_true", help="Quick mode (50 steps, less data)")
    p.add_argument("--topk", type=int, default=0, help="Hard top-k (0=soft routing)")
    p.add_argument("--temp", type=float, default=0.5, help="Softmax temperature")
    p.add_argument("--atoms", type=int, default=N_ATOMS, help="Atoms per layer")
    p.add_argument("--steps", type=int, default=None, help="Training steps per domain")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    p.add_argument("--lifecycle", action="store_true", help="Enable atom lifecycle")
    p.add_argument("--freeze-n", type=int, default=8, help="Atoms to freeze per layer")
    p.add_argument("--graduated", action="store_true",
                   help="Enable graduated gradient protection on active atoms")
    p.add_argument("--ghosts", type=int, default=0, help="Ghosts per frozen atom")
    p.add_argument("--ghost-decay", type=float, default=0.01,
                   help="L2 weight for ghost starvation")
    p.add_argument("--ghost-threshold", type=float, default=0.3,
                   help="Ghost possession threshold (||B|| norm)")
    p.add_argument("--compare", action="store_true",
                   help="Compare static vs lifecycle vs ghosts (1, 2, 5)")
    return p.parse_args()


# ── Data Loading ────────────────────────────────────────────

def load_domain_data(domain_name, tokenizer, n_train, n_eval):
    """Load and tokenize a domain dataset."""
    from datasets import load_dataset

    if domain_name == "wiki":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        texts = [ex["text"] for ex in ds if len(ex["text"].strip()) > 100]
    elif domain_name == "code":
        ds = load_dataset("codeparrot/codeparrot-clean", split="train", streaming=True)
        texts = []
        for ex in ds:
            if len(ex["content"].strip()) > 100:
                texts.append(ex["content"])
            if len(texts) >= n_train + n_eval:
                break
    elif domain_name == "math":
        ds = load_dataset("openai/gsm8k", "main", split="train")
        texts = [f"Question: {ex['question']}\nAnswer: {ex['answer']}" for ex in ds]
    else:
        raise ValueError(f"Unknown domain: {domain_name}")

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

    if len(train_seqs) > n_train:
        train_seqs = train_seqs[:n_train]

    return train_seqs, eval_texts


# ── Training ────────────────────────────────────────────────

def _make_loss_fn(ghost_decay_weight=0.0):
    """Create loss function, optionally with ghost L2 regularization."""
    def _ntp_loss(model, tokens):
        logits = model(tokens[:, :-1])
        targets = tokens[:, 1:]
        loss = nn.losses.cross_entropy(logits, targets, reduction='mean')
        if ghost_decay_weight > 0:
            loss = loss + ghost_regularization_loss(model, weight=ghost_decay_weight)
        return loss
    return _ntp_loss


def train_on_domain(model, train_seqs, steps, lr, lifecycle=False,
                    ghost_config=None):
    """Train atom parameters via next-token prediction.

    Args:
        lifecycle: if True, mask gradients for frozen atoms.
        ghost_config: dict with 'decay', 'threshold', 'check_interval' for ghost lifecycle.
    """
    ghost_decay = ghost_config['decay'] if ghost_config else 0.0
    ghost_threshold = ghost_config['threshold'] if ghost_config else 0.3
    ghost_interval = ghost_config.get('check_interval', 20) if ghost_config else 0

    optimizer = optim.Adam(learning_rate=lr)
    loss_fn = _make_loss_fn(ghost_decay_weight=ghost_decay)
    loss_and_grad = nn.value_and_grad(model, loss_fn)
    losses = []
    n_seqs = len(train_seqs)

    total_possessions = 0
    total_deaths = 0

    for step in range(steps):
        idx = np.random.randint(0, n_seqs, size=min(BATCH_SIZE, n_seqs))
        batch = mx.stack([train_seqs[i] for i in idx])

        loss, grads = loss_and_grad(model, batch)

        if lifecycle:
            grads = mask_frozen_gradients(model, grads)

        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        loss_val = loss.item()
        losses.append(loss_val)

        # Ghost lifecycle check (periodically)
        if ghost_config and ghost_interval > 0 and (step + 1) % ghost_interval == 0:
            events = ghost_lifecycle_step(
                model,
                possession_threshold=ghost_threshold,
                starvation_threshold=1e-4,
                respawn=True,
            )
            total_possessions += events['possessions']
            total_deaths += events['deaths']

        if (step + 1) % max(steps // 5, 1) == 0 or step == 0:
            ppl = math.exp(loss_val) if loss_val < 20 else float('inf')
            ghost_str = ""
            if ghost_config:
                n_alive = sum(l.n_ghosts_alive for _, l in collect_atom_layers(model))
                ghost_str = f", ghosts={n_alive}"
            print(f"      step {step+1:4d}/{steps}: loss={loss_val:.3f}, "
                  f"ppl={ppl:.1f}{ghost_str}")

    if ghost_config and (total_possessions > 0 or total_deaths > 0):
        print(f"    Ghost events: {total_possessions} possessions, "
              f"{total_deaths} deaths")

    return losses


# ── Single Run ──────────────────────────────────────────────

def run_single_config(config, domain_data, baseline_ppl):
    """Run training on all domains for a single configuration."""
    name = config['name']
    lifecycle = config.get('lifecycle', False)
    graduated = config.get('graduated', False)
    n_freeze = config.get('n_freeze', 8)
    n_ghosts = config.get('ghosts', 0)
    ghost_decay = config.get('ghost_decay', 0.01)
    ghost_threshold = config.get('ghost_threshold', 0.3)

    # Compute max_ghosts needed: n_freeze_per_domain * n_domains * n_ghosts_per_frozen
    max_ghosts = n_freeze * len(DOMAINS) * max(n_ghosts, 1) if n_ghosts > 0 else 0

    print(f"\n{'=' * 72}")
    print(f"  CONFIG: {name}")
    print(f"  Atoms: {config['atoms']}/layer, temp={config['temp']}, "
          f"top_k={config['topk']}, lr={config['lr']}, steps={config['steps']}")
    if lifecycle:
        print(f"  Lifecycle: freeze {n_freeze}/layer after each domain")
    if graduated:
        print(f"  Graduated: tiered gradient protection on active atoms")
    if n_ghosts > 0:
        print(f"  Ghosts: {n_ghosts}/frozen, decay={ghost_decay}, "
              f"threshold={ghost_threshold}, max_slots={max_ghosts}")
    print(f"{'=' * 72}")

    # Load fresh model
    model, tokenizer = load_backbone(MODEL_NAME)
    patch_with_atoms(model, n_atoms=config['atoms'], top_k=config['topk'],
                     temperature=config['temp'], scale=SCALE,
                     max_ghosts=max_ghosts)
    freeze_base(model)
    print_atom_summary(model)

    ghost_config = None
    if n_ghosts > 0:
        ghost_config = {
            'decay': ghost_decay,
            'threshold': ghost_threshold,
            'check_interval': 20,
        }

    ppl_matrix = {}

    for task_idx, domain in enumerate(DOMAINS):
        print(f"\n  ── Task {task_idx}: Train on '{domain}' ──")
        train_seqs, _ = domain_data[domain]

        if lifecycle or n_ghosts > 0:
            lifecycle_summary(model)

        train_on_domain(model, train_seqs, steps=config['steps'],
                        lr=config['lr'], lifecycle=lifecycle or n_ghosts > 0,
                        ghost_config=ghost_config)

        # Lifecycle: freeze top atoms, apply graduated protection, spawn ghosts
        if (lifecycle or n_ghosts > 0) and task_idx < len(DOMAINS) - 1:
            frozen_map = freeze_top_atoms(model, n_freeze, domain_label=domain)
            if graduated:
                apply_graduated_protection(model, domain_label=domain)
            if n_ghosts > 0:
                spawn_ghosts_from_frozen(model, frozen_map, n_per_frozen=n_ghosts)
            lifecycle_summary(model)

        # Evaluate on ALL domains
        print(f"    Evaluating...")
        ppl_row = {}
        for eval_domain in DOMAINS:
            _, eval_texts = domain_data[eval_domain]
            ppl = compute_perplexity(model, tokenizer, eval_texts)
            ppl_row[eval_domain] = ppl
            delta = ppl - baseline_ppl[eval_domain]
            marker = "+" if delta < -0.5 else ("-" if delta > 0.5 else "=")
            print(f"      {eval_domain:8s}: ppl={ppl:7.1f} ({marker}{abs(delta):.1f})")

        ppl_matrix[task_idx] = ppl_row

    # Compute forgetting
    forgetting = {}
    for task_idx in range(1, len(DOMAINS)):
        for prev_idx in range(task_idx):
            prev_domain = DOMAINS[prev_idx]
            ppl_before = ppl_matrix[prev_idx][prev_domain]
            ppl_after = ppl_matrix[task_idx][prev_domain]
            fgt = ppl_after - ppl_before
            pct = (fgt / ppl_before) * 100
            key = f"T{task_idx}→{prev_domain}"
            forgetting[key] = {'absolute': fgt, 'percent': pct,
                               'before': ppl_before, 'after': ppl_after}

    return {
        'name': name,
        'ppl_matrix': ppl_matrix,
        'forgetting': forgetting,
        'baseline_ppl': baseline_ppl,
    }


# ── Results Printing ────────────────────────────────────────

def print_results(results_list):
    """Print comparison table across all configs."""
    print("\n" + "=" * 72)
    print("  COMPARISON SUMMARY")
    print("=" * 72)

    # Perplexity matrix per config
    for res in results_list:
        name = res['name']
        ppl = res['ppl_matrix']
        bl = res['baseline_ppl']

        print(f"\n  [{name}] Perplexity Matrix:")
        header = f"  {'Task':>8s}"
        for d in DOMAINS:
            header += f"  {d:>8s}"
        print(header)
        print("  " + "-" * (10 + 10 * len(DOMAINS)))

        print(f"  {'baseline':>8s}", end="")
        for d in DOMAINS:
            print(f"  {bl[d]:8.1f}", end="")
        print()

        for task_idx in range(len(DOMAINS)):
            label = f"T{task_idx}({DOMAINS[task_idx][:4]})"
            print(f"  {label:>8s}", end="")
            for d in DOMAINS:
                print(f"  {ppl[task_idx][d]:8.1f}", end="")
            print()

    # Forgetting comparison
    print(f"\n  Forgetting Comparison:")
    header = f"  {'':>20s}"
    for res in results_list:
        header += f"  {res['name']:>14s}"
    print(header)
    print("  " + "-" * (22 + 16 * len(results_list)))

    all_keys = list(results_list[0]['forgetting'].keys())
    for key in all_keys:
        print(f"  {key:>20s}", end="")
        for res in results_list:
            fgt = res['forgetting'][key]
            print(f"  {fgt['percent']:+13.1f}%", end="")
        print()

    # Mean forgetting
    print(f"  {'MEAN':>20s}", end="")
    for res in results_list:
        fgts = [v['percent'] for v in res['forgetting'].values()]
        mean_fgt = np.mean(fgts)
        print(f"  {mean_fgt:+13.1f}%", end="")
    print()

    # Final perplexity
    print(f"\n  Final Perplexity (after all 3 domains):")
    header = f"  {'Domain':>8s}  {'baseline':>8s}"
    for res in results_list:
        header += f"  {res['name']:>14s}"
    print(header)
    print("  " + "-" * (20 + 16 * len(results_list)))

    for d in DOMAINS:
        bl = results_list[0]['baseline_ppl'][d]
        print(f"  {d:>8s}  {bl:8.1f}", end="")
        for res in results_list:
            final_ppl = res['ppl_matrix'][len(DOMAINS) - 1][d]
            print(f"  {final_ppl:14.1f}", end="")
        print()


# ── Main ────────────────────────────────────────────────────

def run_benchmark():
    args = parse_args()
    quick = args.quick
    train_steps = args.steps or (50 if quick else 200)
    n_train = 200 if quick else 500
    n_eval = 20 if quick else 50

    t0 = time.time()

    # Load data (shared across configs)
    print("  Loading backbone for tokenizer...")
    _, tokenizer = load_backbone(MODEL_NAME)

    print("  Loading domain data...")
    domain_data = {}
    for d in DOMAINS:
        print(f"    {d}...", end=" ", flush=True)
        train_seqs, eval_texts = load_domain_data(d, tokenizer, n_train, n_eval)
        domain_data[d] = (train_seqs, eval_texts)
        print(f"{len(train_seqs)} train, {len(eval_texts)} eval")

    # Baseline perplexity
    print("\n  Baseline perplexity (base model, no atoms)...")
    model_base, _ = load_backbone(MODEL_NAME)
    baseline_ppl = {}
    for d in DOMAINS:
        _, eval_texts = domain_data[d]
        ppl = compute_perplexity(model_base, tokenizer, eval_texts)
        baseline_ppl[d] = ppl
        print(f"    {d:8s}: ppl={ppl:.1f}")
    del model_base

    # Define configs
    base = {"atoms": args.atoms, "steps": train_steps, "lr": args.lr,
            "topk": args.topk, "temp": args.temp}

    if args.compare:
        configs = [
            {**base, "name": "static", "lifecycle": False, "ghosts": 0},
            {**base, "name": "lifecycle", "lifecycle": True, "n_freeze": 8, "ghosts": 0},
            {**base, "name": "graduated", "lifecycle": True, "graduated": True,
             "n_freeze": 8, "ghosts": 0},
            {**base, "name": "ghost_2", "lifecycle": True, "n_freeze": 8,
             "ghosts": 2, "ghost_decay": 0.01, "ghost_threshold": 0.3},
            {**base, "name": "grad+ghost2", "lifecycle": True, "graduated": True,
             "n_freeze": 8, "ghosts": 2, "ghost_decay": 0.01, "ghost_threshold": 0.3},
        ]
    else:
        lc = args.lifecycle or args.ghosts > 0 or args.graduated
        name = "static"
        if args.lifecycle:
            name = "lifecycle"
        if args.graduated:
            name = "graduated"
        if args.ghosts > 0:
            name = f"ghost_{args.ghosts}"
        if args.graduated and args.ghosts > 0:
            name = f"grad+ghost{args.ghosts}"
        configs = [
            {**base, "name": name, "lifecycle": lc,
             "graduated": args.graduated,
             "n_freeze": args.freeze_n, "ghosts": args.ghosts,
             "ghost_decay": args.ghost_decay, "ghost_threshold": args.ghost_threshold},
        ]

    # Run each config
    all_results = []
    for config in configs:
        result = run_single_config(config, domain_data, baseline_ppl)
        all_results.append(result)

    # Print comparison
    print_results(all_results)

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s")
    print("=" * 72)


if __name__ == "__main__":
    run_benchmark()
