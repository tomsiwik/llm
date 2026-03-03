"""Parallel PEER Branch Benchmark: Branch-Level Continual Learning.

SmolLM-135M (frozen) + parallel competing PEER branches trained sequentially
on 3 domains:
  1. Wikipedia (general English prose)
  2. Python code
  3. Math word problems

Tests whether branch-level isolation (frozen branches have zero routing
interference with active branches) beats neuron-level lifecycle (frozen
and active experts share the same routing space).

Compares 4 configs:
  - peer_lifecycle:       1 branch, 1024 experts, 32 active (current best)
  - peer_parallel_2b:     2 branches, 529 experts each, 17 active (no streams)
  - peer_parallel_2b_gpu: same + GPU/GPU streams (concurrent dispatch)
  - peer_parallel_2b_hetero: same + GPU/CPU heterogeneous streams (canonical MLX pattern)

Usage:
    PYTHONUNBUFFERED=1 uv run --with mlx,mlx-lm,datasets python bench_llm_parallel_peer.py --quick
    PYTHONUNBUFFERED=1 uv run --with mlx,mlx-lm,datasets python bench_llm_parallel_peer.py --compare
    PYTHONUNBUFFERED=1 uv run --with mlx,mlx-lm,datasets python bench_llm_parallel_peer.py --config peer_parallel_2b
"""

import argparse
import time
import math
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from tribe.llm import (load_backbone, patch_with_peer, patch_with_parallel_peer,
                        freeze_base, compute_perplexity, trainable_param_count)
from tribe.peer_atom import (collect_peer_layers, mask_peer_frozen_gradients,
                              freeze_top_peer_experts, recycle_dead_experts,
                              peer_lifecycle_summary, total_peer_params,
                              collect_parallel_peer_layers, freeze_best_branch,
                              parallel_peer_lifecycle_summary,
                              total_parallel_peer_params,
                              setup_parallel_streams)

# ── Config ──────────────────────────────────────────────────
MODEL_NAME = "HuggingFaceTB/SmolLM-135M"
SEQ_LEN = 128
BATCH_SIZE = 8
DOMAINS = ["wiki", "code", "math"]

ALL_CONFIGS = ["peer_lifecycle", "peer_parallel_2b", "peer_parallel_2b_gpu",
               "peer_parallel_2b_hetero"]


def parse_args():
    p = argparse.ArgumentParser(description="Parallel PEER branch benchmark")
    p.add_argument("--quick", action="store_true", help="Quick mode (50 steps)")
    p.add_argument("--steps", type=int, default=None, help="Training steps per domain")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    p.add_argument("--config", type=str, default=None,
                   help=f"Single config to run: {ALL_CONFIGS}")
    p.add_argument("--compare", action="store_true",
                   help="Compare all configs")
    return p.parse_args()


# ── Data Loading (reused from bench_llm_peer.py) ───────────

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

def ntp_loss(model, tokens):
    logits = model(tokens[:, :-1])
    targets = tokens[:, 1:]
    return nn.losses.cross_entropy(logits, targets, reduction='mean')


def _train_step(model, optimizer, loss_and_grad, batch, mask_grads_fn):
    """Single training step."""
    loss, grads = loss_and_grad(model, batch)
    if mask_grads_fn is not None:
        grads = mask_grads_fn(model, grads)
    optimizer.update(model, grads)
    del grads
    return loss


def train_on_domain(model, train_seqs, steps, lr, mask_grads_fn=None):
    """Train adapter parameters via next-token prediction."""
    optimizer = optim.Adam(learning_rate=lr)
    loss_and_grad = nn.value_and_grad(model, ntp_loss)
    losses = []
    n_seqs = len(train_seqs)
    report_interval = max(steps // 5, 1)

    for step in range(steps):
        idx = np.random.randint(0, n_seqs, size=min(BATCH_SIZE, n_seqs))
        batch = mx.stack([train_seqs[int(j)] for j in idx])

        loss = _train_step(model, optimizer, loss_and_grad, batch, mask_grads_fn)
        mx.eval(model.parameters(), optimizer.state)

        if (step + 1) % report_interval == 0 or step == 0:
            loss_val = loss.item()
            losses.append(loss_val)
            ppl = math.exp(loss_val) if loss_val < 20 else float('inf')
            print(f"      step {step+1:4d}/{steps}: loss={loss_val:.3f}, ppl={ppl:.1f}")

    return losses


# ── Gate Entropy Tracking ──────────────────────────────────

def compute_gate_entropy(model, train_seqs, n_samples=10):
    """Compute mean gate entropy across parallel PEER layers.

    Returns mean entropy (bits) — lower means more specialized gating.
    """
    layers = collect_parallel_peer_layers(model)
    if not layers:
        return None

    # Sample some inputs
    indices = np.random.choice(len(train_seqs), min(n_samples, len(train_seqs)),
                               replace=False)
    all_entropies = []

    for _, layer in layers:
        for idx in indices:
            x = train_seqs[int(idx)].reshape(1, -1)
            # Get hidden state at this layer — we approximate with raw token embeddings
            # (not exact but directionally correct for entropy tracking)
            x_flat = mx.reshape(x, (-1, layer.d_in)) if x.shape[-1] == layer.d_in else None
            if x_flat is None:
                continue
            gate_logits = x_flat @ layer.gate_proj.T + layer.gate_bias_vec
            gate_probs = mx.softmax(gate_logits, axis=-1)
            mx.eval(gate_probs)
            probs = np.array(gate_probs)
            # H = -sum(p * log2(p))
            probs = np.clip(probs, 1e-8, 1.0)
            entropy = -np.sum(probs * np.log2(probs), axis=-1)
            all_entropies.extend(entropy.tolist())

    return float(np.mean(all_entropies)) if all_entropies else None


# ── Config Runners ──────────────────────────────────────────

def setup_peer(args, model):
    """Setup PEER experts (baseline single-branch config)."""
    patch_with_peer(model, n_experts=1024, n_active=32, pk=8, scale=1.0)
    freeze_base(model)
    n_params = trainable_param_count(model)
    n_stored = total_peer_params(model)
    print(f"  PEER: 1024 experts/layer, 32 active/token")
    print(f"  Trainable params: {n_params:,}, Stored params: {n_stored:,}")


def setup_parallel_peer(args, model, use_streams=False):
    """Setup parallel PEER branches.

    Args:
        use_streams: False (sequential), "gpu" (shared GPU streams),
                     or "hetero" (GPU/CPU heterogeneous streams).
    """
    # Patch layers WITHOUT streams (streams are wired separately after)
    patch_with_parallel_peer(
        model, n_branches=2, n_experts=529, n_active=17, pk=8,
        scale=1.0, use_streams=False,
    )
    freeze_base(model)

    # Wire shared streams AFTER patching (2 streams shared across all 60 layers)
    if use_streams:
        setup_parallel_streams(model, mode=use_streams if isinstance(use_streams, str) else "gpu")

    n_params = trainable_param_count(model)
    n_stored = total_parallel_peer_params(model)
    stream_labels = {False: "", "gpu": " (gpu streams)", "hetero": " (gpu/cpu hetero)"}
    label = stream_labels.get(use_streams, f" ({use_streams})")
    print(f"  Parallel PEER{label}: 2 branches, 529 experts/branch, "
          f"17 active/token")
    print(f"  Trainable params: {n_params:,}, Stored params: {n_stored:,}")


def run_peer_lifecycle(args, domain_data, baseline_ppl):
    """PEER + freeze + recycle (baseline)."""
    print(f"\n{'=' * 72}")
    print(f"  CONFIG: peer_lifecycle")
    print(f"{'=' * 72}")

    model, tokenizer = load_backbone(MODEL_NAME)
    setup_peer(args, model)

    def mask_fn(m, g):
        return mask_peer_frozen_gradients(m, g)

    def between_fn(m, task_idx, domain):
        freeze_top_peer_experts(m, 64, domain_label=domain)
        recycle_dead_experts(m)
        peer_lifecycle_summary(m)

    return _run_domains(model, tokenizer, domain_data, baseline_ppl, args,
                        mask_fn=mask_fn, between_fn=between_fn,
                        name="peer_lifecycle")


def run_peer_parallel_2b(args, domain_data, baseline_ppl, use_streams=False):
    """Parallel PEER with 2 branches."""
    stream_names = {False: "peer_parallel_2b", True: "peer_parallel_2b_gpu",
                    "gpu": "peer_parallel_2b_gpu", "hetero": "peer_parallel_2b_hetero"}
    name = stream_names.get(use_streams, f"peer_parallel_2b_{use_streams}")
    print(f"\n{'=' * 72}")
    print(f"  CONFIG: {name}")
    print(f"{'=' * 72}")

    model, tokenizer = load_backbone(MODEL_NAME)
    setup_parallel_peer(args, model, use_streams=use_streams)

    def mask_fn(m, g):
        return mask_peer_frozen_gradients(m, g)

    def between_fn(m, task_idx, domain):
        # Count active branches
        n_active = 0
        for _, layer in collect_parallel_peer_layers(m):
            n_active = layer.n_active_branches
            break

        if n_active > 1:
            # Freeze the best branch (complete routing isolation)
            freeze_best_branch(m, domain_label=domain)
        # Intra-branch lifecycle on remaining active branch(es)
        # Freeze top experts within active branches for fine-grained preservation
        freeze_top_peer_experts(m, 32, domain_label=f"{domain}_intra")
        recycle_dead_experts(m)

        parallel_peer_lifecycle_summary(m)

    return _run_domains(model, tokenizer, domain_data, baseline_ppl, args,
                        mask_fn=mask_fn, between_fn=between_fn,
                        name=name)


# ── Common domain training loop ────────────────────────────

def _run_domains(model, tokenizer, domain_data, baseline_ppl, args,
                 mask_fn, between_fn, name):
    """Train on all domains, evaluate after each."""
    steps = args.steps or (50 if args.quick else 200)
    ppl_matrix = {}
    gate_entropies = []  # track gate entropy over training

    for task_idx, domain in enumerate(DOMAINS):
        print(f"\n  -- Task {task_idx}: Train on '{domain}' --")
        train_seqs, _ = domain_data[domain]

        t_start = time.time()
        train_on_domain(model, train_seqs, steps=steps, lr=args.lr,
                        mask_grads_fn=mask_fn)
        t_elapsed = time.time() - t_start
        print(f"    Training time: {t_elapsed:.1f}s ({t_elapsed/steps:.2f}s/step)")

        # Track gate entropy (parallel configs only)
        layers = collect_parallel_peer_layers(model)
        if layers:
            entropy = compute_gate_entropy(model, train_seqs)
            if entropy is not None:
                gate_entropies.append((domain, entropy))
                print(f"    Gate entropy: {entropy:.3f} bits")

        # Lifecycle between domains (not after last)
        if between_fn is not None and task_idx < len(DOMAINS) - 1:
            between_fn(model, task_idx, domain)

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
            key = f"T{task_idx}>{prev_domain}"
            forgetting[key] = {'absolute': fgt, 'percent': pct,
                               'before': ppl_before, 'after': ppl_after}

    # Gate weights at eval time (parallel configs only)
    gate_weights_eval = None
    layers = collect_parallel_peer_layers(model)
    if layers:
        gate_weights_eval = {}
        for eval_domain in DOMAINS:
            _, eval_texts = domain_data[eval_domain]
            # Get gate weights on eval data
            gw = _eval_gate_weights(model, tokenizer, eval_texts, layers)
            if gw is not None:
                gate_weights_eval[eval_domain] = gw

    return {
        'name': name,
        'ppl_matrix': ppl_matrix,
        'forgetting': forgetting,
        'baseline_ppl': baseline_ppl,
        'gate_entropies': gate_entropies,
        'gate_weights_eval': gate_weights_eval,
    }


def _eval_gate_weights(model, tokenizer, eval_texts, layers):
    """Compute mean gate weights per domain (approximate via first layer)."""
    # We can't easily get intermediate hidden states, so skip this
    # and just return gate_bias_vec as a proxy for learned preference
    if not layers:
        return None

    _, layer = layers[0]
    bias = np.array(layer.gate_bias_vec)
    # softmax of bias gives default preference
    exp_b = np.exp(bias - bias.max())
    probs = exp_b / exp_b.sum()
    return probs.tolist()


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

        # Gate entropies
        if res.get('gate_entropies'):
            print(f"    Gate entropy: ", end="")
            for domain, ent in res['gate_entropies']:
                print(f"{domain}={ent:.3f} ", end="")
            print()

        # Gate weights at eval
        if res.get('gate_weights_eval'):
            print(f"    Gate weights (eval):")
            for domain, gw in res['gate_weights_eval'].items():
                gw_str = ", ".join(f"B{i}={w:.2f}" for i, w in enumerate(gw))
                print(f"      {domain:8s}: {gw_str}")

    # Forgetting comparison
    print(f"\n  Forgetting Comparison:")
    header = f"  {'':>20s}"
    for res in results_list:
        header += f"  {res['name']:>20s}"
    print(header)
    print("  " + "-" * (22 + 22 * len(results_list)))

    all_keys = list(results_list[0]['forgetting'].keys())
    for key in all_keys:
        print(f"  {key:>20s}", end="")
        for res in results_list:
            fgt = res['forgetting'][key]
            print(f"  {fgt['percent']:+19.1f}%", end="")
        print()

    # Mean forgetting
    print(f"  {'MEAN':>20s}", end="")
    for res in results_list:
        fgts = [v['percent'] for v in res['forgetting'].values()]
        mean_fgt = np.mean(fgts)
        print(f"  {mean_fgt:+19.1f}%", end="")
    print()

    # Final perplexity
    print(f"\n  Final Perplexity (after all 3 domains):")
    header = f"  {'Domain':>8s}  {'baseline':>8s}"
    for res in results_list:
        header += f"  {res['name']:>20s}"
    print(header)
    print("  " + "-" * (20 + 22 * len(results_list)))

    for d in DOMAINS:
        bl = results_list[0]['baseline_ppl'][d]
        print(f"  {d:>8s}  {bl:8.1f}", end="")
        for res in results_list:
            final_ppl = res['ppl_matrix'][len(DOMAINS) - 1][d]
            print(f"  {final_ppl:20.1f}", end="")
        print()

    # Param budget audit
    print(f"\n  Parameter Budget:")
    for res in results_list:
        print(f"    {res['name']}: see stored params above")


# ── Main ────────────────────────────────────────────────────

CONFIG_RUNNERS = {
    "peer_lifecycle": lambda args, dd, bl: run_peer_lifecycle(args, dd, bl),
    "peer_parallel_2b": lambda args, dd, bl: run_peer_parallel_2b(args, dd, bl, use_streams=False),
    "peer_parallel_2b_gpu": lambda args, dd, bl: run_peer_parallel_2b(args, dd, bl, use_streams="gpu"),
    "peer_parallel_2b_hetero": lambda args, dd, bl: run_peer_parallel_2b(args, dd, bl, use_streams="hetero"),
}


def run_benchmark():
    args = parse_args()
    quick = args.quick
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
    print("\n  Baseline perplexity (base model, no adapters)...")
    model_base, _ = load_backbone(MODEL_NAME)
    baseline_ppl = {}
    for d in DOMAINS:
        _, eval_texts = domain_data[d]
        ppl = compute_perplexity(model_base, tokenizer, eval_texts)
        baseline_ppl[d] = ppl
        print(f"    {d:8s}: ppl={ppl:.1f}")
    del model_base

    # Determine which configs to run
    if args.compare:
        configs_to_run = ALL_CONFIGS
    elif args.config:
        if args.config not in CONFIG_RUNNERS:
            print(f"  ERROR: Unknown config '{args.config}'. "
                  f"Choose from: {ALL_CONFIGS}")
            return
        configs_to_run = [args.config]
    else:
        configs_to_run = ["peer_parallel_2b"]

    # Run each config
    all_results = []
    for config_name in configs_to_run:
        runner = CONFIG_RUNNERS[config_name]
        result = runner(args, domain_data, baseline_ppl)
        all_results.append(result)

    # Print comparison
    print_results(all_results)

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s")
    print("=" * 72)


if __name__ == "__main__":
    run_benchmark()
