"""PEER + FoX + Lifecycle Benchmark: Neuron-Level Continual Learning.

SmolLM-135M (frozen) + adapters trained sequentially on 3 domains:
  1. Wikipedia (general English prose)
  2. Python code
  3. Math word problems

Configs:
  Original (self-routing LoRA atoms + PEER):
  - lora_static:       LoRA atoms, no lifecycle
  - lora_lifecycle:    LoRA atoms + freeze
  - peer_static:       PEER routing, no lifecycle
  - peer_lifecycle:    PEER + freeze + recycle
  - peer_fox:          PEER + freeze + clone_with_gate + emancipate
  - peer_version_tree: PEER + snapshot version tree

  Standard CL baselines (true LoRA A@B):
  - lora_seqft:   Sequential fine-tuning (no CL, catastrophic forgetting baseline)
  - lora_ewc:     Elastic Weight Consolidation
  - lora_replay:  Experience Replay (50/50 current + buffer)
  - lora_olora:   Orthogonal LoRA (gradient projection)

Usage:
    PYTHONUNBUFFERED=1 uv run --with mlx,mlx-lm,datasets python bench_llm_peer.py --quick
    PYTHONUNBUFFERED=1 uv run --with mlx,mlx-lm,datasets python bench_llm_peer.py --compare
    PYTHONUNBUFFERED=1 uv run --with mlx,mlx-lm,datasets python bench_llm_peer.py --config lora_seqft
    PYTHONUNBUFFERED=1 uv run --with mlx,mlx-lm,datasets python bench_llm_peer.py --configs lora_seqft,lora_ewc,peer_lifecycle
"""

import argparse
import time
import math
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from tribe.llm import (load_backbone, patch_with_atoms, patch_with_peer,
                        patch_with_standard_lora, freeze_base,
                        compute_perplexity, trainable_param_count)
from tribe.lora_atom import (collect_atom_layers, mask_frozen_gradients,
                              freeze_top_atoms, lifecycle_summary)
from tribe.lora_standard import (collect_standard_lora_layers,
                                  total_standard_lora_params)
from tribe.peer_atom import (collect_peer_layers, mask_peer_frozen_gradients,
                              freeze_top_peer_experts, clone_frozen_to_children,
                              emancipate_mature_children, recycle_dead_experts,
                              peer_lifecycle_summary, total_peer_params,
                              snapshot_top_experts, version_tree_summary)

# ── Config ──────────────────────────────────────────────────
MODEL_NAME = "HuggingFaceTB/SmolLM-135M"
SEQ_LEN = 128
BATCH_SIZE = 8
DOMAINS = ["wiki", "code", "math"]

ALL_CONFIGS = [
    # Original
    "lora_static", "lora_lifecycle",
    "peer_static", "peer_lifecycle", "peer_fox",
    "peer_version_tree",
    # Standard CL baselines
    "lora_seqft", "lora_ewc", "lora_replay", "lora_olora",
]


def parse_args():
    p = argparse.ArgumentParser(description="PEER + FoX + Lifecycle benchmark")
    p.add_argument("--quick", action="store_true", help="Quick mode (50 steps)")
    p.add_argument("--steps", type=int, default=None, help="Training steps per domain")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    p.add_argument("--config", type=str, default=None,
                   help=f"Single config to run: {ALL_CONFIGS}")
    p.add_argument("--configs", type=str, default=None,
                   help="Comma-separated list of configs to compare")
    p.add_argument("--compare", action="store_true",
                   help="Compare all configs")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    # PEER params
    p.add_argument("--n-experts", type=int, default=1024,
                   help="PEER experts per layer (must be perfect square)")
    p.add_argument("--n-active", type=int, default=32,
                   help="Experts selected per token")
    p.add_argument("--pk", type=int, default=8,
                   help="Top-k per sub-key half")
    p.add_argument("--n-freeze", type=int, default=64,
                   help="Experts to freeze per PEER layer")
    # LoRA params (atoms)
    p.add_argument("--atoms", type=int, default=32,
                   help="LoRA atoms per layer")
    p.add_argument("--lora-freeze", type=int, default=8,
                   help="Atoms to freeze per LoRA layer")
    # Standard LoRA params
    p.add_argument("--lora-rank", type=int, default=16,
                   help="Standard LoRA rank")
    # EWC params
    p.add_argument("--ewc-lambda", type=float, default=100.0,
                   help="EWC regularization strength")
    # Replay params
    p.add_argument("--replay-per-domain", type=int, default=100,
                   help="Sequences to store per domain for replay")
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

def ntp_loss(model, tokens):
    logits = model(tokens[:, :-1])
    targets = tokens[:, 1:]
    return nn.losses.cross_entropy(logits, targets, reduction='mean')


def _train_step(model, optimizer, loss_and_grad, batch, mask_grads_fn):
    """Single training step — grads released before eval for memory reuse."""
    loss, grads = loss_and_grad(model, batch)
    if mask_grads_fn is not None:
        grads = mask_grads_fn(model, grads)
    optimizer.update(model, grads)
    del grads  # release before eval per fast-mlx guide
    return loss


def train_on_domain(model, train_seqs, steps, lr, mask_grads_fn=None,
                    loss_fn=None):
    """Train adapter parameters via next-token prediction.

    Args:
        mask_grads_fn: callable(model, grads) → grads for gradient masking.
        loss_fn: custom loss function(model, tokens) → scalar. Default: ntp_loss.
    """
    if loss_fn is None:
        loss_fn = ntp_loss
    optimizer = optim.Adam(learning_rate=lr)
    loss_and_grad = nn.value_and_grad(model, loss_fn)
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


# ── Standard LoRA Setup ────────────────────────────────────

def setup_standard_lora(args, model):
    """Patch model with standard LoRA and freeze base."""
    patch_with_standard_lora(model, rank=args.lora_rank, scale=float(args.lora_rank))
    freeze_base(model)
    n_params = trainable_param_count(model)
    n_lora = total_standard_lora_params(model)
    print(f"  Standard LoRA: rank={args.lora_rank}, targets=q_proj+v_proj")
    print(f"  Trainable params: {n_params:,}, LoRA params: {n_lora:,}")
    return n_lora


# ── EWC ─────────────────────────────────────────────────────

class EWCState:
    """Fisher diagonal + param snapshots for Elastic Weight Consolidation."""

    def __init__(self):
        self.fisher = {}        # flat_key → mx.array (accumulated)
        self.star_params = {}   # flat_key → mx.array (snapshot)

    def compute_fisher(self, model, train_seqs, n_samples=50):
        """Estimate Fisher information diagonal after training on a domain.

        Accumulates squared gradients from n_samples mini-batches.
        Only tracks lora_A and lora_B parameters.
        """
        loss_and_grad = nn.value_and_grad(model, ntp_loss)
        n_seqs = len(train_seqs)

        fisher_acc = {}
        for _ in range(n_samples):
            idx = np.random.randint(0, n_seqs, size=min(BATCH_SIZE, n_seqs))
            batch = mx.stack([train_seqs[int(j)] for j in idx])
            _, grads = loss_and_grad(model, batch)
            mx.eval(grads)

            # Walk grad tree for lora_A/lora_B
            for key, val in _flatten_lora_grads(grads):
                sq = val * val
                if key in fisher_acc:
                    fisher_acc[key] = fisher_acc[key] + sq
                else:
                    fisher_acc[key] = sq

        # Average and accumulate into total Fisher
        for key in fisher_acc:
            fisher_acc[key] = fisher_acc[key] / n_samples
            if key in self.fisher:
                self.fisher[key] = self.fisher[key] + fisher_acc[key]
            else:
                self.fisher[key] = fisher_acc[key]

        # Snapshot current params
        for key, val in _flatten_lora_params(model):
            self.star_params[key] = mx.array(np.array(val))

        mx.eval(*self.fisher.values(), *self.star_params.values())
        n_keys = len(self.fisher)
        print(f"    EWC: computed Fisher over {n_samples} samples, {n_keys} param groups")

    def ewc_penalty(self, model):
        """Compute EWC penalty: sum_i F_i * (theta_i - theta*_i)^2."""
        penalty = mx.array(0.0)
        for key, val in _flatten_lora_params(model):
            if key in self.fisher:
                diff = val - self.star_params[key]
                penalty = penalty + mx.sum(self.fisher[key] * diff * diff)
        return penalty


def _flatten_lora_grads(grads, prefix=""):
    """Yield (flat_key, array) for lora_A/lora_B in nested grad dict."""
    if isinstance(grads, dict):
        for k, v in grads.items():
            yield from _flatten_lora_grads(v, f"{prefix}.{k}" if prefix else k)
    elif isinstance(grads, list):
        for i, v in enumerate(grads):
            yield from _flatten_lora_grads(v, f"{prefix}.{i}")
    elif isinstance(grads, mx.array):
        if prefix.endswith("lora_A") or prefix.endswith("lora_B"):
            yield prefix, grads


def _flatten_lora_params(model):
    """Yield (flat_key, array) for lora_A/lora_B in model parameters."""
    def _walk(params, prefix=""):
        if isinstance(params, dict):
            for k, v in params.items():
                yield from _walk(v, f"{prefix}.{k}" if prefix else k)
        elif isinstance(params, list):
            for i, v in enumerate(params):
                yield from _walk(v, f"{prefix}.{i}")
        elif isinstance(params, mx.array):
            if prefix.endswith("lora_A") or prefix.endswith("lora_B"):
                yield prefix, params
    yield from _walk(model.parameters())


# ── Replay Buffer ───────────────────────────────────────────

class ReplayBuffer:
    """Experience replay buffer using reservoir sampling."""

    def __init__(self):
        self.buffer = []

    def store_domain(self, train_seqs, n=100):
        """Store n sequences from the domain via reservoir sampling."""
        rng = np.random.RandomState(len(self.buffer))
        indices = rng.choice(len(train_seqs), size=min(n, len(train_seqs)),
                             replace=False)
        for i in indices:
            self.buffer.append(train_seqs[int(i)])
        print(f"    REPLAY: stored {len(indices)} seqs, buffer size={len(self.buffer)}")

    def sample_replay(self, batch_size):
        """Sample a batch from the buffer."""
        if not self.buffer:
            return None
        indices = np.random.randint(0, len(self.buffer), size=batch_size)
        return mx.stack([self.buffer[int(i)] for i in indices])


# ── O-LoRA ──────────────────────────────────────────────────

class OLoRAState:
    """Orthogonal basis accumulator for gradient projection (O-LoRA).

    After each domain, accumulate the learned LoRA-A directions into an
    orthonormal basis. During training on the next domain, project out
    these directions from gradients so new learning is orthogonal.
    """

    def __init__(self):
        self.bases = {}  # layer_path → (d_in, accum_rank) orthonormal basis

    def accumulate_basis(self, model):
        """Extract lora_A from all layers, QR-orthogonalize, accumulate."""
        n_new = 0
        for key, val in _flatten_lora_params(model):
            if not key.endswith("lora_A"):
                continue
            # lora_A is (d_in, rank) — columns are the learned directions
            A = np.array(val)

            if key in self.bases:
                # Concatenate with existing basis, re-orthogonalize
                existing = self.bases[key]
                combined = np.concatenate([existing, A], axis=1)
            else:
                combined = A

            # QR decomposition on CPU (MLX QR only on CPU)
            Q, _ = np.linalg.qr(combined)
            # Keep only as many columns as have meaningful content
            # (rank of combined may be less than cols)
            self.bases[key] = Q
            n_new += 1

        print(f"    O-LoRA: accumulated basis for {n_new} layers, "
              f"max rank={max(v.shape[1] for v in self.bases.values()) if self.bases else 0}")

    def project_gradients(self, model, grads):
        """Project out accumulated basis directions from lora_A gradients.

        grad_A -= Q @ Q.T @ grad_A  (remove components in old subspace)
        """
        for key, val in _flatten_lora_grads(grads):
            if not key.endswith("lora_A"):
                continue
            if key not in self.bases:
                continue

            Q = mx.array(self.bases[key])  # (d_in, accum_rank)
            # Project: grad -= Q @ (Q.T @ grad)
            proj = Q @ (Q.T @ val)
            new_val = val - proj

            # Write back into grads tree
            parts = key.split(".")
            node = grads
            for part in parts[:-1]:
                if part.isdigit():
                    node = node[int(part)]
                else:
                    node = node[part]
            node[parts[-1]] = new_val

        return grads


# ── Config Runners: Original ────────────────────────────────

def setup_lora(args, model):
    """Setup LoRA atoms on model."""
    patch_with_atoms(model, n_atoms=args.atoms, temperature=0.5, scale=1.0)
    freeze_base(model)
    n_params = trainable_param_count(model)
    print(f"  LoRA atoms: {args.atoms}/layer, trainable params: {n_params:,}")
    return n_params


def setup_peer(args, model):
    """Setup PEER experts on model."""
    patch_with_peer(model, n_experts=args.n_experts, n_active=args.n_active,
                    pk=args.pk, scale=1.0)
    freeze_base(model)
    n_params = trainable_param_count(model)
    n_stored = total_peer_params(model)
    print(f"  PEER: {args.n_experts} experts/layer, {args.n_active} active/token")
    print(f"  Trainable params: {n_params:,}, Stored params: {n_stored:,}")
    return n_stored


def run_lora_static(args, domain_data, baseline_ppl):
    """LoRA atoms, no lifecycle."""
    print(f"\n{'=' * 72}")
    print(f"  CONFIG: lora_static")
    print(f"{'=' * 72}")

    model, tokenizer = load_backbone(MODEL_NAME)
    n_params = setup_lora(args, model)

    result = _run_domains(model, tokenizer, domain_data, baseline_ppl, args,
                          mask_fn=None, between_fn=None, name="lora_static")
    result['n_params'] = n_params
    return result


def run_lora_lifecycle(args, domain_data, baseline_ppl):
    """LoRA atoms + freeze."""
    print(f"\n{'=' * 72}")
    print(f"  CONFIG: lora_lifecycle")
    print(f"{'=' * 72}")

    model, tokenizer = load_backbone(MODEL_NAME)
    n_params = setup_lora(args, model)

    def mask_fn(m, g):
        return mask_frozen_gradients(m, g)

    def between_fn(m, task_idx, domain):
        freeze_top_atoms(m, args.lora_freeze, domain_label=domain)
        lifecycle_summary(m)

    result = _run_domains(model, tokenizer, domain_data, baseline_ppl, args,
                          mask_fn=mask_fn, between_fn=between_fn,
                          name="lora_lifecycle")
    result['n_params'] = n_params
    return result


def run_peer_static(args, domain_data, baseline_ppl):
    """PEER routing, no lifecycle."""
    print(f"\n{'=' * 72}")
    print(f"  CONFIG: peer_static")
    print(f"{'=' * 72}")

    model, tokenizer = load_backbone(MODEL_NAME)
    n_params = setup_peer(args, model)

    result = _run_domains(model, tokenizer, domain_data, baseline_ppl, args,
                          mask_fn=None, between_fn=None, name="peer_static")
    result['n_params'] = n_params
    return result


def run_peer_lifecycle(args, domain_data, baseline_ppl):
    """PEER + freeze + recycle."""
    print(f"\n{'=' * 72}")
    print(f"  CONFIG: peer_lifecycle")
    print(f"{'=' * 72}")

    model, tokenizer = load_backbone(MODEL_NAME)
    n_params = setup_peer(args, model)

    def mask_fn(m, g):
        return mask_peer_frozen_gradients(m, g)

    def between_fn(m, task_idx, domain):
        freeze_top_peer_experts(m, args.n_freeze, domain_label=domain)
        recycle_dead_experts(m)
        peer_lifecycle_summary(m)

    result = _run_domains(model, tokenizer, domain_data, baseline_ppl, args,
                          mask_fn=mask_fn, between_fn=between_fn,
                          name="peer_lifecycle")
    result['n_params'] = n_params
    return result


def run_peer_fox(args, domain_data, baseline_ppl):
    """PEER + freeze + clone_with_gate + emancipate (full system)."""
    print(f"\n{'=' * 72}")
    print(f"  CONFIG: peer_fox")
    print(f"{'=' * 72}")

    model, tokenizer = load_backbone(MODEL_NAME)
    n_params = setup_peer(args, model)

    def mask_fn(m, g):
        return mask_peer_frozen_gradients(m, g)

    def between_fn(m, task_idx, domain):
        emancipate_mature_children(m, threshold=0.9)
        frozen_map = freeze_top_peer_experts(m, args.n_freeze, domain_label=domain)
        clone_frozen_to_children(m, frozen_map)
        recycle_dead_experts(m)
        peer_lifecycle_summary(m)

    result = _run_domains(model, tokenizer, domain_data, baseline_ppl, args,
                          mask_fn=mask_fn, between_fn=between_fn,
                          name="peer_fox")
    result['n_params'] = n_params
    return result


def run_peer_version_tree(args, domain_data, baseline_ppl):
    """PEER + snapshot version tree (in-place freeze + recycle)."""
    print(f"\n{'=' * 72}")
    print(f"  CONFIG: peer_version_tree")
    print(f"{'=' * 72}")

    model, tokenizer = load_backbone(MODEL_NAME)
    n_params = setup_peer(args, model)

    def mask_fn(m, g):
        return mask_peer_frozen_gradients(m, g)

    def between_fn(m, task_idx, domain):
        snapshot_top_experts(m, args.n_freeze, domain_label=domain, recycle=True)
        recycle_dead_experts(m)
        version_tree_summary(m)

    result = _run_domains(model, tokenizer, domain_data, baseline_ppl, args,
                          mask_fn=mask_fn, between_fn=between_fn,
                          name="peer_version_tree")
    result['n_params'] = n_params
    return result


# ── Config Runners: Standard CL Baselines ───────────────────

def run_lora_seqft(args, domain_data, baseline_ppl):
    """Standard LoRA, sequential fine-tuning, no CL (catastrophic forgetting baseline)."""
    print(f"\n{'=' * 72}")
    print(f"  CONFIG: lora_seqft")
    print(f"{'=' * 72}")

    model, tokenizer = load_backbone(MODEL_NAME)
    n_params = setup_standard_lora(args, model)

    result = _run_domains(model, tokenizer, domain_data, baseline_ppl, args,
                          mask_fn=None, between_fn=None, name="lora_seqft")
    result['n_params'] = n_params
    return result


def run_lora_ewc(args, domain_data, baseline_ppl):
    """Standard LoRA + Elastic Weight Consolidation."""
    print(f"\n{'=' * 72}")
    print(f"  CONFIG: lora_ewc (lambda={args.ewc_lambda})")
    print(f"{'=' * 72}")

    model, tokenizer = load_backbone(MODEL_NAME)
    n_params = setup_standard_lora(args, model)

    ewc = EWCState()
    steps = args.steps or (50 if args.quick else 200)

    ppl_matrix = {}
    for task_idx, domain in enumerate(DOMAINS):
        print(f"\n  -- Task {task_idx}: Train on '{domain}' --")
        train_seqs, _ = domain_data[domain]

        # Build loss function with EWC penalty
        lam = args.ewc_lambda
        if ewc.fisher:
            def ewc_loss(model, tokens, _ewc=ewc, _lam=lam):
                base = ntp_loss(model, tokens)
                penalty = _ewc.ewc_penalty(model)
                return base + (_lam / 2.0) * penalty
            loss_fn = ewc_loss
        else:
            loss_fn = ntp_loss

        train_on_domain(model, train_seqs, steps=steps, lr=args.lr,
                        loss_fn=loss_fn)

        # Compute Fisher after training on this domain (before moving on)
        if task_idx < len(DOMAINS) - 1:
            ewc.compute_fisher(model, train_seqs, n_samples=min(50, len(train_seqs)))

        # Evaluate
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

    result = _compute_forgetting(ppl_matrix, baseline_ppl, "lora_ewc", model)
    result['n_params'] = n_params
    result['total_steps'] = steps * len(DOMAINS)
    return result


def run_lora_replay(args, domain_data, baseline_ppl):
    """Standard LoRA + Experience Replay (50/50 current + buffer)."""
    print(f"\n{'=' * 72}")
    print(f"  CONFIG: lora_replay (buffer={args.replay_per_domain}/domain)")
    print(f"{'=' * 72}")

    model, tokenizer = load_backbone(MODEL_NAME)
    n_params = setup_standard_lora(args, model)

    replay = ReplayBuffer()
    steps = args.steps or (50 if args.quick else 200)

    ppl_matrix = {}
    for task_idx, domain in enumerate(DOMAINS):
        print(f"\n  -- Task {task_idx}: Train on '{domain}' --")
        train_seqs, _ = domain_data[domain]

        # Custom training with replay
        optimizer = optim.Adam(learning_rate=args.lr)
        loss_and_grad = nn.value_and_grad(model, ntp_loss)
        n_seqs = len(train_seqs)
        report_interval = max(steps // 5, 1)

        for step in range(steps):
            # Current domain batch
            half = max(BATCH_SIZE // 2, 1)
            idx = np.random.randint(0, n_seqs, size=min(half, n_seqs))
            current_batch = mx.stack([train_seqs[int(j)] for j in idx])

            # Replay batch (if available)
            replay_batch = replay.sample_replay(half)
            if replay_batch is not None:
                batch = mx.concatenate([current_batch, replay_batch], axis=0)
            else:
                batch = current_batch

            loss = _train_step(model, optimizer, loss_and_grad, batch, None)
            mx.eval(model.parameters(), optimizer.state)

            if (step + 1) % report_interval == 0 or step == 0:
                loss_val = loss.item()
                ppl = math.exp(loss_val) if loss_val < 20 else float('inf')
                print(f"      step {step+1:4d}/{steps}: loss={loss_val:.3f}, ppl={ppl:.1f}")

        # Store sequences for future replay
        if task_idx < len(DOMAINS) - 1:
            replay.store_domain(train_seqs, n=args.replay_per_domain)

        # Evaluate
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

    result = _compute_forgetting(ppl_matrix, baseline_ppl, "lora_replay", model)
    result['n_params'] = n_params
    result['total_steps'] = steps * len(DOMAINS)
    return result


def run_lora_olora(args, domain_data, baseline_ppl):
    """Standard LoRA + Orthogonal LoRA (gradient projection into null space)."""
    print(f"\n{'=' * 72}")
    print(f"  CONFIG: lora_olora")
    print(f"{'=' * 72}")

    model, tokenizer = load_backbone(MODEL_NAME)
    n_params = setup_standard_lora(args, model)

    olora = OLoRAState()
    steps = args.steps or (50 if args.quick else 200)

    ppl_matrix = {}
    for task_idx, domain in enumerate(DOMAINS):
        print(f"\n  -- Task {task_idx}: Train on '{domain}' --")
        train_seqs, _ = domain_data[domain]

        # Gradient projection mask
        if olora.bases:
            def mask_fn(m, g, _olora=olora):
                return _olora.project_gradients(m, g)
        else:
            mask_fn = None

        train_on_domain(model, train_seqs, steps=steps, lr=args.lr,
                        mask_grads_fn=mask_fn)

        # Accumulate basis after training on this domain
        if task_idx < len(DOMAINS) - 1:
            olora.accumulate_basis(model)

        # Evaluate
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

    result = _compute_forgetting(ppl_matrix, baseline_ppl, "lora_olora", model)
    result['n_params'] = n_params
    result['total_steps'] = steps * len(DOMAINS)
    return result


# ── Common domain training loop ────────────────────────────

def _run_domains(model, tokenizer, domain_data, baseline_ppl, args,
                 mask_fn, between_fn, name, loss_fn=None, train_fn=None):
    """Train on all domains, evaluate after each."""
    steps = args.steps or (50 if args.quick else 200)
    ppl_matrix = {}

    for task_idx, domain in enumerate(DOMAINS):
        print(f"\n  -- Task {task_idx}: Train on '{domain}' --")
        train_seqs, _ = domain_data[domain]

        if train_fn is not None:
            train_fn(model, train_seqs, steps, args.lr, mask_fn, task_idx, domain)
        else:
            train_on_domain(model, train_seqs, steps=steps, lr=args.lr,
                            mask_grads_fn=mask_fn, loss_fn=loss_fn)

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

    result = _compute_forgetting(ppl_matrix, baseline_ppl, name, model)
    result['total_steps'] = steps * len(DOMAINS)
    return result


def _compute_forgetting(ppl_matrix, baseline_ppl, name, model):
    """Compute forgetting metrics and gather model info."""
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

    # Gate histogram (PEER layers only)
    gate_hist = None
    peer_layers = collect_peer_layers(model)
    if peer_layers:
        all_gates = []
        for _, l in peer_layers:
            gv = l.gate_values()
            if l._has_parent.any():
                all_gates.extend(gv[l._has_parent])
        if all_gates:
            gate_hist = np.array(all_gates)

    # Routing overlap (PEER layers only)
    routing_info = None
    if peer_layers:
        total_frozen = sum(l.n_frozen for _, l in peer_layers)
        total_gated = sum(l.n_gated for _, l in peer_layers)
        total_experts = sum(l.n_experts for _, l in peer_layers)
        routing_info = {
            'frozen': total_frozen, 'gated': total_gated,
            'total': total_experts,
        }

    return {
        'name': name,
        'ppl_matrix': ppl_matrix,
        'forgetting': forgetting,
        'baseline_ppl': baseline_ppl,
        'gate_hist': gate_hist,
        'routing_info': routing_info,
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
        n_params = res.get('n_params', 0)

        param_str = f" ({n_params:,} params)" if n_params else ""
        print(f"\n  [{name}]{param_str} Perplexity Matrix:")
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

        # Gate histogram
        if res.get('gate_hist') is not None:
            gh = res['gate_hist']
            print(f"    Gates: n={len(gh)}, mean={gh.mean():.3f}, "
                  f"<0.1={np.mean(gh < 0.1):.0%}, >0.9={np.mean(gh > 0.9):.0%}")

        # Routing info
        if res.get('routing_info'):
            ri = res['routing_info']
            print(f"    Experts: {ri['frozen']} frozen, {ri['gated']} gated, "
                  f"{ri['total']} total")

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

    # Efficiency comparison
    has_params = any(res.get('n_params') for res in results_list)
    has_timing = any(res.get('elapsed_s') for res in results_list)
    if has_params or has_timing:
        print(f"\n  Efficiency:")
        for res in results_list:
            n_p = res.get('n_params', 0)
            elapsed = res.get('elapsed_s', 0)
            total_steps = res.get('total_steps', 0)
            parts = [f"    {res['name']:>20s}:"]
            if n_p:
                parts.append(f"{n_p:>10,} params")
            if elapsed:
                parts.append(f"{elapsed:>6.0f}s total")
            if total_steps and elapsed:
                parts.append(f"{elapsed / total_steps:.2f} s/step")
            print("  ".join(parts))

    # Final perplexity
    print(f"\n  Final Perplexity (after all {len(DOMAINS)} domains):")
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

    # Mean final perplexity
    print(f"  {'MEAN':>8s}  {'':>8s}", end="")
    for res in results_list:
        ppls = [res['ppl_matrix'][len(DOMAINS) - 1][d] for d in DOMAINS]
        mean_ppl = np.mean(ppls)
        print(f"  {mean_ppl:14.1f}", end="")
    print()


# ── Main ────────────────────────────────────────────────────

CONFIG_RUNNERS = {
    "lora_static": run_lora_static,
    "lora_lifecycle": run_lora_lifecycle,
    "peer_static": run_peer_static,
    "peer_lifecycle": run_peer_lifecycle,
    "peer_fox": run_peer_fox,
    "peer_version_tree": run_peer_version_tree,
    "lora_seqft": run_lora_seqft,
    "lora_ewc": run_lora_ewc,
    "lora_replay": run_lora_replay,
    "lora_olora": run_lora_olora,
}


def run_benchmark():
    args = parse_args()

    # Seed control
    np.random.seed(args.seed)
    mx.random.seed(args.seed)
    print(f"  Seed: {args.seed}")

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
    elif args.configs:
        configs_to_run = [c.strip() for c in args.configs.split(",")]
        for c in configs_to_run:
            if c not in CONFIG_RUNNERS:
                print(f"  ERROR: Unknown config '{c}'. Choose from: {list(CONFIG_RUNNERS.keys())}")
                return
    elif args.config:
        if args.config not in CONFIG_RUNNERS:
            print(f"  ERROR: Unknown config '{args.config}'. "
                  f"Choose from: {list(CONFIG_RUNNERS.keys())}")
            return
        configs_to_run = [args.config]
    else:
        configs_to_run = ["peer_static"]

    # Run each config
    all_results = []
    for config_name in configs_to_run:
        runner = CONFIG_RUNNERS[config_name]
        t_config = time.time()
        result = runner(args, domain_data, baseline_ppl)
        result['elapsed_s'] = time.time() - t_config
        all_results.append(result)

    # Print comparison
    print_results(all_results)

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s")
    print("=" * 72)


if __name__ == "__main__":
    run_benchmark()
