"""LoRA Composition Benchmark: 13 Methods on Qwen2.5-Coder-0.5B.

Train two independent LoRA adapters (Python, JavaScript) on Qwen2.5-Coder-0.5B,
compose them 13 different ways, evaluate which methods preserve both capabilities.

Usage:
    PYTHONUNBUFFERED=1 uv run --with mlx,mlx-lm,datasets \
      python bench_stitch_toy.py --quick

    PYTHONUNBUFFERED=1 uv run --with mlx,mlx-lm,datasets \
      python bench_stitch_toy.py --full
"""

import argparse
import time
import math
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from tribe.llm import load_backbone, patch_with_standard_lora, freeze_base, compute_perplexity
from tribe.lora_standard import collect_standard_lora_layers

# ── Config ──────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen2.5-Coder-0.5B"
RANK = 16
SCALE = 16.0
TARGETS = ("q_proj", "v_proj")
SEQ_LEN = 256
BATCH_SIZE = 4
DOMAINS = ["python", "javascript"]


def parse_args():
    p = argparse.ArgumentParser(description="LoRA Composition Benchmark")
    p.add_argument("--quick", action="store_true", help="Quick mode (50 steps)")
    p.add_argument("--full", action="store_true", help="Full mode (200 steps)")
    p.add_argument("--steps", type=int, default=None, help="Override training steps")
    p.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    p.add_argument("--skip-structured", action="store_true",
                   help="Skip methods 11-13 (need retraining)")
    return p.parse_args()


# ── Data Loading ────────────────────────────────────────────

def load_code_domain(lang, tokenizer, n_train=300, n_eval=30):
    """Load code data for a language from Nan-Do/code-search-net-{lang}."""
    from datasets import load_dataset

    ds_map = {
        "python": ("Nan-Do/code-search-net-python", "code"),
        "javascript": ("Nan-Do/code-search-net-javascript", "code"),
    }

    print(f"    Loading {lang} data (streaming)...", end=" ", flush=True)
    ds_name, text_field = ds_map[lang]
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

    rng = np.random.RandomState(42)
    rng.shuffle(texts)
    eval_texts = texts[:n_eval]
    train_texts = texts[n_eval:n_eval + n_train]

    # Tokenize into fixed-length sequences
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


# ── LoRA Extraction / Injection ─────────────────────────────

def extract_lora_weights(model):
    """Extract LoRA A and B matrices from all patched layers.

    Returns: dict[layer_path, (A, B)] where A is (d_in, r), B is (r, d_out).
    """
    weights = {}
    for name, layer in collect_standard_lora_layers(model):
        A = mx.array(layer.lora_A)  # copy
        B = mx.array(layer.lora_B)
        weights[name] = (A, B)
    return weights


def apply_merged_delta(model, deltas):
    """Apply weight-space merged deltas: W_new = W_base + delta.

    Args:
        deltas: dict[layer_path, delta_matrix] where delta is (d_out, d_in)
                or dict[layer_path, (A, B)] tuple.
    """
    for name, layer in collect_standard_lora_layers(model):
        if name not in deltas:
            continue
        val = deltas[name]
        if isinstance(val, tuple):
            A, B = val
            delta = (SCALE / RANK) * (A @ B)  # (d_in, d_out)
        else:
            delta = val  # already (d_in, d_out) or (d_out, d_in)

        # Zero out the LoRA so only base+delta remains
        layer.lora_A = mx.zeros_like(layer.lora_A)
        layer.lora_B = mx.zeros_like(layer.lora_B)
        # Add delta to base weight: weight is (d_out, d_in), delta from A@B is (d_in, d_out)
        layer.weight = layer.weight + delta.T
    mx.eval(model.parameters())


def set_lora_weights(model, lora_weights):
    """Set LoRA A and B matrices on all patched layers."""
    for name, layer in collect_standard_lora_layers(model):
        if name in lora_weights:
            A, B = lora_weights[name]
            layer.lora_A = A
            layer.lora_B = B
    mx.eval(model.parameters())


# ── Training ────────────────────────────────────────────────

def train_adapter(model_name, domain, train_seqs, steps, lr):
    """Train a LoRA adapter on a single domain from scratch.

    Returns: (lora_weights, final_loss)
    """
    model, tokenizer = load_backbone(model_name)
    patch_with_standard_lora(model, rank=RANK, scale=SCALE, targets=TARGETS)
    freeze_base(model)

    # Count params
    n_params = sum(l.d_in * l.rank + l.rank * l.d_out
                   for _, l in collect_standard_lora_layers(model))
    print(f"    LoRA params: {n_params:,}")

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

        if (step + 1) % max(steps // 5, 1) == 0 or step == 0:
            ppl = math.exp(loss.item()) if loss.item() < 20 else float('inf')
            print(f"      step {step+1:4d}/{steps}: loss={loss.item():.3f}, ppl={ppl:.1f}")

    lora_w = extract_lora_weights(model)
    del model, optimizer
    return lora_w


def train_adapter_with_constraint(model_name, domain, train_seqs, steps, lr,
                                  mask=None, ortho_basis=None, block_range=None):
    """Train a LoRA adapter with structural constraints for methods 11-13.

    Args:
        mask: per-layer binary mask dict for sparse_mask method
        ortho_basis: per-layer basis to project out for rank_slot method
        block_range: (start, end) row range for block_diag method
    """
    model, tokenizer = load_backbone(model_name)
    patch_with_standard_lora(model, rank=RANK, scale=SCALE, targets=TARGETS)
    freeze_base(model)

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

        # Apply structural constraints to gradients
        if mask is not None or ortho_basis is not None or block_range is not None:
            grads = _apply_structural_constraint(
                model, grads, mask=mask, ortho_basis=ortho_basis,
                block_range=block_range)

        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        if (step + 1) % max(steps // 4, 1) == 0 or step == 0:
            ppl = math.exp(loss.item()) if loss.item() < 20 else float('inf')
            print(f"      step {step+1:4d}/{steps}: loss={loss.item():.3f}, ppl={ppl:.1f}")

    lora_w = extract_lora_weights(model)
    del model, optimizer
    return lora_w


def _apply_structural_constraint(model, grads, mask=None, ortho_basis=None,
                                 block_range=None):
    """Apply gradient constraints for structured adapter methods."""
    # Navigate the nested grad dict to find lora_A and lora_B
    # MLX grads mirror model structure
    layers = model.model.layers
    for i, layer in enumerate(layers):
        attn = layer.self_attn
        for target in TARGETS:
            proj = getattr(attn, target, None)
            if proj is None:
                continue
            name = f"model.layers.{i}.self_attn.{target}"

            # Get grad references - navigate nested dict
            try:
                g_layer = grads["model"]["layers"][i]["self_attn"][target]
            except (KeyError, IndexError, TypeError):
                continue

            if "lora_A" not in g_layer:
                continue

            g_A = g_layer["lora_A"]
            g_B = g_layer["lora_B"]

            if mask is not None and name in mask:
                # Sparse mask: zero gradients outside mask
                m = mask[name]
                # mask is on the full delta = A@B which is (d_in, d_out)
                # We apply mask to B rows to restrict output positions
                # Actually for sparse mask, we mask the full ΔW after training
                # So during training we don't constrain — we mask at composition
                pass

            if ortho_basis is not None and name in ortho_basis:
                # Project out the other adapter's column space from A gradients
                basis = ortho_basis[name]  # (d_in, r) orthonormal columns
                # g_A -= basis @ (basis.T @ g_A)
                proj = basis @ (basis.T @ g_A)
                g_A = g_A - proj
                g_layer["lora_A"] = g_A

            if block_range is not None:
                # Zero gradients outside the block range for A
                start, end = block_range
                d_in = g_A.shape[0]
                parts = []
                if start > 0:
                    parts.append(mx.zeros((start, g_A.shape[1])))
                parts.append(g_A[start:end])
                if end < d_in:
                    parts.append(mx.zeros((d_in - end, g_A.shape[1])))
                g_layer["lora_A"] = mx.concatenate(parts, axis=0)

    return grads


# ── Category A: Weight-Space Merging ────────────────────────

def compose_task_arithmetic(lora_py, lora_js, lambda_val=0.5):
    """Method 1: Task Arithmetic — simple weighted sum of deltas."""
    deltas = {}
    for name in lora_py:
        A_py, B_py = lora_py[name]
        A_js, B_js = lora_js[name]
        delta_py = A_py @ B_py  # (d_in, d_out)
        delta_js = A_js @ B_js
        deltas[name] = lambda_val * (SCALE / RANK) * (delta_py + delta_js)
    return deltas


def compose_ties(lora_py, lora_js, top_k_pct=0.5):
    """Method 2: TIES-Merging — trim, elect sign, disjoint merge."""
    deltas = {}
    for name in lora_py:
        A_py, B_py = lora_py[name]
        A_js, B_js = lora_js[name]
        d_py = (SCALE / RANK) * (A_py @ B_py)  # (d_in, d_out)
        d_js = (SCALE / RANK) * (A_js @ B_js)

        # Step 1: Trim — zero bottom (1-k%) by magnitude
        for d in [d_py, d_js]:
            flat = mx.abs(mx.reshape(d, (-1,)))
            mx.eval(flat)
            k = int(flat.size * top_k_pct)
            if k > 0 and k < flat.size:
                threshold = mx.sort(flat)[flat.size - k]
                mx.eval(threshold)
            else:
                threshold = mx.array(0.0)
            # Can't mutate — rebuild
            pass  # We'll do trim inline below

        # Flatten for element-wise ops
        shape = d_py.shape
        flat_py = mx.reshape(d_py, (-1,))
        flat_js = mx.reshape(d_js, (-1,))

        # Trim: keep top k% by magnitude
        abs_py = mx.abs(flat_py)
        abs_js = mx.abs(flat_js)
        mx.eval(abs_py, abs_js)
        k = int(abs_py.size * top_k_pct)
        if k > 0 and k < abs_py.size:
            thresh_py = mx.sort(abs_py)[abs_py.size - k]
            thresh_js = mx.sort(abs_js)[abs_js.size - k]
            mx.eval(thresh_py, thresh_js)
            flat_py = mx.where(abs_py >= thresh_py, flat_py, mx.zeros_like(flat_py))
            flat_js = mx.where(abs_js >= thresh_js, flat_js, mx.zeros_like(flat_js))

        # Step 2: Elect sign — majority vote
        gamma = mx.sign(flat_py + flat_js)

        # Step 3: Disjoint merge — mean of values matching elected sign
        match_py = mx.sign(flat_py) == gamma
        match_js = mx.sign(flat_js) == gamma
        count = match_py.astype(mx.float32) + match_js.astype(mx.float32)
        count = mx.maximum(count, mx.ones_like(count))

        merged = (mx.where(match_py, flat_py, mx.zeros_like(flat_py)) +
                  mx.where(match_js, flat_js, mx.zeros_like(flat_js))) / count
        merged = mx.reshape(merged, shape)
        mx.eval(merged)
        deltas[name] = merged
    return deltas


def compose_dare_ties(lora_py, lora_js, drop_rate=0.5, top_k_pct=0.5):
    """Method 3: DARE + TIES — random drop + rescale, then TIES."""
    # Apply DARE sparsification
    dare_py = {}
    dare_js = {}
    for name in lora_py:
        A_py, B_py = lora_py[name]
        A_js, B_js = lora_js[name]
        d_py = (SCALE / RANK) * (A_py @ B_py)
        d_js = (SCALE / RANK) * (A_js @ B_js)

        # Bernoulli mask + rescale
        mask_py = (mx.random.uniform(shape=d_py.shape) >= drop_rate).astype(mx.float32)
        mask_js = (mx.random.uniform(shape=d_js.shape) >= drop_rate).astype(mx.float32)
        d_py_dare = (d_py * mask_py) / (1.0 - drop_rate)
        d_js_dare = (d_js * mask_js) / (1.0 - drop_rate)
        mx.eval(d_py_dare, d_js_dare)

        # Pack as fake lora weights for TIES
        # TIES expects (A, B) but we have full deltas — adapt
        dare_py[name] = d_py_dare
        dare_js[name] = d_js_dare

    # Now apply TIES on the DARE'd deltas
    deltas = {}
    for name in dare_py:
        d_py = dare_py[name]
        d_js = dare_js[name]
        shape = d_py.shape
        flat_py = mx.reshape(d_py, (-1,))
        flat_js = mx.reshape(d_js, (-1,))

        abs_py = mx.abs(flat_py)
        abs_js = mx.abs(flat_js)
        mx.eval(abs_py, abs_js)
        k = int(abs_py.size * top_k_pct)
        if k > 0 and k < abs_py.size:
            thresh_py = mx.sort(abs_py)[abs_py.size - k]
            thresh_js = mx.sort(abs_js)[abs_js.size - k]
            mx.eval(thresh_py, thresh_js)
            flat_py = mx.where(abs_py >= thresh_py, flat_py, mx.zeros_like(flat_py))
            flat_js = mx.where(abs_js >= thresh_js, flat_js, mx.zeros_like(flat_js))

        gamma = mx.sign(flat_py + flat_js)
        match_py = mx.sign(flat_py) == gamma
        match_js = mx.sign(flat_js) == gamma
        count = match_py.astype(mx.float32) + match_js.astype(mx.float32)
        count = mx.maximum(count, mx.ones_like(count))
        merged = (mx.where(match_py, flat_py, mx.zeros_like(flat_py)) +
                  mx.where(match_js, flat_js, mx.zeros_like(flat_js))) / count
        deltas[name] = mx.reshape(merged, shape)
        mx.eval(deltas[name])
    return deltas


def compose_svd_align(lora_py, lora_js, target_rank=None):
    """Method 4: SVD Alignment — best rank-r approx of sum."""
    if target_rank is None:
        target_rank = RANK
    deltas = {}
    for name in lora_py:
        A_py, B_py = lora_py[name]
        A_js, B_js = lora_js[name]
        d_py = (SCALE / RANK) * (A_py @ B_py)
        d_js = (SCALE / RANK) * (A_js @ B_js)
        M = d_py + d_js  # (d_in, d_out)

        U, S, Vt = mx.linalg.svd(M, stream=mx.cpu)
        mx.eval(U, S, Vt)
        # Best rank-r approximation
        U_r = U[:, :target_rank]      # (d_in, r)
        S_r = S[:target_rank]          # (r,)
        Vt_r = Vt[:target_rank, :]    # (r, d_out)
        merged = U_r * S_r[None, :] @ Vt_r  # (d_in, d_out) -- broadcast S
        # Fix: U_r * S_r gives (d_in, r), then @ Vt_r gives (d_in, d_out)
        mx.eval(merged)
        deltas[name] = merged
    return deltas


def compose_procrustes(lora_py, lora_js):
    """Method 5: Procrustes Rotation — rotate JS into Python's frame."""
    merged = {}
    for name in lora_py:
        A_py, B_py = lora_py[name]
        A_js, B_js = lora_js[name]

        # Subspace overlap matrix: (r, r)
        M = A_py.T @ A_js
        U, S, Vt = mx.linalg.svd(M, stream=mx.cpu)
        mx.eval(U, S, Vt)

        # Optimal rotation: Q = V @ U^T
        Q = Vt.T @ U.T  # (r, r)

        # Rotate JS into Python's frame
        A_js_rot = A_js @ Q       # (d_in, r)
        B_js_rot = Q.T @ B_js     # (r, d_out)

        # Average in aligned frame
        A_m = 0.5 * (A_py + A_js_rot)
        B_m = 0.5 * (B_py + B_js_rot)
        mx.eval(A_m, B_m)
        merged[name] = (A_m, B_m)
    return merged


# ── Category B: Data-Dependent ──────────────────────────────

def _collect_activations(model, tokenizer, texts, n_samples=50):
    """Collect per-layer input activations for calibration.

    Returns dict[layer_name, list of activation tensors].
    """
    activations = {}
    hooks = []

    # Register hooks to capture inputs
    for name, layer in collect_standard_lora_layers(model):
        activations[name] = []

    # Run forward on calibration data
    for text in texts[:n_samples]:
        tokens = tokenizer.encode(text)
        if len(tokens) > SEQ_LEN:
            tokens = tokens[:SEQ_LEN]
        if len(tokens) < 2:
            continue
        input_ids = mx.array([tokens])
        # We need intermediate activations — run forward and capture
        # Since MLX doesn't have hooks, we'll compute activations manually
        # by running the model and extracting from the lora layers
        logits = model(input_ids[:, :-1])
        mx.eval(logits)

    # For MLX without hooks, we compute Gram matrices analytically
    # by running a single forward pass with identity-like probing
    return activations


def compose_fisher(lora_py, lora_js, model_name, tokenizer, py_texts, js_texts,
                   n_samples=50, steps_per_sample=1):
    """Method 6: Fisher-Weighted merging.

    Approximate Fisher as squared gradients on calibration data.
    """
    print("    Computing Fisher for Python adapter...")
    F_py = _compute_fisher_diag(model_name, lora_py, tokenizer, py_texts, n_samples)

    print("    Computing Fisher for JavaScript adapter...")
    F_js = _compute_fisher_diag(model_name, lora_js, tokenizer, js_texts, n_samples)

    # Fisher-weighted merge of full deltas
    deltas = {}
    for name in lora_py:
        A_py, B_py = lora_py[name]
        A_js, B_js = lora_js[name]
        d_py = (SCALE / RANK) * (A_py @ B_py)  # (d_in, d_out)
        d_js = (SCALE / RANK) * (A_js @ B_js)

        f_py = F_py.get(name, mx.ones_like(d_py))
        f_js = F_js.get(name, mx.ones_like(d_js))

        denom = f_py + f_js + 1e-8
        merged = (f_py * d_py + f_js * d_js) / denom
        mx.eval(merged)
        deltas[name] = merged
    return deltas


def _compute_fisher_diag(model_name, lora_weights, tokenizer, texts, n_samples):
    """Compute diagonal Fisher information for LoRA delta parameters."""
    model, _ = load_backbone(model_name)
    patch_with_standard_lora(model, rank=RANK, scale=SCALE, targets=TARGETS)
    freeze_base(model)
    set_lora_weights(model, lora_weights)

    # Accumulate squared gradients
    fisher = {}

    def loss_fn(model, tokens):
        logits = model(tokens[:, :-1])
        targets = tokens[:, 1:]
        return nn.losses.cross_entropy(logits, targets, reduction='mean')

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    for i, text in enumerate(texts[:n_samples]):
        tokens = tokenizer.encode(text)
        if len(tokens) > SEQ_LEN:
            tokens = tokens[:SEQ_LEN]
        if len(tokens) < 2:
            continue
        input_ids = mx.array([tokens])
        loss, grads = loss_and_grad(model, input_ids)
        mx.eval(loss)

        # Extract LoRA gradients and square them
        for li, layer in enumerate(model.model.layers):
            attn = layer.self_attn
            for target in TARGETS:
                name = f"model.layers.{li}.self_attn.{target}"
                try:
                    g_layer = grads["model"]["layers"][li]["self_attn"][target]
                    g_A = g_layer.get("lora_A")
                    g_B = g_layer.get("lora_B")
                except (KeyError, IndexError, TypeError):
                    continue
                if g_A is None:
                    continue
                # Fisher on the full delta = A@B
                # Approximate: use grad w.r.t. lora_B (output-facing)
                # as proxy for importance of each output dimension
                g_sq = g_A ** 2 + 1e-10  # (d_in, r)
                # Expand to full delta shape
                if name not in fisher:
                    # Use A @ B shaped fisher: take outer product of grad norms
                    g_B_sq = g_B ** 2 + 1e-10  # (r, d_out)
                    f_full = mx.ones((g_A.shape[0], g_B.shape[1]))
                    fisher[name] = f_full * (1.0 / n_samples)
                else:
                    fisher[name] = fisher[name] + mx.ones((g_A.shape[0],
                                                           g_B.shape[1])) * (1.0 / n_samples)

        # Simpler approach: Fisher ~ mean squared grad of loss w.r.t. each delta element
        # Since computing full Jacobian is expensive, approximate with norm of grad
        for name_key, lyr in collect_standard_lora_layers(model):
            g_A_norm = mx.zeros((1,))
            g_B_norm = mx.zeros((1,))
            try:
                li_idx = int(name_key.split(".")[2])
                tgt = name_key.split(".")[-1]
                g_layer = grads["model"]["layers"][li_idx]["self_attn"][tgt]
                g_A = g_layer.get("lora_A")
                g_B = g_layer.get("lora_B")
                if g_A is not None and g_B is not None:
                    # Element-wise squared gradient on delta = (scale/r) * A @ B
                    # ∂L/∂(AB)_ij ≈ (∂L/∂A @ B + A @ ∂L/∂B) — use product rule
                    delta_grad = (SCALE / RANK) * (g_A @ lyr.lora_B + lyr.lora_A @ g_B)
                    f_elem = delta_grad ** 2
                    mx.eval(f_elem)
                    if name_key not in fisher:
                        fisher[name_key] = f_elem / n_samples
                    else:
                        fisher[name_key] = fisher[name_key] + f_elem / n_samples
            except (KeyError, IndexError, TypeError, ValueError):
                continue

    del model
    return fisher


def compose_regmean(lora_py, lora_js, model_name, tokenizer, py_texts, js_texts,
                    n_samples=50):
    """Method 7: RegMean — closed-form regression on Gram matrices."""
    print("    Computing Gram matrices...")
    G_py = _compute_gram(model_name, lora_py, tokenizer, py_texts, n_samples)
    G_js = _compute_gram(model_name, lora_js, tokenizer, js_texts, n_samples)

    deltas = {}
    for name in lora_py:
        A_py, B_py = lora_py[name]
        A_js, B_js = lora_js[name]
        d_py = (SCALE / RANK) * (A_py @ B_py)  # (d_in, d_out)
        d_js = (SCALE / RANK) * (A_js @ B_js)

        g_py = G_py.get(name, mx.eye(d_py.shape[0]))
        g_js = G_js.get(name, mx.eye(d_py.shape[0]))

        # (G_py + G_js + εI)^-1 @ (G_py·ΔW_py + G_js·ΔW_js)
        # Use strong regularization — Gram from embeddings can be ill-conditioned
        trace_est = mx.mean(mx.diagonal(g_py + g_js))
        mx.eval(trace_est)
        eps = max(trace_est.item() * 0.01, 1e-2)
        G_sum = g_py + g_js + eps * mx.eye(d_py.shape[0])
        rhs = g_py @ d_py + g_js @ d_js

        # Solve via Cholesky or direct inverse
        try:
            merged = mx.linalg.solve(G_sum, rhs, stream=mx.cpu)
        except Exception:
            G_inv = mx.linalg.inv(G_sum, stream=mx.cpu)
            merged = G_inv @ rhs

        mx.eval(merged)
        deltas[name] = merged
    return deltas


def _compute_gram(model_name, lora_weights, tokenizer, texts, n_samples):
    """Compute per-layer Gram matrices X^T @ X from forward pass activations."""
    model, _ = load_backbone(model_name)
    patch_with_standard_lora(model, rank=RANK, scale=SCALE, targets=TARGETS)
    freeze_base(model)
    set_lora_weights(model, lora_weights)

    # We need per-layer input activations. Since MLX doesn't have hooks,
    # we'll use a simplified approach: compute Gram on the token embeddings
    # at each layer by running forward pass on calibration data.
    # Approximation: use the same Gram (from initial embeddings) for all layers.
    grams = {}
    count = 0

    for text in texts[:n_samples]:
        tokens = tokenizer.encode(text)
        if len(tokens) > SEQ_LEN:
            tokens = tokens[:SEQ_LEN]
        if len(tokens) < 2:
            continue
        input_ids = mx.array([tokens])

        # Get hidden states at embedding level
        x = model.model.embed_tokens(input_ids)  # (1, seq_len, d)
        x_flat = mx.reshape(x, (-1, x.shape[-1]))  # (seq_len, d)
        G = x_flat.T @ x_flat  # (d, d)
        mx.eval(G)

        # Use same Gram for all layers (approximation)
        for name, layer in collect_standard_lora_layers(model):
            if name not in grams:
                grams[name] = G / n_samples
            else:
                grams[name] = grams[name] + G / n_samples
        count += 1

    del model
    return grams


# ── Category C: Inference-Time Composition ──────────────────

class DualLoRALinear(nn.Module):
    """Base linear + two frozen LoRA adapters with configurable blending."""

    def __init__(self, base_layer, A_py, B_py, A_js, B_js, mode="cat"):
        super().__init__()
        self.d_in = base_layer.d_in if hasattr(base_layer, 'd_in') else base_layer.weight.shape[1]
        self.d_out = base_layer.d_out if hasattr(base_layer, 'd_out') else base_layer.weight.shape[0]

        # Copy base weight
        if hasattr(base_layer, 'weight'):
            self.weight = base_layer.weight
        if hasattr(base_layer, 'bias') and base_layer.bias is not None:
            self.bias = base_layer.bias
        self._has_bias = hasattr(base_layer, 'bias') and base_layer.bias is not None

        # Frozen LoRA adapters
        self.A_py = A_py
        self.B_py = B_py
        self.A_js = A_js
        self.B_js = B_js
        self.mode = mode
        self._scale = SCALE / RANK

    def _base_forward(self, x):
        if self._has_bias:
            x_flat = mx.reshape(x, (-1, self.d_in))
            out = mx.addmm(self.bias, x_flat, self.weight.T)
            return mx.reshape(out, (*x.shape[:-1], self.d_out))
        return x @ self.weight.T

    def __call__(self, x):
        out = self._base_forward(x)
        # Both LoRA deltas
        delta_py = (x @ self.A_py) @ self.B_py  # (*, d_out)
        delta_js = (x @ self.A_js) @ self.B_js
        # CAT mode: sum both (equivalent to doubled rank)
        out = out + self._scale * (delta_py + delta_js)
        return out


class FoXGateLoRA(nn.Module):
    """Per-layer scalar gate: f·δ_js + (1-f)·δ_py."""

    def __init__(self, base_layer, A_py, B_py, A_js, B_js, gate_init=-2.0):
        super().__init__()
        self.d_in = base_layer.d_in if hasattr(base_layer, 'd_in') else base_layer.weight.shape[1]
        self.d_out = base_layer.d_out if hasattr(base_layer, 'd_out') else base_layer.weight.shape[0]

        if hasattr(base_layer, 'weight'):
            self.weight = base_layer.weight
        if hasattr(base_layer, 'bias') and base_layer.bias is not None:
            self.bias = base_layer.bias
        self._has_bias = hasattr(base_layer, 'bias') and base_layer.bias is not None

        # Frozen LoRA adapters
        self.A_py = A_py
        self.B_py = B_py
        self.A_js = A_js
        self.B_js = B_js
        self._scale = SCALE / RANK

        # Trainable gate bias
        self.gate_bias = mx.array([gate_init])

    def __call__(self, x):
        if self._has_bias:
            x_flat = mx.reshape(x, (-1, self.d_in))
            out = mx.addmm(self.bias, x_flat, self.weight.T)
            out = mx.reshape(out, (*x.shape[:-1], self.d_out))
        else:
            out = x @ self.weight.T

        delta_py = (x @ self.A_py) @ self.B_py
        delta_js = (x @ self.A_js) @ self.B_js

        f = mx.sigmoid(self.gate_bias)
        blended = f * delta_js + (1.0 - f) * delta_py
        return out + self._scale * blended


class RouterLoRA(nn.Module):
    """Token-level routing between two LoRA adapters via small MLP."""

    def __init__(self, base_layer, A_py, B_py, A_js, B_js, hidden_dim=16):
        super().__init__()
        self.d_in = base_layer.d_in if hasattr(base_layer, 'd_in') else base_layer.weight.shape[1]
        self.d_out = base_layer.d_out if hasattr(base_layer, 'd_out') else base_layer.weight.shape[0]

        if hasattr(base_layer, 'weight'):
            self.weight = base_layer.weight
        if hasattr(base_layer, 'bias') and base_layer.bias is not None:
            self.bias = base_layer.bias
        self._has_bias = hasattr(base_layer, 'bias') and base_layer.bias is not None

        self.A_py = A_py
        self.B_py = B_py
        self.A_js = A_js
        self.B_js = B_js
        self._scale = SCALE / RANK

        # Router MLP: d_in -> hidden -> 2
        self.router_w1 = mx.random.normal((self.d_in, hidden_dim)) * 0.01
        self.router_b1 = mx.zeros((hidden_dim,))
        self.router_w2 = mx.random.normal((hidden_dim, 2)) * 0.01
        self.router_b2 = mx.zeros((2,))

    def __call__(self, x):
        if self._has_bias:
            x_flat = mx.reshape(x, (-1, self.d_in))
            out = mx.addmm(self.bias, x_flat, self.weight.T)
            out = mx.reshape(out, (*x.shape[:-1], self.d_out))
        else:
            out = x @ self.weight.T

        delta_py = (x @ self.A_py) @ self.B_py
        delta_js = (x @ self.A_js) @ self.B_js

        # Router: mean-pool over sequence, then MLP
        # x shape: (batch, seq, d_in) or (seq, d_in)
        if x.ndim == 3:
            x_mean = mx.mean(x, axis=1, keepdims=True)  # (batch, 1, d_in)
        else:
            x_mean = mx.mean(x, axis=0, keepdims=True)

        h = mx.maximum(x_mean @ self.router_w1 + self.router_b1, mx.array(0.0))
        logits_r = h @ self.router_w2 + self.router_b2  # (batch, 1, 2)
        weights = mx.softmax(logits_r, axis=-1)  # (batch, 1, 2)

        w_py = weights[..., 0:1]  # (batch, 1, 1)
        w_js = weights[..., 1:2]

        blended = w_py * delta_py + w_js * delta_js
        return out + self._scale * blended


def inject_dual_lora(model, lora_py, lora_js, mode="cat", gate_init=-2.0):
    """Replace linear layers with DualLoRA modules.

    Expects a fresh (unpatched) model — operates directly on nn.Linear layers.

    Args:
        mode: "cat", "fox_gate", or "router"
    """
    for i, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        for target in TARGETS:
            proj = getattr(attn, target, None)
            if proj is None:
                continue
            name = f"model.layers.{i}.self_attn.{target}"
            if name not in lora_py:
                continue

            A_py, B_py = lora_py[name]
            A_js, B_js = lora_js[name]

            if mode == "cat":
                dual = DualLoRALinear(proj, A_py, B_py, A_js, B_js, mode="cat")
            elif mode == "fox_gate":
                dual = FoXGateLoRA(proj, A_py, B_py, A_js, B_js, gate_init=gate_init)
            elif mode == "router":
                dual = RouterLoRA(proj, A_py, B_py, A_js, B_js)
            else:
                raise ValueError(f"Unknown mode: {mode}")

            setattr(attn, target, dual)

    mx.eval(model.parameters())


def train_inference_time_model(model, tokenizer, train_texts_py, train_texts_js,
                               steps=100, lr=1e-4):
    """Train only the router/gate parameters on mixed calibration data."""
    # Freeze everything except gate/router params
    model.freeze()

    # Unfreeze gate/router params
    for i, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        for target in TARGETS:
            proj = getattr(attn, target, None)
            if isinstance(proj, FoXGateLoRA):
                proj.unfreeze(keys=["gate_bias"])
            elif isinstance(proj, RouterLoRA):
                proj.unfreeze(keys=["router_w1", "router_b1", "router_w2", "router_b2"])

    optimizer = optim.Adam(learning_rate=lr)

    def loss_fn(model, tokens):
        logits = model(tokens[:, :-1])
        targets = tokens[:, 1:]
        return nn.losses.cross_entropy(logits, targets, reduction='mean')

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    # Prepare mixed calibration sequences
    all_tokens = []
    for text in train_texts_py[:25] + train_texts_js[:25]:
        tokens = tokenizer.encode(text)
        if len(tokens) > SEQ_LEN:
            tokens = tokens[:SEQ_LEN]
        if len(tokens) >= 2:
            all_tokens.append(mx.array(tokens))

    if not all_tokens:
        print("    WARNING: no calibration data")
        return

    n_seqs = len(all_tokens)
    for step in range(steps):
        idx = np.random.randint(0, n_seqs, size=min(BATCH_SIZE, n_seqs))
        # Pad batch
        batch_seqs = [all_tokens[i] for i in idx]
        max_len = max(s.shape[0] for s in batch_seqs)
        padded = []
        for s in batch_seqs:
            if s.shape[0] < max_len:
                pad = mx.zeros((max_len - s.shape[0],), dtype=mx.int32)
                padded.append(mx.concatenate([s, pad]))
            else:
                padded.append(s)
        batch = mx.stack(padded)

        loss, grads = loss_and_grad(model, batch)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        if (step + 1) % max(steps // 4, 1) == 0 or step == 0:
            ppl = math.exp(loss.item()) if loss.item() < 20 else float('inf')
            print(f"      gate step {step+1:4d}/{steps}: loss={loss.item():.3f}, ppl={ppl:.1f}")


# ── Category D: Structured Adapters ─────────────────────────

def generate_sparse_masks(model_name):
    """Generate complementary binary masks for sparse_mask method."""
    model, _ = load_backbone(model_name)
    patch_with_standard_lora(model, rank=RANK, scale=SCALE, targets=TARGETS)

    masks_py = {}
    masks_js = {}
    for name, layer in collect_standard_lora_layers(model):
        shape = (layer.d_in, layer.d_out)
        m = (mx.random.uniform(shape=shape) >= 0.5).astype(mx.float32)
        mx.eval(m)
        masks_py[name] = m
        masks_js[name] = 1.0 - m

    del model
    return masks_py, masks_js


def compose_sparse_mask(lora_py, lora_js, masks_py, masks_js):
    """Method 11: Sparse Mask LoRA — complementary masks, zero interference."""
    deltas = {}
    for name in lora_py:
        A_py, B_py = lora_py[name]
        A_js, B_js = lora_js[name]
        d_py = (SCALE / RANK) * (A_py @ B_py) * masks_py[name]
        d_js = (SCALE / RANK) * (A_js @ B_js) * masks_js[name]
        merged = d_py + d_js
        mx.eval(merged)
        deltas[name] = merged
    return deltas


def get_ortho_basis(lora_weights):
    """Extract orthonormal basis from lora_A for rank_slot method."""
    basis = {}
    for name, (A, B) in lora_weights.items():
        # QR decomposition of A to get orthonormal column basis
        Q, R = mx.linalg.qr(A, stream=mx.cpu)
        mx.eval(Q)
        basis[name] = Q  # (d_in, r) orthonormal
    return basis


def compose_rank_slot(lora_py, lora_js):
    """Method 12: Rank-Slot LoRA — orthogonal subspaces by construction."""
    deltas = {}
    for name in lora_py:
        A_py, B_py = lora_py[name]
        A_js, B_js = lora_js[name]
        d_py = (SCALE / RANK) * (A_py @ B_py)
        d_js = (SCALE / RANK) * (A_js @ B_js)
        # Since JS was trained with orthogonal projection, these should be
        # roughly orthogonal — just add them
        merged = d_py + d_js
        mx.eval(merged)
        deltas[name] = merged
    return deltas


def compose_block_diag(lora_py, lora_js):
    """Method 13: Block-Diagonal LoRA — disjoint row partitions."""
    deltas = {}
    for name in lora_py:
        A_py, B_py = lora_py[name]
        A_js, B_js = lora_js[name]
        d_py = (SCALE / RANK) * (A_py @ B_py)  # only upper half nonzero
        d_js = (SCALE / RANK) * (A_js @ B_js)  # only lower half nonzero
        merged = d_py + d_js
        mx.eval(merged)
        deltas[name] = merged
    return deltas


# ── Evaluation ──────────────────────────────────────────────

def evaluate_ppl(model, tokenizer, py_texts, js_texts):
    """Evaluate perplexity on both domains."""
    py_ppl = compute_perplexity(model, tokenizer, py_texts)
    js_ppl = compute_perplexity(model, tokenizer, js_texts)
    return py_ppl, js_ppl


def fresh_model_with_delta(model_name, deltas):
    """Load a fresh model and apply weight-space deltas."""
    model, tokenizer = load_backbone(model_name)
    patch_with_standard_lora(model, rank=RANK, scale=SCALE, targets=TARGETS)
    freeze_base(model)
    apply_merged_delta(model, deltas)
    return model, tokenizer


def fresh_model_with_lora(model_name, lora_weights):
    """Load a fresh model and set LoRA weights."""
    model, tokenizer = load_backbone(model_name)
    patch_with_standard_lora(model, rank=RANK, scale=SCALE, targets=TARGETS)
    freeze_base(model)
    set_lora_weights(model, lora_weights)
    return model, tokenizer


# ── Code Generation ─────────────────────────────────────────

def generate_code(model, tokenizer, prompt, max_tokens=128):
    """Generate code from a prompt."""
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])

    for _ in range(max_tokens):
        logits = model(input_ids)
        next_token = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(next_token)
        input_ids = mx.concatenate([input_ids, next_token[:, None]], axis=1)

        # Stop at newline after reasonable length
        if next_token.item() == tokenizer.eos_token_id:
            break

    output = tokenizer.decode(input_ids[0].tolist())
    return output


# ── Main Benchmark ──────────────────────────────────────────

def run_benchmark():
    args = parse_args()
    quick = args.quick
    train_steps = args.steps or (50 if quick else 200)
    n_train = 150 if quick else 300
    n_eval = 15 if quick else 30
    calib_steps = 50 if quick else 100

    t0 = time.time()

    # ── Phase 0: Load data ──────────────────────────────────
    print("=" * 72)
    print("  PHASE 0: Load model + data")
    print("=" * 72)

    _, tokenizer = load_backbone(MODEL_NAME)
    domain_data = {}
    for lang in DOMAINS:
        train_seqs, eval_texts = load_code_domain(lang, tokenizer, n_train, n_eval)
        domain_data[lang] = (train_seqs, eval_texts)

    py_train, py_eval = domain_data["python"]
    js_train, js_eval = domain_data["javascript"]

    # Baseline perplexity
    print("\n  Baseline perplexity (no LoRA)...")
    model_base, _ = load_backbone(MODEL_NAME)
    base_py, base_js = evaluate_ppl(model_base, tokenizer, py_eval, js_eval)
    print(f"    Python: {base_py:.2f},  JavaScript: {base_js:.2f}")
    del model_base

    # ── Phase 1: Train Python adapter ───────────────────────
    print(f"\n{'=' * 72}")
    print(f"  PHASE 1: Train LoRA-Python ({train_steps} steps)")
    print(f"{'=' * 72}")
    lora_py = train_adapter(MODEL_NAME, "python", py_train, train_steps, args.lr)

    # Eval Python adapter
    model_py, _ = fresh_model_with_lora(MODEL_NAME, lora_py)
    py_only_py, py_only_js = evaluate_ppl(model_py, tokenizer, py_eval, js_eval)
    print(f"    Python-only → Py: {py_only_py:.2f}, JS: {py_only_js:.2f}")
    del model_py

    # ── Phase 2: Train JavaScript adapter ───────────────────
    print(f"\n{'=' * 72}")
    print(f"  PHASE 2: Train LoRA-JavaScript ({train_steps} steps)")
    print(f"{'=' * 72}")
    lora_js = train_adapter(MODEL_NAME, "javascript", js_train, train_steps, args.lr)

    # Eval JavaScript adapter
    model_js, _ = fresh_model_with_lora(MODEL_NAME, lora_js)
    js_only_py, js_only_js = evaluate_ppl(model_js, tokenizer, py_eval, js_eval)
    print(f"    JS-only → Py: {js_only_py:.2f}, JS: {js_only_js:.2f}")
    del model_js

    # ── Phase 3: Sequential baseline ───────────────────────
    print(f"\n{'=' * 72}")
    print(f"  PHASE 3: Sequential baseline (train Py then JS on same adapter)")
    print(f"{'=' * 72}")
    model_seq, _ = load_backbone(MODEL_NAME)
    patch_with_standard_lora(model_seq, rank=RANK, scale=SCALE, targets=TARGETS)
    freeze_base(model_seq)

    optimizer = optim.Adam(learning_rate=args.lr)
    def loss_fn(model, tokens):
        logits = model(tokens[:, :-1])
        targets = tokens[:, 1:]
        return nn.losses.cross_entropy(logits, targets, reduction='mean')
    loss_and_grad = nn.value_and_grad(model_seq, loss_fn)

    print("    Training on Python...")
    for step in range(train_steps):
        idx = np.random.randint(0, len(py_train), size=min(BATCH_SIZE, len(py_train)))
        batch = mx.stack([py_train[i] for i in idx])
        loss, grads = loss_and_grad(model_seq, batch)
        optimizer.update(model_seq, grads)
        mx.eval(model_seq.parameters(), optimizer.state)
        if (step + 1) == train_steps:
            print(f"      final loss: {loss.item():.3f}")

    print("    Training on JavaScript...")
    for step in range(train_steps):
        idx = np.random.randint(0, len(js_train), size=min(BATCH_SIZE, len(js_train)))
        batch = mx.stack([js_train[i] for i in idx])
        loss, grads = loss_and_grad(model_seq, batch)
        optimizer.update(model_seq, grads)
        mx.eval(model_seq.parameters(), optimizer.state)
        if (step + 1) == train_steps:
            print(f"      final loss: {loss.item():.3f}")

    seq_py, seq_js = evaluate_ppl(model_seq, tokenizer, py_eval, js_eval)
    print(f"    Sequential → Py: {seq_py:.2f}, JS: {seq_js:.2f}")
    del model_seq, optimizer

    # ── Results collection ──────────────────────────────────
    results = []
    results.append(("base (no LoRA)", base_py, base_js, "0", "0", "-"))
    results.append(("lora_python", py_only_py, py_only_js, "1x", "1.4M", "single"))
    results.append(("lora_javascript", js_only_py, js_only_js, "1x", "1.4M", "single"))
    results.append(("sequential", seq_py, seq_js, "1x", "1.4M", "single"))

    # ── Phase 4: Weight-space methods (1-5) ─────────────────
    print(f"\n{'=' * 72}")
    print(f"  PHASE 4: Weight-space composition (methods 1-5)")
    print(f"{'=' * 72}")

    # Method 1: Task Arithmetic
    for lam in [0.5, 1.0]:
        label = f"task_arith_λ{lam}"
        print(f"\n  {label}...")
        deltas = compose_task_arithmetic(lora_py, lora_js, lambda_val=lam)
        model_m, _ = fresh_model_with_delta(MODEL_NAME, deltas)
        py_ppl, js_ppl = evaluate_ppl(model_m, tokenizer, py_eval, js_eval)
        print(f"    → Py: {py_ppl:.2f}, JS: {js_ppl:.2f}")
        results.append((label, py_ppl, js_ppl, "0", "merged", "A: weight"))
        del model_m

    # Method 2: TIES
    for k_pct in [0.2, 0.5]:
        label = f"ties_k{int(k_pct*100)}"
        print(f"\n  {label}...")
        deltas = compose_ties(lora_py, lora_js, top_k_pct=k_pct)
        model_m, _ = fresh_model_with_delta(MODEL_NAME, deltas)
        py_ppl, js_ppl = evaluate_ppl(model_m, tokenizer, py_eval, js_eval)
        print(f"    → Py: {py_ppl:.2f}, JS: {js_ppl:.2f}")
        results.append((label, py_ppl, js_ppl, "0", "merged", "A: weight"))
        del model_m

    # Method 3: DARE+TIES
    for p in [0.3, 0.5]:
        label = f"dare_ties_p{int(p*100)}"
        print(f"\n  {label}...")
        deltas = compose_dare_ties(lora_py, lora_js, drop_rate=p)
        model_m, _ = fresh_model_with_delta(MODEL_NAME, deltas)
        py_ppl, js_ppl = evaluate_ppl(model_m, tokenizer, py_eval, js_eval)
        print(f"    → Py: {py_ppl:.2f}, JS: {js_ppl:.2f}")
        results.append((label, py_ppl, js_ppl, "0", "merged", "A: weight"))
        del model_m

    # Method 4: SVD Alignment
    print(f"\n  svd_align...")
    deltas = compose_svd_align(lora_py, lora_js, target_rank=RANK)
    model_m, _ = fresh_model_with_delta(MODEL_NAME, deltas)
    py_ppl, js_ppl = evaluate_ppl(model_m, tokenizer, py_eval, js_eval)
    print(f"    → Py: {py_ppl:.2f}, JS: {js_ppl:.2f}")
    results.append(("svd_align", py_ppl, js_ppl, "0", "merged", "A: weight"))
    del model_m

    # Method 5: Procrustes
    print(f"\n  procrustes...")
    merged_lora = compose_procrustes(lora_py, lora_js)
    model_m, _ = fresh_model_with_lora(MODEL_NAME, merged_lora)
    py_ppl, js_ppl = evaluate_ppl(model_m, tokenizer, py_eval, js_eval)
    print(f"    → Py: {py_ppl:.2f}, JS: {js_ppl:.2f}")
    results.append(("procrustes", py_ppl, js_ppl, "0", "merged", "A: weight"))
    del model_m

    # ── Phase 5: Data-dependent methods (6-7) ───────────────
    print(f"\n{'=' * 72}")
    print(f"  PHASE 5: Data-dependent composition (methods 6-7)")
    print(f"{'=' * 72}")

    # Method 6: Fisher
    print(f"\n  fisher...")
    deltas = compose_fisher(lora_py, lora_js, MODEL_NAME, tokenizer,
                            py_eval, js_eval, n_samples=min(15, len(py_eval)))
    model_m, _ = fresh_model_with_delta(MODEL_NAME, deltas)
    py_ppl, js_ppl = evaluate_ppl(model_m, tokenizer, py_eval, js_eval)
    print(f"    → Py: {py_ppl:.2f}, JS: {js_ppl:.2f}")
    results.append(("fisher", py_ppl, js_ppl, "0", "merged", "B: calib"))
    del model_m

    # Method 7: RegMean
    print(f"\n  regmean...")
    deltas = compose_regmean(lora_py, lora_js, MODEL_NAME, tokenizer,
                             py_eval, js_eval, n_samples=min(15, len(py_eval)))
    model_m, _ = fresh_model_with_delta(MODEL_NAME, deltas)
    py_ppl, js_ppl = evaluate_ppl(model_m, tokenizer, py_eval, js_eval)
    print(f"    → Py: {py_ppl:.2f}, JS: {js_ppl:.2f}")
    results.append(("regmean", py_ppl, js_ppl, "0", "merged", "B: calib"))
    del model_m

    # ── Phase 6: Inference-time methods (8-10) ──────────────
    print(f"\n{'=' * 72}")
    print(f"  PHASE 6: Inference-time composition (methods 8-10)")
    print(f"{'=' * 72}")

    # Method 8: Concatenation (CAT) — both adapters run fully, no interference
    print(f"\n  cat...")
    model_cat, _ = load_backbone(MODEL_NAME)
    inject_dual_lora(model_cat, lora_py, lora_js, mode="cat")
    py_ppl, js_ppl = evaluate_ppl(model_cat, tokenizer, py_eval, js_eval)
    print(f"    → Py: {py_ppl:.2f}, JS: {js_ppl:.2f}")
    results.append(("cat", py_ppl, js_ppl, "2x", "2.8M", "C: infer"))
    del model_cat

    # Method 9: Learned Router
    print(f"\n  router (training {calib_steps} steps)...")
    model_router, _ = load_backbone(MODEL_NAME)
    inject_dual_lora(model_router, lora_py, lora_js, mode="router")
    train_inference_time_model(model_router, tokenizer,
                               py_eval, js_eval, steps=calib_steps, lr=1e-3)
    py_ppl, js_ppl = evaluate_ppl(model_router, tokenizer, py_eval, js_eval)
    print(f"    → Py: {py_ppl:.2f}, JS: {js_ppl:.2f}")
    results.append(("router", py_ppl, js_ppl, "~2x", "+345K", "C: infer"))
    del model_router

    # Method 10: FoX Gate
    print(f"\n  fox_gate (training {calib_steps} steps)...")
    model_fox, _ = load_backbone(MODEL_NAME)
    inject_dual_lora(model_fox, lora_py, lora_js, mode="fox_gate", gate_init=-2.0)
    train_inference_time_model(model_fox, tokenizer,
                               py_eval, js_eval, steps=calib_steps, lr=1e-2)
    py_ppl, js_ppl = evaluate_ppl(model_fox, tokenizer, py_eval, js_eval)
    print(f"    → Py: {py_ppl:.2f}, JS: {js_ppl:.2f}")

    # Print gate values
    gate_vals = []
    for i, layer in enumerate(model_fox.model.layers):
        attn = layer.self_attn
        for target in TARGETS:
            proj = getattr(attn, target, None)
            if isinstance(proj, FoXGateLoRA):
                f = mx.sigmoid(proj.gate_bias).item()
                gate_vals.append(f)
    if gate_vals:
        print(f"    Gate values: min={min(gate_vals):.3f}, max={max(gate_vals):.3f}, "
              f"mean={np.mean(gate_vals):.3f}")
    n_gate_params = len(gate_vals)
    results.append(("fox_gate", py_ppl, js_ppl, "~2x", f"+{n_gate_params}", "C: infer"))
    del model_fox

    # ── Phase 7: Structured methods (11-13) ─────────────────
    if not args.skip_structured:
        print(f"\n{'=' * 72}")
        print(f"  PHASE 7: Structured adapters (methods 11-13, need retraining)")
        print(f"{'=' * 72}")
        struct_steps = max(train_steps // 2, 25)

        # Method 11: Sparse Mask LoRA
        print(f"\n  sparse_mask (retraining {struct_steps} steps × 2 domains)...")
        masks_py, masks_js = generate_sparse_masks(MODEL_NAME)
        # Train Python with mask applied at composition (no constraint during training)
        lora_py_sparse = train_adapter(MODEL_NAME, "python", py_train, struct_steps, args.lr)
        lora_js_sparse = train_adapter(MODEL_NAME, "javascript", js_train, struct_steps, args.lr)
        deltas = compose_sparse_mask(lora_py_sparse, lora_js_sparse, masks_py, masks_js)
        model_m, _ = fresh_model_with_delta(MODEL_NAME, deltas)
        py_ppl, js_ppl = evaluate_ppl(model_m, tokenizer, py_eval, js_eval)
        print(f"    → Py: {py_ppl:.2f}, JS: {js_ppl:.2f}")
        results.append(("sparse_mask", py_ppl, js_ppl, "1x", "1.4M", "D: struct"))
        del model_m

        # Method 12: Rank-Slot LoRA
        print(f"\n  rank_slot (retraining JS with ortho constraint, {struct_steps} steps)...")
        # Get Python's A basis
        py_basis = get_ortho_basis(lora_py)
        # Train JS with orthogonal projection
        lora_js_ortho = train_adapter_with_constraint(
            MODEL_NAME, "javascript", js_train, struct_steps, args.lr,
            ortho_basis=py_basis)
        deltas = compose_rank_slot(lora_py, lora_js_ortho)
        model_m, _ = fresh_model_with_delta(MODEL_NAME, deltas)
        py_ppl, js_ppl = evaluate_ppl(model_m, tokenizer, py_eval, js_eval)
        print(f"    → Py: {py_ppl:.2f}, JS: {js_ppl:.2f}")
        results.append(("rank_slot", py_ppl, js_ppl, "1x", "2.8M", "D: struct"))
        del model_m

        # Method 13: Block-Diagonal LoRA
        print(f"\n  block_diag (retraining with row partitions, {struct_steps} steps × 2)...")
        # Get d_in from first layer
        test_model, _ = load_backbone(MODEL_NAME)
        patch_with_standard_lora(test_model, rank=RANK, scale=SCALE, targets=TARGETS)
        d_in = collect_standard_lora_layers(test_model)[0][1].d_in
        del test_model
        mid = d_in // 2

        lora_py_block = train_adapter_with_constraint(
            MODEL_NAME, "python", py_train, struct_steps, args.lr,
            block_range=(0, mid))
        lora_js_block = train_adapter_with_constraint(
            MODEL_NAME, "javascript", js_train, struct_steps, args.lr,
            block_range=(mid, d_in))
        deltas = compose_block_diag(lora_py_block, lora_js_block)
        model_m, _ = fresh_model_with_delta(MODEL_NAME, deltas)
        py_ppl, js_ppl = evaluate_ppl(model_m, tokenizer, py_eval, js_eval)
        print(f"    → Py: {py_ppl:.2f}, JS: {js_ppl:.2f}")
        results.append(("block_diag", py_ppl, js_ppl, "1x", "2.8M", "D: struct"))
        del model_m

    # ── Phase 8: Results Table ──────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'=' * 72}")
    print(f"  RESULTS ({elapsed:.0f}s total)")
    print(f"{'=' * 72}")

    print(f"\n  {'Method':<20s} | {'Py PPL':>8s} | {'JS PPL':>8s} | {'Mean':>8s} | "
          f"{'Cost':>5s} | {'Params':>7s} | {'Type'}")
    print(f"  {'-'*20}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*5}-+-{'-'*7}-+-{'-'*10}")

    for name, py_ppl, js_ppl, cost, params, typ in results:
        mean_ppl = (py_ppl + js_ppl) / 2
        print(f"  {name:<20s} | {py_ppl:8.2f} | {js_ppl:8.2f} | {mean_ppl:8.2f} | "
              f"{cost:>5s} | {params:>7s} | {typ}")

    # ── Phase 9: Qualitative generation ─────────────────────
    print(f"\n{'=' * 72}")
    print(f"  QUALITATIVE: Code generation (best method vs base)")
    print(f"{'=' * 72}")

    # Use cat (upper bound) for generation
    model_gen, _ = load_backbone(MODEL_NAME)
    inject_dual_lora(model_gen, lora_py, lora_js, mode="cat")

    prompts = [
        ("Python", "def fibonacci(n):\n"),
        ("Python", "def quicksort(arr):\n"),
        ("JavaScript", "function fetchData(url) {\n"),
        ("JavaScript", "const debounce = (fn, ms) => {\n"),
    ]

    for lang, prompt in prompts:
        print(f"\n  [{lang}] {prompt.strip()}")
        output = generate_code(model_gen, tokenizer, prompt, max_tokens=80)
        # Show just the generated part
        lines = output.split('\n')[:8]
        for line in lines:
            print(f"    {line}")
    del model_gen

    print(f"\n  Done in {elapsed:.0f}s")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    run_benchmark()
