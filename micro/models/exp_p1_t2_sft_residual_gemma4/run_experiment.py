#!/usr/bin/env python3
"""T2.5: SFT-Residual M2P on Gemma 4 E4B.

Replicates Finding #403 (SFT-residual quality_ratio=1.175 on Qwen3-4B) on Gemma 4.

Architecture: B_applied_l = B_sft_l + ΔB_l  with ΔB_l initialized to 0.
Zero-init guarantee: quality at step 0 == SFT quality (82% GSM8K from T2.1).
Training: ΔB_l adapts on GSM8K to improve beyond SFT.

Kill criteria:
  K1044: M2P accuracy after training >= 73.8% (= 0.90 × 82% SFT baseline)
  K1045: B_applied computation time < 10ms on M5 Pro
  K1046: ||B_applied - B_sft||_F = 0 for all 42 layers at step 0

References:
  - He et al. (2016, arXiv:1512.03385) — Residual learning
  - Hu et al. (2022, arXiv:2106.09685) — LoRA
  - Finding #403 — SFT-Residual M2P on Qwen3-4B (quality_ratio=1.175)
  - T2.1 — Gemma 4 math adapter (82% GSM8K)

SMOKE_TEST=1: 20 steps, 5 eval, ~5 min.
Full: 500 steps, 50 eval, ~30 min.
"""

import gc
import json
import os
import re
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

# Memory safety — MANDATORY per CODING_GUIDELINES
device_info = mx.device_info()
total_mem = device_info["memory_size"]
mx.set_memory_limit(total_mem - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

T21_DIR = EXPERIMENT_DIR.parent / "exp_p1_t2_single_domain_training"
ADAPTER_PATH = T21_DIR / "adapters" / "math" / "adapters.safetensors"
TRAIN_JSONL = T21_DIR / "data" / "math" / "train.jsonl"

MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"
IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"
SEED = 42

# Experiment parameters
RANK = 6
SCALE = 6.0
LR = 5e-6         # Conservative LR — ΔB_l adapts carefully around B_sft
GRAD_CLIP = 0.5   # Gradient clipping to prevent ΔB from diverging
BATCH_SIZE = 2
MAX_SEQ_LEN = 256  # Shorter for speed (SFT-residual needs fewer tokens to verify)
N_STEPS = 20 if IS_SMOKE else 500
N_EVAL = 5 if IS_SMOKE else 50

# T2.1 baselines
SFT_ACCURACY = 82.0   # From T2.1 results.json
BASE_ACCURACY = 0.0   # Base model (no adapter) on GSM8K
K1044_THRESHOLD = 0.90 * SFT_ACCURACY  # = 73.8%

# Gemma 4 E4B: global layers every 6th, starting at 5
GLOBAL_INDICES = {5, 11, 17, 23, 29, 35, 41}
N_LAYERS = 42


def log(msg: str) -> None:
    print(msg, flush=True)


def log_memory(label: str = "") -> None:
    active = mx.get_active_memory() / 1e9
    cache = mx.get_cache_memory() / 1e9
    log(f"[MEM {label}] active={active:.2f}GB cache={cache:.2f}GB")


def cleanup(*objects) -> None:
    for obj in objects:
        del obj
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()


# ── Data ─────────────────────────────────────────────────────────────────────

def load_train_samples(tokenizer) -> list:
    """Load tokenized training samples from T2.1 math dataset."""
    samples = []
    lines = TRAIN_JSONL.read_text().strip().split("\n")
    for line in lines:
        if not line.strip():
            continue
        ex = json.loads(line)
        messages = ex.get("messages", [])
        if hasattr(tokenizer, "apply_chat_template"):
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        else:
            text = " ".join(m["content"] for m in messages)
        tokens = tokenizer.encode(text)
        if 4 <= len(tokens) <= MAX_SEQ_LEN:
            samples.append(tokens)
    log(f"  Loaded {len(samples)} training samples")
    return samples


def get_batch(samples: list, step: int) -> mx.array:
    n = len(samples)
    indices = [(step * BATCH_SIZE + i) % n for i in range(BATCH_SIZE)]
    seqs = [samples[i] for i in indices]
    max_len = min(MAX_SEQ_LEN, max(len(s) for s in seqs))
    padded = [s[:max_len] + [0] * (max_len - len(s[:max_len])) for s in seqs]
    return mx.array(padded, dtype=mx.int32)


def load_gsm8k_eval(n: int) -> list:
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
    return [ds[int(i)] for i in idx]


# ── B-matrix loading ──────────────────────────────────────────────────────────

def load_b_sft() -> list:
    """Load SFT B-matrices from T2.1 math adapter. Returns list[42] of mx.array."""
    from safetensors.numpy import load_file
    weights = load_file(str(ADAPTER_PATH))
    B_sft = [None] * N_LAYERS
    for key, val in weights.items():
        if "lora_b" in key:
            parts = key.split(".")
            layer_idx = int(parts[3])  # language_model.model.layers.<idx>.self_attn...
            B_sft[layer_idx] = mx.array(val).astype(mx.float32)
    assert all(b is not None for b in B_sft), "Missing B-matrices for some layers"
    log(f"  Loaded {N_LAYERS} B_sft matrices (local: {RANK}×2048, global: {RANK}×4096)")
    return B_sft


def load_a_matrices() -> list:
    """Load A-matrices from T2.1 math adapter. Returns list[42] of mx.array."""
    from safetensors.numpy import load_file
    weights = load_file(str(ADAPTER_PATH))
    A = [None] * N_LAYERS
    for key, val in weights.items():
        if "lora_a" in key:
            parts = key.split(".")
            layer_idx = int(parts[3])
            A[layer_idx] = mx.array(val).astype(mx.float32)
    assert all(a is not None for a in A), "Missing A-matrices for some layers"
    return A


# ── SFT-Residual Correction Module ───────────────────────────────────────────

class SFTResidualCorrection(nn.Module):
    """B_applied_l = B_sft_l + ΔB_l with ΔB_l initialized to 0.

    Theorem 1 (MATH.md): At step 0, ||ΔB_l||_F = 0 → B_applied_l = B_sft_l exactly.
    Gradient flows from step 1: ∂L/∂ΔB_l = ∂L/∂B_applied_l (chain rule, I identity).
    """

    def __init__(self, b_shapes: list):
        """
        Args:
            b_shapes: List of (rank, d_out_l) tuples for each layer.
        """
        super().__init__()
        # delta_B_l: same shape as B_sft_l, initialized to 0 (Theorem 1)
        self.delta_b = [mx.zeros(shape, dtype=mx.float32) for shape in b_shapes]

    def get_b_applied(self, b_sft: list) -> list:
        """Compute B_applied_l = B_sft_l + ΔB_l for all layers."""
        return [
            b + delta.astype(b.dtype)
            for b, delta in zip(b_sft, self.delta_b)
        ]


# ── Adapter injection ─────────────────────────────────────────────────────────

def inject_lora_a(model, a_matrices: list) -> None:
    """Load SFT A-matrices into the model's LoRA layers (frozen)."""
    from mlx_lm.tuner.lora import LoRALinear
    for li, layer in enumerate(model.layers):
        q = layer.self_attn.q_proj
        if not isinstance(q, LoRALinear):
            lora_q = LoRALinear.from_base(q, r=RANK, scale=SCALE, dropout=0.0)
            layer.self_attn.q_proj = lora_q
        model.layers[li].self_attn.q_proj.lora_a = a_matrices[li].astype(mx.bfloat16)


def set_lora_b(model, b_applied: list) -> None:
    """Set lora_b in each LoRA layer to the given B_applied matrices."""
    for li, layer in enumerate(model.layers):
        model.layers[li].self_attn.q_proj.lora_b = b_applied[li].astype(mx.bfloat16)


# ── K1046: Zero-Init Verification ────────────────────────────────────────────

def verify_zero_init(m2p: SFTResidualCorrection, b_sft: list) -> dict:
    """Verify Theorem 1: ||ΔB_l||_F = 0 for all layers at step 0."""
    log("\n── K1046: Zero-Init Verification ──")
    max_frob = 0.0
    for li, delta in enumerate(m2p.delta_b):
        frob = float(mx.sum(delta ** 2).sqrt().item())
        max_frob = max(max_frob, frob)

    # Also verify B_applied = B_sft via direct comparison
    b_applied = m2p.get_b_applied(b_sft)
    max_diff = 0.0
    for li in range(N_LAYERS):
        diff = float(mx.max(mx.abs(b_applied[li].astype(mx.float32) - b_sft[li])).item())
        max_diff = max(max_diff, diff)

    log(f"  max ||ΔB_l||_F across 42 layers: {max_frob:.2e}")
    log(f"  max |B_applied - B_sft| element-wise: {max_diff:.2e}")
    k1046_pass = max_frob < 1e-6 and max_diff < 1e-6
    log(f"  K1046: {'PASS' if k1046_pass else 'FAIL'} (threshold: < 1e-6)")
    return {"k1046_max_frob": max_frob, "k1046_max_diff": max_diff, "k1046_pass": k1046_pass}


# ── Evaluation ─────────────────────────────────────────────────────────────────

def eval_gsm8k(model, tokenizer, examples: list) -> float:
    """Evaluate GSM8K accuracy with current lora_b in model."""
    from mlx_lm import generate

    correct = 0
    for ex in examples:
        prompt = (
            "Solve the following math problem step by step.\n\n"
            f"{ex['question']}\n\nAnswer:"
        )
        if hasattr(tokenizer, "apply_chat_template"):
            formatted = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            formatted = prompt

        response = generate(model, tokenizer, prompt=formatted, max_tokens=256, verbose=False)

        gt_match = re.search(r"####\s*([\d,\-\.]+)", ex["answer"])
        if not gt_match:
            continue
        gt_ans = gt_match.group(1).replace(",", "").strip()

        pred_match = re.search(r"####\s*([\d,\-\.]+)", response)
        if pred_match and pred_match.group(1).replace(",", "").strip() == gt_ans:
            correct += 1

    return 100.0 * correct / len(examples) if examples else 0.0


# ── Training ──────────────────────────────────────────────────────────────────

def train_m2p(m2p: SFTResidualCorrection, model, tokenizer, b_sft: list,
              samples: list) -> dict:
    """Train delta_B on GSM8K cross-entropy loss."""
    log("\n── Training SFT-Residual M2P ──")

    # Freeze model completely — only m2p.delta_b is trainable
    model.freeze()
    optimizer = optim.Adam(learning_rate=LR)

    def loss_fn(m2p_module):
        B_applied = m2p_module.get_b_applied(b_sft)
        set_lora_b(model, B_applied)
        tokens = get_batch(samples, _step[0])
        logits = model(tokens[:, :-1])
        targets = tokens[:, 1:]
        B, L, V = logits.shape
        return nn.losses.cross_entropy(
            logits.reshape(B * L, V), targets.reshape(B * L), reduction="mean"
        )

    # Use nn.value_and_grad w.r.t. m2p parameters
    val_grad_fn = nn.value_and_grad(m2p, loss_fn)

    losses = []
    _step = [0]  # mutable counter for closure
    report_every = max(1, N_STEPS // 10)
    t0 = time.time()

    for step in range(N_STEPS):
        _step[0] = step
        loss, grads = val_grad_fn(m2p)

        # Gradient clipping — prevents ΔB from growing catastrophically
        flat_grads = [g for _, g in nn.utils.tree_flatten(grads)]
        gnorm_sq = sum(float(mx.sum(g ** 2).item()) for g in flat_grads)
        gnorm = gnorm_sq ** 0.5
        if gnorm > GRAD_CLIP:
            scale_factor = GRAD_CLIP / gnorm
            grads = nn.utils.tree_unflatten(
                [(k, g * scale_factor) for (k, _), g in
                 zip(nn.utils.tree_flatten(grads), flat_grads)]
            )

        optimizer.update(m2p, grads)
        mx.eval(loss, m2p.parameters())

        loss_val = float(loss.item())
        losses.append(loss_val)

        if (step + 1) % report_every == 0 or step == 0 or step == N_STEPS - 1:
            elapsed = time.time() - t0
            log(f"  step {step+1:4d}/{N_STEPS}: loss={loss_val:.4f}  elapsed={elapsed:.1f}s")

    elapsed = time.time() - t0
    log(f"  Training complete: {elapsed:.1f}s, {N_STEPS/elapsed:.1f} steps/s")

    return {
        "train_time_s": round(elapsed, 1),
        "final_loss": round(losses[-1], 4),
        "first_loss": round(losses[0], 4),
        "n_steps": N_STEPS,
    }


# ── K1045: Timing ────────────────────────────────────────────────────────────

def measure_b_applied_time(m2p: SFTResidualCorrection, b_sft: list) -> dict:
    """Measure time to compute B_applied (42 matrix additions)."""
    log("\n── K1045: B_applied Timing ──")
    N_TRIALS = 100

    t0 = time.time()
    for _ in range(N_TRIALS):
        b_app = m2p.get_b_applied(b_sft)
        mx.eval(*b_app)
    elapsed_ms = (time.time() - t0) * 1000 / N_TRIALS

    log(f"  B_applied computation: {elapsed_ms:.3f}ms per call ({N_TRIALS} trials)")
    k1045_pass = elapsed_ms < 10.0
    log(f"  K1045: {'PASS' if k1045_pass else 'FAIL'} (< 10ms threshold)")
    return {"k1045_ms": round(elapsed_ms, 3), "k1045_pass": k1045_pass}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log(f"T2.5: SFT-Residual M2P on Gemma 4 E4B")
    log(f"  SMOKE_TEST={IS_SMOKE}, N_STEPS={N_STEPS}, N_EVAL={N_EVAL}")
    log(f"  K1044 threshold: {K1044_THRESHOLD:.1f}% (= 0.90 × {SFT_ACCURACY:.1f}%)")
    t_start = time.time()

    # ── Phase 0: Load B_sft matrices ──
    log("\n── Phase 0: Load SFT B-Matrices ──")
    b_sft = load_b_sft()
    b_shapes = [b.shape for b in b_sft]
    local_count = sum(1 for s in b_shapes if s[1] == 2048)
    global_count = sum(1 for s in b_shapes if s[1] == 4096)
    log(f"  {local_count} local (6×2048), {global_count} global (6×4096)")

    # ── Phase 0b: Load A-matrices ──
    a_matrices = load_a_matrices()
    log(f"  Loaded {len(a_matrices)} A-matrices")

    # ── Phase 1: Initialize SFTResidualCorrection ──
    log("\n── Phase 1: Initialize M2P Module ──")
    m2p = SFTResidualCorrection(b_shapes)
    n_params = sum(v.size for _, v in nn.utils.tree_flatten(m2p.trainable_parameters()))
    log(f"  M2P trainable params: {n_params:,}")

    # ── Phase 2: K1046 Verification (before any training) ──
    log("\n── Phase 2: K1046 Zero-Init Verification ──")
    k1046_results = verify_zero_init(m2p, b_sft)

    if not k1046_results["k1046_pass"]:
        log("ERROR: K1046 failed! ΔB initialization not zero.")
        json.dump({"is_smoke": IS_SMOKE, **k1046_results, "status": "error"}, open(RESULTS_FILE, "w"))
        return

    # ── Phase 3: Load model ──
    log("\n── Phase 3: Load Gemma 4 E4B ──")
    log_memory("before model load")
    from mlx_lm import load as mlx_load
    model, tokenizer = mlx_load(MODEL_ID)
    log_memory("after model load")

    # Apply LoRA structure with SFT A-matrices
    inject_lora_a(model, a_matrices)
    # Set initial lora_b = B_sft (step-0 state)
    set_lora_b(model, b_sft)
    mx.eval(model.parameters())
    log_memory("after LoRA inject")

    # ── Phase 4: Load training data ──
    log("\n── Phase 4: Load Training Data ──")
    samples = load_train_samples(tokenizer)

    # ── Phase 5: Load eval data ──
    log("\n── Phase 5: Load Eval Data ──")
    eval_examples = load_gsm8k_eval(N_EVAL)
    log(f"  Loaded {len(eval_examples)} eval examples")

    # ── Phase 6: Measure step-0 quality (should match SFT = 82%) ──
    log("\n── Phase 6: Step-0 Quality (K1046 behavioral check) ──")
    # lora_b already = B_sft from Phase 3
    acc_step0 = eval_gsm8k(model, tokenizer, eval_examples)
    log(f"  Step-0 accuracy: {acc_step0:.1f}% (expected ~{SFT_ACCURACY:.1f}%)")

    # ── Phase 7: Training ──
    train_results = train_m2p(m2p, model, tokenizer, b_sft, samples)

    # Set final B_applied in model
    b_final = m2p.get_b_applied(b_sft)
    set_lora_b(model, b_final)
    mx.eval(model.parameters())

    # ── Phase 8: Evaluate after training ──
    log("\n── Phase 8: Final Evaluation ──")
    acc_final = eval_gsm8k(model, tokenizer, eval_examples)
    log(f"  Final accuracy: {acc_final:.1f}%")
    log(f"  SFT accuracy:   {SFT_ACCURACY:.1f}%")

    quality_ratio = acc_final / SFT_ACCURACY if SFT_ACCURACY > 0 else 0.0
    k1044_pass = acc_final >= K1044_THRESHOLD
    log(f"  Quality ratio: {quality_ratio:.3f}")
    log(f"  K1044: {'PASS' if k1044_pass else 'FAIL'} (>= {K1044_THRESHOLD:.1f}%)")

    # ── Phase 9: K1045 Timing ──
    k1045_results = measure_b_applied_time(m2p, b_sft)

    # ── Final delta_B norms ──
    log("\n── ΔB Statistics After Training ──")
    frob_norms = [float(mx.sum(d ** 2).sqrt().item()) for d in m2p.delta_b]
    avg_frob = np.mean(frob_norms)
    max_frob = np.max(frob_norms)
    log(f"  ΔB Frobenius norms: mean={avg_frob:.4f}, max={max_frob:.4f}")
    b_sft_norms = [float(mx.sum(b.astype(mx.float32) ** 2).sqrt().item()) for b in b_sft]
    avg_sft_norm = np.mean(b_sft_norms)
    log(f"  B_sft Frobenius norms: mean={avg_sft_norm:.4f}")
    log(f"  Relative correction: {avg_frob/avg_sft_norm:.4f} × B_sft")

    # ── Summary ──
    total_time = time.time() - t_start
    log(f"\n{'='*60}")
    log(f"K1044: {'PASS' if k1044_pass else 'FAIL'} — accuracy={acc_final:.1f}% >= {K1044_THRESHOLD:.1f}%")
    log(f"K1045: {'PASS' if k1045_results['k1045_pass'] else 'FAIL'} — {k1045_results['k1045_ms']:.3f}ms < 10ms")
    log(f"K1046: {'PASS' if k1046_results['k1046_pass'] else 'FAIL'} — max_diff={k1046_results['k1046_max_diff']:.2e}")
    log(f"Total time: {total_time:.1f}s")

    results = {
        "is_smoke": IS_SMOKE,
        "n_steps": N_STEPS,
        "n_eval": N_EVAL,
        "model_id": MODEL_ID,
        # Baselines
        "sft_accuracy": SFT_ACCURACY,
        "base_accuracy": BASE_ACCURACY,
        "k1044_threshold": K1044_THRESHOLD,
        # Results
        "acc_step0": round(acc_step0, 1),
        "acc_final": round(acc_final, 1),
        "quality_ratio": round(quality_ratio, 4),
        "k1044_pass": k1044_pass,
        # K1045
        **k1045_results,
        # K1046
        **k1046_results,
        # Training
        **{f"train_{k}": v for k, v in train_results.items()},
        # Delta stats
        "delta_b_mean_frob": round(float(avg_frob), 4),
        "delta_b_max_frob": round(float(max_frob), 4),
        "b_sft_mean_frob": round(float(avg_sft_norm), 4),
        "relative_correction": round(float(avg_frob / avg_sft_norm), 4),
        "total_time_s": round(total_time, 1),
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    log(f"\nResults saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
