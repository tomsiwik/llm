"""Experiment: Persistent Expert Tree — version-aware cross-version composition.

Tests two kill criteria:
KC1: cross-version composition >5% worse than same-version composition
KC2: persistent structure overhead >15% memory vs mutable

The experiment tests cross-version composition quality: can you take domain A
leaves fine-tuned at time T1 and compose them with domain B leaves fine-tuned
at time T2 (from the same base), and get comparable quality to composing
contemporaneous experts?

Protocol:
1. Pretrain base model on all data (v0) -- 500 steps
2. Create working copy, fine-tune on domain A -- 200 steps -> store as v1
3. Create working copy from v0, fine-tune on domain B -- 200 steps -> store as v2
4. Same-version: compose both from v0 base + simple weight averaging
5. Cross-version: cherry-pick A-leaves from v1, B-leaves from v2 -> v3
6. Calibrate both composed versions -- 100 steps
7. Compare quality and measure memory overhead
"""

import sys
import time
import random
import json
import copy
import os

os.environ["PYTHONUNBUFFERED"] = "1"

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

sys.path.insert(0, "/Users/tom/Code/tomsiwik/llm")

from micro.data import load_names, CharTokenizer, CharDataset, domain_split, train_val_split
from micro.train import ntp_loss, evaluate
from micro.models import get_model


def count_params(model):
    return sum(v.size for _, v in nn.utils.tree_flatten(model.trainable_parameters()))


def train_model(model, train_ds, val_ds, steps=500, batch_size=32, lr=3e-3,
                seed=42, log_every=100):
    """Train and return metrics."""
    rng = random.Random(seed)
    optimizer = optim.Adam(learning_rate=lr)
    loss_and_grad = nn.value_and_grad(model, ntp_loss)

    t0 = time.time()
    for step in range(1, steps + 1):
        inputs, targets = train_ds.get_batch(batch_size, rng)
        loss, grads = loss_and_grad(model, inputs, targets)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        if step % log_every == 0 or step == steps:
            elapsed = time.time() - t0
            print(f"    step {step:4d}/{steps} | loss {loss.item():.4f} | {elapsed:.1f}s")

    val_loss = evaluate(model, val_ds, batch_size)
    return {"val_loss": val_loss, "elapsed_s": time.time() - t0}


def snapshot_tree_params(model):
    """Snapshot all tree parameters as a dict of arrays."""
    params = {}
    for k, v in nn.utils.tree_flatten(model.parameters()):
        if "tree" in k or "norm" in k or "wte" in k or "wpe" in k or "lm_head" in k or "attn" in k:
            params[k] = mx.array(v)
    return params


def restore_tree_params(model, params):
    """Restore model parameters from a snapshot."""
    model.update(nn.utils.tree_unflatten(list(params.items())))
    mx.eval(model.parameters())


def get_leaf_and_gate_params(model, leaf_indices):
    """Get parameters for specific leaves and their ancestor gates."""
    params = {}
    for k, v in nn.utils.tree_flatten(model.parameters()):
        for layer_idx in range(len(model.layers)):
            # Check if this param belongs to a relevant leaf
            for leaf_idx in leaf_indices:
                if f"layers.{layer_idx}.tree.leaves.{leaf_idx}." in k:
                    params[k] = mx.array(v)
            # Get ALL gate params (gates are shared routing, need recalibration)
            if f"layers.{layer_idx}.tree.gates." in k:
                params[k] = mx.array(v)
    return params


def run_experiment(seed=42):
    """Run the full persistent expert tree experiment."""
    print(f"\n{'='*60}")
    print(f"  Persistent Expert Tree Experiment (seed={seed})")
    print(f"{'='*60}")

    # ── Data setup ──────────────────────────────────────────────────
    docs = load_names()
    tokenizer = CharTokenizer(docs)
    domains = domain_split(docs)
    domain_names = sorted(domains.keys())[:2]
    print(f"\nDomains: {domain_names}")

    all_train, all_val = [], []
    domain_datasets = {}
    for name in domain_names:
        tr, va = train_val_split(domains[name], seed=seed)
        all_train.extend(tr)
        all_val.extend(va)
        domain_datasets[name] = {
            "train": CharDataset(tr, tokenizer, block_size=32),
            "val": CharDataset(va, tokenizer, block_size=32),
        }
    joint_train = CharDataset(all_train, tokenizer, block_size=32)
    joint_val = CharDataset(all_val, tokenizer, block_size=32)

    results = {"seed": seed, "domains": domain_names}

    # ── 1. Joint training (upper bound) ──────────────────────────────
    print("\n--- Phase 1: Joint training (upper bound) ---")
    joint_model = get_model("persistent_expert_tree",
                            vocab_size=tokenizer.vocab_size)
    joint_metrics = train_model(joint_model, joint_train, joint_val,
                                steps=500, seed=seed)
    results["joint"] = joint_metrics
    print(f"  Joint val_loss: {joint_metrics['val_loss']:.4f}")

    # ── 2. Pretrain base model (v0) ──────────────────────────────────
    print("\n--- Phase 2: Pretrain base model (v0) ---")
    base_model = get_model("persistent_expert_tree",
                           vocab_size=tokenizer.vocab_size)
    pretrain_metrics = train_model(base_model, joint_train, joint_val,
                                   steps=300, seed=seed)
    results["pretrain"] = pretrain_metrics
    print(f"  Pretrain val_loss: {pretrain_metrics['val_loss']:.4f}")

    # Save v0 snapshot
    v0_params = snapshot_tree_params(base_model)
    v0_output = base_model(mx.array([[1, 2, 3, 4, 5]]))
    mx.eval(v0_output)

    # ── 3a. Fine-tune on domain A (full model, store as v1) ──────────
    print(f"\n--- Phase 3a: Fine-tune {domain_names[0]} (-> v1) ---")
    # Start from v0 snapshot
    restore_tree_params(base_model, v0_params)
    ft_a_metrics = train_model(base_model,
                               domain_datasets[domain_names[0]]["train"],
                               domain_datasets[domain_names[0]]["val"],
                               steps=200, seed=seed)
    v1_params = snapshot_tree_params(base_model)
    results["finetune_A"] = ft_a_metrics
    print(f"  {domain_names[0]} val_loss: {ft_a_metrics['val_loss']:.4f}")

    # ── 3b. Fine-tune on domain B (full model, store as v2) ──────────
    print(f"\n--- Phase 3b: Fine-tune {domain_names[1]} (-> v2) ---")
    # Start from v0 snapshot
    restore_tree_params(base_model, v0_params)
    ft_b_metrics = train_model(base_model,
                               domain_datasets[domain_names[1]]["train"],
                               domain_datasets[domain_names[1]]["val"],
                               steps=200, seed=seed)
    v2_params = snapshot_tree_params(base_model)
    results["finetune_B"] = ft_b_metrics
    print(f"  {domain_names[1]} val_loss: {ft_b_metrics['val_loss']:.4f}")

    # ── 4. Same-version composition (weight averaging of v1 and v2) ──
    print("\n--- Phase 4a: Same-version composition (weight avg v1+v2) ---")
    avg_params = {}
    for k in v1_params:
        avg_params[k] = (v1_params[k] + v2_params[k]) / 2.0
    restore_tree_params(base_model, avg_params)
    same_v_raw = evaluate(base_model, joint_val, batch_size=32)
    results["same_version_raw"] = {"val_loss": same_v_raw}
    print(f"  Same-version raw val_loss: {same_v_raw:.4f}")

    # Calibrate same-version
    print("  Calibrating same-version (100 steps)...")
    same_cal_metrics = train_model(base_model, joint_train, joint_val,
                                   steps=100, seed=seed, lr=3e-3)
    results["same_version_calibrated"] = same_cal_metrics
    print(f"  Same-version calibrated val_loss: {same_cal_metrics['val_loss']:.4f}")

    # ── 4b. Cross-version composition (cherry-pick leaves) ───────────
    print("\n--- Phase 4b: Cross-version composition (A-leaves from v1, B-leaves from v2) ---")
    # Start from v0 (shared base), then overlay domain-specific leaves
    cross_params = dict(v0_params)  # start with base
    for k in v1_params:
        # Take leaves 0-3 from v1 (domain A)
        for leaf_idx in range(4):
            for layer_idx in range(4):
                if f"layers.{layer_idx}.tree.leaves.{leaf_idx}." in k:
                    cross_params[k] = v1_params[k]
        # Take leaves 4-7 from v2 (domain B)
        for leaf_idx in range(4, 8):
            for layer_idx in range(4):
                if f"layers.{layer_idx}.tree.leaves.{leaf_idx}." in k:
                    cross_params[k] = v2_params[k]
        # Take gates from v0 (base, will be recalibrated)
        # (already in cross_params from v0_params initialization)

    restore_tree_params(base_model, cross_params)
    cross_v_raw = evaluate(base_model, joint_val, batch_size=32)
    results["cross_version_raw"] = {"val_loss": cross_v_raw}
    print(f"  Cross-version raw val_loss: {cross_v_raw:.4f}")

    # Calibrate cross-version
    print("  Calibrating cross-version (100 steps)...")
    cross_cal_metrics = train_model(base_model, joint_train, joint_val,
                                    steps=100, seed=seed, lr=3e-3)
    results["cross_version_calibrated"] = cross_cal_metrics
    print(f"  Cross-version calibrated val_loss: {cross_cal_metrics['val_loss']:.4f}")

    # ── 5. Rollback test ─────────────────────────────────────────────
    print("\n--- Phase 5: Rollback test (restore v0) ---")
    restore_tree_params(base_model, v0_params)
    v0_output_after = base_model(mx.array([[1, 2, 3, 4, 5]]))
    mx.eval(v0_output_after)
    rollback_diff = mx.abs(v0_output - v0_output_after).max().item()
    results["rollback_max_diff"] = rollback_diff
    print(f"  Rollback max diff: {rollback_diff:.2e} (should be ~0)")

    # ── 6. Memory overhead analysis ──────────────────────────────────
    print("\n--- Phase 6: Memory overhead analysis ---")
    # Count unique parameters across versions using structural sharing
    base_param_count = sum(v.size for v in v0_params.values())

    # Persistent storage: count only parameters that DIFFER between versions
    # For each version, count params that changed from v0
    def count_delta_params(version_params, base_params):
        delta = 0
        for k in version_params:
            if k in base_params:
                if not mx.array_equal(version_params[k], base_params[k]).item():
                    delta += version_params[k].size
            else:
                delta += version_params[k].size
        return delta

    v1_delta = count_delta_params(v1_params, v0_params)
    v2_delta = count_delta_params(v2_params, v0_params)

    # Persistent storage = base + deltas
    persistent_total = base_param_count + v1_delta + v2_delta
    # Full copy = 3 * base (v0, v1, v2 as full snapshots)
    full_copy_total = 3 * base_param_count
    # Overhead vs mutable (single tree)
    overhead_pct = (persistent_total - base_param_count) / base_param_count * 100

    results["memory"] = {
        "base_param_count": base_param_count,
        "v1_delta_params": v1_delta,
        "v2_delta_params": v2_delta,
        "v1_delta_pct": v1_delta / base_param_count * 100,
        "v2_delta_pct": v2_delta / base_param_count * 100,
        "persistent_total": persistent_total,
        "full_copy_total": full_copy_total,
        "overhead_vs_mutable_pct": overhead_pct,
        "savings_vs_full_copy_pct": (1 - persistent_total / full_copy_total) * 100,
    }

    print(f"  Base params:    {base_param_count:,}")
    print(f"  v1 delta:       {v1_delta:,} ({v1_delta/base_param_count*100:.1f}% of base)")
    print(f"  v2 delta:       {v2_delta:,} ({v2_delta/base_param_count*100:.1f}% of base)")
    print(f"  Persistent:     {persistent_total:,}")
    print(f"  Full-copy (3x): {full_copy_total:,}")
    print(f"  Overhead vs mutable: {overhead_pct:.1f}%")
    print(f"  Savings vs full-copy: {(1 - persistent_total / full_copy_total) * 100:.1f}%")

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  SUMMARY (seed={seed})")
    print(f"{'='*60}")

    joint_val_loss = results["joint"]["val_loss"]
    same_v_cal = results["same_version_calibrated"]["val_loss"]
    cross_v_cal = results["cross_version_calibrated"]["val_loss"]

    cross_vs_same_pct = (cross_v_cal - same_v_cal) / same_v_cal * 100
    cross_vs_joint_pct = (cross_v_cal - joint_val_loss) / joint_val_loss * 100
    same_vs_joint_pct = (same_v_cal - joint_val_loss) / joint_val_loss * 100

    print(f"  Joint training:            {joint_val_loss:.4f}")
    print(f"  Same-version (calibrated): {same_v_cal:.4f} ({same_vs_joint_pct:+.2f}% vs joint)")
    print(f"  Cross-version (calibrated):{cross_v_cal:.4f} ({cross_vs_joint_pct:+.2f}% vs joint)")
    print(f"  Cross vs Same:             {cross_vs_same_pct:+.2f}%")
    print(f"  Memory overhead:           {overhead_pct:.1f}%")
    print(f"  Rollback fidelity:         {rollback_diff:.2e}")
    print()

    kc1_pass = abs(cross_vs_same_pct) <= 5.0
    kc2_pass = overhead_pct <= 15.0
    print(f"  KC1 (cross-version <=5% worse): {'PASS' if kc1_pass else 'FAIL'} ({cross_vs_same_pct:+.2f}%)")
    print(f"  KC2 (overhead <=15%):           {'PASS' if kc2_pass else 'FAIL'} ({overhead_pct:.1f}%)")

    results["summary"] = {
        "joint_val_loss": joint_val_loss,
        "same_version_val_loss": same_v_cal,
        "cross_version_val_loss": cross_v_cal,
        "cross_vs_same_pct": cross_vs_same_pct,
        "cross_vs_joint_pct": cross_vs_joint_pct,
        "same_vs_joint_pct": same_vs_joint_pct,
        "memory_overhead_pct": overhead_pct,
        "rollback_max_diff": rollback_diff,
        "kc1_pass": kc1_pass,
        "kc2_pass": kc2_pass,
    }

    return results


def main():
    all_results = []
    seeds = [42, 123, 7]

    for seed in seeds:
        result = run_experiment(seed=seed)
        all_results.append(result)

    # ── Aggregate across seeds ──────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  AGGREGATE RESULTS ({len(seeds)} seeds)")
    print(f"{'='*60}")

    joint_losses = [r["summary"]["joint_val_loss"] for r in all_results]
    same_losses = [r["summary"]["same_version_val_loss"] for r in all_results]
    cross_losses = [r["summary"]["cross_version_val_loss"] for r in all_results]
    cross_vs_same = [r["summary"]["cross_vs_same_pct"] for r in all_results]
    overheads = [r["summary"]["memory_overhead_pct"] for r in all_results]
    rollbacks = [r["summary"]["rollback_max_diff"] for r in all_results]

    def mean(xs): return sum(xs) / len(xs)

    print(f"  Joint:         {mean(joint_losses):.4f} (per seed: {[f'{x:.4f}' for x in joint_losses]})")
    print(f"  Same-version:  {mean(same_losses):.4f} (per seed: {[f'{x:.4f}' for x in same_losses]})")
    print(f"  Cross-version: {mean(cross_losses):.4f} (per seed: {[f'{x:.4f}' for x in cross_losses]})")
    print(f"  Cross vs Same: {mean(cross_vs_same):+.2f}% (per seed: {[f'{x:+.2f}%' for x in cross_vs_same]})")
    print(f"  Mem overhead:  {mean(overheads):.1f}%")
    print(f"  Rollback diff: {mean(rollbacks):.2e}")

    mean_cross_vs_same = mean(cross_vs_same)
    mean_overhead = mean(overheads)

    kc1_overall = abs(mean_cross_vs_same) <= 5.0
    kc2_overall = mean_overhead <= 15.0

    print()
    print(f"  KC1 (cross-version <=5% worse): {'PASS' if kc1_overall else 'FAIL'} (mean {mean_cross_vs_same:+.2f}%)")
    print(f"  KC2 (overhead <=15%):           {'PASS' if kc2_overall else 'FAIL'} (mean {mean_overhead:.1f}%)")

    # Save results
    output = {
        "experiment": "persistent_expert_tree",
        "seeds": seeds,
        "per_seed": [r["summary"] for r in all_results],
        "aggregate": {
            "mean_joint_val_loss": mean(joint_losses),
            "mean_same_version_val_loss": mean(same_losses),
            "mean_cross_version_val_loss": mean(cross_losses),
            "mean_cross_vs_same_pct": mean_cross_vs_same,
            "mean_memory_overhead_pct": mean_overhead,
            "mean_rollback_diff": mean(rollbacks),
            "kc1_pass": kc1_overall,
            "kc2_pass": kc2_overall,
        },
        "memory_report_seed42": all_results[0].get("memory", {}),
    }

    out_path = "/Users/tom/Code/tomsiwik/llm/micro/models/persistent_expert_tree/results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
