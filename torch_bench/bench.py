"""7B-scale Continual Learning benchmark: LoRA lifecycle vs baselines.

CLI entry point that loads Qwen2.5-Coder-7B with QLoRA, trains sequentially
on 5 coding language domains, and compares forgetting across methods.

Usage:
    # Single config
    python torch_bench/bench.py --config lora_seqft --steps 50 --domains python,javascript

    # Full comparison
    python torch_bench/bench.py --compare --steps 500 --seed 42

    # Quick smoke test (CPU, 1 domain)
    python torch_bench/bench.py --config lora_seqft --steps 10 --domains python --no-quantize
"""

import argparse
import json
import time
import math
import torch
import numpy as np

from torch_bench.data import DOMAINS, load_domain
from torch_bench.train import (
    train_domain, train_domain_with_replay, evaluate,
    evaluate_all_domains, compute_forgetting,
)
from torch_bench.baselines import EWCState, ReplayBuffer, OLoRAState
from torch_bench.lora_lifecycle import LoRALifecycleWrapper

# ── Config ──────────────────────────────────────────────────

MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"
ALL_CONFIGS = ["lora_seqft", "lora_ewc", "lora_replay", "lora_olora", "lora_lifecycle"]


# ── Checkpointing ────────────────────────────────────────────

def save_adapter_checkpoint(model, config_name, domain, task_idx, checkpoint_dir,
                            lifecycle_wrapper=None, ewc_state=None, olora_state=None):
    """Save PEFT adapter + any extra state after training on a domain.

    Creates: checkpoint_dir/config_name/domain_N_name/
      - adapter_model/  (PEFT adapter via save_pretrained)
      - lifecycle_snapshots.pt  (if lifecycle)
      - ewc_state.pt  (if EWC)
      - olora_bases.pt  (if O-LoRA)
    """
    import os
    save_dir = os.path.join(checkpoint_dir, config_name, f"domain_{task_idx}_{domain}")
    os.makedirs(save_dir, exist_ok=True)

    # Save PEFT adapter
    adapter_dir = os.path.join(save_dir, "adapter")
    model.save_pretrained(adapter_dir)
    print(f"    CHECKPOINT: adapter saved to {adapter_dir}")

    # Save lifecycle snapshots
    if lifecycle_wrapper is not None and lifecycle_wrapper.snapshots:
        snap_path = os.path.join(save_dir, "lifecycle_snapshots.pt")
        lifecycle_wrapper.save_state(snap_path)
        print(f"    CHECKPOINT: {len(lifecycle_wrapper.snapshots)} snapshots saved")

    # Save EWC Fisher + star params
    if ewc_state is not None and ewc_state.fisher:
        ewc_path = os.path.join(save_dir, "ewc_state.pt")
        torch.save({
            "fisher": {k: v.cpu() for k, v in ewc_state.fisher.items()},
            "star_params": {k: v.cpu() for k, v in ewc_state.star_params.items()},
            "lambda_ewc": ewc_state.lambda_ewc,
        }, ewc_path)
        print(f"    CHECKPOINT: EWC state saved ({len(ewc_state.fisher)} param groups)")

    # Save O-LoRA bases
    if olora_state is not None and olora_state.bases:
        olora_path = os.path.join(save_dir, "olora_bases.pt")
        torch.save({
            "bases": {k: v.cpu() for k, v in olora_state.bases.items()},
        }, olora_path)
        max_rank = max(v.shape[1] for v in olora_state.bases.values())
        print(f"    CHECKPOINT: O-LoRA bases saved (max rank={max_rank})")


def parse_args():
    p = argparse.ArgumentParser(description="7B CL benchmark: LoRA lifecycle")
    p.add_argument("--config", type=str, default=None,
                   help=f"Single config: {ALL_CONFIGS}")
    p.add_argument("--configs", type=str, default=None,
                   help="Comma-separated configs to compare")
    p.add_argument("--compare", action="store_true",
                   help="Run all configs")
    p.add_argument("--steps", type=int, default=500,
                   help="Training steps per domain")
    p.add_argument("--lr", type=float, default=2e-4,
                   help="Learning rate")
    p.add_argument("--batch-size", type=int, default=4,
                   help="Batch size")
    p.add_argument("--seq-len", type=int, default=512,
                   help="Sequence length")
    p.add_argument("--domains", type=str, default=None,
                   help="Comma-separated domains (default: all 5)")
    p.add_argument("--n-train", type=int, default=2000,
                   help="Training sequences per domain")
    p.add_argument("--n-eval", type=int, default=200,
                   help="Eval sequences per domain")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed")
    p.add_argument("--lora-rank", type=int, default=16,
                   help="LoRA rank")
    p.add_argument("--lora-targets", type=str, default="q_proj,v_proj",
                   help="LoRA target modules")
    p.add_argument("--ewc-lambda", type=float, default=100.0,
                   help="EWC regularization strength")
    p.add_argument("--replay-per-domain", type=int, default=100,
                   help="Sequences to store per domain for replay")
    p.add_argument("--gate-init", type=float, default=-2.0,
                   help="FoX gate init bias (lifecycle)")
    p.add_argument("--no-quantize", action="store_true",
                   help="Skip quantization (for CPU testing)")
    p.add_argument("--output", type=str, default="results.json",
                   help="Output JSON file")
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                   help="Directory for saving adapter checkpoints")
    return p.parse_args()


# ── Model Loading ──────────────────────────────────────────

def load_model(args, device="cuda"):
    """Load base model with QLoRA via PEFT.

    Returns:
        (peft_model, tokenizer)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    print(f"  Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.no_quantize:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )
    else:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)

    # Apply LoRA
    targets = [t.strip() for t in args.lora_targets.split(",")]
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank,  # scaling = alpha/r = 1.0
        target_modules=targets,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  LoRA: rank={args.lora_rank}, targets={targets}")
    print(f"  Trainable: {n_trainable:,} / {n_total:,} "
          f"({100*n_trainable/n_total:.2f}%)")

    return model, tokenizer


# ── Config Runners ─────────────────────────────────────────

def run_lora_seqft(args, model, domains, domain_data, eval_datasets,
                   baseline_ppl, device):
    """Sequential fine-tuning baseline (catastrophic forgetting)."""
    print(f"\n{'='*72}")
    print(f"  CONFIG: lora_seqft (sequential fine-tuning)")
    print(f"{'='*72}")

    ppl_matrix = {}
    for task_idx, domain in enumerate(domains):
        print(f"\n  -- Task {task_idx}: Train on '{domain}' --")
        train_ds = domain_data[domain]["train"]

        train_domain(model, train_ds, steps=args.steps, lr=args.lr,
                     batch_size=args.batch_size, device=device)

        save_adapter_checkpoint(model, "lora_seqft", domain, task_idx,
                                args.checkpoint_dir)

        print(f"    Evaluating...")
        ppl_matrix[task_idx] = evaluate_all_domains(
            model, eval_datasets, baseline_ppl, device=device
        )

    return {"name": "lora_seqft", "ppl_matrix": ppl_matrix}


def run_lora_ewc(args, model, domains, domain_data, eval_datasets,
                 baseline_ppl, device):
    """LoRA + Elastic Weight Consolidation."""
    print(f"\n{'='*72}")
    print(f"  CONFIG: lora_ewc (lambda={args.ewc_lambda})")
    print(f"{'='*72}")

    ewc = EWCState(lambda_ewc=args.ewc_lambda)
    ppl_matrix = {}

    for task_idx, domain in enumerate(domains):
        print(f"\n  -- Task {task_idx}: Train on '{domain}' --")
        train_ds = domain_data[domain]["train"]

        extra_loss = ewc.penalty if ewc.fisher else None

        train_domain(model, train_ds, steps=args.steps, lr=args.lr,
                     batch_size=args.batch_size, device=device,
                     extra_loss_fn=extra_loss)

        # Compute Fisher after training (before next domain)
        if task_idx < len(domains) - 1:
            from torch.utils.data import DataLoader
            dl = DataLoader(train_ds, batch_size=args.batch_size,
                            shuffle=True, num_workers=0)
            ewc.compute_fisher(model, dl, n_samples=50, device=device)

        save_adapter_checkpoint(model, "lora_ewc", domain, task_idx,
                                args.checkpoint_dir, ewc_state=ewc)

        print(f"    Evaluating...")
        ppl_matrix[task_idx] = evaluate_all_domains(
            model, eval_datasets, baseline_ppl, device=device
        )

    return {"name": "lora_ewc", "ppl_matrix": ppl_matrix}


def run_lora_replay(args, model, domains, domain_data, eval_datasets,
                    baseline_ppl, device):
    """LoRA + Experience Replay (50/50 current + buffer)."""
    print(f"\n{'='*72}")
    print(f"  CONFIG: lora_replay (buffer={args.replay_per_domain}/domain)")
    print(f"{'='*72}")

    replay = ReplayBuffer()
    ppl_matrix = {}

    for task_idx, domain in enumerate(domains):
        print(f"\n  -- Task {task_idx}: Train on '{domain}' --")
        train_ds = domain_data[domain]["train"]

        train_domain_with_replay(model, train_ds, replay, steps=args.steps,
                                 lr=args.lr, batch_size=args.batch_size,
                                 device=device)

        # Store sequences for future replay
        if task_idx < len(domains) - 1:
            replay.store_domain(train_ds, n=args.replay_per_domain)

        save_adapter_checkpoint(model, "lora_replay", domain, task_idx,
                                args.checkpoint_dir)

        print(f"    Evaluating...")
        ppl_matrix[task_idx] = evaluate_all_domains(
            model, eval_datasets, baseline_ppl, device=device
        )

    return {"name": "lora_replay", "ppl_matrix": ppl_matrix}


def run_lora_olora(args, model, domains, domain_data, eval_datasets,
                   baseline_ppl, device):
    """LoRA + Orthogonal LoRA (gradient projection)."""
    print(f"\n{'='*72}")
    print(f"  CONFIG: lora_olora")
    print(f"{'='*72}")

    olora = OLoRAState()
    ppl_matrix = {}

    for task_idx, domain in enumerate(domains):
        print(f"\n  -- Task {task_idx}: Train on '{domain}' --")
        train_ds = domain_data[domain]["train"]

        post_backward = olora.project_gradients if olora.bases else None

        train_domain(model, train_ds, steps=args.steps, lr=args.lr,
                     batch_size=args.batch_size, device=device,
                     post_backward_fn=post_backward)

        # Accumulate basis after training
        if task_idx < len(domains) - 1:
            olora.accumulate_basis(model)

        save_adapter_checkpoint(model, "lora_olora", domain, task_idx,
                                args.checkpoint_dir, olora_state=olora)

        print(f"    Evaluating...")
        ppl_matrix[task_idx] = evaluate_all_domains(
            model, eval_datasets, baseline_ppl, device=device
        )

    return {"name": "lora_olora", "ppl_matrix": ppl_matrix}


def run_lora_lifecycle(args, model, domains, domain_data, eval_datasets,
                       baseline_ppl, device):
    """LoRA + Lifecycle (snapshot + FoX gate blending)."""
    print(f"\n{'='*72}")
    print(f"  CONFIG: lora_lifecycle (gate_init={args.gate_init})")
    print(f"{'='*72}")

    wrapper = LoRALifecycleWrapper(model, gate_init=args.gate_init)
    ppl_matrix = {}

    for task_idx, domain in enumerate(domains):
        print(f"\n  -- Task {task_idx}: Train on '{domain}' --")
        train_ds = domain_data[domain]["train"]

        # Include gate params in optimizer
        extra_params = wrapper.extra_parameters() if wrapper.snapshots else None

        train_domain(model, train_ds, steps=args.steps, lr=args.lr,
                     batch_size=args.batch_size, device=device,
                     extra_params=extra_params)

        # Snapshot and reinit after training (before next domain)
        if task_idx < len(domains) - 1:
            wrapper.snapshot_and_reinit(domain)
            wrapper.gate_summary()

        save_adapter_checkpoint(model, "lora_lifecycle", domain, task_idx,
                                args.checkpoint_dir, lifecycle_wrapper=wrapper)

        print(f"    Evaluating...")
        ppl_matrix[task_idx] = evaluate_all_domains(
            model, eval_datasets, baseline_ppl, device=device
        )

    # Final gate summary
    if wrapper.snapshots:
        print(f"\n    Final gate state:")
        wrapper.gate_summary()
        print(f"    Snapshot params: {wrapper.n_snapshot_params:,}")

    return {"name": "lora_lifecycle", "ppl_matrix": ppl_matrix}


# ── Results Printing ───────────────────────────────────────

def print_results(results_list, domains, baseline_ppl):
    """Print comparison table across all configs."""
    print(f"\n{'='*72}")
    print(f"  COMPARISON SUMMARY")
    print(f"{'='*72}")

    # Perplexity matrix per config
    for res in results_list:
        name = res["name"]
        ppl = res["ppl_matrix"]

        print(f"\n  [{name}] Perplexity Matrix:")
        header = f"  {'Task':>12s}"
        for d in domains:
            header += f"  {d:>12s}"
        print(header)
        print("  " + "-" * (14 + 14 * len(domains)))

        print(f"  {'baseline':>12s}", end="")
        for d in domains:
            print(f"  {baseline_ppl[d]:12.1f}", end="")
        print()

        for task_idx in range(len(domains)):
            label = f"T{task_idx}({domains[task_idx][:4]})"
            print(f"  {label:>12s}", end="")
            for d in domains:
                print(f"  {ppl[task_idx][d]:12.1f}", end="")
            print()

    # Forgetting comparison
    print(f"\n  Forgetting Comparison:")
    header = f"  {'':>20s}"
    for res in results_list:
        header += f"  {res['name']:>14s}"
    print(header)
    print("  " + "-" * (22 + 16 * len(results_list)))

    all_forgetting = []
    for res in results_list:
        fgt = compute_forgetting(res["ppl_matrix"], domains)
        all_forgetting.append(fgt)

    if all_forgetting:
        all_keys = list(all_forgetting[0].keys())
        for key in all_keys:
            print(f"  {key:>20s}", end="")
            for fgt in all_forgetting:
                print(f"  {fgt[key]['percent']:+13.1f}%", end="")
            print()

        # Mean forgetting
        print(f"  {'MEAN':>20s}", end="")
        for fgt in all_forgetting:
            fgts = [v["percent"] for v in fgt.values()]
            mean_fgt = np.mean(fgts) if fgts else 0.0
            print(f"  {mean_fgt:+13.1f}%", end="")
        print()

    # Final perplexity
    print(f"\n  Final Perplexity (after all {len(domains)} domains):")
    header = f"  {'Domain':>12s}  {'baseline':>8s}"
    for res in results_list:
        header += f"  {res['name']:>14s}"
    print(header)
    print("  " + "-" * (24 + 16 * len(results_list)))

    for d in domains:
        print(f"  {d:>12s}  {baseline_ppl[d]:8.1f}", end="")
        for res in results_list:
            final_ppl = res["ppl_matrix"][len(domains) - 1][d]
            print(f"  {final_ppl:14.1f}", end="")
        print()

    # Efficiency
    print(f"\n  Timing:")
    for res in results_list:
        elapsed = res.get("elapsed_s", 0)
        total_steps = len(domains) * res.get("steps_per_domain", 0)
        parts = [f"    {res['name']:>20s}:"]
        if elapsed:
            parts.append(f"{elapsed:>6.0f}s total")
        if total_steps and elapsed:
            parts.append(f"{elapsed / total_steps:.2f} s/step")
        print("  ".join(parts))


# ── Main ───────────────────────────────────────────────────

CONFIG_RUNNERS = {
    "lora_seqft": run_lora_seqft,
    "lora_ewc": run_lora_ewc,
    "lora_replay": run_lora_replay,
    "lora_olora": run_lora_olora,
    "lora_lifecycle": run_lora_lifecycle,
}


def _save_results(all_results, domains, baseline_ppl, args):
    """Save results to JSON incrementally (resume-safe)."""
    serializable = []
    for res in all_results:
        s = {
            "name": res["name"],
            "elapsed_s": res.get("elapsed_s", 0),
            "steps_per_domain": res.get("steps_per_domain", 0),
            "ppl_matrix": {str(k): v for k, v in res["ppl_matrix"].items()},
            "forgetting": compute_forgetting(res["ppl_matrix"], domains),
        }
        serializable.append(s)

    output = {
        "model": MODEL_NAME,
        "domains": domains,
        "baseline_ppl": baseline_ppl,
        "args": vars(args),
        "results": serializable,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)


def main():
    args = parse_args()

    # Seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Device
    device = "cuda" if torch.cuda.is_available() and not args.no_quantize else "cpu"
    print(f"  Device: {device}")
    print(f"  Seed: {args.seed}")

    # Domains
    domains = [d.strip() for d in args.domains.split(",")] if args.domains else DOMAINS
    print(f"  Domains: {domains}")

    t0 = time.time()

    # Load tokenizer (lightweight, for data loading)
    from transformers import AutoTokenizer
    print(f"  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load domain data (shared across configs)
    print(f"  Loading domain data...")
    domain_data = {}
    eval_datasets = {}
    for d in domains:
        print(f"    {d}...", end=" ", flush=True)
        train_ds, eval_ds = load_domain(
            d, tokenizer, n_train=args.n_train, n_eval=args.n_eval,
            seq_len=args.seq_len, seed=args.seed,
        )
        domain_data[d] = {"train": train_ds, "eval": eval_ds}
        eval_datasets[d] = eval_ds
        print(f"{len(train_ds)} train, {len(eval_ds)} eval")

    # Determine configs to run
    if args.compare:
        configs = ALL_CONFIGS
    elif args.configs:
        configs = [c.strip() for c in args.configs.split(",")]
        for c in configs:
            if c not in CONFIG_RUNNERS:
                print(f"  ERROR: Unknown config '{c}'. Choose from: {list(CONFIG_RUNNERS.keys())}")
                return
    elif args.config:
        if args.config not in CONFIG_RUNNERS:
            print(f"  ERROR: Unknown config '{args.config}'. "
                  f"Choose from: {list(CONFIG_RUNNERS.keys())}")
            return
        configs = [args.config]
    else:
        configs = ["lora_seqft"]

    # Resume: load previously completed configs from output file
    import os
    all_results = []
    completed_configs = set()
    baseline_ppl = None

    if os.path.exists(args.output):
        try:
            with open(args.output) as f:
                prev = json.load(f)
            baseline_ppl = prev.get("baseline_ppl")
            for res in prev.get("results", []):
                # Reconstruct ppl_matrix with int keys
                ppl_matrix = {int(k): v for k, v in res["ppl_matrix"].items()}
                all_results.append({
                    "name": res["name"],
                    "elapsed_s": res.get("elapsed_s", 0),
                    "steps_per_domain": res.get("steps_per_domain", 0),
                    "ppl_matrix": ppl_matrix,
                })
                completed_configs.add(res["name"])
            if completed_configs:
                print(f"  Resuming: {len(completed_configs)} configs already done: "
                      f"{completed_configs}")
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  Warning: could not parse {args.output} for resume: {e}")

    remaining = [c for c in configs if c not in completed_configs]
    if not remaining:
        print(f"  All configs already completed. Delete {args.output} to re-run.")
    else:
        print(f"  Configs to run: {remaining}")

    # Run each config (fresh model per config)
    for config_name in remaining:
        # Fresh model for each config
        model, _ = load_model(args, device=device)

        # Baseline perplexity (only compute once)
        if baseline_ppl is None:
            print(f"\n  Baseline perplexity (QLoRA model, before training)...")
            baseline_ppl = {}
            for d in domains:
                ppl = evaluate(model, eval_datasets[d], device=device)
                baseline_ppl[d] = ppl
                print(f"    {d:12s}: ppl={ppl:.1f}")

        runner = CONFIG_RUNNERS[config_name]
        t_config = time.time()
        result = runner(args, model, domains, domain_data, eval_datasets,
                        baseline_ppl, device)
        result["elapsed_s"] = time.time() - t_config
        result["steps_per_domain"] = args.steps
        all_results.append(result)

        # Free GPU memory between configs
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Save incrementally after each config (resume-safe)
        _save_results(all_results, domains, baseline_ppl, args)
        print(f"  Checkpoint saved ({config_name} done)")

    # Print comparison
    print_results(all_results, domains, baseline_ppl)
    print(f"\n  Results saved to {args.output}")

    elapsed = time.time() - t0
    print(f"  Total time: {elapsed:.1f}s")
    print("=" * 72)


if __name__ == "__main__":
    main()
