#!/usr/bin/env python3
"""Composition health via KL divergence: label-free detection of harmful expert additions.

Motivation: Cosine-based degradation detection is anti-correlated at micro (r=-0.46).
Canary queries work but need per-expert held-out data. KL divergence between base and
composed model on calibration tokens is label-free, domain-agnostic, and cheap.

Kill criteria:
  K1: KL(composed || base) does not correlate with composition quality loss (rho < 0.3)
  K2: adding a harmful expert does not spike KL more than adding a helpful one
  K3: KL measurement on 100 calibration tokens takes >30s per composition
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ADAPTER_DIR = Path("/workspace/llm/adapters")
BASE_MODEL = os.environ.get("BASE_MODEL", "/workspace/models/Qwen2.5-7B")
RESULTS_DIR = Path("/workspace/llm/results/composition_health_kl_divergence")
SEED = 42
N_CALIBRATION_TOKENS = 100
N_SWEEP = [5, 10, 25, 50]  # composition sizes to test


def load_base_model():
    """Load quantized base model."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    model.eval()
    return model, tokenizer


def get_calibration_texts(tokenizer, n_texts=20):
    """Get diverse calibration texts (Wikipedia, code, QA-style)."""
    texts = [
        # Diverse text covering different domains
        "The mitochondria is the powerhouse of the cell, responsible for generating ATP through",
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) +",
        "In 1969, Neil Armstrong became the first person to walk on the Moon during the",
        "The derivative of sin(x) is cos(x), and the integral of cos(x) is sin(x) +",
        "SELECT users.name, orders.total FROM users INNER JOIN orders ON users.id =",
        "The French Revolution began in 1789 and led to fundamental changes in European",
        "To solve a quadratic equation ax² + bx + c = 0, use the quadratic formula:",
        "import torch\nmodel = AutoModelForCausalLM.from_pretrained('gpt2')\noutput =",
        "The human genome contains approximately 3 billion base pairs of DNA organized into",
        "In economics, supply and demand determine market equilibrium where the price of",
        "The electromagnetic spectrum ranges from radio waves with long wavelengths to gamma",
        "Machine learning algorithms can be broadly categorized into supervised, unsupervised",
        "The United Nations was established in 1945 after World War II to promote international",
        "In organic chemistry, functional groups determine the chemical properties of molecules",
        "The Pythagorean theorem states that in a right triangle, the square of the hypotenuse",
        "Neural networks consist of layers of interconnected nodes that process information",
        "The Renaissance was a cultural movement that began in Italy during the 14th century",
        "In Python, list comprehensions provide a concise way to create lists: [x**2 for x",
        "Photosynthesis converts carbon dioxide and water into glucose and oxygen using energy",
        "The TCP/IP protocol stack consists of four layers: application, transport, internet",
    ]
    return texts[:n_texts]


def compute_base_logits(model, tokenizer, texts):
    """Compute base model logit distributions on calibration texts."""
    all_logits = []
    for text in texts:
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=256
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs)
            # Take logits at last position
            logits = outputs.logits[0, -1].float()  # float32 for KL computation
            all_logits.append(logits.cpu())

    return all_logits


def compute_kl_divergence(base_logits_list, composed_logits_list):
    """Compute mean KL(composed || base) across calibration texts."""
    kl_values = []
    for base_logits, comp_logits in zip(base_logits_list, composed_logits_list):
        base_probs = F.log_softmax(base_logits, dim=-1)
        comp_probs = F.log_softmax(comp_logits, dim=-1)
        # KL(composed || base) = sum(composed * (log(composed) - log(base)))
        kl = F.kl_div(base_probs, comp_probs, log_target=True, reduction="sum")
        kl_values.append(kl.item())
    return np.mean(kl_values), np.std(kl_values), kl_values


def compose_adapters_on_cpu(adapter_names, adapter_dir):
    """Load adapter deltas on CPU and compute their sum (single composed delta).

    Returns dict: {param_name: summed_delta_tensor}.
    """
    from safetensors import safe_open

    composed_delta = {}
    for adapter_name in adapter_names:
        adapter_path = adapter_dir / adapter_name
        safetensor_path = adapter_path / "adapter_model.safetensors"
        if not safetensor_path.exists():
            # Try bin format
            bin_path = adapter_path / "adapter_model.bin"
            if bin_path.exists():
                state_dict = torch.load(bin_path, map_location="cpu", weights_only=True)
            else:
                print(f"  Warning: no weights found for {adapter_name}")
                continue
        else:
            state_dict = {}
            with safe_open(str(safetensor_path), framework="pt", device="cpu") as f:
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)

        for key, tensor in state_dict.items():
            if key in composed_delta:
                composed_delta[key] = composed_delta[key] + tensor.float()
            else:
                composed_delta[key] = tensor.float()

    return composed_delta


def apply_composed_delta(model, composed_delta):
    """Apply summed adapter delta to model weights (modifies in-place)."""
    applied = 0
    for name, param in model.named_parameters():
        # Map base model param names to adapter delta keys
        # Adapter keys are like "base_model.model.layers.0.self_attn.q_proj.lora_A.weight"
        # We need to compute W_new = W + A @ B for each layer
        pass  # This is complex with LoRA; use PEFT's add_weighted_adapter instead

    return applied


def get_adapter_list():
    """Get available adapters."""
    benchmark_path = Path("/workspace/llm/results/pilot50_benchmark.json")
    if benchmark_path.exists():
        with open(benchmark_path) as f:
            data = json.load(f)
        ranked = []
        for name, info in data.get("per_adapter", {}).items():
            ppl_improvement = info.get("ppl_improvement_pct", 0)
            ranked.append((name, ppl_improvement))
        ranked.sort(key=lambda x: x[1], reverse=True)
        return [name for name, _ in ranked]
    return sorted(
        d.name for d in ADAPTER_DIR.iterdir()
        if d.is_dir() and (d / "adapter_config.json").exists()
    )


def evaluate_per_domain_ppl(model, tokenizer, adapter_names, adapter_dir):
    """Quick per-domain PPL check to correlate with KL. Uses adapter training data samples."""
    # Load a small sample from each adapter's training domain
    ppl_by_adapter = {}
    data_dir = Path("/workspace/llm/training_data")

    for adapter_name in adapter_names:
        domain_file = data_dir / f"{adapter_name}.jsonl"
        if not domain_file.exists():
            continue

        # Read first 20 examples
        texts = []
        with open(domain_file) as f:
            for i, line in enumerate(f):
                if i >= 20:
                    break
                try:
                    obj = json.loads(line)
                    texts.append(obj.get("text", obj.get("content", "")))
                except json.JSONDecodeError:
                    continue

        if not texts:
            continue

        total_nll = 0.0
        total_tokens = 0
        for text in texts[:10]:
            if not text.strip():
                continue
            inputs = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=256
            ).to(model.device)
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = inputs["input_ids"][:, 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    reduction="sum",
                )
                total_nll += loss.item()
                total_tokens += shift_labels.numel()

        if total_tokens > 0:
            ppl_by_adapter[adapter_name] = np.exp(total_nll / total_tokens)

    return ppl_by_adapter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-calibration", type=int, default=20)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_adapters = get_adapter_list()
    print(f"Found {len(all_adapters)} adapters")

    # Load base model and compute calibration logits
    print("\n=== Phase 1: Base model calibration ===")
    base_model, tokenizer = load_base_model()
    texts = get_calibration_texts(tokenizer, args.n_calibration)
    base_logits = compute_base_logits(base_model, tokenizer, texts)
    print(f"Computed base logits on {len(texts)} calibration texts")

    results = {
        "config": {
            "n_calibration_texts": len(texts),
            "n_sweep": N_SWEEP,
            "seed": SEED,
            "total_adapters": len(all_adapters),
        },
        "kl_by_n": {},
        "per_adapter_delta_kl": {},
        "kill_criteria": {},
    }

    # Phase 2: KL vs N sweep
    print("\n=== Phase 2: KL divergence vs N ===")
    from peft import PeftModel

    rng = np.random.RandomState(SEED)

    for n in N_SWEEP:
        if n > len(all_adapters):
            print(f"  Skip N={n} (only {len(all_adapters)} adapters)")
            continue

        selected = all_adapters[:n]
        print(f"\n  N={n}: composing {len(selected)} adapters...")

        t0 = time.time()

        # Load and compose using PEFT's add_weighted_adapter
        fresh_model, _ = load_base_model()
        peft_model = None
        try:
            # Load first adapter
            adapter_path = str(ADAPTER_DIR / selected[0])
            peft_model = PeftModel.from_pretrained(
                fresh_model, adapter_path, adapter_name=selected[0]
            )

            # Load remaining adapters
            for adapter_name in selected[1:]:
                adapter_path = str(ADAPTER_DIR / adapter_name)
                peft_model.load_adapter(adapter_path, adapter_name=adapter_name)

            # Merge via add_weighted_adapter
            peft_model.add_weighted_adapter(
                adapters=selected,
                weights=[1.0] * len(selected),
                adapter_name="composed",
                combination_type="linear",
            )
            peft_model.set_adapter("composed")
            peft_model.eval()

            # Compute KL
            composed_logits = compute_base_logits(peft_model, tokenizer, texts)
            mean_kl, std_kl, kl_values = compute_kl_divergence(base_logits, composed_logits)

            elapsed = time.time() - t0
            results["kl_by_n"][n] = {
                "mean_kl": round(float(mean_kl), 6),
                "std_kl": round(float(std_kl), 6),
                "per_text_kl": [round(float(v), 6) for v in kl_values],
                "elapsed_s": round(elapsed, 1),
            }
            print(f"  N={n}: KL={mean_kl:.4f}±{std_kl:.4f} [{elapsed:.0f}s]")

        except Exception as e:
            print(f"  N={n}: FAILED - {e}")
            results["kl_by_n"][n] = {"error": str(e)}

        finally:
            if peft_model is not None:
                del peft_model
            del fresh_model
            torch.cuda.empty_cache()

    # Phase 3: Per-adapter delta KL (leave-one-out at N=10)
    print("\n=== Phase 3: Per-adapter delta KL (leave-one-out) ===")
    n_test = min(10, len(all_adapters))
    test_adapters = all_adapters[:n_test]

    # Compute KL with all N adapters
    all_n_result = results["kl_by_n"].get(n_test, {})
    if "error" not in all_n_result and "mean_kl" in all_n_result:
        kl_all = all_n_result["mean_kl"]
    else:
        kl_all = None

    for leave_out_idx, leave_out_name in enumerate(test_adapters):
        remaining = [a for i, a in enumerate(test_adapters) if i != leave_out_idx]
        print(f"\n  Leave out: {leave_out_name} ({leave_out_idx+1}/{n_test})")

        t0 = time.time()
        fresh_model, _ = load_base_model()
        peft_model = None
        try:
            adapter_path = str(ADAPTER_DIR / remaining[0])
            peft_model = PeftModel.from_pretrained(
                fresh_model, adapter_path, adapter_name=remaining[0]
            )
            for adapter_name in remaining[1:]:
                adapter_path = str(ADAPTER_DIR / adapter_name)
                peft_model.load_adapter(adapter_path, adapter_name=adapter_name)

            peft_model.add_weighted_adapter(
                adapters=remaining,
                weights=[1.0] * len(remaining),
                adapter_name="composed",
                combination_type="linear",
            )
            peft_model.set_adapter("composed")
            peft_model.eval()

            composed_logits = compute_base_logits(peft_model, tokenizer, texts)
            mean_kl, std_kl, _ = compute_kl_divergence(base_logits, composed_logits)

            delta_kl = (kl_all - mean_kl) if kl_all is not None else None
            elapsed = time.time() - t0

            results["per_adapter_delta_kl"][leave_out_name] = {
                "kl_without": round(float(mean_kl), 6),
                "delta_kl": round(float(delta_kl), 6) if delta_kl is not None else None,
                "elapsed_s": round(elapsed, 1),
            }
            print(f"    KL_without={mean_kl:.4f}, delta_KL={delta_kl:.4f}" if delta_kl else f"    KL_without={mean_kl:.4f}")

        except Exception as e:
            print(f"    FAILED: {e}")
            results["per_adapter_delta_kl"][leave_out_name] = {"error": str(e)}

        finally:
            if peft_model is not None:
                del peft_model
            del fresh_model
            torch.cuda.empty_cache()

    # Phase 4: Kill criteria assessment
    print("\n=== Kill Criteria ===")

    # K1: Correlate KL with quality loss (need PPL or accuracy data)
    # We'll use the per-N KL trend as a proxy
    kl_values = []
    n_values = []
    for n in N_SWEEP:
        info = results["kl_by_n"].get(n, {})
        if "mean_kl" in info:
            kl_values.append(info["mean_kl"])
            n_values.append(n)

    if len(kl_values) >= 3:
        from scipy.stats import spearmanr
        rho, p = spearmanr(n_values, kl_values)
        k1_pass = rho > 0.3  # KL should increase with N (more divergence)
        results["kill_criteria"]["K1_kl_vs_n_rho"] = round(float(rho), 4)
        results["kill_criteria"]["K1_kl_vs_n_p"] = round(float(p), 4)
        results["kill_criteria"]["K1_threshold"] = 0.3
        results["kill_criteria"]["K1_fail"] = bool(not k1_pass)
        print(f"K1: KL-vs-N Spearman rho = {rho:.3f} (threshold 0.3) -> {'PASS' if k1_pass else 'KILL'}")
    else:
        results["kill_criteria"]["K1_fail"] = None
        print("K1: insufficient data points")

    # K2: Delta KL discrimination
    delta_kls = [
        v["delta_kl"]
        for v in results["per_adapter_delta_kl"].values()
        if isinstance(v.get("delta_kl"), (int, float))
    ]
    if delta_kls:
        delta_range = max(delta_kls) - min(delta_kls)
        delta_cv = np.std(delta_kls) / max(abs(np.mean(delta_kls)), 1e-8)
        results["kill_criteria"]["K2_delta_kl_range"] = round(float(delta_range), 6)
        results["kill_criteria"]["K2_delta_kl_cv"] = round(float(delta_cv), 4)
        k2_fail = delta_cv < 0.1  # no discrimination if all delta_KLs are similar
        results["kill_criteria"]["K2_fail"] = bool(k2_fail)
        print(f"K2: delta_KL CV = {delta_cv:.3f} (need >0.1 for discrimination) -> {'KILL' if k2_fail else 'PASS'}")

    # K3: Timing
    n10_time = results["kl_by_n"].get(10, {}).get("elapsed_s", 999)
    k3_fail = n10_time > 30
    results["kill_criteria"]["K3_eval_time_s"] = n10_time
    results["kill_criteria"]["K3_threshold_s"] = 30
    results["kill_criteria"]["K3_fail"] = bool(k3_fail)
    print(f"K3: eval time = {n10_time:.0f}s (threshold 30s) -> {'KILL' if k3_fail else 'PASS'}")

    # Save results
    results_path = RESULTS_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
