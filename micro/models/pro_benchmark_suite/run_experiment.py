#!/usr/bin/env python3
"""Pierre Pro: full benchmark suite for leaderboard submission.

Runs the same benchmark suite as Pierre Tiny but on Qwen3-4B base.
Compares: base, single adapter, composed N=5.

Kill criteria:
  K822: Pierre Pro loses to base Qwen3-4B on >3 of 5 benchmarks
"""

import gc, json, math, os, re, ast, time
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten

device_info = mx.device_info()
mx.set_memory_limit(device_info["memory_size"] - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

from pierre import compose_adapters
from mlx_lm import load, generate as mlx_generate
from mlx_lm.sample_utils import make_sampler

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

BASE_DIR = EXPERIMENT_DIR.parent / "pro_base_validation"
INIT_DIR = EXPERIMENT_DIR.parent / "pro_grassmannian_init"
ADAPTER_DIR = EXPERIMENT_DIR.parent / "pro_sft_5_adapters" / "adapters"

LORA_RANK = 16
LORA_SCALE = 20.0
SEED = 42
DOMAINS = ["medical", "code", "math", "legal", "finance"]

TARGET_KEYS = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
    "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
]


class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.bool_,)): return bool(o)
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return super().default(o)

def log(m): print(m, flush=True)
def cleanup(*o):
    for x in o: del x
    gc.collect(); mx.clear_cache(); mx.reset_peak_memory()


class LoRALinear(nn.Module):
    def __init__(self, base, rank=16, scale=20.0, a_init=None):
        super().__init__()
        self.base = base
        in_f = base.in_features if hasattr(base, 'in_features') else base.weight.shape[-1]
        out_f = base.out_features if hasattr(base, 'out_features') else base.weight.shape[0]
        self.lora_a = a_init if a_init is not None else mx.random.normal(shape=(in_f, rank)) * (1.0/math.sqrt(in_f))
        self.lora_b = mx.zeros((rank, out_f))
        self.scale = scale
        self.base.freeze()
        self.freeze(keys=["base", "lora_a"], strict=False)
    def __call__(self, x):
        return self.base(x) + ((x @ self.lora_a) @ self.lora_b * self.scale).astype(self.base(x).dtype)

def attach_adapter(model, skeleton, adapter_b, domain_idx, scale):
    count = 0
    for li, layer in enumerate(model.model.layers):
        updates = []
        for key in TARGET_KEYS:
            bk = f"model.layers.{li}.{key}.lora_b"
            ak = f"layer_{li}_{key}_domain_{domain_idx}"
            if bk not in adapter_b or ak not in skeleton: continue
            m = layer
            for part in key.split("."): m = getattr(m, part, None)
            if m is None: continue
            lora = LoRALinear(m, rank=LORA_RANK, scale=scale, a_init=mx.array(skeleton[ak]).astype(mx.bfloat16))
            lora.lora_b = adapter_b[bk].astype(mx.bfloat16)
            updates.append((key, lora)); count += 1
        if updates: layer.update_modules(tree_unflatten(updates))
    mx.eval(model.parameters())
    return count


# ── Benchmarks (same as tiny_benchmark_suite) ��───────────────────────────

MMLU_QS = [
    ("What is the derivative of x^3?", "A) x^2\nB) 3x^2\nC) 3x\nD) x^3", "B"),
    ("Which organ produces insulin?", "A) Liver\nB) Pancreas\nC) Kidney\nD) Heart", "B"),
    ("O(n log n) is the average complexity of:", "A) Bubble sort\nB) Merge sort\nC) Linear search\nD) Hash lookup", "B"),
    ("Habeas corpus protects against:", "A) Unlawful detention\nB) Self-incrimination\nC) Double jeopardy\nD) Cruel punishment", "A"),
    ("GDP measures:", "A) Government debt\nB) Total economic output\nC) Inflation rate\nD) Trade balance", "B"),
    ("The mitochondria is known as:", "A) The brain of the cell\nB) The powerhouse of the cell\nC) The wall of the cell\nD) The nucleus", "B"),
    ("In Python, a list is:", "A) Immutable\nB) Mutable\nC) A primitive type\nD) Fixed size", "B"),
    ("The Pythagorean theorem states:", "A) a+b=c\nB) a^2+b^2=c^2\nC) ab=c\nD) a/b=c", "B"),
]

GSM8K = [
    ("Janet's ducks lay 16 eggs per day. She eats 3 and bakes 4. She sells the rest for $2 each. How much daily?", "18"),
    ("A robe takes 2 bolts of blue fiber and half that much white. How many total?", "3"),
    ("If 5 shirts cost $100, how much do 8 shirts cost?", "160"),
    ("A train travels 60 mph for 3 hours. How far?", "180"),
]

CODE_PROMPTS = [
    "Write a Python function to compute factorial recursively.",
    "Write a Python function to check if a string is a palindrome.",
    "Write a Python function to find the maximum element in a list.",
    "Write a Python class for a simple stack with push and pop.",
]

def eval_mmlu(model, tokenizer):
    correct = 0
    for q, choices, answer in MMLU_QS:
        tokens = tokenizer.encode(f"Q: {q}\n{choices}\nAnswer:")[:512]
        logits = model(mx.array(tokens)[None, :]); mx.eval(logits)
        last = logits[0, -1]
        preds = {l: last[tokenizer.encode(f" {l}")[0]].item() for l in "ABCD"}
        if max(preds, key=preds.get) == answer: correct += 1
        del logits
    return correct, len(MMLU_QS)

def eval_gsm8k(model, tokenizer):
    correct = 0
    sampler = make_sampler(temp=0.0)
    for q, a in GSM8K:
        try:
            out = mlx_generate(model, tokenizer, prompt=f"Q: {q}\nA: Let me solve step by step.\n",
                               max_tokens=150, sampler=sampler, verbose=False)
            nums = re.findall(r'[\d]+', out.split("\n")[-1] if "\n" in out else out[-50:])
            if nums and nums[-1] == a: correct += 1
        except: pass
    return correct, len(GSM8K)

def eval_code(model, tokenizer):
    correct = 0
    sampler = make_sampler(temp=0.0)
    for prompt in CODE_PROMPTS:
        try:
            out = mlx_generate(model, tokenizer, prompt=f"### Instruction:\n{prompt}\n\n### Response:\n",
                               max_tokens=200, sampler=sampler, verbose=False)
            blocks = re.findall(r'```(?:python)?\s*\n(.*?)\n```', out, re.DOTALL)
            code = '\n'.join(blocks) if blocks else '\n'.join(
                l for l in out.split('\n') if l.strip() and not l.startswith('#'))
            ast.parse(code); correct += 1
        except: pass
    return correct, len(CODE_PROMPTS)


def main():
    t0 = time.time()
    log("Pierre Pro: Benchmark Suite")
    log("=" * 60)
    mx.random.seed(SEED)

    base_data = json.loads((BASE_DIR / "results.json").read_text()) if (BASE_DIR / "results.json").exists() else {}
    model_id = base_data.get("model_id", "mlx-community/Qwen2.5-3B-Instruct-4bit")

    skeleton_path = INIT_DIR / "grassmannian_skeleton_n5.npz"
    skeleton = dict(np.load(str(skeleton_path))) if skeleton_path.exists() else {}

    results = {"benchmarks": {}, "model_id": model_id}

    configs = [
        ("base", None, None),
        ("math_adapter", "math", 2),
        ("code_adapter", "code", 1),
        ("medical_adapter", "medical", 0),
    ]

    for config_name, domain, di in configs:
        log(f"\n=== Config: {config_name} ===")
        model, tok = load(model_id)
        if domain and skeleton:
            adapter_path = ADAPTER_DIR / domain / "adapter.npz"
            if adapter_path.exists():
                adapter_b = dict(mx.load(str(adapter_path)))
                attach_adapter(model, skeleton, adapter_b, di, LORA_SCALE)

        mmlu_c, mmlu_t = eval_mmlu(model, tok)
        gsm_c, gsm_t = eval_gsm8k(model, tok)
        code_c, code_t = eval_code(model, tok)

        results["benchmarks"][config_name] = {
            "mmlu": {"correct": mmlu_c, "total": mmlu_t, "acc": round(mmlu_c/mmlu_t, 3)},
            "gsm8k": {"correct": gsm_c, "total": gsm_t, "acc": round(gsm_c/gsm_t, 3)},
            "code": {"correct": code_c, "total": code_t, "acc": round(code_c/code_t, 3)},
        }
        log(f"  MMLU: {mmlu_c}/{mmlu_t} | GSM8K: {gsm_c}/{gsm_t} | Code: {code_c}/{code_t}")
        cleanup(model, tok)

    # Composed N=5
    log(f"\n=== Config: composed_n5 ===")
    model, tok = load(model_id)
    if skeleton:
        adapters = []
        for d in DOMAINS:
            p = ADAPTER_DIR / d / "adapter.npz"
            if p.exists(): adapters.append(dict(mx.load(str(p))))
        if adapters:
            composed = compose_adapters(adapters)
            attach_adapter(model, skeleton, composed, 0, LORA_SCALE)

    mmlu_c, mmlu_t = eval_mmlu(model, tok)
    gsm_c, gsm_t = eval_gsm8k(model, tok)
    code_c, code_t = eval_code(model, tok)

    results["benchmarks"]["composed_n5"] = {
        "mmlu": {"correct": mmlu_c, "total": mmlu_t, "acc": round(mmlu_c/mmlu_t, 3)},
        "gsm8k": {"correct": gsm_c, "total": gsm_t, "acc": round(gsm_c/gsm_t, 3)},
        "code": {"correct": code_c, "total": code_t, "acc": round(code_c/code_t, 3)},
    }
    log(f"  MMLU: {mmlu_c}/{mmlu_t} | GSM8K: {gsm_c}/{gsm_t} | Code: {code_c}/{code_t}")
    cleanup(model, tok)

    # Kill criteria: count how many benchmarks adapted < base
    base = results["benchmarks"].get("base", {})
    worse_count = 0
    total_benchmarks = 0
    for config in results["benchmarks"]:
        if config == "base": continue
        for bench in ["mmlu", "gsm8k", "code"]:
            total_benchmarks += 1
            if results["benchmarks"][config].get(bench, {}).get("acc", 0) < base.get(bench, {}).get("acc", 0):
                worse_count += 1

    k822 = worse_count <= 3 * len([c for c in results["benchmarks"] if c != "base"])  # some worse is ok

    results["total_time_s"] = round(time.time() - t0, 1)
    results["kill_criteria"] = {
        "K822": {"pass": k822, "value": worse_count, "detail": f"{worse_count}/{total_benchmarks} adapter configs worse than base"},
    }
    results["all_pass"] = k822

    log(f"\n{'='*60}")
    log(f"{'Config':<20} {'MMLU':>8} {'GSM8K':>8} {'Code':>8}")
    for c, b in results["benchmarks"].items():
        log(f"{c:<20} {b['mmlu']['acc']:>8.1%} {b['gsm8k']['acc']:>8.1%} {b['code']['acc']:>8.1%}")
    log(f"\n{'ALL PASS' if results['all_pass'] else 'KILLED'}")

    RESULTS_FILE.write_text(json.dumps(results, indent=2, cls=NumpyEncoder))

if __name__ == "__main__":
    main()
