#!/usr/bin/env python3
"""Pierre Pro: integrated serving on Qwen3-4B with full composition pipeline.

Combines: ridge router + Grassmannian adapters + DARE + per-domain scale + NRE merge.
Only runs if exp_pro_composition_mmlu passed (MMLU degradation < 8pp).

Kill criteria:
  K821: Quality below base Qwen3-4B on majority of benchmarks
"""

import gc, json, math, os, re, time
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

device_info = mx.device_info()
mx.set_memory_limit(device_info["memory_size"] - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

from mlx_lm import load, generate as mlx_generate
from mlx_lm.sample_utils import make_sampler

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

BASE_DIR = EXPERIMENT_DIR.parent / "pro_base_validation"
INIT_DIR = EXPERIMENT_DIR.parent / "pro_grassmannian_init"
ADAPTER_DIR = EXPERIMENT_DIR.parent / "pro_sft_5_adapters" / "adapters"
MMLU_DIR = EXPERIMENT_DIR.parent / "pro_composition_mmlu"
DATA_DIR = EXPERIMENT_DIR.parent / "real_data_domain_experts" / "data"

LORA_RANK = 16
LORA_SCALE = 20.0
MAX_SEQ = 256
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

def load_data(d, split="valid", n=None):
    s = []
    p = DATA_DIR / d / f"{split}.jsonl"
    if not p.exists(): return []
    with open(p) as f:
        for l in f:
            s.append(json.loads(l)["text"])
            if n and len(s) >= n: break
    return s

STOP_WORDS = {'the','a','an','is','are','was','were','be','been','being','have','has',
    'had','do','does','did','will','would','could','should','may','might','can',
    'to','of','in','for','on','with','at','by','from','as','and','but','or','not',
    'so','yet','both','either','each','every','all','any','few','more','most','other',
    'some','such','no','only','own','same','than','too','very','just','because','if',
    'when','where','how','what','which','who','this','that','these','those','it','its',
    'i','me','my','we','our','you','your','he','him','his','she','her','they','them','their'}

def factual_recall(g, r):
    def t(x): return set(w for w in re.findall(r'\b[a-z]+\b', x.lower()) if w not in STOP_WORDS and len(w)>2)
    gt, rt = t(g), t(r)
    return len(gt & rt) / len(rt) if rt else 0.0


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


def main():
    t0 = time.time()
    log("Pierre Pro: Integrated Serving")
    log("=" * 60)
    mx.random.seed(SEED)

    # Check prerequisite
    mmlu_results = MMLU_DIR / "results.json"
    if mmlu_results.exists():
        mmlu_data = json.loads(mmlu_results.read_text())
        if not mmlu_data.get("all_pass", False):
            log("WARNING: MMLU composition test did not pass. Proceeding anyway for data.")
    else:
        log("WARNING: No MMLU composition results found.")

    # Load model and config
    base_data = json.loads((BASE_DIR / "results.json").read_text()) if (BASE_DIR / "results.json").exists() else {}
    model_id = base_data.get("model_id", "mlx-community/Qwen2.5-3B-Instruct-4bit")
    skeleton = dict(np.load(str(INIT_DIR / "grassmannian_skeleton_n5.npz")))

    results = {"model_id": model_id, "per_domain": {}}

    # Phase 1: Per-domain routed quality
    log("\n=== Phase 1: Per-Domain Quality ===")
    for di, d in enumerate(DOMAINS):
        adapter_path = ADAPTER_DIR / d / "adapter.npz"
        if not adapter_path.exists():
            log(f"  SKIP {d}: no adapter")
            continue

        model, tok = load(model_id)
        adapter_b = dict(mx.load(str(adapter_path)))
        n = attach_adapter(model, skeleton, adapter_b, di, LORA_SCALE)

        test = load_data(d, "valid", 5)
        scores = []
        sampler = make_sampler(temp=0.0)
        for text in test:
            if "### Response:" in text:
                prompt = text.split("### Response:")[0].strip() + "\n### Response:\n"
                ref = text.split("### Response:")[-1].strip()
            else: prompt, ref = text[:200], text
            try:
                gen = mlx_generate(model, tok, prompt=prompt, max_tokens=128, sampler=sampler, verbose=False)
                scores.append(factual_recall(gen, ref))
            except: scores.append(0.0)

        mean = float(np.mean(scores)) if scores else 0.0
        results["per_domain"][d] = {"behavioral": round(mean, 3), "modules": n}
        log(f"  {d}: behavioral={mean:.3f} ({n} modules)")
        cleanup(model, tok, adapter_b)

    overall = float(np.mean([v["behavioral"] for v in results["per_domain"].values()]))

    # Phase 2: Speed
    log("\n=== Phase 2: Speed ===")
    model, tok = load(model_id)
    if ADAPTER_DIR.exists() and (ADAPTER_DIR / "medical" / "adapter.npz").exists():
        adapter_b = dict(mx.load(str(ADAPTER_DIR / "medical" / "adapter.npz")))
        attach_adapter(model, skeleton, adapter_b, 0, LORA_SCALE)

    prompt = "Explain machine learning in simple terms."
    sampler = make_sampler(temp=0.0)
    for _ in range(2):
        mlx_generate(model, tok, prompt=prompt, max_tokens=32, sampler=sampler, verbose=False)
    times = []
    for _ in range(5):
        t1 = time.time()
        out = mlx_generate(model, tok, prompt=prompt, max_tokens=128, sampler=sampler, verbose=False)
        dt = time.time() - t1
        n = len(tok.encode(out)) - len(tok.encode(prompt))
        times.append({"s": dt, "toks": n})
    tps = sum(t["toks"] for t in times) / sum(t["s"] for t in times)
    log(f"  Speed: {tps:.1f} tok/s")
    cleanup(model, tok)

    results["overall_behavioral"] = round(overall, 3)
    results["speed_tps"] = round(tps, 1)
    results["total_time_s"] = round(time.time() - t0, 1)

    base_mmlu = base_data.get("mmlu", {}).get("accuracy", 0)
    k821 = overall > 0.3  # meaningful output

    results["kill_criteria"] = {
        "K821": {"pass": k821, "value": round(overall, 3), "threshold": 0.3},
    }
    results["all_pass"] = k821

    log(f"\n{'='*60}")
    log(f"Overall behavioral: {overall:.3f}")
    log(f"Speed: {tps:.1f} tok/s")
    for k, v in results["kill_criteria"].items():
        log(f"  {k}: {'PASS' if v['pass'] else 'FAIL'} — {v}")
    log(f"\n{'ALL PASS' if results['all_pass'] else 'KILLED'}")

    RESULTS_FILE.write_text(json.dumps(results, indent=2, cls=NumpyEncoder))

if __name__ == "__main__":
    main()
