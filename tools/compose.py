#!/usr/bin/env python3
"""compose — Plug-and-Play LoRA Composition CLI.

Register, compose, route, and serve LoRA adapters on any base model.

Usage:
    compose init --base Qwen/Qwen2.5-0.5B       # init registry
    compose add expert.safetensors --name python  # register adapter
    compose list                                  # show expert registry
    compose calibrate --steps 100                 # train softmax router
    compose serve --port 8080                     # serve with routing
    compose bench                                 # benchmark composition
    compose remove python                         # remove adapter
    compose export merged.safetensors             # export merged weights
"""

import argparse
import hashlib
import json
import os
import sys
import time
from bisect import bisect_right
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

REGISTRY_FILE = "compose_registry.json"


# ── Registry ────────────────────────────────────────────────────────────

class ExpertRegistry:
    def __init__(self, registry_dir="."):
        self.dir = Path(registry_dir)
        self.file = self.dir / REGISTRY_FILE
        self.data = self._load()

    def _load(self):
        if self.file.exists():
            with open(self.file) as f:
                return json.load(f)
        return {"base_model": None, "experts": {}, "router": None}

    def save(self):
        with open(self.file, "w") as f:
            json.dump(self.data, f, indent=2)

    def init(self, base_model):
        self.data = {"base_model": base_model, "experts": {}, "router": None}
        self.save()
        print(f"Registry initialized with base: {base_model}")

    def add(self, name, path, domain=None, rank=None):
        self.data["experts"][name] = {
            "path": str(path),
            "domain": domain,
            "rank": rank,
            "added": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.save()
        print(f"Added expert '{name}' from {path}")

    def remove(self, name):
        if name in self.data["experts"]:
            del self.data["experts"][name]
            self.save()
            print(f"Removed expert '{name}'")
        else:
            print(f"Expert '{name}' not found")

    def list_experts(self):
        if not self.data["experts"]:
            print("No experts registered. Use 'compose add' to register.")
            return
        print(f"Base: {self.data['base_model']}")
        print(f"{'Name':<20} {'Domain':<15} {'Rank':<6} {'Path'}")
        print("-" * 70)
        for name, info in self.data["experts"].items():
            print(f"{name:<20} {info.get('domain',''):<15} {info.get('rank',''):<6} {info['path']}")
        print(f"\n{len(self.data['experts'])} experts registered")


# ── Hash Ring Router ────────────────────────────────────────────────────

class HashRingRouter:
    """Consistent hash ring for zero-shot expert routing."""
    def __init__(self, expert_names, virtual_nodes=150):
        self.expert_names = list(expert_names)
        self.virtual_nodes = virtual_nodes
        self.ring = []
        for name in self.expert_names:
            for vn in range(virtual_nodes):
                h = int(hashlib.md5(f"{name}_vn_{vn}".encode()).hexdigest(), 16)
                self.ring.append((h, name))
        self.ring.sort()
        self._hashes = [h for h, _ in self.ring]
        self._names = [n for _, n in self.ring]

    def route(self, token_text, top_k=2):
        h = int(hashlib.md5(token_text.encode()).hexdigest(), 16)
        idx = bisect_right(self._hashes, h) % len(self.ring)
        experts = []
        seen = set()
        for offset in range(len(self.ring)):
            e = self._names[(idx + offset) % len(self.ring)]
            if e not in seen:
                seen.add(e)
                experts.append(e)
                if len(experts) >= top_k:
                    break
        return experts


# ── Softmax Router ──────────────────────────────────────────────────────

class LearnedRouter(nn.Module):
    def __init__(self, hidden_dim, n_experts, top_k=2):
        super().__init__()
        self.gate = nn.Linear(hidden_dim, n_experts)
        self.top_k = top_k
        self.expert_names = []

    def forward(self, hidden_states):
        logits = self.gate(hidden_states)
        topk_vals, topk_idx = logits.topk(self.top_k, dim=-1)
        weights = F.softmax(topk_vals, dim=-1)
        return weights, topk_idx


# ── LoRA Expert Loader ──────────────────────────────────────────────────

class ExpertLoader:
    """Loads and caches LoRA expert states with LRU eviction."""
    def __init__(self, registry, cache_size=8):
        self.registry = registry
        self.cache_size = cache_size
        self.cache = {}  # name -> (state_dict, last_access)
        self.access_count = 0
        self.stats = {"hits": 0, "misses": 0, "loads": 0, "evictions": 0}

    def load(self, name):
        self.access_count += 1
        if name in self.cache:
            self.stats["hits"] += 1
            state, _ = self.cache[name]
            self.cache[name] = (state, self.access_count)
            return state

        self.stats["misses"] += 1
        self.stats["loads"] += 1

        # Load from disk
        info = self.registry.data["experts"][name]
        path = info["path"]

        if path.endswith(".safetensors"):
            from safetensors.torch import load_file
            state = load_file(path)
        elif path.endswith(".pt") or path.endswith(".pth"):
            state = torch.load(path, map_location="cpu", weights_only=True)
        elif path.endswith(".bin"):
            state = torch.load(path, map_location="cpu", weights_only=True)
        else:
            # Try as directory (peft adapter)
            adapter_path = Path(path)
            if (adapter_path / "adapter_model.safetensors").exists():
                from safetensors.torch import load_file
                state = load_file(str(adapter_path / "adapter_model.safetensors"))
            elif (adapter_path / "adapter_model.bin").exists():
                state = torch.load(str(adapter_path / "adapter_model.bin"),
                                  map_location="cpu", weights_only=True)
            else:
                raise FileNotFoundError(f"No adapter found at {path}")

        # Evict if cache full
        if len(self.cache) >= self.cache_size:
            oldest = min(self.cache, key=lambda k: self.cache[k][1])
            del self.cache[oldest]
            self.stats["evictions"] += 1

        self.cache[name] = (state, self.access_count)
        return state

    def merge_experts(self, names, weights=None):
        """Merge multiple expert states with optional weights."""
        if weights is None:
            weights = [1.0 / len(names)] * len(names)

        states = [self.load(n) for n in names]
        merged = {}
        for key in states[0]:
            merged[key] = sum(w * s[key].float() for w, s in zip(weights, states))
        return merged


# ── Compose Engine ──────────────────────────────────────────────────────

class ComposeEngine:
    """Main composition engine: base model + LoRA experts + router."""
    def __init__(self, registry_dir="."):
        self.registry = ExpertRegistry(registry_dir)
        self.loader = ExpertLoader(self.registry)
        self.base_model = None
        self.tokenizer = None
        self.router = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_base(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model

        model_name = self.registry.data["base_model"]
        hf_home = os.environ.get("HF_HOME", None)

        print(f"Loading base model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=hf_home, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name, cache_dir=hf_home, trust_remote_code=True,
            torch_dtype=torch.float32).to(self.device)

        # Wrap with LoRA scaffold (rank from first expert or default)
        experts = self.registry.data["experts"]
        rank = 16
        if experts:
            first = next(iter(experts.values()))
            rank = first.get("rank", 16) or 16

        cfg = LoraConfig(r=rank, lora_alpha=rank,
                        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                       "up_proj", "gate_proj", "down_proj"],
                        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
        self.base_model = get_peft_model(self.base_model, cfg)
        self.base_model.eval()
        print(f"  Loaded on {self.device}")

    def route_and_generate(self, text, max_new_tokens=100, top_k=2):
        """Route input to experts and generate."""
        if self.base_model is None:
            self.load_base()

        expert_names = list(self.registry.data["experts"].keys())
        if not expert_names:
            raise ValueError("No experts registered")

        # Hash-ring routing (zero-shot)
        hr = HashRingRouter(expert_names)
        selected = hr.route(text, top_k=top_k)
        print(f"  Routed to: {selected}")

        # Merge selected experts
        merged_state = self.loader.merge_experts(selected)
        self.base_model.load_state_dict(merged_state, strict=False)

        # Generate
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.base_model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=True, temperature=0.7, top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id)
        return self.tokenizer.decode(out[0], skip_special_tokens=True)

    def benchmark(self, n_prompts=20):
        """Benchmark composition quality and latency."""
        if self.base_model is None:
            self.load_base()

        expert_names = list(self.registry.data["experts"].keys())
        n_experts = len(expert_names)

        test_prompts = [
            "def fibonacci(n):",
            "function quickSort(arr) {",
            "The patient presents with acute",
            "Under Section 42 of the Act,",
            "Calculate the integral of",
        ]

        results = {"experts": n_experts, "prompts": []}

        for prompt in test_prompts[:n_prompts]:
            hr = HashRingRouter(expert_names)
            selected = hr.route(prompt, top_k=min(2, n_experts))

            t0 = time.perf_counter()
            merged = self.loader.merge_experts(selected)
            merge_ms = (time.perf_counter() - t0) * 1000

            self.base_model.load_state_dict(merged, strict=False)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            t0 = time.perf_counter()
            with torch.no_grad():
                out = self.base_model(**inputs)
            fwd_ms = (time.perf_counter() - t0) * 1000

            results["prompts"].append({
                "prompt": prompt[:50],
                "experts": selected,
                "merge_ms": round(merge_ms, 2),
                "forward_ms": round(fwd_ms, 2),
            })

        results["cache_stats"] = self.loader.stats
        avg_merge = sum(p["merge_ms"] for p in results["prompts"]) / len(results["prompts"])
        avg_fwd = sum(p["forward_ms"] for p in results["prompts"]) / len(results["prompts"])
        results["avg_merge_ms"] = round(avg_merge, 2)
        results["avg_forward_ms"] = round(avg_fwd, 2)
        results["overhead_pct"] = round(avg_merge / avg_fwd * 100, 1)

        print(f"\n  Benchmark Results:")
        print(f"    Experts: {n_experts}")
        print(f"    Avg merge: {avg_merge:.1f}ms")
        print(f"    Avg forward: {avg_fwd:.1f}ms")
        print(f"    Overhead: {results['overhead_pct']}%")
        print(f"    Cache: {self.loader.stats}")

        return results


# ── CLI ─────────────────────────────────────────────────────────────────

def cmd_init(args):
    reg = ExpertRegistry(args.dir)
    reg.init(args.base)

def cmd_add(args):
    reg = ExpertRegistry(args.dir)
    reg.add(args.name, args.path, domain=args.domain, rank=args.rank)

def cmd_remove(args):
    reg = ExpertRegistry(args.dir)
    reg.remove(args.name)

def cmd_list(args):
    reg = ExpertRegistry(args.dir)
    reg.list_experts()

def cmd_bench(args):
    engine = ComposeEngine(args.dir)
    results = engine.benchmark(n_prompts=args.prompts)
    out_path = Path(args.dir) / "bench_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to {out_path}")

def cmd_generate(args):
    engine = ComposeEngine(args.dir)
    output = engine.route_and_generate(args.prompt, max_new_tokens=args.max_tokens, top_k=args.top_k)
    print(f"\n{output}")

def cmd_serve(args):
    """Simple HTTP server for compose inference."""
    try:
        from http.server import HTTPServer, BaseHTTPRequestHandler
    except ImportError:
        print("http.server not available")
        return

    engine = ComposeEngine(args.dir)
    engine.load_base()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            prompt = body.get("prompt", "")
            max_tokens = body.get("max_tokens", 100)
            top_k = body.get("top_k", 2)

            output = engine.route_and_generate(prompt, max_new_tokens=max_tokens, top_k=top_k)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"output": output}).encode())

        def log_message(self, format, *a):
            print(f"  {self.address_string()} - {format % a}")

    server = HTTPServer(("0.0.0.0", args.port), Handler)
    print(f"Serving on port {args.port}...")
    print(f"  POST /generate {{\"prompt\": \"...\", \"max_tokens\": 100, \"top_k\": 2}}")
    server.serve_forever()


def main():
    parser = argparse.ArgumentParser(description="compose - Plug-and-Play LoRA Composition")
    parser.add_argument("--dir", default=".", help="Registry directory")
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("init", help="Initialize registry")
    p.add_argument("--base", required=True, help="Base model name/path")

    p = sub.add_parser("add", help="Register an adapter")
    p.add_argument("path", help="Path to adapter weights")
    p.add_argument("--name", required=True, help="Expert name")
    p.add_argument("--domain", help="Domain description")
    p.add_argument("--rank", type=int, help="LoRA rank")

    p = sub.add_parser("remove", help="Remove an adapter")
    p.add_argument("name", help="Expert name")

    sub.add_parser("list", help="List registered experts")

    p = sub.add_parser("bench", help="Benchmark composition")
    p.add_argument("--prompts", type=int, default=20, help="Number of test prompts")

    p = sub.add_parser("generate", help="Generate with routing")
    p.add_argument("prompt", help="Input prompt")
    p.add_argument("--max-tokens", type=int, default=100)
    p.add_argument("--top-k", type=int, default=2)

    p = sub.add_parser("serve", help="Start HTTP server")
    p.add_argument("--port", type=int, default=8080)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    handlers = {
        "init": cmd_init, "add": cmd_add, "remove": cmd_remove,
        "list": cmd_list, "bench": cmd_bench, "generate": cmd_generate,
        "serve": cmd_serve,
    }
    handlers[args.command](args)


if __name__ == "__main__":
    main()
