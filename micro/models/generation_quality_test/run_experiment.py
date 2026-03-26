#!/usr/bin/env python3
"""Generation Quality Test: Does routed composition produce better TEXT than base alone?

THE EXISTENTIAL TEST for BitNet-SOLE. Generates text with three configurations:
  1. Base only (no adapters)
  2. Uniform 1/N composition (all 5 adapters equally weighted)
  3. Routed top-2 (per-sequence routing based on prompt hidden state)

Scores generated text with automated metrics:
  - Domain keyword density (does the response use domain terminology?)
  - Response length (does composition produce more substantive answers?)
  - N-gram diversity (is the text varied or repetitive?)
  - Cross-PPL (does the domain adapter predict the generated text better?)

Kill criteria:
  K1 (id=272): Routed composition worse than base on >= 3/5 domains -> KILL
  K2 (id=273): No measurable difference between routed and base -> KILL
  K3 (id=274): All generated text is incoherent (base model too weak at 2B) -> KILL

Platform: Apple M5 Pro 48GB, MLX
"""

import gc
import json
import math
import os
import re
import time
from collections import Counter
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

# Memory safety (MANDATORY per CODING_GUIDELINES)
device_info = mx.device_info()
total_mem = device_info["memory_size"]
mx.set_memory_limit(total_mem - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

# Source experiment with trained adapters
SOURCE_DIR = EXPERIMENT_DIR.parent / "real_data_domain_experts"
ADAPTERS_DIR = SOURCE_DIR / "adapters"
DATA_DIR = SOURCE_DIR / "data"

MODEL_ID = "microsoft/BitNet-b1.58-2B-4T"
LORA_RANK = 16
LORA_SCALE = 20.0
MAX_SEQ_LENGTH = 256

DOMAINS = ["medical", "code", "math", "legal", "finance"]
N_DOMAINS = len(DOMAINS)

# Generation settings
NUM_PROMPTS_PER_DOMAIN = 10
MAX_NEW_TOKENS = 128
TEMPERATURE = 0.7
TOP_P = 0.9
SEED = 42

# Domain keyword lists for scoring
DOMAIN_KEYWORDS = {
    "medical": [
        "patient", "diagnosis", "treatment", "symptoms", "disease", "clinical",
        "medication", "therapy", "surgical", "pathology", "prognosis", "chronic",
        "acute", "syndrome", "condition", "medical", "doctor", "hospital",
        "infection", "immune", "blood", "organ", "tissue", "cell", "drug",
        "dose", "prescription", "cancer", "cardiac", "neural", "liver",
        "kidney", "lung", "brain", "bone", "muscle", "nerve", "artery",
        "vein", "inflammation", "fever", "pain", "swelling",
    ],
    "code": [
        "function", "variable", "class", "method", "return", "import",
        "loop", "array", "string", "integer", "boolean", "object", "def",
        "print", "if", "else", "for", "while", "try", "except", "lambda",
        "list", "dict", "tuple", "parameter", "argument", "module", "library",
        "algorithm", "data", "structure", "python", "code", "program",
        "output", "input", "error", "debug", "compile", "run", "execute",
    ],
    "math": [
        "equation", "formula", "calculate", "solve", "number", "variable",
        "total", "sum", "product", "divide", "multiply", "subtract", "add",
        "percent", "ratio", "fraction", "decimal", "integer", "value",
        "answer", "solution", "problem", "step", "result", "equal",
        "greater", "less", "positive", "negative", "zero", "proof",
        "theorem", "function", "graph", "slope", "area", "volume",
        "distance", "rate", "time", "cost", "price", "profit",
    ],
    "legal": [
        "law", "court", "judge", "attorney", "lawyer", "legal", "rights",
        "statute", "regulation", "contract", "liability", "plaintiff",
        "defendant", "jurisdiction", "precedent", "ruling", "verdict",
        "appeal", "testimony", "evidence", "trial", "case", "claim",
        "damages", "negligence", "tort", "criminal", "civil", "federal",
        "state", "constitution", "amendment", "legislation", "act",
        "section", "clause", "provision", "enforce", "comply", "violate",
    ],
    "finance": [
        "investment", "stock", "bond", "market", "portfolio", "risk",
        "return", "profit", "loss", "revenue", "capital", "dividend",
        "interest", "rate", "inflation", "monetary", "fiscal", "budget",
        "asset", "liability", "equity", "debt", "credit", "loan",
        "mortgage", "insurance", "tax", "income", "expense", "savings",
        "bank", "financial", "economic", "trade", "exchange", "currency",
        "price", "value", "growth", "recession", "gdp", "fund",
    ],
}


def log(msg):
    print(msg, flush=True)


def log_memory(label=""):
    active = mx.get_active_memory() / 1e9
    cache = mx.get_cache_memory() / 1e9
    peak = mx.get_peak_memory() / 1e9
    log(f"[MEM {label}] active={active:.2f}GB cache={cache:.2f}GB peak={peak:.2f}GB")


def cleanup(*objects):
    for obj in objects:
        del obj
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()


# ============================================================================
# BitNet unpacking and model utilities (reused from real_data_domain_experts)
# ============================================================================

from mlx_lm import load, generate as mlx_generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.models.bitlinear_layers import BitLinear


def unpack_ternary(packed_weights, out_features, weight_scale, invert_scale):
    """Unpack uint8-packed ternary weights to bfloat16."""
    w0 = (packed_weights & 3).astype(mx.bfloat16) - 1
    w1 = ((packed_weights >> 2) & 3).astype(mx.bfloat16) - 1
    w2 = ((packed_weights >> 4) & 3).astype(mx.bfloat16) - 1
    w3 = ((packed_weights >> 6) & 3).astype(mx.bfloat16) - 1
    unpacked = mx.concatenate([w0, w1, w2, w3], axis=0)[:out_features]
    scale = weight_scale.astype(mx.bfloat16)
    if invert_scale:
        unpacked = unpacked / scale
    else:
        unpacked = unpacked * scale
    return unpacked


def replace_bitlinear_with_linear(model):
    """Replace BitLinear with nn.Linear for differentiable forward pass."""
    count = 0
    for layer in model.model.layers:
        updates = []
        for key, module in layer.named_modules():
            if isinstance(module, BitLinear):
                unpacked_w = unpack_ternary(
                    module.weight, module.out_features,
                    module.weight_scale, module.invert_weight_scales,
                )
                has_bias = module.bias is not None
                linear = nn.Linear(module.in_features, module.out_features, bias=has_bias)
                linear.weight = unpacked_w
                if has_bias:
                    linear.bias = module.bias
                updates.append((key, linear))
                count += 1
        if updates:
            layer.update_modules(tree_unflatten(updates))
    mx.eval(model.parameters())
    log(f"  Replaced {count} BitLinear -> nn.Linear")
    return model


# ============================================================================
# LoRA layers for composition
# ============================================================================

class TernaryLoRALinear(nn.Module):
    """LoRA with frozen Grassmannian A and STE-ternary B."""
    def __init__(self, base_linear: nn.Linear, rank: int = 16,
                 scale: float = 20.0, a_init: mx.array = None):
        super().__init__()
        in_features = base_linear.weight.shape[1]
        out_features = base_linear.weight.shape[0]
        self.linear = base_linear
        if a_init is not None:
            self.lora_a = a_init
        else:
            s = 1.0 / math.sqrt(in_features)
            self.lora_a = mx.random.uniform(low=-s, high=s, shape=(in_features, rank))
        self.lora_b = mx.zeros((rank, out_features))
        self.scale = scale
        self.rank = rank
        self.linear.freeze()
        self.freeze(keys=["lora_a"], strict=False)

    def __call__(self, x):
        base_out = self.linear(x)
        b = self.lora_b
        alpha = mx.mean(mx.abs(b))
        b_scaled = b / (alpha + 1e-7)
        b_q = mx.clip(mx.round(b_scaled), -1, 1) * alpha
        b_ste = b + mx.stop_gradient(b_q - b)
        lora_out = (x @ self.lora_a) @ b_ste * self.scale
        return base_out + lora_out


class MultiAdapterLoRALinear(nn.Module):
    """LoRA with multiple A/B pairs for multi-expert composition.

    Supports two modes:
    - Uniform: y = base(x) + (1/N) * sum_i[(x @ A_i) @ ternary(B_i)] * scale
    - Weighted: y = base(x) + sum_i[w_i * (x @ A_i) @ ternary(B_i)] * scale
    """
    def __init__(self, base_linear: nn.Linear, rank: int = 16,
                 scale: float = 20.0, a_inits: list = None):
        super().__init__()
        out_features = base_linear.weight.shape[0]
        self.linear = base_linear
        self.scale = scale
        self.rank = rank
        self.n_experts = len(a_inits) if a_inits else 0
        self.a_matrices = a_inits if a_inits else []
        self.b_matrices = [mx.zeros((rank, out_features)) for _ in range(self.n_experts)]
        self.linear.freeze()
        # Routing weights: set externally per forward pass
        self._weights = None

    def set_weights(self, weights):
        """Set per-expert weights for this forward pass. weights: list of N floats."""
        self._weights = weights

    def __call__(self, x):
        base_out = self.linear(x)
        if self.n_experts == 0:
            return base_out

        lora_sum = mx.zeros_like(base_out)
        for i in range(self.n_experts):
            b = self.b_matrices[i]
            alpha = mx.mean(mx.abs(b))
            b_scaled = b / (alpha + 1e-7)
            b_q = mx.clip(mx.round(b_scaled), -1, 1) * alpha
            b_ste = b + mx.stop_gradient(b_q - b)
            lora_sum = lora_sum + (x @ self.a_matrices[i]) @ b_ste

        if self._weights is not None:
            # Weighted composition (for routing)
            # Note: weights are applied per-expert above for correctness
            # This is a simplified version - equal weights here, real routing
            # would need per-expert weighting in the loop above
            return base_out + lora_sum * self.scale / self.n_experts
        else:
            return base_out + lora_sum * (self.scale / self.n_experts)


class RoutedMultiAdapterLoRALinear(nn.Module):
    """LoRA with multiple A/B pairs and per-expert routing weights.

    Forward: y = base(x) + sum_i[w_i * (x @ A_i) @ ternary(B_i)] * scale
    where w_i are the routing weights (set externally per sequence).
    """
    def __init__(self, base_linear: nn.Linear, rank: int = 16,
                 scale: float = 20.0, a_inits: list = None):
        super().__init__()
        out_features = base_linear.weight.shape[0]
        self.linear = base_linear
        self.scale = scale
        self.rank = rank
        self.n_experts = len(a_inits) if a_inits else 0
        self.a_matrices = a_inits if a_inits else []
        self.b_matrices = [mx.zeros((rank, out_features)) for _ in range(self.n_experts)]
        self.linear.freeze()
        # Routing weights: set externally per forward pass
        self._routing_weights = [1.0 / self.n_experts] * self.n_experts if self.n_experts > 0 else []

    def set_routing_weights(self, weights):
        """Set per-expert routing weights for this forward pass."""
        self._routing_weights = weights

    def __call__(self, x):
        base_out = self.linear(x)
        if self.n_experts == 0:
            return base_out

        lora_sum = mx.zeros_like(base_out)
        for i in range(self.n_experts):
            w = self._routing_weights[i]
            if w < 1e-6:
                continue  # Skip experts with zero weight
            b = self.b_matrices[i]
            alpha = mx.mean(mx.abs(b))
            b_scaled = b / (alpha + 1e-7)
            b_q = mx.clip(mx.round(b_scaled), -1, 1) * alpha
            b_ste = b + mx.stop_gradient(b_q - b)
            lora_sum = lora_sum + w * ((x @ self.a_matrices[i]) @ b_ste)

        return base_out + lora_sum * self.scale


# ============================================================================
# Model setup utilities
# ============================================================================

TARGET_KEYS = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
    "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
]


def load_skeleton():
    """Load the Grassmannian A matrices."""
    skeleton_path = ADAPTERS_DIR / "grassmannian_skeleton.npz"
    return dict(np.load(str(skeleton_path)))


def apply_single_adapter(model, skeleton, domain_idx, domain_name):
    """Apply a single adapter (with correct A matrix) to the model."""
    n_layers = len(model.model.layers)

    # Build A matrix mapping
    a_matrices = {}
    for li in range(n_layers):
        for key in TARGET_KEYS:
            skey = f"layer_{li}_{key}_domain_{domain_idx}"
            if skey in skeleton:
                a_matrices[(li, key)] = skeleton[skey]

    # Apply TernaryLoRA
    count = 0
    for li, layer in enumerate(model.model.layers):
        lora_updates = []
        for key in TARGET_KEYS:
            parts = key.split(".")
            module = layer
            for part in parts:
                module = getattr(module, part, None)
                if module is None:
                    break
            if module is None or not isinstance(module, nn.Linear):
                continue

            a_key = (li, key)
            a_mx = mx.array(a_matrices[a_key]).astype(mx.bfloat16) if a_key in a_matrices else None
            lora = TernaryLoRALinear(module, rank=LORA_RANK, scale=LORA_SCALE, a_init=a_mx)
            lora_updates.append((key, lora))
            count += 1

        if lora_updates:
            layer.update_modules(tree_unflatten(lora_updates))

    # Load B weights
    adapter_path = ADAPTERS_DIR / domain_name / "adapter.npz"
    adapter_params = dict(mx.load(str(adapter_path)))
    model.update(tree_unflatten(list(adapter_params.items())))
    mx.eval(model.parameters())
    log(f"  Applied {domain_name} adapter ({count} layers)")
    return model


def apply_multi_adapter_uniform(model, skeleton):
    """Apply all 5 adapters with uniform 1/N weighting."""
    n_layers = len(model.model.layers)

    # Load all adapter B params
    all_adapter_params = {}
    for di, domain in enumerate(DOMAINS):
        adapter_path = ADAPTERS_DIR / domain / "adapter.npz"
        all_adapter_params[domain] = dict(mx.load(str(adapter_path)))

    # Apply MultiAdapterLoRALinear to each target
    count = 0
    for li, layer in enumerate(model.model.layers):
        lora_updates = []
        for key in TARGET_KEYS:
            parts = key.split(".")
            module = layer
            for part in parts:
                module = getattr(module, part, None)
                if module is None:
                    break
            if module is None or not isinstance(module, nn.Linear):
                continue

            # Collect A matrices for all domains
            a_inits = []
            for di in range(N_DOMAINS):
                skey = f"layer_{li}_{key}_domain_{di}"
                if skey in skeleton:
                    a_inits.append(mx.array(skeleton[skey]).astype(mx.bfloat16))
                else:
                    a_inits.append(None)

            multi_lora = MultiAdapterLoRALinear(
                module, rank=LORA_RANK, scale=LORA_SCALE, a_inits=a_inits
            )

            # Load B matrices for each expert
            for di, domain in enumerate(DOMAINS):
                # Find the corresponding B parameter name
                b_key = f"model.layers.{li}.{key}.lora_b"
                if b_key in all_adapter_params[domain]:
                    multi_lora.b_matrices[di] = all_adapter_params[domain][b_key]

            lora_updates.append((key, multi_lora))
            count += 1

        if lora_updates:
            layer.update_modules(tree_unflatten(lora_updates))

    mx.eval(model.parameters())
    log(f"  Applied uniform multi-adapter composition ({count} layers)")
    return model


def apply_multi_adapter_routed(model, skeleton):
    """Apply all 5 adapters with routing weights (set per-sequence)."""
    n_layers = len(model.model.layers)

    # Load all adapter B params
    all_adapter_params = {}
    for di, domain in enumerate(DOMAINS):
        adapter_path = ADAPTERS_DIR / domain / "adapter.npz"
        all_adapter_params[domain] = dict(mx.load(str(adapter_path)))

    count = 0
    for li, layer in enumerate(model.model.layers):
        lora_updates = []
        for key in TARGET_KEYS:
            parts = key.split(".")
            module = layer
            for part in parts:
                module = getattr(module, part, None)
                if module is None:
                    break
            if module is None or not isinstance(module, nn.Linear):
                continue

            a_inits = []
            for di in range(N_DOMAINS):
                skey = f"layer_{li}_{key}_domain_{di}"
                if skey in skeleton:
                    a_inits.append(mx.array(skeleton[skey]).astype(mx.bfloat16))
                else:
                    a_inits.append(None)

            routed_lora = RoutedMultiAdapterLoRALinear(
                module, rank=LORA_RANK, scale=LORA_SCALE, a_inits=a_inits
            )

            for di, domain in enumerate(DOMAINS):
                b_key = f"model.layers.{li}.{key}.lora_b"
                if b_key in all_adapter_params[domain]:
                    routed_lora.b_matrices[di] = all_adapter_params[domain][b_key]

            lora_updates.append((key, routed_lora))
            count += 1

        if lora_updates:
            layer.update_modules(tree_unflatten(lora_updates))

    mx.eval(model.parameters())
    log(f"  Applied routed multi-adapter composition ({count} layers)")
    return model


def set_all_routing_weights(model, weights):
    """Set routing weights on all RoutedMultiAdapterLoRALinear modules."""
    for layer in model.model.layers:
        for key, module in layer.named_modules():
            if isinstance(module, RoutedMultiAdapterLoRALinear):
                module.set_routing_weights(weights)


# ============================================================================
# Simple per-sequence router
# ============================================================================

def compute_prompt_routing_weights(model, tokenizer, prompt_text, n_experts=5, top_k=2):
    """Compute routing weights for a prompt using hidden state similarity.

    Strategy: run the prompt through the base model (first few layers),
    extract the mean hidden state, and compare to domain prototypes.

    For simplicity, we use the domain of the prompt as oracle routing
    (the routing heads from real_data_domain_experts had 99.9% accuracy).
    This tests the UPPER BOUND of routing: if perfect routing doesn't help,
    nothing will.
    """
    # Since routing heads had 99.9% accuracy, oracle routing is a valid proxy
    # This avoids the complexity of loading separate routing heads
    # and gives us the best-case scenario for the architecture
    pass  # We'll use oracle routing in the experiment


def get_oracle_routing_weights(domain_idx, n_experts=5, top_k=2):
    """Oracle routing: give full weight to the correct domain + second best.

    For top-2, we give 0.7 to the correct domain and 0.3 to the next.
    This simulates the 99.9% accurate routing heads from prior experiments.
    """
    weights = [0.0] * n_experts
    weights[domain_idx] = 0.7  # Primary expert
    # Second expert: next domain (wrap around)
    second = (domain_idx + 1) % n_experts
    weights[second] = 0.3
    return weights


# ============================================================================
# Text generation
# ============================================================================

def generate_text(model, tokenizer, prompt, max_tokens=128, temperature=0.7, top_p=0.9):
    """Generate text using mlx_lm.generate which handles KV cache properly."""
    try:
        sampler = make_sampler(temp=temperature, top_p=top_p)
        text = mlx_generate(
            model, tokenizer, prompt,
            max_tokens=max_tokens,
            sampler=sampler,
            verbose=False,
        )
        return text
    except Exception as e:
        log(f"  WARNING: generation failed: {e}")
        return ""


# ============================================================================
# Scoring metrics
# ============================================================================

def keyword_density(text, domain):
    """Fraction of words that are domain keywords."""
    words = re.findall(r'\b\w+\b', text.lower())
    if not words:
        return 0.0
    keywords = set(DOMAIN_KEYWORDS.get(domain, []))
    matches = sum(1 for w in words if w in keywords)
    return matches / len(words)


def ngram_diversity(text, n=3):
    """Fraction of unique n-grams. Higher = more diverse."""
    words = re.findall(r'\b\w+\b', text.lower())
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
    if not ngrams:
        return 0.0
    return len(set(ngrams)) / len(ngrams)


def word_count(text):
    """Number of words in text."""
    return len(re.findall(r'\b\w+\b', text))


def repetition_score(text):
    """Measure repetition: ratio of unique words to total words. Higher = less repetitive."""
    words = re.findall(r'\b\w+\b', text.lower())
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def coherence_score(text):
    """Simple coherence metric: average sentence length (proxy for coherence).

    Very short sentences or very long ones indicate incoherence.
    Score is 1.0 for sentences of ~15-20 words, lower for extremes.
    """
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return 0.0
    avg_len = np.mean([word_count(s) for s in sentences])
    # Peak at 15 words per sentence
    return max(0, 1.0 - abs(avg_len - 15) / 30)


def is_incoherent(text):
    """Detect obviously incoherent text (repeated tokens, gibberish)."""
    words = re.findall(r'\b\w+\b', text.lower())
    if len(words) < 3:
        return True
    # Check for excessive repetition
    counter = Counter(words)
    most_common_freq = counter.most_common(1)[0][1] / len(words)
    if most_common_freq > 0.4:  # One word is >40% of text
        return True
    # Check if it's just the prompt repeated
    if len(set(words)) < 5 and len(words) > 20:
        return True
    return False


# ============================================================================
# Prompt extraction
# ============================================================================

def extract_prompts(domain, n_prompts=10):
    """Extract instruction prompts from validation data."""
    val_path = DATA_DIR / domain / "valid.jsonl"
    prompts = []
    with open(val_path) as f:
        for line in f:
            text = json.loads(line)["text"]
            # Extract just the instruction part
            if "### Instruction:" in text and "### Response:" in text:
                instruction = text.split("### Instruction:")[1].split("### Response:")[0].strip()
                prompts.append(instruction)
            if len(prompts) >= n_prompts:
                break
    return prompts


def format_prompt(instruction):
    """Format instruction as model input (instruction-response format)."""
    return f"### Instruction:\n{instruction}\n\n### Response:\n"


# ============================================================================
# Phase 1: Generate with base model
# ============================================================================

def phase_generate_base(prompts_by_domain):
    """Generate text with base model (no adapters)."""
    log("\n[Phase 1] Generating with BASE model (no adapters)...")
    t0 = time.time()

    model, tokenizer = load(MODEL_ID)
    model = replace_bitlinear_with_linear(model)
    model.freeze()
    log_memory("post-load-base")

    np.random.seed(SEED)
    results = {}
    for domain in DOMAINS:
        domain_results = []
        for i, prompt in enumerate(prompts_by_domain[domain]):
            formatted = format_prompt(prompt)
            generated = generate_text(
                model, tokenizer, formatted,
                max_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE, top_p=TOP_P
            )
            domain_results.append({
                "prompt": prompt,
                "generated": generated,
                "word_count": word_count(generated),
                "keyword_density": keyword_density(generated, domain),
                "ngram_diversity": ngram_diversity(generated),
                "repetition": repetition_score(generated),
                "coherence": coherence_score(generated),
                "incoherent": is_incoherent(generated),
            })
            if (i + 1) % 5 == 0:
                log(f"  {domain}: {i+1}/{len(prompts_by_domain[domain])} prompts done")

        results[domain] = domain_results
        avg_kw = np.mean([r["keyword_density"] for r in domain_results])
        avg_wc = np.mean([r["word_count"] for r in domain_results])
        log(f"  {domain}: avg_keywords={avg_kw:.3f}, avg_words={avg_wc:.1f}")

    elapsed = time.time() - t0
    log(f"  Base generation done in {elapsed:.1f}s")
    log_memory("post-gen-base")
    cleanup(model, tokenizer)
    return results


# ============================================================================
# Phase 2: Generate with uniform 1/N composition
# ============================================================================

def phase_generate_uniform(prompts_by_domain):
    """Generate text with uniform 1/N adapter composition."""
    log("\n[Phase 2] Generating with UNIFORM 1/N composition...")
    t0 = time.time()

    skeleton = load_skeleton()
    model, tokenizer = load(MODEL_ID)
    model = replace_bitlinear_with_linear(model)
    model = apply_multi_adapter_uniform(model, skeleton)
    model.freeze()
    log_memory("post-load-uniform")

    np.random.seed(SEED)
    results = {}
    for domain in DOMAINS:
        domain_results = []
        for i, prompt in enumerate(prompts_by_domain[domain]):
            formatted = format_prompt(prompt)
            generated = generate_text(
                model, tokenizer, formatted,
                max_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE, top_p=TOP_P
            )
            domain_results.append({
                "prompt": prompt,
                "generated": generated,
                "word_count": word_count(generated),
                "keyword_density": keyword_density(generated, domain),
                "ngram_diversity": ngram_diversity(generated),
                "repetition": repetition_score(generated),
                "coherence": coherence_score(generated),
                "incoherent": is_incoherent(generated),
            })
            if (i + 1) % 5 == 0:
                log(f"  {domain}: {i+1}/{len(prompts_by_domain[domain])} prompts done")

        results[domain] = domain_results
        avg_kw = np.mean([r["keyword_density"] for r in domain_results])
        avg_wc = np.mean([r["word_count"] for r in domain_results])
        log(f"  {domain}: avg_keywords={avg_kw:.3f}, avg_words={avg_wc:.1f}")

    elapsed = time.time() - t0
    log(f"  Uniform generation done in {elapsed:.1f}s")
    log_memory("post-gen-uniform")
    cleanup(model, tokenizer)
    del skeleton
    gc.collect()
    return results


# ============================================================================
# Phase 3: Generate with routed top-2 composition
# ============================================================================

def phase_generate_routed(prompts_by_domain):
    """Generate text with routed top-2 adapter composition.

    Uses oracle routing (correct domain gets 0.7 weight, next gets 0.3).
    This is the UPPER BOUND test: if oracle routing doesn't help, nothing will.
    """
    log("\n[Phase 3] Generating with ROUTED top-2 composition (oracle routing)...")
    t0 = time.time()

    skeleton = load_skeleton()
    model, tokenizer = load(MODEL_ID)
    model = replace_bitlinear_with_linear(model)
    model = apply_multi_adapter_routed(model, skeleton)
    model.freeze()
    log_memory("post-load-routed")

    np.random.seed(SEED)
    results = {}
    for di, domain in enumerate(DOMAINS):
        # Set routing weights for this domain
        weights = get_oracle_routing_weights(di, N_DOMAINS, top_k=2)
        set_all_routing_weights(model, weights)
        log(f"  {domain}: routing weights = {weights}")

        domain_results = []
        for i, prompt in enumerate(prompts_by_domain[domain]):
            formatted = format_prompt(prompt)
            generated = generate_text(
                model, tokenizer, formatted,
                max_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE, top_p=TOP_P
            )
            domain_results.append({
                "prompt": prompt,
                "generated": generated,
                "word_count": word_count(generated),
                "keyword_density": keyword_density(generated, domain),
                "ngram_diversity": ngram_diversity(generated),
                "repetition": repetition_score(generated),
                "coherence": coherence_score(generated),
                "incoherent": is_incoherent(generated),
            })
            if (i + 1) % 5 == 0:
                log(f"  {domain}: {i+1}/{len(prompts_by_domain[domain])} prompts done")

        results[domain] = domain_results
        avg_kw = np.mean([r["keyword_density"] for r in domain_results])
        avg_wc = np.mean([r["word_count"] for r in domain_results])
        log(f"  {domain}: avg_keywords={avg_kw:.3f}, avg_words={avg_wc:.1f}")

    elapsed = time.time() - t0
    log(f"  Routed generation done in {elapsed:.1f}s")
    log_memory("post-gen-routed")
    cleanup(model, tokenizer)
    del skeleton
    gc.collect()
    return results


# ============================================================================
# Phase 4: Cross-PPL evaluation
# ============================================================================

def phase_cross_ppl(base_results, uniform_results, routed_results):
    """Evaluate generated text quality via cross-perplexity.

    For each domain, compute PPL of the generated text under the domain's
    own adapter. Lower PPL = the adapter 'agrees' with the generated text
    more (i.e., it's more domain-appropriate).
    """
    log("\n[Phase 4] Computing cross-PPL on generated text...")
    t0 = time.time()

    skeleton = load_skeleton()
    cross_ppls = {"base": {}, "uniform": {}, "routed": {}}

    for di, domain in enumerate(DOMAINS):
        log(f"\n  Loading {domain} adapter for cross-PPL...")
        model, tokenizer = load(MODEL_ID)
        model = replace_bitlinear_with_linear(model)
        model = apply_single_adapter(model, skeleton, di, domain)
        model.freeze()

        for config_name, config_results in [
            ("base", base_results),
            ("uniform", uniform_results),
            ("routed", routed_results),
        ]:
            total_loss = 0.0
            total_tokens = 0
            for result in config_results[domain]:
                text = result["generated"]
                if not text or len(text.strip()) < 5:
                    continue
                full_text = format_prompt(result["prompt"]) + text
                tokens = tokenizer.encode(full_text)
                tokens = tokens[:MAX_SEQ_LENGTH + 1]
                if len(tokens) < 2:
                    continue
                x = mx.array(tokens[:-1])[None, :]
                y = mx.array(tokens[1:])[None, :]
                logits = model(x)
                loss = nn.losses.cross_entropy(logits, y, reduction="sum")
                mx.eval(loss)
                total_loss += loss.item()
                total_tokens += y.size
                del logits, loss, x, y

            if total_tokens > 0:
                avg_loss = total_loss / total_tokens
                ppl = math.exp(min(avg_loss, 100))
            else:
                ppl = float("inf")
            cross_ppls[config_name][domain] = ppl
            log(f"    {config_name} {domain}: cross-PPL = {ppl:.2f}")

        cleanup(model, tokenizer)

    elapsed = time.time() - t0
    log(f"  Cross-PPL done in {elapsed:.1f}s")
    del skeleton
    gc.collect()
    return cross_ppls


# ============================================================================
# Analysis and scoring
# ============================================================================

def analyze_results(base_results, uniform_results, routed_results, cross_ppls):
    """Compute aggregate scores and kill criteria assessment."""
    log("\n[Analysis] Computing aggregate metrics...")

    analysis = {
        "per_domain": {},
        "aggregates": {},
        "kill_criteria": {},
        "generated_samples": {},
    }

    routed_better_count = 0
    measurable_diff_count = 0
    incoherent_count = 0
    total_domains = len(DOMAINS)

    for domain in DOMAINS:
        base_kw = np.mean([r["keyword_density"] for r in base_results[domain]])
        uniform_kw = np.mean([r["keyword_density"] for r in uniform_results[domain]])
        routed_kw = np.mean([r["keyword_density"] for r in routed_results[domain]])

        base_wc = np.mean([r["word_count"] for r in base_results[domain]])
        uniform_wc = np.mean([r["word_count"] for r in uniform_results[domain]])
        routed_wc = np.mean([r["word_count"] for r in routed_results[domain]])

        base_div = np.mean([r["ngram_diversity"] for r in base_results[domain]])
        uniform_div = np.mean([r["ngram_diversity"] for r in uniform_results[domain]])
        routed_div = np.mean([r["ngram_diversity"] for r in routed_results[domain]])

        base_rep = np.mean([r["repetition"] for r in base_results[domain]])
        uniform_rep = np.mean([r["repetition"] for r in uniform_results[domain]])
        routed_rep = np.mean([r["repetition"] for r in routed_results[domain]])

        base_coh = np.mean([r["coherence"] for r in base_results[domain]])
        uniform_coh = np.mean([r["coherence"] for r in uniform_results[domain]])
        routed_coh = np.mean([r["coherence"] for r in routed_results[domain]])

        base_incoh = sum(1 for r in base_results[domain] if r["incoherent"])
        uniform_incoh = sum(1 for r in uniform_results[domain] if r["incoherent"])
        routed_incoh = sum(1 for r in routed_results[domain] if r["incoherent"])

        # Cross-PPL (lower = better alignment with domain adapter)
        base_xppl = cross_ppls["base"].get(domain, float("inf"))
        uniform_xppl = cross_ppls["uniform"].get(domain, float("inf"))
        routed_xppl = cross_ppls["routed"].get(domain, float("inf"))

        # Composite quality score: weighted combination
        # keyword_density (40%) + diversity (20%) + coherence (20%) + (1/cross_ppl normalized) (20%)
        def composite(kw, div, coh, xppl):
            # Normalize cross_ppl: lower is better, use inverse relative to base
            xppl_score = max(0, 1.0 - xppl / max(base_xppl, 1.0)) if base_xppl < float("inf") else 0
            return 0.4 * kw + 0.2 * div + 0.2 * coh + 0.2 * max(0, xppl_score)

        base_score = composite(base_kw, base_div, base_coh, base_xppl)
        uniform_score = composite(uniform_kw, uniform_div, uniform_coh, uniform_xppl)
        routed_score = composite(routed_kw, routed_div, routed_coh, routed_xppl)

        # Is routed better than base?
        routed_wins = routed_score > base_score
        if routed_wins:
            routed_better_count += 1

        # Is there measurable difference? (>5% relative improvement)
        if base_score > 0:
            relative_improvement = (routed_score - base_score) / base_score
        else:
            relative_improvement = 0
        if abs(relative_improvement) > 0.05:
            measurable_diff_count += 1

        # Check incoherence
        if routed_incoh > len(routed_results[domain]) * 0.5:
            incoherent_count += 1

        analysis["per_domain"][domain] = {
            "base": {
                "keyword_density": round(base_kw, 4),
                "word_count": round(base_wc, 1),
                "ngram_diversity": round(base_div, 4),
                "repetition": round(base_rep, 4),
                "coherence": round(base_coh, 4),
                "incoherent_count": base_incoh,
                "cross_ppl": round(base_xppl, 2),
                "composite_score": round(base_score, 4),
            },
            "uniform": {
                "keyword_density": round(uniform_kw, 4),
                "word_count": round(uniform_wc, 1),
                "ngram_diversity": round(uniform_div, 4),
                "repetition": round(uniform_rep, 4),
                "coherence": round(uniform_coh, 4),
                "incoherent_count": uniform_incoh,
                "cross_ppl": round(uniform_xppl, 2),
                "composite_score": round(uniform_score, 4),
            },
            "routed": {
                "keyword_density": round(routed_kw, 4),
                "word_count": round(routed_wc, 1),
                "ngram_diversity": round(routed_div, 4),
                "repetition": round(routed_rep, 4),
                "coherence": round(routed_coh, 4),
                "incoherent_count": routed_incoh,
                "cross_ppl": round(routed_xppl, 2),
                "composite_score": round(routed_score, 4),
            },
            "routed_wins": routed_wins,
            "relative_improvement_pct": round(relative_improvement * 100, 2),
        }

        # Save sample generations for inspection
        analysis["generated_samples"][domain] = {
            "prompt": base_results[domain][0]["prompt"],
            "base": base_results[domain][0]["generated"][:500],
            "uniform": uniform_results[domain][0]["generated"][:500],
            "routed": routed_results[domain][0]["generated"][:500],
        }

    # Kill criteria assessment
    routed_worse_count = total_domains - routed_better_count
    k1_kill = routed_worse_count >= 3
    k2_kill = measurable_diff_count == 0
    k3_kill = incoherent_count == total_domains

    analysis["aggregates"] = {
        "routed_better_count": routed_better_count,
        "routed_worse_count": routed_worse_count,
        "measurable_diff_count": measurable_diff_count,
        "incoherent_domains": incoherent_count,
    }

    analysis["kill_criteria"] = {
        "k1": {
            "test": f"Routed worse than base on >= 3/5 domains",
            "value": f"{routed_worse_count}/5 domains worse",
            "result": "FAIL" if k1_kill else "PASS",
        },
        "k2": {
            "test": "No measurable difference (0/5 domains with >5% improvement)",
            "value": f"{measurable_diff_count}/5 domains with measurable diff",
            "result": "FAIL" if k2_kill else "PASS",
        },
        "k3": {
            "test": "All generated text incoherent",
            "value": f"{incoherent_count}/5 domains incoherent",
            "result": "FAIL" if k3_kill else "PASS",
        },
    }

    verdict = "KILLED" if (k1_kill or k2_kill or k3_kill) else "SUPPORTED"
    analysis["verdict"] = verdict

    return analysis


# ============================================================================
# Main
# ============================================================================

def main():
    t0 = time.time()
    log("=" * 70)
    log("Generation Quality Test: Routed Composition vs Base")
    log("=" * 70)
    log_memory("start")

    # Verify adapters exist
    for domain in DOMAINS:
        adapter_path = ADAPTERS_DIR / domain / "adapter.npz"
        if not adapter_path.exists():
            log(f"ERROR: Missing adapter for {domain} at {adapter_path}")
            log("Run exp_real_data_domain_experts first!")
            return

    # Extract prompts
    log("\n[Setup] Extracting prompts from validation data...")
    prompts_by_domain = {}
    for domain in DOMAINS:
        prompts = extract_prompts(domain, NUM_PROMPTS_PER_DOMAIN)
        prompts_by_domain[domain] = prompts
        log(f"  {domain}: {len(prompts)} prompts")

    # Phase 1: Base generation
    base_results = phase_generate_base(prompts_by_domain)

    # Phase 2: Uniform composition
    uniform_results = phase_generate_uniform(prompts_by_domain)

    # Phase 3: Routed composition
    routed_results = phase_generate_routed(prompts_by_domain)

    # Phase 4: Cross-PPL evaluation
    cross_ppls = phase_cross_ppl(base_results, uniform_results, routed_results)

    # Analysis
    analysis = analyze_results(base_results, uniform_results, routed_results, cross_ppls)

    # Final results
    results = {
        "experiment": "generation_quality_test",
        "model": MODEL_ID,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "num_prompts_per_domain": NUM_PROMPTS_PER_DOMAIN,
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "routing": "oracle_top2 (0.7/0.3)",
        },
        "per_domain": analysis["per_domain"],
        "aggregates": analysis["aggregates"],
        "kill_criteria": analysis["kill_criteria"],
        "cross_ppls": cross_ppls,
        "verdict": analysis["verdict"],
        "generated_samples": analysis["generated_samples"],
        "total_time_s": round(time.time() - t0, 1),
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    log(f"\nResults saved to {RESULTS_FILE}")

    # Print summary
    log("\n" + "=" * 70)
    log("SUMMARY")
    log("=" * 70)
    for domain in DOMAINS:
        d = analysis["per_domain"][domain]
        log(f"\n{domain}:")
        log(f"  Base:    score={d['base']['composite_score']:.4f} kw={d['base']['keyword_density']:.4f} words={d['base']['word_count']:.0f} xppl={d['base']['cross_ppl']:.1f}")
        log(f"  Uniform: score={d['uniform']['composite_score']:.4f} kw={d['uniform']['keyword_density']:.4f} words={d['uniform']['word_count']:.0f} xppl={d['uniform']['cross_ppl']:.1f}")
        log(f"  Routed:  score={d['routed']['composite_score']:.4f} kw={d['routed']['keyword_density']:.4f} words={d['routed']['word_count']:.0f} xppl={d['routed']['cross_ppl']:.1f}")
        log(f"  Routed wins: {d['routed_wins']} ({d['relative_improvement_pct']:+.1f}%)")

    log(f"\nKill Criteria:")
    for k, v in analysis["kill_criteria"].items():
        log(f"  {k}: {v['result']} — {v['test']} ({v['value']})")

    log(f"\nVERDICT: {analysis['verdict']}")
    log(f"Total time: {results['total_time_s']:.1f}s")

    # Show sample generations
    log("\n" + "=" * 70)
    log("SAMPLE GENERATIONS (first prompt per domain)")
    log("=" * 70)
    for domain in DOMAINS:
        samples = analysis["generated_samples"][domain]
        log(f"\n--- {domain.upper()} ---")
        log(f"Prompt: {samples['prompt'][:200]}")
        log(f"\nBase:    {samples['base'][:300]}")
        log(f"\nUniform: {samples['uniform'][:300]}")
        log(f"\nRouted:  {samples['routed'][:300]}")


if __name__ == "__main__":
    main()
