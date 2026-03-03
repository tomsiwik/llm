# Self-Routing LoRA Atoms for Continual Learning on LLMs

> **Goal:** Evolve the tribe lifecycle system from vision-only (CIFAR-100/ViT) to language models,
> using self-routing LoRA atoms (MoRAM-style) where each adapter atom routes itself based on
> input relevance — true tribe behavior where experts self-select their work.

---

## Motivation

Our CIFAR-100 results show the lifecycle works (MNIST +12.8%, ResNet 6x over baselines) but
the vision benchmark has hit diminishing returns:

- **10e/10t:** lifecycle is a no-op (0% forgetting by construction)
- **5e/10t:** preservation-coverage trade-off makes it net zero
- **CIL routing:** 44% vs 78% oracle — routing is the bottleneck, not learning

The EASE paper (CVPR 2024, 91.5% on CIFAR-100) shows that running ALL adapters and using
prototype classification eliminates the routing problem entirely. But for LLMs:

- **EASE doesn't scale** — running 10+ full adapter sets per token is expensive for sequences
- **Self-routing scales** — each atom decides per-token whether to fire, giving sparse efficiency
- **LLMs are the real target** — continual learning on language domains (code, math, legal, etc.)

The MoRAM insight: decompose LoRA into rank-1 atoms where the A vector serves as both
projection AND routing key. Each atom literally "knows what it's good at" and self-selects.
This IS tribe behavior — no external router deciding from outside.

---

## Architecture

```
Frozen LLM backbone (SmolLM-135M → Llama-3.2-1B → larger)
    │
    ├─ Layer 0: Q_proj ← SelfRoutingLoRALinear (n_atoms, top_k)
    │           V_proj ← SelfRoutingLoRALinear (n_atoms, top_k)
    ├─ Layer 1: Q_proj ← ...
    │           V_proj ← ...
    │   ...
    └─ Layer N: Q_proj ← ...
                V_proj ← ...

Per SelfRoutingLoRALinear:
    atom_A: (n_atoms, d_in)   — key vectors (routing + projection)
    atom_B: (n_atoms, d_out)  — value vectors (LoRA delta)

    Forward:
        base_out = x @ frozen_weight.T           # frozen linear
        projections = x @ atom_A.T               # (batch, seq, n_atoms)
        scores = |projections| / temperature      # routing relevance
        weights = softmax(scores)                 # soft routing (peaky)
        delta = (weights * projections) @ atom_B  # gated LoRA output
        return base_out + delta

    Top-k (optional, for efficiency):
        mask = top_k_mask(scores, k)              # hard selection
        ste_weights = mask + weights - stop_grad(weights)  # STE trick
        delta = (ste_weights * projections) @ atom_B
```

### Parameter Budget

SmolLM-135M: hidden=576, 30 layers, Q=(576,576), V=(192,576)

| Config | Atoms/layer | Params/Q atom | Params/V atom | Total atoms | Total params |
|--------|-------------|---------------|---------------|-------------|--------------|
| Small  | 16          | 576+576=1,152 | 576+192=768   | 960         | ~0.9M        |
| Medium | 32          | 1,152         | 768           | 1,920       | ~1.8M        |
| Large  | 64          | 1,152         | 768           | 3,840       | ~3.7M        |

Compare: full model is 135M params. Even "Large" atoms are 2.7% overhead.

---

## Phase 1: Self-Routing LoRA Atom Module

**File:** `tribe/lora_atom.py`

```python
class SelfRoutingLoRALinear(nn.Module):
    """Drop-in replacement for nn.Linear with self-routing LoRA atoms.

    Each atom has:
    - A row (d_in,): key vector — determines routing AND projection
    - B row (d_out,): value vector — the LoRA delta contribution

    The projection x @ a_i simultaneously:
    1. Measures how relevant this atom is for the input (|projection|)
    2. Computes the LoRA contribution (projection * b_i)
    """
```

Key design decisions:
- **Soft routing initially** (temperature-controlled softmax), hard top-k later
- **Atom activation tracking**: per-atom, per-domain activation counts for analysis
- **Compatible with MLX nn.Module**: freeze/unfreeze for selective training
- **B initialized to zero**: standard LoRA practice (initial adapter = identity)

### Lifecycle Integration Points

Each atom maps to a potential TribeMember:
- **Freeze:** lock A and B, but atom STILL fires when input aligns (self-routing!)
- **Clone:** copy A and B to new slot — inherits parent's "knowledge" and routing key
- **Bond:** merge two atoms' A and B vectors (or use SVD delta merge)
- **Recycle:** reinitialize A randomly (new routing key), zero B (clean slate)
- **Wither:** atom with near-zero activation across all domains gets recycled

---

## Phase 2: LLM Backbone Utilities

**File:** `tribe/llm.py`

- `load_backbone(model_name)` — load via mlx-lm, return (model, tokenizer)
- `patch_with_atoms(model, n_atoms, top_k, targets)` — replace target Linear layers
- `freeze_base(model)` — freeze all, unfreeze only atom params
- `compute_perplexity(model, tokenizer, texts, max_tokens)` — per-domain evaluation
- `get_atom_stats(model)` — extract activation statistics across all patched layers
- `get_atom_params(model)` — extract trainable atom parameters for optimizer

Targets: Q and V projections in self-attention (standard LoRA targets).
Optional: gate_proj in MLP (for richer adaptation).

---

## Phase 3: Toy LLM Benchmark

**File:** `bench_llm_toy.py`

### Setup
- Backbone: SmolLM-135M (frozen, ~135M params)
- Atoms: 32 per target, 2 targets (Q, V) per layer, 30 layers = 1,920 atoms (~1.8M params)
- Top-k: start with soft (temperature=0.1), then test k=8, k=4

### Sequential Domains (3-4 tasks)
1. **Wikipedia** — general English prose (wikitext-2)
2. **Python code** — code completion/understanding
3. **Math** — mathematical reasoning text

Each domain: ~500 training sequences, 100 eval sequences.
Sequence length: 128-256 tokens (short for fast iteration).

### Training Protocol
- Optimizer: Adam (lr=1e-3 for atoms, standard for LoRA)
- Steps: 200-500 per domain
- Batch size: 8-16 sequences
- Loss: cross-entropy next-token prediction

### Evaluation (after each domain)
- Per-domain perplexity on held-out eval set
- Forgetting: perplexity increase on previously seen domains
- Atom activation heatmap: which atoms fire for which domain
- Sparsity: effective number of atoms per token (entropy of routing weights)

### Expected Outcomes
1. **Atoms specialize**: wiki atoms fire on wiki text, code atoms fire on code
2. **Low forgetting**: frozen atoms preserve old domain knowledge
3. **Self-routing works**: activation patterns show domain-specific clustering
4. **Baseline comparison**: full fine-tune (catastrophic forgetting) vs atoms (preservation)

---

## Phase 4: Lifecycle on Atoms

After Phase 3 validates that self-routing works, add lifecycle management:

- **Auto-freeze**: after domain training, freeze atoms whose activation is highly domain-specific
  and whose contribution exceeds threshold
- **Clone-and-deploy**: when new domain arrives, clone the most relevant frozen atom
  (highest activation on new domain's initial samples) into a fresh active slot
- **Wither detection**: atoms with < 1% activation across all domains get recycled
- **Bond**: two atoms with > 80% activation overlap get merged, freeing a slot

### Clone-and-Deploy (the user's idea)
```
Frozen atom (domain A expert)
    │ clone
    ▼
New active atom (inherits A's key + value)
    │ train on domain B
    ▼
Either:
  (a) Thrives → new expert, but needs HIGHER freeze threshold than parent
      (lineage-aware: gen+1 experts must prove MORE to earn freeze)
  (b) Withers → recycled (wasn't useful for domain B)
  (c) Bonds → merges with nearby atom to expand coverage

Anti-dynasty rule: max frozen atoms = n_atoms / 3 per layer
Clone cooldown: frozen atom can only be cloned once per N domains
```

---

## Phase 5: Scale to Real LLM Benchmarks

- Backbone: Llama-3.2-1B or Phi-2
- Benchmark: TRACE (8 NLP tasks) or domain-incremental from The Pile
- Compare: LoRA fine-tune, EWC, O-LoRA, InfLoRA
- Full lifecycle: freeze + clone + bond + wither
- Multiple seeds + error bars

---

## Files

| File | Status | Purpose |
|------|--------|---------|
| `PLAN.md` | This file | Master plan |
| `tribe/lora_atom.py` | New | SelfRoutingLoRALinear module |
| `tribe/llm.py` | New | LLM backbone load/patch/evaluate |
| `bench_llm_toy.py` | New | Toy benchmark (SmolLM + 3 domains) |
| `tribe/core.py` | Adapt later | TribeMember wraps LoRA atoms |
| `tribe/__init__.py` | Update | Export new modules |

### Dependencies
```bash
uv run --with mlx,mlx-lm,datasets python bench_llm_toy.py
```

---

## Success Criteria

### Phase 3 (Toy)
- [ ] Atoms show domain-specific activation patterns (heatmap)
- [ ] Per-domain perplexity improves during training
- [ ] Forgetting < 20% perplexity increase on old domains after new domain
- [ ] Self-routing is sparser than uniform (effective atoms < n_atoms/2)

### Phase 4 (Lifecycle)
- [ ] Clone-and-deploy reduces forgetting vs no-clone baseline
- [ ] Frozen atoms maintain old domain performance
- [ ] Withering removes genuinely useless atoms
- [ ] Bond reduces redundancy without hurting accuracy

### Phase 5 (Scale)
- [ ] Competitive with O-LoRA / InfLoRA on standard benchmarks
- [ ] Lifecycle provides measurable benefit over static atom pool
- [ ] Self-routing matches or beats external SwitchRouter
