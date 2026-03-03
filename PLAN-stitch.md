# Expert Stitching: Composable Model Capabilities via Compatibility Maps

## One-Line Summary

Train small specialist models independently, map their frozen experts into a shared capability space, compose them into systems that match large models — modularly and updatably.

---

## The Problem

Today's model improvement is monolithic. To add Rust expertise to a Python model, you retrain everything. Fine-tuning risks regression. LoRA merging destroys specialization. MoE requires joint training. There is no way to independently train specialists and compose them post-hoc with automatic routing.

**Who has this problem:**
- Enterprise teams running multiple fine-tuned models per department
- AI coding assistants wanting per-language specialization
- Edge deployment (can't run 7B, could run 1B + relevant experts)
- Model marketplaces (HuggingFace) with thousands of LoRA adapters and no way to compose them

---

## Key Insight

Our continual learning experiments reveal a scale-dependent finding:

| Scale | Forgetting | Expert Specialization | Composability |
|-------|-----------|----------------------|---------------|
| 135M | Severe (77%) | Strong — experts partition cleanly | High potential |
| 7B | Near-zero | Weak — base model already knows everything | Low value |

**The sweet spot is 1-3B**: forgetting is real enough that experts specialize during lifecycle, but models are still small enough to compose 5-10 of them cheaply. A composed system of N×1B specialists could match a single 7B model while being modular and updatable.

---

## Architecture

### Three Layers

```
┌─────────────────────────────────────────────────────────┐
│ Layer 3: COMPOSITION                                    │
│   FoX gates learn trust between cross-model experts     │
│   "How much should my model trust this external expert?"│
├─────────────────────────────────────────────────────────┤
│ Layer 2: COMPATIBILITY MAP                              │
│   Translates between model coordinate systems           │
│   "Where does Model B keep what Model A calls 'types'?" │
├─────────────────────────────────────────────────────────┤
│ Layer 1: EXPERT LIFECYCLE (existing)                    │
│   Freeze → Snapshot → FoX gate → Recycle                │
│   "What does this model know and where is it stored?"   │
└─────────────────────────────────────────────────────────┘
```

### Layer 1: Expert Lifecycle (Built)

Already implemented in our tribe system. Training produces:
- **Frozen experts** with routing keys (`weight_down`) — "what inputs I respond to"
- **Snapshot chains** with FoX gates — "how much to trust me vs. newer knowledge"
- **Importance scores** — "how critical am I to this model's performance"

These frozen experts are the **landmarks** on the map.

### Layer 2: Compatibility Map (Novel)

A learned translation layer between any two models' internal coordinate systems.

**What it is:** Two small projection matrices per model, mapping from model-specific space to a shared canonical space.

```
Model A's space ──→ map_A ──→ Shared Space ←── map_B ←── Model B's space
  (d_model_A)       (linear)    (d_shared)      (linear)    (d_model_B)
```

**How to build it:**

1. **Probe dataset**: Curated code snippets covering fundamental concepts
   - Types/type systems (shared across all languages)
   - Control flow (loops, conditionals, pattern matching)
   - Functions/closures (universal)
   - Data structures (lists, maps, trees)
   - Language-specific (DOM, borrow checker, JVM, etc.)

2. **Record activations**: Run each model on probe dataset, record which frozen experts fire for which concepts. This produces a sparse matrix:
   ```
   Expert_A_17: fires on [types, generics, interfaces] with scores [0.9, 0.7, 0.3]
   Expert_A_42: fires on [loops, iterators] with scores [0.8, 0.6]
   Expert_B_8:  fires on [types, annotations] with scores [0.85, 0.5]
   ```

3. **Align**: Learn projections that minimize distance between experts that fire on the same concepts:
   ```
   loss = Σ ||map_A(expert_A_i) - map_B(expert_B_j)||² × concept_overlap(i, j)
   ```
   This is a standard Procrustes alignment — closed-form solution, no training loop needed.

4. **Result**: A compatibility matrix showing which experts across models are substitutable, complementary, or incompatible.

**Cost**: One-time per model pair. Run probe dataset (~1000 examples), record activations, solve alignment. Hours, not days. The map itself is tiny — two matrices of shape `(d_model, d_shared)`.

### Layer 3: Composition (Novel)

Given a compatibility map, compose experts from different models:

**Expert Transfer:**
```python
def transfer_expert(expert_A, map_A, map_B):
    """Transfer a frozen expert from Model A's space to Model B's space."""
    # Project routing key to shared space, then to Model B's space
    shared_key = map_A(expert_A.weight_down)      # what it responds to (canonical)
    new_key = map_B.inverse(shared_key)            # same meaning in B's coordinates

    # Weights stay frozen — they encode knowledge, not routing
    # Only the routing key gets translated
    return FrozenExpert(
        weight_down=new_key,           # translated routing
        weight_up=expert_A.weight_up,  # original knowledge (frozen)
        gate_bias=-2.0,                # start with low trust, let it learn
    )
```

**Calibration:** After plugging in external experts, run a short calibration (100-500 steps) on a small mixed dataset. The FoX gates learn how much to trust each transplanted expert. No full retraining needed.

**Inference:**
```
Input → Base Model → At each adapter layer:
  1. Compute routing scores against LOCAL frozen experts (normal)
  2. Compute routing scores against TRANSPLANTED experts (via translated keys)
  3. Top-k selection across both pools
  4. FoX gates blend: trust_local vs trust_transplant
  5. Output = base + Σ(gate_i × expert_i_output)
```

---

## The Lifecycle of an Expert (Full Picture)

```
BIRTH          TRAINING        MASTERY         EMANCIPATION      TRANSFER
  │               │               │                │                │
  ▼               ▼               ▼                ▼                ▼
New hire →    Learns from    → Becomes good  → Outperforms    → Redundant HERE
(random       senior via       (importance      senior           but valuable
 init)        FoX gate         rises)           (gate>0.9)       ELSEWHERE
              (88% senior,                      Bakes senior's
               12% self)                        knowledge in,    Check compatibility
                                                becomes          map for offices
                                                independent      with matching gaps
                                                                        │
                                                                        ▼
                                                                 Transfer via map
                                                                 projection, plug
                                                                 into new model,
                                                                 FoX gate calibrates
                                                                 trust automatically
```

---

## Experiment Plan

### Phase 1: Validate Expert Specialization at 1-3B (Cost: ~$15)

**Goal:** Confirm that lifecycle produces cleanly specialized experts at this scale.

- Model: Qwen2.5-Coder-1.5B or CodeLlama-3B
- Adapter: LoRA rank=32 (higher rank → more room to specialize)
- Domains: Python, JavaScript, Rust (3 languages, clear boundaries)
- Training: 1000 steps/domain sequential with lifecycle
- Measure: Do frozen experts cluster by domain? (activation analysis)

**Success = experts clearly partition by language.** If they don't, composition won't work.

### Phase 2: Build Compatibility Map (Cost: ~$5)

**Goal:** Show that two independently trained models' experts can be aligned.

- Train Model A: Python-specialist (1.5B + LoRA + lifecycle)
- Train Model B: JavaScript-specialist (same base + LoRA + lifecycle)
- Build probe dataset: 500 code snippets covering shared concepts
- Record expert activations on probes
- Compute Procrustes alignment
- Measure: Do "types" experts from A and B align? Do "DOM" experts (JS-only) correctly show no match?

**Success = high cosine similarity for shared concepts, low for language-specific ones.**

### Phase 3: Cross-Model Composition (Cost: ~$10)

**Goal:** Transfer experts and show the composed system outperforms individuals.

- Take Model A (Python) and Model B (JavaScript)
- Transfer Model B's top JavaScript experts into Model A via compatibility map
- Calibrate FoX gates (500 steps on mixed Python+JS data)
- Evaluate: Does Model A+B_experts outperform Model A alone on JavaScript?
- Evaluate: Does Model A's Python performance survive?

**Success = JavaScript performance improves without Python regression.**

### Phase 4: Expert Marketplace Prototype (Cost: ~$20)

**Goal:** Demonstrate N-way composition and the marketplace concept.

- Train 5 specialist models (Python, JS, Rust, SQL, C++)
- Build pairwise compatibility maps (10 pairs, automated)
- Compose all 5 into one system
- Benchmark against single 7B model on all languages
- Measure: total params, inference cost, per-language performance

**Success = composed 5×1.5B system matches 7B on multilingual coding.**

---

## Why This Is Novel

| Existing Approach | Limitation | Our Solution |
|---|---|---|
| LoRA merging (TIES/DARE) | Averages weights, destroys specialization | Keeps experts separate, routes dynamically |
| Multi-adapter (PEFT) | Manual switching, no routing | Automatic routing via compatibility map |
| MoE (Mixtral) | Joint training required, fixed experts | Independent training, composable post-hoc |
| Model stitching | Layer-level, fragile, same architecture only | Expert-level, robust, any same-base model |
| Cross-lingual embeddings | Maps representations, not capabilities | Maps what a model CAN DO, not what it sees |

**The novel contribution in one sentence:** A self-describing expert format with learned compatibility maps that enables post-hoc composition of independently trained model capabilities.

---

## The Product Vision

```
┌─────────────────────────────────────────────┐
│           Expert Registry (Hub)             │
│                                             │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐    │
│  │ Python   │ │ Rust     │ │ SQL      │    │
│  │ Expert   │ │ Expert   │ │ Expert   │    │
│  │ Pack     │ │ Pack     │ │ Pack     │    │
│  │          │ │          │ │          │    │
│  │ map: ✓   │ │ map: ✓   │ │ map: ✓   │    │
│  │ gate: ✓  │ │ gate: ✓  │ │ gate: ✓  │    │
│  │ 50MB     │ │ 50MB     │ │ 50MB     │    │
│  └──────────┘ └──────────┘ └──────────┘    │
│                                             │
│  Download what you need. Compose locally.   │
│  Auto-calibrate in 5 minutes.               │
└─────────────────────────────────────────────┘

User: "I need Python + Rust for my systems project"
  → Download 2 expert packs (100MB total)
  → Plug into their 1.5B base model
  → Calibrate gates (5 min on their codebase)
  → Done: specialized coding assistant, updatable,
    fraction of 7B cost
```

**Expert packs are tiny** (just frozen LoRA weights + routing keys + compatibility map vectors). Megabytes, not gigabytes. Downloadable, shareable, composable.

---

## Risks and Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Experts don't specialize at 1-3B | Medium | Increase LoRA rank, more training steps, try PEER routing |
| Compatibility maps don't align | Low | Same-base models share feature space; Procrustes is proven |
| Calibration undoes transferred knowledge | Medium | Freeze transferred experts, only train gates |
| Composed system slower than single model | Low | Top-k routing limits active experts; same inference cost |
| Nobody needs this (7B is cheap enough) | Medium | Edge deployment, updatability, and modularity are real needs even if raw perf isn't |

---

## Relation to Existing Work

- **Tribe lifecycle** → produces the frozen experts (Layer 1)
- **PEER routing** → provides the routing infrastructure at scale
- **FoX gates** → enable trust calibration between local and foreign experts
- **Version tree snapshots** → the expert format that gets transferred
- **This plan** → adds the compatibility map (Layer 2) and composition protocol (Layer 3)

Everything we've built is a prerequisite. This is the application layer.

---

## Next Steps

1. Finish current 7B benchmark (in progress, ~30 min remaining)
2. Record the null result (no forgetting at 7B = validates the scale insight)
3. Run Phase 1 on same RunPod instance (1.5B model, ~2 hours)
4. If experts specialize → proceed to Phase 2-3
5. Write up as research paper: "Expert Stitching: Composable Model Capabilities via Learned Compatibility Maps"
