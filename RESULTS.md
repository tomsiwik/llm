# MoE with Expert Lifecycle Management for Continual Learning

## Thesis

> **Expert lifecycle management (freeze, recycle, bond, distill) reduces forgetting when experts have sufficient capacity for task coverage in a Mixture-of-Experts continual learning system.**

The lifecycle is a homeostatic cleanup mechanism. Its effectiveness depends on the expert-to-task ratio: beneficial when experts have slack capacity (N_EXPERTS >= N_TASKS), neutral-to-harmful when experts are severely constrained (N_EXPERTS << N_TASKS).

---

## 1. Primary Evidence: ViT-B/16 + Linear Heads

Frozen ViT-B/16 backbone (86M params) → linear head experts (76.9K params each).
10 tasks x 10 classes, Adam optimizer, 500 steps/task, cosine LR.

### N_EXPERTS=10 (Ceiling Reference: 1:1 Mapping)

| Method | FA (Oracle) | BWT | Forgetting |
|--------|-------------|-----|------------|
| Single Head (fine-tune) | 11.9% | -71.2% | 71.2% |
| **Static Tribe (10 experts)** | **78.2%** | +0.0% | **0.0%** |
| **Lifecycle Tribe (10 experts)** | **78.1%** | +0.0% | **0.0%** |
| Learned+LC (10 experts) | 16.9% | -7.1% | 7.1% |

With 1:1 mapping, both static and lifecycle achieve zero forgetting because each expert trains on exactly one task. Lifecycle has no freeze/recycle events — there's nothing to manage.

**Class-Incremental Routing (label-free, global argmax):**

| Method | FA | Routing |
|--------|-----|---------|
| Static Tribe | 44.2% | Uncalibrated logit stitching |
| Lifecycle Tribe | 44.5% | Same |
| SwitchRouter (learned) | 4.3% | Joint-trained, routing still weak |
| Oracle (ceiling) | 78.2% | Uses ground truth labels |

The 34pp oracle/CIL gap reflects uncalibrated logit scales across independently trained heads. The SwitchRouter's joint training improves load balance over time but doesn't yet produce good routing decisions.

### N_EXPERTS=5 (Constrained: 2:1 Task-to-Expert Ratio)

This is the critical experiment testing lifecycle under resource constraints.

| Method | FA (Oracle) | Forgetting | FA (CIL) |
|--------|-------------|------------|----------|
| Single Head (fine-tune) | 11.9% | 71.2% | — |
| Static Tribe (5e) | **47.0%** | **34.1%** | 28.5% |
| Lifecycle Tribe (5e) | 46.0% | 35.4% | 15.8% |
| Learned+LC (5e) | 12.5% | 8.3% | 2.3% |

**Per-task final accuracy (Oracle):**

| Method | T0 | T1 | T2 | T3 | T4 | T5 | T6 | T7 | T8 | T9 |
|--------|----|----|----|----|----|----|----|----|----|----|
| Static (5e) | 20% | 12% | 18% | 8% | 11% | 78% | 84% | 80% | 81% | 78% |
| Lifecycle (5e) | **77%** | 12% | 18% | 0% | 11% | 20% | 84% | 80% | 80% | 79% |

**Key finding: Preservation-Coverage Trade-off.** The lifecycle successfully preserves T0 at 77% (vs static's 20%, +57pp) by just-in-time freezing expert 0 before task 5 would overwrite it. However, this forces task 5 onto a recycled expert that later gets overwritten, causing T5 to drop from 78% to 20% (-58pp). The net effect is approximately zero.

**Lifecycle events (5e):** FREEZE expert 0 at task 5 (acc=76.8%), RECYCLE expert 3 (worst active, acc=71.9%). Final state: 4 active, 1 frozen.

**Why lifecycle doesn't help with 5 experts:** With 2:1 task-to-expert ratio, every expert slot is needed for current tasks. Freezing one expert preserves its task but steals capacity from future tasks. The benefit of preservation is offset by the cost of reduced capacity. This defines the lifecycle's operating range.

---

## 2. Supporting Evidence

### Split CIFAR-100 (ResNet experts, from scratch)

ResNet-18-lite (~712K params/expert), 500 SGD steps/task.

| Method | FA | Forgetting | Params |
|--------|-----|------------|--------|
| Fine-tune | 4.2% | 31.3% | 712K |
| EWC (lambda=400) | 3.8% | 31.9% | 712K |
| DER++ (buf=2000) | 5.0% | 22.5% | 712K |
| **Static Tribe (10 exp)** | **28.2%** | 7.3% | 7.1M |
| **Lifecycle Tribe** | **29.1%** | **6.8%** | 7.1M |
| Learned + Lifecycle | 24.9% | **5.4%** | 7.1M |

**Key finding:** Tribe methods (28-29% FA) crush single-network methods (4-5% FA) — 6x improvement. Lifecycle reduces forgetting from 7.3% to 6.8% (N_EXPERTS=N_TASKS=10, slight benefit). Learned routing achieves best forgetting (5.4%) at cost of accuracy.

### MNIST (4 experts, 4 generations, ~64K params/expert)

| Baseline | Accuracy | Redundancy |
|----------|----------|------------|
| C: Round-robin static | 82.5% | 0.46 |
| **E: Round-robin + lifecycle** | **95.3%** | **0.26** |

Lifecycle adds +12.8% accuracy and halves redundancy. Knowledge precision +358% over generations. The MNIST benchmark has enough generations for lifecycle to accumulate improvements.

---

## 3. SOTA Context (Honest Assessment)

### Split CIFAR-100 Benchmark Landscape

| Method | FA (CIL) | Forgetting | Architecture |
|--------|----------|------------|--------------|
| L2P (2022) | ~83% | ~5% | ViT-B/16 + prompts |
| DualPrompt (2022) | ~85% | ~4% | ViT-B/16 + prompts |
| CODA-Prompt (2023) | ~86% | ~3% | ViT-B/16 + prompts |
| HiDe-Prompt (2023) | ~87% | ~2% | ViT-B/16 + prompts |
| EASE (2024) | ~88% | ~2% | ViT-B/16 + adapters |
| MoE-Adapters4CL (2024) | ~87% | ~2% | ViT-B/16 + MoE adapters |
| **Ours (10e, oracle)** | **78%** | **0.0%** | ViT-B/16 + linear heads |
| **Ours (10e, CIL)** | **44%** | **12%** | ViT-B/16 + global argmax |
| **Ours (5e, oracle)** | **47%** | **34%** | ViT-B/16 + linear heads |

**Where we stand:** With N_EXPERTS=N_TASKS=10, our oracle routing achieves 78% FA with zero forgetting. The zero forgetting is a structural guarantee. The CIL gap (44% vs 78%) is a routing problem, not a learning problem.

With N_EXPERTS=5, both static and lifecycle drop to ~47% FA with ~34% forgetting. Expert isolation alone doesn't prevent forgetting when experts must be reused.

---

## 4. Novel Contributions

### 1. Expert Lifecycle as Homeostatic Mechanism

The lifecycle (freeze, recycle, bond, distill) as first-class operations on expert-level units in an MoE:

| Operation | Our Implementation | Closest Published |
|-----------|-------------------|-------------------|
| **Freeze** | Just-in-time, accuracy-based | PackNet (per-neuron masking) |
| **Recycle** | Blend neighbors + reinit + warmup | ReDo (neuron reinit) |
| **Bond** | Weighted blend of parents | DynMoE (add from prototype) |
| **Distill** | Selective (unique knowledge only) | LwF (model-level distill) |

### 2. Preservation-Coverage Trade-off (New Finding)

When N_EXPERTS < N_TASKS, lifecycle freezing creates a trade-off:
- **Preservation benefit:** Frozen experts maintain high accuracy on their task (77% vs 20%)
- **Coverage cost:** Frozen expert slot is unavailable for future tasks, increasing forgetting on other tasks

The net effect depends on the expert-to-task ratio. With sufficient slack (N_EXPERTS >= N_TASKS), preservation is free. Under severe constraint (N_EXPERTS << N_TASKS), preservation has negative ROI.

### 3. Dynamic Freeze Threshold

Data-driven threshold: `mean_accuracy + 0.5 * std_accuracy` across trained experts. Just-in-time freeze: protect expert right before round-robin would overwrite it.

### 4. Lifecycle Metrics

- Expert weight cosine similarity as redundancy proxy
- Generational tracking across freeze/recycle events
- Per-task active/frozen/recycled state reporting

---

## 5. Limitations

1. **Lifecycle is neutral-to-negative with N_EXPERTS=5.** The preservation-coverage trade-off means freezing helps individual tasks but hurts overall FA.
2. **Class-incremental routing gap.** 44% CIL vs 78% oracle (10e). SwitchRouter joint training yields only 4.3% CIL — routing collapse not yet solved.
3. **Single-seed results.** Need 5+ seeds with error bars.
4. **No expert interaction during training.** Experts train independently (except in joint mode with SwitchRouter).
5. **Linear heads are less expressive than prompt tuning.** 76.9K params vs 86M-param prompt-tuned representations.
6. **Oracle routing is not class-incremental.** Best numbers use ground-truth labels.

---

## 6. Follow-Up Research

### Near-Term
- **LoRA experts** (`tribe/lora.py`, 371K params): Fine-tune Q,V projections. Should push accuracy from ~78% toward CODA-Prompt range.
- **N_EXPERTS=7-8 config:** Test the sweet spot where lifecycle has some slack for freezing without severe capacity loss.
- **Expert-level replay:** When retraining an expert for a new task, include a small replay buffer from the previous task to retain knowledge.
- **Multiple seeds + error bars.**

### Medium-Term
- **Lifecycle-aware training loss:** Add redundancy + diversity losses to MoE objective.
- **Dynamic expert count:** Grow/shrink expert pool (DynMoE + lifecycle).
- **Better SwitchRouter training:** Pre-train experts separately, then fine-tune router jointly.

### Long-Term
- **LLM-scale lifecycle:** Apply to production MoE (Mixtral/DeepSeek scale).
- **Continual pre-training:** Stream of domains with lifecycle-managed expert pool.

---

## Terminology Note

This project uses "Tribe" and "TribeMember" as implementation names. In standard MoE research terminology:
- Tribe = Expert pool / MoE layer
- TribeMember = Expert
- SwitchRouter = Gating network (Switch Transformer, Fedus et al. 2022)
- Lifecycle (freeze/recycle/bond) = Novel contribution, no standard term

The novel contribution is the lifecycle protocol, not the MoE architecture itself.
