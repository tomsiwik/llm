# LoRA Composition: Why It Degrades LLMs But Improves Vision Models

## Complete Mathematical Analysis and Solutions

**Research date:** 2026-03-28
**NotebookLM notebook:** `72ef5131-c589-45ce-b100-cff44b064f64`
**Sources:** 40+ papers from deep web research + targeted arxiv fetches

---

## 1. THE FUNDAMENTAL ASYMMETRY: Vision vs Language

### Why Vision Model Composition Works

**Model Soups** (Wortsman et al., arXiv:2203.05482): Averaging weights of multiple fine-tuned CLIP/ViT models *improves* accuracy without increasing inference cost.

- ViT-G model soup: **90.94% top-1 on ImageNet** (new SOTA at time)
- Works for CLIP, ALIGN, ViT-G pre-trained on JFT

**The mathematical reason: Linear Mode Connectivity (LMC)**

When two models are fine-tuned from the same pretrained initialization, the interpolated model attains *at least* the accuracy of the endpoints. Formally:

```
L(alpha * W1 + (1-alpha) * W2) <= alpha * L(W1) + (1-alpha) * L(W2)
```

This holds when fine-tuned models lie in a **single low-error basin** of the loss landscape. The loss surface between them is approximately convex (no barrier).

**Why vision models have this property:**
1. **Flat loss landscape:** CLIP/ViT fine-tuning produces models in broad, flat minima. The loss is approximately linear between fine-tuned checkpoints.
2. **Task similarity:** Vision classification tasks (ImageNet variants, domain shifts) share substantial feature extraction structure. The task vectors point in similar directions.
3. **Overparameterization:** Vision models are heavily overparameterized relative to the fine-tuning task, creating wide basins where averaging stays within the basin.

**LoRA Soups for Diffusion:** Works because diffusion model adapters (style, subject, composition) modify largely orthogonal aspects of the generation process. Style affects global statistics; subject affects local features.

### Why LLM Composition Fails

**The core problem:** LLM tasks (math, code, medical, SQL) require *different reasoning pathways* through the same layers. When you average the weights, you get a model that does none of them well.

**Key evidence (arXiv:2506.13479 - "Pause Recycling LoRAs"):**
- Data-agnostic merging (parameter averaging, dynamic adapter selection) fails to logically integrate knowledge across disjoint fine-tuning datasets
- Models resort to **shallow pattern matching** rather than genuine compositional generalization
- LoRA has inherently **limited expressiveness** that makes composition fundamentally harder

**Mathematical explanation:**
- Vision tasks: Task vectors are approximately **co-linear** (pointing in similar directions in weight space) -> averaging stays in the basin
- Language tasks: Task vectors are **competing** (pulling weights in conflicting directions) -> averaging cancels critical updates

---

## 2. THE INTERFERENCE MECHANISM: Exact Mathematics

### The Merged Weight Formula

Given base model weights W_0 and two LoRA adapters:

```
W_merged = W_0 + B_1 @ A_1 + B_2 @ A_2
```

For input features h from task T1:

```
W_merged @ h_1 = (W_0 + B_1 @ A_1) @ h_1 + B_2 @ A_2 @ h_1
                = W_1 @ h_1              + [INTERFERENCE TERM]
```

**The interference term is `B_2 @ A_2 @ h_1`** -- task T2's adapter operating on task T1's features.

### Interference Magnitude (OSRM, arXiv:2505.22934)

The interference is quantified via Frobenius norm:

```
||A_2 @ H_1^T||_F
```

where H_1 is the feature matrix of task T1 data (k samples x n features).

**This measures how strongly task T2's down-projection A_2 amplifies the principal directions of task T1's feature distribution.**

### Why Nonlinearities Amplify Interference

**Attention layers:** When Q, K, V matrices are all adapted, the attention score becomes:

```
attn = softmax((W_Q + dQ) @ x @ (W_K + dK)^T @ x^T / sqrt(d))
```

The cross-terms `dQ_1 @ x @ dK_2^T @ x^T` create spurious attention patterns that don't correspond to any single task's learned attention distribution.

**MLP layers (SwiGLU/GELU):** The gating mechanism in SwiGLU:

```
output = SiLU(W_gate @ x) * (W_up @ x)
```

When W_gate and W_up are both perturbed by merged adapters, the gate and value paths become misaligned. Small perturbations to the gate can zero out or amplify value signals, creating catastrophic interference.

### Layer-Specific Interference

From research findings in this project (macro experiments):
- **Attention layers** are more interference-sensitive than FFN layers
- **Early layers** (embedding-adjacent) show higher interference because they affect all downstream computation
- **Middle layers** show the most task-specific divergence in adapter weights

---

## 3. ALL KNOWN SOLUTIONS (With Benchmark Numbers)

### A. Static Merging Methods (Zero Inference Cost)

#### Task Arithmetic (arXiv:2212.04089)
- Compute task vectors: tau_i = W_finetuned_i - W_pretrained
- Merge: W_merged = W_pretrained + lambda * sum(tau_i)
- **Result:** Baseline method, works for similar tasks, degrades for diverse tasks

#### TIES-Merging (arXiv:2306.01708)
Three-step process:
1. **Trim:** Zero out small-magnitude changes (keep top-k%)
2. **Elect sign:** Majority vote on sign conflicts per parameter
3. **Merge:** Average only the agreeing, significant parameters

**Benchmark:** DARE-TIES reaches online score of **60.0** on LLM merging competitions

#### DARE (arXiv:2311.03099) - "Language Models are Super Mario"
- Randomly prune (1-density) fraction of task vector values
- Rescale remaining by 1/density
- **Key insight:** Even dropping 90-99% of parameters works
- Combines with TIES for DARE-TIES

#### DO-Merging (arXiv:2505.15875) - Decouple and Orthogonalize
Separates parameters into magnitude and direction:
```
W = alpha * W_bar  (magnitude * normalized direction)
```

Applies orthogonal constraints data-free:
```
minimize sum_i,j (W_i + delta_i)^T (W_j + delta_j) + sum_i ||delta_i||_2
```

**Benchmark results:**
| Model | DO-Merging | TIES-Merging | Task Arithmetic |
|-------|-----------|-------------|-----------------|
| ViT-B/32 (8 tasks) | **77.88%** | 73.11% | 74.06% |
| ViT-L/14 (8 tasks) | **84.58%** | - | 79.80% |
| T5-base (8 tasks) | **80.9%** | - | - |
| LLaMA3-8B (6 tasks) | **87.11%** | - | 85.79% |

#### Twin-Merging (OpenReview)
**Benchmark:** Improves over Task Arithmetic by **28.34%**, TIES-Merging by **32.46%**, DARE by **30.56%** in absolute normalized score.

#### TC-LoRA - Tensorized Clustered LoRA (arXiv:2508.03999)
SVD-based LoRA merging with clustering.
- **+1.4% improvement on Phi-3**
- **+2.3% improvement on Mistral-7B**

### B. Orthogonal/Subspace Methods (Training-Time Constraint)

#### OSRM - Orthogonal Subspaces for Robust Merging (arXiv:2505.22934, ACL 2025)

**The key optimization:**
```
minimize_A ||A @ H_1^T||^2_F  subject to  A @ A^T = I
```

Analytical solution via eigendecomposition of covariance S = (1/(k-1)) H_1^T H_1:
```
A_tilde_2 = V^T_{:, n-r:n}  (r smallest eigenvalues)
```

This places A_2 in the subspace where task T1's feature variance is **minimal**.

**Benchmark (RoBERTa-large, 8 GLUE tasks):**
- Task Arithmetic + OSRM: **76.59%** vs 70.04% baseline (+6.55pp)
- CoLA: **32.25%** vs 18.57% (+13.68pp)
- MNLI: **81.24%** vs 74.01% (+7.23pp)
- **Llama3-8B:** 72.75% vs 63.28% baseline (+9.47pp)
- Individual task degradation: only **-0.18pp**

#### LoRI - LoRA with Reduced Interference (arXiv:2504.07448)
- Freezes A matrices as **random projections**
- Sparsifies B matrices using **task-specific masks**
- Leverages orthogonality between adapter subspaces
- **95% fewer trainable parameters** than standard LoRA
- Enables continual learning with minimal catastrophic forgetting

#### NP-LoRA - Null Space Projection (arXiv:2511.11051)
- Identifies that separately trained LoRAs occupy **non-orthogonal, overlapping subspaces**
- Projects each adapter into the null space of others
- Enforces subspace separation to prevent structural interference

#### OPLoRA - Orthogonal Projection LoRA (arXiv:2510.13003)
- Decomposes pretrained weights via SVD
- Constructs projection matrices removing components aligned with top-k singular triples
- Introduces metric rho_k measuring update energy in dominant pretrained subspace

#### Flat-LoRA (arXiv:2409.14396, ICML)
**Training for flat loss landscape to improve mergeability.**

Bayesian expectation loss:
```
min_{A,B} E_{eps ~ N(0, sigma^2)} L(W + BA + eps_W)
```

Critical insight: Standard LoRA-SAM only optimizes sharpness within column space of A:
```
eps_W ≈ c * (nabla_W L) @ A^T @ A  [restricted subspace!]
```

Flat-LoRA perturbs in full m x n parameter space.

**Benchmark improvements:**
- GSM8K math: **+3.18%** (57.47% -> 60.65%)
- HumanEval code: **+3.08%** (24.85% -> 27.93%)
- Memory overhead: only 0.12GB additional
- Training time: +2.5% (11 min on 7h22min run)
- **Flat-LoRA rank 8 surpasses standard LoRA rank 16** (87.20% vs 86.92%)

### C. Dynamic Routing Methods (Inference Cost)

#### LoRA Soups / CAT (arXiv:2410.13025)

Learnable concatenation:
```
delta_W^l = alpha_1^l * B_1 @ A_1^T + alpha_2^l * B_2 @ A_2^T
```

Layer-specific learnable weights alpha trained on 5% of data (frozen LoRA modules).

**BEST KNOWN RESULT for skill composition:**
- CAT outperforms model-merging by **43%** average
- CAT outperforms data-mixing by **12%** average
- GSM-Hard: **21.11%** (vs 18.8% data-mix, vs 14.18% math-only, vs 8.04% code-only)
- **257% improvement** over baseline -- exceeds SUM of individual improvements (superlinear!)
- **First work showing model merging > data mixing for binary skill composition**

#### PHATGOOSE (arXiv:2402.05859)
- Per-token, per-layer adaptive routing of PEFT modules
- Post-hoc (no access to original training data needed)
- Sometimes outperforms explicit multitask training

#### SpectR - Spectral Routing (arXiv:2504.03454)
- Training-free dynamic composition at each timestep
- Token-wise and layer-wise combinations
- No retraining required

#### LoraHub (arXiv:2307.13269)
- Few-shot assembly of multiple LoRA modules
- No additional parameters or gradients needed
- Better upper bound than in-context learning

#### LoRAMoE / Mixture-of-LoRAs (arXiv:2403.03432)
- Treats LoRAs as experts with learned router
- Token-level routing decisions
- From this project's findings: **MoE beats joint training by -0.70%**

### D. Evolutionary / Search Methods

#### Sakana AI Evolutionary Model Merging (arXiv:2403.13187, Nature Machine Intelligence)

Uses **CMA-ES** to optimize merging recipes in two spaces:
1. **Parameter Space (PS):** Layer-wise weight mixing coefficients
2. **Data Flow Space (DFS):** Optimal inference paths through layers from different models

Search space: reduced from (M+1)^T to 2^T via indicator arrays.

**Results:**
- EvoLLM-JP (7B): **52.0% MGSM-JA**, surpassing 70B parameter models
- EvoVLM-JP: **51.2 ROUGE-L** on VLM benchmark (vs 41.1 baseline)
- Cross-domain merging: Japanese LLM + English math + English VLM

---

## 4. ZERO-COST COMPOSITION (No Speed Penalty)

### Methods That Merge Statically While Maintaining Quality

| Method | Mechanism | Quality Loss | Speed Penalty |
|--------|-----------|-------------|---------------|
| OSRM (constrain before training) | Eigendecomposition of feature covariance | -0.18pp individual task | **None** (merged weights) |
| LoRI (random A + sparse B) | Orthogonal subspaces by construction | Minimal | **None** (merged weights) |
| DO-Merging | Decouple magnitude/direction + orthogonalize | +4.77pp over TIES | **None** (merged weights) |
| Flat-LoRA (train in flat region) | Bayesian perturbation during training | +3.18pp on math | **None** (merged weights) |
| DARE-TIES | Drop 90-99% + sign election | Moderate | **None** (merged weights) |
| LoRA Soups CAT | Concatenation (higher rank merged adapter) | **+43% over merging** | Slightly higher rank |

**Best zero-cost approach:** Train with OSRM constraints + Flat-LoRA, then merge with DO-Merging.

**Best quality approach:** LoRA Soups CAT with 5% calibration data (technically near-zero cost, as rank doubles but is still low).

---

## 5. PRODUCTION DEPLOYMENT

### Apple Intelligence (arXiv:2407.21075)

**Architecture:**
- ~3B parameter on-device model
- LoRA adapters of **rank 16**
- Adapts: all attention matrices, attention projection, FFN layers
- Adapter size: **~10s of megabytes** (16-bit)
- Base model: mixed 2-bit/4-bit quantization, **3.7 bits-per-weight** average

**Runtime composition:**
- Adapters **dynamically loaded, cached in memory, and swapped** at runtime
- Each task (proofreading, email reply, summarization) gets its own adapter
- Base model shared across all tasks and apps
- **NOT merged** -- switched at runtime (one adapter at a time)

**Performance on iPhone 15 Pro:**
- Time-to-first-token: **0.6ms per prompt token**
- Generation: **30 tokens/second**

**Key insight:** Apple does NOT compose multiple LoRAs simultaneously. They switch between task-specific adapters. This sidesteps the interference problem entirely.

### Hugging Face PEFT Multi-Adapter Serving

**Hotswapping:** New adapter weights swapped in-place (no memory accumulation):
```python
model.load_adapter("new_adapter", hotswap=True)
```

**Production strategies:**
1. **Merge for single-task:** Merge adapter into base for latency reduction
2. **Keep separate for multi-task:** Load base once, attach appropriate adapter at inference
3. **Optimization recipe:** Flash Attention 3 + torch.compile + FP8 = **2.23x speedup on H100**

### Multi-LoRA Serving Frameworks

- **S-LoRA (arXiv:2311.03285):** Serves thousands of concurrent LoRA adapters
- **Punica:** Batched multi-adapter inference with CUDA kernels
- **dLoRA:** Distributed LoRA serving
- **Together AI Serverless Multi-LoRA:** Dynamic adapter switching at scale, **90% base model performance**
- **LoRAX:** Production LoRA serving with adapter merging support

### From This Project's Findings

- **Batched LoRA k=1 overhead: -4%** (faster than monolithic!)
- **Hash routing is plug-and-play:** 5.3% displacement at N=20
- **1/N scaling resolves composition catastrophe:** PPL from trillions to 2.36

---

## 6. SYNTHESIS: THE BEST APPROACH FOR THIS PROJECT

### Tier 1: Proven Methods (Use These)
1. **Train with OSRM constraints** (arXiv:2505.22934) -- constrain adapter subspaces BEFORE training using feature covariance eigendecomposition. +6-9pp improvement on merging with only -0.18pp individual degradation.
2. **Apply Flat-LoRA** (arXiv:2409.14396) during training -- Bayesian perturbation for flat loss landscape. +3pp on math/code with negligible overhead.
3. **Use LoRA Soups CAT** (arXiv:2410.13025) for composition -- learnable layer-wise coefficients with 5% calibration data. 43% better than static merging.

### Tier 2: Promising Methods (Test These)
4. **DO-Merging** (arXiv:2505.15875) for post-hoc improvement -- decouple magnitude/direction + orthogonalize. Data-free, +4.77pp over TIES.
5. **LoRI sparse masks** (arXiv:2504.07448) for extreme parameter efficiency -- 95% fewer parameters with orthogonal guarantees.

### Tier 3: If Routing Budget Available
6. **PHATGOOSE** per-token routing or **SpectR** spectral routing for maximum quality at inference cost.

### The Answer to the Core Question

**Vision models compose well because:**
- Fine-tuned models share a flat, connected loss basin (linear mode connectivity holds)
- Task vectors are approximately co-directional (features generalize across vision tasks)
- Vision task diversity is lower (classification variants vs. reasoning types)

**LLMs compose poorly because:**
- Task vectors point in conflicting directions (math vs. code vs. language)
- Interference term B_2 @ A_2 @ h_1 is large when task feature distributions overlap in weight space but require different transformations
- Nonlinearities (SwiGLU gating, softmax attention) amplify small perturbations catastrophically
- LLM tasks require different *reasoning pathways* through the same layers, not just different *features*

**The fix:** Constrain adapters to occupy orthogonal subspaces in the directions where other tasks' features have minimal variance (OSRM), or train for flat loss landscape (Flat-LoRA), then use learnable layer-wise composition (LoRA Soups CAT).

---

## Key ArXiv IDs Reference

| Paper | ArXiv ID | Key Contribution |
|-------|----------|------------------|
| LoRA | 2106.09685 | Original low-rank adaptation |
| Model Soups | 2203.05482 | Weight averaging works for vision |
| Task Arithmetic | 2212.04089 | Task vector algebra |
| TIES-Merging | 2306.01708 | Trim, elect sign, merge |
| LoraHub | 2307.13269 | Dynamic few-shot LoRA composition |
| DARE | 2311.03099 | Drop and rescale (90-99% pruning works) |
| S-LoRA | 2311.03285 | Serving thousands of concurrent adapters |
| DoRA | 2402.09353 | Weight-decomposed adaptation |
| PHATGOOSE | 2402.05859 | Per-token per-layer routing |
| Mixture-of-LoRAs | 2403.03432 | MoE-style LoRA routing |
| Sakana Evolutionary | 2403.13187 | CMA-ES model merging optimization |
| Apple Intelligence | 2407.21075 | On-device LoRA switching (rank 16, 3.7bpw) |
| Flat-LoRA | 2409.14396 | Flat loss landscape for mergeability |
| LoRA Soups | 2410.13025 | CAT: 43% better than averaging, superlinear |
| LoRI | 2504.07448 | Random A + sparse B, 95% fewer params |
| SpectR | 2504.03454 | Training-free spectral routing |
| DO-Merging | 2505.15875 | Decouple magnitude/direction + orthogonalize |
| OSRM | 2505.22934 | Pre-training subspace constraints via eigendecomp |
| OPLoRA | 2510.13003 | SVD-based orthogonal projection |
| NP-LoRA | 2511.11051 | Null space projection for fusion |
| TC-LoRA | 2508.03999 | Tensorized clustered merging |
| Pause Recycling LoRAs | 2506.13479 | Position paper: limits of LoRA composition |
