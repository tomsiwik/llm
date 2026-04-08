# Adversarial Review: Pierre Research Program

**Reviewer:** Automated peer review, NeurIPS calibration
**Scope:** Full program (427 experiments, 358 findings, architecture, product vision)
**Date:** 2026-04-07

---

## 1. Mathematical Gaps

### 1.1 The Grassmannian Guarantee: Proven but Narrower Than Claimed

The central mathematical claim is correct within its stated scope: QR-constructed
A-matrices produce A_i^T A_j = 0, and therefore the Frobenius inner product
of LoRA adapters is zero. This is valid linear algebra. No dispute.

**However, the claim "composition cannot fail" (VISION.md line 41) is false.**
The guarantee covers PARAMETER-SPACE interference only. The architecture
documents acknowledge this but understate it. The actual output is:

    h_out = W_base*x + B_1(A_1*x) + B_2(A_2*x)

Even with orthogonal A-slots, B_1(A_1*x) and B_2(A_2*x) can destructively interfere
in activation space. Finding #351 measured B-matrix cosine max |cos| = 0.291.
This is not negligible. The VISION.md claim should read "parameter-space interference
is impossible; activation-space interference is empirically small but unbounded."

**Severity: Medium.** The math is right, the framing overpromises. A reviewer
at a top venue would flag "composition cannot fail" as misleading.

### 1.2 Theorems That Are Not Theorems

**Conjecture 2 (Enrichment Monotonicity):** The MATH.md for m2p_cross_domain_graph
originally labeled this a "Theorem." It was circular (assumed its conclusion) and was
empirically refuted (parity loss increased 6.3x). Credit to the team for relabeling
it as a refuted conjecture. But this reveals a pattern: the proof-first workflow
can produce "theorems" that are really "descriptions dressed in equations."

**The Bartlett Scaling Heuristic (m2p_macro_quality):** Applied Bartlett et al.
(linear regression, isotropic sub-Gaussian features) to a nonlinear transformer.
The prediction was 48.8% quality; the actual was 101%. The document correctly
labels this an "engineering estimate" but then derives the K882 threshold (85%)
from it. The threshold has no mathematical basis. It was falsified by 16pp.

**Hardt et al. Bound (m2p_data_scale):** Applied convex-loss bound to non-convex
transformer. Predicted train-val gap of 0.001 nats; measured 0.337 nats (270x off).
Again correctly flagged in the paper, but the quantitative predictions from
"Theorem 2" are vacuous.

**Pattern:** The program cites real theorems (Ghadimi-Lan, Bartlett, Hardt) but
applies them outside their preconditions. The qualitative conclusions happen to
be correct, but the quantitative predictions from these misapplied theorems are
consistently orders of magnitude wrong. A top-venue reviewer would reject any
claim that these constitute "verified proofs."

**Severity: High for claims of mathematical rigor.** The actual finding -- that
M2P quality scales with d_model -- is empirically well-supported. But the
theoretical scaffolding around it is decorative, not load-bearing.

### 1.3 Circular Proofs and Missing Proofs

**"Why Promotion Works"** (ARCHITECTURE.md section) cites ReLoRA as proof that
"adapter promotion = continued pre-training." ReLoRA shows repeated merge and
retrain converges for FULL pre-training runs. Pierre promotes ONCE with scale=5.
These are different operations. Finding #333 shows 0pp MMLU degradation for a
single promotion. But Finding #331 shows catastrophic interference at promotion 3.
The claim "promotion works" is proven for exactly one cycle.

**The "model gets better with usage" flywheel** has no mathematical backing.
The dissolve-recrystallize cycle (m2p_cross_domain_graph) refuted enrichment
monotonicity. The promotion cycle (#333) works once. There is no proof,
conjecture, or even heuristic argument that repeated cycles converge.

**Severity: High for the product vision.** The flywheel is the entire business case.

### 1.4 The d_model-Independence Claim Rests on 2 Valid Domains

At d=512, arithmetic was excluded by the parity guard. Only sort and reverse
remained. At d=1024, same situation: only sort and reverse.

The "three-point scaling law" (current_direction.md) is actually a TWO-domain
result at each of three d_model values. The domains are structurally similar
(both involve reordering sequences). No natural-language, no reasoning, no
code generation -- nothing remotely resembling real LLM workloads.

**Severity: High.** The scaling claim is the strongest result in the program
and it rests on the thinnest evidence.

---

## 2. Scale Blindspots

### 2.1 Everything Is Toy Scale

Every M2P experiment uses ToyGPT (d=64 to d=1024, 2-4 layers, 100K-25M params)
on synthetic domains (arithmetic, sort, reverse, repeat, parity).

**What breaks at real scale (d=3584, 36 layers, Qwen3-4B):**

1. **M2P output head dimensionality.** At d=3584, the fc1 output head must generate
   2 * 4 * (4*3584) = 114,688 parameters from d_M2P=64. That is a 1792:1 compression
   ratio. The program tested up to 512:1 (d=1024). The 3.5x jump is untested.

2. **Layer count.** Qwen3-4B has ~36 layers. ToyGPT has 2. The M2P must generate
   B-matrices for 36 layers per call (Option A) or make 36 separate calls (Option B).
   Neither has been tested. This is correctly identified as the next experiment
   (current_direction.md) but it is the single largest gap between toy and real.

3. **GQA attention.** Qwen3-4B uses grouped-query attention. Finding #318 verified
   Grassmannian holds on GQA, but no M2P experiment has run on a GQA architecture.

4. **Natural language.** Every domain is a synthetic sequence task with a simple
   grammar. Natural language has long-range dependencies, polysemy, pragmatics.
   The parity-guard exclusion of arithmetic at d>=512 (base already solves it)
   is a preview of what happens with real pretrained models: the base is already
   competent, and the adapter margin shrinks. At scale, the SFT improvement
   over base on most tasks may be tiny, making quality_ratio undefined or noisy.

5. **Tokenization.** Toy models use character-level tokenization. Real models use
   BPE. The TF-IDF routing that achieved 95% accuracy on toy domains may not
   transfer to BPE-tokenized natural language where domain boundaries are fuzzy.

### 2.2 The d=512 Result Has Only 2 Valid Domains

Already noted in 1.4 above. The paper itself correctly identifies this limitation,
which is appreciated. But the current_direction.md treats d_model scaling as "CLOSED"
based on this evidence. It is not closed; it is provisionally supported.

### 2.3 The 97.6% and 100.6% Numbers Are on Trivially Learnable Tasks

Arithmetic (add two numbers), sort (sort a sequence), reverse (reverse a sequence).
These are regular-language tasks with simple compositional structure. Any model
with enough parameters to represent the mapping will solve them.

The real question is not "can M2P generate adapters for sort?" but "can M2P
generate adapters for medical reasoning, legal analysis, or code generation?"
These tasks have no analogy in the toy setup.

### 2.4 M2P Compression Ratio at Qwen3-4B

| Module | d_out at Qwen3-4B | B params per layer | Compression (d_M2P=64) |
|--------|------------------|--------------------|----------------------|
| wq | 3584 | 4 * 3584 = 14,336 | 224:1 |
| wk | 512 (GQA) | 4 * 512 = 2,048 | 32:1 |
| wv | 512 (GQA) | 4 * 512 = 2,048 | 32:1 |
| wo | 3584 | 4 * 3584 = 14,336 | 224:1 |
| fc1 | 14336 (MLP 4x) | 4 * 14336 = 57,344 | 896:1 |

The fc1 compression ratio at Qwen3-4B is 896:1 PER LAYER. Across 36 layers,
a single M2P call must generate 36 * (14336 + 2048 + 2048 + 14336 + 57344) * 4
= 36 * 90,112 * 4 = 12,976,128 B-matrix parameters. From 64 hidden dimensions.

The Aghajanyan intrinsic dimensionality argument says this should work if the
actual parameter update lives in a <64-dimensional subspace. This is plausible
for SFT of a well-pretrained model (the original Aghajanyan result), but
UNVERIFIED for hypernetwork-generated adapters.

---

## 3. Product Vision Gaps

### 3.1 The Flywheel Has No Evidence

The claimed flywheel: generate adapters -> compose -> promote to base -> base improves -> repeat.

Evidence for each step:
- **Generate adapters:** Supported at toy scale (93-101% of SFT, Findings #339, #359, #361, #362)
- **Compose:** Grassmannian orthogonality proven; routing fixed by TF-IDF (Finding #352)
- **Promote to base:** Single promotion at scale=5 works (Finding #333)
- **Base improves:** Promotion from random init fails at step 3 (#331). SVD breaks Grassmannian (#329). Dissolve-recrystallize REFUTED monotonicity (#353).
- **Repeat:** COMPLETELY UNTESTED. Zero evidence.

The flywheel requires ALL steps to work AND to compound. Step 4 has been tested
once (single promotion) and failed in every multi-step variant attempted. The
product vision rests on an unproven hypothesis.

### 3.2 Per-User Adapters (Tier 3) Are Entirely Unproven

No experiment has tested:
- Accumulating session adapters into a persistent user profile
- Whether user-level adaptation even produces measurably different B-matrices
- How many sessions are needed for a useful user adapter
- Whether user adapters compose with domain adapters

This is a core differentiator in the product vision ("model personalizes to you")
and has zero experimental backing.

### 3.3 "Model Gets Better With Usage" -- Where?

The only evidence is a single promotion cycle (#333) that preserved quality.
"Preserved" is not "improved." The dissolve-recrystallize experiment (#353)
showed 3/5 domains improved but parity was catastrophically destroyed.

No experiment has shown the base model getting BETTER through the promotion cycle.
At best, it gets not-worse.

### 3.4 Cross-Domain Links: Parity Destroyed, Safe Dissolve Not Run

Finding #353 showed that dissolving 10 cross-domain adapters at scale=5 destroyed
parity (0.59 -> 3.73, 6.3x regression). The PAPER recommended PROMOTE_SCALE/N as
the fix. This has not been tested. The cross-domain pipeline is stuck at
"provisional" status.

---

## 4. Comparison Honesty

### 4.1 Pierre vs MoE: Misleading Comparison

VISION.md and ARCHITECTURE.md compare Pierre to Mixtral and DeepSeek MoE.
This comparison is misleading in several ways:

1. **MoE experts are INSIDE the model.** They share the same residual stream,
   are trained jointly end-to-end, and route at the FFN level. Pierre's experts
   are OUTSIDE the model (LoRA side-paths), trained independently, and composed
   post-hoc. These are fundamentally different architectures.

2. **MoE at 7B scale already has real benchmarks.** Pierre has zero real-world
   benchmarks. Comparing architectures without comparable evaluations is empty.

3. **"Simpler than Mixtral"** -- Pierre is also orders of magnitude less capable.
   Simplicity is not a virtue if it comes from doing less.

### 4.2 "No One Does All Three" -- Novel or Broken?

The claim that no existing system "generates, composes, and promotes" adapters
is presented as evidence of novelty. An equally valid interpretation: nobody
does it because it does not work. Until the promotion cycle is demonstrated
beyond a single step, this is an open question.

### 4.3 66.6% of SFT Quality Is Not Competitive

Finding #339 reports M2P generates adapters at 66.6% of SFT quality. This was
later improved to 93-101% in subsequent experiments. But the VISION.md still
leads with 66.6% in the architecture diagram (line 22). This number is stale
and should be updated or removed.

More importantly: even 93-101% of SFT quality is 93-101% of a TOY SFT adapter
on SYNTHETIC domains. The absolute quality bar is unknown. If SFT on Qwen3-4B
for a real domain gives MMLU+3pp, then 93% of that is MMLU+2.8pp. Is that
useful? Unclear.

---

## 5. Missing Experiments (Critical Gaps)

### TOP 5 That MUST Pass Before This Is Credible

1. **M2P on Qwen3-4B with real NLP benchmarks (MMLU, GSM8K, HumanEval).**
   Everything so far is toy. Until M2P generates an adapter that measurably
   improves a real model on a real benchmark, Pierre is a hypothesis.

2. **Layer-depth scaling (L=4, 8, 16, 36).**
   Correctly identified as the next experiment. This is the #1 architectural risk.
   If M2P cannot handle 36 layers, the Qwen3-4B product does not exist.

3. **Multi-cycle promotion (promote -> train new adapter -> promote again -> x5).**
   The flywheel requires this. Finding #331 killed it from random init.
   Finding #333 showed one cycle works from pretrained. Need 5+ cycles.

4. **Natural language domain routing.**
   TF-IDF at 95% on toy domains. What about routing between "medical" and "legal"
   on real text? Domain boundaries are much fuzzier in natural language.

5. **Activation-space interference at N=24+ adapters.**
   The Grassmannian guarantee is parameter-space only. At N=24 with 224 available
   slots, what is the activation-space interference? B-matrix cos was 0.29 at N=5.
   Does it grow with N?

### TOP 5 That Could KILL the Entire Approach

1. **Layer-depth scaling fails.**
   If M2P quality degrades significantly when generating adapters for 36 layers
   from d_M2P=64, the single-pass adapter generation dies. Option B (per-layer
   calls) survives but loses the "one forward pass" selling point and adds
   36x latency.

2. **Natural language quality ratio collapses.**
   If the SFT improvement over a pretrained base on real NLP tasks is tiny
   (e.g., MMLU+1pp), then the quality ratio metric becomes noise. The parity
   guard already excludes domains where base is competent. At Qwen3-4B scale,
   MOST domains may be "parity-like" (base already competent).

3. **Multi-cycle promotion diverges.**
   If promotion cycle 2 or 3 degrades quality, the flywheel breaks and Pierre
   becomes a static adapter library (which already exists -- see LoRAX, S-LoRA).

4. **Activation-space interference grows with N.**
   If B-matrix interference scales as O(sqrt(N)) or worse, composition quality
   will degrade at scale. The parameter-space guarantee does not help.

5. **TF-IDF routing fails on real domains.**
   If sequence-level routing cannot distinguish "Python code" from "Rust code"
   or "cardiology" from "oncology," the routing architecture needs replacement.
   Per-token MLP routing already failed (#351). If both fail, routing is dead.

---

## 6. The Strongest Attack

### If I Were a Competitor Trying to Prove Pierre Is a Dead End

**The argument:** Pierre's theoretical foundation is a tautology, its empirical
evidence is toy, and its product vision is unproven.

1. **The Grassmannian guarantee is trivially true and practically irrelevant.**
   Any set of LoRA adapters with orthogonal A-matrices has zero parameter-space
   interference. This is freshman linear algebra, not a novel contribution.
   FlyLoRA (arXiv:2510.08396) already uses frozen random A-matrices with JL-lemma
   guarantees. The parameter-space guarantee tells you nothing about whether
   the composed model actually works -- that depends on activation-space behavior,
   which Pierre does not guarantee.

2. **M2P is a hypernetwork. Hypernetworks have a 7-year track record of not scaling.**
   Ha et al. (2016) introduced hypernetworks. Despite continuous research, no
   production LLM system uses hypernetwork-generated weights. The reason:
   hypernetworks generate worse weights than direct training, and the generation
   speed advantage disappears when you can cache the trained weights. Pierre's
   M2P is a hypernetwork that generates LoRA B-matrices. It achieves 93-101% of
   SFT quality on toy tasks. The question is whether this gap widens or closes
   on real tasks. The 7-year base rate says: widens.

3. **The promotion cycle is just continual learning, which is a solved problem.**
   CL methods (PackNet, HAT, ProgressiveNets, SUPR) handle sequential expert
   addition with proven guarantees. Pierre's promotion (base + scale*adapter)
   is the simplest possible CL strategy and was killed at step 3 (#331).
   The "self-growing model" claim is rebranded continual learning that does
   not actually work beyond one step.

4. **The product just reinvents S-LoRA / LoRAX with extra steps.**
   Serving multiple LoRA adapters per request is a solved engineering problem
   (S-LoRA, LoRAX, vLLM with LoRA). These systems compose adapters, route
   requests, and serve at production scale. Pierre adds a hypernetwork on top
   of this stack, generating inferior adapters faster. But "faster to generate"
   is not a product differentiator when adapter training takes minutes and
   adapters are cached indefinitely.

### The Single Weakest Link

**The promotion cycle.** Everything else has fallback paths:
- M2P fails -> fall back to SFT adapters (proven)
- Routing fails -> fall back to TF-IDF or manual selection (proven)
- Activation interference grows -> fall back to fewer adapters (safe)

But if the promotion cycle does not compound, Pierre is just another LoRA
serving framework with a hypernetwork bolted on. The flywheel is dead.
The "model gets better with usage" claim dies. The product differentiation dies.

And the evidence says: promotion works once (#333), fails at step 3 from
random (#331), fails via SVD (#329), causes catastrophic regression when
dissolving multiple adapters (#353). One success, three failures.

---

## 7. What Is Actually Strong

In fairness, the program has genuine strengths:

1. **Methodological rigor.** The self-correction is impressive. Conjectures get
   relabeled when refuted. Papers honestly report failures. Kill criteria are
   enforced. This is better than most research programs.

2. **Grassmannian orthogonality is real.** The math is trivial but the
   implementation is correct and the empirical verification is thorough
   (cos = 0 at float32 precision across all experiments).

3. **M2P quality scaling.** The three-point progression (97.6% -> 101% -> 99.6%)
   at d=256/512/1024 is genuinely interesting. If it holds at d=3584, the
   hypernetwork approach has a real speed/quality tradeoff worth exploring.

4. **Honest failure modes.** The killed experiments (#329, #331, #334, #341, #342)
   are well-documented with root-cause analysis. The program learns from failures.

5. **Apple Silicon focus.** Running everything on-device with MLX is a real
   engineering constraint that, if it works, would be genuinely differentiated.

---

## 8. Recommendations

### For Academic Publication

The M2P quality scaling result (d_model-independence of intrinsic adapter
dimensionality) is publishable as a workshop paper if:
- Extended to L=8+ layers
- Validated on at least one natural-language task
- The Aghajanyan connection is made formally (not just cited)
- The Bartlett misapplication is removed (currently weakens the paper)

The Grassmannian composition result is not publishable alone -- it is too
trivially true. It could be a section of a systems paper.

### For Product Development

1. **Run the layer-depth experiment immediately.** This is the #1 blocker.
2. **Run a single real-NLP benchmark.** Even one MMLU category with M2P-generated
   adapters on Qwen3-4B would be more convincing than all toy experiments combined.
3. **Stop claiming the flywheel works.** Claim "single-cycle promotion works;
   multi-cycle is under investigation." The current language overpromises.
4. **Drop the Bartlett citations.** They are misapplied and hurt credibility.
5. **Rename "composition cannot fail" to "parameter-space interference is zero."**

### Verdict on Research Program

**REVISE.** The micro-scale foundation is solid but the claims far exceed the
evidence. The program needs:

1. Layer-depth scaling to L >= 8 (blocking for Qwen3-4B claim)
2. At least one natural-language validation (blocking for product claim)
3. Multi-cycle promotion evidence (blocking for flywheel claim)
4. Retraction of "composition cannot fail" language
5. Honest accounting that the theoretical scaffolding (Bartlett, Hardt) is
   decorative, not load-bearing -- the actual evidence is empirical

---

## 9. Resolution Tracker

| Critique | Status | Resolution |
|----------|--------|------------|
| #1 Layer depth L=2 only | **RESOLVED** — #365: L=36 at 89.1%, K894 PASS. Aghajanyan d_int<64 holds at 2304:1 | DONE |
| #2 Only 2 valid domains | **RESOLVED** — #378: GSM8K (real NLP) is structurally different from toy domains. M2P works on both. | DONE |
| #3 No natural language | **RESOLVED** — #376/#378: M2P on Qwen3-0.6B + GSM8K achieves 28.6% (base 20%, SFT 26%). quality_ratio=143%. | DONE |
| #4 Promotion works once | → Level 4A (multi-cycle) | OPEN |
| #5 Safe dissolve untested | **RESOLVED** — #366: S3 selective routing wins (5/5 protected). S1 merges 0/10 (no safe uniform merge exists). | DONE |
| #6 Activation-space unbounded | **RESOLVED** — #372: α=0.38 (sub-linear), per-token max\|cos\|=0.339 at N=10, plateaus. | DONE |
| #7 "Composition cannot fail" | **RESOLVED** — reframed in VISION.md, ARCHITECTURE.md | DONE |
| #8 Bartlett/Hardt misapplied | **RESOLVED** — dropped quantitative predictions from M2P_DISTILLATION.md | DONE |
| #9 Per-user adapters unproven | → Level 5A | OPEN |
| #10 Hypernetworks don't scale | **RESOLVED** — #378: M2P on Qwen3-0.6B matches SFT (p=0.36, not stat. sig. different). #370: d=3072 L=36 at 90% on toy. | DONE |
