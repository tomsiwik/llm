# LEARNINGS: Boundary Detection via Sliding-Window Domain Classification

## Core Finding

**PPL-based sliding-window boundary detection is structurally impossible on two independent axes:
latency (O(N*W) forward passes = 3 seconds for 256 tokens) and quality (correlated false positives
from overlapping windows cascade into misrouted segments, making detected-boundary routing WORSE
than no-boundary routing). The domain signal IS detectable (88.7% window accuracy) but needs a
fundamentally different delivery mechanism.**

## Why This Happened

### The Independence Assumption Failure
Corollary 1 predicted 0.013 false positives per sequence; measured 0.26 (20x). The root cause:
overlapping sliding windows share most of their tokens, creating correlated PPL estimates. When
one window produces noisy PPL (e.g., near a boundary or on ambiguous text), adjacent windows
produce similarly noisy PPL, causing burst argmax flickering. This is a known failure mode in
change-point detection: Basseville & Nikiforov (1993) specifically warn that windowed statistics
violate independence when windows overlap, and recommend CUSUM or sequential probability ratio
tests instead of windowed argmax.

### The Cascade Failure Mode
Theorem 3 modeled boundary error as epsilon misassigned tokens. The actual failure mode is
qualitatively different: false positive boundaries CREATE entire misrouted segments (32-128
tokens each), and the Phase 2 implementation used the FIRST detected boundary (often a false
positive before the true boundary). This is not a quantitative prediction error — the proof
modeled the wrong failure mode entirely.

### The Latency Wall
PPL-based classification requires a full model forward pass per (window, adapter) pair.
At 40ms per forward pass and 75 passes per 256-token sequence, this is 3 seconds — 600x over
the 5ms serving budget. This is not improvable within the PPL framework: even with batching,
the compute is O(N_windows * N_adapters * forward_pass_cost).

## Confirming Evidence

1. **CUSUM vs windowed statistics** — Basseville & Nikiforov (1993) established that overlapping
   windows violate independence assumptions required for reliable change-point detection.
   Sequential methods (CUSUM, SPRT) are designed specifically to avoid this failure mode.

2. **Online Neural Networks for Change-Point Detection** (arXiv:2010.01388) — Demonstrates that
   neural change-point detectors achieve linear computational complexity, confirming our finding
   that the O(N*W) approach is structurally wrong. Practical detectors must be O(T), not O(N*W*T).

3. **Neural Network-Based Change Point Detection** (arXiv:2503.09541) — Uses a single trained
   network over a moving window with calibrated test error, not exhaustive evaluation of all
   candidate models per window. Confirms: the classifier must be trained, not brute-forced.

4. **ERGO: Entropy-Guided Resetting** (arXiv:2510.14077) — Uses token-level entropy from the
   model's next-token distribution as a low-cost temporal signal to detect context degradation.
   Confirms entropy is a viable zero-cost proxy for domain shift detection during inference.

## Contradicting Evidence

1. **MoLoRA** (arXiv:2603.15965) — Per-token routing CAN work with jointly trained routers
   and adapters. Our failure is specific to post-hoc frozen adapters with brute-force PPL
   classification. A jointly trained lightweight router would not have the independence or
   latency problems.

2. **X-LoRA** (arXiv:2402.07148) — Uses hidden states to dynamically mix adapted layers at
   the token level. The gating network operates on hidden representations (already computed),
   not separate PPL forward passes. Achieves dynamic composition without the O(N*W) cost.

3. **TT-LoRA MoE** (arXiv:2504.21190) — Lightweight noisy top-1 gating routers using base
   model hidden representations for dynamic expert selection. The router is a tiny MLP on
   hidden states, not exhaustive evaluation. Works in a task-agnostic manner.

## Alternative Approaches (with paper evidence)

### 1. Hidden-State Probe Router (STRONGEST CANDIDATE)
**X-LoRA** (arXiv:2402.07148): A small MLP on the model's hidden states selects adapter
mixing weights. Cost: one matrix multiply per token (negligible vs forward pass).
**LoRAuter** (arXiv:2601.21795): Routes via task embeddings from small validation sets,
scales with number of tasks not adapters.
**TT-LoRA MoE** (arXiv:2504.21190): Noisy top-1 gating on hidden representations.

Why it fixes our failure: Hidden states are already computed during generation (zero
additional forward passes). A trained probe learns smooth routing functions (no argmax
flickering). The router IS the classifier — no separate detection pipeline.

### 2. Entropy-Based Boundary Detection
**ERGO** (arXiv:2510.14077): Token-level entropy as temporal signal for context shifts.
**Entropy-Driven Pre-Tokenization** (arXiv:2506.15889): Entropy spikes at coherent
boundaries achieve F1=58.7% for tokenization boundaries.

Why it fixes our failure: Entropy is computed from the base model's logits (already
available during generation). No adapter-specific forward passes needed. But: entropy
signals domain uncertainty, not domain identity — would detect boundaries without
knowing which adapter to route to.

### 3. Segment-Level Task Embedding Router
**LoRAuter** (arXiv:2601.21795): Organizes adapters into task-indexed catalog using
lightweight validation embeddings. Routes by input-task similarity, not exhaustive PPL.
**L-MoE** (arXiv:2510.17898): Lightweight gating network dynamically composes LoRA
adapters via weighted parameter averaging per token.

Why it fixes our failure: Routes at task/segment level via embedding similarity
(O(1) per segment), not O(N*W) forward passes. Tested at scale.

## Implications for Next Experiments

1. **PPL-based anything is dead for online routing.** Any approach that requires adapter-
   specific forward passes is O(N) per decision point. The router must use representations
   already computed by the base model (hidden states, logits, entropy).

2. **The false-positive cascade is the structural lesson.** Any boundary detector that uses
   hard argmax decisions on noisy signals will produce false positives that cascade.
   Soft routing (weighted adapter mixing) sidesteps boundary detection entirely.

3. **Two viable paths forward:**
   - **Path A (segment-level):** Entropy/hidden-state boundary detection + segment isolation
     from Finding #305. Preserves the +16% segment isolation benefit but replaces PPL
     classification with a cheap signal.
   - **Path B (token-level soft routing):** Hidden-state probe that outputs soft adapter
     weights per token. No boundaries needed — routing is continuous. Requires trained
     router (not post-hoc). X-LoRA and L-MoE prove this works.

4. **Path B is more general but requires joint training.** Path A is deployable with
   frozen adapters. The choice depends on whether we can train a router on our adapter set.

## Recommended Follow-Up

**exp_hidden_state_probe_router (P1)**

- **Motivation:** Finding #307 (this experiment) proved PPL-based routing is O(N*W) and
  quality-impossible. Finding #305 proved segment isolation works with oracle routing (+16%).
  The gap: need a cheap, smooth router that uses existing hidden states.
- **Literature:** X-LoRA (arXiv:2402.07148) proves hidden-state routing works for LoRA
  mixtures. TT-LoRA MoE (arXiv:2504.21190) proves noisy top-1 gating on hidden states
  works in task-agnostic settings. LoRAuter (arXiv:2601.21795) proves task-level routing
  via embeddings scales efficiently.
- **Why it fixes the failure:** Hidden states are free (already computed). A trained MLP
  produces smooth outputs (no argmax flickering). Router overhead is one small matmul per
  token (~0.01ms), not 75 forward passes (3000ms).
- **Design:** Train a small MLP (hidden_dim -> N_adapters) on per-token adapter classification
  using the existing 5 domain adapters. Input: base model hidden states at layer L.
  Output: adapter selection weights. Evaluate: adapter selection accuracy AND end-to-end
  PPL on mixed-domain sequences from Finding #305's test set.

## Paper References Added

- arXiv:2010.01388 — Online Neural Networks for Change-Point Detection (O(T) complexity)
- arXiv:2402.07148 — X-LoRA: Hidden-state routing for LoRA adapter mixtures
- arXiv:2504.21190 — TT-LoRA MoE: Noisy top-1 gating on hidden representations
- arXiv:2510.14077 — ERGO: Entropy-guided temporal signal for context degradation
- arXiv:2601.21795 — LoRAuter: Task-level routing via embeddings
- arXiv:2510.17898 — L-MoE: Lightweight gating for LoRA expert composition
