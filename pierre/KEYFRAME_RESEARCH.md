# Keyframe Adapters: Deterministic Verification in Composable Ternary Experts

## The Insight

Video codecs (H.264/H.265) compress most frames as deltas (P-frames) but periodically
insert keyframes (I-frames) that are fully self-contained. Keyframes reset error
accumulation and guarantee the decoder can always recover ground truth.

**Applied to LLMs:** Most tokens are generated probabilistically (P-frames). But certain
tokens carry information that MUST be correct — numerical facts, logical steps, code
syntax. These "keyframe tokens" should flow through a DETERMINISTIC adapter path that
guarantees correctness, not a statistical one that guesses.

## The Architecture

```
y = BitLinear(x)                          # base: probabilistic generation
  + scale_stat * (x @ A_stat) @ B_stat    # statistical adapter: domain knowledge
  + γ * A_det @ Φ(B_det @ x)              # deterministic adapter: verifiable rules
```

Where:
- `A_stat ⊥ A_det` (Grassmannian orthogonality — non-interference guaranteed)
- `Φ` is a hard threshold function (sgn or step) — enforces Boolean logic
- `γ` is large enough that deterministic signal overrides probabilistic base
- `B_det ∈ {-1, 0, +1}` encodes verifiable rules as ternary logic gates

## The Ternary Connection

Ternary weights {-1, 0, +1} naturally encode three logical states:
- **+1**: Strengthen/excite this feature (logical TRUE)
- **-1**: Suppress/inhibit this feature (logical NOT)
- **0**: Don't care / skip / neutral (logical UNKNOWN)

This is not a statistical model — it's a **deterministic filter**. Prior art:
- Boolean Threshold Functions: "weights of irrelevant inputs can be set to zero"
- DFA encoding in RNNs: +H excites, -H inhibits, 0 = no connection (Omlin & Giles)
- Ternary hardware: '0' is a computational skip signal, eliminates ~50% of MAC ops

## 8 Supporting Research Pillars

### 1. Anchor Tokens (AnLLMs) — the keyframe mechanism exists
Anchor-based LLMs train specific tokens to compress entire preceding sequences.
During inference, non-anchor KV caches are deleted — only the anchor survives.
99% KV cache reduction, 3.5x inference speedup.
*Paper: Anchor-based Large Language Models*

### 2. DFAs in Neural Networks — deterministic automata as weight matrices
Deterministic Finite Automata can be encoded exactly in RNN weights.
Weights of ±H force/inhibit state transitions deterministically.
Stable for infinite sequence lengths when H is large enough.
*Paper: Omlin & Giles, "Constructing Deterministic Finite-State Automata in RNNs"*

### 3. WFA-RNN Equivalence — automata ≡ neural layers (proven)
Weighted Finite Automata and second-order RNNs with linear activations are
expressively equivalent. Neural adapter vectors can perfectly map to exact
state-machine logic.
*Paper: "Connecting Weighted Automata and Recurrent Neural Networks through Spectral Learning"*

### 4. Logical Neural Networks (IBM) — differentiable deterministic verification
Every neuron represents a logical formula component. Outputs are strict
upper/lower bounds [L, U] on truth values, not probabilistic guesses.
Supports forward AND backward logical inference (like a theorem prover).
*Paper: "Logical Neural Networks" (IBM Research)*

### 5. Boolean Threshold Functions — ternary gates from weight constraints
Strict margin constraint |w·x| ≥ μ forces neurons to act as pure logic gates
(AND, OR, Majority). Weights naturally converge to {-1, 0, +1} via
divide-and-concur projection.
*Paper: "Learning with Boolean Threshold Functions"*

### 6. Gate-Level Boolean Attention (GL-BEGANN)
Replaces dot-product attention with XNOR similarity scoring.
Hard threshold: attention = 1 if similar, 0 if not.
Feedforward network replaced by trainable logic kernel.
*Paper: "Gate-Level Boolean Evolutionary Geometric Attention Neural Networks"*

### 7. Continuous Relaxation for Training
Discrete logic can't backpropagate directly. Solution: train with continuous
relaxation (T-norms, sigmoid approximations), then discretize at inference.
"Soft training, hard execution" paradigm.
*Papers: Neurosymbolic AI survey, Logical Activation Functions*

### 8. Ternary Epistemic Humility
The "0" state in ternary is not a statistical zero-mean — it's an explicit
"UNKNOWN" state. This maps to the knowable/unknowable distinction in LLMs.
The adapter can express: "I know this is true (+1), I know this is false (-1),
I don't know (0) — so leave the base model's prediction unchanged."
*Paper: "Beyond Binary: Ternary Dynamics"*

## What We Need to Prove (MATH.md)

### Theorem 1: Non-Interference
For any input x, the statistical and deterministic corrections are orthogonal:
⟨A_stat B_stat x, A_det Φ(B_det x)⟩ = 0
Proof: follows from Grassmannian A_stat ⊥ A_det (Finding #3, cos=0.0002).

### Theorem 2: Override Guarantee
The deterministic signal γ·A_det·Φ(B_det·x) dominates the maximum possible
logit from BitLinear(x) + A_stat·B_stat·x when a keyframe rule is violated.
Proof: requires bounding the base model's output magnitude and setting γ accordingly.

### Theorem 3: Composability
New deterministic adapters A_det_new can be added without retraining, as long as
A_det_new ⊥ [A_stat | A_det_1 | ... | A_det_k].
Proof: follows from Grassmannian packing at d=2560 (Finding #3).

### Theorem 4: Speed Preservation
Using the precomputed concatenated delta approach (from SPEED_RESEARCH.md),
the combined statistical+deterministic adapter adds ≤60 kernel dispatches.
At 60 dispatches: ~10% overhead → ~126 tok/s (target: 100+).

## Concrete Example: Math Verifier Adapter

A deterministic adapter for mathematical correctness:
- B_det encodes: "if the token is a digit following '=', check if the
  preceding arithmetic is consistent"
- Φ is a step function: outputs +1 if consistent, -1 if inconsistent
- A_det projects this into the logit space to boost correct digits /
  suppress incorrect ones
- The adapter is a ternary automaton that can verify: 2+3=5 ✓, 2+3=7 ✗

This is not a learned statistical pattern — it's a compiled finite automaton
embedded in a ternary weight matrix. It runs at the same speed as the base
model because it uses the same kernel.

## What Doesn't Exist Yet (the gap)

1. No paper combines ternary deterministic adapters + probabilistic base + 
   Grassmannian orthogonality in one architecture
2. No paper trains deterministic ternary adapters using STE on a BitNet base
3. No paper demonstrates the keyframe/anchor concept with ternary verification
4. The speed optimization (precomputed concatenated deltas) is new

This is the research contribution.
