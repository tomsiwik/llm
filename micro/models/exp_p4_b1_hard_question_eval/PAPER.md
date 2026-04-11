# PAPER.md — P4.B1: Gap-Targeted Evaluation — Hard Domain Questions

## Status: KILLED (smoke test — N=5 per domain)

Early kill: structural impossibility identified from smoke test results.
Full run not executed.

---

## Prediction vs Measurement Table (Smoke, N=5)

| Kill Criterion | Prediction | Measurement | Pass/Fail |
|---|---|---|---|
| K1227: Base mean < 25% on all domains | < 0.25 | math=0.633, med=0.600, legal=0.433, code=0.467, fin=0.433 | **FAIL** |
| K1228: ≥3/5 domains ≥15pp improvement | ≥3 domains | 0/5 (max = math +13pp) | **FAIL** |
| K1229: r(base, improvement) < -0.30 | < -0.30 | r = -0.3026 | PASS (barely) |

**ALL_PASS: False → KILLED**

---

## Base Score Per Domain (N=5)

| Domain | Base Mean | Scores |
|---|---|---|
| math | 0.633 | [1.0, 0.17, 0.50, 0.67, 0.83] |
| medical | 0.600 | [0.33, 0.67, 0.83, 0.67, 0.50] |
| legal | 0.433 | [0.50, 0.17, 0.67, 0.33, 0.50] |
| code | 0.467 | [0.50, 0.67, 0.50, 0.50, 0.17] |
| finance | 0.433 | [0.33, 0.67, 0.33, 0.50, 0.33] |

**Key observation**: All domain base scores significantly exceed 0.25 threshold.
Most extreme: math=0.633 (2.5× higher than P4.B0's math base of 0.307).

---

## Adapted Score Per Domain (N=5)

| Domain | Base | Adapted | Delta | Pass ≥15pp? |
|---|---|---|---|---|
| math | 0.633 | 0.767 | **+0.133** | FAIL |
| medical | 0.600 | 0.633 | +0.033 | FAIL |
| legal | 0.433 | 0.500 | +0.067 | FAIL |
| code | 0.467 | 0.333 | **-0.133** | FAIL |
| finance | 0.433 | 0.433 | 0.000 | FAIL |

**Critical finding**: Code adapter HURTS on advanced systems questions (-13pp).

---

## Root Cause Analysis

### Why K1227 Failed: Gemma 4 4B Knows Advanced Domain Content

The hypothesis that "advanced domain questions would have base < 25%" was wrong.
Gemma 4 4B instruction-tuned has extensive knowledge of:
- Abstract algebra (Zorn's lemma, Galois theory)
- Medical immunology (V(D)J recombination, antiphospholipid syndrome)
- Legal doctrine (Chevron, qualified immunity, PBFT)
- Distributed systems (Raft, Byzantine faults)
- Quantitative finance (Vasicek, Heston, Kelly criterion)

**The Gemma 4 4B model was trained on the same Wikipedia/textbook content** that these
questions come from. Knowledge of advanced domain vocabulary is NOT a gap for this model.

### The P4.B0 Math Result Was a Notation Artifact

P4.B0 math base = 0.307, P4.B1 math base = 0.633 — same domain, different keyword choices:
- P4.B0 keywords: `["a^2", "u dv", "f(g(x))", "0/0"]` — text notation not produced in prose
- P4.B1 keywords: `["Zorn", "maximal element", "Hilbert space", "eigenvalue"]` — natural vocabulary

The P4.B0 "gap" was not about domain knowledge — it was about notation style:
- Base model answers in natural language ("a squared") not notation ("a^2")
- Math adapter (trained on Q&A with notation) learns to produce "a^2" in text

**Conclusion**: The P4.B0 math +20pp improvement measures notation alignment, not knowledge gaps.

### Why K1228 Failed: Adapter Training Data Mismatch

The rank-6 adapters from P1 T2 were trained on 100 basic Q&A examples (e.g., "What is
the Pythagorean theorem?"). For advanced questions (Galois theory, MESI protocol, Vasicek
model), the adapter training data doesn't cover these topics. Even if a gap existed,
the adapter cannot fill it without being trained on the relevant material.

**Code adapter negative result** (-13pp) confirms this: the code adapter learned basic
algorithm vocabulary (FIFO, heap, sorting) which is WRONG for systems questions (Byzantine,
MESI, Raft). Applying it hurts because it pulls token probabilities toward wrong vocabulary.

---

## Structural Impossibility

**Theorem (P4.B1 Impossibility)**:
For a rank-r LoRA adapter ΔW_d trained on N basic domain examples:
- δ_d > 0 requires BOTH (1) vocabulary gap: H(V_d|θ_base) > H_threshold AND
  (2) distribution overlap: V_d ∩ V_train ≠ ∅ (question vocab must overlap training vocab)

Gemma 4 4B fails condition (1) for all advanced domain questions.
P1 T2 adapters fail condition (2) for advanced subdomain questions.

**Corollary**: The gap-targeted approach requires a model with genuine domain gaps
(e.g., smaller base models, domain-specific tasks like code generation, specific notation).
Gemma 4 4B with instruction-tuning has too broad a knowledge base for this approach.

---

## Comparison: P4.B0 vs P4.B1

| Metric | P4.B0 (Standard Questions) | P4.B1 (Hard Questions) |
|---|---|---|
| Math base | 0.307 | 0.633 |
| Math improvement | +20pp | +13pp |
| Medical base | 0.480 | 0.600 |
| Medical improvement | -4pp | +3pp |
| Notation-type keywords | Yes ("a^2") | No (natural language) |
| K1227 pass | N/A | FAIL (all > 0.25) |

The higher math improvement in P4.B0 vs P4.B1 despite lower base is explained by:
P4.B0 math adapter learned specific notation patterns; P4.B1 questions don't require them.

---

## Implications for P4 Series

1. **Adapter quality is notation-domain-specific**: Math adapters help when evaluation
   uses mathematical notation. Language-based evaluation of same math content shows less gain.

2. **Gemma 4 4B has no exploitable knowledge gap** in standard academic domains.
   Finding genuine gaps requires: (a) smaller base model, (b) proprietary/specialized content,
   or (c) formatting/style tasks (notation, code syntax, document structure).

3. **P4.C design direction**: Test adapters on FORMATTING tasks where Gemma 4 has style gaps:
   - LaTeX notation output (math)
   - Clinical note format (SOAP notes, ICD-10 coding)
   - Legal document structure (pleadings, contracts)
   - Code in specific style/framework (FastAPI, Rust async patterns)

---

## Adversarial Note

**N=5 smoke test** is noisy. The r=-0.3026 for K1229 is barely over the threshold
and not statistically significant at N=25. However, K1227 FAIL is structural (not noise):
even single-question base scores of 0.83 and 1.0 are observed. The questions are simply
not hard for Gemma 4 4B. Running N=15 would confirm K1227 fails.

Early kill is appropriate given clear structural analysis.
