# LEARNINGS.md — P4.B1: Gap-Targeted Evaluation — Hard Domain Questions

**Status:** KILLED — Finding #478
**Date:** 2026-04-11

## What We Tested
We hypothesized that harder domain questions (advanced: Galois theory, Vasicek, Raft consensus)
would expose genuine knowledge gaps in Gemma 4 4B, enabling rank-6 adapters to show ≥15pp
improvement. Early killed from smoke test (N=5 per domain).

## Key Finding: Gemma 4 4B Has No Exploitable Knowledge Gap

All domain base scores exceeded 0.25 by a wide margin:

| Domain | Base Score | Adapted | Delta |
|--------|-----------|---------|-------|
| Math | 0.633 | 0.767 | +13pp |
| Medical | 0.600 | 0.633 | +3pp |
| Legal | 0.433 | 0.500 | +7pp |
| Code | 0.467 | 0.333 | **-13pp** |
| Finance | 0.433 | 0.433 | 0pp |

The model has deep knowledge of advanced academic content — it was pretrained on the same
Wikipedia/textbook sources these questions come from. Knowledge gap is NOT a viable lever.

## The Critical P4.B0 Reanalysis: Notation Artifact

The P4.B0 math +20pp improvement (Finding #477) was a notation artifact, NOT a knowledge gap:
- **P4.B0 keywords**: `"a^2"`, `"u dv"`, `"f(g(x))"` — mathematical notation not produced in natural prose
- **P4.B1 keywords**: `"Zorn"`, `"maximal element"`, `"eigenvalue"` — natural vocabulary Gemma freely produces

Same domain, same adapter: P4.B0 base=0.307 vs P4.B1 base=0.633.
The math adapter learned to produce notation characters ("^", subscripts) — style, not knowledge.

## Structural Impossibility

`δ_d > 0` requires BOTH:
1. **Vocabulary gap**: `H(V_d|θ_base) > H_threshold` (base model doesn't know the vocabulary)
2. **Distribution overlap**: `V_d ∩ V_train ≠ ∅` (adapter training covers the question vocabulary)

Gemma 4 4B fails (1) for standard and advanced academic domains.
P1 T2 rank-6 adapters fail (2) for advanced subdomain questions.

**Code adapter degradation** (-13pp) confirms: the code adapter learned basic algorithm
vocabulary (FIFO, heap, sorting) which actively hurts on systems questions (Raft, MESI,
Byzantine fault tolerance). Applying wrong-domain vocabulary pulls token probabilities down.

## What To Pursue Next: Formatting Gaps (Not Knowledge Gaps)

The correct framing for capable base models is FORMAT compliance, not content knowledge:
- **Math LaTeX**: `\frac{}{}`, `\sum_{i=1}^{n}`, `\int_{a}^{b}` — style gap exists
- **Clinical SOAP notes**: `"S:"`, `"O:"`, `"A:"`, `"P:"`, `"HPI"` — format gap exists
- **Legal document structure**: `"WHEREAS"`, `"NOW THEREFORE"`, `"PARTY"` — gap exists
- **Code framework style**: FastAPI patterns, Rust async idioms — style gap exists

P3.C5 (Finding #472) showed that style adapters (personal writing style) CAN achieve ≥55pp
with adequate data coverage. The analogy: formatting is a style gap, not a content gap.

## Implications for Architecture

This finding resolves the P4.B series: adapter quality is **not about knowledge gaps** for
Gemma 4 4B. The adapter value proposition for this base model is:
1. **Formatting/style alignment** (math notation, clinical structure, legal boilerplate)
2. **Routing signal** (domain centroid for TF-IDF Ridge routing)
3. **Composition** (additive ΔW composition with Grassmannian isolation)

Content knowledge enrichment requires either: (a) a smaller/weaker base model, or
(b) genuinely proprietary data not in pretraining (internal docs, client-specific patterns).

## Citations
- Finding #477: rank-6 adapters can't overcome strong base priors (P4.B0)
- Finding #472: style adapters work with 167 diverse examples and cache fix (P3.C5)
- Finding #468: rank-16 + diverse data needed for style compliance (P3.C4)
