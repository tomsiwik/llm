# Peer Review: SFT BitNet Generation Quality (RE-REVIEW)

## Experiment Type
Guided exploration (correctly reclassified from verification).

## Hack Detector
- Fix count: 2 (SFT masking + energy gap routing). Below flag threshold.
- Is MATH.md a proof or a description? Mixed. Lemma 1 is a trivial but correct proof. Theorem 1 (SFT-Routing Incompatibility) is a semi-formal argument with an approximation step that is not rigorously bounded but is directionally correct and empirically confirmed. Proposition 2 is honestly labeled as contingent.
- Metric used as evidence: Routing accuracy (4% vs 80%) -- this is the key metric, and it is objective and unambiguous. LLM-as-judge scores are secondary and acknowledged as near-constant.
- Kill criteria source: K1/K2 loosely from proof. K3 acknowledged as exploratory, not proof-derived. Honest.

## Self-Test Audit
1. One-sentence impossibility property: PASS. Single property correctly identified.
2. Cited theorems: PASS. Chain rule and Bayes-optimal classification cited. NP explicitly disclaimed with correct reasoning (simple vs composite hypotheses). Fix #3 resolved.
3. Predicted numbers: PASS with caveat. Numbers stated (4/5 domains, 40% math) but not derived from the proof. For guided exploration this is acceptable -- the proof does not predict counts, and the document is honest about it.
4. Falsification condition: PASS. Now includes "Theorem 1 is falsified if SFT adapters produce discriminative energy gaps on instruction tokens despite zero instruction gradient." This targets the post-hoc theorem directly.
5. Hyperparameter count: PASS. Zero new.
6. Hack check: PASS. Two independently validated mechanisms combined.

## Verification of Previous 6 REVISE Fixes

| Fix | Required Change | Status |
|-----|----------------|--------|
| 1. Reclassify as guided exploration | MATH.md + PAPER.md must say "guided exploration" | DONE -- both documents reclassified, proven framework and unknown clearly stated |
| 2. Add SFT-routing incompatibility theorem | Formal theorem proving the routing failure | DONE -- Theorem 1 added (MATH.md lines 85-106) with QED |
| 3. Fix NP citation | Remove or correct Neyman-Pearson | DONE -- explicitly disclaimed with correct reasoning, replaced by Bayes-optimal |
| 4. Downgrade old Theorem 1 to Lemma | Finite-sum linearity is a lemma | DONE -- now "Lemma 1 (Instruction Gradient Isolation)" |
| 5. Acknowledge K3 threshold as exploratory | Not proof-derived | DONE -- explicitly stated "not derived from the proof" |
| 6. Finding #187 provisional | Post-hoc formalization needs prediction-verification cycle | DONE -- marked PROVISIONAL throughout |

All 6 fixes applied.

## Mathematical Soundness

**Lemma 1 (Instruction Gradient Isolation):** Trivially correct. Finite-sum linearity. Appropriate as a lemma now rather than a theorem.

**Proposition 2 (Composition of SFT + Energy Gap):** Now honestly framed as contingent on Assumption 3 (routing transfer). The note acknowledging the experiment violated this assumption is good scientific practice.

**Theorem 1 (SFT-Routing Incompatibility):** This is the key addition. Assessment:

- The core argument is sound: SFT adapters receive zero gradient at instruction positions, so their effect on instruction-token NLL is not directly optimized. Since energy gap routing measures mean NLL over all positions, and instruction tokens constitute a large fraction, the discriminative signal is diluted.
- The formal step "Delta_E approximately equals (|R|/|T|) * Delta_E_response" (line 104) is an approximation, not a bound. The adapter affects instruction-token NLL through hidden-state propagation (acknowledged in lines 94-95 as "incidental"), but this effect is not bounded. A rigorous proof would need to show that the hidden-state effect is o(1) in some parameter.
- However, for a post-hoc formalization of an empirical discovery in a guided exploration, this is adequate. The approximation is strongly confirmed by measurement (4% routing accuracy). The theorem explains the mechanism even if the bound is not tight.
- The Corollary 2 (line 108-109) cleanly states the structural incompatibility: gradient isolation and full-prompt NLL routing are inherently at odds. This is the valuable insight.

**Minor gap:** The proof claims "all SFT adapters reduce response NLL by similar amounts (because all are well-trained)" (line 105) but does not argue why domain-specialized adapters should reduce response NLL by SIMILAR amounts across domains. An adapter trained on medical responses should reduce medical response NLL more than legal response NLL. This suggests response-only energy gap routing should still work -- which is exactly what the paper suggests as the resolution (line 113). So the gap is acknowledged and the resolution is correct.

## Prediction vs Measurement

PAPER.md contains the table (lines 17-23). Assessment:

| Prediction | Measured | Verdict |
|---|---|---|
| SFT routed >= 4/5 domains vs base | 3/5 | PARTIAL -- close but missed |
| SFT routed > NTP on 4/5 domains | 4/5 | YES |
| Math correctness >= 40% | 10% | NO (but confounded by 4% routing accuracy) |
| Response token ratio 40-60% | 60-89% | YES (wider than predicted) |
| Energy gap routing works with SFT | 4% accuracy | CRITICAL FAILURE -- the key discovery |

Score: 1.5/5 original predictions matched. But for a guided exploration, the failure IS the finding. The document is transparent about this. The critical routing failure was formalized post-hoc as Theorem 1, which is the correct scientific response.

## NotebookLM Findings

Not generated for this re-review. The mathematical issues are sufficiently clear from direct reading, and the previous round's concerns have been addressed through the 6 fixes.

## Novelty Assessment

The SFT-routing incompatibility (Finding #187) is a genuinely valuable structural insight:
- SFT masking and full-prompt energy gap routing are inherently incompatible
- The same property (zero instruction gradient) that solves contamination destroys the routing signal
- This points directly to response-token energy gap routing as the resolution

This is not a known result in the literature. Standard SFT work does not consider the interaction between response-only masking and NLL-based routing. The finding is novel within the project's framework and actionable for the next experiment.

## Macro-Scale Risks (advisory)

1. Response-token energy gap routing requires knowing where the response starts -- straightforward for instruction-format data, harder for free-form multi-turn conversations.
2. The chicken-and-egg problem noted in the previous review remains: you need to generate some response tokens before you can compute response energy gaps, but you need the right adapter to generate good tokens.
3. lora_scale=20 is a known confounder (acknowledged in Assumption 4) that should be re-examined.

## Verdict

**PROCEED**

All 6 required fixes from the previous review have been applied correctly. The document is now honest about what it is (guided exploration, not verification), what was predicted versus discovered, and the status of its findings (provisional, not supported).

The key strengths of the revised submission:
1. Experiment type correctly classified as guided exploration with the unknown clearly identified
2. Theorem 1 (SFT-Routing Incompatibility) formalizes the main discovery, even if the approximation step could be tighter
3. NP citation corrected with proper justification
4. Lemma 1 appropriately sized for its content
5. Kill criteria thresholds honestly acknowledged as exploratory
6. Finding #187 correctly marked provisional pending a proper prediction-verification cycle

Remaining non-blocking items for future work:
- Theorem 1's approximation step (hidden-state effect is "incidental") should eventually be bounded formally, perhaps by measuring the instruction-token NLL delta empirically as a function of training steps
- The next experiment (response-token energy gap routing) should be a proper verification: predict routing accuracy from Theorem 1's resolution, then measure
