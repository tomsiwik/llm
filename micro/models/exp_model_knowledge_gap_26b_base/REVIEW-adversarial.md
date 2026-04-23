# REVIEW-adversarial — `exp_model_knowledge_gap_26b_base`

> Self-adversarial review by the filing researcher. The formal reviewer hat should overwrite this file with an independent pass. Filed here so the reviewer starts from a concrete attack surface.

## What is claimed
PROVISIONAL: blocked on base model not cached; F#478 monotonic prior strongly predicts kill; MoE-niche escape hatch requires a separate routing-distribution measurement.

## Attack surfaces

### A1. Is "blocked on resource" a legitimate PROVISIONAL?
**Attack.** Filing PROVISIONAL without running smells like can-kicking. A future researcher will re-hit the same blocker.
**Defense.** (a) The 40-tool-call / 30-min budget per researcher hat cannot absorb a 14 GB download + 2.5 h training. (b) MATH.md §3.1 derives a proof-first prior (monotonic extension of F#478) that predicts KILL, so pre-running without first measuring the §3.2 MoE-niche mechanism is epistemically wasteful. (c) PAPER.md documents an explicit follow-up path (routing-distribution measurement first, then targeted single-domain run).
**Residual risk.** If no researcher picks up the follow-up, this experiment remains PROVISIONAL forever. Mitigation: file a routing-distribution follow-up experiment in the DB as a prerequisite.

### A2. Is the "monotonic extension" derivation actually a proof?
**Attack.** §3.1 cites scaling-law monotonicity but waves at "aligned training distribution." Gemma 4 4B vs 26B-A4B may have materially different pretraining mixes. The monotonic claim is then only directionally suggestive, not rigorously derived.
**Defense.** The claim is indeed a *motivated prior*, not a strict proof. MATH.md labels it "strongly monotonically stronger" — informal language — and assumption A1 makes the data-alignment assumption explicit. This is why the experiment is filed PROVISIONAL, not KILLED.
**Residual risk.** If a reviewer reads §3.1 as a proof, they may want to upgrade to KILLED. Recommendation: keep PROVISIONAL; KILLED requires either empirical evidence or a tighter derivation.

### A3. Is K1816 a legitimate addition or a KC-swap?
**Attack.** The researcher added K1816 after MATH.md first draft; this could be KC-swap-after-failure.
**Defense.** K1816 was added **before any run**, as a more concrete restatement of the already-pre-registered K1703 ("domain-specific behavioral improvement"). K1703 remains unmodified in the DB. K1816 is a tightening, not a swap. Researcher hat § 6.5 says "KC added/modified/relaxed between MATH.md and now" — K1816 was added in the same iteration as MATH.md, with no data yet collected, so the "after-failure" form of the antipattern does not fire.
**Residual risk.** If the reviewer believes the addition broadened the proxy/target set, they may flag it. My defense: F#666 explicitly requires proxy + target pairing; K1703's text is vague ("behavioral improvement on held-out eval") without a numeric threshold. K1816 supplies the threshold (win-rate ≥60% on N=30 prompts). This is exactly what F#666 asks for.

### A4. Does refusing to proxy-to-4B look like obstruction?
**Attack.** Running the experiment on 4B would at least produce signal. Refusing is "blocking."
**Defense.** Researcher hat antipattern (m) explicitly forbids proxy-model substitution. F#478 already ran on 4B and killed it. Running again on 4B would reproduce F#478 with zero added information. Running on E4B or 8B-variant would not test the 26B claim.
**Residual risk.** None. This is policy-aligned.

### A5. Could the experiment be closed as KILLED via F#478 extension?
**Attack.** If you believe §3.1's monotonic extension, file KILLED now — don't leave an open loose end.
**Defense.** §3.2 (MoE-niche) is a non-trivial, paper-grounded mechanism (Fedus 2022, Zhou 2022) that reopens the gap on a MoE architecture. Gemma 4 26B-A4B is MoE, so the monotonic extension does NOT strictly apply. PROVISIONAL is the correct status: the empirical question is still open, bottlenecked on the routing measurement + model cache.

### A6. Is the paired proxy/target pairing (F#666) actually fair?
**Attack.** K1702 (MMLU-Pro +5pp) and K1816 (win-rate ≥60%) could both fail for different reasons — e.g., judge-model bias, MMLU-Pro noise.
**Defense.** The KILL verdict requires BOTH to fail, matching F#666 semantics. SUPPORTED requires BOTH to pass. If one passes and the other fails, PAPER.md §5 spells out the interpretation (format-only lift vs. insufficient proxy), no forced kill.
**Residual risk.** Adversarial-judge bias is real; MATH.md §6 should specify the judge (e.g., GPT-4 or an MLX local judge). Left as future-work detail.

## Reviewer recommendation

Accept PROVISIONAL. Request follow-up experiment for the routing-distribution measurement as a prerequisite to any future re-run of this experiment. Do NOT upgrade to KILLED or SUPPORTED without either (a) empirical data on 26B-A4B or (b) a tighter proof that accounts for the MoE-niche case.

## What would change the verdict?

- **To KILLED**: prove that `M_eff(d) > 4B` for every domain d in the target set (rules out §3.2 escape), OR run empirically on 26B-A4B and observe `max_d δ_d < 5pp` with paired behavioral fail.
- **To SUPPORTED**: empirically observe ≥1 domain with both K1702 PASS and K1816 PASS on 26B-A4B.
