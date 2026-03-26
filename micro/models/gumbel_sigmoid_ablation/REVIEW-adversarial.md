# Peer Review: Gumbel-Sigmoid Routing Ablation

## NotebookLM Findings

Skipped -- not authenticated. Review conducted via direct code and results inspection.

## Mathematical Soundness

### Load-Balancing Loss: Comment Contradicts Implementation

Line 307 of `run_experiment.py` says "Loss: sum of squared deviations from uniform" but the actual implementation on line 309 is:

```python
lb_loss = N * mx.sum(gate_means * uniform)  # Switch Transformer style
```

This computes `N * sum(gate_means * (1/N)) = sum(gate_means)`, which equals the mean activation level across all experts. This is NOT the Switch Transformer load-balancing loss. The Switch Transformer loss is `N * sum(f_i * p_i)` where `f_i` is the fraction of tokens dispatched to expert i and `p_i` is the mean gate probability for expert i. Here, with batch_size=1, `gate_means = gates[0]` (the gate vector itself), so `lb_loss = sum(gates[0])` -- this penalizes total gate activation, not imbalance. It acts as an L1 regularizer on gate outputs, not a balance enforcer.

Why it accidentally works: penalizing total activation with sigmoid gates pushes logits negative for non-target experts, which indirectly forces the router to be more selective. Combined with the BCE target loss that pushes the correct gate up, the net effect is: raise target gate, suppress all others. This is load-balancing-adjacent but for the wrong reason. The comment, the MATH.md description, and the code are three different things.

**Severity: Medium.** The mechanism produces a useful result but the claimed mechanism (Switch Transformer load-balancing) is not what is implemented. The paper's explanation of WHY load-balancing fixes zero-accuracy domains may be incorrect -- it could be L1 regularization improving generalization rather than load-balancing preventing collapse.

### BCE Target Construction: Batch Dimension Issue

Line 293:
```python
target = mx.zeros((h_batch.shape[0], N))
target = target.at[:, target_idx].add(1.0)
```

Since `target_idx` is a scalar (the domain index), this sets column `target_idx` to 1.0 for ALL rows in the batch. This is correct when batch_size=1 (always the case here, line 327: `h_batch = h_all[batch_idx:batch_idx + 1]`), but would be a bug with mixed-domain batches. Not blocking since batch_size=1 throughout.

### Gumbel Noise Formula: Correct

Line 204: `gumbel_noise = -mx.log(-mx.log(u))` with `u ~ Uniform(1e-6, 1-1e-6)` -- standard Gumbel(0,1) sampling. Correct.

### Temperature Annealing: Correct

Linear interpolation from start to end. Matches MATH.md.

### MATH.md Parameter Count Discrepancy

MATH.md claims router with h=256: `2560*256 + 256 + 256*49 + 49 = 668,977`. The code uses `nn.Linear(input_dim, hidden_dim)` + `nn.Linear(hidden_dim, n_adapters)`. With d=2560, h=256, N=49: params = 2560*256 + 256 + 256*49 + 49 = 655,360 + 256 + 12,544 + 49 = 668,209. The MATH.md has an arithmetic error (668,977 vs 668,209). This was already noted in prior N=50 caveats. Minor.

### Inference Evaluation Uses Logits, Not Gates

Line 373: `logits, _ = router(h_val, hard=True)` -- evaluation ranks by raw logits, discarding the gate values. This is correct for sigmoid (raw logit > 0 means gate > 0.5, and logit magnitude reflects confidence). But note: during training with Gumbel noise, the router learns to produce logits that are correct AFTER noise addition. At inference, noise is removed. For well-trained routers this is fine (logits converge to large magnitudes), but for marginal domains the train/inference distribution mismatch could hurt. This is standard practice and not a flaw, but it contributes to the brittleness on borderline domains.

## Novelty Assessment

### Prior Art

This is a standard hyperparameter ablation study for mixture-of-experts routing. The individual components (Gumbel-sigmoid, Gumbel-softmax, load-balancing loss, temperature annealing, straight-through estimation) are all well-established:

- Gumbel-Softmax: Jang et al. 2017, Maddison et al. 2017
- Switch Transformer load-balancing: Fedus et al. 2022
- Non-competing sigmoid routing: L2R (cited), recent MoLoRA work

The novelty is in the specific application context (ternary LoRA adapters on BitNet-2B at N=49) and the diagnostic analysis of zero-accuracy domains (hidden state cosine similarity + intra-domain variance). The diagnostic findings are the most valuable contribution -- they identify the failure mode taxonomy (expert collapse vs. unroutable domains) and show that these have different fixes.

### Delta Over Closest Work

The zero-accuracy domain analysis and the identification of two distinct failure modes (collapse fixable by load-balancing, unroutability requiring architectural change) is a useful contribution within the project. No novel mechanisms proposed.

## Experimental Design

### Critical: Single Seed, Low Statistical Power

All 21 configurations use seed=42. With 10 val samples per domain (490 total), the difference between 85.10% and 90.00% is 24 additional correct predictions out of 490. For a binomial test: p-value for observing 441/490 vs 417/490 under the null that both have the same accuracy is approximately 0.004, which is statistically significant. However, the seed affects both training (random domain sampling, Gumbel noise) and validation (same fixed set). The paper acknowledges this limitation.

### Top-k=4 "Improvement" Is Partially Mechanical

The highest-scoring config is k=4 (90.82%), but increasing k mechanically increases top-k accuracy by giving more chances to include the correct domain. The fair comparison is between configs at the same k. At k=2: baseline 85.10% vs lb+6000steps 90.00% (+4.90pp). This is the meaningful comparison and the paper correctly identifies it.

### Load-Balancing + 6000 Steps Confounds Two Variables

The "best" k=2 config changes both load_balance_alpha (0 -> 0.1) and n_steps (3000 -> 6000). Looking at the leaderboard: lb=0.1 at 3000 steps scores 83.88% (rank 12, WORSE than baseline). This means the improvement comes primarily from more training steps, not load-balancing per se. The lb=0.1 config needs 6000 steps to overcome the regularization penalty from the L1-like auxiliary loss. The paper does not isolate these effects -- a "6000 steps, no load-balancing" control is missing. This is a significant experimental design gap.

**Missing control: baseline at 6000 steps.** Without this, we cannot attribute the improvement to load-balancing vs. simply training longer.

### Hidden State Caching: Acknowledged Limitation

Hidden states are extracted from the base model without any LoRA adapters applied. This is acknowledged and standard for the project, but means the routing operates on representations that do not reflect the adapted model. In a deployed system, the base model's hidden states would be modified by the always-on instruction adapter, potentially changing the domain separability landscape.

### Per-Domain Accuracy at 10 Samples

With 10 validation samples per domain, domain-level accuracy granularity is 0%, 10%, 20%, ..., 100%. The difference between "0% accuracy" and "10% accuracy" is a single correct prediction. The chemistry recovery from 0% to 100% (10/10 correct) is robust. The wikitext recovery from 0% to 40% (4/10) is more fragile. The paper's claim that load-balancing "fixes 3/4 zero-accuracy domains" should be read as "moves 3/4 domains from 0/10 to at least 4/10 correct" -- directionally useful but not high-confidence at per-domain level.

### Softmax Zero-Accuracy Domains Differ

An underexplored finding: softmax (rank 4, 86.12%) has only 1 zero-accuracy domain (wikitext), while baseline sigmoid has 2 (wikitext, dialogue). The softmax router routes dialogue at 20% while sigmoid cannot. This suggests that competing gates naturally prevent the total collapse that independent gates suffer from (if one expert gets too much probability mass, softmax mechanically reduces all others). This is briefly noted but deserves more attention.

## Hypothesis Graph Consistency

The experiment does not appear in HYPOTHESES.yml. The kill criterion from the script header is:

> K1 (id=264): No config beats current default (86.33% top-2 accuracy) by >5% -> KILL

The results claim K1 PASS based on beating the in-experiment baseline (85.10%, not 86.33%). The paper honestly notes this discrepancy: "comparing against the N=50 original (86.33%): the improvement over that is 3.67% (lb+6000) or 4.49% (k=4). This is borderline."

Using the original 86.33% baseline, no configuration exceeds the 5% threshold. K1 would be a borderline FAIL under the original criterion. The paper's honest acknowledgment partially mitigates this, but the K1 PASS claim in results.json (`"k1_result": "PASS"`) uses the lower in-experiment baseline, which is favorable self-comparison.

## Macro-Scale Risks (advisory)

1. **Load-balancing implementation needs fixing.** The current "load-balancing" is actually L1 gate regularization. At macro scale with proper batching (batch_size > 1), this would need to be replaced with actual Switch Transformer load-balancing (per-batch routing statistics). The current form would behave differently.

2. **6000 steps may not be sufficient at N=100+.** The improvement from 3000 to 6000 steps suggests the router is undertrained. At higher N, convergence would take proportionally longer.

3. **Dialogue-type domains remain architecturally unroutable.** Mean-pooled hidden states fundamentally cannot separate high-variance domains. Per-token or attention-weighted routing is needed. This is correctly identified.

4. **Hidden state similarity > 0.99 between some domain pairs.** Chemistry/science_qa (0.992) and wikitext/history (0.996) are nearly indistinguishable to the router. At macro scale with more similar domains, this problem worsens. Hierarchical routing (first coarse, then fine) may be needed.

## Verdict

**REVISE**

The experiment is a solid ablation study with useful diagnostic findings, but has specific issues that should be addressed before the conclusions are committed to FINDINGS.md.

### Required Fixes

1. **Fix load-balancing loss description or implementation.** Either (a) update MATH.md and PAPER.md to correctly describe what the code does (L1 gate regularization, not Switch Transformer load-balancing), or (b) implement actual Switch Transformer load-balancing and re-run. The mismatch between documentation and code is a scientific integrity issue. The comment on line 307 ("sum of squared deviations from uniform") describes neither the Switch Transformer loss nor what the code computes.

2. **Add a "baseline at 6000 steps" control.** Run the baseline config (anneal 2->0.5, k=2, no load-balancing) at 6000 steps. This is necessary to separate the effect of more training from the effect of the auxiliary loss. Without this, the claim "load-balancing + more training is the single most impactful change" is confounded.

3. **Correct the K1 assessment.** Either (a) use the original 86.33% baseline from N=50, which makes K1 borderline FAIL (max improvement 4.49%), or (b) explicitly redefine the baseline in the experiment setup with justification for why the in-experiment baseline (85.10%) is more appropriate. The results.json `k1_result: PASS` is misleading as currently stated.

4. **Add HYPOTHESES.yml entry.** The experiment has no entry in the hypothesis graph. It should be added with correct kill criteria and evidence, and the status should reflect the corrected K1 assessment.

### Advisory (Not Blocking)

5. The softmax router's natural resistance to domain collapse (dialogue 20% vs sigmoid 0%) deserves explicit discussion as a finding. Softmax at 86.12% with 1 zero-acc domain may be preferable to sigmoid at 85.10% with 2 zero-acc domains, even at lower headline accuracy.

6. State explicitly that the "recommended configuration" has not been validated on downstream composition PPL -- routing accuracy improvement does not guarantee composition quality improvement.
