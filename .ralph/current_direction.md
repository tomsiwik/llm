# Current Direction: P1 — T4.5 Adapter Format Compatibility (COMPLETE)

## Active Experiment: exp_p1_t4_tfidf_routing_gemma4

**Purpose:** Validate TF-IDF nearest-centroid routing for N=5 and N=25 domain adapters.
Zero neural parameters. Sub-ms CPU latency.

**Kill criteria:**
- K1073: N=5 routing accuracy >= 95%
- K1074: N=25 routing accuracy >= 85%
- K1075: p99 latency < 1ms on CPU (smoke test: 0.61ms PASS)
- K1076: Zero LLM params added (structural guarantee)

**Design:**
- N=5 real domains: math (GSM8K), code (HumanEval), medical (PubMedQA), legal (MMLU professional_law), finance (MMLU macroeconomics)
- N=25: add 20 MMLU subjects (carefully selected to avoid biomedical/physics overlap)
- N_TRAIN=300 per domain, N_TEST=100 per domain
- TF-IDF(max_features=20000, ngram=(1,2)) + nearest centroid

**Smoke test results (N=30 per domain):**
- N=5: 86.7% (low at N=30, expected to hit 95%+ at N=300)
- N=25: 66.9% (low at N=30, expected to hit 85%+ at N=300)
- K1075 latency: 0.61ms PASS
- K1076: 0 LLM params PASS

**Expected runtime:** ~10-15 min (pure Python, no model loading needed)

**Blocks:** T4.2 (LSH routing), T4.3 (e2e latency)

## T3 Tier Complete
T3.1 KILLED: Simultaneous N=5 activation catastrophic (math 82→8%)
T3.2 KILLED: Scale ≥ 12 degrades MMLU; scale=6 safe
T3.3 SUPPORTED: Power law alpha=0.15, routing load-bearing
T3.4 SUPPORTED: N=25 Grassmannian max|cos|=2.2e-8
T3.6 SUPPORTED: Hot-add bit-exact, 0.004ms, immediate
T3.7 SUPPORTED: Hot-remove bit-exact, 0.001ms, slot reusable

## T4.3 Complete (Finding #432)
exp_p1_t4_vllm_adapter_serving: SUPPORTED
- K1081 PASS: 5/5 adapters load+generate
- K1082 PASS: p99=4.77ms (10.5× margin under 50ms budget)
- K1083 PASS: 90.8% throughput (predicted 99.5%, gap = LoRALinear double-read)
- K1084 PASS: routing <1μs, 5/5 correct
- NOTE: vLLM is CUDA-only; MLX-native hot-swap is the right serving primitive
- NEXT: emit experiment.done → Reviewer writes REVIEW-adversarial.md
