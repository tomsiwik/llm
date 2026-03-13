# Current Direction (2026-03-13)

## Completed This Iteration
- exp_distillation_pilot_50: **REVISED** per adversarial review (REVIEW-adversarial.md)
  - Status downgraded from "proven" to "supported"
  - Three required fixes applied:
    1. Contamination caveat added to PAPER.md after results tables
    2. Status downgraded in HYPOTHESES.yml, evidence claim updated
    3. Limitations section rewritten to state eval data IS training data
  - FINDINGS.md: moved from Conclusive to Supported section with caveat
  - VISION.md: updated table, roadmap, and gate criteria with contamination note
  - HYPOTHESES.yml meta: last_proven_node updated (no longer this experiment)
  - To fully prove: run MMLU subset or HumanEval evaluation on the 50 experts

## Active Tasks

### Task: exp_gpu_latency_validation (P4)
- **Status**: RUNNING on RunPod (nohup, restarted after crash fix)
- **Script**: `/workspace/llm/scripts/gpu_latency_bench.py`
- **Log**: `/workspace/gpu_latency_bench.log`
- **Results**: `/workspace/llm/results/gpu_latency_benchmark.json`
- **ETA**: ~30-60 min (model load + 3 phases x multiple N values)
- **Kill criteria**: Pre-merge >5% overhead at any N; dynamic top-k scales with N

## Check Commands (next iteration)
```bash
# Check latency benchmark
ssh runpod 'tail -30 /workspace/gpu_latency_bench.log 2>/dev/null; ls -la /workspace/llm/results/gpu_latency_benchmark.json 2>/dev/null || echo "not done yet"'

# Process status
ssh runpod 'ps aux | grep gpu_latency | grep -v grep'
```

## After Results Are Available
1. Pull latency results: `scp runpod:/workspace/llm/results/gpu_latency_benchmark.json results/`
2. Update `micro/models/inference_latency_gpu/PAPER.md` with actual numbers
3. Update HYPOTHESES.yml: exp_gpu_latency_validation

## Budget Tracking
- pilot50_bench: ~$0.09 (done)
- latency_bench: ~$0.09 (running)
- Total this session: ~$0.18
