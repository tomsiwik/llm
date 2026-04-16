# LEARNINGS.md — P11.L0: RSD Aligned Traces

## Core Finding

NLL-filtered s1K traces ("practical RSD") provide a cheap quality gate (one forward pass per trace) for reducing distribution shift between teacher-generated traces and the 4-bit student model. This is a design-stage experiment — full run results TBD.

## Why

True Rejection Sampling Decoding (von Neumann 1951; Liu et al. arXiv:2309.06657) requires the teacher's log-probs, which are unavailable locally. The proxy — filtering on absolute student NLL ≤ 6.9 (250× chance level at vocab_size=256k) — is directionally correct: d_TV(D_rsd, π_S) ≤ d_TV(D_raw, π_S) by construction, but the acceptance rate may be trivially high.

## Key Design Decision

- Separate data subdirs `data/rsd/` and `data/sert/` with standard `train.jsonl`/`valid.jsonl` — mlx_lm.lora only accepts `--data DIR` (no `--train-splits`/`--val-splits` flags).
- Two adapter conditions: RSD-filtered s1K (cross-domain traces) vs. SERT (self-generated GSM8K traces). SERT should show strong GSM8K specialization but limited breadth.

## Implications for Next Experiment

If K1541 passes (RSD adapter ≥ raw+3pp on MMLU-Pro), NLL filtering is a reusable quality gate for all future domain adapter training data pipelines. If it fails, the threshold (6.9) may need tuning or the experiment collapses to "NLL filtering has no effect" — derive impossibility from the gap between absolute and ratio-based acceptance criteria.
