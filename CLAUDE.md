# Project: Composable Ternary Experts

## Platform
Apple M5 Pro, 48GB. MLX only. No CUDA.

## Before Writing MLX Code
Invoke `/fast-mlx` and `/mlx-dev` skills first.

## Before/After Experiments
```bash
experiment claim <worker-id>                    # get next experiment + full details
# ... run experiment ...
experiment complete <id> --status supported \   # finish in one shot
  --dir micro/models/<name>/ \
  --k 183:pass --k 184:fail \
  --evidence "K1 PASS: metric=value"
```

## Recording Findings
```bash
experiment finding-add \
  --title "..." --status conclusive|supported|killed|provisional \
  --result "..." --caveat "..." \
  --failure-mode "What degenerate behavior?" \
  --impossibility-structure "What math makes it impossible?" \
  --experiment <id> --scale micro|macro

experiment finding-list [--status conclusive]   # list all findings
experiment finding-get <id>                     # full details
experiment finding-update <id> --status killed  # update
experiment query "search term"                  # searches experiments + evidence + findings
```

## Proof-First Research (Constructive Mathematics)
Every experiment follows this order — no exceptions:

1. **Derive the proof** — Theorem/Proof/QED in MATH.md BEFORE any code
2. **Ground in prior math** — cite existing theorems (JL-lemma, Welch bound, etc.)
3. **Predict specific numbers** — the proof must make quantitative predictions
4. **Run experiment to verify** — measurements confirm or refute the proof
5. **Kill criteria from proof** — not arbitrary thresholds

An experiment that "improves a metric" without a theorem predicting the
improvement is not a finding. PAPER.md must contain a prediction-vs-measurement table.

Experiment types:
- **Verification** — proof complete, experiment confirms predictions
- **Guided exploration** — proven framework, unknown parameter/function to discover
- **Frontier extension** — proven result being extended into new territory

Finding status definitions:
- `conclusive` = formal proof verified, all predictions match (Type 1 only)
- `supported` = proof mostly verified, or exploration narrowed an unknown (Type 1-2)
- `provisional` = frontier extension, or empirical observation awaiting proof (Type 2-3)
- `killed` = proof's predictions refuted, or proof found incorrect

Architecture reference: https://sebastianraschka.com/llm-architecture-gallery/

## Hypothesis Generation Rules
Every new experiment MUST cite an arxiv paper or prior finding.
No data-structure analogies. No inventing mechanisms without paper evidence.
Killed experiments: derive what structure makes the failure impossible, then re-test.

## Key Files
- `PROMPT.md` — research loop instructions
- `VISION.md` — architecture, proven results, readiness
- `experiment finding-list` — all findings (DB-backed, queryable)
- `experiment query "<term>"` — full-text search across experiments + evidence + findings
- `CODING_GUIDELINES.md` — mandatory MLX memory/cleanup patterns
- `.ralph/current_direction.md` — what's in progress
