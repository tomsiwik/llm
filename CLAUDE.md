# Project: Composable Ternary Experts

## Platform
Apple M5 Pro, 48GB. MLX only. No CUDA.

## Before Writing MLX Code
Invoke `/fast-mlx` and `/mlx-dev` skills first.

## Running Experiments
```bash
experiment claim <worker-id>                    # get next experiment + full details
# Run via pueue (MANDATORY — never bare `uv run python`):
experiment run <id>                             # run by experiment ID (looks up experiment_dir)
experiment run micro/models/<name>/run_experiment.py  # run by path
experiment run --no-wait <id>                   # submit and return immediately
experiment run --status                         # check queue
experiment run --kill <id>                      # kill running experiment
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
- `VISION.md` — v1 architecture, proven results, readiness (473 experiments)
- `../VISION_P1.md` — P1 enhanced vision: Gemma 4 + polar adapters + PLE-M2P
- `../ARCHITECTURE_P1.md` — complete mathematical reference for P1 (every Gemma 4 mechanism formalized)
- `docs/MLX_GEMMA4_GUIDE.md` — HOW TO: load, train, adapt Gemma 4 on MLX (verified commands + code)
- `experiment finding-list` — all findings (DB-backed, queryable)
- `experiment query "<term>"` — full-text search across experiments + evidence + findings
- `experiment list -t p1` — Pierre P1 experiment collection (40 experiments, 7 tiers)
- `CODING_GUIDELINES.md` — mandatory MLX memory/cleanup patterns
- `.ralph/current_direction.md` — what's in progress

## Pierre P1 Experiments
P1 experiments are tagged `p1` in the experiment DB. Start with T0 (math foundation):
```bash
experiment list -t p1 -t t0-foundation    # 5 foundation experiments
experiment list -t p1 -t critical-path    # critical path only
experiment claim <worker> --tag p1        # claim next P1 experiment
```

## Gemma 4 on MLX (Quick Reference)
```bash
# Inference
uv run python -m mlx_lm.generate --model mlx-community/gemma-4-e4b-it-4bit --prompt "test"

# LoRA training
uv run python -m mlx_lm.lora --model mlx-community/gemma-4-e4b-it-4bit --train --data ./data/ --iters 1000

# With adapter
uv run python -m mlx_lm.generate --model mlx-community/gemma-4-e4b-it-4bit --adapter-path ./adapters/math/
```
Full guide: `docs/MLX_GEMMA4_GUIDE.md`
