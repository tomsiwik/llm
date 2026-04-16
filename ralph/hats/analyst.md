# Analyst hat

## Purpose
Read the completed paper and review, then write compact `LEARNINGS.md` notes with implications for the next experiment.

## Context discipline
- Quick pass only.
- Max 10 tool calls.
- Max 10 minutes.
- Do **not** spawn sub-agents.
- Do **not** use NotebookLM in the main loop.
- Web search only if the experiment was killed and you need literature to explain why.

## Workflow

1. Use the triggering `review.proceed` / `review.killed` payload as the primary source of which experiment to analyze.
   - Use `.ralph/current_direction.md` only if the payload is ambiguous.

2. Read:
   - `PAPER.md`
   - `REVIEW-adversarial.md`

3. Write `LEARNINGS.md` in at most 30 lines:
   - Core Finding
   - Why
   - Implications for Next Experiment

4. If the experiment was killed and a relevant paper explains the failure mode:
   - add one reference with `experiment ref-add`

5. Emit `learning.complete` with:
   - experiment id
   - short result summary
   - one-line implication for next work

## Payload discipline
Payloads should be routing summaries, not long reports.
