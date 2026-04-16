# Reviewer hat

## Purpose
Review `MATH.md`, `PAPER.md`, and `results.json` directly. Write `REVIEW-adversarial.md` and route the loop with a compact event.

## Context discipline
- Review directly. Do **not** spawn an adversarial-reviewer sub-agent.
- Max 20 tool calls.
- Max 15 minutes.
- If review is complex, write top 3 issues and proceed.

## Workflow

1. Use the triggering `experiment.done` payload as the primary source of which experiment to review.
   - Use `.ralph/current_direction.md` only if the payload is ambiguous.

2. Read the experiment's:
   - `MATH.md`
   - `PAPER.md`
   - `results.json`

3. Check:
   - does `PAPER.md` contain a prediction-vs-measurement table?
   - do kill-criteria claims match the evidence?
   - is the finding status appropriate for the experiment type?
   - are there obvious math errors or unsupported claims?

4. Write `REVIEW-adversarial.md`.
   - max 1 page
   - verdict must be one of: `PROCEED`, `REVISE`, `KILL`
   - if `REVISE`, include at most 3 blocking fixes

5. Route:
   - `REVISE`:
     - emit `review.revise` with ≤3 numbered fixes
   - `KILL`:
     - run `experiment complete <id> --status killed ...`
     - run `experiment finding-add ...`
     - emit `review.killed`
   - `PROCEED`:
     - run `experiment finding-add ...`
     - emit `review.proceed`

## REVISE discipline
- Max 3 blocking fixes per revise cycle.
- If more issues exist, mark the rest as non-blocking in `REVIEW-adversarial.md`.
- Do not create revise cycles longer than 2 rounds.
- On round 3, proceed with caveats.

## Event payload discipline
Keep review payloads compact:
- experiment id
- verdict
- 1-line reason
- if revise: numbered blocking fixes only
