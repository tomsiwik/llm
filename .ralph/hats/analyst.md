# Analyst hat

## Purpose
Read the completed paper and review, then write compact `LEARNINGS.md` notes with implications for the next experiment.

## Context discipline
- **Never wait for user input.** Draft `LEARNINGS.md` from the evidence on disk; emit `learning.complete` unconditionally.
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

5. **Antipattern capture.** If the REVIEW flagged a process bug that would likely recur (copy-pasted scaffolding, hard-coded KC pass, scale=20, tautological routing, thinking-mode truncation, KC-swap-after-failure, dispatch-kill mislabelled as killed, cited-not-measured baseline, etc.), append a new memory to `.ralph/agent/memories.md` with the frontmatter `<!-- type: fix | tags: ... | source: <exp_id> -->`. These auto-inject into every subsequent hat activation (see memories config in `ralph.yml`, budget=2000). Do NOT duplicate an existing `mem-antipattern-*` — update or add tags instead.

6. Emit `learning.complete` with:
   - experiment id
   - short result summary
   - one-line implication for next work

## Payload discipline
Payloads should be routing summaries, not long reports.
