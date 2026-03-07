# Current Direction: REVISE exp_persistent_expert_tree

## Task
Fix 6 issues identified by adversarial reviewer. The persistent tree data structure
is well-implemented; the experiment just needs to actually USE it.

## Key Fixes
1. Correct memory reporting (200%, not 100%)
2. Use persistent tree API (update_leaves/compose_versions/set_active_version)
3. Fine-tune leaves only (freeze embeddings, attention, norms, lm_head)
4. Report memory_report() from tree API
5. Add flat-dict baseline
6. Add same-version cherry-pick control (isolate version-crossing factor)

## Kill Criteria
- cross-version composition >5% worse than same-version
- persistent structure overhead >15% memory vs mutable (leaf-only regime)
