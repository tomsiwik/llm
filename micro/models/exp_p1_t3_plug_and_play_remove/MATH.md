# T3.7: Hot-Remove Adapter Without Affecting Remaining Adapters

## Setup

Registry R = {domain_k → path_k}_{k=1}^{N}.  
Exclusive routing: for query x tagged to domain j, the system loads adapter j only.  
Remove operation: R' = R \ {domain_k} (Python dict deletion, O(1) amortized).

Prior: T3.6 (Finding #429) proved hot-add is structurally free under exclusive routing.  
T3.7 proves the symmetric operation — hot-remove — is also structurally free.

---

## Theorem 1 (Remove Invariance)

**Theorem:** Under exclusive routing, removing adapter k from registry R produces registry  
R' = R \ {k}. For all queries x routed to domain j ≠ k, the output f_j(x) is identical  
before and after removal.

**Proof:**  
For any query x routed to domain j, the forward pass computes:

  f_j(x) = LLM_base(x) + ΔW_j(x)

where ΔW_j is loaded from R[j] = path_j.

After hot-remove of domain k:  
  R'[j] = R[j] = path_j  (unchanged, since j ≠ k)

The forward pass is identical:  
  f_j(x)|_{R'} = LLM_base(x) + ΔW_j(x) = f_j(x)|_{R}

Exclusive routing ensures no other adapter fires for domain j.  
Therefore f_j(x) is INVARIANT to the removal of adapter k. QED.

**Quantitative Prediction (K1070):**  
All N_eval outputs for each remaining domain are bit-exact before and after removal.  
Max token-level differences = 0 for all tested domains.

---

## Theorem 2 (Freed Slot Reusable)

**Theorem:** After removing adapter k (label k freed from R), adding a new adapter m with  
label k (or any new label m) via hot-add produces a functional registry R'' = R' ∪ {m}.  
The new adapter is immediately functional (K1071).

**Proof:**  
Hot-remove clears R[k] → R'. The key k is no longer in the dict.  
Hot-add (T3.6 Theorem 2): R''[m] = path_m is a new O(1) dict insert.  
No coupling exists between removed adapter k and new adapter m at any level:  
  (a) Memory: adapters are loaded from disk on each inference call (not persistent in GPU RAM)
  (b) Registry: only the dict key matters — there is no geometric "slot" allocation  
  (c) Routing: new domain m is routed independently by the router

By T3.6 Theorem 2 (Finding #429), the new adapter is immediately functional.  
The freed label is reusable because Python dict keys have no ghost state.  QED.

**Quantitative Prediction (K1071):**  
After removing adapter k and adding adapter m, adapter m achieves accuracy > base  
on its domain (same criterion as T3.6 K1068 = 4% threshold), and existing adapters  
remain bit-exact.

---

## Theorem 3 (Remove Latency)

**Theorem:** Hot-remove latency is O(1) = Python dict deletion, bounded by  
the same constant as hot-add (T3.6 Theorem 3).

**Proof:**  
Python dict `del d[k]` is O(1) amortized (hash table deletion).  
Wall-clock bound: identical to dict insert (T3.6 Theorem 3 measured 0.004ms).  
Therefore hot-remove latency << 10ms. QED.

**Quantitative Prediction (K1072):**  
Hot-remove latency (p99 over N=100 trials) < 1ms.  
Observed value expected ≈ 0.005ms (same order as T3.6 hot-add 0.004ms).

---

## Kill Criteria Summary

| Kill | Prediction | From |
|------|-----------|------|
| K1070: bit-exact remaining outputs | 0/N differences | Theorem 1 |
| K1071: freed slot reusable, new adapter functional | >4% on new domain | Theorem 2 |
| K1072: remove latency < 10ms | ~0.005ms (p99 < 1ms) | Theorem 3 |

---

## Connection to Prior Work

- Finding #429 (T3.6): Hot-add is structurally free under exclusive routing. T3.7 proves symmetry.
- Finding #425 (T3.1): Simultaneous N-adapter activation catastrophic → routing is load-bearing.
- The Room Model (memory): W_combined = Σ ΔW_i with routing = selective composition.
  Hot-add and hot-remove are the "enter room" / "exit room" operations.

## References

- HRA (2405.17484): Cheapest orthogonal adapter construction (cited as basis for Grassmannian A-matrices)
- T3.6 MATH.md: Theorems 1-3 (hot-add case) which this work symmetrizes
