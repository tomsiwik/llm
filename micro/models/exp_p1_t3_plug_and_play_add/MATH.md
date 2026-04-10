# MATH.md — T3.6: Hot-Add Adapter Without Retraining

## Setting

Let the system at time t have N domain adapters registered:
```
Registry_t = {(A_i, B_i) : i = 1..N}
```
Each adapter contributes Δ_i = A_i B_i to the weight update for domain i.

With exclusive routing, the effective weight for a query from domain i is:
```
W_eff(q, domain=i) = W_base + A_i B_i
```
No other adapter enters the computation — only the single matched adapter is applied.

---

## Theorem 1: Exclusive Routing Invariance

**Claim:** Adding adapter (A_{N+1}, B_{N+1}) to Registry_t to form Registry_{t+1} does
NOT change W_eff(q, domain=i) for any i ≤ N and any query q.

**Proof:**
```
W_eff(q, domain=i) = W_base + A_i B_i    [exclusive routing: only adapter i]
```
The quantity A_i B_i depends solely on {A_i, B_i}, which are not modified by
the registry update. Registry_{t+1} = Registry_t ∪ {(A_{N+1}, B_{N+1})}.

For i ≤ N:
```
W_eff_{t+1}(q, domain=i) = W_base + A_i B_i = W_eff_t(q, domain=i)
```

The key property is that we never form W_merged = W_base + Σ A_i B_i.
Instead, routing selects exactly one adapter: the sum has exactly one term.
Adding a new term to the registry cannot change which single term is selected
for existing domains (routing is many-to-one by domain label, not by index).  **QED**

**Quantitative prediction:** Token-level outputs (logit vectors) are bit-identical
before and after hot-add for any query routed to an existing domain.

**Prediction (K1067):** max |output_before - output_after| = 0.0 (float32 exact equality)

---

## Theorem 2: Immediate Functionality of New Adapter

**Claim:** Adapter (A_{N+1}, B_{N+1}) is immediately functional after registration —
no retraining of existing adapters or the routing table is required.

**Proof:**
The adapter was trained on domain N+1's data. Its output Δ_{N+1} = A_{N+1} B_{N+1}
encodes the domain-specific weight update. The exclusive routing rule is:
```
route(q) = argmax_i Sim(q, c_i)    where c_i is domain i's centroid/label
```
After adding centroid c_{N+1}, queries from domain N+1 route to adapter N+1.
Existing domain centroids c_1..c_N are unchanged.

For domain N+1 queries, the effective weight is:
```
W_eff(q, domain=N+1) = W_base + A_{N+1} B_{N+1}
```
This is identical to running adapter N+1 in isolation — the same computation as
any other domain's adapter, which T2.1 and T2.6 verified achieves +22pp to +82pp.

Additionally, T3.4 (Finding #428) showed that any adapter trained on a domain
enables MCQ format compliance universally (56-88% on neutral MMLU subjects).
Therefore, domain N+1 is immediately functional.   **QED**

**Prediction (K1068):** New adapter (domain 6, synthetic with B_train from training
or MCQ-format transfer from base) achieves > base accuracy on its domain immediately.

---

## Theorem 3: Hot-Add Latency Bound

**Claim:** Hot-add latency is O(|adapter_file|) — limited by I/O, not by computation.

**Proof:**
The registry is a Python dict: domain_label → adapter_path (or loaded weights).
The hot-add operation consists of:
1. Write adapter weights to disk: O(M) where M = file size (~4.77 MB per adapter)
2. Update dict: O(1) amortized
3. No model forward passes, no gradient computation, no training

I/O bound: NVMe sequential write speed on M5 Pro ≈ 6 GB/s → 4.77 MB / 6 GB/s ≈ 0.8ms
Memory mapping (mlx_lm): adapter load via safetensors mmap = O(1) first access.

Total hot-add latency ≪ 100ms.   **QED**

**Prediction (K1069):** Hot-add latency (registry update + file write) < 100ms.

---

## Kill Criteria Summary

| Criterion | Prediction | Source |
|-----------|------------|--------|
| K1067: existing outputs unchanged after hot-add | max\|diff\| = 0.0 (bit-exact) | Theorem 1 |
| K1068: new adapter functional immediately | acc > base on domain 6 | Theorem 2 |
| K1069: hot-add latency < 100ms | ~1ms (I/O bound) | Theorem 3 |

---

## References

- Finding #428: T3.4 N=25 Grassmannian composition on Gemma 4 (K1059-K1062 PASS)
- Finding #421: T2.1 LoRA r=6 achieves 22-82pp domain improvement on Gemma 4 E4B
- HRA (arxiv 2405.17484): Orthogonal adapter construction
- T3.1 (Finding #425, KILLED): Simultaneous activation destroys math/code —
  exclusive routing is the structural fix that makes this experiment possible
