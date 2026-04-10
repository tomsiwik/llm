# LEARNINGS: T3.7 Plug-and-Play Hot-Remove

## Core Finding

Hot-remove of an adapter from a live registry is structurally free under exclusive routing:
0/40 existing outputs changed (bit-exact), freed slot immediately reusable, p99 latency = 0.001ms.

## Why

Exclusive routing means R[j] is untouched when adapter k ≠ j is removed — algebraically exact,
no approximation. Python dict deletion is O(1) with no ghost state (Theorem 3). The same
structural argument that makes hot-add free (T3.6, Finding #429) applies symmetrically to remove.

## Implications for Next Experiment

T3 is structurally complete. The plug-and-play interface (add/remove < 1ms, zero peer interference,
N=25 Grassmannian max|cos|=2.2e-8) is fully verified. T4 should focus on real routing (PLE-M2P):
activating the right adapter per query is the remaining load-bearing piece (T3.1 showed
simultaneous N=5 activation collapses math/code to 8%). Base MCQ parsing (4% without adapters)
should also be verified once real per-token routing is in place.
