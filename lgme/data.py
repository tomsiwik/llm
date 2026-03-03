"""Data utilities — splitting and sampling for continual learning experiments."""

import random


def split_by_initial(docs, boundary='n'):
    """Split docs into two sets by first character.

    Args:
        docs: list of name strings
        boundary: split point (exclusive upper bound for set_a)

    Returns:
        (set_a, set_b) — names starting with a..(boundary-1) vs boundary..z
    """
    set_a = [d for d in docs if d[0].lower() < boundary]
    set_b = [d for d in docs if d[0].lower() >= boundary]
    return set_a, set_b


def split_by_initial_multi(docs, boundaries):
    """Split docs into multiple sets by first character boundaries.

    Args:
        docs: list of name strings
        boundaries: list of boundary chars, e.g. ['f','k','p','u']
                    produces len(boundaries)+1 splits

    Returns:
        list of lists — e.g. 5 splits for 4 boundaries:
        [a-e, f-j, k-o, p-t, u-z]
    """
    boundaries = sorted(boundaries)
    splits = [[] for _ in range(len(boundaries) + 1)]
    for d in docs:
        c = d[0].lower()
        placed = False
        for i, b in enumerate(boundaries):
            if c < b:
                splits[i].append(d)
                placed = True
                break
        if not placed:
            splits[-1].append(d)
    return splits


def reservoir_sample(docs, k, rng=None):
    """Reservoir sampling — uniformly sample k items from docs.

    Args:
        docs: list of items
        k: sample size
        rng: optional random.Random instance

    Returns:
        list of k items (or all items if len(docs) <= k)
    """
    if rng is None:
        rng = random.Random(42)
    if len(docs) <= k:
        return list(docs)
    reservoir = list(docs[:k])
    for i in range(k, len(docs)):
        j = rng.randint(0, i)
        if j < k:
            reservoir[j] = docs[i]
    return reservoir
