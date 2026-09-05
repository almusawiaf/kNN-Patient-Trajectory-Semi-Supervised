"""Predicting held-out diagnosis codes from neighbour votes.

What changed from the original implementation
---------------------------------------------
The earlier version read the neighbour's code *at the masked position*: if the
mask sat at index 7, it looked up ``neighbour_sequence[7]``. Positions inside a
MIMIC admission come from the billing sequence number, not from clinical
ordering, so position 7 in one admission has no correspondence with position 7
in another. The lookup also silently skipped any neighbour shorter than the
mask position, quietly discarding votes.

Here every code a neighbour carries is a candidate, weighted by that
neighbour's similarity to the query. Codes the query already shows are removed
from the ranking, since they cannot be the held-out answer. The output is a
ranked list rather than a single argmax, which is what makes hits@k reportable.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Sequence

import numpy as np

log = logging.getLogger(__name__)


def vote_weights(similarities: np.ndarray, scheme: str, epsilon: float) -> np.ndarray:
    """Convert neighbour similarities into non-negative vote weights."""
    if scheme == "uniform":
        return np.ones_like(similarities, dtype=np.float64)

    if scheme == "rank":
        return 1.0 / (np.arange(len(similarities), dtype=np.float64) + 1.0)

    if scheme == "similarity":
        sims = np.asarray(similarities, dtype=np.float64)
        # Distance-derived measures return negative similarity; shift so the
        # weakest neighbour in the set gets weight ~0 rather than a negative
        # weight that would subtract votes.
        if sims.min() < 0:
            sims = sims - sims.min()
        return sims + epsilon

    raise ValueError(f"Unknown vote weighting {scheme!r}")


def rank_candidates(
    observed_codes: set[str],
    neighbour_codes: Sequence[Sequence[str]],
    similarities: np.ndarray,
    scheme: str = "similarity",
    epsilon: float = 1e-6,
    exclude_observed: bool = True,
) -> list[tuple[str, float]]:
    """Rank candidate codes for one query.

    Parameters
    ----------
    observed_codes:
        Codes visible in the query (everything except the held-out target).
    neighbour_codes:
        Code lists of the retrieved neighbours, nearest first.
    similarities:
        Similarity of each neighbour to the query, same order.

    Returns
    -------
    Ranked ``(code, score)`` pairs, highest score first.
    """
    weights = vote_weights(np.asarray(similarities), scheme, epsilon)
    scores: dict[str, float] = defaultdict(float)

    for codes, weight in zip(neighbour_codes, weights):
        for code in set(codes):
            if exclude_observed and code in observed_codes:
                continue
            scores[code] += float(weight)

    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


def predict_batch(
    observed: Sequence[set[str]],
    neighbour_indices: np.ndarray,
    neighbour_similarities: np.ndarray,
    reference_sequences: Sequence[Sequence[str]],
    scheme: str = "similarity",
    epsilon: float = 1e-6,
    top_n: int = 50,
) -> list[list[tuple[str, float]]]:
    """Ranked candidates for a batch of queries.

    ``top_n`` truncates each ranking to keep memory bounded; it must be at
    least as large as the biggest k used at evaluation time.
    """
    predictions = []
    for row, codes in enumerate(observed):
        neighbours = [reference_sequences[i] for i in neighbour_indices[row]]
        ranked = rank_candidates(
            observed_codes=codes,
            neighbour_codes=neighbours,
            similarities=neighbour_similarities[row],
            scheme=scheme,
            epsilon=epsilon,
        )
        predictions.append(ranked[:top_n])
    return predictions
