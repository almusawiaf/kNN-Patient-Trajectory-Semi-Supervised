"""Dynamic time warping matching the paper's formulation.

Paper: "Dynamic Hierarchical KNN for Patient Trajectory Reconstruction"

The paper defines:
  S_i = [s_1, s_2, ..., s_n]   a sequence of disease tags (ICD-9 codes)
  d(S_new, S_i)                 DTW distance between two such sequences
  w_i = 1 / (d + epsilon)       inverse-distance weight
  s_hat_j = sum(w_i * s_i_j) / sum(w_i)   weighted imputation at position j

Two levels of DTW are now supported:

1. TAG-LEVEL  (within_visit task, paper's primary formulation)
   Sequence unit = one ICD-9 code.
   S_i is the ordered list of codes in ONE admission.
   Local cost between two codes = 0 if identical, 1 otherwise (Hamming on strings).
   This matches the paper exactly: s_j is a disease tag, not a visit.
   Note: code order within a MIMIC admission comes from SEQ_NUM (billing),
   so 'time point' here means billing sequence position, not clock time.
   Still more principled than integer-index arithmetic.

2. VISIT-LEVEL  (next_visit task)
   Sequence unit = one hospital visit (a set of codes).
   S_i is the time-ordered list of admissions for ONE patient.
   Local cost = 1 - Jaccard(visit_a, visit_b).
   This is the trajectory task described in the paper's recommendation section.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

log = logging.getLogger(__name__)

# Type aliases matching the paper's notation
TagSequence   = Sequence[str]        # S_i for within_visit:  ["D_250", "D_401", ...]
VisitSequence = Sequence[frozenset]  # S_i for next_visit:    [{"D_250",...}, {"D_401",...}]


# ============================================================================
# TAG-LEVEL DTW  (paper's primary formulation, within_visit task)
# ============================================================================

def tag_cost(a: str, b: str) -> float:
    """Local cost between two ICD-9 code tags.

    Paper: d(s_j, s_k) between individual disease tags.
    Uses 0/1 Hamming: 0 if same code, 1 if different.
    This avoids the fatal flaw of the original (arithmetic on vocab indices).
    """
    return 0.0 if a == b else 1.0


def tag_cost_matrix(seq_a: TagSequence, seq_b: TagSequence) -> np.ndarray:
    """Pairwise local cost matrix between two tag sequences."""
    n, m = len(seq_a), len(seq_b)
    cost = np.ones((n, m), dtype=np.float64)
    for i, a in enumerate(seq_a):
        for j, b in enumerate(seq_b):
            cost[i, j] = tag_cost(a, b)
    return cost


def dtw_tag_distance(
    seq_a: TagSequence,
    seq_b: TagSequence,
    window: int | None = None,
    normalise: bool = True,
) -> float:
    """DTW distance between two sequences of ICD-9 code tags.

    This is the d(S_new, S_i) in the paper's main formula.

    Parameters
    ----------
    window:
        Sakoe-Chiba band width. Keeps the warping path from stretching
        arbitrarily; recommended value = max(5, abs(len_a - len_b)).
    normalise:
        Divide by warping path length (n + m) so longer sequences don't
        accumulate more cost simply by being longer.
    """
    n, m = len(seq_a), len(seq_b)
    if n == 0 or m == 0:
        return float("inf")

    local = tag_cost_matrix(seq_a, seq_b)
    acc = np.full((n + 1, m + 1), np.inf)
    acc[0, 0] = 0.0

    band = max(window, abs(n - m)) if window is not None else None

    for i in range(1, n + 1):
        j_lo = max(1, i - band) if band is not None else 1
        j_hi = min(m, i + band) if band is not None else m
        for j in range(j_lo, j_hi + 1):
            acc[i, j] = local[i - 1, j - 1] + min(
                acc[i - 1, j],        # insertion (skip a step in seq_a)
                acc[i, j - 1],        # deletion  (skip a step in seq_b)
                acc[i - 1, j - 1],    # match
            )

    distance = acc[n, m]
    if normalise and np.isfinite(distance):
        distance /= float(n + m)
    return float(distance)


def dtw_tag_distance_row(
    query: TagSequence,
    references: Sequence[TagSequence],
    window: int | None = None,
    normalise: bool = True,
) -> np.ndarray:
    """DTW distances from one query tag sequence to all references."""
    return np.array(
        [dtw_tag_distance(query, ref, window=window, normalise=normalise)
         for ref in references],
        dtype=np.float64,
    )


def dtw_tag_topk(
    query: TagSequence,
    references: Sequence[TagSequence],
    k: int,
    window: int | None = None,
    exclude: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Top-k nearest tag sequences by DTW.

    Returns (indices, weights) where weight_i = 1 / (distance_i + epsilon),
    exactly as in the paper's imputation formula.
    """
    eps = 1e-6
    distances = dtw_tag_distance_row(query, references, window=window)
    if exclude is not None and 0 <= exclude < len(distances):
        distances[exclude] = np.inf
    k = min(k, int(np.isfinite(distances).sum()))
    if k == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64)
    part = np.argpartition(distances, kth=k - 1)[:k]
    order = part[np.argsort(distances[part])]
    weights = 1.0 / (distances[order] + eps)   # w_i from the paper
    return order, weights


def impute_position(
    query: TagSequence,
    neighbour_sequences: Sequence[TagSequence],
    weights: np.ndarray,
    mask_position: int,
) -> list[tuple[str, float]]:
    """Weighted vote for the code at mask_position.

    Implements the paper's formula exactly:
        s_hat_j = sum(w_i * s_{i,j}) / sum(w_i)

    Since s_{i,j} is a categorical code (not a number), 'multiplication'
    becomes a weighted vote: each neighbour's code at position j contributes
    weight w_i to that code's score.

    Returns a ranked list of (code, score) pairs, highest score first.
    """
    scores: dict[str, float] = {}
    weight_sum = 0.0
    for seq, w in zip(neighbour_sequences, weights):
        if mask_position < len(seq):
            code = seq[mask_position]
            scores[code] = scores.get(code, 0.0) + float(w)
            weight_sum += float(w)
    if weight_sum > 0:
        scores = {c: s / weight_sum for c, s in scores.items()}
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


# ============================================================================
# VISIT-LEVEL DTW  (next_visit task, trajectory recommendation)
# ============================================================================

def visit_cost_matrix(traj_a: VisitSequence, traj_b: VisitSequence) -> np.ndarray:
    """Pairwise 1 - Jaccard cost between every visit of A and every visit of B."""
    n, m = len(traj_a), len(traj_b)
    cost = np.empty((n, m), dtype=np.float64)
    for i, set_a in enumerate(traj_a):
        for j, set_b in enumerate(traj_b):
            union = len(set_a | set_b)
            cost[i, j] = 1.0 if union == 0 else 1.0 - len(set_a & set_b) / union
    return cost


def dtw_distance(
    traj_a: VisitSequence,
    traj_b: VisitSequence,
    window: int | None = None,
    normalise: bool = True,
) -> float:
    """DTW distance between two patient trajectories (visit-level)."""
    n, m = len(traj_a), len(traj_b)
    if n == 0 or m == 0:
        return float("inf")

    local = visit_cost_matrix(traj_a, traj_b)
    acc = np.full((n + 1, m + 1), np.inf)
    acc[0, 0] = 0.0

    band = max(window, abs(n - m)) if window is not None else None

    for i in range(1, n + 1):
        j_lo = max(1, i - band) if band is not None else 1
        j_hi = min(m, i + band) if band is not None else m
        for j in range(j_lo, j_hi + 1):
            acc[i, j] = local[i - 1, j - 1] + min(
                acc[i - 1, j], acc[i, j - 1], acc[i - 1, j - 1]
            )

    distance = acc[n, m]
    if normalise and np.isfinite(distance):
        distance /= float(n + m)
    return float(distance)


def dtw_distance_row(
    query: VisitSequence,
    references: Sequence[VisitSequence],
    window: int | None = None,
    normalise: bool = True,
) -> np.ndarray:
    """DTW distances from one visit-sequence query to all references."""
    return np.array(
        [dtw_distance(query, ref, window=window, normalise=normalise)
         for ref in references],
        dtype=np.float64,
    )


def dtw_topk(
    query: VisitSequence,
    references: Sequence[VisitSequence],
    k: int,
    window: int | None = None,
    exclude: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Top-k nearest visit-sequences by DTW.

    Returns (indices, similarities) where similarity = -distance,
    matching the SimilarityIndex convention.
    """
    distances = dtw_distance_row(query, references, window=window)
    if exclude is not None and 0 <= exclude < len(distances):
        distances[exclude] = np.inf
    k = min(k, len(distances))
    part = np.argpartition(distances, kth=k - 1)[:k]
    order = part[np.argsort(distances[part])]
    return order, -distances[order]
