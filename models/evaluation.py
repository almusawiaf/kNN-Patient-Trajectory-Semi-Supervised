"""Ranking metrics for held-out diagnosis prediction.

Exact-match accuracy on a 900-way problem with a heavy-tailed label
distribution is close to uninformative: it hides whether the model was nearly
right, and a frequency prior can look respectable on it. These metrics answer
the question the meeting raised, namely accuracy at top-k, and add the
breakdowns that make a near-zero aggregate score interpretable.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Sequence

import numpy as np

log = logging.getLogger(__name__)

Ranking = Sequence[tuple[str, float]]


def hits_at_k(ranking: Ranking, target: str, k: int) -> float:
    """1.0 if ``target`` appears in the top ``k`` candidates."""
    return float(any(code == target for code, _ in ranking[:k]))


def reciprocal_rank(ranking: Ranking, target: str) -> float:
    for position, (code, _) in enumerate(ranking, start=1):
        if code == target:
            return 1.0 / position
    return 0.0


def recall_at_k(ranking: Ranking, targets: set[str], k: int) -> float:
    """Fraction of a target *set* recovered in the top ``k`` (next-visit task)."""
    if not targets:
        return float("nan")
    retrieved = {code for code, _ in ranking[:k]}
    return len(retrieved & targets) / len(targets)


def evaluate_single_target(
    rankings: Sequence[Ranking],
    targets: Sequence[str],
    k_values: Sequence[int] = (1, 5, 10, 20),
    strata: dict[str, str] | None = None,
) -> dict:
    """Aggregate metrics for the within-visit task (one held-out code each)."""
    if len(rankings) != len(targets):
        raise ValueError("rankings and targets must be the same length")

    n = len(targets)
    result: dict = {"n_queries": n}
    if n == 0:
        return result

    for k in k_values:
        result[f"hits@{k}"] = float(
            np.mean([hits_at_k(r, t, k) for r, t in zip(rankings, targets)])
        )

    result["mrr"] = float(
        np.mean([reciprocal_rank(r, t) for r, t in zip(rankings, targets)])
    )
    result["coverage"] = float(
        np.mean([1.0 if len(r) > 0 else 0.0 for r in rankings])
    )
    result["mean_candidates"] = float(np.mean([len(r) for r in rankings]))

    if strata:
        by_stratum: dict[str, list[int]] = defaultdict(list)
        for i, target in enumerate(targets):
            by_stratum[strata.get(target, "unknown")].append(i)

        result["by_frequency_stratum"] = {}
        for label in sorted(by_stratum):
            rows = by_stratum[label]
            entry = {"n": len(rows)}
            for k in k_values:
                entry[f"hits@{k}"] = float(
                    np.mean([hits_at_k(rankings[i], targets[i], k) for i in rows])
                )
            entry["mrr"] = float(
                np.mean([reciprocal_rank(rankings[i], targets[i]) for i in rows])
            )
            result["by_frequency_stratum"][label] = entry

    return result


def evaluate_set_target(
    rankings: Sequence[Ranking],
    targets: Sequence[set[str]],
    k_values: Sequence[int] = (1, 5, 10, 20),
) -> dict:
    """Aggregate metrics for the next-visit task (a held-out code set each)."""
    n = len(targets)
    result: dict = {"n_queries": n}
    if n == 0:
        return result

    for k in k_values:
        result[f"recall@{k}"] = float(
            np.nanmean([recall_at_k(r, t, k) for r, t in zip(rankings, targets)])
        )
        result[f"any_hit@{k}"] = float(
            np.mean(
                [
                    1.0 if {c for c, _ in r[:k]} & t else 0.0
                    for r, t in zip(rankings, targets)
                ]
            )
        )

    result["mean_target_size"] = float(np.mean([len(t) for t in targets]))
    return result


def format_table(results: dict[str, dict], k_values: Sequence[int]) -> str:
    """Render a comparison table across models for the log."""
    metric_names = [f"hits@{k}" for k in k_values if f"hits@{k}" in
                    next(iter(results.values()), {})]
    if not metric_names:
        metric_names = [f"recall@{k}" for k in k_values]
    metric_names = [m for m in metric_names if any(m in r for r in results.values())]
    if "mrr" in next(iter(results.values()), {}):
        metric_names.append("mrr")

    header = f"{'model':<22}" + "".join(f"{m:>12}" for m in metric_names)
    lines = [header, "-" * len(header)]
    for name, metrics in results.items():
        row = f"{name:<22}"
        for metric in metric_names:
            value = metrics.get(metric)
            row += f"{value:>12.4f}" if isinstance(value, float) else f"{'-':>12}"
        lines.append(row)
    return "\n".join(lines)
