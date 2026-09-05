#!/usr/bin/env python3
"""End-to-end smoke test on synthetic data - no MIMIC access needed.

Generates patients from a small number of latent "clinical profiles" so that
genuine neighbour structure exists. A working kNN pipeline should clearly beat
the frequency prior here. If it does not, the bug is in the code and not in the
data, which makes this the first thing to run after any change.

    python scratch/test_pipeline_synthetic.py
"""

from __future__ import annotations

import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.baselines import FrequencyPrior
from models.evaluation import evaluate_single_target, format_table
from models.hierarchical import HierarchicalKNN
from models.imputation import predict_batch
from models.representations import CodeVocabulary, build_matrix, frequency_strata
from models.similarity import SimilarityIndex

SEED = 7


def make_synthetic(n_patients=1500, n_profiles=12, vocab_size=200, seed=SEED):
    """Patients drawn from latent profiles, with noise codes mixed in."""
    rng = random.Random(seed)
    codes = [f"D_{i:03d}" for i in range(vocab_size)]

    profiles = []
    for _ in range(n_profiles):
        core = rng.sample(codes, k=rng.randint(6, 12))
        profiles.append(core)

    sequences = []
    for _ in range(n_patients):
        profile = rng.choice(profiles)
        picked = set(rng.sample(profile, k=max(3, len(profile) - rng.randint(0, 2))))
        # background noise: codes unrelated to the profile
        picked.update(rng.sample(codes, k=rng.randint(0, 3)))
        sequences.append(sorted(picked))
    return sequences


def mask_one(sequences, seed=SEED):
    rng = random.Random(seed)
    observed, targets = [], []
    for seq in sequences:
        pos = rng.randrange(len(seq))
        targets.append(seq[pos])
        observed.append(seq[:pos] + seq[pos + 1 :])
    return observed, targets


def main() -> None:
    sequences = make_synthetic()
    split = int(0.8 * len(sequences))
    reference, held_out = sequences[:split], sequences[split:]
    observed, targets = mask_one(held_out)

    vocab = CodeVocabulary.from_sequences(reference)
    ref_matrix = build_matrix(reference, vocab)
    query_matrix = build_matrix(observed, vocab)

    k_values = [1, 5, 10, 20]
    results = {}

    for measure in ("jaccard", "cosine", "dice", "overlap", "hamming", "mahalanobis"):
        index = SimilarityIndex(ref_matrix, measure=measure, mahalanobis_components=32)
        idx, sims = index.topk(query_matrix, k=25)
        preds = predict_batch(
            observed=[set(o) for o in observed],
            neighbour_indices=idx,
            neighbour_similarities=sims,
            reference_sequences=reference,
        )
        results[f"knn_{measure}"] = evaluate_single_target(
            preds, targets, k_values, frequency_strata(vocab)
        )

    prior = FrequencyPrior(vocab)
    results["frequency_prior"] = evaluate_single_target(
        [prior.rank(set(o)) for o in observed], targets, k_values
    )

    print("\n" + format_table(results, k_values) + "\n")

    # -- assertions ----------------------------------------------------------
    baseline_h10 = results["frequency_prior"]["hits@10"]
    knn_h10 = results["knn_jaccard"]["hits@10"]
    assert knn_h10 > baseline_h10, (
        f"kNN ({knn_h10:.3f}) failed to beat frequency prior ({baseline_h10:.3f}) "
        "on data with planted neighbour structure - the retrieval is broken."
    )

    # Euclidean, Manhattan and Hamming must agree exactly on binary vectors.
    e = SimilarityIndex(ref_matrix, measure="euclidean").topk(query_matrix, k=10)[0]
    m = SimilarityIndex(ref_matrix, measure="manhattan").topk(query_matrix, k=10)[0]
    assert np.array_equal(e, m), "euclidean and manhattan should rank identically"

    # Hierarchical update should touch far fewer rows than a full recompute.
    engine = HierarchicalKNN(ref_matrix, measure="jaccard", k=10, recompute_threshold=3)
    for start in range(0, query_matrix.shape[0], 100):
        end = min(start + 100, query_matrix.shape[0])
        engine.ingest(query_matrix[start:end], np.arange(start, end))
    summary = engine.summary()
    print(f"hierarchical: {summary['total_recomputations']} recomputations "
          f"across {summary['n_batches']} batches")

    print(f"\nPASS  knn_jaccard hits@10={knn_h10:.3f} vs prior {baseline_h10:.3f}")


if __name__ == "__main__":
    main()
