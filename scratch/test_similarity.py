#!/usr/bin/env python3
"""Unit checks for similarity measures and DTW.

    python scratch/test_similarity.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
from scipy import sparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.dtw import dtw_distance
from models.imputation import rank_candidates
from models.representations import CodeVocabulary, build_matrix
from models.similarity import SimilarityIndex


def test_jaccard_values():
    seqs = [["A", "B", "C"], ["A", "B", "D"], ["X", "Y", "Z"]]
    vocab = CodeVocabulary.from_sequences(seqs)
    matrix = build_matrix(seqs, vocab)
    index = SimilarityIndex(matrix, measure="jaccard")
    sims = index.similarity_block(matrix)

    assert abs(sims[0, 0] - 1.0) < 1e-6, "self-similarity must be 1"
    assert abs(sims[0, 1] - 0.5) < 1e-6, f"|AB| / |ABCD| = 0.5, got {sims[0, 1]}"
    assert abs(sims[0, 2]) < 1e-6, "disjoint sets must score 0"
    print("PASS  jaccard values")


def test_binary_metric_equivalence():
    rng = np.random.default_rng(3)
    dense = (rng.random((200, 60)) < 0.15).astype(np.float32)
    matrix = sparse.csr_matrix(dense)

    rankings = {}
    for measure in ("euclidean", "manhattan", "hamming"):
        idx, _ = SimilarityIndex(matrix, measure=measure).topk(matrix, k=8)
        rankings[measure] = idx

    assert np.array_equal(rankings["euclidean"], rankings["manhattan"])
    assert np.array_equal(rankings["euclidean"], rankings["hamming"])
    print("PASS  euclidean == manhattan == hamming on binary vectors")


def test_exclude_self():
    seqs = [["A", "B"], ["A", "B"], ["C", "D"]]
    vocab = CodeVocabulary.from_sequences(seqs)
    matrix = build_matrix(seqs, vocab)
    index = SimilarityIndex(matrix, measure="jaccard")
    idx, _ = index.topk(matrix, k=1, exclude=np.array([0, 1, 2]))
    assert idx[0, 0] == 1, "row 0 should match row 1, not itself"
    print("PASS  self-exclusion")


def test_dtw_properties():
    a = [frozenset({"A", "B"}), frozenset({"C"})]
    b = [frozenset({"A", "B"}), frozenset({"C"})]
    c = [frozenset({"X"}), frozenset({"Y"})]

    assert dtw_distance(a, b) == 0.0, "identical trajectories -> 0"
    assert dtw_distance(a, c) > dtw_distance(a, b), "disjoint should cost more"

    # Warping: a repeated visit should stay cheap against the unrepeated version.
    stretched = [frozenset({"A", "B"}), frozenset({"A", "B"}), frozenset({"C"})]
    assert dtw_distance(a, stretched) < dtw_distance(a, c)
    print("PASS  dtw properties")


def test_imputation_excludes_observed():
    ranked = rank_candidates(
        observed_codes={"A", "B"},
        neighbour_codes=[["A", "B", "C"], ["A", "C", "D"]],
        similarities=np.array([0.9, 0.5]),
    )
    codes = [c for c, _ in ranked]
    assert "A" not in codes and "B" not in codes, "observed codes must be filtered"
    assert codes[0] == "C", f"C appears in both neighbours, expected first, got {codes}"
    print("PASS  imputation excludes observed codes")


def test_negative_similarity_weights():
    # Distance-based measures give negative similarities; weights must stay >= 0.
    ranked = rank_candidates(
        observed_codes=set(),
        neighbour_codes=[["A"], ["B"]],
        similarities=np.array([-1.0, -5.0]),
        scheme="similarity",
    )
    assert all(score >= 0 for _, score in ranked), "vote weights must be non-negative"
    assert ranked[0][0] == "A", "nearer neighbour should win"
    print("PASS  negative similarities handled")


if __name__ == "__main__":
    test_jaccard_values()
    test_binary_metric_equivalence()
    test_exclude_self()
    test_dtw_properties()
    test_imputation_excludes_observed()
    test_negative_similarity_weights()
    print("\nAll similarity tests passed.")
