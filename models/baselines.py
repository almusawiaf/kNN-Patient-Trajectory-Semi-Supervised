"""Baselines the kNN model has to beat to be worth reporting.

The original run predicted ``D_250`` (diabetes) and ``D_401`` (hypertension)
for most queries. Those are the two most common codes in MIMIC-III, which means
the pipeline had collapsed onto an implicit frequency prior. Making that prior
an explicit baseline turns a failure mode into a measurement: if the kNN model
does not clear :class:`FrequencyPrior`, the neighbour structure is contributing
nothing.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
from scipy import sparse

from .representations import CodeVocabulary

log = logging.getLogger(__name__)


class FrequencyPrior:
    """Rank every code by corpus frequency, ignoring the query entirely."""

    name = "frequency_prior"

    def __init__(self, vocab: CodeVocabulary) -> None:
        self.vocab = vocab
        freqs = vocab.frequency_array()
        total = freqs.sum() or 1.0
        self.scores = freqs / total
        self.order = np.argsort(-self.scores)

    def rank(self, observed_codes: set[str], top_n: int = 50) -> list[tuple[str, float]]:
        out = []
        for col in self.order:
            code = self.vocab.codes[col]
            if code in observed_codes:
                continue
            out.append((code, float(self.scores[col])))
            if len(out) >= top_n:
                break
        return out


class CooccurrenceScorer:
    """Score candidates by co-occurrence with the query's observed codes.

    Uses positive pointwise mutual information, which damps the pull of very
    common codes that co-occur with everything. This is the baseline that tests
    whether patient-level neighbour retrieval adds anything over plain
    code-to-code association.
    """

    name = "cooccurrence"

    def __init__(
        self, matrix: sparse.csr_matrix, vocab: CodeVocabulary, smoothing: float = 1.0
    ) -> None:
        self.vocab = vocab
        n_docs = matrix.shape[0]

        binary = matrix.copy()
        binary.data = np.ones_like(binary.data)
        cooc = (binary.T @ binary).toarray().astype(np.float64)
        marginal = np.diag(cooc).copy()
        np.fill_diagonal(cooc, 0.0)

        p_joint = (cooc + smoothing) / (n_docs + smoothing)
        p_marg = (marginal + smoothing) / (n_docs + smoothing)
        with np.errstate(divide="ignore", invalid="ignore"):
            pmi = np.log(p_joint / np.outer(p_marg, p_marg))
        self.ppmi = np.maximum(np.nan_to_num(pmi), 0.0)
        log.info("Co-occurrence PPMI matrix: %s", self.ppmi.shape)

    def rank(self, observed_codes: set[str], top_n: int = 50) -> list[tuple[str, float]]:
        cols = [self.vocab.code2idx[c] for c in observed_codes if c in self.vocab.code2idx]
        if not cols:
            return []
        scores = self.ppmi[cols].sum(axis=0)
        for col in cols:
            scores[col] = -np.inf
        top = np.argpartition(-scores, kth=min(top_n, len(scores) - 1))[:top_n]
        top = top[np.argsort(-scores[top])]
        return [
            (self.vocab.codes[c], float(scores[c])) for c in top if np.isfinite(scores[c])
        ]


def build_baselines(
    names: Sequence[str], matrix: sparse.csr_matrix, vocab: CodeVocabulary
) -> dict:
    """Instantiate the requested baselines by name."""
    available = {
        FrequencyPrior.name: lambda: FrequencyPrior(vocab),
        CooccurrenceScorer.name: lambda: CooccurrenceScorer(matrix, vocab),
    }
    built = {}
    for name in names:
        if name not in available:
            log.warning("Skipping unknown baseline %r", name)
            continue
        log.info("Building baseline: %s", name)
        built[name] = available[name]()
    return built
