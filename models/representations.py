"""Turning diagnosis code sets into vocabulary indices and sparse matrices.

Design note
-----------
The previous implementation mapped each ICD-9 code to an integer by order of
first appearance, then fed those integers to a numeric distance. That makes
``|idx(D_250) - idx(D_337)|`` a meaningful-looking number when in fact it
encodes nothing but file ordering.

Here integers are used *only* as column positions in a sparse binary matrix.
Distance is computed on set overlap, so the index values themselves never enter
any arithmetic.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Iterable, Sequence

import numpy as np
from scipy import sparse

log = logging.getLogger(__name__)

MASK_TOKEN = "D_X"


class CodeVocabulary:
    """Bidirectional map between diagnosis codes and column indices.

    Codes are ordered by descending corpus frequency, which makes the index
    deterministic across runs and makes frequency-stratified evaluation cheap.
    """

    def __init__(self, counts: Counter) -> None:
        self.counts = counts
        ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        self.codes: list[str] = [code for code, _ in ordered]
        self.code2idx: dict[str, int] = {c: i for i, c in enumerate(self.codes)}

    def __len__(self) -> int:
        return len(self.codes)

    def __contains__(self, code: str) -> bool:
        return code in self.code2idx

    def index(self, code: str) -> int | None:
        return self.code2idx.get(code)

    def frequency(self, code: str) -> int:
        return self.counts.get(code, 0)

    def frequency_array(self) -> np.ndarray:
        """Corpus frequency per column, aligned with matrix columns."""
        return np.array([self.counts[c] for c in self.codes], dtype=np.float64)

    @classmethod
    def from_sequences(
        cls, sequences: Iterable[Sequence[str]], min_frequency: int = 1
    ) -> "CodeVocabulary":
        counts: Counter = Counter()
        for seq in sequences:
            counts.update(set(seq))
        if min_frequency > 1:
            dropped = {c for c, n in counts.items() if n < min_frequency}
            for code in dropped:
                del counts[code]
            log.info(
                "Vocabulary: dropped %d codes below frequency %d",
                len(dropped),
                min_frequency,
            )
        vocab = cls(counts)
        log.info("Vocabulary: %d codes retained", len(vocab))
        return vocab

    def to_dict(self) -> dict:
        return {"codes": self.codes, "counts": dict(self.counts)}

    @classmethod
    def from_dict(cls, payload: dict) -> "CodeVocabulary":
        return cls(Counter(payload["counts"]))


def build_matrix(
    sequences: Sequence[Sequence[str]], vocab: CodeVocabulary
) -> sparse.csr_matrix:
    """Multi-hot sparse matrix, one row per sequence, one column per code.

    Codes outside the vocabulary are silently skipped. Duplicate codes within a
    row collapse to a single 1.
    """
    indptr = [0]
    indices: list[int] = []
    for seq in sequences:
        row = {vocab.code2idx[c] for c in seq if c in vocab.code2idx}
        indices.extend(sorted(row))
        indptr.append(len(indices))

    data = np.ones(len(indices), dtype=np.float32)
    matrix = sparse.csr_matrix(
        (data, np.array(indices, dtype=np.int32), np.array(indptr, dtype=np.int64)),
        shape=(len(sequences), len(vocab)),
    )
    log.info(
        "Built %d x %d matrix, density %.5f",
        matrix.shape[0],
        matrix.shape[1],
        matrix.nnz / max(1, matrix.shape[0] * matrix.shape[1]),
    )
    return matrix


def frequency_strata(
    vocab: CodeVocabulary, n_strata: int = 4
) -> dict[str, str]:
    """Assign every code to a frequency band, for stratified reporting.

    Returns a mapping ``code -> stratum label`` where stratum 1 holds the
    rarest codes. Rare-code performance is where a frequency-prior baseline
    looks good on aggregate accuracy while being clinically useless, so it is
    worth reporting separately.
    """
    freqs = np.array([vocab.counts[c] for c in vocab.codes], dtype=np.float64)
    if len(freqs) == 0:
        return {}
    quantiles = np.quantile(freqs, np.linspace(0, 1, n_strata + 1)[1:-1])
    labels = {}
    for code in vocab.codes:
        band = int(np.searchsorted(quantiles, vocab.counts[code], side="right"))
        labels[code] = f"Q{band + 1}"
    return labels
