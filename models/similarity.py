"""Similarity measures over binary diagnosis-code vectors.

All measures are computed from the intersection count between two code sets,
which for binary sparse matrices is a single sparse matrix product. Everything
here returns a *similarity* (higher is more similar) so downstream code has one
convention to reason about.

A note that matters for the metric sweep
----------------------------------------
On binary vectors, Manhattan distance, squared Euclidean distance and Hamming
distance are all equal to the size of the symmetric difference:

    L1(a, b) = L2(a, b)^2 = |a| + |b| - 2|a n b|

They therefore induce *identical* neighbour rankings. Running all three as if
they were separate conditions produces three copies of one result. The measures
that genuinely differ are the ones that normalise the intersection differently:
Jaccard, Dice, cosine and overlap.
"""

from __future__ import annotations

import logging
from typing import Iterator

import numpy as np
from scipy import sparse

log = logging.getLogger(__name__)

SET_MEASURES = ("jaccard", "dice", "cosine", "overlap")
VECTOR_MEASURES = ("euclidean", "manhattan", "hamming", "mahalanobis")
ALL_MEASURES = SET_MEASURES + VECTOR_MEASURES

#: Measures that produce the same ranking, kept for documentation and so the
#: sweep script can warn instead of silently duplicating work.
EQUIVALENT_ON_BINARY = {
    "euclidean": "hamming",
    "manhattan": "hamming",
}


class SimilarityIndex:
    """Blocked similarity search against a fixed reference matrix.

    Parameters
    ----------
    reference:
        ``(n_reference, n_codes)`` binary CSR matrix.
    measure:
        One of :data:`ALL_MEASURES`.
    block_size:
        Number of query rows scored at a time.
    mahalanobis_components:
        Rank of the whitened projection used when ``measure='mahalanobis'``.
    """

    def __init__(
        self,
        reference: sparse.csr_matrix,
        measure: str = "jaccard",
        block_size: int = 512,
        mahalanobis_components: int = 128,
        random_state: int = 0,
    ) -> None:
        if measure not in ALL_MEASURES:
            raise ValueError(
                f"Unknown measure {measure!r}; expected one of {ALL_MEASURES}"
            )
        self.measure = measure
        self.block_size = block_size
        self.reference = reference.astype(np.float32).tocsr()
        self.reference_sizes = np.asarray(self.reference.sum(axis=1)).ravel()
        self._projection = None

        if measure == "mahalanobis":
            self._projection = _fit_whitening(
                self.reference, mahalanobis_components, random_state
            )
            self._reference_proj = self._projection.transform(self.reference)
            self._reference_sqnorm = (self._reference_proj ** 2).sum(axis=1)

    # -- public API ----------------------------------------------------------
    def similarity_block(self, queries: sparse.csr_matrix) -> np.ndarray:
        """Dense ``(n_queries, n_reference)`` similarity matrix for one block."""
        queries = queries.astype(np.float32).tocsr()

        if self.measure == "mahalanobis":
            proj = self._projection.transform(queries)
            sq = (proj ** 2).sum(axis=1)[:, None]
            d2 = sq + self._reference_sqnorm[None, :] - 2.0 * (proj @ self._reference_proj.T)
            np.maximum(d2, 0.0, out=d2)
            return -np.sqrt(d2)

        inter = np.asarray((queries @ self.reference.T).todense(), dtype=np.float32)
        q_sizes = np.asarray(queries.sum(axis=1)).ravel()[:, None]
        r_sizes = self.reference_sizes[None, :]

        if self.measure == "jaccard":
            union = q_sizes + r_sizes - inter
            return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)

        if self.measure == "dice":
            total = q_sizes + r_sizes
            return np.divide(
                2.0 * inter, total, out=np.zeros_like(inter), where=total > 0
            )

        if self.measure == "cosine":
            denom = np.sqrt(q_sizes * r_sizes)
            return np.divide(inter, denom, out=np.zeros_like(inter), where=denom > 0)

        if self.measure == "overlap":
            denom = np.minimum(q_sizes, r_sizes)
            return np.divide(inter, denom, out=np.zeros_like(inter), where=denom > 0)

        # hamming / manhattan / euclidean all reduce to symmetric difference
        symdiff = q_sizes + r_sizes - 2.0 * inter
        if self.measure == "euclidean":
            return -np.sqrt(np.maximum(symdiff, 0.0))
        return -symdiff

    def blocks(self, queries: sparse.csr_matrix) -> Iterator[tuple[int, np.ndarray]]:
        """Yield ``(start_row, similarity_block)`` over the query matrix."""
        n = queries.shape[0]
        for start in range(0, n, self.block_size):
            end = min(start + self.block_size, n)
            yield start, self.similarity_block(queries[start:end])

    def topk(
        self,
        queries: sparse.csr_matrix,
        k: int,
        exclude: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Top-``k`` reference rows per query.

        Parameters
        ----------
        exclude:
            Optional array of reference indices, one per query row, to mask out
            (used to drop self-matches when query and reference coincide).

        Returns
        -------
        indices, similarities:
            Both ``(n_queries, k)``, sorted by descending similarity.
        """
        n_ref = self.reference.shape[0]
        k = min(k, n_ref - (1 if exclude is not None else 0))
        idx_out = np.zeros((queries.shape[0], k), dtype=np.int64)
        sim_out = np.zeros((queries.shape[0], k), dtype=np.float32)

        for start, sims in self.blocks(queries):
            end = start + sims.shape[0]
            if exclude is not None:
                rows = np.arange(sims.shape[0])
                valid = exclude[start:end] >= 0
                sims[rows[valid], exclude[start:end][valid]] = -np.inf

            part = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
            part_sims = np.take_along_axis(sims, part, axis=1)
            order = np.argsort(-part_sims, axis=1)
            idx_out[start:end] = np.take_along_axis(part, order, axis=1)
            sim_out[start:end] = np.take_along_axis(part_sims, order, axis=1)

        return idx_out, sim_out


class _Whitening:
    """Truncated SVD followed by variance normalisation."""

    def __init__(self, components: np.ndarray, scale: np.ndarray) -> None:
        self.components = components
        self.scale = scale

    def transform(self, matrix: sparse.csr_matrix) -> np.ndarray:
        return (matrix @ self.components.T) / self.scale


def _fit_whitening(
    matrix: sparse.csr_matrix, n_components: int, random_state: int
) -> _Whitening:
    """Whitened low-rank projection, a tractable stand-in for Mahalanobis.

    A full Mahalanobis distance needs the inverse covariance of the code space.
    With hundreds to thousands of sparse binary columns that matrix is
    rank-deficient and inverting it is both expensive and unstable. Projecting
    onto the leading singular directions and dividing by their singular values
    gives the same decorrelating effect at a rank you control.
    """
    from sklearn.decomposition import TruncatedSVD

    n_components = int(min(n_components, min(matrix.shape) - 1))
    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    svd.fit(matrix)
    scale = np.sqrt(np.maximum(svd.explained_variance_, 1e-12))
    log.info(
        "Mahalanobis projection: rank %d, %.1f%% variance retained",
        n_components,
        100.0 * svd.explained_variance_ratio_.sum(),
    )
    return _Whitening(svd.components_.astype(np.float32), scale.astype(np.float32))
