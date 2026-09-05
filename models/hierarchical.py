"""Tiered KNN update as new sequences arrive.

Works with both:
- Sparse matrix similarity (Jaccard, cosine, etc.)  -- within_visit default
- DTW distance matrix                               -- within_visit with dtw_tag

The paper defines:
  F(S_i) = frequency of S_i appearing in KNN lists of new sequences
  Sequences with F(S_i) >= threshold get their own neighbours recomputed.
  Each sequence is refreshed at most once (critical for bounding total work).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Callable, Sequence

import numpy as np
from scipy import sparse

log = logging.getLogger(__name__)


class HierarchicalKNN:
    """Maintain neighbour lists across successive arrival batches.

    Supports two retrieval backends:
      - sparse_fn: a callable(query_matrix) -> (indices, weights) for set measures
      - dtw_fn:    a callable(query_sequences) -> (indices, weights) for DTW

    Exactly one must be provided.
    """

    def __init__(
        self,
        reference: sparse.csr_matrix | None,
        measure: str = "jaccard",
        k: int = 25,
        recompute_threshold: int = 2,
        block_size: int = 512,
        mahalanobis_components: int = 128,
        # DTW backend (optional)
        reference_sequences: list | None = None,
        dtw_fn: Callable | None = None,
    ) -> None:
        self.k = k
        self.recompute_threshold = recompute_threshold
        self.knn_map: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self.frequency: dict[int, int] = defaultdict(int)
        self.refreshed: set[int] = set()
        self.total_recomputations = 0
        self.stats: list[dict] = []

        # -- backend setup ---------------------------------------------------
        self._use_dtw = dtw_fn is not None
        self._dtw_fn = dtw_fn
        self._ref_seqs = reference_sequences or []

        if not self._use_dtw:
            from .similarity import SimilarityIndex
            self.index = SimilarityIndex(
                reference,
                measure=measure,
                block_size=block_size,
                mahalanobis_components=mahalanobis_components,
            )
            self._n_ref = reference.shape[0]
        else:
            self._n_ref = len(self._ref_seqs)

    def _retrieve(
        self,
        batch_matrix: sparse.csr_matrix | None,
        batch_seqs: list | None,
        exclude: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Retrieve top-k for a batch using the configured backend."""
        if self._use_dtw:
            # DTW: one query at a time
            all_idx, all_sim = [], []
            for i, seq in enumerate(batch_seqs):
                ex = int(exclude[i]) if exclude is not None else None
                idx, sim = self._dtw_fn(seq, self._ref_seqs, self.k, exclude=ex)
                # Pad to k if fewer results
                pad = self.k - len(idx)
                if pad > 0:
                    idx = np.concatenate([idx, np.full(pad, -1, dtype=np.int64)])
                    sim = np.concatenate([sim, np.zeros(pad)])
                all_idx.append(idx)
                all_sim.append(sim)
            return np.stack(all_idx), np.stack(all_sim)
        else:
            return self.index.topk(batch_matrix, k=self.k, exclude=exclude)

    def ingest(
        self,
        batch_matrix: sparse.csr_matrix | None,
        batch_ids: np.ndarray,
        batch_seqs: list | None = None,
    ) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        """Process one arrival batch and return their neighbour lists.

        Parameters
        ----------
        batch_matrix : sparse matrix of new queries (set-similarity path)
        batch_ids    : integer IDs for the new queries
        batch_seqs   : list of tag sequences (DTW path only)
        """
        indices, sims = self._retrieve(batch_matrix, batch_seqs)

        # Record neighbour lists and update frequency counter
        for row, qid in enumerate(batch_ids):
            self.knn_map[int(qid)] = (indices[row], sims[row])
            for nb in indices[row]:
                if nb >= 0:
                    self.frequency[int(nb)] += 1

        # Refresh hub neighbourhoods (paper: F(S_i) >= threshold, first time only)
        flagged = [
            ref_id
            for ref_id, count in self.frequency.items()
            if count >= self.recompute_threshold and ref_id not in self.refreshed
        ]
        self.refreshed.update(flagged)
        self.total_recomputations += len(flagged)

        if flagged:
            if self._use_dtw:
                hub_seqs = [self._ref_seqs[i] for i in flagged]
                hub_idx, hub_sim = self._retrieve(
                    None, hub_seqs,
                    exclude=np.array(flagged, dtype=np.int64),
                )
            else:
                hub_matrix = self.index.reference[np.array(flagged)]
                hub_idx, hub_sim = self.index.topk(
                    hub_matrix, k=self.k,
                    exclude=np.array(flagged, dtype=np.int64),
                )
            for pos, ref_id in enumerate(flagged):
                self.knn_map[ref_id] = (hub_idx[pos], hub_sim[pos])

        self.stats.append({
            "batch_size": int(len(batch_ids)),
            "recomputed": len(flagged),
            "cumulative_recomputations": self.total_recomputations,
            "full_recompute_would_be": self._n_ref,
            "cumulative_vs_full_rebuild": (
                self.total_recomputations / self._n_ref if self._n_ref else 0.0
            ),
        })
        log.info(
            "Batch of %d: recomputed %d hub neighbourhoods (%.2f%% of reference)",
            len(batch_ids), len(flagged),
            100.0 * len(flagged) / max(1, self._n_ref),
        )
        return {int(q): self.knn_map[int(q)] for q in batch_ids}

    def summary(self) -> dict:
        ratio = self.total_recomputations / self._n_ref if self._n_ref else 0.0
        return {
            "backend": "dtw" if self._use_dtw else "sparse",
            "n_batches": len(self.stats),
            "n_ingested": sum(s["batch_size"] for s in self.stats),
            "total_recomputations": self.total_recomputations,
            "unique_rows_refreshed": len(self.refreshed),
            "reference_size": self._n_ref,
            "cumulative_vs_full_rebuild": ratio,
            "beats_full_rebuild": bool(ratio < 1.0),
            "mean_recomputations_per_batch": (
                self.total_recomputations / len(self.stats) if self.stats else 0.0
            ),
            "per_batch": self.stats,
        }
