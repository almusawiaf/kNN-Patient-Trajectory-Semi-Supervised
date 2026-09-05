#!/usr/bin/env python3
"""Stage 3 - run neighbour retrieval, impute held-out codes, score everything.

Similarity options
------------------
Set measures (fast, vectorised):
  jaccard | dice | cosine | overlap | hamming | mahalanobis

DTW measures:
  dtw_tag       -- tag-level DTW on within_visit sequences (paper's formulation)
  dtw_jaccard   -- visit-level DTW on next_visit trajectories

Usage
-----
    python run_experiment.py --task within_visit --similarity jaccard
    python run_experiment.py --task within_visit --similarity dtw_tag
    python run_experiment.py --task next_visit   --similarity dtw_jaccard
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

import numpy as np
from scipy import sparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.baselines import build_baselines
from models.config import load_config
from models.dtw import dtw_tag_topk, dtw_topk, impute_position
from models.evaluation import (
    evaluate_set_target,
    evaluate_single_target,
    format_table,
)
from models.hierarchical import HierarchicalKNN
from models.imputation import predict_batch, rank_candidates
from models.io_utils import load_pickle, save_json, setup_logging
from models.representations import (
    CodeVocabulary,
    build_matrix,
    frequency_strata,
)

log = logging.getLogger("run_experiment")


# ============================================================================
# Retrieval backends
# ============================================================================

def run_knn_sparse(cfg, payload, vocab):
    """Set/vector similarity on the sparse binary matrix (fast, vectorised)."""
    from models.similarity import SimilarityIndex

    ref_seqs = payload["reference_sequences"]
    queries  = payload["queries"]

    ref_matrix   = build_matrix(ref_seqs, vocab)
    query_matrix = build_matrix([q["observed"] for q in queries], vocab)

    index = SimilarityIndex(
        ref_matrix,
        measure=cfg.model["similarity"],
        block_size=int(cfg.runtime["block_size"]),
        mahalanobis_components=int(cfg.model["mahalanobis_components"]),
    )

    t0 = time.time()
    indices, sims = index.topk(query_matrix, k=int(cfg.model["k"]))
    log.info("Retrieved %d neighbours for %d queries in %.1fs",
             cfg.model["k"], query_matrix.shape[0], time.time() - t0)

    return predict_batch(
        observed=[set(q["observed"]) for q in queries],
        neighbour_indices=indices,
        neighbour_similarities=sims,
        reference_sequences=ref_seqs,
        scheme=cfg.model["vote_weighting"],
        epsilon=float(cfg.model["epsilon"]),
    ), ref_matrix, query_matrix


def run_knn_dtw_tag(cfg, payload):
    """Tag-level DTW on within_visit sequences — paper's primary formulation.

    Each patient's observed code list is treated as an ordered sequence S_i.
    DTW with 0/1 local cost (same code = 0, different code = 1) finds the
    k nearest reference sequences. Imputation uses the paper's weighted formula:

        s_hat_j = sum(w_i * s_{i,j}) / sum(w_i),  w_i = 1 / (d_i + eps)

    where j is the masked position.
    """
    queries   = payload["queries"]
    ref_seqs  = payload["reference_sequences"]

    max_ref = cfg.runtime.get("max_reference")
    if max_ref and len(ref_seqs) > int(max_ref):
        log.warning("Capping DTW reference from %d to %d", len(ref_seqs), int(max_ref))
        ref_seqs = ref_seqs[:int(max_ref)]

    k = int(cfg.model["k"])
    window = cfg.model.get("dtw_window")   # optional Sakoe-Chiba band

    predictions = []
    t0 = time.time()
    for n, q in enumerate(queries):
        observed = q["observed"]  # list of codes with one removed
        target   = q["target"]

        # Find k nearest neighbours by tag-level DTW
        idx, weights = dtw_tag_topk(observed, ref_seqs, k=k, window=window)

        if len(idx) == 0:
            predictions.append([])
            continue

        neighbour_seqs = [ref_seqs[i] for i in idx]

        # Paper's imputation: weighted vote at the masked position.
        # We don't know the exact position the mask was at in the original
        # sequence, so we vote over the entire neighbour code pool, excluding
        # codes already observed. This preserves the weighted formula while
        # handling MIMIC's unordered billing sequences.
        ranked = rank_candidates(
            observed_codes=set(observed),
            neighbour_codes=neighbour_seqs,
            similarities=weights,   # already 1/(d+eps) from dtw_tag_topk
            scheme="similarity",
            exclude_observed=True,
        )
        predictions.append(ranked[:50])

        if n and n % 500 == 0:
            log.info("DTW tag: %d/%d queries (%.1fs)", n, len(queries), time.time() - t0)

    log.info("DTW tag retrieval done: %d queries in %.1fs", len(queries), time.time() - t0)
    return predictions


def run_knn_dtw_visit(cfg, payload):
    """Visit-level DTW on next_visit trajectories."""
    if "reference_trajectories" not in payload:
        raise ValueError("dtw_jaccard requires the next_visit task.")

    ref_trajs = payload["reference_trajectories"]
    flat_ref  = [sorted({c for v in t for c in v}) for t in ref_trajs]

    max_ref = cfg.runtime.get("max_reference")
    if max_ref and len(ref_trajs) > int(max_ref):
        log.warning("Capping DTW reference from %d to %d", len(ref_trajs), int(max_ref))
        ref_trajs = ref_trajs[:int(max_ref)]
        flat_ref  = flat_ref[:int(max_ref)]

    k = int(cfg.model["k"])
    predictions = []
    t0 = time.time()
    for n, q in enumerate(payload["queries"]):
        history = [frozenset(v) for v in q["history"]]
        idx, sims = dtw_topk(history, ref_trajs, k=k)
        ranked = rank_candidates(
            observed_codes=set(q["observed"]),
            neighbour_codes=[flat_ref[i] for i in idx],
            similarities=sims,
            scheme=cfg.model["vote_weighting"],
            epsilon=float(cfg.model["epsilon"]),
        )
        predictions.append(ranked[:50])
        if n and n % 100 == 0:
            log.info("DTW visit: %d/%d (%.1fs)", n, len(payload["queries"]), time.time() - t0)
    return predictions


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--project_root", default=None)
    parser.add_argument("--task",
                        choices=["within_visit", "next_visit"], default=None)
    parser.add_argument("--similarity",
                        default=None,
                        help="jaccard | cosine | dtw_tag | dtw_jaccard | ...")
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--vote_weighting", default=None)
    parser.add_argument("--icd_digits", type=int, default=None)
    parser.add_argument("--no_hierarchical", action="store_true")
    parser.add_argument("--dtw_window", type=int, default=None,
                        help="Sakoe-Chiba band width for DTW (default: no constraint)")
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()

    cfg = load_config(
        args.config,
        {
            "paths.project_root": args.project_root,
            "task.name": args.task,
            "model.similarity": args.similarity,
            "model.k": args.k,
            "model.vote_weighting": args.vote_weighting,
            "data.icd_digits": args.icd_digits,
            "hierarchical.enabled": False if args.no_hierarchical else None,
            "model.dtw_window": args.dtw_window,
        },
    )
    setup_logging(cfg.paths.logs_dir, "run_experiment", cfg.runtime["log_level"])
    cfg.paths.ensure()

    task    = cfg.task["name"]
    sim     = cfg.model["similarity"]
    payload = load_pickle(cfg.paths.data(f"task_{task}.pkl"))
    vocab   = CodeVocabulary.from_dict(load_pickle(cfg.paths.data("vocabulary.pkl")))
    log.info("Task=%s | similarity=%s | queries=%d | vocab=%d",
             task, sim, len(payload["queries"]), len(vocab))

    # -- retrieval -----------------------------------------------------------
    ref_matrix   = None
    query_matrix = None

    if sim == "dtw_tag":
        if task != "within_visit":
            raise ValueError("dtw_tag is for the within_visit task.")
        predictions = run_knn_dtw_tag(cfg, payload)

    elif sim == "dtw_jaccard":
        predictions = run_knn_dtw_visit(cfg, payload)

    else:
        predictions, ref_matrix, query_matrix = run_knn_sparse(cfg, payload, vocab)

    all_rankings = {f"knn_{sim}": predictions}
    all_rankings.update(run_baselines(cfg, payload, vocab))

    # -- scoring -------------------------------------------------------------
    k_values = cfg.evaluation["k_values"]
    strata   = frequency_strata(vocab) if cfg.evaluation["stratify_by_frequency"] else None
    results  = {}

    for name, rankings in all_rankings.items():
        if task == "within_visit":
            targets = [q["target"] for q in payload["queries"]]
            results[name] = evaluate_single_target(rankings, targets, k_values, strata)
        else:
            targets = [set(q["target_set"]) for q in payload["queries"]]
            results[name] = evaluate_set_target(rankings, targets, k_values)

    log.info("\n%s", format_table(results, k_values))

    # -- hierarchical update -------------------------------------------------
    hierarchical_summary = None
    if cfg.hierarchical["enabled"]:
        if sim == "dtw_tag":
            # DTW-based hierarchical update (paper's formulation end-to-end)
            def dtw_fn(seq, refs, k, exclude=None):
                from models.dtw import dtw_tag_topk
                return dtw_tag_topk(seq, refs, k=k,
                                    window=cfg.model.get("dtw_window"),
                                    exclude=exclude)

            engine = HierarchicalKNN(
                reference=None,
                k=int(cfg.model["k"]),
                recompute_threshold=int(cfg.hierarchical["recompute_threshold"]),
                reference_sequences=payload["reference_sequences"],
                dtw_fn=dtw_fn,
            )
            queries_seqs = [q["observed"] for q in payload["queries"]]
            batch_size   = int(cfg.hierarchical["batch_size"])
            for start in range(0, len(queries_seqs), batch_size):
                end = min(start + batch_size, len(queries_seqs))
                engine.ingest(
                    batch_matrix=None,
                    batch_ids=np.arange(start, end),
                    batch_seqs=queries_seqs[start:end],
                )

        elif sim not in ("dtw_jaccard",) and ref_matrix is not None:
            # Sparse-matrix hierarchical update
            engine = HierarchicalKNN(
                reference=ref_matrix,
                measure=sim,
                k=int(cfg.model["k"]),
                recompute_threshold=int(cfg.hierarchical["recompute_threshold"]),
                block_size=int(cfg.runtime["block_size"]),
            )
            batch_size = int(cfg.hierarchical["batch_size"])
            for start in range(0, query_matrix.shape[0], batch_size):
                end = min(start + batch_size, query_matrix.shape[0])
                engine.ingest(
                    batch_matrix=query_matrix[start:end],
                    batch_ids=np.arange(start, end),
                )
        else:
            engine = None

        if engine:
            hierarchical_summary = engine.summary()

    # -- persist -------------------------------------------------------------
    tag = args.tag or cfg.tag()
    save_json(
        {"config": cfg.to_dict(), "results": results, "hierarchical": hierarchical_summary},
        cfg.paths.results(f"{tag}.json"),
    )


def run_baselines(cfg, payload, vocab):
    ref_matrix = build_matrix(payload["reference_sequences"], vocab)
    models     = build_baselines(cfg.evaluation["baselines"], ref_matrix, vocab)
    return {
        name: [model.rank(set(q["observed"]), top_n=50) for q in payload["queries"]]
        for name, model in models.items()
    }


if __name__ == "__main__":
    main()
    