#!/usr/bin/env python3
"""Stage 2 - construct the masked prediction task and the train/test split.

Two task formulations are supported.

``within_visit``
    Hold out one diagnosis code from an admission and predict it from the rest.
    This is the original formulation. The change from the previous version is
    that the masked code is *removed* rather than replaced by a ``D_X`` token
    at a fixed position: nothing downstream depends on position any more, and
    leaving a placeholder in the sequence only created a phantom code that
    every similarity computation had to account for.

``next_visit``
    Hold out a patient's final visit and predict its code set from the earlier
    visits. This is the trajectory task the project is actually aiming at.

Only test-set queries are masked. Reference sequences stay intact, so a
neighbour never votes with a hole in it.

Usage
-----
    python build_labels.py --task within_visit
    python build_labels.py --task next_visit
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.config import load_config
from models.io_utils import load_pickle, save_json, save_pickle, setup_logging

log = logging.getLogger("build_labels")


def build_within_visit(cfg) -> dict:
    """Mask one code per held-out admission."""
    paths = cfg.paths
    rng = random.Random(int(cfg.task["seed"]))

    admission_diagnoses = load_pickle(paths.data("admission_diagnoses.pkl"))
    admission_ids = sorted(admission_diagnoses)
    rng.shuffle(admission_ids)

    n_test = int(len(admission_ids) * float(cfg.task["test_fraction"]))
    test_ids = admission_ids[:n_test]
    train_ids = admission_ids[n_test:]

    queries = []
    for adm_id in test_ids:
        codes = list(admission_diagnoses[adm_id])
        if len(codes) < 2:
            continue  # nothing would remain observable after masking
        position = rng.randrange(len(codes))
        target = codes[position]
        observed = codes[:position] + codes[position + 1 :]
        queries.append(
            {"id": adm_id, "observed": observed, "target": target}
        )

    log.info(
        "within_visit: %d reference admissions, %d masked queries",
        len(train_ids),
        len(queries),
    )
    return {
        "task": "within_visit",
        "reference_ids": train_ids,
        "reference_sequences": [admission_diagnoses[i] for i in train_ids],
        "queries": queries,
    }


def build_next_visit(cfg) -> dict:
    """Hold out each test patient's final visit."""
    paths = cfg.paths
    rng = random.Random(int(cfg.task["seed"]))

    trajectories = load_pickle(paths.data("patient_trajectories.pkl"))
    min_visits = int(cfg.data["min_visits_per_patient"])

    eligible = [p for p, v in trajectories.items() if len(v) >= min_visits]
    log.info(
        "Patients with >=%d visits: %d of %d",
        min_visits,
        len(eligible),
        len(trajectories),
    )
    if not eligible:
        raise RuntimeError(
            "No patient has enough visits for the next_visit task. "
            "MIMIC-III is single-admission for most subjects; lower "
            "data.min_visits_per_patient or use the within_visit task."
        )

    rng.shuffle(eligible)
    n_test = max(1, int(len(eligible) * float(cfg.task["test_fraction"])))
    test_patients = eligible[:n_test]
    train_patients = eligible[n_test:]

    # Reference: every visit of every training patient, plus the observed
    # prefix visits of test patients. Held-out final visits never appear.
    reference_ids, reference_sequences, reference_trajectories = [], [], []
    for patient in train_patients:
        history = [frozenset(codes) for _, _, codes in trajectories[patient]]
        reference_trajectories.append(history)
        for hadm_id, _, codes in trajectories[patient]:
            reference_ids.append(hadm_id)
            reference_sequences.append(codes)

    queries = []
    for patient in test_patients:
        visits = trajectories[patient]
        history, future = visits[:-1], visits[-1]
        observed = sorted({c for _, _, codes in history for c in codes})
        queries.append(
            {
                "id": patient,
                "observed": observed,
                "target_set": sorted(set(future[2])),
                "history": [[c for c in codes] for _, _, codes in history],
                "n_history_visits": len(history),
            }
        )

    log.info(
        "next_visit: %d reference visits from %d patients, %d query patients",
        len(reference_sequences),
        len(train_patients),
        len(queries),
    )
    return {
        "task": "next_visit",
        "reference_ids": reference_ids,
        "reference_sequences": reference_sequences,
        "reference_trajectories": reference_trajectories,
        "queries": queries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--project_root", default=None)
    parser.add_argument("--task", choices=["within_visit", "next_visit"], default=None)
    parser.add_argument("--test_fraction", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(
        args.config,
        {
            "paths.project_root": args.project_root,
            "task.name": args.task,
            "task.test_fraction": args.test_fraction,
            "task.seed": args.seed,
        },
    )
    setup_logging(cfg.paths.logs_dir, "build_labels", cfg.runtime["log_level"])
    cfg.paths.ensure()

    task = cfg.task["name"]
    builder = {"within_visit": build_within_visit, "next_visit": build_next_visit}[task]
    payload = builder(cfg)

    out = cfg.paths.data(f"task_{task}.pkl")
    save_pickle(payload, out)

    sizes = [len(q["observed"]) for q in payload["queries"]]
    save_json(
        {
            "task": task,
            "n_reference": len(payload["reference_sequences"]),
            "n_queries": len(payload["queries"]),
            "mean_observed_codes": float(np.mean(sizes)) if sizes else 0.0,
            "median_observed_codes": float(np.median(sizes)) if sizes else 0.0,
        },
        cfg.paths.results(f"task_summary_{task}.json"),
    )


if __name__ == "__main__":
    main()
