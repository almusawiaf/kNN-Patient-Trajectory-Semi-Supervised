#!/usr/bin/env python3
"""Stage 1 - build diagnosis code sets and time-ordered patient trajectories.

Reads the raw MIMIC-III CSVs and writes three artefacts to ``data/``:

``admission_diagnoses.pkl``
    ``{hadm_id: [code, ...]}`` - the per-admission code set. Same shape as the
    original repo's file, so existing notebooks still load.

``patient_trajectories.pkl``
    ``{subject_id: [(hadm_id, admittime, [code, ...]), ...]}`` sorted by
    admission time. The original pipeline computed patient-to-admission
    groupings and then never used them; the trajectory task needs them.

``vocabulary.pkl``
    The frequency-ordered code vocabulary.

Usage
-----
    python prepare_data.py
    python prepare_data.py --icd_digits 4 --min_code_frequency 10
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections import Counter

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.config import load_config
from models.io_utils import save_json, save_pickle, setup_logging
from models.representations import CodeVocabulary

log = logging.getLogger("prepare_data")


def truncate_code(code: object, digits: int) -> str | None:
    """Normalise an ICD-9 code and truncate it to ``digits`` characters.

    ``digits=0`` keeps the full code. V and E codes are kept whole when
    truncation would cut into the numeric part in a way that loses the code
    class, which is why the prefix is checked explicitly rather than blindly
    slicing.
    """
    if code is None or pd.isna(code):
        return None
    text = str(code).strip().upper()
    if not text:
        return None
    if digits <= 0:
        return f"D_{text}"
    if text[0] in {"V", "E"}:
        # V codes carry 3 meaningful characters, E codes 4.
        keep = max(digits, 3 if text[0] == "V" else 4)
        return f"D_{text[:keep]}"
    return f"D_{text[:digits]}"


def build(cfg) -> None:
    paths = cfg.paths
    paths.ensure()
    digits = int(cfg.data["icd_digits"])

    log.info("Reading MIMIC-III from %s", paths.mimic_dir)
    admissions = pd.read_csv(
        paths.mimic("ADMISSIONS.csv"),
        usecols=["SUBJECT_ID", "HADM_ID", "ADMITTIME"],
    )
    diagnoses = pd.read_csv(
        paths.mimic("DIAGNOSES_ICD.csv"),
        usecols=["SUBJECT_ID", "HADM_ID", "SEQ_NUM", "ICD9_CODE"],
    )

    admissions = admissions.dropna(subset=["SUBJECT_ID", "HADM_ID"])
    admissions["ADMITTIME"] = pd.to_datetime(admissions["ADMITTIME"], errors="coerce")
    admissions = admissions.dropna(subset=["ADMITTIME"])
    diagnoses = diagnoses.dropna(subset=["HADM_ID", "ICD9_CODE"])

    log.info(
        "Raw: %d admissions, %d patients, %d diagnosis rows",
        admissions["HADM_ID"].nunique(),
        admissions["SUBJECT_ID"].nunique(),
        len(diagnoses),
    )

    # -- code normalisation --------------------------------------------------
    diagnoses["CODE"] = diagnoses["ICD9_CODE"].map(lambda c: truncate_code(c, digits))
    diagnoses = diagnoses.dropna(subset=["CODE"])
    log.info(
        "ICD-9 truncated to %s digits: %d distinct codes",
        digits or "full",
        diagnoses["CODE"].nunique(),
    )

    # SEQ_NUM ordering is retained here only so the artefact is reproducible.
    # Nothing downstream treats position as clinically meaningful.
    diagnoses = diagnoses.sort_values(["HADM_ID", "SEQ_NUM"])

    # -- rare-code filtering -------------------------------------------------
    min_freq = int(cfg.data["min_code_frequency"])
    if min_freq > 1:
        per_admission = diagnoses.drop_duplicates(subset=["HADM_ID", "CODE"])
        counts = per_admission["CODE"].value_counts()
        keep = set(counts[counts >= min_freq].index)
        before = diagnoses["CODE"].nunique()
        diagnoses = diagnoses[diagnoses["CODE"].isin(keep)]
        log.info(
            "Rare-code filter (>=%d admissions): %d -> %d codes",
            min_freq,
            before,
            diagnoses["CODE"].nunique(),
        )

    # -- admission -> ordered unique code list -------------------------------
    admission_diagnoses: dict[int, list[str]] = {}
    for hadm_id, group in diagnoses.groupby("HADM_ID", sort=False):
        seen: list[str] = []
        for code in group["CODE"]:
            if code not in seen:
                seen.append(code)
        admission_diagnoses[int(hadm_id)] = seen

    min_codes = int(cfg.data["min_codes_per_admission"])
    admission_diagnoses = {
        k: v for k, v in admission_diagnoses.items() if len(v) >= min_codes
    }
    log.info(
        "Admissions with >=%d distinct codes: %d",
        min_codes,
        len(admission_diagnoses),
    )

    # -- patient -> time-ordered trajectory ----------------------------------
    admissions = admissions[admissions["HADM_ID"].isin(admission_diagnoses)]
    admissions = admissions.sort_values(["SUBJECT_ID", "ADMITTIME"])

    trajectories: dict[int, list[tuple[int, str, list[str]]]] = {}
    for subject_id, group in admissions.groupby("SUBJECT_ID", sort=False):
        visits = [
            (int(row.HADM_ID), row.ADMITTIME.isoformat(), admission_diagnoses[int(row.HADM_ID)])
            for row in group.itertuples()
        ]
        trajectories[int(subject_id)] = visits

    visit_counts = Counter(len(v) for v in trajectories.values())
    multi = sum(n for length, n in visit_counts.items() if length >= 2)
    log.info(
        "Patients: %d total, %d with >=2 visits (%.1f%%)",
        len(trajectories),
        multi,
        100.0 * multi / max(1, len(trajectories)),
    )
    log.info(
        "Visit-count distribution (top 10): %s",
        sorted(visit_counts.items())[:10],
    )

    # -- vocabulary ----------------------------------------------------------
    vocab = CodeVocabulary.from_sequences(admission_diagnoses.values())

    # -- write ---------------------------------------------------------------
    save_pickle(admission_diagnoses, paths.data("admission_diagnoses.pkl"))
    save_pickle(trajectories, paths.data("patient_trajectories.pkl"))
    save_pickle(vocab.to_dict(), paths.data("vocabulary.pkl"))

    save_json(
        {
            "icd_digits": digits,
            "n_admissions": len(admission_diagnoses),
            "n_patients": len(trajectories),
            "n_patients_multi_visit": multi,
            "vocabulary_size": len(vocab),
            "mean_codes_per_admission": float(
                sum(len(v) for v in admission_diagnoses.values())
                / max(1, len(admission_diagnoses))
            ),
            "visit_count_distribution": dict(sorted(visit_counts.items())),
        },
        paths.results(f"data_summary_icd{digits}.json"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--mimic_dir", default=None)
    parser.add_argument("--project_root", default=None)
    parser.add_argument("--icd_digits", type=int, default=None)
    parser.add_argument("--min_code_frequency", type=int, default=None)
    parser.add_argument("--min_codes_per_admission", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(
        args.config,
        {
            "paths.mimic_dir": args.mimic_dir,
            "paths.project_root": args.project_root,
            "data.icd_digits": args.icd_digits,
            "data.min_code_frequency": args.min_code_frequency,
            "data.min_codes_per_admission": args.min_codes_per_admission,
        },
    )
    setup_logging(cfg.paths.logs_dir, "prepare_data", cfg.runtime["log_level"])
    build(cfg)


if __name__ == "__main__":
    main()
