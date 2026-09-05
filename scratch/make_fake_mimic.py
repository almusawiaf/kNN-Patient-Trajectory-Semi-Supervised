#!/usr/bin/env python3
"""Write a miniature fake MIMIC-III directory for testing the CSV stages.

Real MIMIC lives behind a data use agreement and is not always mounted on the
node you are debugging on. This produces CSVs with the same column names and
dtypes so ``prepare_data.py`` can be exercised anywhere.

    python scratch/make_fake_mimic.py --out /tmp/fake_mimic
"""

from __future__ import annotations

import argparse
import os
import random
from datetime import datetime, timedelta

import pandas as pd

ICD_POOL = [
    "4019", "4280", "42731", "41401", "5849", "25000", "2724", "51881",
    "5990", "53081", "2859", "2449", "486", "0389", "99592", "78552",
    "V5861", "V4581", "E8798", "1970", "1749", "34590", "56400", "71590",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="/tmp/fake_mimic")
    parser.add_argument("--n_patients", type=int, default=400)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(args.out, exist_ok=True)

    # Latent profiles so neighbour structure exists in the fake data too.
    profiles = [rng.sample(ICD_POOL, k=rng.randint(5, 9)) for _ in range(8)]

    adm_rows, dx_rows = [], []
    hadm_id = 100000
    base = datetime(2150, 1, 1)

    for subject_id in range(1, args.n_patients + 1):
        profile = rng.choice(profiles)
        n_visits = rng.choices([1, 2, 3, 4], weights=[55, 25, 13, 7])[0]
        when = base + timedelta(days=rng.randint(0, 900))

        for _ in range(n_visits):
            hadm_id += 1
            when += timedelta(days=rng.randint(30, 400))
            adm_rows.append(
                {
                    "SUBJECT_ID": subject_id,
                    "HADM_ID": hadm_id,
                    "ADMITTIME": when.strftime("%Y-%m-%d %H:%M:%S"),
                    "DISCHTIME": (when + timedelta(days=rng.randint(1, 14))).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                }
            )
            codes = set(rng.sample(profile, k=max(2, len(profile) - rng.randint(0, 3))))
            codes.update(rng.sample(ICD_POOL, k=rng.randint(0, 3)))
            for seq_num, code in enumerate(sorted(codes), start=1):
                dx_rows.append(
                    {
                        "ROW_ID": len(dx_rows) + 1,
                        "SUBJECT_ID": subject_id,
                        "HADM_ID": hadm_id,
                        "SEQ_NUM": seq_num,
                        "ICD9_CODE": code,
                    }
                )

    pd.DataFrame(adm_rows).to_csv(os.path.join(args.out, "ADMISSIONS.csv"), index=False)
    pd.DataFrame(dx_rows).to_csv(os.path.join(args.out, "DIAGNOSES_ICD.csv"), index=False)

    print(f"Wrote {len(adm_rows)} admissions, {len(dx_rows)} diagnosis rows to {args.out}")


if __name__ == "__main__":
    main()
