#!/usr/bin/env python3
"""Collapse a directory of result JSONs into one CSV table.

    python scratch/collect_results.py --pattern "sweep_sim_*"

Writes ``results/collected_<pattern>.csv`` and prints the table sorted by the
headline metric, which is what goes into the Overleaf results section.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--project_root", default=None)
    parser.add_argument("--pattern", default="*")
    parser.add_argument("--sort_by", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config, {"paths.project_root": args.project_root})
    files = sorted(glob.glob(cfg.paths.results(f"{args.pattern}.json")))
    if not files:
        print(f"No result files matching {args.pattern!r} in {cfg.paths.results_dir}")
        return

    rows = []
    for path in files:
        with open(path) as fh:
            payload = json.load(fh)
        conf = payload.get("config", {})
        for model_name, metrics in payload.get("results", {}).items():
            row = {
                "file": os.path.basename(path),
                "model": model_name,
                "task": conf.get("task", {}).get("name"),
                "similarity": conf.get("model", {}).get("similarity"),
                "k": conf.get("model", {}).get("k"),
                "icd_digits": conf.get("data", {}).get("icd_digits"),
                "vote_weighting": conf.get("model", {}).get("vote_weighting"),
            }
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    row[key] = value
            rows.append(row)

    frame = pd.DataFrame(rows)
    sort_key = args.sort_by or next(
        (c for c in ("hits@10", "recall@10", "mrr") if c in frame.columns), None
    )
    if sort_key:
        frame = frame.sort_values(sort_key, ascending=False)

    out = cfg.paths.results(f"collected_{args.pattern.strip('*_')}.csv")
    frame.to_csv(out, index=False)

    with pd.option_context("display.width", 200, "display.max_columns", 40):
        print(frame.to_string(index=False))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
