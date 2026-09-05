#!/bin/bash
# =============================================================================
# Stage 3 — run one experiment configuration end to end.
#
# Usage (interactive):
#   ./scripts/02_run_single.sh [TASK] [SIM] [K] [ICD_DIGITS] [VOTE] [DTW_WIN]
#
# Examples:
#   ./scripts/02_run_single.sh within_visit jaccard   25          # Jaccard baseline
#   ./scripts/02_run_single.sh within_visit dtw_tag   25          # paper's DTW method
#   ./scripts/02_run_single.sh within_visit dtw_tag   25 4        # 4-digit ICD codes
#   ./scripts/02_run_single.sh within_visit dtw_tag   25 3 rank 5 # rank weighting, window=5
#   ./scripts/02_run_single.sh next_visit   dtw_jaccard 15        # trajectory task
#
# SLURM (adjust time/mem for dtw_tag — it is much slower than set measures):
#   sbatch --time=02:00:00 --mem=32G  scripts/02_run_single.sh within_visit jaccard 25
#   sbatch --time=12:00:00 --mem=64G  scripts/02_run_single.sh within_visit dtw_tag 25
#   sbatch --time=24:00:00 --mem=64G  scripts/02_run_single.sh within_visit dtw_tag 25 4
#
# Outputs:
#   results/<TAG>.json   — full metrics + config + hierarchical summary
#   logs/run_<TAG>_<SLURM_JOB_ID>.log
# =============================================================================

#SBATCH --job-name=knn_single
#SBATCH --output=logs/slurm_run_%j.out
#SBATCH --error=logs/slurm_run_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

set -euo pipefail

# SLURM copies the script to its spool directory before running it, so
# dirname "$0" points into /var/spool/slurm/... rather than your project.
# SLURM_SUBMIT_DIR is always the directory you ran sbatch from, which is
# what we want. Fall back to dirname "$0" for interactive use.
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_ROOT"

# ── arguments (all optional, sensible defaults) ────────────────────────────
TASK="${1:-within_visit}"       # within_visit | next_visit
SIM="${2:-jaccard}"             # jaccard | cosine | dtw_tag | dtw_jaccard | ...
K="${3:-25}"                    # number of neighbours
ICD_DIGITS="${4:-3}"            # ICD-9 truncation depth: 3 | 4 | 5
VOTE="${5:-similarity}"         # vote weighting: similarity | rank | uniform
DTW_WIN="${6:-}"                # Sakoe-Chiba window (leave blank = no constraint)

# ── derived tag used for the result filename ───────────────────────────────
TAG="${TASK}_icd${ICD_DIGITS}_${SIM}_k${K}"
[[ -n "$DTW_WIN" ]] && TAG="${TAG}_w${DTW_WIN}"

# ── announce ──────────────────────────────────────────────────────────────
echo "============================================================"
echo "  Job         : ${SLURM_JOB_ID:-local}"
echo "  Task        : $TASK"
echo "  Similarity  : $SIM"
echo "  K           : $K"
echo "  ICD digits  : $ICD_DIGITS"
echo "  Vote weight : $VOTE"
echo "  DTW window  : ${DTW_WIN:-none}"
echo "  Result tag  : $TAG"
echo "  Started     : $(date)"
echo "============================================================"

# ── warn if dtw_tag is asked without adequate time ────────────────────────
if [[ "$SIM" == "dtw_tag" || "$SIM" == "dtw_jaccard" ]]; then
    echo "NOTE: DTW similarity is O(n*m*L^2) — allow 4-24 h depending on cohort size."
fi

# ── optional: rebuild data if ICD digit depth differs from cached ──────────
SUMMARY="results/data_summary_icd${ICD_DIGITS}.json"
if [[ ! -f "$SUMMARY" ]]; then
    echo "→ data_summary_icd${ICD_DIGITS}.json not found — running prepare_data.py"
    python prepare_data.py --icd_digits "$ICD_DIGITS"
fi

# ── optional: rebuild labels if task pickle is missing ────────────────────
LABELS="data/task_${TASK}.pkl"
if [[ ! -f "$LABELS" ]]; then
    echo "→ task_${TASK}.pkl not found — running build_labels.py"
    python build_labels.py --task "$TASK"
fi

# ── build the python command ───────────────────────────────────────────────
CMD="python run_experiment.py \
    --task         $TASK \
    --similarity   $SIM \
    --k            $K \
    --icd_digits   $ICD_DIGITS \
    --vote_weighting $VOTE \
    --tag          $TAG"

[[ -n "$DTW_WIN" ]] && CMD="$CMD --dtw_window $DTW_WIN"

echo "→ $CMD"
echo "------------------------------------------------------------"
eval "$CMD"

# ── print headline metrics from the result JSON ───────────────────────────
RESULT="results/${TAG}.json"
if [[ -f "$RESULT" ]] && command -v python &>/dev/null; then
    echo "------------------------------------------------------------"
    echo "Results summary:"
    python - "$RESULT" "$SIM" << 'PYEOF'
import json, sys

path, sim_key = sys.argv[1], f"knn_{sys.argv[2]}"
data    = json.load(open(path))
results = data.get("results", {})

models_order = [sim_key] + [k for k in results if k != sim_key]
header = f"  {'model':<26}" + "".join(f"{'hits@'+str(k):>10}" for k in [1,5,10,20]) + f"{'mrr':>10}"
print(header)
print("  " + "-" * (len(header) - 2))
for name in models_order:
    if name not in results:
        continue
    m = results[name]
    row = f"  {name:<26}"
    for k in [1, 5, 10, 20]:
        v = m.get(f"hits@{k}")
        row += f"{v:>10.4f}" if v is not None else f"{'—':>10}"
    mrr = m.get("mrr")
    row += f"{mrr:>10.4f}" if mrr is not None else f"{'—':>10}"
    print(row)

hier = data.get("hierarchical")
if hier:
    ratio = hier.get("cumulative_vs_full_rebuild", float("nan"))
    saved = max(0.0, 1.0 - ratio)
    print(f"\n  Hierarchical update: {hier['total_recomputations']} recomputations "
          f"across {hier['n_batches']} batches "
          f"({saved*100:.1f}% work saved vs full rebuild)")
PYEOF
fi

echo "============================================================"
echo "  Finished    : $(date)"
echo "  Result      : $RESULT"
echo "============================================================"