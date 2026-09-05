#!/bin/bash
# =============================================================================
# GPU-accelerated experiment run.
#
# Automatically picks the fastest available backend:
#   CuPy CUDA  → ~2-4 min  for DTW (vs 5.8h pure Python)
#   Numba JIT  → ~7 min    for DTW (no GPU needed)
#   CPU sparse → ~6 sec    for Jaccard/cosine
#
# Usage:
#   sbatch scripts/05_run_gpu.sh within_visit jaccard  25
#   sbatch scripts/05_run_gpu.sh within_visit dtw_tag  25
#   sbatch scripts/05_run_gpu.sh within_visit dtw_tag  25 3 similarity 5
#
# =============================================================================

#SBATCH --job-name=knn_gpu
#SBATCH --output=logs/slurm_gpu_%j.out
#SBATCH --error=logs/slurm_gpu_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1              # request one GPU
#SBATCH --partition=gpu           # change to your cluster's GPU partition name

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_ROOT"

# ── arguments ──────────────────────────────────────────────────────────────
TASK="${1:-within_visit}"
SIM="${2:-dtw_tag}"
K="${3:-25}"
ICD_DIGITS="${4:-3}"
VOTE="${5:-similarity}"
DTW_WIN="${6:-}"

TAG="${TASK}_icd${ICD_DIGITS}_${SIM}_k${K}"
[[ -n "$DTW_WIN" ]] && TAG="${TAG}_w${DTW_WIN}"

# ── environment setup ──────────────────────────────────────────────────────
echo "============================================================"
echo "  Job         : ${SLURM_JOB_ID:-local}"
echo "  Node        : $(hostname)"
echo "  Task        : $TASK"
echo "  Similarity  : $SIM"
echo "  K           : $K"
echo "  ICD digits  : $ICD_DIGITS"
echo "  DTW window  : ${DTW_WIN:-none}"
echo "  Result tag  : $TAG"
echo "  Started     : $(date)"
echo "============================================================"

# Print GPU info
nvidia-smi --query-gpu=name,memory.total,driver_version \
           --format=csv,noheader 2>/dev/null || echo "nvidia-smi not available"

# Check which Python acceleration is available
python - << 'PYEOF'
print("Checking acceleration backends...")
try:
    import cupy as cp
    print(f"  CuPy {cp.__version__} — CUDA kernel will be used")
    dev = cp.cuda.runtime.getDeviceProperties(0)
    print(f"  GPU: {dev['name'].decode()}")
except ImportError:
    print("  CuPy not available")
try:
    import numba
    print(f"  Numba {numba.__version__} — JIT kernel will be used")
except ImportError:
    print("  Numba not available — falling back to pure Python")
PYEOF

# ── check prerequisites ────────────────────────────────────────────────────
SUMMARY="results/data_summary_icd${ICD_DIGITS}.json"
if [[ ! -f "$SUMMARY" ]]; then
    echo "→ Running prepare_data.py (icd_digits=${ICD_DIGITS})"
    python prepare_data.py --icd_digits "$ICD_DIGITS"
fi

LABELS="data/task_${TASK}.pkl"
if [[ ! -f "$LABELS" ]]; then
    echo "→ Running build_labels.py (task=${TASK})"
    python build_labels.py --task "$TASK"
fi

# ── build command ──────────────────────────────────────────────────────────
CMD="python run_experiment.py \
    --task           $TASK \
    --similarity     $SIM \
    --k              $K \
    --icd_digits     $ICD_DIGITS \
    --vote_weighting $VOTE \
    --no_hierarchical \
    --tag            $TAG"

# DTW hierarchical is prohibitively slow — disable unless specifically requested
if [[ "${7:-}" == "--with_hierarchical" ]]; then
    CMD="${CMD/--no_hierarchical/}"
    echo "NOTE: hierarchical update enabled — may take many hours with DTW"
fi

[[ -n "$DTW_WIN" ]] && CMD="$CMD --dtw_window $DTW_WIN"

echo "→ $CMD"
echo "------------------------------------------------------------"
eval "$CMD"

# ── results summary ────────────────────────────────────────────────────────
RESULT="results/${TAG}.json"
if [[ -f "$RESULT" ]]; then
    echo "------------------------------------------------------------"
    python - "$RESULT" "$SIM" << 'PYEOF'
import json, sys
path, sim_key = sys.argv[1], f"knn_{sys.argv[2]}"
data    = json.load(open(path))
results = data.get("results", {})
order   = [sim_key] + [k for k in results if k != sim_key]
header  = f"  {'model':<28}" + "".join(f"{'hits@'+str(k):>10}" for k in [1,5,10,20]) + f"{'mrr':>10}"
print(header)
print("  " + "-" * (len(header)-2))
for name in order:
    if name not in results: continue
    m   = results[name]
    row = f"  {name:<28}"
    for k in [1, 5, 10, 20]:
        v = m.get(f"hits@{k}")
        row += f"{v:>10.4f}" if v is not None else f"{'—':>10}"
    mrr = m.get("mrr")
    row += f"{mrr:>10.4f}" if mrr is not None else f"{'—':>10}"
    print(row)
PYEOF
fi

echo "============================================================"
echo "  Finished    : $(date)"
echo "  Result      : ${RESULT}"
echo "============================================================"
