#!/bin/bash
#SBATCH --job-name=sweep_gran
#SBATCH --output=logs/slurm_sweep_gran_%j.out
#SBATCH --error=logs/slurm_sweep_gran_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

# Sweep ICD-9 truncation depth: 3 (category) -> 5 (full billable code).
#
# This is the "do we go from 3 digits to 4 or 5" question. Each level needs a
# fresh data-preparation pass, because truncation changes the vocabulary, the
# rare-code filter and therefore the whole matrix.
#
# Read the result alongside the vocabulary size in results/data_summary_icd*.json:
# finer codes carry more clinical detail but split each concept across more
# columns, so overlap between any two patients drops. Whether that trade helps
# is exactly what this sweep measures.
#
#   ./scripts/04_sweep_granularity.sh within_visit jaccard

set -euo pipefail
cd "$(dirname "$0")/.."

TASK="${1:-within_visit}"
SIM="${2:-jaccard}"

for DIGITS in 3 4 5; do
  echo "=== Preparing data at ${DIGITS} digits ==="
  python prepare_data.py --icd_digits "${DIGITS}"

  echo "=== Building labels ==="
  python build_labels.py --task "${TASK}"

  echo "=== Running ${SIM} at ${DIGITS} digits ==="
  python run_experiment.py \
    --task "${TASK}" \
    --similarity "${SIM}" \
    --icd_digits "${DIGITS}" \
    --no_hierarchical \
    --tag "sweep_gran_${TASK}_${SIM}_icd${DIGITS}"
done

python scratch/collect_results.py --pattern "sweep_gran_${TASK}_*"
