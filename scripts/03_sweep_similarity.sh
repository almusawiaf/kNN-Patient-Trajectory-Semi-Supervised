#!/bin/bash
#SBATCH --job-name=sweep_sim
#SBATCH --output=logs/slurm_sweep_sim_%j.out
#SBATCH --error=logs/slurm_sweep_sim_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

# Sweep similarity measure x k.
#
# Note on which measures are actually distinct: on binary code vectors,
# Manhattan, Euclidean and Hamming distance are all equal to the size of the
# symmetric difference, so they produce identical neighbour rankings. Only
# `hamming` is run below; adding the other two would triple the runtime to
# reproduce the same numbers three times.
#
#   ./scripts/03_sweep_similarity.sh within_visit

set -euo pipefail
cd "$(dirname "$0")/.."

TASK="${1:-within_visit}"

MEASURES=(jaccard dice cosine overlap hamming mahalanobis)
KS=(5 10 25 50 100)

for SIM in "${MEASURES[@]}"; do
  for K in "${KS[@]}"; do
    echo "=== ${TASK} | ${SIM} | k=${K} ==="
    python run_experiment.py \
      --task "${TASK}" \
      --similarity "${SIM}" \
      --k "${K}" \
      --no_hierarchical \
      --tag "sweep_sim_${TASK}_${SIM}_k${K}"
  done
done

echo "Sweep complete. Collecting results..."
python scratch/collect_results.py --pattern "sweep_sim_${TASK}_*"
