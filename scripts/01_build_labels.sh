#!/bin/bash
#SBATCH --job-name=build_labels
#SBATCH --output=logs/slurm_labels_%j.out
#SBATCH --error=logs/slurm_labels_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=16G

# Stage 2: build the masked prediction task.

set -euo pipefail
cd "$(dirname "$0")/.."

TASK="${1:-within_visit}"

echo "Building labels for task: ${TASK}"
python build_labels.py --task "${TASK}"
