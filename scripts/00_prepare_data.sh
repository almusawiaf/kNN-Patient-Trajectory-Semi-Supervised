#!/bin/bash
#SBATCH --job-name=prep_data
#SBATCH --output=logs/slurm_prep_%j.out
#SBATCH --error=logs/slurm_prep_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# Stage 1: MIMIC-III CSVs -> admission code sets + patient trajectories.
# Run once per ICD-9 granularity you want to study.

set -euo pipefail
cd "$(dirname "$0")/.."

DIGITS="${1:-3}"

echo "Preparing data at ICD-9 granularity: ${DIGITS} digits"
python prepare_data.py --icd_digits "${DIGITS}"
