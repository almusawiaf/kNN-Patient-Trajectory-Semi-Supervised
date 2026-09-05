#!/usr/bin/env bash
#SBATCH --job-name=knn_single
#SBATCH --output=logs/full_pipeline_%j.out
#SBATCH --error=logs/full_pipeline_%j.err
#SBATCH --time=4-00:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

set -e  # Exit immediately if any command fails

echo "=== Step 1: Cleaning and preparing data (vocab=888) ==="
python prepare_data.py --icd_digits 3 --min_code_frequency 5

echo "=== Step 2: Building labels ==="
python build_labels.py --task within_visit

echo "=== Step 3: Running Jaccard baseline ==="
# Running directly via bash inside the job allocation
bash scripts/02_run_single.sh within_visit jaccard 25

echo "=== Step 4: Running DTW without hierarchical update ==="
# Passing --no_hierarchical directly to run_experiment inside the script
bash scripts/02_run_single.sh within_visit dtw_tag 25 3 similarity --no_hierarchical

echo "=== Pipeline finished successfully! ==="