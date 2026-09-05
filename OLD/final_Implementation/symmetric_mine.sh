#!/bin/bash

#SBATCH --output=X_output_%j.out            # Standard output and error log (%j expands to jobId)
#SBATCH --error=X_error_%j.err              # Error file
#SBATCH --partition=cpu-small               # Specify the partition
#SBATCH --ntasks=1                          # Total number of tasks across all nodes
#SBATCH --cpus-per-task=18                  # Number of CPU cores per task (maximum for this partition)
#SBATCH --mem=64G                          # Memory per node (choose the upper range for safety)

#SBATCH --job-name=DTW_experiment1_25percent          # Job name

jupyter nbconvert --to notebook --execute main.ipynb --output main_1.ipynb