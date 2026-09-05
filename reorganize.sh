#!/bin/bash
# ============================================================================
# Reorganize a flat dump of project files into the intended folder structure.
#
# Run this from inside the folder that currently holds all the .py/.sh files
# sitting side by side:
#
#     bash reorganize.sh
#
# Safe to run twice: files already in the right place are left alone. Nothing
# is deleted. Your OLD/ folder is not touched.
# ============================================================================

set -euo pipefail

echo "Reorganizing project layout in: $(pwd)"
echo

# -- create the directories --------------------------------------------------
for d in models data logs results scripts scratch; do
    mkdir -p "$d"
done
for d in data logs results; do
    touch "$d/.gitkeep"
done

# -- helper ------------------------------------------------------------------
move() {
    local file="$1" dest="$2"
    if [ -f "$file" ]; then
        mv "$file" "$dest/"
        echo "  $file -> $dest/"
    elif [ -f "$dest/$file" ]; then
        echo "  $file already in $dest/"
    else
        echo "  ! $file not found (skipping)"
    fi
}

# -- library code ------------------------------------------------------------
echo "models/ (library code, imported as 'from models.x import y')"
for f in __init__.py config.py io_utils.py representations.py similarity.py \
         dtw.py imputation.py baselines.py hierarchical.py evaluation.py; do
    move "$f" models
done
echo

# -- shell wrappers ----------------------------------------------------------
echo "scripts/ (SLURM + shell wrappers)"
for f in 00_prepare_data.sh 01_build_labels.sh 02_run_single.sh \
         03_sweep_similarity.sh 04_sweep_granularity.sh; do
    move "$f" scripts
done
chmod +x scripts/*.sh 2>/dev/null || true
echo

# -- tests and throwaway analysis -------------------------------------------
echo "scratch/ (tests, fixtures, ad-hoc analysis)"
for f in test_similarity.py test_pipeline_synthetic.py make_fake_mimic.py \
         collect_results.py; do
    move "$f" scratch
done
echo

# -- entry points stay at the root -------------------------------------------
echo "Staying at root (pipeline entry points + config):"
for f in prepare_data.py build_labels.py run_experiment.py \
         config.yaml README.md requirements.txt; do
    [ -f "$f" ] && echo "  $f"
done
echo

# -- verify ------------------------------------------------------------------
echo "Verifying imports resolve..."
if python3 -c "import sys; sys.path.insert(0, '.'); import models; print('  models package OK, version', models.__version__)" 2>/dev/null; then
    :
else
    echo "  ! Import check failed. Is models/__init__.py present?"
fi

echo
echo "Done. Final layout:"
find . -maxdepth 2 -not -path './.git*' -not -path './OLD*' -not -name '.gitkeep' \
    | sort | sed 's|^\./||' | sed '/^\.$/d' | sed 's|^|  |'