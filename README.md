# Diagnostic Trajectory Generation and Prediction

Semi-supervised k-nearest-neighbour imputation over patient diagnosis
trajectories from MIMIC-III. Rewrite of
[`kNN-Patient-Trajectory-Semi-Supervised`](https://github.com/almusawiaf/kNN-Patient-Trajectory-Semi-Supervised)
with the representation and evaluation layers rebuilt.

---

## Layout

```
.
├── config.yaml              # every knob, one place
├── prepare_data.py          # stage 1: MIMIC CSVs -> code sets + trajectories
├── build_labels.py          # stage 2: masking and train/test split
├── run_experiment.py        # stage 3: retrieval, imputation, scoring
├── models/                  # library code
│   ├── config.py            #   config loading, path resolution
│   ├── io_utils.py          #   pickle/JSON, logging
│   ├── representations.py   #   vocabulary, sparse multi-hot matrices
│   ├── similarity.py        #   Jaccard, Dice, cosine, overlap, Hamming, Mahalanobis
│   ├── dtw.py               #   DTW over visit sequences
│   ├── imputation.py        #   neighbour voting -> ranked candidates
│   ├── baselines.py         #   frequency prior, PPMI co-occurrence
│   ├── hierarchical.py      #   tiered kNN update
│   └── evaluation.py        #   hits@k, MRR, recall@k, stratified reporting
├── data/                    # generated pickles (git-ignored)
├── logs/                    # timestamped run logs + SLURM output
├── results/                 # one JSON per run, plus collected CSVs
├── scripts/                 # SLURM-ready shell wrappers
└── scratch/                 # tests and throwaway analysis
```

Paths default to the cluster locations from the original repo:

- MIMIC-III CSVs: `/lustre/home/almusawiaf/PhD_Projects/MIMIC_resources`
- Project root: `/lustre/home/almusawiaf/PhD_Projects/Satyaki/Semi_Supervised`

Override with `--mimic_dir` / `--project_root` or by editing `config.yaml`.

---

## Quick start

```bash
pip install -r requirements.txt

# Verify the code works before touching real data
python scratch/test_similarity.py
python scratch/test_pipeline_synthetic.py

# Real pipeline
python prepare_data.py --icd_digits 3
python build_labels.py --task within_visit
python run_experiment.py --similarity jaccard --k 25
```

On SLURM:

```bash
sbatch scripts/00_prepare_data.sh 3
sbatch scripts/01_build_labels.sh within_visit
sbatch scripts/02_run_single.sh within_visit jaccard 25
sbatch scripts/03_sweep_similarity.sh within_visit
```

---

## The two tasks

**`within_visit`** — hold out one diagnosis code from an admission, predict it
from the remaining codes. This is the original formulation, kept so results are
comparable.

**`next_visit`** — hold out a patient's final visit, predict its code set from
the earlier visits. This is the trajectory task the project is aiming at, and
the only setting where DTW is meaningful.

---

## What changed from the original, and why

### 1. Codes are no longer treated as numbers

The original built a vocabulary by order of first appearance (`D_250` → 1,
`D_337` → 2, …) and then computed DTW with local cost
`|index_a − index_b|`. That makes the distance between two diagnoses a function
of where they happened to appear in the CSV. `D_250` and `D_337` scored as
near-identical; `D_250` and `D_486` scored 23 apart. The neighbour graph was
essentially noise, which is the main reason accuracy sat near zero.

Integers are now only column positions in a sparse binary matrix. Distance
comes from set overlap, so the index values never enter arithmetic.

### 2. Imputation no longer depends on position

The original read the neighbour's code *at the masked index*: mask at position
7 → look up `neighbour[7]`. Diagnosis order inside a MIMIC admission comes from
`SEQ_NUM`, a billing artefact, so position 7 in two different admissions has no
correspondence. Neighbours shorter than the mask position were silently
skipped, dropping their votes entirely.

Now every code a neighbour carries is a candidate, weighted by that neighbour's
similarity. Codes already visible in the query are removed, since they cannot
be the answer.

### 3. Ranked output instead of argmax

The old pipeline emitted one code and scored exact match. On a ~900-way problem
with a heavy-tailed label distribution, that number is close to uninformative.
The model now returns a ranked list, and evaluation reports hits@{1,5,10,20}
and MRR — the top-k accuracy raised in the meeting.

### 4. Baselines that expose the failure mode

The original predicted `D_250` (diabetes) or `D_401` (hypertension) for nearly
every query — the two most frequent codes in MIMIC. That is a frequency prior
wearing a kNN costume. `FrequencyPrior` and `CooccurrenceScorer` now run on the
same queries every time. **If the kNN model does not clear both, the neighbour
structure is contributing nothing** and no amount of hyperparameter tuning will
fix it.

### 5. DTW moved to the axis that has time on it

DTW was being applied to the code list inside a *single* admission — an
unordered set, with no time axis to warp. It now operates on the patient's
sequence of visits, ordered by `ADMITTIME`, with local cost
`1 − Jaccard(visit_a, visit_b)`, plus an optional Sakoe-Chiba band and
path-length normalisation so long trajectories aren't penalised for being long.

### 6. Patient trajectories are actually built

The original computed `patient_admissions` in notebook 1 and never used it.
`prepare_data.py` now writes time-ordered trajectories, which the `next_visit`
task consumes.

---

## Two things to know before running the sweeps

**Manhattan, Euclidean and Hamming are the same measure here.** On binary
vectors all three equal the symmetric-difference size:

```
L1(a,b) = L2(a,b)² = |a| + |b| − 2·|a ∩ b|
```

They induce identical neighbour rankings — `scratch/test_similarity.py`
asserts it. Running all three produces one result three times. The measures
that genuinely differ are those that normalise the intersection differently:
Jaccard (by union), Dice (by total size), cosine (geometrically), overlap (by
the smaller set). `scripts/03_sweep_similarity.sh` runs only the distinct ones.

**Most MIMIC-III patients have exactly one admission.** `prepare_data.py`
logs the visit-count distribution; check it before committing to the
`next_visit` task. If the multi-visit cohort is too small to support the
trajectory experiments, that is a finding worth reporting rather than a
blocker — it is precisely the data limitation that came up in the meeting, and
it is the argument for bringing in a registry dataset with real longitudinal
follow-up.

---

## Interpreting results

Each run writes `results/<tag>.json` containing the full config, per-model
metrics, and the hierarchical-update summary. Collect a sweep with:

```bash
python scratch/collect_results.py --pattern "sweep_sim_within_visit_*"
```

Read `by_frequency_stratum` before the headline number. Q4 (common codes) will
always look best; Q1 (rare codes) is where a method either has real signal or
is riding the prior. A model that beats the baseline only on Q4 has learned
which diagnoses are common, which is not the contribution being claimed.

---

## Next steps

- [ ] Run `03_sweep_similarity.sh` — settles the distance-measure question
- [ ] Run `04_sweep_granularity.sh` — settles 3 vs 4 vs 5 digits
- [ ] Report visit-count distribution; decide if `next_visit` is viable on MIMIC
- [ ] Quantify hierarchical-update savings vs. full recompute
- [ ] Port the algorithm blocks to Overleaf
