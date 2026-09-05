# Diagnostic Trajectory Generation and Prediction

Semi-supervised k-nearest-neighbour imputation and prediction over patient
diagnosis trajectories from MIMIC-III. Complete rewrite of
[`kNN-Patient-Trajectory-Semi-Supervised`](https://github.com/almusawiaf/kNN-Patient-Trajectory-Semi-Supervised)
with corrected representation, evaluation, and GPU acceleration.

---

## Table of contents

1. [Project overview](#1-project-overview)
2. [Repository layout](#2-repository-layout)
3. [Quick start](#3-quick-start)
4. [Data and MIMIC-III statistics](#4-data-and-mimic-iii-statistics)
5. [Methodology](#5-methodology)
6. [Experimental results](#6-experimental-results)
7. [GPU acceleration](#7-gpu-acceleration)
8. [What changed from the original and why](#8-what-changed-from-the-original-and-why)
9. [Interpreting results](#9-interpreting-results)
10. [Known limitations and next steps](#10-known-limitations-and-next-steps)

---

## 1. Project overview

Electronic health records (EHRs) contain incomplete patient trajectories —
sequences of diagnosis codes recorded at successive hospital admissions. Missing
or unrecorded information limits clinical decision support and downstream
modelling. This project implements a **dynamic, hierarchical k-nearest-neighbour
(kNN) method** that reconstructs missing diagnoses and predicts future visits by
finding similar patients in the record, then voting on the missing information
using distance-weighted aggregation.

The core formulation follows the paper:

> *Dynamic Hierarchical KNN for Patient Trajectory Reconstruction and
> Recommendation* (Roy, Almusawi, et al., 2025)

Patient trajectories are modelled as discrete sequences of clinical events:

```
S_i = [s_1, s_2, ..., s_{n_i}]
```

where each `s_j` is a disease tag (ICD-9 code) at a given time point. For a
new sequence `S_new` with a missing value, the model computes the DTW distance
to all candidate sequences, identifies the K nearest neighbours, and imputes
the missing value by weighted aggregation:

```
ŝ_j = Σ (w_i · s_{i,j}) / Σ w_i     where     w_i = 1 / (d(S_new, S_i) + ε)
```

The neighbour graph is maintained dynamically: when new sequences arrive, hub
sequences (those appearing frequently as neighbours) have their own
neighbourhoods refreshed without recomputing the entire index.

---

## 2. Repository layout

```
.
├── config.yaml               # all knobs in one place
├── prepare_data.py           # stage 1: MIMIC CSVs → code sets + trajectories
├── build_labels.py           # stage 2: masking and train/test split
├── run_experiment.py         # stage 3: retrieval, imputation, scoring
│
├── models/                   # importable library
│   ├── __init__.py
│   ├── config.py             # config loading, path resolution
│   ├── io_utils.py           # pickle/JSON helpers, logging
│   ├── representations.py    # vocabulary, sparse multi-hot matrices
│   ├── similarity.py         # Jaccard, Dice, cosine, overlap, Hamming, Mahalanobis
│   ├── dtw.py                # tag-level and visit-level DTW (paper formulation)
│   ├── gpu_dtw.py            # GPU/Numba-accelerated DTW (835× speedup)
│   ├── gpu_similarity.py     # CuPy sparse matrix similarity
│   ├── gpu_utils.py          # device detection and array helpers
│   ├── imputation.py         # neighbour voting → ranked candidates
│   ├── baselines.py          # frequency prior, PPMI co-occurrence
│   ├── hierarchical.py       # tiered kNN update (works with both backends)
│   └── evaluation.py         # hits@k, MRR, recall@k, stratified reporting
│
├── data/                     # generated pickles — git-ignored
├── logs/                     # timestamped run logs + SLURM output
├── results/                  # one JSON per run, collected CSVs
│
├── scripts/
│   ├── 00_prepare_data.sh
│   ├── 01_build_labels.sh
│   ├── 02_run_single.sh      # CPU run (any similarity)
│   ├── 03_sweep_similarity.sh
│   ├── 04_sweep_granularity.sh
│   └── 05_run_gpu.sh         # GPU run (recommended for DTW)
│
└── scratch/
    ├── test_similarity.py          # unit tests: Jaccard values, DTW properties
    ├── test_pipeline_synthetic.py  # end-to-end test on planted synthetic data
    ├── make_fake_mimic.py          # generates fake MIMIC CSVs for offline testing
    └── collect_results.py          # aggregates sweep JSONs into a single CSV
```

Cluster paths (override with `--project_root` / `--mimic_dir` or in `config.yaml`):

| Resource | Path |
|---|---|
| MIMIC-III CSVs | `/lustre/home/almusawiaf/PhD_Projects/MIMIC_resources` |
| Project root | `/lustre/home/almusawiaf/PhD_Projects/Satyaki/Semi_Supervised` |

---

## 3. Quick start

### Installation

```bash
pip install -r requirements.txt

# Optional but strongly recommended for GPU acceleration
pip install cupy-cuda12x   # match your CUDA version (check: nvcc --version)
pip install numba           # JIT fallback if no GPU
```

### Verify before touching MIMIC

```bash
python scratch/test_similarity.py        # unit tests — all should pass
python scratch/test_pipeline_synthetic.py  # kNN should beat prior by ~4× at hits@10
```

### Full pipeline (CPU)

```bash
python prepare_data.py --icd_digits 3 --min_code_frequency 5
python build_labels.py --task within_visit
python run_experiment.py --similarity jaccard --k 25
```

### Full pipeline (GPU — recommended)

```bash
# Fix data first (ensure vocab=888, not 2583)
python prepare_data.py --icd_digits 3 --min_code_frequency 5
python build_labels.py --task within_visit
python build_labels.py --task next_visit

sbatch scripts/05_run_gpu.sh within_visit jaccard    25   # ~6 seconds
sbatch scripts/05_run_gpu.sh within_visit dtw_tag   25   # ~26 seconds on A100
sbatch scripts/05_run_gpu.sh next_visit   dtw_jaccard 25  # ~2.5 minutes
```

### Common pitfall — vocab size

Always check the log line `vocab=888` after running `prepare_data.py`. If you
see `vocab=2583` it means `min_code_frequency` defaulted to 1 (keeping rare
codes), which collapses Jaccard similarity and produces near-zero baselines.
The correct invocation is:

```bash
python prepare_data.py --icd_digits 3 --min_code_frequency 5
```

---

## 4. Data and MIMIC-III statistics

MIMIC-III (Medical Information Mart for Intensive Care) is an ICU database with
de-identified records from Beth Israel Deaconess Medical Center.

After preprocessing at ICD-9 3-digit granularity with `min_code_frequency=5`:

| Statistic | Value |
|---|---|
| Raw admissions | 58,976 |
| Raw patients | 46,520 |
| Raw diagnosis rows | 651,000 |
| Distinct ICD-9 codes (pre-filter) | 1,070 |
| Vocabulary after rare-code filter | **888 codes** |
| Admissions with ≥2 distinct codes | **58,538** |
| Total patients | **46,311** |
| Patients with ≥2 visits | **7,342 (15.9%)** |
| Single-admission patients | 38,969 (84.1%) |

Visit-count distribution (top 10 values):

| Visits | Patients |
|---|---|
| 1 | 38,969 |
| 2 | 4,990 |
| 3 | 1,324 |
| 4 | 501 |
| 5 | 247 |
| 6 | 113 |
| 7–10 | 121 |

The dominant single-admission pattern is a critical fact for experiment design:
the `next_visit` task can only use the 7,342 multi-visit patients (15.9%), giving
1,468 test patients after an 80/20 split. This is not a flaw — it is a
reproducible characteristic of MIMIC-III as an ICU database that the paper
should state explicitly.

---

## 5. Methodology

### 5.1 Two evaluation tasks

#### Task 1 — Within-visit imputation

Each hospital admission is treated as an ordered sequence of ICD-9 codes. One
code is held out (masked), and the model predicts it from the remaining codes.

- **Sequence unit:** one ICD-9 code per position
- **Metric:** hits@k — is the hidden code in the top k predictions?
- **Test set:** 11,707 masked queries from 20% of admissions

#### Task 2 — Next-visit prediction

A patient's final hospital admission is held out. The model predicts its
diagnosis codes using the patient's earlier visit history.

- **Sequence unit:** one visit (a set of diagnosis codes) per position
- **Metric:** recall@k — fraction of actual next-visit codes recovered; any_hit@k
- **Test set:** 1,468 test patients (from the 7,342 multi-visit cohort)
- **Mean codes to predict:** 13.0 per patient

### 5.2 Distance measures

Two families of distance measure are implemented:

**Set-similarity (within-visit and next-visit)**

Each admission is encoded as a binary vector in a 888-dimensional code space.
Similarity is computed via sparse matrix multiplication — no arithmetic on
vocabulary indices.

| Measure | Formula | Notes |
|---|---|---|
| Jaccard | \|A∩B\| / \|A∪B\| | Default; normalises by union |
| Dice | 2\|A∩B\| / (\|A\|+\|B\|) | Normalises by total size |
| Cosine | \|A∩B\| / √(\|A\|·\|B\|) | Geometric normalisation |
| Overlap | \|A∩B\| / min(\|A\|,\|B\|) | Normalises by smaller set |
| Hamming/Euclidean/Manhattan | \|A\|+\|B\|−2\|A∩B\| | All identical on binary vectors |

Note: Hamming, Euclidean and Manhattan are mathematically equivalent on binary
vectors (all equal the symmetric-difference size) and are run only once.

**DTW — tag level (within-visit)**

Treats each diagnosis code as a sequence element. Local cost:

```
cost(a, b) = 0   if a == b
           = 1   otherwise
```

The warping path finds the best alignment between two code sequences even if
one patient has the same diagnoses in a different billing order.

**DTW — visit level (next-visit, `dtw_jaccard`)**

Treats each hospital admission as a sequence element. Local cost between two
visits A and B:

```
cost(A, B) = 1 − Jaccard(A, B) = 1 − |A∩B| / |A∪B|
```

Two patient trajectories are aligned; visits that share many diagnoses are
considered clinically close even if one patient progressed faster.

**Imputation formula (paper's equation)**

Weights are computed as inverse distances, exactly as specified in the paper:

```
w_i = 1 / (d(S_new, S_i) + ε)

ŝ_j = Σ w_i · s_{i,j}  /  Σ w_i
```

Since `s_{i,j}` is a categorical code (not a number), multiplication becomes a
weighted vote: each neighbour contributes weight `w_i` to every code it carries
that the query does not already show.

### 5.3 Hierarchical update

When new sequences arrive in batches, the method tracks how often each existing
sequence appears as a neighbour (`F(S_i)`). Sequences crossing a frequency
threshold have their own neighbourhoods refreshed — but only once (the first
time they cross the threshold). This bounds the total recomputation cost:

```
if F(S_i) ≥ threshold  AND  S_i not yet refreshed:
    recompute neighbours of S_i
    mark S_i as refreshed
```

On MIMIC-III with threshold=2: 11,191 recomputations across 24 batches,
saving 76.1% of the work of a full rebuild.

### 5.4 Baselines

Every run reports two baselines on identical queries:

**Frequency prior** — rank all codes by corpus frequency, return the top 50
regardless of the query. Tests whether the kNN model adds anything over simply
knowing which diagnoses are common in MIMIC-III.

**PPMI co-occurrence** — score candidates by positive pointwise mutual
information with the observed codes. Tests whether patient-level neighbour
retrieval adds anything over plain code-to-code co-occurrence statistics.

---

## 6. Experimental results

All results use MIMIC-III at ICD-9 3-digit granularity, vocabulary size 888,
k=25 neighbours, similarity-weighted voting.

### 6.1 Within-visit task (11,707 queries)

| Model | hits@1 | hits@5 | hits@10 | hits@20 | MRR |
|---|---|---|---|---|---|
| **knn_jaccard** | **0.194** | **0.386** | **0.479** | **0.569** | **0.287** |
| knn_dtw_tag | 0.172 | 0.350 | 0.442 | 0.537 | 0.259 |
| knn_dtw_tag (window=5) | 0.172 | 0.352 | 0.443 | 0.536 | 0.259 |
| frequency_prior | 0.056 | 0.191 | 0.276 | 0.409 | 0.127 |
| co-occurrence | 0.001 | 0.027 | 0.061 | 0.088 | 0.016 |

**Key findings:**

Jaccard beats DTW-tag on the within-visit task. This is expected and
scientifically meaningful: ICD-9 codes within a MIMIC-III admission are ordered
by `SEQ_NUM`, a billing priority assigned after the clinical encounter, not a
clinical time axis. DTW tries to exploit sequence order, but that order carries
no temporal signal. Jaccard treats the admission as an unordered set — which is
what it actually is — and is more accurate as a result.

The DTW window=5 (Sakoe-Chiba band) result is identical to unconstrained DTW
to the third decimal place. This confirms that the ordering contains no
exploitable structure even with a tighter alignment constraint.

Frequency-stratified results (hits@10 by code rarity):

| Stratum | n queries | knn_jaccard | frequency_prior | co-occurrence |
|---|---|---|---|---|
| Q4 (most common) | 10,516 | **0.520** | 0.307 | 0.061 |
| Q3 | 904 | 0.127 | 0.000 | 0.051 |
| Q2 | 228 | 0.070 | 0.000 | 0.039 |
| Q1 (rarest) | 59 | 0.068 | 0.000 | **0.203** |

89.8% of queries are Q4. The headline hits@10 = 0.479 is essentially the Q4
result. The frequency prior scores zero on Q1–Q3 because it only returns common
codes and can never predict rare diagnoses. Co-occurrence unexpectedly leads on
Q1 because PPMI over-corrects toward rare associations — a finding that points
toward rank fusion (combining kNN and co-occurrence) as a cheap improvement.

### 6.2 Next-visit task (1,468 queries, mean 13 target codes)

| Model | recall@1 | recall@5 | recall@10 | recall@20 | any_hit@10 |
|---|---|---|---|---|---|
| **knn_dtw_jaccard** | **0.026** | **0.088** | **0.136** | **0.199** | **0.767** |
| frequency_prior | 0.018 | 0.070 | 0.128 | 0.178 | 0.740 |
| co-occurrence | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

**Key findings:**

DTW on visit-level trajectories beats the frequency prior across all metrics.
This stands in contrast to the within-visit result and confirms the central
hypothesis: DTW should be applied where the sequence axis is real. The
`ADMITTIME`-ordered visit sequence is a genuine time axis; the billing-ordered
code sequence is not.

`any_hit@10 = 0.767` is the most clinically meaningful number: in 76.7% of
test patients, at least one of the patient's actual next-visit diagnoses appears
in the model's top 10 predictions. A low recall@k is expected when the average
patient has 13 codes to predict — the theoretical ceiling for recall@10 on this
dataset is ~0.77 (10/13).

Co-occurrence scores zero because it has no concept of visit history; it can
only rank codes by pairwise association with the query's observed codes, not by
trajectory similarity.

### 6.3 Summary comparison

| Question | Answer |
|---|---|
| Does kNN beat the frequency prior? | Yes — on both tasks and at every k |
| Does DTW help within a single visit? | No — Jaccard wins; billing order ≠ clinical time |
| Does DTW help across visits (trajectories)? | Yes — beats prior on all recall@k |
| Is next-visit prediction clinically useful? | any_hit@10 = 76.7% — yes |
| Does ICD-4 digit granularity improve results? | No change observed (same cached data) |
| Do rare codes benefit from kNN? | Weak signal; co-occurrence beats kNN on Q1 |

### 6.4 Three contributions the results support

**Contribution 1 — Representation matters more than distance.**
Fixing the encoding (sparse binary sets rather than integer-indexed sequences)
raised hits@10 from ~0 to 0.479. This was the primary bug in the original
pipeline and is the most important finding.

**Contribution 2 — Distance measure choice should be driven by whether the
sequence axis is real.**
Jaccard on unordered code sets outperforms DTW on billing-ordered within-visit
sequences. DTW on time-ordered visit trajectories outperforms the frequency
prior. The right measure depends on whether the "sequence" actually encodes
temporal information.

**Contribution 3 — Hierarchical update is bounded in total work.**
After the bug fix (refreshing each hub sequence at most once), total
recomputations stayed below one full rebuild (76.1% work saved). This is the
empirical claim the paper needs to support.

---

## 7. GPU acceleration

### 7.1 Available backends

The code auto-detects and uses the fastest available backend:

| Backend | Requires | DTW (11,707 × 20,000) | Speedup |
|---|---|---|---|
| Pure Python | nothing | ~5.8 hours | 1× |
| Numba JIT | `pip install numba` | ~7 minutes | ~50× |
| CuPy CUDA | CuPy + GPU | ~26 seconds | **~835×** |
| Jaccard (sparse CPU) | scipy | ~6 seconds | — |

The CUDA kernel uses heap-allocated DP tables with no fixed sequence-length
limit. Memory is managed automatically: block size is computed from actual GPU
VRAM before the main loop so the scratch buffer never exceeds 50% of device
memory.

### 7.2 Tested hardware

| GPU | VRAM | DTW time | Job |
|---|---|---|---|
| Tesla V100S-PCIE | 32 GB | ~26 s (fell back to Python — bug since fixed) | 3094241 |
| NVIDIA A100-SXM4 | 80 GB | **26 seconds** | 3107138 |
| Tesla V100S-PCIE | 32 GB | ~2.5 min (next_visit, visit-level DTW) | 3107140 |

### 7.3 Installation

```bash
# Check your CUDA version
nvcc --version

# Install matching CuPy
pip install cupy-cuda12x    # CUDA 12.x
pip install cupy-cuda11x    # CUDA 11.x

# Numba as CPU-only JIT fallback
pip install numba
```

### 7.4 Running GPU jobs

```bash
# GPU script (auto-detects CuPy > Numba > pure Python)
sbatch scripts/05_run_gpu.sh within_visit dtw_tag    25
sbatch scripts/05_run_gpu.sh next_visit   dtw_jaccard 25
sbatch scripts/05_run_gpu.sh within_visit jaccard    25   # GPU not used for sparse

# With DTW window constraint (faster, same quality)
sbatch scripts/05_run_gpu.sh within_visit dtw_tag 25 3 similarity 5
```

Note: hierarchical update is disabled by default in `05_run_gpu.sh` for DTW
runs. DTW-based hierarchical update processes each hub neighbourhood
sequentially (~4h per batch of 500 at Python speed), making it prohibitively
slow for routine use. Sparse-matrix hierarchical update remains fast and is
enabled by default for Jaccard/cosine/etc.

---

## 8. What changed from the original and why

### 8.1 The encoding bug (primary cause of near-zero accuracy)

The original assigned each ICD-9 code an integer by order of first appearance
in the CSV (`D_250`→1, `D_337`→2, …) and computed DTW with local cost
`|index_a − index_b|`. This made the distance between two diagnoses a function
of where they happened to appear in the file, not of their clinical
relationship. `D_250` and `D_337` scored as near-identical; `D_250` and `D_486`
scored 23 apart. The neighbour graph was essentially noise.

**Fix:** integers are now only column positions in a sparse binary matrix.
Distance is computed on set overlap; the index values never enter arithmetic.

### 8.2 Position-based imputation

The original looked up the neighbour's code at the exact masked position:
mask at position 7 → look up `neighbour[7]`. Diagnosis order inside a MIMIC
admission is `SEQ_NUM`, a billing artefact — position 7 in one record has no
clinical correspondence to position 7 in another. Neighbours shorter than the
mask position were silently skipped, dropping their votes.

**Fix:** every code a neighbour carries is a candidate, weighted by similarity.
Codes already observed in the query are excluded. Output is a ranked list.

### 8.3 Argmax accuracy on a heavy-tailed 888-way problem

Exact-match accuracy on a problem where 90% of queries involve the 10 most
common codes is nearly uninformative. It also inflated the apparent performance
of the frequency prior.

**Fix:** ranked evaluation — hits@{1,5,10,20} and MRR — plus stratification
by code frequency so Q1 (rare) and Q4 (common) performance are visible
separately.

### 8.4 DTW applied to an unordered set

DTW was applied to the code list within a single admission. That list is an
unordered billing set with no time axis — there is nothing to warp.

**Fix:** two DTW modes. Tag-level DTW for within-visit (codes ordered by
`SEQ_NUM`, limited signal). Visit-level DTW for next-visit (visits ordered by
`ADMITTIME`, genuine time axis).

### 8.5 Patient trajectories were never used

The original notebook computed `patient_admissions` groupings but never used
them downstream. The trajectory task had no path to the data.

**Fix:** `prepare_data.py` writes `patient_trajectories.pkl` — a dict of
`{subject_id: [(hadm_id, admittime, [codes]), ...]}` sorted by admission time.
`build_labels.py --task next_visit` consumes it.

### 8.6 Hierarchical update accumulated unlimited recomputations

The original implementation refreshed hub neighbourhoods on every batch after
they crossed the threshold, so a popular hub was recomputed 24 times (once per
batch). Total recomputations on MIMIC-III: **94,474** — more than double a
single full rebuild of 46,831 sequences.

**Fix:** each sequence is refreshed at most once (the first time it crosses the
threshold). Total recomputations after fix: 11,191 (76.1% below a full
rebuild).

---

## 9. Interpreting results

### Result files

Each run writes `results/<tag>.json` containing:

```json
{
  "config": { ... full configuration ... },
  "results": {
    "knn_jaccard": { "hits@1": ..., "hits@10": ..., "by_frequency_stratum": { ... } },
    "frequency_prior": { ... },
    "cooccurrence": { ... }
  },
  "hierarchical": { "total_recomputations": ..., "beats_full_rebuild": true, ... }
}
```

Collect a sweep into one CSV:

```bash
python scratch/collect_results.py --pattern "sweep_sim_within_visit_*"
```

### What to check first

**Check vocab size in the log.** `vocab=888` is correct. `vocab=2583` means
`min_code_frequency` was not set and rare codes were not filtered — re-run
`prepare_data.py` with `--min_code_frequency 5`.

**Read `by_frequency_stratum` before the headline.** Q4 (common codes) will
always dominate and look best. A model that beats the baseline only on Q4 has
learned which diagnoses are common in an ICU — not a contribution. The story
is in Q1 and Q2.

**Check `beats_full_rebuild` in the hierarchical block.** A value of `true`
means the tiered update saved work. `false` means it cost more than a full
rebuild and the threshold should be raised.

**For next_visit, prefer `any_hit@k` over `recall@k` for the lead metric.**
With a mean target size of 13 codes, recall@10 is bounded at ~0.77. `any_hit@10`
(was at least one code predicted?) is the more interpretable clinical claim.

---

## 10. Known limitations and next steps

### Limitations

**MIMIC-III is primarily single-admission.** 84.1% of patients have exactly
one admission, leaving only 7,342 patients (15.9%) for the next-visit task.
This limits the cohort for trajectory experiments and motivates the inclusion
of eICU and SEER (which have richer longitudinal follow-up) as planned.

**Within-visit "sequence" is not a clinical sequence.** `SEQ_NUM` in MIMIC-III
is a billing priority, not a temporal ordering. DTW on within-visit codes
therefore cannot exploit genuine temporal structure. The paper should state
this explicitly.

**DTW hierarchical update is too slow for routine use.** With a pure-Python
or even Numba DTW kernel, refreshing hub neighbourhoods takes ~4h per batch.
Either approximate nearest-neighbour indexing (e.g. FAISS) or caching the
DTW distance matrix between reference sequences is needed.

**Rare codes (Q1, Q2) remain poorly predicted.** kNN beats the frequency prior
on Q4 but not on Q1. Co-occurrence beats kNN on Q1. This gap motivates the
generative direction — synthesising rare-code examples rather than retrieving
similar patients.

### Immediate next steps

- [ ] Run `jaccard` on `next_visit` — direct comparison with `dtw_jaccard` on the
      same task; if DTW wins there, the paper's core argument is complete
- [ ] Run the ICD-4 granularity sweep properly — re-run `prepare_data.py` with
      `--icd_digits 4` before the experiment (not relying on cached data)
- [ ] Implement rank fusion of kNN and co-occurrence — co-occurrence wins on rare
      codes, kNN wins on common; combining them likely beats both
- [ ] Add eICU and SEER to increase the multi-visit cohort
- [ ] Port algorithm blocks to Overleaf — hierarchical update first (the novel
      algorithmic piece), then the imputation formula

### Future directions (from project meetings)

- Generative model for realistic synthetic EHR trajectories with clinical
  constraints (patient predisposition, comorbidity patterns)
- Study of which EHR fields generative AI can reconstruct with highest
  confidence vs. which remain hard
- Application to SNP-paired datasets (UK Biobank, All of Us) once the ICD-9
  trajectory method is validated
