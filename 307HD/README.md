# Duplicate-Aware Stacking for Heart Disease Prediction

**Manmeet Kaur** · SIT307 Machine Learning, Deakin University · 11.1HD Machine Learning Research Task

This package reconstructs the stacking ensemble reported by Bhagat *et al.* (2025) and evaluates a
duplicate-aware alternative. The proposed protocol keeps identical predictor patterns on one side of both the
outer evaluation split and the inner folds used to train the meta-learner.

The repository contains the dataset, training code, notebooks, generated analysis assets and execution logs.
Saved result CSVs and prediction arrays are bundled in `results_snapshot.zip`; restore them with
`python restore_saved_results.py` to inspect the measured analysis without retraining.

**Uploading this folder:** follow [GITHUB_UPLOAD.md](GITHUB_UPLOAD.md).

## Main measured findings

| Result | Value |
| --- | --- |
| Published stacking accuracy (Bhagat *et al.*) | 98.53% |
| Seed-42 row-split reconstruction (Protocol A) | 98.54% |
| Row-split test rows with an exact training-pattern twin | 97.1% mean over 30 seeds |
| Stack on unseen patterns, row-fold meta-training (B) | 78.6% mean accuracy |
| Stack on unseen patterns, grouped meta-training (C) | 81.2% mean accuracy; +2.61 pp, 21/30 wins |
| Soft vote on the same grouped test rows | 80.2% mean accuracy |
| Accuracy MAE: row CV → grouped CV | 16.87 pp → 4.98 pp |
| Arm D stack; arm D logistic regression | 83.5%; 84.5% mean accuracy |

The 1,025-row Kaggle file contains 302 distinct predictor patterns, each repeated three to eight times. The duplicate
audit, LR's leading leakage-free AUC and limited tuning benefit are consistency checks against Kilim (2026,
doi:10.54569/aair.1998842); the new analysis is
stacking-specific grouping of the inner OOF folds and the combiner control. The analysis therefore distinguishes row-weighted results from equal-pattern results and avoids treating
row-split accuracy as a generalisation estimate for unseen patterns.

## Repository layout

```
SIT307_11.1HD/
├── run_all.py                    # runs experiments 1–8 and rebuilds analysis assets
├── restore_saved_results.py      # restores the supplied result files
├── results_snapshot.zip          # compact saved outputs for browser upload
├── GITHUB_UPLOAD.md              # upload instructions
├── verify_results.py             # checks saved predictions and headline metrics
├── run_experiment.py             # dataset audit/download utilities and Part 1 run
├── requirements.txt              # pinned reconstruction versions
├── config/reproduction_v1.json   # fixed model settings and assumptions
├── data/                         # pinned CSV and provenance record
├── src/                          # audit, evaluation, reproduction and protocols
├── experiments/                  # numbered experiment scripts
├── notebooks/
│   ├── SIT307_HD_Colab.ipynb     # full fresh run for Google Colab
│   └── SIT307_HD_Verification.ipynb # executed saved-output verification
├── results/                      # created by restoration or a training run
├── verification/                 # clean-run record, stage logs and verification summary
└── paper/                        # generated figures/tables only; no report included
```

## Installation

Run commands from the repository root, the folder containing `run_all.py`. Python 3.12 is recommended.
With conda (macOS, Windows or Linux):

```bash
conda create -n sit307hd python=3.12 -y
conda activate sit307hd
python -m pip install -r requirements.txt
```

On Google Colab, open `notebooks/SIT307_HD_Colab.ipynb`, install `requirements.txt`, and run all cells. A CPU
runtime is sufficient, although a fresh full run takes longer or shorter depending on the runtime.

## Reproducing and checking the evidence

```bash
python restore_saved_results.py   # unpack the supplied measured outputs
python run_experiment.py audit     # hash, schema, missingness and duplicate audit
python verify_results.py           # quick check of the packaged saved evidence
python run_all.py                  # fresh experiments 1–8 and regenerated analysis assets
```

A full training run does not require restoring the snapshot first. The restore command verifies its checksum
and refuses to overwrite different existing results. The generated `results/` folder is ignored by Git; the
supplied snapshot remains available in the repository.

Useful options are `python run_all.py --only 2 8` and `python run_all.py --from 5`. The default is one worker;
`SIT307_JOBS` can be set to a positive value for a parallel run. Fits are seeded and single-threaded, but bitwise
identity across different software builds or hardware is not assumed. Each run records its environment, stage logs and
source hashes under `verification/runs/`.

| Experiment | Output folder | Generated output |
| --- | --- | --- |
| 1 Reproduction (seed 42) | `results/01_reproduction/` | published comparison and discrepancy audit |
| 2 Protocols A/B/C | `results/02_protocols/` | grouped-versus-row results and ROC/mechanism inputs |
| 3 Nested validation | `results/03_nested_validation/` | validation reliability table and figure |
| 4 Preprocessing ablations | `results/04_preprocessing/` | one-hot/deduplication results |
| 5 Tuning study | `results/05_tuning/` | development-only selected rows |
| 6 Mechanism experiments | `results/06_mechanism/` | overlap, combiner and fold-count analyses |
| 7 Statistical checks | `results/07_statistics/` | corrected tests, bootstrap and McNemar checks |
| 8 Analysis assets | `paper/generated/`, `paper/figs/` | generated measured tables, figures and macros |

After restoration or training, the saved prediction arrays are under the `runs/` folders for experiments 2–5. The executed verification notebook
loads these outputs and prints the data audit, anchor metrics, grouped comparison and evidence-file checks.
All reported experimental results were produced by a full serial `python run_all.py` from a fresh source copy with
no cached outputs. All eight stages passed. The 26 measured CSV files, ten generated table bodies and numeric
macros match the reported values; only elapsed fitting times were excluded from the comparison. The run record,
clean-start manifest and `verification/fresh_run_comparison.json` document these checks. Historical execution logs
are retained alongside the successful fresh run. Holm adjustment covers all four reported D contrasts, including
D stack minus D LR, separately for each metric and weighting.


## Dataset

`data/heart.csv` is version 2 of the Kaggle resource cited by Bhagat *et al.*:
<https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset>. It contains 1,025 rows, 13 predictors and a
binary target. SHA-256: `ddb2996b2f4db2e00aad13f4518200179ff69f79093838e3c21ffa672ebec0f1`. The published paper does not
state its dataset version, so this version is pinned as the reconstruction snapshot. To re-download it, run
`python run_experiment.py download`.

