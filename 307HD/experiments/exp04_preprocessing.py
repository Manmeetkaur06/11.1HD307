"""Experiment 4: encoding and training-weight ablations on the grouped protocol (seeds 0-29).

C_dedup  : one development row per predictor pattern (test rows untouched)
C_onehot : one-hot cp, restecg, slope, ca, thal; standardise the 5 continuous inputs;
           sex, fbs, exang pass through (27 columns); Gaussian NB keeps 13 scaled inputs
D        : both changes
All three use the same grouped outer split and grouped inner folds as Protocol C.

Outputs (results/04_preprocessing/): metrics.csv, summary.csv
"""
import pandas as pd

from common import Timer, fresh_dir, load, map_seeds, save_npz

from src.evaluation import ANCHOR_SEED, SEEDS, score_run, summarise
from src.protocols import ARMS, run_arm


def one_seed(seed, config, X, y, groups, out):
    rows = []
    for name in ("C_dedup", "C_onehot", "D"):
        run = run_arm(ARMS[name], config, X, y, groups, seed)
        save_npz(out / "runs" / f"seed{seed}_{name}.npz", run)
        rows += score_run(run, groups)
    print(f"  seed {seed} done", flush=True)
    return rows


def main():
    config, X, y, groups, _ = load()
    out = fresh_dir("04_preprocessing")
    (out / "runs").mkdir()
    with Timer("exp04 preprocessing ablations"):
        res = map_seeds(one_seed, SEEDS + [ANCHOR_SEED], config, X, y, groups, out)
    metrics = pd.DataFrame([r for rows in res for r in rows])
    metrics.to_csv(out / "metrics.csv", index=False)
    rep = metrics[metrics.seed.isin(SEEDS)]
    summary = summarise(rep, ["arm", "weighting", "model"])
    summary.to_csv(out / "summary.csv", index=False)
    print(summary[(summary.model == "Stacking") & (summary.weighting == "row")]
          [["arm", "accuracy_mean", "recall_mean", "f1_mean", "roc_auc_mean"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
