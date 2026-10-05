"""Experiment 3: does development CV predict performance on unseen patterns? (seeds 0-9)

For each seed the Protocol-C outer test set (20% of groups) is the final holdout. On the
development groups, row CV and grouped CV (5 folds each) evaluate the same grouped-inner
stack: every CV-training subset builds its own grouped OOF matrix, LR meta-learner and
base refits, then predicts its validation fold. Pooled validation predictions give the CV
estimate; the holdout reference is the Protocol-C stack trained on all development rows
(loaded from experiment 2).

Outputs (results/03_nested_validation/): estimates.csv, summary.csv
"""
import numpy as np
import pandas as pd

from common import RESULTS, Timer, fresh_dir, load, map_seeds

from src.evaluation import METRICS
from src.protocols import fit_stack, inner_folds, make_models, outer_split
from src.reproduction import evaluate

NESTED_SEEDS = list(range(10))


def one_seed(seed, config, X, y, groups, out):
    dev, test = outer_split("group", len(X), groups, seed)
    X_dev, y_dev, g_dev = X.iloc[dev], y.iloc[dev], groups[dev]
    models = make_models(config, seed)
    rows = []
    for scheme in ("row", "group"):
        pooled = np.full(len(dev), np.nan)
        for fit, val in inner_folds(scheme, y_dev.to_numpy(), g_dev, seed, k=5):
            res = fit_stack(config, models, X_dev.iloc[fit], y_dev.iloc[fit], g_dev[fit], "group", seed,
                            X_pred=X_dev.iloc[val])
            pooled[val] = res["p_stack"]
        assert np.isfinite(pooled).all()
        np.savez_compressed(out / "runs" / f"seed{seed}_{scheme}.npz",
                            dev=dev, test=test, y_dev=y_dev.to_numpy(), scores=pooled)
        m, _ = evaluate(y_dev.to_numpy(), pooled, 0.5)
        rows.append({"seed": seed, "estimate": f"{scheme}_cv", **{k: m[k] for k in METRICS}})
    hold = np.load(RESULTS / "02_protocols" / "runs" / f"seed{seed}_C.npz")
    assert np.array_equal(hold["test"], test)
    m, _ = evaluate(hold["y_test"], hold["p_stack"], 0.5)
    rows.append({"seed": seed, "estimate": "holdout", **{k: m[k] for k in METRICS}})
    print(f"  seed {seed} done", flush=True)
    return rows


def main():
    config, X, y, groups, _ = load()
    out = fresh_dir("03_nested_validation")
    (out / "runs").mkdir()
    with Timer("exp03 nested validation"):
        res = map_seeds(one_seed, NESTED_SEEDS, config, X, y, groups, out)
    est = pd.DataFrame([r for rows in res for r in rows])
    est.to_csv(out / "estimates.csv", index=False)
    wide = est.pivot(index="seed", columns="estimate")
    summary = []
    for m in METRICS:
        row, grp, hold = wide[(m, "row_cv")], wide[(m, "group_cv")], wide[(m, "holdout")]
        summary.append({"metric": m, "row_cv_mean": row.mean(), "group_cv_mean": grp.mean(),
                        "holdout_mean": hold.mean(), "row_cv_mae": (row - hold).abs().mean(),
                        "group_cv_mae": (grp - hold).abs().mean(),
                        "row_cv_bias": (row - hold).mean(), "group_cv_bias": (grp - hold).mean(),
                        "group_closer": int(((grp - hold).abs() < (row - hold).abs()).sum()),
                        "n_seeds": len(hold)})
    pd.DataFrame(summary).to_csv(out / "summary.csv", index=False)
    print(pd.DataFrame(summary).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
