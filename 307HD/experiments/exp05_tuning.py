"""Experiment 5: development-only model, hyperparameter and threshold selection (seeds 0-29).

Uses the arm-D representation (one row per development pattern, one-hot encoding) and the
same grouped outer split. Four families (LR, RBF-SVM, RF, XGBoost) x 16 configurations are
scored with three grouped development folds. The configuration with the highest mean fold
accuracy at cut-off 0.5 is selected (ties: mean fold AUC, then fixed family/candidate order).
A decision threshold is then chosen from 0.10-0.90 (step 0.02) by cross-fitting the selected
configuration on the same folds (ties: F1, closeness to 0.5, lower cut-off). The selected
model is refitted on all development rows and evaluated once on the locked test rows at 0.5
and at the selected threshold. No outer-test information is used for any choice.

Outputs (results/05_tuning/): metrics.csv, selections.csv, candidates.csv, summary.csv
"""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import ParameterGrid, ParameterSampler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from xgboost import XGBClassifier

from common import Timer, fresh_dir, load, map_seeds

from src.evaluation import ANCHOR_SEED, SEEDS, pattern_positions, summarise
from src.protocols import first_per_group, inner_folds, onehot_preprocessor, outer_split
from src.reproduction import evaluate

N_CANDIDATES = 16
THRESHOLDS = np.round(np.arange(0.10, 0.9001, 0.02), 2)
SPACES = {
    "LR": {"C": [0.001, 0.01, 0.03, 0.1, 0.3, 1, 3, 10], "class_weight": [None, "balanced"]},
    "SVM": {"C": [0.01, 0.1, 0.3, 1, 3, 10, 30, 100], "gamma": ["scale", 0.01, 0.03, 0.1, 0.3, 1],
            "class_weight": [None, "balanced"]},
    "RF": {"n_estimators": [100, 250], "max_depth": [3, 5, 8, None], "min_samples_leaf": [1, 2, 4, 8],
           "max_features": ["sqrt", 0.5, 1.0], "class_weight": [None, "balanced"]},
    "XGB": {"n_estimators": [75, 150, 300], "max_depth": [1, 2, 3, 4], "learning_rate": [0.01, 0.03, 0.1, 0.2],
            "min_child_weight": [1, 3, 5, 10], "subsample": [0.7, 1.0], "colsample_bytree": [0.7, 1.0],
            "reg_lambda": [1, 5, 10], "reg_alpha": [0, 0.1, 1]},
}


def estimator(family, params, seed):
    if family == "LR":
        m = LogisticRegression(solver="lbfgs", max_iter=4000, random_state=seed, **params)
    elif family == "SVM":
        m = SVC(kernel="rbf", random_state=seed, **params)
    elif family == "RF":
        m = RandomForestClassifier(random_state=seed, n_jobs=1, **params)
    else:
        m = XGBClassifier(objective="binary:logistic", eval_metric="logloss", tree_method="hist",
                          device="cpu", n_jobs=1, verbosity=0, random_state=seed, **params)
    return Pipeline([("encode", onehot_preprocessor()), ("model", m)])


def score(model, X):
    """Class-1 score in [0, 1]. SVM margins are mapped with a logistic function (not a
    calibrated probability; the 0.5 cut-off equals the SVM decision boundary)."""
    if isinstance(model.named_steps["model"], SVC):
        return 1.0 / (1.0 + np.exp(-model.decision_function(X)))
    return model.predict_proba(X)[:, list(model.classes_).index(1)]


def candidates(seed):
    out = []
    for family, space in SPACES.items():
        grid = list(ParameterGrid(space))
        params = grid if len(grid) == N_CANDIDATES else list(
            ParameterSampler(space, n_iter=N_CANDIDATES, random_state=seed))
        out += [(family, i, p) for i, p in enumerate(params)]
    return out


def crossfit(family, params, seed, X, y, folds):
    oof, accs, aucs = np.full(len(y), np.nan), [], []
    for fit, val in folds:
        m = estimator(family, params, seed).fit(X.iloc[fit], y.iloc[fit])
        s = score(m, X.iloc[val])
        oof[val] = s
        accs.append(accuracy_score(y.iloc[val], (s > 0.5).astype(int)))
        aucs.append(roc_auc_score(y.iloc[val], s))
    return oof, float(np.mean(accs)), float(np.mean(aucs))


def one_seed(seed, config, X, y, groups, out):
    dev, test = outer_split("group", len(X), groups, seed)
    dev = first_per_group(dev, groups)
    X_dev, y_dev = X.iloc[dev], y.iloc[dev]
    folds = inner_folds("group", y_dev.to_numpy(), groups[dev], seed, k=3)
    cand_rows = []
    for order, (family, idx, params) in enumerate(candidates(seed)):
        _, acc, auc = crossfit(family, params, seed, X_dev, y_dev, folds)
        cand_rows.append({"seed": seed, "order": order, "family": family, "candidate": idx,
                          "params": repr(params), "cv_accuracy": acc, "cv_auc": auc, "_p": params})
    best = sorted(cand_rows, key=lambda r: (-r["cv_accuracy"], -r["cv_auc"], r["order"]))[0]
    oof, _, _ = crossfit(best["family"], best["_p"], seed, X_dev, y_dev, folds)
    yv = y_dev.to_numpy()
    ranked = sorted(THRESHOLDS, key=lambda t: (-accuracy_score(yv, (oof > t).astype(int)),
                                                -f1_score(yv, (oof > t).astype(int), zero_division=0),
                                                abs(t - 0.5), t))
    threshold = float(ranked[0])
    final = estimator(best["family"], best["_p"], seed).fit(X_dev, y_dev)
    s_test, y_test = score(final, X.iloc[test]), y.iloc[test].to_numpy()
    np.savez_compressed(out / "runs" / f"seed{seed}.npz", dev=dev, test=test,
                        y_test=y_test, score=s_test, threshold=threshold)
    pat = pattern_positions(test, groups)
    rows = []
    for arm, thr in (("Selected_0.5", 0.5), ("Selected_cutoff", threshold)):
        for weighting, sel in (("row", slice(None)), ("pattern", pat)):
            m, _ = evaluate(y_test[sel], s_test[sel], thr)
            rows.append({"seed": seed, "arm": arm, "model": arm, "weighting": weighting, **m})
    selection = {"seed": seed, "family": best["family"], "params": best["params"],
                 "cv_accuracy": best["cv_accuracy"], "cv_auc": best["cv_auc"], "threshold": threshold}
    for r in cand_rows:
        r.pop("_p")
    print(f"  seed {seed}: {best['family']} thr={threshold:.2f}", flush=True)
    return rows, selection, cand_rows


def main():
    config, X, y, groups, _ = load()
    out = fresh_dir("05_tuning")
    (out / "runs").mkdir()
    with Timer("exp05 tuning"):
        res = map_seeds(one_seed, SEEDS + [ANCHOR_SEED], config, X, y, groups, out)
    metrics = pd.DataFrame([r for rows, _, _ in res for r in rows])
    metrics.to_csv(out / "metrics.csv", index=False)
    pd.DataFrame([s for _, s, _ in res]).to_csv(out / "selections.csv", index=False)
    pd.DataFrame([c for _, _, cands in res for c in cands]).to_csv(out / "candidates.csv", index=False)
    rep = metrics[metrics.seed.isin(SEEDS)]
    summary = summarise(rep, ["arm", "weighting"])
    summary.to_csv(out / "summary.csv", index=False)
    sel = pd.DataFrame([s for _, s, _ in res])
    print(sel[sel.seed.isin(SEEDS)].family.value_counts().to_dict())
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
