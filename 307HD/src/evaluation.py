"""Metrics (row- and pattern-weighted), paired statistics and small I/O helpers."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .reproduction import MODEL_ORDER, evaluate

METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]
ALL_MODELS = MODEL_ORDER + ["Stacking"]
SEEDS = list(range(30))
ANCHOR_SEED = 42


def pattern_positions(test: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Positions (within the test vector) of the first row of each test pattern.
    Rows of one pattern share predictors and label, so this gives one vote per pattern."""
    _, first = np.unique(groups[test], return_index=True)
    return np.sort(first)


def score_run(run: dict, groups: np.ndarray, threshold: float = 0.5) -> list[dict]:
    """Row- and pattern-weighted metrics for the six base learners and the stack."""
    probs = {n: run["Z_test"][:, j] for j, n in enumerate(MODEL_ORDER)}
    probs["Stacking"] = run["p_stack"]
    pat = pattern_positions(run["test"], groups)
    rows = []
    for weighting, sel in (("row", slice(None)), ("pattern", pat)):
        for name, p in probs.items():
            m, _ = evaluate(run["y_test"][sel], p[sel], threshold)
            rows.append({"seed": run["seed"], "arm": run["arm"], "model": name, "weighting": weighting, **m})
    return rows


def summarise(df: pd.DataFrame, by: list[str], metrics=METRICS) -> pd.DataFrame:
    g = df.groupby(by, sort=False)[metrics]
    out = g.mean().add_suffix("_mean").join(g.std(ddof=1).add_suffix("_sd"))
    return out.reset_index()


def paired_diff(df, a: dict, b: dict, metric: str, seeds=SEEDS) -> np.ndarray:
    """(a - b) per seed in percentage points; a/b are column filters."""
    def pick(f):
        sub = df
        for k, v in f.items():
            sub = sub[sub[k] == v]
        return sub.set_index("seed")[metric].loc[seeds].to_numpy()
    return (pick(a) - pick(b)) * 100


def corrected_t(d, ratio: float):
    """Nadeau-Bengio corrected resampled t-test, two-sided, J-1 dof."""
    d = np.asarray(d, float)
    J, var = len(d), d.var(ddof=1)
    if var == 0:
        return float("nan"), float("nan")
    t = d.mean() / np.sqrt((1 / J + ratio) * var)
    return float(t), float(2 * stats.t.sf(abs(t), J - 1))


def holm(pvalues):
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj


def wilcoxon_p(d):
    d = np.asarray(d, float)
    return float("nan") if np.allclose(d, 0) else float(stats.wilcoxon(d).pvalue)


def win_tie_loss(d, tol=1e-9):
    d = np.asarray(d, float)
    return int((d > tol).sum()), int((np.abs(d) <= tol).sum()), int((d < -tol).sum())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=float) + "\n")
