"""Evaluation protocols, representations and stacking used by every experiment.

Protocols
---------
A        row outer split (80/20, unstratified)  + stratified row inner folds   (reproduction)
B        grouped outer split (20% of patterns)   + stratified row inner folds
C        grouped outer split (same as B)         + StratifiedGroupKFold inner folds
C_dedup  C, keeping one development row per predictor pattern
C_onehot C, one-hot encoding of the five multi-level categorical predictors
D        C_dedup + C_onehot

A "group" is an exact 13-predictor pattern. The target is never part of the key.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import (GroupShuffleSplit, StratifiedGroupKFold,
                                     StratifiedKFold, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .reproduction import MODEL_ORDER, build_meta, build_models, positive_probability

CONTINUOUS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
BINARY = ["sex", "fbs", "exang"]
CATEGORICAL = {"cp": [0, 1, 2, 3], "restecg": [0, 1, 2], "slope": [0, 1, 2],
               "ca": [0, 1, 2, 3, 4], "thal": [0, 1, 2, 3]}  # audited category sets


@dataclass(frozen=True)
class Arm:
    name: str
    outer: str          # "row" | "group"
    inner: str          # "row" | "group"
    encoding: str = "numeric"   # "numeric" | "onehot"
    dedup: bool = False


ARMS = {
    "A": Arm("A", "row", "row"),
    "B": Arm("B", "group", "row"),
    "C": Arm("C", "group", "group"),
    "C_dedup": Arm("C_dedup", "group", "group", dedup=True),
    "C_onehot": Arm("C_onehot", "group", "group", encoding="onehot"),
    "D": Arm("D", "group", "group", encoding="onehot", dedup=True),
}


# ----------------------------------------------------------------- groups
def pattern_groups(X: pd.DataFrame) -> np.ndarray:
    """Integer id per exact predictor pattern, numbered by first appearance in the CSV.

    GroupShuffleSplit permutes the unique labels, so the numbering fixes the partitions;
    first-appearance numbering is used throughout this project.
    """
    keys = pd.Series(list(X.itertuples(index=False, name=None)))
    return pd.factorize(keys)[0]


def first_per_group(positions: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Keep the first listed position of every group (order preserved)."""
    _, first = np.unique(groups[positions], return_index=True)
    return positions[np.sort(first)]


# ----------------------------------------------------------------- splits
def outer_split(kind: str, n: int, groups: np.ndarray, seed: int, test_size: float = 0.2):
    idx = np.arange(n)
    if kind == "row":
        return train_test_split(idx, test_size=test_size, random_state=seed, shuffle=True)
    dev, test = next(GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed).split(idx, groups=groups))
    assert not set(groups[dev]) & set(groups[test]), "group leak across outer split"
    return dev, test


def inner_folds(kind: str, y: np.ndarray, groups: np.ndarray, seed: int, k: int = 5):
    dummy = np.zeros(len(y))
    if kind == "row":
        folds = list(StratifiedKFold(k, shuffle=True, random_state=seed).split(dummy, y))
    else:
        folds = list(StratifiedGroupKFold(k, shuffle=True, random_state=seed).split(dummy, y, groups))
    seen = np.zeros(len(y), dtype=int)
    for fit, val in folds:
        assert not set(fit) & set(val), "row overlap across inner folds"
        assert set(np.unique(y[fit])) == {0, 1} and set(np.unique(y[val])) == {0, 1}, "both classes required in every fold"
        if kind == "group":
            assert not set(groups[fit]) & set(groups[val]), "group leak across inner folds"
        seen[val] += 1
    assert (seen == 1).all(), "each row must be validated exactly once"
    return folds


# ----------------------------------------------------------------- models
def onehot_preprocessor():
    """27 columns in a fixed order: 5 scaled continuous, 19 one-hot, 3 binary."""
    return ColumnTransformer([
        ("continuous", StandardScaler(), CONTINUOUS),
        ("categorical", OneHotEncoder(categories=list(CATEGORICAL.values()), handle_unknown="ignore",
                                      sparse_output=False), list(CATEGORICAL)),
        ("binary", "passthrough", BINARY),
    ], verbose_feature_names_out=False)


def make_models(config: dict, seed: int, encoding: str = "numeric") -> dict:
    """The six fixed base learners. One-hot replaces the scaler for every learner except
    Gaussian NB, which keeps the 13 standardised numeric predictors."""
    models = build_models(dict(config, seed=seed))
    if encoding == "numeric":
        return models
    out = {}
    for name, pipe in models.items():
        if name == "NB":
            out[name] = pipe
        else:
            out[name] = Pipeline([("encode", onehot_preprocessor()), ("model", pipe.named_steps["model"])])
    return out


def base_predictions(models: dict, X_fit, y_fit, X_pred) -> np.ndarray:
    return np.column_stack([positive_probability(clone(models[n]).fit(X_fit, y_fit), X_pred)
                            for n in MODEL_ORDER])


def oof_matrix(models: dict, X, y, folds) -> np.ndarray:
    Z = np.full((len(X), len(MODEL_ORDER)), np.nan)
    seen = np.zeros(len(X), dtype=int)
    for fit, val in folds:
        seen[val] += 1
        for j, n in enumerate(MODEL_ORDER):
            Z[val, j] = positive_probability(clone(models[n]).fit(X.iloc[fit], y.iloc[fit]), X.iloc[val])
    assert (seen == 1).all() and np.isfinite(Z).all(), "each development row needs exactly one OOF vector"
    return Z


def fit_stack(config, models, X_dev, y_dev, g_dev, inner, seed, k=5, X_pred=None):
    """Cross-fit OOF meta-features, fit the LR meta-learner, refit base learners on all
    development rows and (optionally) predict X_pred."""
    folds = inner_folds(inner, y_dev.to_numpy(), g_dev, seed, k)
    Z_dev = oof_matrix(models, X_dev, y_dev, folds)
    meta = build_meta(dict(config, seed=seed)).fit(Z_dev, y_dev)
    out = {"Z_dev": Z_dev, "meta": meta}
    if X_pred is not None:
        Z_pred = base_predictions(models, X_dev, y_dev, X_pred)
        out.update(Z_pred=Z_pred, p_stack=positive_probability(meta, Z_pred))
    return out


def run_arm(arm: Arm, config, X, y, groups, seed, k=5, base_cache=None):
    """One outer evaluation of one arm. Returns predictions for every model and the stack."""
    dev, test = outer_split(arm.outer, len(X), groups, seed)
    dev_used = first_per_group(dev, groups) if arm.dedup else dev
    models = make_models(config, seed, arm.encoding)
    X_dev, y_dev, X_test = X.iloc[dev_used], y.iloc[dev_used], X.iloc[test]
    key = (seed, arm.outer, arm.encoding, arm.dedup)
    if base_cache is not None and key in base_cache:
        Z_test = base_cache[key]
    else:
        Z_test = base_predictions(models, X_dev, y_dev, X_test)
        if base_cache is not None:
            base_cache[key] = Z_test
    folds = inner_folds(arm.inner, y_dev.to_numpy(), groups[dev_used], seed, k)
    Z_dev = oof_matrix(models, X_dev, y_dev, folds)
    meta = build_meta(dict(config, seed=seed)).fit(Z_dev, y_dev)
    return {"arm": arm.name, "seed": seed, "dev": dev_used, "test": test,
            "y_dev": y_dev.to_numpy(), "y_test": y.iloc[test].to_numpy(),
            "Z_dev": Z_dev, "Z_test": Z_test, "p_stack": positive_probability(meta, Z_test),
            "meta_coef": meta.coef_[0].copy(), "meta_intercept": float(meta.intercept_[0])}
