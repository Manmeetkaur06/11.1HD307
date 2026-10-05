"""Part 1: six classifiers and an explicit five-fold out-of-fold stacking reconstruction
of Bhagat et al. (2025) on one shuffled 80/20 row split.

Nothing here is tuned against the test set. Every numerical result written by this module
comes from the pinned dataset. The model definitions are reused by src/protocols.py.
"""
from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

from .data_audit import audit_dataset, load_config, sha256, utc_now, write_json

MODEL_ORDER = ["LR", "DT", "RF", "XGB", "NB", "KNN"]
METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]

# Transcribed published reference values, never treated as measured project results.
# Source: Bhagat et al., Table 11, supplied PDF page 18 (journal page 36368).
PUBLISHED = {
    "LR": [0.8439, 0.8196, 0.9090, 0.8620, 0.9204],
    "DT": [0.9268, 0.9702, 0.8909, 0.9289, 0.9702],
    "RF": [0.9268, 0.9130, 0.9545, 0.9333, 0.9734],
    "XGB": [0.9073, 0.9174, 0.9090, 0.9132, 0.9830],
    "NB": [0.8439, 0.8250, 0.9000, 0.8608, 0.9185],
    "KNN": [0.8585, 0.8648, 0.8727, 0.8687, 0.9304],
    "Stacking": [0.9853, 1.0000, 0.9727, 0.9861, 0.9880],
}
ALTERNATIVE_REPORTS = [
    ("XGB", "accuracy", 0.9317, "Introduction, PDF p. 3"),
    ("Stacking", "accuracy", 0.9658, "Introduction, PDF p. 3"),
    ("XGB", "roc_auc", 0.9652153110047846, "Figure 4(f) legend, PDF p. 20"),
]


def build_models(config):
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError("XGBoost is required; install requirements.txt. No substitute model is used.") from exc
    seed = config["seed"]
    settings = config["models"]
    classifiers = {
        "LR": LogisticRegression(**settings["LR"], random_state=seed),
        "DT": DecisionTreeClassifier(**settings["DT"], random_state=seed),
        "RF": RandomForestClassifier(**settings["RF"], random_state=seed),
        "XGB": XGBClassifier(**settings["XGB"], random_state=seed),
        "NB": GaussianNB(**settings["NB"]),
        "KNN": KNeighborsClassifier(**settings["KNN"]),
    }
    return {name: Pipeline([("scale", StandardScaler()), ("model", classifiers[name])])
            for name in MODEL_ORDER}


def build_meta(config):
    params = dict(config["meta_classifier"])
    if params.pop("type") != "LogisticRegression":
        raise ValueError("Only the documented LR meta-classifier is implemented in v1.")
    return LogisticRegression(**params, random_state=config["seed"])


def positive_probability(estimator, X):
    classes = np.asarray(estimator.classes_)
    matches = np.flatnonzero(classes == 1)
    if len(matches) != 1:
        raise ValueError("Cannot locate the documented positive class, 1.")
    values = np.asarray(estimator.predict_proba(X))[:, int(matches[0])]
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Invalid predicted probabilities.")
    return values


def evaluate(y_true, probability, threshold):
    predicted = (probability > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    metrics = {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "precision": float(precision_score(y_true, predicted, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "specificity": float(tn / (tn + fp)),
        "mcc": float(matthews_corrcoef(y_true, predicted)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "test_records": int(len(y_true)),
    }
    return metrics, predicted


def compare_to_paper(measured):
    rows = []
    indexed = measured.set_index("model")
    for name, values in PUBLISHED.items():
        for metric, published in zip(METRICS, values):
            actual = float(indexed.loc[name, metric])
            rows.append({"model": name, "metric": metric, "measured": actual,
                         "published": published, "difference": actual - published,
                         "published_source": "Table 11, PDF p. 18", "report_type": "table_value"})
    for name, metric, published, source in ALTERNATIVE_REPORTS:
        actual = float(indexed.loc[name, metric])
        rows.append({"model": name, "metric": metric, "measured": actual,
                     "published": published, "difference": actual - published,
                     "published_source": source, "report_type": "conflicting_published_alternative"})
    return pd.DataFrame(rows)


def environment_record():
    packages = {}
    for name in ["numpy", "pandas", "scipy", "scikit-learn", "xgboost", "xgboost-cpu",
                 "matplotlib", "joblib", "threadpoolctl", "kagglehub"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": sys.version, "platform": platform.platform(), "packages": packages}


def overlap_audit(X, train_positions, test_positions):
    train_keys = set(X.iloc[train_positions].itertuples(index=False, name=None))
    test_keys = list(X.iloc[test_positions].itertuples(index=False, name=None))
    matched = np.asarray([key in train_keys for key in test_keys])
    return {
        "test_rows_with_predictor_pattern_in_training": int(matched.sum()),
        "test_rows": int(len(test_keys)),
        "unique_predictor_patterns_shared": int(len(train_keys.intersection(test_keys))),
        "note": "Exact predictor matches do not establish patient identity. Overlap limits claims about unseen-record generalisation."
    }, matched


def plot_outputs(out, y, probabilities, measured):
    styles = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1)), (0, (1, 1))]
    fig, ax = plt.subplots(figsize=(8, 6))
    curve_rows = []
    for i, (name, prob) in enumerate(probabilities.items()):
        fpr, tpr, thresholds = roc_curve(y, prob)
        ax.plot(fpr, tpr, color=str(0.08 + i * 0.10), linestyle=styles[i], linewidth=1.5,
                label=f"{name} (AUC {roc_auc_score(y, prob):.4f})")
        for x, z, threshold in zip(fpr, tpr, thresholds):
            curve_rows.append({"model": name, "fpr": x, "tpr": z, "threshold": threshold})
    ax.plot([0, 1], [0, 1], color="0.7", linewidth=.7)
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="False positive rate", ylabel="True positive rate",
           title="Measured holdout ROC curves - documented reconstruction")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "figures/roc_curves.png", dpi=180)
    plt.close(fig)
    pd.DataFrame(curve_rows).to_csv(out / "tables/roc_points.csv", index=False)
    fig, axes = plt.subplots(2, 4, figsize=(13, 6))
    for ax, (_, row) in zip(axes.flat, measured.iterrows()):
        matrix = np.array([[row.tn, row.fp], [row.fn, row.tp]], dtype=int)
        ax.imshow(matrix, cmap="Greys", vmin=0, vmax=max(matrix.max(), 1))
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(j, i, str(value), ha="center", va="center", color="white" if value > matrix.max()/2 else "black")
        ax.set(xticks=[0, 1], yticks=[0, 1], xlabel="Predicted", ylabel="Actual", title=row.model)
    axes.flat[-1].axis("off")
    fig.suptitle("Measured confusion matrices (positive class = 1)")
    fig.tight_layout()
    fig.savefig(out / "figures/confusion_matrices.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4))
    values = measured["accuracy"].to_numpy() * 100
    ax.bar(measured["model"], values, color="0.45")
    for i, value in enumerate(values):
        ax.text(i, value + 1, f"{value:.2f}", ha="center", fontsize=9)
    ax.set(ylim=(0, 106), ylabel="Accuracy (%)", title="Measured holdout accuracy")
    fig.tight_layout()
    fig.savefig(out / "figures/accuracy.png", dpi=180)
    plt.close(fig)


def run_reproduction(root, output_dir=None, save_models=False):
    root = Path(root).resolve()
    config = load_config(root)
    df, data_report = audit_dataset(root)
    # These policies are fixed in v1. Editing a descriptive string must not silently
    # claim that the underlying code has implemented a different procedure.
    required = {"duplicate_policy": "retain_and_report_overlap", "missing_value_policy": "stop_if_present",
                "feature_selection": "none", "stacking_stratify": True}
    for key, value in required.items():
        if config[key] != value:
            raise ValueError(f"Unsupported v1 policy {key}={config[key]!r}; code and specification must be revised together.")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + f"_seed{config['seed']}"
    out = Path(output_dir).resolve() if output_dir else root / "results/runs" / run_id
    if out.exists():
        raise FileExistsError(f"Run directory already exists: {out}. Existing results will not be overwritten.")
    for folder in [out, out / "tables", out / "figures", out / "predictions"]:
        folder.mkdir(parents=True, exist_ok=True)
    status = {"status": "running", "started_at_utc": utc_now(), "run_id": run_id,
              "dataset_sha256": data_report["dataset_sha256"], "specification_id": config["specification_id"]}
    write_json(out / "RUN_STATUS.json", status)
    write_json(out / "configuration.json", config)
    write_json(out / "dataset_audit.json", data_report)
    write_json(out / "environment.json", environment_record())
    provenance_path = root / "data/provenance.json"
    write_json(out / "data_provenance.json", json.loads(provenance_path.read_text()))
    source_hashes = {str(p.relative_to(root)): sha256(p) for p in sorted((root / "src").glob("*.py"))}
    source_hashes["config/reproduction_v1.json"] = sha256(root / "config/reproduction_v1.json")
    write_json(out / "source_hashes.json", source_hashes)
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=False)
    (out / "installed_packages.txt").write_text(freeze.stdout or "pip freeze unavailable\n")
    try:
        with warnings.catch_warnings(record=True) as captured, threadpool_limits(limits=1):
            warnings.simplefilter("always")
            X = df.loc[:, config["features"]]
            y = df[config["target"]].astype(int)
            assert config["target"] not in X.columns and X.shape[1] == 13
            train_pos, test_pos = train_test_split(
                np.arange(len(df)), test_size=config["test_size"], random_state=config["seed"],
                shuffle=config["split_shuffle"], stratify=y if config["split_stratify"] else None)
            X_train, X_test = X.iloc[train_pos], X.iloc[test_pos]
            y_train, y_test = y.iloc[train_pos], y.iloc[test_pos]
            if y_train.nunique() != 2 or y_test.nunique() != 2:
                raise ValueError("Both partitions must contain both target classes.")
            partition = np.full(len(df), "train", dtype=object)
            partition[test_pos] = "test"
            pd.DataFrame({"row_id": np.arange(len(df)), "partition": partition, "target": y}).to_csv(out / "tables/split_membership.csv", index=False)
            overlap, overlap_mask = overlap_audit(X, train_pos, test_pos)
            write_json(out / "split_overlap_audit.json", overlap)
            print(f"Training records: {len(train_pos)}; test records: {len(test_pos)}", flush=True)
            print("Test class counts:", y_test.value_counts().sort_index().to_dict(), flush=True)
            print("Test rows with exact predictor matches in training:", int(overlap_mask.sum()), flush=True)
            if overlap_mask.any():
                print("Interpret this as a row-level reproduction: exact record overlap exists across the split.", flush=True)
            models = build_models(config)
            fitted, probabilities, measured_rows, prediction_frames = {}, {}, [], []
            parameter_log = {name: estimator.get_params(deep=True) for name, estimator in models.items()}
            parameter_log["meta_classifier"] = build_meta(config).get_params(deep=True)
            # String representations only apply to estimator objects; numeric settings remain numeric.
            (out / "model_parameters.json").write_text(json.dumps(parameter_log, indent=2, default=str) + "\n")
            for name in MODEL_ORDER:
                print("Fitting individual classifier:", name, flush=True)
                start = time.perf_counter()
                model = clone(models[name]).fit(X_train, y_train)
                fit_seconds = time.perf_counter() - start
                probability = positive_probability(model, X_test)
                metrics, predicted = evaluate(y_test, probability, config["probability_threshold"])
                fitted[name], probabilities[name] = model, probability
                measured_rows.append({"model": name, "seed": config["seed"], **metrics, "fit_seconds": fit_seconds})
                prediction_frames.append(pd.DataFrame({"model": name, "row_id": test_pos, "y_true": y_test.to_numpy(),
                                                       "probability_class_1": probability, "y_pred": predicted,
                                                       "predictor_pattern_seen_in_train": overlap_mask}))
            folds = StratifiedKFold(n_splits=config["stacking_folds"], shuffle=config["stacking_shuffle"],
                                    random_state=config["seed"] if config["stacking_shuffle"] else None)
            oof = np.full((len(X_train), len(MODEL_ORDER)), np.nan)
            fold_id, seen = np.full(len(X_train), -1), np.zeros(len(X_train), dtype=int)
            fold_overlap_rows = []
            stack_start = time.perf_counter()
            for fold_number, (fit_pos, valid_pos) in enumerate(folds.split(X_train, y_train), start=1):
                print(f"Stacking fold {fold_number}/{config['stacking_folds']}", flush=True)
                if set(fit_pos).intersection(valid_pos):
                    raise AssertionError("Stacking fold index overlap.")
                seen[valid_pos] += 1
                fold_id[valid_pos] = fold_number
                fold_overlap, _ = overlap_audit(X_train, fit_pos, valid_pos)
                fold_overlap_rows.append({"fold": fold_number, **fold_overlap})
                for column, name in enumerate(MODEL_ORDER):
                    model = clone(models[name]).fit(X_train.iloc[fit_pos], y_train.iloc[fit_pos])
                    oof[valid_pos, column] = positive_probability(model, X_train.iloc[valid_pos])
            if not (seen == 1).all() or not np.isfinite(oof).all():
                raise AssertionError("Every training record must receive exactly one held-out prediction per base learner.")
            meta = build_meta(config).fit(oof, y_train)
            refit_seconds = sum(row["fit_seconds"] for row in measured_rows)
            stack_seconds = time.perf_counter() - stack_start + refit_seconds
            meta_test = np.column_stack([probabilities[name] for name in MODEL_ORDER])
            probability = positive_probability(meta, meta_test)
            metrics, predicted = evaluate(y_test, probability, config["probability_threshold"])
            # Base refits are the six already-fitted individual models. Total stack
            # build time includes these refits plus OOF fitting and meta fitting.
            measured_rows.append({"model": "Stacking", "seed": config["seed"], **metrics, "fit_seconds": stack_seconds})
            probabilities["Stacking"] = probability
            prediction_frames.append(pd.DataFrame({"model": "Stacking", "row_id": test_pos, "y_true": y_test.to_numpy(),
                                                   "probability_class_1": probability, "y_pred": predicted,
                                                   "predictor_pattern_seen_in_train": overlap_mask}))
            oof_table = pd.DataFrame(oof, columns=[name + "_probability_1" for name in MODEL_ORDER])
            oof_table.insert(0, "target", y_train.to_numpy())
            oof_table.insert(0, "validation_fold", fold_id)
            oof_table.insert(0, "row_id", train_pos)
            oof_table.to_csv(out / "predictions/stacking_oof_predictions.csv", index=False)
            pd.DataFrame(fold_overlap_rows).to_csv(out / "tables/stacking_fold_overlap.csv", index=False)
            measured = pd.DataFrame(measured_rows)
            measured.to_csv(out / "tables/measured_metrics.csv", index=False)
            compare_to_paper(measured).to_csv(out / "tables/published_vs_measured.csv", index=False)
            pd.concat(prediction_frames, ignore_index=True).to_csv(out / "predictions/test_predictions.csv", index=False)
            plot_outputs(out, y_test, probabilities, measured)
            if save_models:
                (out / "models").mkdir()
                joblib.dump({"base_models": fitted, "meta_classifier": meta, "base_order": MODEL_ORDER,
                             "feature_order": config["features"], "configuration": config}, out / "models/reconstruction.joblib")
            write_json(out / "warnings.json", [{"category": type(w.message).__name__, "message": str(w.message)} for w in captured])
        status.update(status="complete", completed_at_utc=utc_now(), models_evaluated=len(measured_rows),
                      test_rows_with_predictor_overlap=int(overlap_mask.sum()))
        write_json(out / "RUN_STATUS.json", status)
        print(measured[["model"] + METRICS].to_string(index=False), flush=True)
        print("Results saved to:", out, flush=True)
        return out, measured
    except Exception as exc:
        status.update(status="failed", error_type=type(exc).__name__, error=str(exc), failed_at_utc=utc_now())
        write_json(out / "RUN_STATUS.json", status)
        raise
