"""Validate the saved evidence package without retraining the models.

The full experiment pipeline is available through ``run_all.py``.  This
lightweight check is intended for a quick hand-off review: it verifies the
pinned data, saved prediction arrays, key metric rows, generated paper assets,
and the recorded clean execution.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parent
EXPECTED_HASH = "ddb2996b2f4db2e00aad13f4518200179ff69f79093838e3c21ffa672ebec0f1"
METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def metric_row(df: pd.DataFrame, seed: int, arm: str, model: str, weighting: str = "row") -> pd.Series:
    rows = df[(df.seed == seed) & (df.arm == arm) & (df.model == model) & (df.weighting == weighting)]
    require(len(rows) == 1, f"expected one metrics row for {seed}/{arm}/{model}/{weighting}")
    return rows.iloc[0]


def main() -> None:
    data_path = ROOT / "data" / "heart.csv"
    digest = hashlib.sha256(data_path.read_bytes()).hexdigest()
    require(digest == EXPECTED_HASH, f"dataset hash mismatch: {digest}")
    data = pd.read_csv(data_path)
    require(data.shape == (1025, 14), f"unexpected dataset shape: {data.shape}")

    m02_path = ROOT / "results" / "02_protocols" / "metrics.csv"
    m02 = pd.read_csv(m02_path)
    require(set(METRICS).issubset(m02.columns), "protocol metrics are incomplete")
    require(set(m02.seed) == set(range(30)) | {42}, "protocol seeds are incomplete")
    require(set(m02.arm) == {"A", "B", "C"}, "protocol arms are incomplete")

    # Recompute the anchor stack metrics from the saved prediction array.
    run = np.load(ROOT / "results" / "02_protocols" / "runs" / "seed42_C.npz")
    y = run["y_test"]
    p = run["p_stack"]
    pred = (p > 0.5).astype(int)
    recomputed = {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, p),
    }
    anchor = metric_row(m02, 42, "C", "Stacking")
    for name, value in recomputed.items():
        require(abs(float(anchor[name]) - value) < 1e-12, f"anchor {name} does not match saved predictions")

    # Recompute the headline paired C-minus-B accuracy result from the table.
    b = m02[(m02.seed.isin(range(30))) & (m02.arm == "B") & (m02.model == "Stacking") & (m02.weighting == "row")]
    c = m02[(m02.seed.isin(range(30))) & (m02.arm == "C") & (m02.model == "Stacking") & (m02.weighting == "row")]
    require(len(b) == len(c) == 30, "paired C/B stack rows are incomplete")
    paired_pp = (c.sort_values("seed").accuracy.to_numpy() - b.sort_values("seed").accuracy.to_numpy()) * 100
    require(abs(float(paired_pp.mean()) - 2.6102) < 0.02, "headline C-minus-B result changed")

    d_stack = pd.read_csv(ROOT / "results" / "04_preprocessing" / "metrics.csv")
    d_stack = d_stack[(d_stack.seed.isin(range(30))) & (d_stack.arm == "D") & (d_stack.weighting == "row")]
    d_lr = d_stack[d_stack.model == "LR"].sort_values("seed").reset_index(drop=True)
    d_stack_model = d_stack[d_stack.model == "Stacking"].sort_values("seed").reset_index(drop=True)
    require(len(d_lr) == len(d_stack_model) == 30, "matched D stack/LR rows are incomplete")
    d_lr_delta = (d_stack_model.accuracy - d_lr.accuracy) * 100
    require(abs(float(d_lr_delta.mean()) - (-0.9719)) < 0.02, "matched D stack-D LR contrast changed")
    contrasts = pd.read_csv(ROOT / "results/07_statistics/contrasts.csv")
    matched = contrasts[(contrasts.contrast == "D stack-D LR") &
                        (contrasts.weighting == "row") & (contrasts.metric == "accuracy")].iloc[0]
    corrected_t = d_lr_delta.mean() / np.sqrt((1 / 30 + 61 / 241) * d_lr_delta.var(ddof=1))
    corrected_p = 2 * stats.t.sf(abs(corrected_t), 29)
    require(abs(float(matched.mean_pp) - d_lr_delta.mean()) < 1e-12, "D/LR contrast mean mismatch")
    require(abs(float(matched.p_corr) - corrected_p) < 1e-12, "D/LR corrected p-value mismatch")
    holm_names = {"D-C stack", "D-B stack", "D stack-C LR", "D stack-D LR"}
    family = contrasts[(contrasts.contrast.isin(holm_names)) &
                       (contrasts.weighting == "row") & (contrasts.metric == "accuracy")].sort_values("p_corr")
    require(len(family) == 4, "Holm family does not include all four D contrasts")
    independent_holm = np.maximum.accumulate(np.minimum(1, family.p_corr.to_numpy() * np.arange(4, 0, -1)))
    require(np.allclose(family.holm_p, independent_holm, atol=1e-12), "four-contrast Holm values mismatch")

    expected_runs = {
        "results/02_protocols/runs": 93,
        "results/03_nested_validation/runs": 20,
        "results/04_preprocessing/runs": 93,
        "results/05_tuning/runs": 31,
    }
    for relative, count in expected_runs.items():
        files = list((ROOT / relative).glob("*.npz"))
        require(len(files) == count, f"{relative}: expected {count} NPZ files, found {len(files)}")
        for path in files:
            with np.load(path) as saved:
                require(all(np.isfinite(saved[key]).all() for key in saved.files if saved[key].dtype.kind in "fc"),
                        f"non-finite values in {path}")

    execution_files = sorted((ROOT / "verification" / "runs").glob("*/execution.json"))
    require(execution_files, "no recorded verification execution")
    full_runs = [p for p in execution_files
                 if (r := json.loads(p.read_text())).get("status") == "passed"
                 and r.get("steps_requested") == list(range(1, 9))]
    require(full_runs, "no successful full eight-stage training run is recorded")
    execution = json.loads(execution_files[-1].read_text())
    require(execution.get("status") == "passed", "latest recorded execution did not pass")
    require(all(stage.get("returncode") == 0 for stage in execution.get("stages", [])),
            "a recorded experiment stage failed")

    for relative in (
        "paper/generated/numbers.tex",
        "paper/generated/tab_avc.tex",
        "paper/generated/tab_combiner.tex",
        "paper/generated/tab_ablate.tex",
        "notebooks/SIT307_HD_Verification.ipynb",
    ):
        require((ROOT / relative).is_file(), f"missing generated evidence asset: {relative}")

    summary = {
        "status": "PASS",
        "dataset_rows": int(data.shape[0]),
        "dataset_predictors": int(data.shape[1] - 1),
        "dataset_sha256": digest,
        "anchor_protocol": "C / seed 42 / Stacking",
        "anchor_metrics": {name: float(value) for name, value in recomputed.items()},
        "paired_C_minus_B_accuracy_pp": float(paired_pp.mean()),
        "paired_D_stack_minus_D_LR_accuracy_pp": float(d_lr_delta.mean()),
        "D_stack_minus_D_LR_corrected_p": float(corrected_p),
        "D_stack_minus_D_LR_Holm_p": float(matched.holm_p),
        "full_training_execution": str(full_runs[-1].relative_to(ROOT)),
        "latest_execution_steps": execution.get("steps_requested"),
        "saved_prediction_runs": expected_runs,
        "verification_execution": str(execution_files[-1].relative_to(ROOT)),
    }
    out = ROOT / "verification" / "verification_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print("PASS: saved evidence, generated assets and anchor metrics verified")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
