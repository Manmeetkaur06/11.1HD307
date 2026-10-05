"""Experiment 2: matched protocols A, B and C over seeds 0-29 plus the seed-42 anchor.

A = row outer / row inner (reproduction); B = grouped outer / row inner;
C = grouped outer / grouped inner (proposed). B and C share the outer split and the
refitted base-learner predictions, so C - B isolates the inner-fold strategy.

Outputs (results/02_protocols/):
  metrics.csv        row- and pattern-weighted metrics for every seed, arm and model
  summary.csv        mean and SD over seeds 0-29
  meta.csv           meta-learner coefficients and base-learner OOF accuracy
  overlap.csv        Protocol A test rows whose exact pattern appears in training
  runs/*.npz         predictions reused by experiments 3 and 6
"""
import numpy as np
import pandas as pd

from common import Timer, fresh_dir, load, map_seeds, save_npz

from src.evaluation import ANCHOR_SEED, SEEDS, score_run, summarise
from src.protocols import ARMS
from src.reproduction import MODEL_ORDER


def one_seed(seed, config, X, y, groups, out):
    cache, rows, meta_rows, overlap = {}, [], [], None
    for name in ("A", "B", "C"):
        run = run_arm_cached(name, config, X, y, groups, seed, cache)
        save_npz(out / "runs" / f"seed{seed}_{name}.npz", run)
        rows += score_run(run, groups)
        oof_acc = ((run["Z_dev"] > 0.5).astype(int) == run["y_dev"][:, None]).mean(axis=0)
        meta_rows.append({"seed": seed, "arm": name, "intercept": run["meta_intercept"],
                          **{f"coef_{n}": c for n, c in zip(MODEL_ORDER, run["meta_coef"])},
                          **{f"oof_acc_{n}": a for n, a in zip(MODEL_ORDER, oof_acc)},
                          "n_dev": len(run["dev"]), "n_test": len(run["test"]),
                          "dev_groups": len(set(groups[run["dev"]])),
                          "test_groups": len(set(groups[run["test"]]))})
        if name == "A":
            seen = np.isin(groups[run["test"]], groups[run["dev"]])
            overlap = {"seed": seed, "test_rows": len(seen), "seen_rows": int(seen.sum())}
    print(f"  seed {seed} done", flush=True)
    return rows, meta_rows, overlap


def run_arm_cached(name, config, X, y, groups, seed, cache):
    from src.protocols import run_arm
    return run_arm(ARMS[name], config, X, y, groups, seed, base_cache=cache)


def main():
    config, X, y, groups, _ = load()
    out = fresh_dir("02_protocols")
    (out / "runs").mkdir()
    with Timer("exp02 protocols A/B/C"):
        results = map_seeds(one_seed, SEEDS + [ANCHOR_SEED], config, X, y, groups, out)
    metrics = pd.DataFrame([r for res in results for r in res[0]])
    meta = pd.DataFrame([r for res in results for r in res[1]])
    overlap = pd.DataFrame([res[2] for res in results])
    metrics.to_csv(out / "metrics.csv", index=False)
    meta.to_csv(out / "meta.csv", index=False)
    overlap.to_csv(out / "overlap.csv", index=False)
    rep = metrics[metrics.seed.isin(SEEDS)]
    summarise(rep, ["arm", "weighting", "model"]).to_csv(out / "summary.csv", index=False)
    s = summarise(rep[rep.weighting == "row"], ["arm", "model"]).set_index(["arm", "model"])
    print(s.loc[(slice(None), "Stacking"), ["accuracy_mean", "accuracy_sd", "roc_auc_mean"]].round(4))


if __name__ == "__main__":
    main()
