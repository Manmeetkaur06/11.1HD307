"""Experiment 7: paired statistics for every reported contrast (seeds 0-29) and the
seed-42 pattern-level bootstrap / McNemar test.

Paired differences are in percentage points. The Nadeau-Bengio corrected resampled t-test
uses n_test/n_train = 61/241 (counted in patterns). Holm adjustment is applied within each
metric/weighting family of four reported D contrasts (D-C stack, D-B stack, D stack-C LR, D stack-D LR).
Wilcoxon signed-rank p-values are descriptive only (seeds are not independent samples).

Reads results/02, 04, 05 and 06. Outputs to results/07_statistics/.
"""
import numpy as np
import pandas as pd
from scipy import stats

from common import RESULTS, Timer, fresh_dir, load

from src.evaluation import (METRICS, SEEDS, corrected_t, holm, pattern_positions, wilcoxon_p,
                            win_tie_loss, write_json)
from src.reproduction import MODEL_ORDER

RATIO = 61 / 241
BOOT_DRAWS, BOOT_SEED = 2000, 20261001
HOLM_FAMILY = ["D-C stack", "D-B stack", "D stack-C LR", "D stack-D LR"]


def all_metrics():
    frames = [pd.read_csv(RESULTS / "02_protocols" / "metrics.csv"),
              pd.read_csv(RESULTS / "04_preprocessing" / "metrics.csv"),
              pd.read_csv(RESULTS / "05_tuning" / "metrics.csv")]
    df = pd.concat(frames, ignore_index=True)
    return df[df.seed.isin(SEEDS)]


def contrast_specs():
    st, lr, nb = "Stacking", "LR", "NB"
    return {
        "C-B stack": (("C", st), ("B", st)),
        "D-C stack": (("D", st), ("C", st)),
        "D-B stack": (("D", st), ("B", st)),
        "D stack-C LR": (("D", st), ("C", lr)),
        "C_dedup-C stack": (("C_dedup", st), ("C", st)),
        "C_onehot-C stack": (("C_onehot", st), ("C", st)),
        "D-C_onehot stack": (("D", st), ("C_onehot", st)),
        "D-C_dedup stack": (("D", st), ("C_dedup", st)),
        "C stack-C LR": (("C", st), ("C", lr)),
        "C stack-C NB": (("C", st), ("C", nb)),
        "D stack-D LR": (("D", st), ("D", lr)),
        "Selected0.5-D stack": (("Selected_0.5", "Selected_0.5"), ("D", st)),
        "SelectedCut-D stack": (("Selected_cutoff", "Selected_cutoff"), ("D", st)),
        "Selected0.5-D LR": (("Selected_0.5", "Selected_0.5"), ("D", lr)),
        "SelectedCut-D LR": (("Selected_cutoff", "Selected_cutoff"), ("D", lr)),
        "SelectedCut-Selected0.5": (("Selected_cutoff", "Selected_cutoff"), ("Selected_0.5", "Selected_0.5")),
    }


def series(df, arm, model, weighting, metric):
    s = df[(df.arm == arm) & (df.model == model) & (df.weighting == weighting)].set_index("seed")[metric]
    return s.loc[SEEDS].to_numpy()


def contrasts(df):
    rows = []
    for name, ((a_arm, a_mod), (b_arm, b_mod)) in contrast_specs().items():
        for weighting in ("row", "pattern"):
            for metric in METRICS:
                d = (series(df, a_arm, a_mod, weighting, metric) - series(df, b_arm, b_mod, weighting, metric)) * 100
                t, p = corrected_t(d, RATIO)
                w, ti, lo = win_tie_loss(d)
                rows.append({"contrast": name, "weighting": weighting, "metric": metric, "mean_pp": d.mean(),
                             "sd_pp": d.std(ddof=1), "wins": w, "ties": ti, "losses": lo,
                             "t_corr": t, "p_corr": p, "wilcoxon_p": wilcoxon_p(d)})
    out = pd.DataFrame(rows)
    out["holm_p"] = np.nan
    for (weighting, metric), idx in out[out.contrast.isin(HOLM_FAMILY)].groupby(["weighting", "metric"]).groups.items():
        out.loc[idx, "holm_p"] = holm(out.loc[idx, "p_corr"].to_numpy())
    return out


def anchor_bootstrap(groups):
    """Seed-42 Protocol C: stacking versus Gaussian NB, one vote per test pattern."""
    r = dict(np.load(RESULTS / "02_protocols" / "runs" / "seed42_C.npz"))
    pos = pattern_positions(r["test"], groups)
    y = r["y_test"][pos]
    p_stack, p_nb = r["p_stack"][pos], r["Z_test"][pos, MODEL_ORDER.index("NB")]
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    def metrics(idx):
        yy = y[idx]
        out = {}
        for name, fn in (("accuracy", accuracy_score), ("f1", f1_score)):
            out[name] = fn(yy, (p_stack[idx] > 0.5).astype(int)) - fn(yy, (p_nb[idx] > 0.5).astype(int))
        out["roc_auc"] = (roc_auc_score(yy, p_stack[idx]) - roc_auc_score(yy, p_nb[idx])
                          if len(set(yy)) == 2 else np.nan)
        return out

    observed = metrics(np.arange(len(y)))
    rng = np.random.default_rng(BOOT_SEED)
    draws = pd.DataFrame([metrics(rng.integers(0, len(y), len(y))) for _ in range(BOOT_DRAWS)])
    rows = [{"metric": m, "observed": observed[m], "ci_low": np.nanpercentile(draws[m], 2.5),
             "ci_high": np.nanpercentile(draws[m], 97.5), "n_patterns": len(y)} for m in observed]
    ok_s, ok_n = (p_stack > 0.5) == y, (p_nb > 0.5) == y
    b, c = int((ok_s & ~ok_n).sum()), int((~ok_s & ok_n).sum())
    mcnemar_p = float(stats.binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0
    return pd.DataFrame(rows), {"stack_only_correct": b, "nb_only_correct": c, "mcnemar_exact_p": mcnemar_p,
                                "stack_correct": int(ok_s.sum()), "nb_correct": int(ok_n.sum()), "n_patterns": len(y)}


def main():
    _, _, _, groups, _ = load()
    out = fresh_dir("07_statistics")
    with Timer("exp07 statistics"):
        con = contrasts(all_metrics())
        con.to_csv(out / "contrasts.csv", index=False)
        boot, mcn = anchor_bootstrap(groups)
        boot.to_csv(out / "anchor_bootstrap.csv", index=False)
        write_json(out / "anchor_mcnemar.json", mcn)
    acc = con[(con.metric == "accuracy")]
    print(acc[["contrast", "weighting", "mean_pp", "sd_pp", "wins", "ties", "losses", "p_corr", "holm_p"]]
          .round(4).to_string(index=False))
    print(boot.round(4).to_string(index=False)); print(mcn)


if __name__ == "__main__":
    main()
