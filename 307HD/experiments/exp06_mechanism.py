"""Experiment 6: why grouping the inner folds helps (E1-E3), seeds 0-29.

E1  Overlap decomposition of Protocol A: accuracy on test rows whose exact pattern also
    appears in training ("seen") versus rows whose pattern does not ("unseen").
E2  Combiner control on the matched B/C partitions: LR meta-learner (the stack), the single
    learner with the highest OOF accuracy, and an unweighted soft vote. The soft vote never
    uses OOF predictions, so it is identical under B and C by construction.
E3  Inner-fold count: LR-meta stack under B and C with k = 3, 5, 10 (k = 5 from exp02).

Reads results/02_protocols/runs/*.npz. Outputs to results/06_mechanism/.
"""
import numpy as np
import pandas as pd

from common import RESULTS, Timer, fresh_dir, load, map_seeds

from src.evaluation import SEEDS
from src.protocols import fit_stack, make_models, outer_split
from src.reproduction import MODEL_ORDER, evaluate, positive_probability

RUNS = RESULTS / "02_protocols" / "runs"


def e1(groups):
    rows = []
    for seed in SEEDS:
        r = dict(np.load(RUNS / f"seed{seed}_A.npz"))
        seen = np.isin(groups[r["test"]], groups[r["dev"]])
        unseen_patterns = len(set(groups[r["test"]][~seen]))
        probs = {n: r["Z_test"][:, j] for j, n in enumerate(MODEL_ORDER)}
        probs["Stacking"] = r["p_stack"]
        for name, p in probs.items():
            ok = (p > 0.5).astype(int) == r["y_test"]
            rows.append({"seed": seed, "model": name, "n_seen": int(seen.sum()), "n_unseen": int((~seen).sum()),
                         "unseen_patterns": unseen_patterns, "correct_seen": int(ok[seen].sum()),
                         "correct_unseen": int(ok[~seen].sum()), "accuracy": float(ok.mean())})
    per_seed = pd.DataFrame(rows)
    agg = per_seed.groupby("model", sort=False).agg(
        n_seen=("n_seen", "sum"), n_unseen=("n_unseen", "sum"), correct_seen=("correct_seen", "sum"),
        correct_unseen=("correct_unseen", "sum"), accuracy_mean=("accuracy", "mean")).reset_index()
    agg["acc_seen"] = agg.correct_seen / agg.n_seen
    agg["acc_unseen"] = agg.correct_unseen / agg.n_unseen
    return per_seed, agg


def e2():
    rows = []
    for seed in SEEDS:
        for arm in ("B", "C"):
            r = dict(np.load(RUNS / f"seed{seed}_{arm}.npz"))
            yt, yd = r["y_test"], r["y_dev"]
            m, _ = evaluate(yt, r["p_stack"], 0.5)
            rows.append({"seed": seed, "arm": arm, "combiner": "lr_meta", "selected": "", **m})
            oof_acc = ((r["Z_dev"] > 0.5).astype(int) == yd[:, None]).mean(axis=0)
            best = int(np.argmax(oof_acc))  # ties -> first in MODEL_ORDER
            m, _ = evaluate(yt, r["Z_test"][:, best], 0.5)
            rows.append({"seed": seed, "arm": arm, "combiner": "oof_best_single", "selected": MODEL_ORDER[best], **m})
            m, _ = evaluate(yt, r["Z_test"].mean(axis=1), 0.5)
            rows.append({"seed": seed, "arm": arm, "combiner": "soft_vote", "selected": "", **m})
    return pd.DataFrame(rows)


def e3_seed(seed, config, X, y, groups):
    dev, test = outer_split("group", len(X), groups, seed)
    models = make_models(config, seed)
    base = dict(np.load(RUNS / f"seed{seed}_C.npz"))
    rows = []
    for k in (3, 10):
        for inner, arm in (("row", "B"), ("group", "C")):
            res = fit_stack(config, models, X.iloc[dev], y.iloc[dev], groups[dev], inner, seed, k=k)
            p = positive_probability(res["meta"], base["Z_test"])
            m, _ = evaluate(base["y_test"], p, 0.5)
            rows.append({"seed": seed, "arm": arm, "k": k, **m})
    print(f"  seed {seed} done", flush=True)
    return rows


def main():
    config, X, y, groups, _ = load()
    out = fresh_dir("06_mechanism")
    with Timer("exp06 E1 overlap decomposition"):
        per_seed, agg = e1(groups)
        per_seed.to_csv(out / "e1_per_seed.csv", index=False)
        agg.to_csv(out / "e1_summary.csv", index=False)
        print(agg[["model", "acc_seen", "acc_unseen", "n_seen", "n_unseen"]].round(4).to_string(index=False))
    with Timer("exp06 E2 combiner control"):
        comb = e2()
        comb.to_csv(out / "e2_combiners.csv", index=False)
        print(comb.groupby(["combiner", "arm"]).accuracy.mean().round(4))
    with Timer("exp06 E3 inner-fold count"):
        res = map_seeds(e3_seed, SEEDS, config, X, y, groups)
        e3 = pd.DataFrame([r for rows in res for r in rows])
        k5 = comb[comb.combiner == "lr_meta"].drop(columns=["combiner", "selected"]).assign(k=5)
        e3 = pd.concat([e3, k5[e3.columns]], ignore_index=True).sort_values(["k", "arm", "seed"])
        e3.to_csv(out / "e3_fold_count.csv", index=False)
        print(e3.groupby(["k", "arm"]).accuracy.mean().round(4))


if __name__ == "__main__":
    main()
