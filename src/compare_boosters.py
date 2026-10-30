"""Compare scikit-learn HistGradientBoosting vs LightGBM vs XGBoost under the SAME
profile-holdout CV used everywhere else, then write one submission file per model.

Usage:
    python -m pip install lightgbm xgboost
    python src/compare_boosters.py --data data            # full run (~15-25 min)
    python src/compare_boosters.py --data data --quick    # 1 repeat, quick check

Outputs (never overwrites submission.csv):
    submission_hgb.csv, submission_lgbm.csv, submission_xgb.csv, submission_ensemble.csv
    (the ensemble averages the probabilities of the three models)
Adopt a booster only if its CV score beats the baseline by clearly more than the
noise (the +/- printed next to each score); otherwise keep the current submission.
"""
import argparse
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score

from features import build_features, port_counts, CLASSES
from model import make_model, sample_weights, downsample_flood
from validate import profile_groups

try:
    import lightgbm as lgb
except ImportError:
    lgb = None
try:
    import xgboost as xgb
except ImportError:
    xgb = None


def make_hgb(seed, rounds):
    return make_model(seed)                      # baseline: early stopping decides the rounds


def make_lgbm(seed, rounds):
    return lgb.LGBMClassifier(
        n_estimators=rounds, learning_rate=0.1, num_leaves=31, min_child_samples=20,
        reg_lambda=1.0, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        random_state=seed, n_jobs=-1, verbose=-1)


def make_xgb(seed, rounds):
    return xgb.XGBClassifier(
        n_estimators=rounds, learning_rate=0.1, max_depth=6, min_child_weight=1,
        reg_lambda=1.0, subsample=0.8, colsample_bytree=0.8, tree_method="hist",
        objective="multi:softprob", random_state=seed, n_jobs=-1, verbosity=0)


def candidates():
    c = {"hgb": [("hgb", make_hgb, 0)]}
    if lgb is not None:
        c["lgbm"] = [(f"lgbm_{r}r", make_lgbm, r) for r in (60, 120)]
    if xgb is not None:
        c["xgb"] = [(f"xgb_{r}r", make_xgb, r) for r in (60, 120)]
    return c


def fit(maker, seed, rounds, X, y):
    m = maker(seed, rounds)
    m.fit(X.values, y, sample_weight=sample_weights(y))   # numpy: avoids feature-name restrictions
    return m


def main(data_dir, quick):
    full = pd.read_csv(f"{data_dir}/train.csv")
    test = pd.read_csv(f"{data_dir}/test.csv")
    pc_train = port_counts(full)
    X_test = build_features(test, port_counts(test))

    tr = downsample_flood(full)
    y = tr["Target"].map({c: i for i, c in enumerate(CLASSES)}).values
    X = build_features(tr, pc_train)
    n_rep = 1 if quick else 3
    groups = [profile_groups(X, y, seed=s) for s in range(n_rep)]

    cand = candidates()
    if lgb is None: print("lightgbm not installed -> skipped (python -m pip install lightgbm)")
    if xgb is None: print("xgboost not installed  -> skipped (python -m pip install xgboost)")

    results, oofp = {}, {}
    for family, configs in cand.items():
        for name, maker, rounds in configs:
            scores, P_all = [], []
            for s in range(n_rep):
                P = np.zeros((len(y), len(CLASSES)))
                for a, b in StratifiedGroupKFold(5, shuffle=True, random_state=s).split(X, y, groups[s]):
                    m = fit(maker, s, rounds, X.iloc[a], y[a])
                    P[b] = m.predict_proba(X.iloc[b].values)
                P_all.append(P)
                scores.append(f1_score(y, P.argmax(1), average="macro"))
            oofp[name] = P_all
            results[name] = (family, maker, rounds, np.mean(scores), np.std(scores))
            print(f"{name:12s} profile-holdout macro F1 = {np.mean(scores):.4f} +/- {np.std(scores):.4f}", flush=True)

    best = {fam: max((n for n, r in results.items() if r[0] == fam), key=lambda n: results[n][3]) for fam in cand}

    # ensemble CV: average the out-of-fold probabilities of each family's best config (same folds)
    ens_score = None
    if len(best) >= 2:
        sc = [f1_score(y, np.mean([oofp[best[f]][s] for f in best], axis=0).argmax(1), average="macro")
              for s in range(n_rep)]
        ens_score = np.mean(sc)
        print(f"ENSEMBLE ({' + '.join(best.values())}) profile-holdout macro F1 = {np.mean(sc):.4f} +/- {np.std(sc):.4f}", flush=True)

    # write one submission per family (best config, 5 seeds averaged) and the ensemble
    existing = pd.read_csv("submission.csv") if os.path.exists("submission.csv") else None

    def write(proba, out, label):
        sub = pd.DataFrame({"id": test["id"], "Target": np.array(CLASSES)[proba.argmax(1)]})
        sub.to_csv(out, index=False)
        diff = "" if existing is None else f", differs from submission.csv on {(sub.Target.values != existing.Target.values).sum()} rows"
        print(f"wrote {out} ({label}){diff}")
        print(sub["Target"].value_counts().to_string())

    fam_proba = {}
    for family in cand:
        name = best[family]
        _, maker, rounds, score, _ = results[name]
        proba = np.zeros((len(test), len(CLASSES)))
        for s in range(5):
            trs = downsample_flood(full, seed=s)
            ys = trs["Target"].map({c: i for i, c in enumerate(CLASSES)}).values
            Xs = build_features(trs, pc_train)
            proba += fit(maker, s, rounds, Xs, ys).predict_proba(X_test[Xs.columns].values) / 5
        fam_proba[family] = proba
        write(proba, f"submission_{family}.csv", f"config {name}, CV {score:.4f}")
    if len(fam_proba) >= 2:
        write(np.mean(list(fam_proba.values()), axis=0), "submission_ensemble.csv", f"ensemble CV {ens_score:.4f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    main(a.data, a.quick)
