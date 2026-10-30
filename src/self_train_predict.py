"""HIGH-VARIANCE variant: self-training (pseudo-labelling) on the unlabeled test features.

Usage:  python src/self_train_predict.py --data data --out submission_selftrain.csv

How it works (no test labels are ever used):
  1. Train the normal model on the labelled training data and predict the test set.
  2. Keep the test rows where the predicted probability is >= THRESH, treat the
     predicted class as a pseudo-label, add them to the training set, retrain.
  3. Repeat for ROUNDS rounds; average 5 seeds.

Why it is a gamble: in profile-holdout CV it averaged 0.765 (vs 0.805 for the normal
model) with a huge spread -- one split scored 0.839, another collapsed to 0.63.
It can help when the unseen profiles form tight clusters the model is mostly right
about, and hurt when confident mistakes reinforce themselves.

Never overwrites submission.csv.
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight

from features import build_features, port_counts, CLASSES
from model import make_model, downsample_flood

THRESH, ROUNDS, SEEDS = 0.95, 2, [0, 1, 2, 3, 4]


def main(data_dir, out):
    full = pd.read_csv(f"{data_dir}/train.csv")
    test = pd.read_csv(f"{data_dir}/test.csv")
    X_test = build_features(test, port_counts(test))
    pc_train = port_counts(full)

    proba = np.zeros((len(test), len(CLASSES)))
    n_pseudo = []
    for s in SEEDS:
        tr = downsample_flood(full, seed=s)
        y = tr["Target"].map({c: i for i, c in enumerate(CLASSES)}).values
        X = build_features(tr, pc_train)
        w = compute_sample_weight("balanced", y)
        m = make_model(s); m.fit(X, y, sample_weight=w)
        P = m.predict_proba(X_test[X.columns])
        for r in range(ROUNDS):
            conf = P.max(1) >= THRESH
            X2 = pd.concat([X, X_test.loc[conf, X.columns]], ignore_index=True)
            y2 = np.r_[y, P.argmax(1)[conf]]
            m = make_model(s); m.fit(X2, y2, sample_weight=compute_sample_weight("balanced", y2))
            P = m.predict_proba(X_test[X.columns])
        n_pseudo.append(int(conf.sum()))
        proba += P / len(SEEDS)
        print(f"seed {s}: used {conf.sum()} confident test rows as pseudo-labels in last round")

    sub = pd.DataFrame({"id": test["id"], "Target": np.array(CLASSES)[proba.argmax(1)]})
    sub.to_csv(out, index=False)
    print(f"wrote {out}: {len(sub)} rows")
    print(sub["Target"].value_counts().to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="submission_selftrain.csv")
    a = ap.parse_args()
    main(a.data, a.out)
