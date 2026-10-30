"""Train the final model on all training data and write submission.csv.

Usage:  python src/train_predict.py --data data/ --out submission.csv
"""
import argparse
import numpy as np
import pandas as pd

from features import build_features, port_counts, CLASSES
from model import make_model, sample_weights, downsample_flood

SEEDS = [0, 1, 2, 3, 4]


def main(data_dir, out):
    full = pd.read_csv(f"{data_dir}/train.csv")
    test = pd.read_csv(f"{data_dir}/test.csv")

    # Source-port reuse counts are computed inside each file separately (no labels
    # involved): a scanner reusing one port is visible within whichever capture it is in.
    X_test = build_features(test, port_counts(test))
    pc_train = port_counts(full)

    proba = np.zeros((len(test), len(CLASSES)))
    for s in SEEDS:
        tr = downsample_flood(full, seed=s)          # a different flood subsample per seed
        y = tr["Target"].map({c: i for i, c in enumerate(CLASSES)}).values
        X = build_features(tr, pc_train)
        m = make_model(s)
        m.fit(X, y, sample_weight=sample_weights(y))
        proba += m.predict_proba(X_test[X.columns]) / len(SEEDS)
        print(f"seed {s} done ({m.n_iter_} boosting rounds)")

    sub = pd.DataFrame({"id": test["id"], "Target": np.array(CLASSES)[proba.argmax(1)]})
    sub.to_csv(out, index=False)
    print(f"wrote {out}: {len(sub)} rows")
    print(sub["Target"].value_counts().to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="submission.csv")
    a = ap.parse_args()
    main(a.data, a.out)
