"""Validation: why random CV is misleading here, and the profile-holdout CV used instead.

Usage:  python src/validate.py --data data/

1. Random stratified 5-fold CV (the naive baseline; very optimistic).
2. Profile-holdout CV: every class is clustered (KMeans) into ~8 "behaviour
   profiles"; StratifiedGroupKFold then holds out whole profiles, so the model is
   scored on traffic patterns it never saw -- the same situation as the hidden test.
   Repeated with 3 different clusterings/splits to measure noise (std).
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import f1_score, classification_report

from features import build_features, port_counts, CLASSES
from model import make_model, sample_weights, downsample_flood


def profile_groups(X, y, k_per_class=8, seed=0):
    Z = StandardScaler().fit_transform(np.log1p(X.clip(lower=0)))
    g, nxt = np.zeros(len(y), dtype=int), 0
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        k = min(k_per_class, len(idx) // 20)
        g[idx] = KMeans(k, n_init=4, random_state=seed).fit_predict(Z[idx]) + nxt
        nxt += k
    return g


def oof_predict(X, y, splitter, groups=None, seed=0):
    oof = np.zeros(len(y), dtype=int)
    for a, b in splitter.split(X, y, groups):
        m = make_model(seed)
        m.fit(X.iloc[a], y[a], sample_weight=sample_weights(y[a]))
        oof[b] = m.predict(X.iloc[b])
    return oof


def main(data_dir):
    full = pd.read_csv(f"{data_dir}/train.csv")
    pc = port_counts(full)                      # label-free, from the whole train file
    tr = downsample_flood(full)
    y = tr["Target"].map({c: i for i, c in enumerate(CLASSES)}).values
    X = build_features(tr, pc)

    oof = oof_predict(X, y, StratifiedKFold(5, shuffle=True, random_state=0))
    print(f"Random stratified CV   macro F1 = {f1_score(y, oof, average='macro'):.4f}")

    scores = []
    for s in range(3):
        g = profile_groups(X, y, seed=s)
        oof = oof_predict(X, y, StratifiedGroupKFold(5, shuffle=True, random_state=s), g, seed=s)
        scores.append(f1_score(y, oof, average="macro"))
        print(f"Profile-holdout CV #{s} macro F1 = {scores[-1]:.4f}")
    print(f"Profile-holdout CV     macro F1 = {np.mean(scores):.4f} +/- {np.std(scores):.4f}")
    print(classification_report(y, oof, target_names=CLASSES, digits=3))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    main(ap.parse_args().data)
