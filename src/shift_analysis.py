"""Train-vs-test shift analysis (uses test FEATURES only, never labels).

Usage:  python src/shift_analysis.py --data data/

Prints the findings that drove the design:
  1. Protocol / service mix differs strongly between train and test.
  2. The port-21 SYN flood: in train every flow has a reply and a 1-6 us
     duration; in test most have no reply and duration exactly 0.
  3. Nmap-style source-port reuse (one source port used by ~1000 flows).
  4. Adversarial validation: a classifier that separates train rows from test
     rows, with permutation importance showing which features carry the shift.
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance

from features import build_features, port_counts


def adversarial(A, B, name):
    A = A.sample(min(len(A), 20000), random_state=0)
    X = pd.concat([A, B], ignore_index=True)
    y = np.r_[np.zeros(len(A)), np.ones(len(B))]
    m = HistGradientBoostingClassifier(max_iter=150, random_state=0)
    auc = roc_auc_score(y, cross_val_predict(m, X, y, cv=3, method="predict_proba")[:, 1])
    m.fit(X, y)
    s = np.random.default_rng(0).choice(len(y), min(5000, len(y)), replace=False)
    imp = permutation_importance(m, X.iloc[s], y[s], n_repeats=3, random_state=0, scoring="roc_auc")
    top = pd.Series(imp.importances_mean, index=X.columns).sort_values(ascending=False).head(5)
    print(f"\n[adversarial: {name}] train-vs-test AUC = {auc:.3f}; top shifting features:")
    print(top.round(4).to_string())


def main(d):
    tr = pd.read_csv(f"{d}/train.csv"); te = pd.read_csv(f"{d}/test.csv")
    print("1) proto/service mix (share of rows)")
    mix = pd.concat([(tr.proto + "/" + tr.service).value_counts(normalize=True).rename("train"),
                     (te.proto + "/" + te.service).value_counts(normalize=True).rename("test")], axis=1)
    print(mix.fillna(0).round(3).to_string())

    print("\n2) port-21 SYN flood")
    for n, df in [("train", tr), ("test", te)]:
        f = df[df["id.resp_p"] == 21]
        print(f"  {n}: rows={len(f)}, with reply={(f.bwd_pkts_tot > 0).mean():.1%}, "
              f"duration==0: {(f.flow_duration == 0).mean():.1%}, "
              f"median flow_pkts_per_sec={f.flow_pkts_per_sec.median():,.0f}")

    print("\n3) source ports used by >50 flows (scanner signature)")
    for n, df in [("train", tr), ("test", te)]:
        vc = df["id.orig_p"].value_counts()
        print(f"  {n}: {(vc > 50).sum()} ports, covering {vc[vc > 50].sum()} flows")

    # orig_port_reuse is excluded: its scale depends on file size by construction
    A = build_features(tr, port_counts(tr)).drop(columns="orig_port_reuse")
    B = build_features(te, port_counts(te)).drop(columns="orig_port_reuse")
    m21, t21 = (tr["id.resp_p"] == 21).values, (te["id.resp_p"] == 21).values
    adversarial(A[m21], B[t21], "port-21 flood rows, after timing normalisation")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data")
    main(ap.parse_args().data)
