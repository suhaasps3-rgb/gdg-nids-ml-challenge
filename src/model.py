"""Model definition + training-set preparation (shared by validate.py and train_predict.py)."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight

N_FLOOD_KEEP = 15000   # SYN-flood rows kept (of ~79.7k near-identical rows)


def make_model(seed=0):
    # Chosen on profile-holdout CV (see README, "Model comparison").
    # early_stopping is on by default for >10k rows (10% internal holdout), so the
    # seed changes the model -> we average several seeds in the final submission.
    return HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.1, max_leaf_nodes=31,
        l2_regularization=1.0, random_state=seed)


def sample_weights(y):
    # 'balanced' = n_samples / (n_classes * class_count): every class contributes
    # equally to the loss, which matches the macro-F1 metric.
    return compute_sample_weight("balanced", y)


def downsample_flood(df, n_keep=N_FLOOD_KEEP, seed=0):
    """79.7k of 84.7k train rows are one identical-looking SYN flood on port 21.
    Keeping 15k loses no information, speeds training ~4x, and every other
    row (incl. all non-port-21 Availability traffic) is kept."""
    rng = np.random.default_rng(seed)
    flood = df.index[(df["Target"] == "Availability") & (df["id.resp_p"] == 21)]
    keep = df.index.difference(flood).union(rng.choice(flood, min(n_keep, len(flood)), replace=False))
    return df.loc[keep].reset_index(drop=True)
