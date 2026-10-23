"""Feature selection - pick a small, non-redundant, individually-informative subset.

Why this exists: a model given 300+ mostly-redundant columns and only a few thousand
training rows will fit noise. Fewer, stronger, less-overlapping features generally
generalise better (and are far easier to explain).

LEAKAGE RULE (the important part): every function here is fitted on TRAINING ROWS ONLY
and is called from inside each walk-forward fold. Choosing features once on the whole
dataset would let information from the test period influence the choice, which inflates
every score computed afterwards - the exact "accuracy" illusion this project is built
to avoid.

Three stages, matching standard feature-shortlisting practice:
  1. drop near-constant columns   - nothing to learn from them
  2. rank by univariate strength  - |AUC - 0.5| of each feature against the train target
  3. greedy correlation prune     - of any correlated pair, keep only the stronger one
"""
import numpy as np
from scipy.stats import rankdata


def univariate_auc(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Rank-based AUC of every column treated as a standalone classifier.

    Vectorised Mann-Whitney U, so all columns are scored in one pass (this runs inside
    every fold of every model, so per-column sklearn calls would be far too slow).
    Returns 0.5 for degenerate columns.
    """
    y = np.asarray(y).astype(bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return np.full(X.shape[1], 0.5)
    ranks = rankdata(X, axis=0)                      # average ranks handle ties correctly
    auc = (ranks[y].sum(axis=0) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return np.nan_to_num(auc, nan=0.5)


def select_features(X: np.ndarray, y: np.ndarray, max_features: int = 30,
                    corr_threshold: float = 0.95, min_edge: float = 0.0) -> np.ndarray:
    """Return column indices to keep. `X`/`y` must be TRAINING rows only.

    min_edge: require |AUC - 0.5| above this before a feature is eligible at all.
    Falls back to the strongest `max_features` columns if nothing clears the bar, so a
    fold can never end up with an empty feature set.
    """
    n_cols = X.shape[1]
    if n_cols <= 1:
        return np.arange(n_cols)

    # 1. drop near-constant columns
    std = np.nanstd(X, axis=0)
    usable = np.where(std > 1e-12)[0]
    if len(usable) == 0:
        return np.arange(min(n_cols, max_features))

    # 2. rank by univariate strength against the training target
    edge = np.abs(univariate_auc(X[:, usable], y) - 0.5)
    rank = np.argsort(-edge)
    order, ordered_edge = usable[rank], edge[rank]
    eligible = order[ordered_edge >= min_edge]
    if len(eligible) == 0:
        eligible = order

    # 3. greedy correlation prune, strongest first
    pool = eligible[:max(max_features * 8, 60)]      # cap the correlation matrix size
    corr = np.corrcoef(np.nan_to_num(X[:, pool]), rowvar=False)
    corr = np.atleast_2d(np.nan_to_num(corr))
    keep = []
    for i in range(len(pool)):
        if all(abs(corr[i, j]) < corr_threshold for j in keep):
            keep.append(i)
        if len(keep) >= max_features:
            break
    return pool[keep]
