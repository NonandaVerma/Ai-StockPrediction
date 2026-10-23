"""Step 4 - Time-series validation: locked holdout + walk-forward folds with embargo."""
import pandas as pd


def split_holdout(df: pd.DataFrame, years: float, purge: int = 0):
    """Development / locked-holdout split. The last `purge` development rows are dropped:
    their target windows (row t looks `purge` rows ahead) would otherwise reach into the
    holdout, and the final model would be fitted on holdout prices."""
    cutoff = df.index.max() - pd.DateOffset(days=int(365.25 * years))
    dev, hold = df[df.index <= cutoff], df[df.index > cutoff]
    return (dev.iloc[:len(dev) - purge] if purge else dev), hold


def effective_embargo(embargo: int, horizon: int) -> int:
    """An embargo shorter than the target horizon lets training labels overlap the test block."""
    return max(int(embargo), int(horizon))


def walk_forward_splits(n: int, n_splits: int, embargo: int, min_train_frac: float = 0.3):
    """Expanding-window splits. Yields (train_idx, test_idx) as integer ranges.
    The `embargo` rows before each test block are dropped from training so that
    multi-day targets cannot overlap the test period."""
    start_test = int(n * min_train_frac)
    fold = (n - start_test) // n_splits
    for k in range(n_splits):
        t0 = start_test + k * fold
        t1 = n if k == n_splits - 1 else t0 + fold
        train = range(0, max(0, t0 - embargo))
        test = range(t0, t1)
        yield train, test
