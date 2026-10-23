"""Platt scaling fitted on a purged, chronologically later slice of the training window."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

EPS = 1e-6


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


class PlattCalibrator:
    def fit(self, p_raw: np.ndarray, y: np.ndarray) -> "PlattCalibrator":
        self.lr_ = LogisticRegression(C=1e4).fit(logit(np.asarray(p_raw)).reshape(-1, 1), np.asarray(y).astype(int))
        return self

    def transform(self, p_raw: np.ndarray) -> np.ndarray:
        return self.lr_.predict_proba(logit(np.asarray(p_raw)).reshape(-1, 1))[:, 1]
