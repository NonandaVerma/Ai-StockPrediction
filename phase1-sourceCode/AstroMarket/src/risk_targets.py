"""Modelling-scale conversions for the risk pipeline's regression targets (realized volatility,
max drawdown), used by src/risk_eval.py so each target is fit on a well-behaved scale and
converted back for reporting."""
import numpy as np

# Modelling scale for each regression target (and its inverse).
DD_FLOOR = 0.005    # 0.5%: drawdowns smaller than this are treated as equivalent


def to_model_scale(name: str, y: np.ndarray) -> np.ndarray:
    if name == "realized_volatility_next_20d":
        return np.log(y)
    if name == "max_drawdown_next_20d":
        return np.log(DD_FLOOR + np.abs(y))
    raise ValueError(name)


def from_model_scale(name: str, z: np.ndarray) -> np.ndarray:
    if name == "realized_volatility_next_20d":
        return np.exp(z)
    if name == "max_drawdown_next_20d":
        return -np.maximum(np.exp(z) - DD_FLOOR, 0.0)
    raise ValueError(name)
