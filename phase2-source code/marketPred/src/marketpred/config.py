"""Configuration loading and validation (YAML + environment variables)."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"


class ProjectCfg(BaseModel):
    seed: int = 42
    data_dir: str = "data"
    artifacts_dir: str = "artifacts"


class MarketCfg(BaseModel):
    id: str
    name: str
    ticker: str
    calendar: str
    timezone: str
    currency: str
    start_date: str
    weather_location: dict[str, Any] | None = None
    cftc_contract_code: str | None = None
    vol_index: str = "^VIX"  # implied-volatility index of the target market


class HttpCfg(BaseModel):
    timeout_s: float = 90
    max_retries: int = 5
    backoff_factor: float = 1.5
    min_interval_s: float = 1.0
    cache_ttl_hours: float = 20


class FeaturesCfg(BaseModel):
    categories: list[str]
    winsorize_quantile: float = 0.005


class TargetsCfg(BaseModel):
    horizons: list[int]
    downside: dict[int, float]
    large_move: dict[str, float]
    regime: dict[str, float]
    vol_horizons: list[int]


class ValidationCfg(BaseModel):
    dev_start: str
    holdout_start: str
    retrain_every_days: int = 252
    embargo_days: int = 5
    min_train_obs: int = 1500
    calibration_fraction: float = 0.2

    @model_validator(mode="after")
    def _ordered(self) -> "ValidationCfg":
        if self.holdout_start <= self.dev_start:
            raise ValueError("holdout_start must be after dev_start")
        return self


class ComputeCfg(BaseModel):
    n_jobs: int = 1


class EvaluationCfg(BaseModel):
    bootstrap_reps: int = 500
    bootstrap_block: int = 20
    alpha: float = 0.05
    cost_bps_per_turnover: float = 5.0
    slippage_bps: float = 2.0


class ModelsCfg(BaseModel):
    classification: list[str]
    regression: list[str]
    volatility: list[str]
    regime: list[str]
    ablation_classification: list[str]
    ablation_regression: list[str]
    ablation_volatility: list[str]
    params: dict[str, dict[str, Any]] = Field(default_factory=dict)
    calibrate: list[str] = Field(default_factory=list)
    tuning: dict[str, dict[str, list[Any]]] = Field(default_factory=dict)


class Config(BaseModel):
    project: ProjectCfg
    market: MarketCfg
    sources: dict[str, dict[str, Any]]
    http: HttpCfg = HttpCfg()
    features: FeaturesCfg
    targets: TargetsCfg
    validation: ValidationCfg
    compute: ComputeCfg = ComputeCfg()
    evaluation: EvaluationCfg = EvaluationCfg()
    models: ModelsCfg

    @property
    def data_dir(self) -> Path:
        p = Path(os.environ.get("MARKETPRED_DATA_DIR") or self.project.data_dir)
        return p if p.is_absolute() else PROJECT_ROOT / p

    @property
    def artifacts_dir(self) -> Path:
        p = Path(os.environ.get("MARKETPRED_ARTIFACTS_DIR") or self.project.artifacts_dir)
        return p if p.is_absolute() else PROJECT_ROOT / p

    def source(self, name: str) -> dict[str, Any]:
        return self.sources.get(name, {})

    def fingerprint(self) -> str:
        """Stable hash of the full configuration, used for run/version metadata."""
        blob = json.dumps(self.model_dump(), sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:12]


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_dotenv(path: Path | None = None) -> None:
    """Minimal .env loader (never overrides variables already set in the environment)."""
    path = path or PROJECT_ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        if val.strip():
            os.environ.setdefault(key.strip(), val.strip())


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> Config:
    """Load the default config, deep-merged with an optional YAML file and a dict of overrides."""
    load_dotenv()
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    if path and Path(path).resolve() != DEFAULT_CONFIG.resolve():
        raw = _deep_merge(raw, yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})
    if overrides:
        raw = _deep_merge(raw, overrides)
    return Config.model_validate(raw)
