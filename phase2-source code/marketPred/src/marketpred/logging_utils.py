"""Structured (JSON-lines) logging to file plus human-readable console output."""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

_CONFIGURED = False


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        extra = getattr(record, "fields", None)
        if extra:
            payload.update(extra)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def setup_logging(log_dir: Path | None = None, level: str | None = None) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    level = level or os.environ.get("MARKETPRED_LOG_LEVEL", "INFO")
    root = logging.getLogger("marketpred")
    root.setLevel(level)
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S"))
    root.addHandler(console)
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / "marketpred.jsonl", encoding="utf-8")
        fh.setFormatter(JsonFormatter())
        root.addHandler(fh)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"marketpred.{name}")


def log_event(logger: logging.Logger, msg: str, **fields: object) -> None:
    """Log a message with structured key/value fields (kept as JSON keys in the file log)."""
    suffix = " ".join(f"{k}={v}" for k, v in fields.items())
    logger.info(f"{msg} {suffix}".strip(), extra={"fields": fields})
