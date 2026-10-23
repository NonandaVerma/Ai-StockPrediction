"""HTTP client with retries, per-host rate limiting, and an on-disk response cache."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .logging_utils import get_logger

try:  # Use the OS trust store (needed behind TLS-inspecting corporate proxies).
    import truststore

    truststore.inject_into_ssl()
except Exception:  # pragma: no cover - optional dependency
    pass

log = get_logger("http")
USER_AGENT = "Mozilla/5.0 (compatible; marketpred research client)"


class HttpClient:
    def __init__(
        self,
        cache_dir: Path,
        timeout_s: float = 90,
        max_retries: int = 5,
        backoff_factor: float = 1.5,
        min_interval_s: float = 1.0,
        cache_ttl_hours: float = 20,
    ) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout_s = timeout_s
        self.min_interval_s = min_interval_s
        self.cache_ttl_s = cache_ttl_hours * 3600
        self._last_call: dict[str, float] = {}
        self._lock = threading.Lock()
        retry = Retry(
            total=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET", "POST"),
            respect_retry_after_header=True,
        )
        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        self.session.mount("https://", HTTPAdapter(max_retries=retry))
        self.session.mount("http://", HTTPAdapter(max_retries=retry))

    def _cache_path(self, url: str, params: dict | None, body: Any) -> Path:
        key = json.dumps([url, params or {}, body], sort_keys=True, default=str)
        return self.cache_dir / (hashlib.sha256(key.encode()).hexdigest()[:24] + ".bin")

    def _throttle(self, url: str) -> None:
        host = urlparse(url).netloc
        with self._lock:
            wait = self._last_call.get(host, 0) + self.min_interval_s - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last_call[host] = time.monotonic()

    def get_bytes(self, url: str, params: dict | None = None, use_cache: bool = True,
                  method: str = "GET", json_body: Any = None) -> bytes:
        """Fetch a URL; cached responses younger than the TTL are reused without a network call."""
        path = self._cache_path(url, params, json_body)
        if use_cache and path.exists() and time.time() - path.stat().st_mtime < self.cache_ttl_s:
            return path.read_bytes()
        self._throttle(url)
        t0 = time.monotonic()
        resp = self.session.request(method, url, params=params, json=json_body, timeout=self.timeout_s)
        resp.raise_for_status()
        log.debug(f"{method} {url} {resp.status_code} {len(resp.content)}B {time.monotonic() - t0:.1f}s")
        path.write_bytes(resp.content)
        return resp.content

    def get_json(self, url: str, params: dict | None = None, **kw: Any) -> Any:
        return json.loads(self.get_bytes(url, params=params, **kw))
