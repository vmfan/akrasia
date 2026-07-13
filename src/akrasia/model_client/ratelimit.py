"""Minimal blocking rate limiter (CLAUDE.md §6: "throttle/backoff rather than
hammer" the HKUST endpoint).

A fixed-interval limiter, not a bucket — simplest thing that guarantees no two
calls through one limiter are closer than ``60 / requests_per_minute`` seconds
apart, which is all the ~60 req/min HKUST constraint requires.
"""

from __future__ import annotations

import threading
import time


class RateLimiter:
    def __init__(self, requests_per_minute: float | None):
        self._min_interval = (
            60.0 / requests_per_minute if requests_per_minute else 0.0
        )
        self._lock = threading.Lock()
        self._last_call: float | None = None

    def wait(self) -> None:
        if self._min_interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            if self._last_call is not None:
                elapsed = now - self._last_call
                remaining = self._min_interval - elapsed
                if remaining > 0:
                    time.sleep(remaining)
            self._last_call = time.monotonic()
