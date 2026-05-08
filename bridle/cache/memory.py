"""In-memory cache backend.

Process-local, threading-safe, optional TTL. The default backend when
nothing else is configured.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from . import MISS


class MemoryCache:
    """Simple ``dict``-backed cache with thread safety and TTL."""

    def __init__(self) -> None:
        self._store: dict[str, tuple[Any, float | None]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Any:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return MISS
            value, expires_at = entry
            if expires_at is not None and time.time() > expires_at:
                del self._store[key]
                return MISS
            return value

    def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        expires_at = (time.time() + ttl) if ttl is not None else None
        with self._lock:
            self._store[key] = (value, expires_at)

    def delete(self, key: str) -> None:
        with self._lock:
            self._store.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()


__all__ = ["MemoryCache"]
