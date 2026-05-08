"""File-backed cache.

One pickle file per key, atomic writes via ``tempfile`` + ``os.replace``.
Survives process restart. Useful for local development runs that want
durability without standing up Redis.
"""

from __future__ import annotations

import os
import pickle
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from . import MISS


class FileCache:
    """Filesystem cache with atomic writes and optional TTL."""

    def __init__(self, path: str | Path) -> None:
        self._root = Path(path)
        self._root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _path_for(self, key: str) -> Path:
        # Keys come from ``default_cache_key`` (sha256-derived), so they're
        # filesystem-safe. Defensive: replace path separators just in case.
        safe = key.replace("/", "_").replace(os.sep, "_")
        return self._root / f"{safe}.pkl"

    def get(self, key: str) -> Any:
        path = self._path_for(key)
        if not path.exists():
            return MISS
        try:
            with path.open("rb") as fh:
                value, expires_at = pickle.load(fh)
        except (OSError, pickle.PickleError, EOFError):
            return MISS
        if expires_at is not None and time.time() > expires_at:
            self.delete(key)
            return MISS
        return value

    def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        path = self._path_for(key)
        expires_at = (time.time() + ttl) if ttl is not None else None
        with self._lock:
            with tempfile.NamedTemporaryFile(
                dir=str(self._root), delete=False, suffix=".tmp"
            ) as tmp:
                pickle.dump((value, expires_at), tmp)
                tmp_name = tmp.name
            os.replace(tmp_name, path)

    def delete(self, key: str) -> None:
        import contextlib

        path = self._path_for(key)
        with contextlib.suppress(FileNotFoundError):
            path.unlink()


__all__ = ["FileCache"]
