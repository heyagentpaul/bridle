"""Redis cache backend — reserved for v2.

Importing this module is fine; instantiating :class:`RedisCache` raises so
users discover the gap loudly rather than mysteriously.
"""

from __future__ import annotations

from typing import Any


class RedisCache:
    """Placeholder; planned for v0.2.0 with an optional ``bridle[redis]`` extra."""

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise NotImplementedError(
            "Redis backend is reserved for v0.2.0. Use MemoryCache or FileCache "
            "in v0.1.0; see https://github.com/heyagentpaul/bridle for status."
        )


__all__ = ["RedisCache"]
