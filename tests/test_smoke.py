"""Smoke tests — package imports and version is set."""

import bridle


def test_version() -> None:
    assert bridle.__version__ == "0.1.0"
