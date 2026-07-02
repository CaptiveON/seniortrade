"""Tests for watch's bar-close-aware timing (the two-cadence radar)."""
from __future__ import annotations

import time

from src.cli import _next_bar_close_ts, _tf_seconds


def test_tf_seconds():
    assert _tf_seconds("4h") == 14400
    assert _tf_seconds("1h") == 3600
    assert _tf_seconds("15m") == 900
    assert _tf_seconds("1d") == 86400
    assert _tf_seconds("bogus") == 14400          # safe fallback


def test_next_bar_close_is_aligned_and_within_one_bar():
    now = time.time()
    for tf_s in (900, 3600, 14400, 86400):
        nb = _next_bar_close_ts(tf_s)
        assert nb > now                           # always in the future
        assert nb % tf_s == 0                     # aligned to the UTC candle boundary
        assert nb - now <= tf_s                   # never more than one bar away
