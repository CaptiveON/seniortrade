"""Tests for OHLCV fetching — especially pagination past the exchange per-request cap."""
from __future__ import annotations

import pytest

from src.data_fetch import _MAX_CANDLES_PER_CALL, fetch_ohlcv


class _FakeEx:
    """Emulates Binance klines: any single fetch returns at most `cap` of the most-recent
    candles up to the request's endTime (default: now). Records how many calls were made."""

    def __init__(self, total: int, cap: int = _MAX_CANDLES_PER_CALL, tf_ms: int = 4 * 3600 * 1000,
                 start: int = 1_700_000_000_000):
        self.rows = [[start + i * tf_ms, 1.0, 2.0, 0.5, 1.5, 10.0] for i in range(total)]  # ascending ts
        self.cap = cap
        self.calls = 0

    def milliseconds(self) -> int:
        return self.rows[-1][0] + 1

    def fetch_ohlcv(self, symbol, timeframe=None, since=None, limit=None, params=None):
        self.calls += 1
        end = (params or {}).get("endTime", self.rows[-1][0])
        eligible = [r for r in self.rows if r[0] <= end]
        n = min(limit or self.cap, self.cap)
        return eligible[-n:] if eligible else []


def test_single_call_when_within_cap():
    ex = _FakeEx(total=2500)
    df = fetch_ohlcv(ex, "X/USDT:USDT", "4h", 800)
    assert len(df) == 800 and ex.calls == 1          # no pagination under the cap
    assert df.index.is_monotonic_increasing


def test_paginates_and_stitches_beyond_cap():
    ex = _FakeEx(total=2500)
    df = fetch_ohlcv(ex, "X/USDT:USDT", "4h", 2000)
    assert len(df) == 2000                            # honours the requested limit past the cap
    assert ex.calls >= 2                              # required more than one request
    assert df.index.is_monotonic_increasing          # sorted ascending
    assert df.index.duplicated().sum() == 0          # deduped across pages
    # returns the MOST-RECENT 2000 (ends at the latest candle)
    assert int(df["timestamp"].iloc[-1]) == ex.rows[-1][0]


def test_pagination_stops_when_history_exhausted():
    ex = _FakeEx(total=1500)                          # only 1500 exist, ask for 2000
    df = fetch_ohlcv(ex, "X/USDT:USDT", "4h", 2000)
    assert len(df) == 1500                            # all available, no infinite loop
    assert df.index.is_monotonic_increasing


def test_exact_cap_is_single_call():
    ex = _FakeEx(total=2500)
    df = fetch_ohlcv(ex, "X/USDT:USDT", "4h", _MAX_CANDLES_PER_CALL)
    assert len(df) == _MAX_CANDLES_PER_CALL and ex.calls == 1
