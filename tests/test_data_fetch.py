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


# --- candle sanitation (audit finding 6) ------------------------------------- #
import numpy as np
import pandas as pd

from src.data_fetch import DataError, sanitize_ohlcv


def _frame(rows, freq="4h"):
    idx = pd.date_range("2024-01-01", periods=len(rows), freq=freq, tz="UTC")
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"], index=idx)
    df.insert(0, "timestamp", (idx.view("int64") // 10**6))
    return df


def test_sanitize_drops_only_provably_broken_rows():
    rows = [
        [100, 101, 99, 100, 10],     # clean
        [100, 99, 101, 100, 10],     # high < low  -> broken
        [100, 101, 99, float("nan"), 10],   # NaN close -> broken
        [100, 101, 99, -5, 10],      # non-positive price -> broken
        [100, 100.5, 99, 102, 10],   # close above high (wick-inconsistent) -> broken
        [100, 101, 99, 100, -3],     # negative volume -> broken
        [100, 101, 99, 100.5, 10],   # clean
    ]
    out = sanitize_ohlcv(_frame(rows), "4h")
    q = out.attrs["quality"]
    assert len(out) == 2 and q["dropped"] == 5
    assert q["raw_rows"] == 7 and q["rows"] == 2


def test_sanitize_flags_gaps_and_zero_volume_without_dropping():
    rows = [[100, 101, 99, 100, 10], [100, 101, 99, 100, 0],
            [100, 101, 99, 100, 0], [100, 101, 99, 100, 10]]
    df = _frame(rows)
    df = df.drop(df.index[2])                      # create a 1-bar gap
    out = sanitize_ohlcv(df, "4h")
    q = out.attrs["quality"]
    assert len(out) == 3                           # nothing dropped
    assert q["gaps"] == 1 and q["zero_volume"] == 1


def test_sanitize_flags_suspect_spike_but_keeps_it():
    rows = [[100, 100.6, 99.4, 100, 10] for _ in range(30)]
    rows[15] = [100, 160, 40, 100.2, 10]           # 120-point range vs ~1.2 median, reverts next bar
    out = sanitize_ohlcv(_frame(rows), "4h")
    q = out.attrs["quality"]
    assert len(out) == 30                          # flagged, NOT dropped (a real crash must stay data)
    assert q["suspect_spikes"] == 1


def test_sanitize_clean_tape_untouched():
    rng = np.random.default_rng(3)
    close = 100 + rng.normal(0, 1, 200).cumsum()
    rows = [[c, c + abs(rng.normal(0, .4)) + .01, c - abs(rng.normal(0, .4)) - .01, c, 5.0] for c in close]
    out = sanitize_ohlcv(_frame(rows), "4h")
    q = out.attrs["quality"]
    assert len(out) == 200 and q["dropped"] == 0 and q["gaps"] == 0


def test_fetch_ohlcv_raises_on_fully_broken_tape():
    class _BrokenEx(_FakeEx):
        def fetch_ohlcv(self, symbol, timeframe=None, since=None, limit=None, params=None):
            self.calls += 1
            return [[1_700_000_000_000 + i * 14_400_000, 100, 99, 101, 100, 10] for i in range(50)]  # high<low
    with pytest.raises(DataError, match="sanitation"):
        fetch_ohlcv(_BrokenEx(total=1), "X/USDT:USDT", "4h", 50)
