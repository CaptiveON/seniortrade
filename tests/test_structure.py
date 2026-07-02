"""Tests for expert-grade structure detection (no network)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import StructureConfig
from src.structure import (
    BUY_LIQ,
    DOWN,
    HIGH,
    LOW,
    RESISTANCE,
    SUPPORT,
    UP,
    Level,
    Swing,
    analyze_structure,
    classify_structure,
    cluster_levels,
    find_swings,
    prune_levels,
    range_edges,
    retracement_levels,
    round_levels,
    volume_profile,
)


def _ohlc(close, high=None, low=None, volume=None):
    n = len(close)
    idx = pd.date_range("2024-01-01", periods=n, freq="1D", tz="UTC")
    return pd.DataFrame(
        {
            "open": close,
            "high": list(close) if high is None else high,
            "low": list(close) if low is None else low,
            "close": close,
            "volume": [1.0] * n if volume is None else volume,
        },
        index=idx,
    )


def test_zigzag_finds_alternating_swings_above_threshold():
    close = [100, 102, 104, 106, 108, 110, 108, 106, 104, 102, 100, 102, 104, 106, 108, 110, 112]
    df = _ohlc(close)
    swings = find_swings(df, atr_values=1.0, reversal_atr=2.0)
    # A confirmed high at 110 and a confirmed low at 100; the final up-leg is unconfirmed.
    assert any(s.kind == HIGH and abs(s.price - 110) < 1e-6 for s in swings)
    assert any(s.kind == LOW and abs(s.price - 100) < 1e-6 for s in swings)
    # a meaningful ~10-ATR leg was measured between swings
    assert max(s.leg_atr for s in swings) > 5.0


def test_classify_structure_uptrend_and_choch():
    swings = [
        Swing(0, None, 100.0, LOW),
        Swing(1, None, 110.0, HIGH),
        Swing(2, None, 105.0, LOW),    # higher low
        Swing(3, None, 115.0, HIGH),   # higher high
    ]
    up = classify_structure(swings, last_price=113.0)
    assert up.trend == UP
    assert up.event == "none"
    # break below the last higher-low -> change of character (bearish)
    choch = classify_structure(swings, last_price=104.0)
    assert choch.event == "CHoCH_down"


def test_classify_structure_downtrend():
    swings = [
        Swing(0, None, 120.0, HIGH),
        Swing(1, None, 110.0, LOW),
        Swing(2, None, 115.0, HIGH),   # lower high
        Swing(3, None, 105.0, LOW),    # lower low
    ]
    st = classify_structure(swings, last_price=107.0)
    assert st.trend == DOWN


def test_cluster_levels_side_and_kind():
    swings = [Swing(0, None, 110.0, HIGH, leg_atr=5.0),
              Swing(1, None, 110.1, HIGH, leg_atr=4.0)]
    levels = cluster_levels(swings, tolerance=0.5, last_price=100.0,
                            n_bars=10, recency_window=120)
    assert len(levels) == 1
    lv = levels[0]
    assert lv.touches == 2
    assert lv.side == BUY_LIQ        # made of swing highs
    assert lv.kind == RESISTANCE     # above last price


def test_prune_levels_separation_and_cap():
    levels = [
        Level(price=100.0, touches=3, kind=SUPPORT, side="sell_liq", last_touch_index=5, strength=3.0),
        Level(price=100.2, touches=1, kind=SUPPORT, side="sell_liq", last_touch_index=4, strength=2.0),
        Level(price=110.0, touches=2, kind=RESISTANCE, side="buy_liq", last_touch_index=6, strength=4.0),
    ]
    pruned = prune_levels(levels, min_separation=1.0, max_levels=8)
    prices = sorted(round(lv.price, 1) for lv in pruned)
    assert prices == [100.0, 110.0]   # the 100.2 (weaker, within 1.0 of 100) is dropped


def test_retracement_levels_up_leg():
    swings = [Swing(0, None, 100.0, LOW), Swing(1, None, 110.0, HIGH)]
    retr, measured = retracement_levels(swings)
    assert abs(retr["0.618"] - 103.82) < 0.01     # 110 - 0.618*10
    assert abs(measured - 120.0) < 1e-6           # measured move up = hi + span


def test_round_levels_scale_to_magnitude():
    levels = round_levels(0.6, span_pct=20)
    assert any(abs(v - 0.6) < 1e-9 for v in levels)
    assert all(0.6 * 0.8 - 1e-9 <= v <= 0.6 * 1.2 + 1e-9 for v in levels)


def test_range_edges():
    df = _ohlc([5, 7, 3, 6, 4], high=[5, 9, 3, 6, 4], low=[5, 7, 1, 6, 4])
    hi, lo = range_edges(df, window=5)
    assert hi == 9.0 and lo == 1.0


def test_range_edges_excludes_current_bar():
    # the current (last) bar makes a fresh extreme — it must NOT be inside the range,
    # so detectors can see it sweep/break the prior range (regression: failed_breakout was dead).
    df = _ohlc([10, 11, 9, 10, 8], high=[10, 12, 9, 10, 8], low=[9, 11, 8, 10, 5])
    hi, lo = range_edges(df, window=4)
    assert lo == 8.0                       # the PRIOR min, not the last bar's 5
    assert hi == 12.0
    assert float(df["low"].iloc[-1]) < lo  # so the current bar genuinely sweeps below the range


def test_volume_profile_poc_at_high_volume_price():
    close = [105] * 10 + [100, 110, 101, 109, 102, 108, 103, 107, 104, 106]
    high = [c + 0.5 for c in close]
    low = [c - 0.5 for c in close]
    volume = [100.0] * 10 + [1.0] * 10     # volume concentrated at ~105
    df = _ohlc(close, high=high, low=low, volume=volume)
    vp = volume_profile(df, window=20, bins=40, value_area_pct=70)
    assert vp is not None
    assert abs(vp.poc - 105.0) < 1.0
    assert vp.val <= vp.poc <= vp.vah


def test_analyze_structure_integration():
    # ~5 oscillations between 100 and 110 with volume -> levels + state + profile
    one = [100, 103, 106, 109, 110, 107, 104, 101, 100]
    close = (one * 5) + [102, 105]
    df = _ohlc(close, high=[c + 0.5 for c in close], low=[c - 0.5 for c in close],
               volume=[10.0] * len(close))
    s = analyze_structure(df, StructureConfig(range_window=30, vp_window=40))
    assert s.state is not None
    assert s.levels                              # found at least one key level
    assert s.volume_profile is not None
    assert len(s.levels) <= StructureConfig().max_levels
    # combined stop/target primitives still work
    assert s.support_below(s.last_price) is None or s.support_below(s.last_price) <= s.last_price
    assert s.resistance_above(s.last_price) is None or s.resistance_above(s.last_price) >= s.last_price
