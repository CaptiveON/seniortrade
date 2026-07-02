"""Tests for setup detection (pure, look-ahead-safe) — no network."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import Market, SetupConfig
from src import structure as st
from src.markets import LONG, SHORT
from src.setups import (
    SetupContext,
    detect_all,
    detect_breakout_momentum,
    detect_breakout_retest,
    detect_choch_reversal,
    detect_divergence_reversal,
    detect_failed_breakout,
    detect_liquidity_sweep_reversal,
    detect_momentum_flag,
    detect_range_fade,
    detect_squeeze_breakout,
    detect_trend_pullback,
)

CFG = SetupConfig()
CTX_UP = SetupContext(htf_trend=st.UP, tide="neutral")
CTX_RANGE = SetupContext(htf_trend=st.RANGE, tide="neutral")


def _bars(open_, high, low, close):
    n = len(close)
    idx = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                         "volume": np.ones(n)}, index=idx)


def _level(price, kind):
    return st.Level(price=price, touches=2, kind=kind, side="mixed",
                    last_touch_index=10, strength=2.0)


def _structure(last, trend, levels, range_low, range_high, last_low=np.nan, last_high=np.nan):
    state = st.StructureState(trend=trend, event="none",
                              last_high=last_high, last_low=last_low, sequence=["HL", "HH", "HL"])
    return st.Structure(last_price=last, levels=levels, state=state,
                        range_low=range_low, range_high=range_high)


def _uptrend_bars(last_o, last_h, last_l, last_c, n=25):
    close = [100.0 + i for i in range(n - 1)] + [last_c]
    open_ = [c - 0.5 for c in close[:-1]] + [last_o]
    high = [c + 0.3 for c in close[:-1]] + [last_h]
    low = [c - 0.3 for c in close[:-1]] + [last_l]
    return _bars(open_, high, low, close)


def test_trend_pullback_long_grade_a():
    bars = _uptrend_bars(last_o=120.2, last_h=121.5, last_l=119.8, last_c=121.0)
    struct = _structure(121.0, st.UP, [_level(120.0, st.SUPPORT), _level(130.0, st.RESISTANCE)],
                        range_low=100.0, range_high=131.0, last_low=119.0)
    sig = detect_trend_pullback(bars, struct, CTX_UP, CFG, Market.USDM)
    assert sig is not None
    assert sig.setup == "trend_pullback" and sig.direction == LONG
    assert sig.stop < sig.entry < sig.targets[0]
    assert sig.rr > 1.2
    assert sig.grade == "A"            # rr>=2, 3 confluence reasons, with-tide + HTF aligned


def test_spot_has_no_short_pullback():
    # downtrend pullback is a short -> not allowed on spot
    close = [124.0 - i for i in range(24)] + [101.0]
    bars = _bars([c + 0.5 for c in close[:-1]] + [101.8],
                 [c + 0.3 for c in close[:-1]] + [102.2],
                 [c - 0.3 for c in close[:-1]] + [100.5], close)
    struct = _structure(101.0, st.DOWN, [_level(102.0, st.RESISTANCE)],
                        range_low=90.0, range_high=125.0, last_high=103.0)
    assert detect_trend_pullback(bars, struct, CTX_RANGE, CFG, Market.SPOT) is None


def test_range_fade_long():
    close = [110.0, 100.5, 119.0, 101.0, 118.0, 102.0] * 4 + [101.0]
    bars = _bars([c + 0.2 for c in close[:-1]] + [100.5],
                 [c + 0.5 for c in close[:-1]] + [101.5],
                 [c - 0.5 for c in close[:-1]] + [100.1], close)
    struct = _structure(101.0, st.RANGE, [], range_low=100.0, range_high=120.0)
    sig = detect_range_fade(bars, struct, CTX_RANGE, CFG, Market.USDM)
    assert sig is not None and sig.setup == "range_fade" and sig.direction == LONG
    assert abs(sig.targets[0] - 110.0) < 1e-6     # mid of the range
    assert sig.entry_type == "limit"


def test_failed_breakout_long_spring():
    close = [105.0 + (i % 3) for i in range(24)] + [101.0]
    bars = _bars([c for c in close[:-1]] + [99.0],
                 [c + 0.5 for c in close[:-1]] + [101.5],
                 [c - 0.5 for c in close[:-1]] + [98.0], close)   # last bar swept 98 < range_low, reclaimed
    struct = _structure(101.0, st.RANGE, [], range_low=100.0, range_high=120.0)
    sig = detect_failed_breakout(bars, struct, CTX_RANGE, CFG, Market.USDM)
    assert sig is not None and sig.setup == "failed_breakout" and sig.direction == LONG
    assert sig.stop < 100.0                         # below the sweep wick


def test_breakout_retest_long():
    # recent closes dipped below 110 then above -> a genuine break; last bar retests 110
    close = [108.0, 109.0, 112.0, 113.0, 111.0, 114.0] * 4 + [111.0]
    bars = _bars([c - 0.3 for c in close[:-1]] + [110.5],
                 [c + 0.3 for c in close[:-1]] + [111.5],
                 [c - 0.3 for c in close[:-1]] + [109.8], close)
    struct = _structure(111.0, st.UP, [_level(110.0, st.SUPPORT), _level(125.0, st.RESISTANCE)],
                        range_low=105.0, range_high=126.0)
    sig = detect_breakout_retest(bars, struct, CTX_UP, CFG, Market.USDM)
    assert sig is not None and sig.setup == "breakout_retest" and sig.direction == LONG
    assert sig.entry_type == "limit" and abs(sig.entry - 110.0) < 1e-6


def test_breakout_momentum_long_on_decisive_break():
    # a steep, clean uptrend (high ADX) then a bar that decisively clears the prior high
    close = [100.0 + 1.2 * i for i in range(29)] + [139.0]
    bars = _bars([c - 0.6 for c in close[:-1]] + [134.0],
                 [c + 0.4 for c in close[:-1]] + [139.5],
                 [c - 0.4 for c in close[:-1]] + [133.9], close)
    struct = _structure(139.0, st.UP, [], range_low=100.0, range_high=160.0)
    sig = detect_breakout_momentum(bars, struct, CTX_UP, CFG, Market.USDM)
    assert sig is not None and sig.setup == "breakout_momentum" and sig.direction == LONG
    assert sig.entry_type == "stop"                 # enter ON continuation, not a retest
    assert sig.stop < sig.entry                      # stop back inside the broken range


def test_breakout_momentum_skips_choppy_regime():
    # an oscillating range -> low ADX -> no momentum breakout even if a bar pokes out
    close = [110.0 + (8.0 if i % 2 else -8.0) for i in range(29)] + [121.0]
    bars = _bars([c for c in close[:-1]] + [112.0],
                 [c + 1.0 for c in close[:-1]] + [121.5],
                 [c - 1.0 for c in close[:-1]] + [111.5], close)
    struct = _structure(121.0, st.RANGE, [], range_low=100.0, range_high=120.0)
    assert detect_breakout_momentum(bars, struct, CTX_RANGE, CFG, Market.USDM) is None


def test_liquidity_sweep_reversal_long_and_short():
    base = [105.0 + (i % 3) for i in range(20)]
    # LONG: wick below a sell-side pool at 100, close back above with a big lower wick
    long_bars = _bars([c for c in base] + [100.5], [c + 0.5 for c in base] + [101.6],
                      [c - 0.5 for c in base] + [98.0], base + [101.2])
    s_long = st.Structure(last_price=101.2, levels=[], range_low=100.0, range_high=120.0,
                          sell_side_liquidity=[100.0],
                          state=st.StructureState(st.RANGE, "none", np.nan, np.nan, []))
    sig = detect_liquidity_sweep_reversal(long_bars, s_long, CTX_RANGE, CFG, Market.USDM)
    assert sig is not None and sig.direction == LONG and sig.stop < 98.0

    # SHORT (usdm): wick above a buy-side pool at 120, reject back below
    short_bars = _bars([c for c in base] + [119.5], [c + 0.5 for c in base] + [122.0],
                       [c - 0.5 for c in base] + [118.5], base + [119.0])
    s_short = st.Structure(last_price=119.0, levels=[], range_low=100.0, range_high=120.0,
                           buy_side_liquidity=[120.0],
                           state=st.StructureState(st.RANGE, "none", np.nan, np.nan, []))
    sig2 = detect_liquidity_sweep_reversal(short_bars, s_short, CTX_RANGE, CFG, Market.USDM)
    assert sig2 is not None and sig2.direction == SHORT and sig2.stop > 122.0


def test_momentum_flag_bull():
    close = [100.0 + 1.2 * i for i in range(21)] + [122.5, 123.0, 122.8]
    bars = _bars([c - 0.6 for c in close[:21]] + [122.6, 122.5, 123.0],
                 [c + 0.4 for c in close[:21]] + [123.2, 123.5, 123.3],
                 [c - 0.4 for c in close[:21]] + [122.0, 122.4, 122.2], close)
    swings = [st.Swing(0, None, 100.0, st.LOW, 0.0, False),
              st.Swing(20, None, 124.0, st.HIGH, 3.0, True)]
    struct = st.Structure(last_price=122.8, swings=swings, levels=[], range_low=100.0, range_high=150.0,
                          state=st.StructureState(st.UP, "none", 124.0, 100.0, ["HL", "HH"]))
    sig = detect_momentum_flag(bars, struct, CTX_UP, CFG, Market.USDM)
    assert sig is not None and sig.setup == "momentum_flag" and sig.direction == LONG
    assert sig.entry_type == "stop" and sig.targets[0] > sig.entry and sig.stop < sig.entry


def test_squeeze_breakout_long_from_a_coil():
    # a volatile history (high ATR), then a tight coil (low ATR%ile), then a release up
    vol = [100.0 + (5.0 if i % 2 else -5.0) for i in range(90)]   # big bar-to-bar swings
    coil = [100.0 + (0.2 if i % 2 else -0.2) for i in range(30)]  # tight coil ~100
    close = vol + coil + [103.0]                                  # breakout bar
    wide = [4.0] * 90 + [0.3] * 30 + [0.5]                        # half-range per bar
    bars = _bars([c - 0.1 for c in close[:-1]] + [100.2],
                 [c + w for c, w in zip(close, wide)],
                 [c - w for c, w in zip(close, wide)], close)
    struct = _structure(103.0, st.RANGE, [], range_low=90.0, range_high=115.0)
    sig = detect_squeeze_breakout(bars, struct, CTX_RANGE, CFG, Market.USDM)
    assert sig is not None and sig.setup == "squeeze_breakout" and sig.direction == LONG
    assert sig.stop < 100.0                                       # back inside / below the coil


def _state(event, last_high, last_low, trend=st.RANGE):
    return st.StructureState(trend=trend, event=event, last_high=last_high,
                             last_low=last_low, sequence=["LH", "LL"])


def test_choch_reversal_long_and_short():
    bars_up = _bars([110.0] * 19 + [110.0], [c + 0.5 for c in [110.0] * 20],
                    [c - 0.5 for c in [110.0] * 20], [110.0] * 19 + [111.0])
    s_up = st.Structure(last_price=111.0, levels=[], range_low=95.0, range_high=130.0,
                        state=_state("CHoCH_up", last_high=110.0, last_low=100.0))
    sig = detect_choch_reversal(bars_up, s_up, CTX_RANGE, CFG, Market.USDM)
    assert sig is not None and sig.setup == "choch_reversal" and sig.direction == LONG
    assert sig.stop < 100.0 and sig.targets[0] > sig.entry

    bars_dn = _bars([100.0] * 19 + [100.5], [c + 0.5 for c in [100.0] * 20],
                    [c - 0.5 for c in [100.0] * 20], [100.0] * 19 + [99.0])
    s_dn = st.Structure(last_price=99.0, levels=[], range_low=80.0, range_high=110.0,
                        state=_state("CHoCH_down", last_high=110.0, last_low=100.0))
    sig2 = detect_choch_reversal(bars_dn, s_dn, CTX_RANGE, CFG, Market.USDM)
    assert sig2 is not None and sig2.direction == SHORT and sig2.stop > 110.0


def test_divergence_reversal_bearish_short():
    # strong rally (high RSI at the first high, index >= rsi length) then a choppy,
    # marginally-higher second high (weaker momentum -> lower RSI) -> bearish divergence
    rally = [100.0 + 2.0 * i for i in range(17)]                  # idx 0..16, RSI very high at 16
    chop = ([131.0, 132.0] * 8) + [131.5]                         # idx 17..33, RSI fades to ~50
    close = rally + chop
    bars = _bars([c - 0.3 for c in close[:-1]] + [132.5],         # last bar opens 132.5 > close (down bar)
                 [c + 0.4 for c in close[:-1]] + [132.6],
                 [c - 0.4 for c in close[:-1]] + [131.4], close)
    swings = [st.Swing(16, None, 132.0, st.HIGH, 0.0, False),
              st.Swing(33, None, 132.5, st.HIGH, 0.0, False)]
    struct = st.Structure(last_price=131.5, swings=swings, levels=[], range_low=100.0, range_high=140.0,
                          state=_state("none", last_high=132.5, last_low=131.0))
    sig = detect_divergence_reversal(bars, struct, CTX_RANGE, CFG, Market.USDM)
    assert sig is not None and sig.setup == "divergence_reversal" and sig.direction == SHORT
    assert sig.stop > 132.5                                       # beyond the divergent high


def test_detect_all_returns_ranked_list():
    bars = _uptrend_bars(last_o=120.2, last_h=121.5, last_l=119.8, last_c=121.0)
    struct = _structure(121.0, st.UP, [_level(120.0, st.SUPPORT), _level(130.0, st.RESISTANCE)],
                        range_low=100.0, range_high=131.0, last_low=119.0)
    sigs = detect_all(bars, struct, CTX_UP, CFG, Market.USDM)
    assert any(s.setup == "trend_pullback" for s in sigs)
    grades = [s.grade for s in sigs]
    assert grades == sorted(grades, key=lambda g: {"A": 0, "B": 1, "C": 2}[g])
