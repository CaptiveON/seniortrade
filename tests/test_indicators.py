"""Behavioural tests for native indicators (no network)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.indicators import (
    EXPANDING,
    SQUEEZE,
    adx,
    atr,
    atr_percent,
    cvd_proxy,
    extension_atr,
    macd,
    obv,
    regime_label,
    rsi,
    true_range,
    volatility_regime,
)


def _df(open_, high, low, close, n):
    idx = pd.date_range("2024-01-01", periods=n, freq="1D", tz="UTC")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close,
         "volume": np.ones(n)},
        index=idx,
    )


def test_atr_of_constant_range_equals_that_range():
    n = 60
    df = _df(100.0, 101.0, 99.0, 100.0, n)  # flat bars, range = 2.0
    val = float(atr(df, 14).iloc[-1])
    assert abs(val - 2.0) < 1e-6


def test_true_range_uses_previous_close_gap():
    df = _df(
        open_=[100, 100],
        high=[101, 110],
        low=[99, 108],
        close=[100, 109],
        n=2,
    )
    tr = true_range(df)
    # Second bar gaps up: high(110) - prev_close(100) = 10 dominates the 2-wide range.
    assert abs(float(tr.iloc[-1]) - 10.0) < 1e-6


def test_atr_percent_scales_by_price():
    n = 40
    df = _df(50.0, 51.0, 49.0, 50.0, n)  # range 2 on price 50 -> 4%
    val = float(atr_percent(df, 14).iloc[-1])
    assert abs(val - 4.0) < 1e-6


def test_adx_high_on_strong_uptrend_and_direction_up():
    n = 120
    close = np.linspace(100, 300, n)        # clean, strong uptrend
    high = close + 1.0
    low = close - 1.0
    df = _df(close, high, low, close, n)
    out = adx(df, 14)
    assert float(out["adx"].iloc[-1]) > 25.0
    assert float(out["plus_di"].iloc[-1]) > float(out["minus_di"].iloc[-1])


def test_adx_low_on_choppy_market():
    n = 120
    base = 100.0
    # tight oscillation around a flat mean -> no directional structure
    close = base + np.array([0.5 if i % 2 == 0 else -0.5 for i in range(n)])
    high = close + 0.5
    low = close - 0.5
    df = _df(close, high, low, close, n)
    out = adx(df, 14)
    assert float(out["adx"].iloc[-1]) < 25.0


def test_regime_label_thresholds():
    assert regime_label(30.0, 25.0, 20.0) == "trending"
    assert regime_label(22.0, 25.0, 20.0) == "ranging"
    assert regime_label(10.0, 25.0, 20.0) == "choppy"
    assert regime_label(float("nan"), 25.0, 20.0) == "choppy"


def test_volatility_regime_expanding_when_range_grows():
    n = 60
    rng = np.linspace(1.0, 10.0, n)          # widening bars -> volatility rising
    close = np.full(n, 100.0)
    df = _df(close, close + rng / 2, close - rng / 2, close, n)
    label, pctl = volatility_regime(df, length=5, window=40)
    assert label == EXPANDING
    assert pctl >= 75.0


def test_volatility_regime_squeeze_when_range_shrinks():
    n = 60
    rng = np.linspace(10.0, 1.0, n)          # narrowing bars -> volatility falling
    close = np.full(n, 100.0)
    df = _df(close, close + rng / 2, close - rng / 2, close, n)
    label, pctl = volatility_regime(df, length=5, window=40)
    assert label == SQUEEZE
    assert pctl <= 25.0


def test_extension_atr_positive_far_above_ma():
    n = 60
    close = np.linspace(100, 200, n)          # steady climb -> price above its MA
    df = _df(close, close + 1, close - 1, close, n)
    ext = float(extension_atr(df, ma_len=20, atr_len=14).iloc[-1])
    assert ext > 1.0                          # price sits well above the mean (extended)


def test_extension_atr_near_zero_when_flat():
    n = 60
    close = np.full(n, 100.0)
    df = _df(close, close + 1, close - 1, close, n)
    ext = float(extension_atr(df, ma_len=20, atr_len=14).iloc[-1])
    assert abs(ext) < 0.5                      # at the mean -> ~0 ATRs of extension


def test_rsi_extremes():
    n = 40
    up = np.linspace(100, 140, n)
    down = np.linspace(140, 100, n)
    assert float(rsi(_df(up, up, up, up, n), 14).iloc[-1]) > 99.0
    assert float(rsi(_df(down, down, down, down, n), 14).iloc[-1]) < 1.0


def test_macd_sign_follows_trend():
    n = 60
    up = np.linspace(100, 200, n)
    down = np.linspace(200, 100, n)
    assert float(macd(_df(up, up, up, up, n))["macd"].iloc[-1]) > 0
    assert float(macd(_df(down, down, down, down, n))["macd"].iloc[-1]) < 0


def test_obv_rises_with_advancing_price():
    n = 30
    close = np.linspace(100, 130, n)
    df = _df(close, close, close, close, n)
    series = obv(df)
    assert float(series.iloc[-1]) > float(series.iloc[0])


def test_cvd_proxy_accumulates_when_closes_near_highs():
    n = 30
    close = np.full(n, 100.0)
    high = close + 1.0
    low = close - 1.0
    near_high = close + 0.9          # close sits near the high each bar -> accumulation
    df = _df(near_high, high, low, near_high, n)
    assert float(cvd_proxy(df).iloc[-1]) > 0
