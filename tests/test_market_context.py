"""Tests for the market-context layer (BTC posture, relative strength, beta)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import Market
from src.market_context import (
    NEUTRAL,
    RISK_OFF,
    RISK_ON,
    assess_market,
    assign_groups,
    btc_reference_symbol,
    relative_strength,
)


def _df(close, high, low, n):
    idx = pd.date_range("2024-01-01", periods=n, freq="1D", tz="UTC")
    return pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close,
         "volume": np.ones(n)},
        index=idx,
    )


def test_btc_reference_symbol():
    assert btc_reference_symbol(Market.SPOT, "USDT") == "BTC/USDT"
    assert btc_reference_symbol(Market.USDM, "USDT") == "BTC/USDT:USDT"


def test_assess_market_risk_on_uptrend():
    n = 120
    close = np.linspace(100, 300, n)
    df = _df(close, close + 1, close - 1, n)
    ctx = assess_market(df, "BTC/USDT", 14, 25.0, 20.0)
    assert ctx.posture == RISK_ON
    assert ctx.trend == "up"


def test_assess_market_risk_off_downtrend():
    n = 120
    close = np.linspace(300, 100, n)
    df = _df(close, close + 1, close - 1, n)
    ctx = assess_market(df, "BTC/USDT", 14, 25.0, 20.0)
    assert ctx.posture == RISK_OFF
    assert ctx.trend == "down"


def test_assess_market_neutral_when_choppy():
    n = 120
    close = 100.0 + np.array([0.5 if i % 2 == 0 else -0.5 for i in range(n)])
    df = _df(close, close + 0.5, close - 0.5, n)
    ctx = assess_market(df, "BTC/USDT", 14, 25.0, 20.0)
    assert ctx.posture == NEUTRAL


def test_relative_strength_identical_series():
    rng = np.random.default_rng(0)
    rets = rng.normal(0.001, 0.02, 60)
    btc = pd.Series(100.0 * np.cumprod(1 + rets))
    coin = btc.copy()
    rs = relative_strength(coin, btc, window=40)
    assert abs(rs.rel_strength_pct) < 1e-6   # no out/under-performance vs itself
    assert rs.beta == pytest.approx(1.0)
    assert rs.corr == pytest.approx(1.0)


def test_relative_strength_leader_outperforms_with_higher_beta():
    rng = np.random.default_rng(1)
    btc_ret = rng.normal(0.002, 0.015, 60)               # BTC drifts up
    # A genuine leader: moves ~1.3x with BTC AND adds its own positive drift
    # (alpha) — higher beta alone wouldn't guarantee outperformance (vol drag).
    coin_ret = 1.3 * btc_ret + 0.003 + rng.normal(0, 0.001, 60)
    btc = pd.Series(100.0 * np.cumprod(1 + btc_ret))
    coin = pd.Series(100.0 * np.cumprod(1 + coin_ret))
    rs = relative_strength(coin, btc, window=50)
    assert rs.rel_strength_pct > 0          # outperformed BTC over the window
    assert 1.1 < rs.beta < 1.6              # ~1.3x sensitivity
    assert rs.corr > 0.9


def test_assign_groups_anchor_leads_and_independent_alone():
    n = 80
    rng = np.random.default_rng(3)
    base = np.cumsum(rng.normal(0, 1, n)) + 100.0
    closes = {
        "BTC/USDT": pd.Series(base),
        "AAA/USDT": pd.Series(base + rng.normal(0, 0.05, n)),   # tracks BTC
        "ZZZ/USDT": pd.Series(np.cumsum(rng.normal(0, 1, n)) + 100.0),  # independent
    }
    groups = assign_groups(
        closes, ["BTC/USDT"], {"AAA/USDT": 1e8, "ZZZ/USDT": 1e7},
        window=60, threshold=0.85,
    )
    # AAA joins the BTC-led group; the anchor wins leadership.
    assert groups["AAA/USDT"].leader == "BTC/USDT"
    assert groups["AAA/USDT"].size == 2
    # ZZZ is its own one-member group.
    assert groups["ZZZ/USDT"].leader == "ZZZ/USDT"
    assert groups["ZZZ/USDT"].size == 1
