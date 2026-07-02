"""Tests for the risk engine: management state machine, net costs, scaling, Kelly."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import Market, MarketsConfig, RiskConfig
from src.markets import LONG, make_market
from src.risk import (
    drawdown_scaled_risk,
    kelly_capped_risk,
    kelly_fraction,
    plan_trade,
    simulate,
)

MGMT = RiskConfig()


class FakeExchange:
    def __init__(self):
        spec = {"limits": {"amount": {"min": 0.0001, "max": None}, "cost": {"min": 5.0},
                           "leverage": {"max": 125}}, "contractSize": 1.0}
        self.markets = {"BTC/USDT": dict(spec), "BTC/USDT:USDT": dict(spec)}

    def price_to_precision(self, s, p): return f"{float(p):.2f}"
    def amount_to_precision(self, s, a): return f"{float(a):.4f}"
    def fetch_market_leverage_tiers(self, s): raise RuntimeError("offline")
    def fetch_leverage_tiers(self, syms): raise RuntimeError("offline")


def _bars(rows):
    idx = pd.date_range("2024-01-01", periods=len(rows), freq="1h", tz="UTC")
    return pd.DataFrame(rows, index=idx, columns=["open", "high", "low", "close"]).assign(volume=1.0)


# --- management state machine ---
def test_simulate_stop_hit():
    bars = _bars([[96, 97, 94, 95]])             # low pierces 95 stop
    r = simulate(LONG, 100, 95, [110, 120], bars, MGMT)
    assert r == pytest.approx(-1.0)


def test_simulate_gap_through_stop_fills_at_gap():
    bars = _bars([[90, 91, 89, 90]])             # opens below the stop -> gap fill
    r = simulate(LONG, 100, 95, [110, 120], bars, MGMT)
    assert r == pytest.approx(-2.0)              # (90-100)/5


def test_simulate_tp1_then_tp2():
    bars = _bars([[101, 111, 101, 110], [110, 121, 105, 120]])
    r = simulate(LONG, 100, 95, [110, 120], bars, MGMT)   # 0.5*2R + 0.5*4R
    assert r == pytest.approx(3.0)


def test_simulate_tp1_then_breakeven():
    bars = _bars([[101, 111, 101, 110], [100, 100, 99, 99]])  # TP1, then back to BE
    r = simulate(LONG, 100, 95, [110, 120], bars, MGMT)       # 0.5*2R + 0.5*0
    assert r == pytest.approx(1.0)


# --- drawdown scaling ---
def test_drawdown_scaled_risk():
    assert drawdown_scaled_risk(1.0, 0.0, MGMT) == pytest.approx(1.0)
    assert drawdown_scaled_risk(1.0, 20.0, MGMT) == pytest.approx(0.5)     # at full -> floor
    assert drawdown_scaled_risk(1.0, 12.5, MGMT) == pytest.approx(0.75)    # midpoint


# --- Kelly ---
def test_kelly_fraction_and_cap():
    assert kelly_fraction(0.6, 2.0, 1.0) == pytest.approx(0.4)   # 0.6 - 0.4/2
    off = RiskConfig(kelly_enabled=False)
    assert kelly_capped_risk(2.0, 0.6, 2.0, 1.0, off) == 2.0     # disabled -> base
    on = RiskConfig(kelly_enabled=True, kelly_fraction=0.25)
    assert kelly_capped_risk(2.0, 0.52, 1.0, 1.0, on) == pytest.approx(1.0)  # weak edge -> reduced
    assert kelly_capped_risk(2.0, 0.6, 2.0, 1.0, on) == 2.0      # strong edge -> capped at base


# --- plan_trade (net costs) ---
def test_plan_trade_net_below_gross_and_valid():
    m = make_market(Market.USDM, FakeExchange(), MarketsConfig())
    plan = plan_trade(m, symbol="BTC/USDT:USDT", side=LONG, entry=100, stop=95,
                      targets=[110, 120], account_equity=1000, risk_pct=2.0, mgmt=MGMT)
    assert plan.valid
    assert plan.risk_actual == pytest.approx(20.0)
    assert plan.gross_rr == pytest.approx(2.0)
    assert MGMT.min_net_rr <= plan.net_rr < plan.gross_rr     # costs eat into it
    assert plan.total_cost > 0


def test_plan_trade_thin_rr_rejected():
    m = make_market(Market.USDM, FakeExchange(), MarketsConfig())
    plan = plan_trade(m, symbol="BTC/USDT:USDT", side=LONG, entry=100, stop=95,
                      targets=[100.5, 101], account_equity=1000, risk_pct=2.0, mgmt=MGMT)
    assert not plan.valid
    assert any("net R:R" in n for n in plan.notes)
