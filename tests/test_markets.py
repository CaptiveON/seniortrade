"""Tests for market mechanics (sizing, leverage, liquidation) — no network."""
from __future__ import annotations

import pytest

from src.config import Market, MarketsConfig
from src.markets import (
    LONG,
    SHORT,
    SpotMarket,
    UsdmMarket,
    liquidation_price,
    make_market,
    safe_leverage,
)

CFG = MarketsConfig()


class FakeExchange:
    """Minimal ccxt stand-in: markets metadata + precision rounding, no network."""

    def __init__(self):
        spec = {
            "limits": {"amount": {"min": 0.0001, "max": None},
                       "cost": {"min": 5.0},
                       "leverage": {"max": 125}},
            "contractSize": 1.0,
        }
        self.markets = {"BTC/USDT": dict(spec), "BTC/USDT:USDT": dict(spec)}

    def price_to_precision(self, symbol, price):
        return f"{float(price):.2f}"

    def amount_to_precision(self, symbol, amount):
        return f"{float(amount):.4f}"

    def fetch_market_leverage_tiers(self, symbol):
        raise RuntimeError("offline")

    def fetch_leverage_tiers(self, symbols):
        raise RuntimeError("offline")


def test_liquidation_price_long_and_short():
    assert liquidation_price(100, 10, 0.005, LONG) == pytest.approx(90.5)
    assert liquidation_price(100, 10, 0.005, SHORT) == pytest.approx(109.5)


def test_safe_leverage_math():
    # stop 5% away: 1/(0.05*1.5 + 0.005) = 1/0.08 = 12.5 -> floor 12
    assert safe_leverage(100, 95, 0.005, 1.5, 20) == 12
    # wide 30% stop: 1/(0.30*1.5 + 0.005) = 1/0.455 = 2.19 -> floor 2
    assert safe_leverage(100, 70, 0.005, 1.5, 20) == 2


def test_spot_sizing_long():
    m = make_market(Market.SPOT, FakeExchange(), CFG)
    r = m.size_from_risk("BTC/USDT", entry=100, stop=95, risk_amount=10, side=LONG)
    assert r.valid
    assert r.size == pytest.approx(2.0)          # $10 / $5 per-unit
    assert r.notional == pytest.approx(200.0)
    assert r.leverage == 1.0
    assert r.margin == pytest.approx(200.0)      # spot pays full cost
    assert r.liquidation_price is None


def test_spot_short_rejected():
    m = make_market(Market.SPOT, FakeExchange(), CFG)
    r = m.size_from_risk("BTC/USDT", entry=100, stop=105, risk_amount=10, side=SHORT)
    assert not r.valid
    assert any("long-only" in n for n in r.notes)


def test_usdm_sizing_long_leverage_margin_liquidation():
    m = make_market(Market.USDM, FakeExchange(), CFG)
    r = m.size_from_risk("BTC/USDT:USDT", entry=100, stop=95, risk_amount=10, side=LONG)
    assert r.valid
    assert r.leverage == 3.0                     # default; safe(12) and cap(20) don't bind
    assert r.margin == pytest.approx(200.0 / 3.0)
    assert r.liquidation_price == pytest.approx(100 * (1 - 1 / 3 + 0.005))
    assert r.liq_distance_pct > r.stop_distance_pct   # liquidation safely beyond the stop


def test_usdm_leverage_capped_when_stop_is_wide():
    m = make_market(Market.USDM, FakeExchange(), CFG)
    r = m.size_from_risk("BTC/USDT:USDT", entry=100, stop=70, risk_amount=10,
                         side=LONG, leverage=10)
    assert r.leverage == 2.0                     # reduced from requested 10
    assert any("capped" in n for n in r.notes)
    assert r.liq_distance_pct > r.stop_distance_pct


def test_make_market_types():
    assert isinstance(make_market(Market.SPOT, FakeExchange(), CFG), SpotMarket)
    assert isinstance(make_market(Market.USDM, FakeExchange(), CFG), UsdmMarket)
