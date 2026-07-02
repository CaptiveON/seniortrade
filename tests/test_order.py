"""Tests for order-staging pure helpers (ticket, drift/validity/duplicate)."""
from __future__ import annotations

from types import SimpleNamespace

from src.config import Market
from src.order import (
    LIMIT,
    MARKET,
    build_proposed,
    build_ticket,
    check_drift,
    check_validity,
    has_open_or_staged,
)


def _plan():
    return SimpleNamespace(symbol="SOL/USDT:USDT", entry=72.0, stop=73.0, targets=[70.0, 68.0],
                           liquidation_price=95.0, leverage=3.0, risk_actual=10.0, net_rr=2.2)


def _signal(direction="short", entry_type=LIMIT, invalidation=73.0):
    return SimpleNamespace(direction=direction, entry_type=entry_type, invalidation=invalidation,
                           setup="breakout_retest", grade="A")


def test_build_ticket_usdm_short():
    t = build_ticket(_plan(), _signal(), Market.USDM, tp1_fraction=0.5)
    kinds = {leg.kind: leg for leg in t.legs}
    assert t.leverage == 3.0 and t.margin_mode == "isolated"
    assert kinds["entry"].side == "sell" and not kinds["entry"].reduce_only
    assert kinds["stop"].side == "buy" and kinds["stop"].reduce_only and kinds["stop"].order_type == "stop_market"
    assert kinds["tp1"].qty_fraction == 0.5 and kinds["tp2"].qty_fraction == 0.5
    assert all(leg.reduce_only for leg in (kinds["stop"], kinds["tp1"], kinds["tp2"]))


def test_build_ticket_spot_long_no_leverage():
    plan = SimpleNamespace(symbol="BTC/USDT", entry=60000, stop=58000, targets=[64000, 66000],
                           liquidation_price=None, leverage=1.0, risk_actual=10.0)
    t = build_ticket(plan, _signal(direction="long", invalidation=58000), Market.SPOT, 0.5)
    assert t.leverage is None and t.margin_mode is None
    assert {leg.kind for leg in t.legs} == {"entry", "stop", "tp1", "tp2"}
    assert t.legs[0].side == "buy"


def test_check_drift_market_vs_limit():
    ok, _ = check_drift(MARKET, 100.0, 100.5, drift_pct=1.0)
    assert ok                                         # 0.5% within 1%
    bad, _ = check_drift(MARKET, 100.0, 102.0, drift_pct=1.0)
    assert not bad                                    # 2% drift
    rest, _ = check_drift(LIMIT, 100.0, 120.0, drift_pct=1.0)
    assert rest                                       # limit rests at its level — no drift race


def test_check_validity_invalidation_hit():
    long_ok, _ = check_validity(_signal(direction="long", invalidation=95.0), 100.0)
    assert long_ok
    long_void, _ = check_validity(_signal(direction="long", invalidation=95.0), 94.0)
    assert not long_void                              # price already below the invalidation
    short_void, _ = check_validity(_signal(direction="short", invalidation=73.0), 74.0)
    assert not short_void


def test_has_open_or_staged():
    recs = [SimpleNamespace(symbol="SOL/USDT:USDT", status="open"),
            SimpleNamespace(symbol="ETH/USDT:USDT", status="closed")]
    assert has_open_or_staged(recs, "SOL/USDT:USDT")
    assert not has_open_or_staged(recs, "ETH/USDT:USDT")     # closed doesn't block
    assert not has_open_or_staged(recs, "BTC/USDT:USDT")


def test_build_proposed_maps_fields():
    p = build_proposed(_plan(), _signal(), group="indep", risk_pct=1.0)
    assert p.symbol == "SOL/USDT:USDT" and p.side == "short"
    assert p.entry == 72.0 and p.stop == 73.0 and p.liquidation == 95.0
