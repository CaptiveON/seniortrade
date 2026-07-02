"""Tests for the portfolio/behaviour guards (no network)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.config import GuardsConfig
from src.guards import (
    OpenPosition,
    PortfolioState,
    ProposedTrade,
    allocate_basket,
    check_trade,
    group_heat_usd,
    _in_funding_window,
)
from src.markets import LONG, SHORT

CFG = GuardsConfig()


def _items(n, group="indep", score=1.0):
    return [dict(symbol=f"C{i}", group=group, side="long", score=score) for i in range(n)]


def test_allocate_equal_risk_within_cap():
    allocs = allocate_basket(_items(3), 1000.0, CFG, base_risk_pct=1.0)
    assert len(allocs) == 3
    assert all(abs(a.risk_pct - 1.0) < 1e-9 for a in allocs)        # 3% heat < 6% cap → unscaled
    assert all(abs(a.risk_amount - 10.0) < 1e-9 for a in allocs)


def test_allocate_scales_whole_basket_to_heat_cap():
    allocs = allocate_basket(_items(5), 1000.0, CFG, base_risk_pct=2.0)   # 5×2% = 10% > 6%
    assert abs(sum(a.risk_pct for a in allocs) - CFG.heat_cap_pct) < 1e-6  # scaled to the cap
    assert all(abs(a.risk_pct - 1.2) < 1e-6 for a in allocs)


def test_allocate_caps_positions_and_per_group():
    assert len(allocate_basket(_items(10), 1000.0, CFG, 1.0)) == CFG.max_positions
    assert len(allocate_basket(_items(4, group="BTC"), 1000.0, CFG, 1.0)) == CFG.max_per_group


def test_allocate_tilt_weights_by_edge():
    items = [dict(symbol="HI", group="indep", side="long", score=3.0),
             dict(symbol="LO", group="indep", side="long", score=1.0)]
    a = {x.symbol: x for x in allocate_basket(items, 1000.0, CFG, base_risk_pct=5.0, tilt=True)}
    assert a["HI"].risk_pct > a["LO"].risk_pct          # more risk to the stronger edge
    assert a["HI"].risk_pct <= 5.0                       # never above base
QUIET = datetime(2024, 1, 1, 3, 30, tzinfo=timezone.utc)   # far from any funding window


def _state(**kw):
    base = dict(equity=1000.0)
    base.update(kw)
    return PortfolioState(**base)


def _long(risk_pct=1.0, group="indep"):
    return ProposedTrade(symbol="X/USDT:USDT", group=group, side=LONG,
                         risk_pct=risk_pct, entry=100.0, stop=95.0)


def test_clean_trade_allowed():
    r = check_trade(_state(), _long(), CFG, now=QUIET)
    assert r.allowed and not r.hard_blocks
    assert r.heat_pct == pytest.approx(1.0)


def test_sanity_blocks_bad_geometry():
    bad = ProposedTrade("X", "indep", LONG, 1.0, entry=100.0, stop=100.0)
    assert not check_trade(_state(), bad, CFG, now=QUIET).allowed


def test_daily_loss_lockout():
    r = check_trade(_state(realized_r_today=-3.0), _long(), CFG, now=QUIET)
    assert not r.allowed
    assert any("daily-loss" in h for h in r.hard_blocks)


def test_cooldown_recent_loss_and_streak():
    recent = check_trade(_state(last_loss_time=QUIET - timedelta(minutes=10)), _long(), CFG, now=QUIET)
    assert not recent.allowed and any("cooldown" in h for h in recent.hard_blocks)
    streak = check_trade(_state(consecutive_losses=3), _long(), CFG, now=QUIET)
    assert not streak.allowed


def test_max_trades_per_day_guard():
    from dataclasses import replace
    cfg = replace(CFG, max_trades_per_day=3)
    blocked = check_trade(_state(trades_today=3), _long(), cfg, now=QUIET)
    assert not blocked.allowed and any("max trades/day" in h for h in blocked.hard_blocks)
    assert check_trade(_state(trades_today=2), _long(), cfg, now=QUIET).allowed
    # 0 = unlimited (default)
    assert check_trade(_state(trades_today=99), _long(), CFG, now=QUIET).allowed


def test_max_positions_blocks():
    opens = [OpenPosition(f"C{i}", "indep", 10.0, LONG) for i in range(5)]
    r = check_trade(_state(open_positions=opens), _long(), CFG, now=QUIET)
    assert not r.allowed and any("max positions" in h for h in r.hard_blocks)


def test_group_aware_heat_below_naive_sum():
    pos = [OpenPosition("A1", "ETH", 20.0, LONG), OpenPosition("A2", "ETH", 20.0, LONG),
           OpenPosition("A3", "ETH", 20.0, LONG)]
    # 20 + 0.8*(20+20) = 52, well below the naive 60
    assert group_heat_usd(pos, 0.8) == pytest.approx(52.0)


def test_heat_scale_to_fit_then_block():
    # one $50 position in group A; cap is $60 (6% of 1000)
    opens = [OpenPosition("A1", "A", 50.0, LONG)]
    scaled = check_trade(_state(open_positions=opens), _long(risk_pct=2.0, group="B"), CFG, now=QUIET)
    assert scaled.allowed                       # not blocked — scaled down
    assert scaled.adjusted_risk_pct == pytest.approx(1.0)   # $20 -> $10 headroom
    # already at the cap -> block
    full = [OpenPosition("A1", "A", 60.0, LONG)]
    blocked = check_trade(_state(open_positions=full), _long(risk_pct=2.0, group="B"), CFG, now=QUIET)
    assert not blocked.allowed and any("heat" in h for h in blocked.hard_blocks)


def test_funding_window_warn():
    assert _in_funding_window(datetime(2024, 1, 1, 0, 5, tzinfo=timezone.utc), 15.0)
    assert not _in_funding_window(QUIET, 15.0)
    r = check_trade(_state(), _long(), CFG, now=datetime(2024, 1, 1, 8, 3, tzinfo=timezone.utc))
    assert any("funding" in w for w in r.soft_warns)


def test_correlation_soft_warn_same_group():
    opens = [OpenPosition("A1", "ETH", 10.0, LONG)]
    r = check_trade(_state(open_positions=opens), _long(group="ETH"), CFG, now=QUIET)
    assert r.allowed and any("correlation" in w for w in r.soft_warns)
