"""Tests for journal storage, account-state derivation, and review."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.config import JournalConfig
from src.journal import (
    CLOSED,
    OPEN,
    STAGED,
    TradeRecord,
    append,
    build_review,
    load_records,
    portfolio_state,
    record_close,
    record_fill,
    record_staged,
)


def _iso(dt):
    return dt.isoformat()


def _closed(symbol, realized_r, risk_amount=10.0, setup="trend_pullback", regime="up",
            grade="B", closed_at=None, side="long", planned_stop=95.0, actual_stop=95.0):
    return TradeRecord(
        id=symbol + str(realized_r), timestamp=_iso(datetime(2024, 1, 1, tzinfo=timezone.utc)),
        market="usdm", symbol=symbol, side=side, setup=setup, grade=grade, status=CLOSED,
        group="indep", entry_planned=100.0, stop_planned=planned_stop, stop_actual=actual_stop,
        targets=[110.0, 120.0], risk_pct_planned=1.0, risk_amount=risk_amount,
        regime=regime, realized_r=realized_r, pnl_usd=realized_r * risk_amount,
        closed_at=closed_at or _iso(datetime.now(timezone.utc)))


def test_storage_keeps_latest_per_id(tmp_path):
    p = tmp_path / "j.jsonl"
    rec = TradeRecord(id="abc", timestamp="t", market="usdm", symbol="X", side="long", status=STAGED)
    append(rec, p)
    rec.status = CLOSED
    rec.realized_r = 1.5
    append(rec, p)
    loaded = load_records(p)
    assert len(loaded) == 1
    assert loaded[0].status == CLOSED and loaded[0].realized_r == 1.5


def test_lifecycle_helpers(tmp_path):
    p = tmp_path / "j.jsonl"
    tid = record_staged(market="usdm", symbol="ETH/USDT:USDT", side="long", setup="trend_pullback",
                        grade="A", group="indep", entry=100, stop=95, targets=[110, 120],
                        risk_pct=1.0, risk_amount=10.0, path=p)
    record_fill(tid, entry_actual=100.5, path=p)
    record_close(tid, exit_price=110.0, realized_r=1.8, pnl_usd=18.0, path=p)
    recs = load_records(p)
    assert len(recs) == 1 and recs[0].status == CLOSED and recs[0].realized_r == 1.8


def test_portfolio_state_derivation():
    now = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)        # fixed mid-day (no UTC-midnight flake)
    recs = [
        _closed("A/USDT:USDT", +2.0, closed_at=_iso(now - timedelta(minutes=30))),
        _closed("B/USDT:USDT", -1.0, closed_at=_iso(now - timedelta(minutes=10))),  # latest = loss
        TradeRecord(id="open1", timestamp="t", market="usdm", symbol="C/USDT:USDT", side="long",
                    status=OPEN, group="ETH", risk_amount=15.0),
    ]
    st = portfolio_state(recs, JournalConfig(starting_equity=1000.0), now=now)
    assert st.equity == 1000.0 + 20.0 - 10.0          # +2R*$10 -1R*$10
    assert len(st.open_positions) == 1 and st.open_positions[0].risk_amount == 15.0
    assert st.realized_r_today == 1.0                  # +2 - 1
    assert st.consecutive_losses == 1                  # last closed was a loss
    assert st.last_loss_time is not None


def test_build_review_verdicts_and_buckets():
    cfg = JournalConfig(starting_equity=1000.0, min_sample=4, min_bucket_sample=2)
    recs = [
        _closed("A", +1.5, setup="trend_pullback", regime="up"),
        _closed("B", +1.0, setup="trend_pullback", regime="up"),
        _closed("C", -1.0, setup="range_fade", regime="range"),
        _closed("D", -1.0, setup="range_fade", regime="range"),
    ]
    rep = build_review(recs, cfg)
    assert rep.n_closed == 4 and rep.enough_sample
    assert "trend_pullback" in rep.by_setup and rep.by_setup["trend_pullback"].expectancy > 0
    assert any("edge is in 'trend_pullback'" in v for v in rep.verdicts)
    assert any("stop taking 'range_fade'" in v for v in rep.verdicts)


def test_build_review_too_few():
    rep = build_review([_closed("A", 1.0)], JournalConfig(min_sample=20))
    assert not rep.enough_sample
    assert any("too few" in v for v in rep.verdicts)


def test_portfolio_state_drawdown_and_trades_today():
    import pytest
    now = datetime(2024, 1, 1, 12, tzinfo=timezone.utc)
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    recs = [
        _closed("A", +5.0, closed_at=_iso(base + timedelta(hours=1))),   # peak 1050
        _closed("B", -3.0, closed_at=_iso(base + timedelta(hours=2))),   # 1020
        _closed("C", -3.0, closed_at=_iso(base + timedelta(hours=3))),   # 990
    ]
    st = portfolio_state(recs, JournalConfig(), now=now)
    assert st.drawdown_pct == pytest.approx(60 / 1050 * 100, abs=0.01)   # (1050-990)/1050
    assert st.trades_today == 3                                          # all staged 2024-01-01 == today
