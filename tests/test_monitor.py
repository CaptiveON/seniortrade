"""Tests for Layer 10 trade management: the shared state machine, paper-fill, warnings."""
from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

from src.config import MarketsConfig, RiskConfig
from src.monitor import (
    CLOSE_TP2,
    HOLD,
    SCALE_TP1,
    STOP_EXIT,
    detect_paper_fill,
    manage_step,
    unrealized_r,
    usdm_warnings,
)
from src.risk import EV_TP1, EV_TP2, walk_management


def _bars(rows):
    idx = pd.date_range("2024-01-01", periods=len(rows), freq="4h", tz="UTC")
    return pd.DataFrame(rows, index=idx, columns=["open", "high", "low", "close"])


def _open(**kw):
    base = dict(side="long", entry_actual=100.0, entry_planned=100.0, stop_planned=95.0,
                targets=[110.0, 120.0], current_stop=None, remaining_fraction=1.0,
                tp1_filled=False, locked_r=0.0, managed_at="", liquidation_price=None)
    base.update(kw)
    return SimpleNamespace(**base)


def _staged(**kw):
    base = dict(side="long", entry_type="limit", entry_planned=100.0, stop_planned=95.0,
                targets=[110.0, 120.0], invalidation=0.0, managed_at="",
                timestamp="2023-12-01T00:00:00+00:00")   # before the test bars (staged earlier)
    base.update(kw)
    return SimpleNamespace(**base)


CFG = RiskConfig()


# --- the SHARED state machine (same generator the backtest walks) ----------- #
def test_walk_seeded_skips_already_filled_tp1():
    bars = _bars([(112, 121, 111, 119)])  # seeded post-TP1, runner reaches TP2
    evs = list(walk_management("long", 100, 95, [110, 120], bars, CFG,
                               remaining=0.5, cur_stop=100, hit_tp1=True))
    assert [e.kind for e in evs] == [EV_TP2]
    assert evs[0].closes
    assert abs(evs[0].realized_delta - 0.5 * (120 - 100) / 5) < 1e-9   # +2.0R on the half


def test_walk_fresh_yields_tp1_then_tp2():
    bars = _bars([(101, 111, 100, 108), (112, 121, 111, 119)])
    evs = list(walk_management("long", 100, 95, [110, 120], bars, CFG))
    assert [e.kind for e in evs] == [EV_TP1, EV_TP2]
    assert evs[0].new_stop == 100 and not evs[0].closes          # breakeven, partial
    assert evs[1].closes


# --- manage_step: one proposed action, worst-case, stop-first --------------- #
def test_manage_proposes_tp1_scale_and_breakeven():
    rec = _open()
    act = manage_step(rec, _bars([(101, 111, 100, 108)]), current_price=108.0, cfg=CFG)
    assert act.kind == SCALE_TP1
    assert act.new_stop == 100.0 and act.remaining_after == 0.5 and act.tp1_filled_after
    assert abs(act.realized_delta_r - 1.0) < 1e-9                 # 0.5 * 2R


def test_manage_closes_runner_at_tp2_after_tp1():
    rec = _open(current_stop=100.0, remaining_fraction=0.5, tp1_filled=True, locked_r=1.0)
    act = manage_step(rec, _bars([(112, 121, 111, 119)]), current_price=119.0, cfg=CFG)
    assert act.kind == CLOSE_TP2 and act.closes
    assert abs(act.total_realized_r - 3.0) < 1e-9                 # 1.0 locked + 2.0 runner


def test_manage_proposes_stop_exit():
    rec = _open()
    act = manage_step(rec, _bars([(99, 100, 94, 96)]), current_price=96.0, cfg=CFG)
    assert act.kind == STOP_EXIT and act.closes
    assert abs(act.total_realized_r + 1.0) < 1e-9                 # -1.0R


def test_manage_stop_wins_a_bar_that_spans_both():
    rec = _open()  # bar touches TP1 AND the stop → stop fills first (worst case)
    act = manage_step(rec, _bars([(100, 111, 94, 105)]), current_price=105.0, cfg=CFG)
    assert act.kind == STOP_EXIT


def test_manage_holds_when_nothing_triggers():
    rec = _open()
    act = manage_step(rec, _bars([(101, 104, 99, 102)]), current_price=102.0, cfg=CFG)
    assert act.kind == HOLD


def test_manage_live_price_triggers_intrabar():
    rec = _open()  # closed bars quiet, but the LIVE tick is already through TP1
    act = manage_step(rec, _bars([(101, 104, 99, 102)]), current_price=111.0, cfg=CFG)
    assert act.kind == SCALE_TP1


# --- paper-fill: a dry-run STAGED trade reaching its entry ------------------ #
def test_paper_fill_limit_touched_and_not():
    filled, price, _ = detect_paper_fill(_staged(), _bars([(102, 103, 99, 101)]), current_price=101.0)
    assert filled and price == 100.0                              # dipped to the limit
    no, _, _ = detect_paper_fill(_staged(), _bars([(102, 103, 101, 102)]), current_price=102.0)
    assert not no                                                 # never reached 100


def test_paper_fill_market_is_immediate():
    filled, price, _ = detect_paper_fill(_staged(entry_type="market"),
                                         _bars([(102, 103, 101, 102)]), current_price=102.5)
    assert filled and price == 102.5


def test_paper_fill_voided_only_when_unfilled():
    # long breakout buy-stop @110 that never triggered, but price broke DOWN through
    # the invalidation (96) first → void before fill.
    rec = _staged(side="long", entry_type="stop", entry_planned=110.0, invalidation=96.0)
    filled, _, why = detect_paper_fill(rec, _bars([(99, 100, 95, 97)]), current_price=95.0)
    assert not filled and "void" in why
    # but if the entry WAS reached, reaching it fills — invalidation doesn't block a fill
    rec2 = _staged(side="short", entry_type="limit", entry_planned=100.0, invalidation=100.0)
    f2, p2, _ = detect_paper_fill(rec2, _bars([(99, 101, 98, 100)]), current_price=100.5)
    assert f2 and p2 == 100.0                       # retest short: entry==invalidation still fills


def test_paper_fill_ignores_pre_stage_history():
    # staged "now" (no bars since); an OLD bar touched the level but the live price hasn't
    # → must stay PENDING, not fill off stale history (regression: caught on live WLD).
    rec = _staged(side="short", entry_type="limit", entry_planned=0.58,
                  managed_at="2099-01-01T00:00:00+00:00")
    filled, _, why = detect_paper_fill(rec, _bars([(0.59, 0.60, 0.58, 0.585)]), current_price=0.55)
    assert not filled and "not reached" in why


# --- USD-M watch + open-R --------------------------------------------------- #
def test_usdm_warns_when_liquidation_is_near():
    quiet = datetime(2024, 1, 1, 3, 0, tzinfo=timezone.utc)      # far from funding
    near = usdm_warnings(_open(liquidation_price=93.0, current_stop=95.0), 100.0,
                         MarketsConfig(), now=quiet)
    assert any("liquidation" in w for w in near)
    far = usdm_warnings(_open(liquidation_price=80.0, current_stop=95.0), 100.0,
                        MarketsConfig(), now=quiet)
    assert not far


def test_usdm_warns_inside_funding_window():
    pre_funding = datetime(2024, 1, 1, 7, 50, tzinfo=timezone.utc)   # ~10 min to 08:00
    warns = usdm_warnings(_open(liquidation_price=80.0), 100.0, MarketsConfig(), now=pre_funding)
    assert any("funding" in w for w in warns)


def test_usdm_warns_on_crowded_funding_and_oi_trend():
    quiet = datetime(2024, 1, 1, 3, 0, tzinfo=timezone.utc)
    w = usdm_warnings(_open(), 100.0, MarketsConfig(), now=quiet,
                      funding_signal="crowded_long", oi_trend="rising", oi_change_pct=9.2)
    assert any("CROWDED long" in x for x in w)              # we are long on the crowded side
    assert any("open interest rising" in x for x in w)
    # a short would NOT be flagged crowded by a crowded-long book
    w2 = usdm_warnings(_open(side="short"), 100.0, MarketsConfig(), now=quiet, funding_signal="crowded_long")
    assert not any("CROWDED" in x for x in w2)


def test_unrealized_r():
    assert abs(unrealized_r(_open(), 105.0) - 1.0) < 1e-9         # +1R at +5 on a 5-wide stop


# --- audit-walk findings 5+6: stale pending expiry + replay-after-fill ---------- #
from src.monitor import replay_after_fill


def test_pending_entry_expires_stale_after_expiry_bars():
    rec = _staged(side="long", entry_planned=90.0)          # dip-buy never reached
    rows = [[100, 101, 99, 100]] * 12                        # 12 bars since staging, no touch
    filled, _, why = detect_paper_fill(rec, _bars(rows), 100.0, expiry_bars=8)
    assert not filled and "stale" in why and "void" in why   # cli's cancel branch keys on 'void'
    # under the expiry window → still just pending
    filled, _, why2 = detect_paper_fill(rec, _bars(rows[:5]), 100.0, expiry_bars=8)
    assert not filled and "void" not in why2
    # no expiry arg → legacy behaviour (never stale)
    filled, _, why3 = detect_paper_fill(rec, _bars(rows), 100.0)
    assert not filled and "void" not in why3


def test_replay_detects_fill_then_stop_breach_as_completed_loss():
    # the PAXG case: short limit touched in history, then price ran THROUGH the stop.
    rec = _staged(side="short", entry_planned=110.0, stop_planned=115.0, targets=[100.0, 95.0])
    rows = [
        [105, 106, 104, 105],       # staged; not touched
        [106, 111, 105, 110],       # pops to 111 → limit 110 TOUCHED (fill bar)
        [110, 116, 109, 115.5],     # runs through the stop 115 → stopped
        [115, 121, 114, 120],       # keeps running (already out)
    ]
    out = replay_after_fill(rec, _bars(rows), RiskConfig())
    assert out is not None and out["closed"] is True
    assert out["exit"] == 115.0 and out["r"] == pytest.approx(-1.0)


def test_replay_still_open_marks_to_market():
    rec = _staged(side="short", entry_planned=110.0, stop_planned=115.0, targets=[100.0, 95.0])
    rows = [
        [105, 106, 104, 105],
        [106, 111, 105, 110],       # touched; fill bar (favourable extreme clamped)
        [109, 111, 107, 108],       # drifts our way; no stop, no TP
    ]
    out = replay_after_fill(rec, _bars(rows), RiskConfig())
    assert out is not None and out["closed"] is False
    assert out["r"] == pytest.approx((110 - 108) / 5.0)      # +0.4R mark
    assert out["fill_ts"]                                     # retro managed_at anchor


def test_replay_never_credits_same_bar_target_before_fill():
    # fill bar's favourable extreme is clamped to entry — a TP that printed in the same
    # bar as the touch must NOT be credited (the backtest's anti-look-ahead convention).
    rec = _staged(side="short", entry_planned=110.0, stop_planned=115.0, targets=[104.0, 95.0])
    rows = [
        [105, 106, 104, 105],
        [106, 111, 103.9, 105],     # touches entry AND trades through TP1 in the SAME bar
    ]
    out = replay_after_fill(rec, _bars(rows), RiskConfig())
    assert out is not None and out["closed"] is False        # TP not credited on the fill bar
