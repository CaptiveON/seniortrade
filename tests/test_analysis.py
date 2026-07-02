"""Tests for the analysis synthesis (pure scoring/aggregation) — no network."""
from __future__ import annotations

import pandas as pd

from types import SimpleNamespace

from src.analysis import (
    BEAR,
    BULL,
    NEUTRAL,
    LensRead,
    aggregate,
    classify_location,
    detect_rsi_divergence,
    rank_board,
)
from src.config import AnalysisConfig


def _board_res(symbol, setup=False, plan_valid=None, grade="B", net_rr=0.0):
    setups = [SimpleNamespace(grade=grade, direction="short", setup="breakout_retest")] if setup else []
    plan = SimpleNamespace(valid=plan_valid, net_rr=net_rr) if plan_valid is not None else None
    return SimpleNamespace(symbol=symbol, setups=setups, plan=plan)


def test_rank_board_orders_plans_then_setups_then_no_trade():
    a = _board_res("A", setup=True, plan_valid=True, grade="B", net_rr=1.8)
    b = _board_res("B", setup=True, plan_valid=True, grade="A", net_rr=1.2)   # better grade wins
    c = _board_res("C", setup=True, plan_valid=False, grade="C")             # setup, no valid plan
    d = _board_res("D")                                                      # no setup → NO TRADE
    assert [r.symbol for r in rank_board([a, c, d, b])] == ["B", "A", "C", "D"]


def test_rank_board_ties_break_on_net_rr():
    lo = _board_res("LO", setup=True, plan_valid=True, grade="B", net_rr=1.5)
    hi = _board_res("HI", setup=True, plan_valid=True, grade="B", net_rr=2.4)
    assert [r.symbol for r in rank_board([lo, hi])] == ["HI", "LO"]


def test_rank_board_setups_only_drops_no_trade():
    a = _board_res("A", setup=True, plan_valid=True, grade="B", net_rr=1.5)
    d = _board_res("D")
    assert [r.symbol for r in rank_board([a, d], setups_only=True)] == ["A"]
from src import structure as st

CFG = AnalysisConfig()


def _lenses(score):
    return [
        LensRead("trend", "", score, ""),
        LensRead("momentum", "", score, ""),
        LensRead("volume", "", score, ""),
        LensRead("levels", "", score, ""),
        LensRead("context", "", score, ""),
        LensRead("volatility", "", 0.0, ""),
    ]


def test_aggregate_all_bullish_gives_bull_bias_when_htf_up():
    net, bias, conf, align, agree = aggregate(_lenses(1.0), CFG, st.UP)
    assert bias == BULL
    assert conf == 1.0          # weights sum to 1.0
    assert align == "aligned"
    assert agree == 5


def test_aggregate_htf_gating_caps_countertrend():
    # all lenses bullish, but the higher timeframe is DOWN -> gated to neutral
    net, bias, conf, align, agree = aggregate(_lenses(1.0), CFG, st.DOWN)
    assert net <= 0.05
    assert bias == NEUTRAL


def test_aggregate_all_bearish_gives_bear():
    net, bias, conf, align, agree = aggregate(_lenses(-1.0), CFG, st.DOWN)
    assert bias == BEAR
    assert align == "aligned"


def _level(price, kind, side="mixed"):
    return st.Level(price=price, touches=2, kind=kind, side=side,
                    last_touch_index=10, strength=2.0)


def test_classify_location_at_support_and_resistance():
    sup = st.Structure(last_price=100.0, levels=[_level(99.7, st.SUPPORT)],
                       range_high=110.0, range_low=90.0)
    assert classify_location(sup, 100.0, atr_val=1.0, cfg=CFG) == "at_support"

    res = st.Structure(last_price=100.0, levels=[_level(100.4, st.RESISTANCE)],
                       range_high=110.0, range_low=90.0)
    assert classify_location(res, 100.0, atr_val=1.0, cfg=CFG) == "at_resistance"


def test_classify_location_breakout():
    s = st.Structure(last_price=111.0, levels=[], range_high=110.0, range_low=90.0)
    assert classify_location(s, 111.0, atr_val=1.0, cfg=CFG) == "breakout_up"


def test_detect_bearish_divergence():
    swings = [st.Swing(2, None, 110.0, st.HIGH), st.Swing(5, None, 115.0, st.HIGH)]
    rsi_series = pd.Series([50, 50, 70, 60, 55, 60])   # idx2=70 > idx5=60 while price HH
    kind, note, score = detect_rsi_divergence(swings, rsi_series)
    assert kind == "bearish" and score < 0


def test_detect_bullish_divergence():
    swings = [st.Swing(2, None, 90.0, st.LOW), st.Swing(5, None, 85.0, st.LOW)]
    rsi_series = pd.Series([50, 50, 30, 35, 40, 40])   # idx2=30 < idx5=40 while price LL
    kind, note, score = detect_rsi_divergence(swings, rsi_series)
    assert kind == "bullish" and score > 0
