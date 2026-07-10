"""A1 funding + A3 OI/L-S: PIT alignment, buckets, collector, and gate flow."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import src.derivs as dv
from src.config import Market, Settings
from src.backtest import run_setups

_8H = 8 * 3600 * 1000


def test_merge_dedupes_and_sorts():
    merged = dv._merge([[2, 0.1], [1, 0.2]], [[2, 0.3], [3, 0.4]])
    assert merged == [[1, 0.2], [2, 0.3], [3, 0.4]]          # ascending, newer row wins the tie


def test_fund_buckets():
    assert dv.fund_bucket(2.5) == "ext+" and dv.fund_bucket(1.2) == "hi+"
    assert dv.fund_bucket(0.0) == "mid"
    assert dv.fund_bucket(-1.2) == "hi-" and dv.fund_bucket(-2.5) == "ext-"


def test_align_funding_is_pit_and_buckets_extremes():
    # 30 settlements at a flat 0.0001, then one violent 0.01 print
    events = [[i * _8H, 0.0001] for i in range(30)] + [[30 * _8H, 0.01]]
    bars = [-1, 5 * _8H, 25 * _8H, 30 * _8H]
    ctx, rates = dv.align_funding(bars, events)
    assert ctx[0] is None and rates[0] is None               # PIT: before any settlement → nothing
    assert ctx[1] is None                                     # < _FUND_MIN trailing events → no bucket
    assert ctx[2] == "mid" and rates[2] == pytest.approx(0.0001)
    assert ctx[3] == "ext+"                                   # the spike vs its trailing window
    # strict PIT: a bar 1ms before the spike settles must NOT see it
    ctx2, rates2 = dv.align_funding([30 * _8H - 1], events)
    assert rates2[0] == pytest.approx(0.0001)


def test_align_oi_needs_history_then_buckets():
    rows = [[i * _8H, 100.0] for i in range(10)] + [[10 * _8H, 112.0]]   # +12% vs flat
    bars = [0, 3 * _8H, 10 * _8H]
    out = dv.align_oi(bars, rows, "4h")                      # lookback 6 snapshots
    assert out[0] is None                                     # not enough local history yet
    assert out[2] == "expand"
    assert dv.align_oi([9 * _8H], rows, "4h") == ["flat"]


class _FakeDerivEx:
    def __init__(self):
        self.calls = 0

    def market(self, symbol):
        return {"id": symbol.split("/")[0] + "USDT"}

    def _rows(self, field):
        return [{"timestamp": i * _8H, field: "1.0"} for i in range(5)]

    def fapiDataGetOpenInterestHist(self, params):
        self.calls += 1
        return [{"timestamp": i * _8H, "sumOpenInterest": 100 + i} for i in range(5)]

    def fapiDataGetTopLongShortPositionRatio(self, params):
        return self._rows("longShortRatio")

    def fapiDataGetTopLongShortAccountRatio(self, params):
        return self._rows("longShortRatio")

    def fapiDataGetGlobalLongShortAccountRatio(self, params):
        return self._rows("longShortRatio")

    def fapiDataGetTakerlongshortRatio(self, params):
        return self._rows("buySellRatio")


def test_collector_appends_and_dedupes(tmp_path, monkeypatch):
    monkeypatch.setattr(dv, "_DIR", tmp_path)
    ex = _FakeDerivEx()
    added1 = dv.collect_snapshots(ex, Market.USDM, ["BTC/USDT:USDT"])
    assert added1["oi"] == 5
    added2 = dv.collect_snapshots(ex, Market.USDM, ["BTC/USDT:USDT"])   # same window again
    assert added2["oi"] == 0                                  # dedupe: nothing new, nothing lost
    assert len(dv.oi_series(Market.USDM, "BTC/USDT:USDT")) == 5
    assert dv.collect_snapshots(ex, Market.SPOT, ["BTC/USDT"]) == {}    # USD-M only


def _walk_bars(seed=7, n=420):
    rng = np.random.default_rng(seed)
    close = 100.0 + rng.normal(0, 1, n).cumsum()
    high = close + np.abs(rng.normal(0, 0.4, n))
    low = close - np.abs(rng.normal(0, 0.4, n))
    open_ = np.r_[close[0], close[:-1]]
    idx = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    return pd.DataFrame({"open": open_, "high": high, "low": low,
                         "close": close, "volume": 1.0}, index=idx)


def test_run_setups_carries_fund_oi_context_and_actual_funding_cost():
    df = _walk_bars()
    n = len(df)
    fund_ctx = ["hi+"] * n
    oi_ctx = ["expand"] * n
    hot = [0.002] * n                                        # 20× the flat assumption
    res_hot = run_setups(df, Market.USDM, Settings(), "4h", seed=3,
                         fund_ctx=fund_ctx, fund_rate=hot, oi_ctx=oi_ctx)
    res_flat = run_setups(df, Market.USDM, Settings(), "4h", seed=3)
    trades_hot = [t for ts in res_hot.trades.values() for t in ts]
    trades_flat = [t for ts in res_flat.trades.values() for t in ts]
    assert trades_hot, "no trades fired — vacuous"
    assert all(t.context.get("fund") == "hi+" and t.context.get("oi") == "expand" for t in trades_hot)
    assert all(t.context.get("fund") is None for t in trades_flat)      # absent stays None, never invented
    # actual funding (0.002/8h) must cost measurably more than the flat 0.0001 assumption
    assert sum(t.funding_r for t in trades_hot) > sum(t.funding_r for t in trades_flat) * 3


def test_fund_feature_flows_through_the_same_gate():
    from types import SimpleNamespace
    from src.config import EdgeScoreConfig
    from src.edge_score import CONTEXT_FEATURES, evaluate_conditioning
    assert "fund" in CONTEXT_FEATURES and "oi" in CONTEXT_FEATURES
    rng = np.random.default_rng(11)
    tr = ([SimpleNamespace(r=float(rng.normal(0.6, 0.4)), regime="up", context={"fund": "ext-"})
           for _ in range(120)]
          + [SimpleNamespace(r=float(rng.normal(-0.05, 0.4)), regime="up", context={"fund": "mid"})
             for _ in range(120)])
    refs = evaluate_conditioning({"s": tr}, EdgeScoreConfig(), min_child_n=20)
    proven = [r for r in refs if r["feature"] == "fund" and r["value"] == "ext-" and r["proven"]]
    assert proven, "a real funding-extreme edge must clear money+Bonferroni+OOS like any feature"
