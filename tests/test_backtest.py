"""Tests for the backtest engine: tf parsing, entry fills, smoke run."""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src import expectancy as ex
from src.backtest import (
    BacktestResult,
    NullTrade,
    _archetype,
    _cost_r,
    _resolve_entry,
    _tf_hours,
    backtest_all,
    run_setups,
)
from src.config import Market, Settings
from src.markets import LONG, SHORT
from src.setups import DETECTORS, LIMIT, MARKET, STOP


def _bars(rows):
    idx = pd.date_range("2024-01-01", periods=len(rows), freq="4h", tz="UTC")
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)
    df["volume"] = 1.0
    return df


def test_tf_hours():
    assert _tf_hours("4h") == 4.0
    assert _tf_hours("1d") == 24.0
    assert _tf_hours("15m") == 0.25


def test_resolve_entry_market_fills_next_open():
    bars = _bars([[100, 101, 99, 100], [102, 103, 101, 102]])
    sig = SimpleNamespace(entry_type=MARKET, direction=LONG, entry=100, expiry_bars=5)
    j, price = _resolve_entry(bars, 0, sig, 5)
    assert j == 1 and price == 102.0           # next bar's open


def test_resolve_entry_limit_fills_then_expires():
    # long limit at 100: fills only if a later bar dips to 100
    bars = _bars([[105, 106, 104, 105], [104, 105, 100, 101], [101, 102, 100.5, 101]])
    sig = SimpleNamespace(entry_type=LIMIT, direction=LONG, entry=100.0, expiry_bars=5)
    res = _resolve_entry(bars, 0, sig, 5)
    assert res == (1, 100.0)
    # never reaches 100 within expiry -> no fill
    bars2 = _bars([[105, 106, 104, 105], [106, 107, 105, 106]])
    assert _resolve_entry(bars2, 0, sig, 1) is None


def test_resolve_entry_stop_breaks_up():
    bars = _bars([[100, 101, 99, 100], [101, 111, 100, 110]])
    sig = SimpleNamespace(entry_type=STOP, direction=LONG, entry=110.0, expiry_bars=5)
    assert _resolve_entry(bars, 0, sig, 5) == (1, 110.0)


def test_cost_r_positive_and_cheaper_for_limit():
    s = Settings()
    market_sig = SimpleNamespace(entry_type=MARKET, direction=LONG, entry=100.0, stop=96.0)
    limit_sig = SimpleNamespace(entry_type=LIMIT, direction=LONG, entry=100.0, stop=96.0)
    cm = _cost_r(market_sig, 100.0, Market.USDM, s, bars_held=2, tf_hours=4.0)
    cl = _cost_r(limit_sig, 100.0, Market.USDM, s, bars_held=2, tf_hours=4.0)
    assert cm > 0 and cl > 0
    assert cl < cm                              # maker + no entry slippage is cheaper


_ARCH_VALUES = {"panic", "euphoria", "vol_expansion", "vol_compression", "accumulation",
                "distribution", "trending_up", "trending_down", "range"}


def test_archetype_classifier_priority_and_disjoint():
    s = Settings().setups
    # priority tree: most-specific first, exactly one label per state.
    assert _archetype("down", "expanding", -2.0, 0.0, s) == "panic"
    assert _archetype("up", "expanding", 2.0, 0.0, s) == "euphoria"
    assert _archetype("down", "expanding", -0.5, 0.0, s) == "vol_expansion"   # expanding, no thrust extreme
    assert _archetype("range", "normal", 0.0, -3.0, s) == "accumulation"
    assert _archetype("range", "normal", 0.0, 3.0, s) == "distribution"
    assert _archetype("range", "squeeze", 0.0, 0.0, s) == "vol_compression"
    assert _archetype("up", "squeeze", 0.0, 0.0, s) == "vol_compression"
    assert _archetype("up", "normal", 0.0, 0.0, s) == "trending_up"
    assert _archetype("down", "normal", 0.0, 0.0, s) == "trending_down"
    assert _archetype("range", "normal", 0.0, 0.0, s) == "range"


def test_run_setups_tags_every_trade_with_an_archetype():
    res = run_setups(_random_walk(7), Market.USDM, Settings(), "4h", seed=7)
    seen = 0
    for trades in res.trades.values():
        for tr in trades:
            assert tr.context.get("arch") in _ARCH_VALUES
            seen += 1
    assert seen >= 1, "no trades produced — test would be vacuous"


def test_run_setups_records_gross_and_funding_for_cost_decomposition():
    # P10: every Trade carries gross_r (pre-cost) and funding_r; costs must REDUCE net R.
    res = run_setups(_random_walk(7), Market.USDM, Settings(), "4h", seed=7)
    seen = 0
    for trades in res.trades.values():
        for tr in trades:
            assert tr.gross_r >= tr.r - 1e-9            # net = gross − costs ⇒ gross ≥ net
            assert tr.funding_r >= 0.0                  # USD-M carry is a cost
            seen += 1
    assert seen >= 1


def test_backtest_all_smoke_runs_and_is_wellformed():
    # an oscillating range so range/reversal setups can trigger
    n = 160
    t = np.arange(n)
    close = 110 + 8 * np.sin(t / 3.0)
    df = _bars(np.column_stack([close, close + 1.0, close - 1.0, close]).tolist())
    out = backtest_all(df, Market.USDM, Settings(), "4h")
    assert isinstance(out, dict)
    for trades in out.values():
        for tr in trades:
            assert tr.exit_index >= tr.entry_index
            assert isinstance(tr.r, float)


def test_backtest_all_disables_null_run_setups_emits_it():
    n = 160
    t = np.arange(n)
    close = 110 + 8 * np.sin(t / 3.0)
    df = _bars(np.column_stack([close, close + 1.0, close - 1.0, close]).tolist())
    # backtest_all is the null-disabled wrapper (optimizer/tests path)
    assert isinstance(backtest_all(df, Market.USDM, Settings(), "4h"), dict)
    # run_setups emits matched-geometry shadow nulls alongside the real trades
    res = run_setups(df, Market.USDM, Settings(), "4h", seed=3)
    assert isinstance(res, BacktestResult)
    for nm, nulls in res.nulls.items():
        assert nm in res.trades
        for sh in nulls:
            assert isinstance(sh, NullTrade) and isinstance(sh.r, float)


def _random_walk(seed, n=420):
    """Pure noise: a driftless random walk. No setup should show real edge here."""
    rng = np.random.default_rng(seed)
    close = 100.0 + rng.normal(0, 1, n).cumsum()
    high = close + np.abs(rng.normal(0, 0.4, n))
    low = close - np.abs(rng.normal(0, 0.4, n))
    open_ = np.r_[close[0], close[:-1]]
    return _bars(np.column_stack([open_, high, low, close]).tolist())


def test_null_baseline_kills_fabrication_on_random_walk():
    # THE regression for the audit finding: on pure noise the scale-out+breakeven exit
    # makes raw expectancy positive ("free option"). The null baseline must subtract it,
    # so NO setup with a real sample is stamped +EV on noise.
    s = replace(Settings(), backtest=replace(
        Settings().backtest, min_sample=20, null_min=40, null_k=12))
    pooled, pooled_null = {}, {}
    for seed in range(5):
        res = run_setups(_random_walk(seed), Market.USDM, s, "4h", seed=seed)
        for nm, tr in res.trades.items():
            pooled.setdefault(nm, []).extend(tr)
        for nm, nl in res.nulls.items():
            pooled_null.setdefault(nm, []).extend(nl)

    checked = 0
    for nm, tr in pooled.items():
        prof = ex.evaluate(tr, setup=nm, n_combos_tested=len(DETECTORS), cfg=s.backtest,
                           null_trades=pooled_null.get(nm, []))
        if prof.overall.n >= s.backtest.min_sample and prof.null_n >= s.backtest.null_min:
            checked += 1
            assert prof.verdict != ex.VERDICT_EV, f"{nm} fabricated +EV on noise: {prof.notes}"
    assert checked >= 1, "no setup reached the sample minimum — test would be vacuous"
