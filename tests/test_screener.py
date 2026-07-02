"""Tests for screener logic that needs no network: filtering, regime, clustering."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import Market, ScreenerConfig
from src.indicators import CHOPPY, RANGING, TRENDING
from src.market_context import RISK_OFF, RISK_ON, MarketContext
from src.screener import (
    Candidate,
    _assign_groups,
    _rank_by_volume,
    _regime_from_adx,
    _score,
    _trend_dir,
    eligible_symbols,
)

CFG = ScreenerConfig()


def _ctx(posture):
    return MarketContext(proxy_symbol="BTC/USDT", regime="trending",
                         trend="up" if posture == RISK_ON else "down",
                         posture=posture, adx=30.0, last_price=50000.0)


class FakeExchange:
    """Minimal stand-in exposing only .markets, as eligible_symbols needs."""

    def __init__(self, markets):
        self.markets = markets


def _spot_market(base, quote="USDT", active=True):
    return {"base": base, "quote": quote, "active": active,
            "spot": True, "swap": False, "linear": False}


def _swap_market(base, quote="USDT", active=True, expiry=None):
    return {"base": base, "quote": quote, "active": active,
            "spot": False, "swap": True, "linear": True, "expiry": expiry}


def test_regime_thresholds():
    assert _regime_from_adx(30.0, CFG) == TRENDING
    assert _regime_from_adx(22.0, CFG) == RANGING
    assert _regime_from_adx(10.0, CFG) == CHOPPY
    assert _regime_from_adx(float("nan"), CFG) == CHOPPY


def test_trend_direction():
    assert _trend_dir(30.0, 10.0) == "up"
    assert _trend_dir(10.0, 30.0) == "down"
    assert _trend_dir(20.0, 20.0) == "flat"


def test_score_penalises_choppy_below_trending():
    trend = Candidate(symbol="A/USDT", quote_volume=1e8, last_price=1, spread_bps=1,
                      depth_usd=1e6, atr_pct=3, adx=30, plus_di=30, minus_di=10,
                      regime=TRENDING, trend="up", n_candles=200)
    chop = Candidate(symbol="B/USDT", quote_volume=1e8, last_price=1, spread_bps=1,
                     depth_usd=1e6, atr_pct=3, adx=15, plus_di=15, minus_di=14,
                     regime=CHOPPY, trend="flat", n_candles=200)
    assert _score(trend, CFG) > _score(chop, CFG)


def test_score_rewards_leader_over_laggard():
    common = dict(quote_volume=1e8, last_price=1, spread_bps=1, depth_usd=1e6,
                  atr_pct=3, adx=30, plus_di=30, minus_di=10, regime=TRENDING,
                  trend="up", n_candles=200, vol_regime="normal")
    leader = Candidate(symbol="L/USDT", rel_strength=15.0, btc_beta=1.0, **common)
    laggard = Candidate(symbol="G/USDT", rel_strength=-15.0, btc_beta=1.0, **common)
    assert _score(leader, CFG) > _score(laggard, CFG)


def test_score_prefers_fresh_over_extended():
    common = dict(quote_volume=1e8, last_price=1, spread_bps=1, depth_usd=1e6,
                  atr_pct=3, adx=30, plus_di=30, minus_di=10, regime=TRENDING,
                  trend="up", n_candles=200, rel_strength=10.0, btc_beta=1.0)
    fresh = Candidate(symbol="F/USDT", vol_regime="normal", freshness="fresh",
                      extension=0.5, **common)
    extended = Candidate(symbol="X/USDT", vol_regime="normal", freshness="extended",
                         extension=3.5, **common)
    assert _score(fresh, CFG) > _score(extended, CFG)


def test_score_penalises_spot_long_fighting_riskoff_btc():
    cand = Candidate(symbol="X/USDT", quote_volume=1e8, last_price=1, spread_bps=1,
                     depth_usd=1e6, atr_pct=3, adx=30, plus_di=30, minus_di=10,
                     regime=TRENDING, trend="up", n_candles=200,
                     rel_strength=5.0, btc_beta=1.6, btc_corr=0.9, vol_regime="normal")
    risk_on = _score(cand, CFG, _ctx(RISK_ON), Market.SPOT)
    cand.fighting_tide = False
    risk_off = _score(cand, CFG, _ctx(RISK_OFF), Market.SPOT)
    assert risk_off < risk_on
    assert cand.fighting_tide is True


def test_decoupled_leader_not_flagged_fighting_tide():
    # Leads BTC, low correlation/beta -> defying the tide, not fighting it.
    cand = Candidate(symbol="W/USDT", quote_volume=1e8, last_price=1, spread_bps=1,
                     depth_usd=1e6, atr_pct=10, adx=45, plus_di=40, minus_di=10,
                     regime=TRENDING, trend="up", n_candles=200,
                     rel_strength=120.0, btc_beta=0.5, btc_corr=0.1, vol_regime="expanding")
    _score(cand, CFG, _ctx(RISK_OFF), Market.SPOT)
    assert cand.fighting_tide is False


def test_eligible_symbols_spot_excludes_stables_and_leveraged_tokens():
    markets = {
        "BTC/USDT": _spot_market("BTC"),
        "ETH/USDT": _spot_market("ETH"),
        "USDC/USDT": _spot_market("USDC"),       # stable/stable -> excluded
        "BTCUP/USDT": _spot_market("BTCUP"),     # leveraged token -> excluded
        "ETHDOWN/USDT": _spot_market("ETHDOWN"), # leveraged token -> excluded
        "ADA/BTC": _spot_market("ADA", quote="BTC"),  # wrong quote -> excluded
        "DOGE/USDT": _spot_market("DOGE", active=False),  # inactive -> excluded
        "SOL/USDT": _swap_market("SOL"),         # not spot -> excluded on SPOT
    }
    out = set(eligible_symbols(FakeExchange(markets), CFG, Market.SPOT))
    assert out == {"BTC/USDT", "ETH/USDT"}


def test_eligible_symbols_usdm_only_perpetual_swaps():
    markets = {
        "BTC/USDT:USDT": _swap_market("BTC"),
        "ETH/USDT:USDT": _swap_market("ETH"),
        "OLD/USDT:USDT": _swap_market("OLD", expiry=1900000000000),  # dated -> excluded
        "BTC/USDT": _spot_market("BTC"),         # spot -> excluded on USD-M
        "USDC/USDT:USDT": _swap_market("USDC"),  # stable -> excluded
    }
    out = set(eligible_symbols(FakeExchange(markets), CFG, Market.USDM))
    assert out == {"BTC/USDT:USDT", "ETH/USDT:USDT"}


def test_rank_by_volume_applies_floor_and_top_n():
    cfg = ScreenerConfig(top_n=2, min_quote_volume=1_000_000)
    tickers = {
        "A/USDT": {"quoteVolume": 9e6, "last": 1.0},
        "B/USDT": {"quoteVolume": 5e6, "last": 2.0},
        "C/USDT": {"quoteVolume": 3e6, "last": 3.0},
        "D/USDT": {"quoteVolume": 500_000, "last": 4.0},  # below floor
    }
    kept, rejected = _rank_by_volume(tickers, list(tickers), cfg)
    assert [s for s, _, _ in kept] == ["A/USDT", "B/USDT"]  # top-2 by volume
    assert rejected == 2  # C (top-N cut) + D (floor)


def test_assign_groups_clusters_with_btc_anchor():
    n = 90
    rng = np.random.default_rng(7)
    btc = np.cumsum(rng.normal(0, 1, n)) + 100.0
    # X and Y track BTC; Z is independent.
    x = btc + rng.normal(0, 0.05, n)
    y = btc + rng.normal(0, 0.05, n)
    z = np.cumsum(rng.normal(0, 1, n)) + 100.0

    def cand(sym, series):
        c = Candidate(symbol=sym, quote_volume=1e8, last_price=1, spread_bps=1,
                      depth_usd=1e6, atr_pct=3, adx=30, plus_di=30, minus_di=10,
                      regime=TRENDING, trend="up", n_candles=n)
        c._close = pd.Series(series).reset_index(drop=True)
        return c

    cands = [cand("X/USDT", x), cand("Y/USDT", y), cand("Z/USDT", z)]
    anchors = {"BTC/USDT": pd.Series(btc).reset_index(drop=True)}
    _assign_groups(cands, anchors, ScreenerConfig(corr_window=60, corr_threshold=0.85))
    by = {c.symbol: c for c in cands}
    # X and Y land in the BTC group and are flagged as one bet.
    assert by["X/USDT"].group == "BTC" and by["X/USDT"].clustered
    assert by["Y/USDT"].group == "BTC"
    assert by["X/USDT"].driver == "BTC/USDT"
    # Z stands alone (independent of the anchors).
    assert by["Z/USDT"].group == "indep"
    assert not by["Z/USDT"].clustered
