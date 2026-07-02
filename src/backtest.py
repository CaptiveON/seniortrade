"""Layer 5 — the backtest engine.

Runs the SAME detect() + simulate() code as live across real history, with honest
fills and no look-ahead, producing trades that expectancy.py turns into an
EdgeProfile + verdict.

  - Walk closed bars; compute structure on a rolling window of bars-up-to-i
    (point-in-time, no look-ahead); run every setup detector.
  - Fill BY ENTRY TYPE: market → next bar's open; limit/stop → fills only if a
    later bar (within expiry) trades to the level, else the order EXPIRES (a
    pending order that never fills is NOT a trade).
  - One position at a time per setup (no pyramiding/overlap).
  - Manage via simulate_detailed(); subtract costs in R (maker for limits,
    taker + slippage for market/stop; usdm funding carry assumption).
"""
from __future__ import annotations

from collections import namedtuple
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
import pandas as pd

from . import data_fetch as dfetch
from . import expectancy as ex
from . import indicators as ind
from . import risk as rk
from . import setups as su
from . import structure as st
from .config import Market
from .markets import LONG, SHORT


@dataclass
class Trade:
    setup: str
    direction: str
    entry_index: int
    entry_price: float
    exit_index: int
    exit_price: float
    r: float                 # realized R, NET of costs
    regime: str              # point-in-time regime at entry
    bars_held: int
    context: dict = field(default_factory=dict)   # P1: point-in-time market context at the signal bar
    gross_r: float = 0.0     # P10: realized R BEFORE costs (raw statistical edge)
    funding_r: float = 0.0   # P10: funding/carry portion of the cost (R)
    entry_ts: float = 0.0    # P7: entry-bar epoch seconds (for time-decay weighting)


# A shadow (null) trade: the SAME exit + risk geometry as a real trade, entered at a
# RANDOM bar. Carries the real trade's regime so the null can be pooled by regime.
NullTrade = namedtuple("NullTrade", ["r", "regime"])


@dataclass
class BacktestResult:
    trades: dict[str, list]          # setup -> [Trade]   (real)
    nulls: dict[str, list]           # setup -> [NullTrade] (matched-geometry random entries)


def _tf_hours(tf: str) -> float:
    tf = tf.strip().lower()
    unit = tf[-1]
    try:
        val = float(tf[:-1])
    except ValueError:
        return 24.0
    return {"m": val / 60.0, "h": val, "d": val * 24.0, "w": val * 24.0 * 7.0}.get(unit, 24.0)


# --------------------------------------------------------------------------- #
# P1 — point-in-time market CONTEXT at the signal bar (no look-ahead; same
# structure/indicators the live read uses). Captured on every Trade so the edge
# can later be CONDITIONED on context (gated; see edge_score).
# --------------------------------------------------------------------------- #
def _divergence_flag(struct, rsi_series) -> str:
    """bull / bear / none — regular RSI divergence at the last two same-kind swings."""
    if rsi_series is None or len(struct.swings) < 2:
        return "none"
    def rsi_at(i):
        return float(rsi_series.iloc[i]) if 0 <= i < len(rsi_series) else float("nan")
    lows = [s for s in struct.swings if s.kind == st.LOW][-2:]
    if len(lows) == 2:
        a, b = lows
        ra, rb = rsi_at(a.index), rsi_at(b.index)
        if b.price < a.price and ra == ra and rb == rb and rb > ra:
            return "bull"
    highs = [s for s in struct.swings if s.kind == st.HIGH][-2:]
    if len(highs) == 2:
        a, b = highs
        ra, rb = rsi_at(a.index), rsi_at(b.index)
        if b.price > a.price and ra == ra and rb == rb and rb < ra:
            return "bear"
    return "none"


def _archetype(trend: str, vol: str, thrust: float, leg: float, scfg) -> str:
    """P14 — the named regime ARCHETYPE for this bar: a disjoint, priority-ordered label
    derived ONLY from primitives the engine already computes (structural trend + ATR-%ile
    vol + recent thrust + the leg into a range). Most-specific first, so every bar maps to
    exactly ONE archetype (clean pooling). The label is DESCRIPTIVE — the statistical edge
    still comes from the gated conditional expectancy of the trend×vol cell, not the name."""
    expanding = vol == "expanding"
    if trend == st.DOWN and expanding and thrust <= -scfg.arch_thrust_atr:
        return "panic"
    if trend == st.UP and expanding and thrust >= scfg.arch_thrust_atr:
        return "euphoria"
    if trend == st.RANGE:                                 # a base: which way did price come IN?
        if leg <= -scfg.arch_leg_atr:
            return "accumulation"                         # fell into the range -> basing
        if leg >= scfg.arch_leg_atr:
            return "distribution"                         # rose into the range -> topping
    if expanding:
        return "vol_expansion"                            # trending + expanding, no thrust extreme
    if vol == "squeeze":
        return "vol_compression"                          # coiling (any trend / neutral range)
    if trend == st.UP:
        return "trending_up"
    if trend == st.DOWN:
        return "trending_down"
    return "range"                                        # neutral range, normal vol, no clear leg


def _bar_context(window: pd.DataFrame, struct, settings, htf_trend: str | None) -> dict:
    """Bar-level context (direction-independent). htf_align is added per-signal."""
    scfg = settings.setups
    feats: dict = {}
    try:
        pctl = float(ind.atr_percentile(window, scfg.atr_length, 100))
        feats["vol"] = "squeeze" if pctl <= 30 else "expanding" if pctl >= 75 else "normal"
    except Exception:
        feats["vol"] = "normal"
    rsi_series = None
    try:
        rsi_series = ind.rsi(window, scfg.rsi_length)
        r = float(rsi_series.iloc[-1])
        feats["mom"] = "bull" if r > 55 else "bear" if r < 45 else "neutral"
    except Exception:
        feats["mom"] = "neutral"
    c = float(window["close"].iloc[-1])
    try:
        atr = float(ind.atr(window, scfg.atr_length).iloc[-1])
    except Exception:
        atr = 0.0
    res, sup = struct.resistance_above(c), struct.support_below(c)
    if atr > 0 and res is not None and (res - c) <= 0.6 * atr:
        feats["loc"] = "at_res"
    elif atr > 0 and sup is not None and (c - sup) <= 0.6 * atr:
        feats["loc"] = "at_sup"
    else:
        feats["loc"] = "mid"
    feats["div"] = _divergence_flag(struct, rsi_series)
    feats["htf"] = htf_trend or "na"
    # P14 archetype: recent thrust + the leg into a range, in ATRs (0 when undefined).
    closes = window["close"]
    thrust = leg = 0.0
    if atr > 0:
        if len(closes) > scfg.arch_thrust_bars:
            thrust = (c - float(closes.iloc[-1 - scfg.arch_thrust_bars])) / atr
        if len(closes) > scfg.arch_leg_bars:
            leg = (c - float(closes.iloc[-1 - scfg.arch_leg_bars])) / atr
    trend = struct.state.trend if struct.state else st.RANGE
    feats["arch"] = _archetype(trend, feats["vol"], thrust, leg, scfg)
    return feats


def _htf_align(htf_trend: str | None, direction: str) -> str:
    """aligned / opposed / neutral — the bias-TF trend vs the trade direction."""
    if htf_trend in (None, "na", st.RANGE):
        return "neutral"
    if (direction == LONG and htf_trend == st.UP) or (direction == SHORT and htf_trend == st.DOWN):
        return "aligned"
    return "opposed"


def htf_trend_aligned(market: Market, symbol: str, settings, trigger_index, exch=None) -> list | None:
    """Point-in-time BIAS-TF trend aligned to each trigger bar (the latest CLOSED bias bar's
    structure trend). Returns a list[str] the length of trigger_index, or None on failure."""
    bias_tf = settings.analysis.tf_bias
    try:
        e = exch or dfetch.make_exchange(market, settings.api_key, settings.api_secret)
        if exch is None:
            dfetch.load_markets(e)
        bias = dfetch.drop_unclosed(e, dfetch.fetch_ohlcv(e, symbol, bias_tf, 300), bias_tf)
    except dfetch.DataError:
        return None
    sw = settings.backtest.struct_window
    pts: list = []                                   # (timestamp, trend) per bias bar, point-in-time
    for j in range(min(40, len(bias)), len(bias)):
        s = st.analyze_structure(bias.iloc[max(0, j - sw + 1):j + 1], settings.structure,
                                 with_volume_profile=False)
        pts.append((bias.index[j], s.state.trend if s.state else st.RANGE))
    if not pts:
        return None
    out, k = [], 0
    for t in trigger_index:
        while k + 1 < len(pts) and pts[k + 1][0] <= t:
            k += 1
        out.append(pts[k][1] if pts[0][0] <= t else "na")
    return out


def _resolve_entry(bars: pd.DataFrame, signal_i: int, sig, expiry: int):
    """Honest fill by entry type. Returns (entry_index, entry_price) or None."""
    n = len(bars)
    if sig.entry_type == su.MARKET:
        j = signal_i + 1
        if j >= n:
            return None
        return j, float(bars["open"].iloc[j])
    hi = min(n, signal_i + 1 + expiry)
    for j in range(signal_i + 1, hi):
        h = float(bars["high"].iloc[j])
        l = float(bars["low"].iloc[j])
        if sig.entry_type == su.LIMIT:
            if sig.direction == LONG and l <= sig.entry:
                return j, sig.entry
            if sig.direction != LONG and h >= sig.entry:
                return j, sig.entry
        else:  # STOP entry
            if sig.direction == LONG and h >= sig.entry:
                return j, sig.entry
            if sig.direction != LONG and l <= sig.entry:
                return j, sig.entry
    return None  # expired un-filled


def _cost_breakdown(sig, entry: float, market: Market, settings, bars_held: int,
                    tf_hours: float) -> tuple[float, float]:
    """(exec_r, funding_r): the execution cost (fees + slippage, both sides) and the funding/carry
    cost, each in R. Split out so the report can SEPARATE edge from execution (P10)."""
    per_unit = abs(entry - sig.stop)
    if per_unit <= 0:
        return 0.0, 0.0
    mcfg = settings.markets
    maker, taker = (mcfg.spot_maker, mcfg.spot_taker) if market is Market.SPOT \
        else (mcfg.usdm_maker, mcfg.usdm_taker)
    entry_fee = maker if sig.entry_type == su.LIMIT else taker
    slip = mcfg.slippage_bps / 1e4
    entry_slip = 0.0 if sig.entry_type == su.LIMIT else slip
    exec_frac = (entry_fee + taker) + (entry_slip + slip)        # both sides
    exec_r = (entry / per_unit) * exec_frac
    funding_r = 0.0
    if market is Market.USDM:
        hours = bars_held * tf_hours
        funding_frac = settings.backtest.assumed_funding_per_8h * (hours / 8.0)
        funding_r = (entry / per_unit) * funding_frac           # carry as a cost (conservative)
    return exec_r, funding_r


def _cost_r(sig, entry: float, market: Market, settings, bars_held: int, tf_hours: float) -> float:
    exec_r, funding_r = _cost_breakdown(sig, entry, market, settings, bars_held, tf_hours)
    return exec_r + funding_r


def _spawn_nulls(bars: pd.DataFrame, sig, regime: str, market: Market, settings,
                 tf_hours: float, rng, k: int, horizon: int, warmup: int) -> list:
    """The NULL baseline for one real trade: k SHADOW trades with the SAME direction
    and SAME risk geometry (stop distance + TP R-multiples), entered at RANDOM bars
    and managed by the same state machine. Their R distribution is what the exit +
    directional drift produce with NO predictive entry — the bar the real edge must beat."""
    n = len(bars)
    lo, hi = warmup, n - 2
    if k <= 0 or hi <= lo or sig.entry <= 0:
        return []
    risk_dist = sig.entry - sig.stop                         # signed (long > 0, short < 0)
    if risk_dist == 0:
        return []
    stop_frac = abs(risk_dist) / sig.entry                   # stop distance as a fraction of price
    if not (0.0 < stop_frac < 1.0):
        return []
    rr_targets = [(t - sig.entry) / risk_dist for t in sig.targets]   # R-multiples (>0 toward profit)
    cost_ns = SimpleNamespace(entry_type=sig.entry_type, stop=0.0)
    opens = bars["open"].to_numpy()
    out: list = []
    for j in rng.integers(lo, hi, size=k):
        j = int(j)
        e = float(opens[j])
        if e <= 0:
            continue
        shadow_stop = e * (1.0 - stop_frac) if sig.direction == LONG else e * (1.0 + stop_frac)
        shadow_risk = e - shadow_stop                        # same sign as risk_dist
        targets = [e + rr * shadow_risk for rr in rr_targets]
        seg = bars.iloc[j:j + horizon]
        r, held, _ = rk.simulate_detailed(sig.direction, e, shadow_stop, targets, seg, settings.risk_mgmt)
        cost_ns.stop = shadow_stop
        out.append(NullTrade(r - _cost_r(cost_ns, e, market, settings, held, tf_hours), regime))
    return out


def run_setups(bars: pd.DataFrame, market: Market, settings, tf: str, *,
               null_k: int | None = None, seed: int = 0, htf_trend: list | None = None) -> BacktestResult:
    """Backtest all setups on one coin -> real Trades (each tagged with point-in-time CONTEXT)
    AND the matched-geometry null shadows (unless ``null_k`` is 0). The shared detection walk
    feeds both. ``htf_trend`` (per-bar bias-TF trend) populates the real HTF context + ctx gate."""
    bt = settings.backtest
    n = len(bars)
    tf_hours = _tf_hours(tf)
    k = bt.null_k if null_k is None else null_k
    rng = np.random.default_rng(seed) if k > 0 else None
    next_free = {fn: bt.warmup for fn in su.DETECTORS}
    out: dict[str, list[Trade]] = {}
    nulls: dict[str, list] = {}

    for i in range(bt.warmup, n - 1):
        ws = max(0, i - bt.struct_window + 1)
        window = bars.iloc[ws:i + 1]
        struct = st.analyze_structure(window, settings.structure, with_volume_profile=False)
        if struct.state is None:
            continue
        htf_i = htf_trend[i] if (htf_trend is not None and i < len(htf_trend)) else None
        bar_ctx = _bar_context(window, struct, settings, htf_i)
        ctx = su.SetupContext(htf_trend=(htf_i if htf_i not in (None, "na") else struct.state.trend),
                              tide="neutral")
        regime = struct.state.trend

        for fn in su.DETECTORS:
            if i < next_free[fn]:
                continue                       # a position for this setup is open
            try:
                sig = fn(window, struct, ctx, settings.setups, market)
            except Exception:
                sig = None
            if sig is None:
                continue
            fill = _resolve_entry(bars, i, sig, sig.expiry_bars)
            if fill is None:
                continue
            entry_idx, entry_price = fill
            # Simulate FROM the entry bar (not entry_idx+1) so a same-bar stop-out is COUNTED
            # — skipping it hid same-bar stops and FABRICATED edge for tight-stop setups.
            # The entry bar opens at the fill price. For an INTRA-BAR (limit/stop) fill we can't
            # know the order of the bar's high/low vs the fill, so we clamp the FAVOURABLE extreme
            # to the entry — never crediting a same-bar target that may have printed BEFORE the fill
            # (that was the look-ahead) — while still catching a same-bar stop (the conservative side).
            after = bars.iloc[entry_idx:entry_idx + 400].copy()
            after.iloc[0, after.columns.get_loc("open")] = entry_price
            if sig.entry_type != su.MARKET:
                fav = "high" if sig.direction == LONG else "low"
                after.iloc[0, after.columns.get_loc(fav)] = entry_price
            r, held, exit_price = rk.simulate_detailed(
                sig.direction, entry_price, sig.stop, sig.targets, after, settings.risk_mgmt)
            exec_r, funding_r = _cost_breakdown(sig, entry_price, market, settings, held, tf_hours)
            r_net = r - exec_r - funding_r
            exit_idx = entry_idx + max(held - 1, 0)
            tctx = dict(bar_ctx)
            tctx["htf_align"] = _htf_align(bar_ctx.get("htf"), sig.direction)
            try:
                entry_ts = float(bars.index[entry_idx].timestamp())
            except Exception:
                entry_ts = 0.0
            out.setdefault(sig.setup, []).append(Trade(
                setup=sig.setup, direction=sig.direction, entry_index=entry_idx,
                entry_price=entry_price, exit_index=exit_idx, exit_price=exit_price,
                r=r_net, regime=regime, bars_held=held, context=tctx,
                gross_r=r, funding_r=funding_r, entry_ts=entry_ts))
            next_free[fn] = exit_idx + 1       # no overlap for this setup

            if rng is not None:
                bucket = nulls.setdefault(sig.setup, [])
                if len(bucket) < bt.null_cap:
                    bucket.extend(_spawn_nulls(bars, sig, regime, market, settings, tf_hours,
                                               rng, k, bt.null_horizon, bt.warmup))

    return BacktestResult(trades=out, nulls=nulls)


def backtest_all(bars: pd.DataFrame, market: Market, settings, tf: str) -> dict[str, list[Trade]]:
    """Backtest all setups on one coin; returns {setup_name: [Trade, ...]}.

    Thin wrapper over :func:`run_setups` with the null disabled — for the optimizer
    and callers that only need the real trades."""
    return run_setups(bars, market, settings, tf, null_k=0).trades


def fetch_history(market: Market, symbols, settings, tf: str) -> dict:
    """Fetch + clean closed history once per coin (shared by the pooled edge profiler
    and the optimizer). Coins that fail to fetch are skipped, not raised."""
    exch = dfetch.make_exchange(market, settings.api_key, settings.api_secret)
    dfetch.load_markets(exch)
    out: dict = {}
    for sym in symbols:
        try:
            out[sym] = dfetch.drop_unclosed(
                exch, dfetch.fetch_ohlcv(exch, sym, tf, settings.backtest.candle_limit), tf)
        except dfetch.DataError:
            pass
    return out


def run_backtest(market: Market, symbol: str, settings, tf: str,
                 setup_name: str | None = None) -> tuple[dict[str, ex.EdgeProfile], int, dict]:
    """Fetch real history and evaluate every setup (or one) into EdgeProfiles. Returns
    (profiles, n_candles, data_quality) — the ACTUAL closed candles used (post drop-unclosed)
    and the sanitation report (audit finding 6), for honest reporting."""
    bcfg = settings.backtest
    exch = dfetch.make_exchange(market, settings.api_key, settings.api_secret)
    dfetch.load_markets(exch)
    raw = dfetch.fetch_ohlcv(exch, symbol, tf, bcfg.candle_limit)
    quality = dict(raw.attrs.get("quality") or {})
    bars = dfetch.drop_unclosed(exch, raw, tf)

    res = run_setups(bars, market, settings, tf)
    n_combos = len(su.DETECTORS)
    profiles: dict[str, ex.EdgeProfile] = {}
    names = [setup_name] if setup_name else sorted(res.trades)
    for name in names:
        profiles[name] = ex.evaluate(res.trades.get(name, []), setup=name,
                                     n_combos_tested=n_combos, cfg=bcfg,
                                     null_trades=res.nulls.get(name, []))
    return profiles, len(bars), quality
