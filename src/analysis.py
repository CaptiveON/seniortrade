"""Layer 2 — analysis: the six-lens, multi-timeframe synthesis.

Runs the six lenses across a TF stack (BIAS → TRIGGER) and FUSES them into one
seasoned-trader read: an overall bias + a confluence score, the explicit CONFLICT
(bear case), where price IS (location), and the concrete INVALIDATION level.

It DESCRIBES — it never predicts. Readings are observations, not signals; only a
backtested setup (Layer 3/5) becomes a signal. "No clean read" is a valid output.

Pure scoring/aggregation functions are network-free and unit-tested; analyze()
orchestrates the live fetch + structure + markets + context.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import data_fetch as dfetch
from . import indicators as ind
from . import market_context as mc
from . import risk as rk
from . import setups as su
from . import structure as st
from .config import AnalysisConfig, Market
from .markets import make_market
from .markets.base import FundingRead, OIRead

BULL = "bull"
BEAR = "bear"
NEUTRAL = "neutral"


@dataclass
class LensRead:
    name: str
    direction: str        # BULL / BEAR / NEUTRAL
    score: float          # signed strength in [-1, +1]
    note: str


@dataclass
class AnalysisResult:
    symbol: str
    market: Market
    tf_bias: str
    tf_trigger: str
    last_price: float
    htf_trend: str                       # structure trend on the bias TF
    bias: str                            # overall BULL / BEAR / NEUTRAL
    net_score: float
    confluence: float                    # 0..1 (|net|)
    alignment: str                       # aligned / mixed / opposed
    agree_count: int
    lenses: list[LensRead] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    location: str = "mid_range"
    freshness: str = "fresh"
    invalidation: float | None = None
    invalidation_note: str = ""
    rs_btc: float = float("nan")
    posture: str = mc.NEUTRAL
    funding: FundingRead | None = None
    oi: OIRead | None = None
    structure: st.Structure | None = None
    htf_structure: st.Structure | None = None
    # illustrative trade scaffold (entry/stop/target by bias + structure)
    side: str = "long"
    entry: float = float("nan")
    stop: float | None = None
    target: float | None = None
    sizing: object | None = None     # markets.SizingResult (leverage/margin/liq)
    setups: list = field(default_factory=list)   # setups.Signal list (ranked)
    plan: object | None = None       # risk.TradePlan for the best setup (net + blended)
    context: dict = field(default_factory=dict)  # P1: point-in-time context at the trigger bar
    notes: list[str] = field(default_factory=list)
    risk_pct: float = 0.0            # the EFFECTIVE risk % the plan was sized with
    risk_notes: list[str] = field(default_factory=list)   # which effective-risk stages fired
    risk_inputs: dict = field(default_factory=dict)       # P12: the sizing evidence (edge/conf/σ_R/sample)


# --------------------------------------------------------------------------- #
# Pure helpers (testable)
# --------------------------------------------------------------------------- #
def _dir(score: float, thr: float = 0.15) -> str:
    if score >= thr:
        return BULL
    if score <= -thr:
        return BEAR
    return NEUTRAL


def _clip(x: float) -> float:
    return max(-1.0, min(1.0, x))


def detect_rsi_divergence(swings, rsi_series: pd.Series) -> tuple[str, str, float]:
    """Regular divergence between price swings and RSI at those swings."""
    n = len(rsi_series)

    def rsi_at(i: int) -> float:
        return float(rsi_series.iloc[i]) if 0 <= i < n else float("nan")

    highs = [s for s in swings if s.kind == st.HIGH][-2:]
    lows = [s for s in swings if s.kind == st.LOW][-2:]
    if len(highs) == 2:
        a, b = highs
        ra, rb = rsi_at(a.index), rsi_at(b.index)
        if b.price > a.price and rb < ra and ra == ra and rb == rb:
            return "bearish", "price higher-high but RSI lower-high (bearish divergence)", -1.0
    if len(lows) == 2:
        a, b = lows
        ra, rb = rsi_at(a.index), rsi_at(b.index)
        if b.price < a.price and rb > ra and ra == ra and rb == rb:
            return "bullish", "price lower-low but RSI higher-low (bullish divergence)", 1.0
    return "none", "", 0.0


def classify_location(struct: st.Structure, last: float, atr_val: float, cfg: AnalysisConfig) -> str:
    tol = cfg.location_atr_frac * atr_val if atr_val and atr_val > 0 else last * 0.005
    res = struct.resistance_above(last)
    sup = struct.support_below(last)
    if res is not None and (res - last) <= tol:
        return "at_resistance"
    if sup is not None and (last - sup) <= tol:
        return "at_support"
    # retracement golden pocket (0.5–0.618 of the last leg)
    r = struct.retracements
    if r:
        try:
            lo = min(r["0.618"], r["0.500"])
            hi = max(r["0.618"], r["0.500"])
            if lo <= last <= hi:
                return "golden_pocket"
        except KeyError:
            pass
    vp = struct.volume_profile
    if vp is not None:
        if abs(last - vp.vah) <= tol or abs(last - vp.val) <= tol:
            return "value_edge"
    if not np.isnan(struct.range_high) and last >= struct.range_high:
        return "breakout_up"
    if not np.isnan(struct.range_low) and last <= struct.range_low:
        return "breakout_down"
    return "mid_range"


def _lens_trend(struct: st.Structure, ma_fast: float, ma_slow: float) -> LensRead:
    state = struct.state
    tc = 1.0 if state.trend == st.UP else -1.0 if state.trend == st.DOWN else 0.0
    mc_ = 1.0 if ma_fast > ma_slow else -1.0 if ma_fast < ma_slow else 0.0
    score = _clip(0.6 * tc + 0.4 * mc_)
    if state.event == "CHoCH_down":
        score = _clip(score - 0.3)
    elif state.event == "CHoCH_up":
        score = _clip(score + 0.3)
    seq = "/".join(state.sequence[-3:]) if state.sequence else "—"
    note = f"structure {state.trend} [{seq}] event={state.event}; MA{'+' if mc_>0 else '-' if mc_<0 else '~'}"
    return LensRead("trend", _dir(score), score, note)


def _lens_momentum(rsi_last: float, macd_hist: float, divergence: tuple[str, str, float],
                   cfg: AnalysisConfig) -> LensRead:
    rsi_comp = _clip((rsi_last - 50.0) / 50.0) if rsi_last == rsi_last else 0.0
    macd_comp = 1.0 if macd_hist > 0 else -1.0 if macd_hist < 0 else 0.0
    div_kind, div_note, div_comp = divergence
    score = _clip(0.5 * rsi_comp + 0.3 * macd_comp + 0.2 * div_comp)
    note = f"RSI {rsi_last:.0f}; MACD hist {'+' if macd_hist>0 else '-'}"
    if div_kind != "none":
        note += f"; {div_note}"
    return LensRead("momentum", _dir(score), score, note)


def _lens_volatility(vol_regime: str, freshness: str, extension: float) -> LensRead:
    note = f"{vol_regime}; {freshness} ({extension:+.1f}σ from MA)"
    return LensRead("volatility", NEUTRAL, 0.0, note)


def _lens_volume(obv_slope: float, cvd_slope: float) -> LensRead:
    ocomp = 1.0 if obv_slope > 0 else -1.0 if obv_slope < 0 else 0.0
    ccomp = 1.0 if cvd_slope > 0 else -1.0 if cvd_slope < 0 else 0.0
    score = _clip(0.5 * ocomp + 0.5 * ccomp)
    note = f"OBV {'rising' if ocomp>0 else 'falling' if ocomp<0 else 'flat'}; " \
           f"CVD {'accumulation' if ccomp>0 else 'distribution' if ccomp<0 else 'flat'}"
    return LensRead("volume", _dir(score), score, note)


def _lens_levels(location: str) -> LensRead:
    mapping = {
        "at_support": 0.6, "at_resistance": -0.6,
        "breakout_up": 0.5, "breakout_down": -0.5,
        "golden_pocket": 0.3, "value_edge": 0.0, "mid_range": 0.0,
    }
    score = mapping.get(location, 0.0)
    return LensRead("levels", _dir(score), score, f"location: {location}")


def _lens_context(rs_btc: float, posture: str, funding: FundingRead | None,
                  market: Market) -> LensRead:
    rs_comp = _clip(rs_btc / 20.0) if rs_btc == rs_btc else 0.0
    posture_comp = 1.0 if posture == mc.RISK_ON else -1.0 if posture == mc.RISK_OFF else 0.0
    fund_comp = 0.0
    note = f"RS {rs_btc:+.0f}% vs BTC; tide {posture}"
    if market is Market.USDM and funding is not None:
        if funding.signal == "crowded_long":
            fund_comp = -0.3   # contrarian
            note += "; funding crowded-long (contrarian -)"
        elif funding.signal == "crowded_short":
            fund_comp = 0.3
            note += "; funding crowded-short (contrarian +)"
    score = _clip(0.5 * rs_comp + 0.4 * posture_comp + fund_comp)
    return LensRead("context", _dir(score), score, note)


def aggregate(lenses: list[LensRead], cfg: AnalysisConfig, htf_trend: str) -> tuple[float, str, float, str, int]:
    """Weighted fusion of lens scores → (net, bias, confluence, alignment, agree_count)."""
    weights = {
        "trend": cfg.w_trend, "momentum": cfg.w_momentum, "volume": cfg.w_volume,
        "levels": cfg.w_levels, "context": cfg.w_context, "volatility": 0.0,
    }
    net = sum(weights.get(l.name, 0.0) * l.score for l in lenses)

    # HTF gating: don't fight the higher-timeframe bias.
    gated = False
    if htf_trend == st.DOWN and net > 0.05:
        net, gated = 0.05, True
    elif htf_trend == st.UP and net < -0.05:
        net, gated = -0.05, True

    bias = _dir(net, cfg.bias_threshold)
    confluence = min(abs(net), 1.0)
    sign = 1 if net > 0 else -1 if net < 0 else 0
    agree = sum(1 for l in lenses if weights.get(l.name, 0.0) > 0 and np.sign(l.score) == sign and sign != 0)

    if htf_trend == st.RANGE:
        alignment = "mixed"
    elif (htf_trend == st.UP and bias == BULL) or (htf_trend == st.DOWN and bias == BEAR):
        alignment = "aligned"
    elif bias == NEUTRAL:
        alignment = "mixed"
    else:
        alignment = "opposed"
    return net, bias, confluence, alignment, agree


def _invalidation(struct: st.Structure, bias: str) -> tuple[float | None, str]:
    last = struct.last_price
    if bias == BULL:
        lvl = struct.state.last_low if struct.state and not np.isnan(struct.state.last_low) else None
        lvl = lvl if (lvl and lvl < last) else struct.support_below(last)
        if lvl:
            return lvl, f"a close below {lvl:,.4g} breaks the structure (thesis void)"
    elif bias == BEAR:
        lvl = struct.state.last_high if struct.state and not np.isnan(struct.state.last_high) else None
        lvl = lvl if (lvl and lvl > last) else struct.resistance_above(last)
        if lvl:
            return lvl, f"a close above {lvl:,.4g} breaks the structure (thesis void)"
    return None, "no directional thesis — stand aside"


def _collect_conflicts(lenses: list[LensRead], bias: str, location: str,
                       divergence, oi: OIRead | None, extension: float,
                       posture: str, htf_trend: str) -> list[str]:
    conflicts: list[str] = []
    sign = 1 if bias == BULL else -1 if bias == BEAR else 0
    for l in lenses:
        if l.name == "volatility" or sign == 0:
            continue
        if np.sign(l.score) == -sign and l.score != 0:
            conflicts.append(f"{l.name} disagrees ({l.note})")
    if divergence[0] != "none":
        conflicts.append(divergence[1])
    if bias == BULL and location == "at_resistance":
        conflicts.append("price into resistance on a long bias")
    if bias == BEAR and location == "at_support":
        conflicts.append("price into support on a short bias")
    if oi is not None and oi.trend == "falling":
        conflicts.append("open interest falling — move may be hollow (deleveraging)")
    if abs(extension) >= 2.5:
        conflicts.append(f"extended {extension:+.1f}σ from its MA — late entry risk")
    if htf_trend == st.DOWN and bias == BULL:
        conflicts.append("counter-tide: bullish read against a bearish higher timeframe")
    if htf_trend == st.UP and bias == BEAR:
        conflicts.append("counter-tide: bearish read against a bullish higher timeframe")
    return conflicts


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def _slope(series: pd.Series, lookback: int = 10) -> float:
    s = series.dropna()
    if len(s) <= lookback:
        return 0.0
    return float(s.iloc[-1] - s.iloc[-1 - lookback])


def analyze(
    market: Market,
    symbol: str,
    settings,
    tf_trigger: str | None = None,
    ex=None,
    drawdown_pct: float = 0.0,
) -> AnalysisResult:
    cfg: AnalysisConfig = settings.analysis
    tf_trigger = tf_trigger or cfg.tf_trigger
    if ex is None:                       # batch callers pass a per-thread exchange (avoids re-loading markets)
        ex = dfetch.make_exchange(market, settings.api_key, settings.api_secret)
        dfetch.load_markets(ex)

    def fetch(sym: str, tf: str) -> pd.DataFrame:
        raw = dfetch.fetch_ohlcv(ex, sym, tf, cfg.candle_limit)
        return dfetch.drop_unclosed(ex, raw, tf)

    trig = fetch(symbol, tf_trigger)
    bias_df = fetch(symbol, cfg.tf_bias)

    struct = st.analyze_structure(trig, settings.structure)
    htf_struct = st.analyze_structure(bias_df, settings.structure)
    last = float(trig["close"].iloc[-1])
    atr_val = float(ind.atr(trig, settings.structure.atr_length).iloc[-1])
    try:                                             # P1: point-in-time context (same fn the backtest tags trades with)
        from . import backtest as _bt
        context = _bt._bar_context(trig, struct, settings, htf_struct.state.trend if htf_struct.state else None)
    except Exception:
        context = {}
    if market is Market.USDM:
        try:                                          # A1/A3 live buckets (PIT: trailing data only)
            from . import derivs as dv
            ts_ms = [int(t.timestamp() * 1000) for t in trig.index]
            f_ctx, _ = dv.align_funding(ts_ms, dv.funding_series(ex, market, symbol))
            o_ctx = dv.align_oi(ts_ms, dv.oi_series(market, symbol, tf_trigger), tf_trigger)
            context["fund"] = f_ctx[-1] if f_ctx else None
            context["oi"] = o_ctx[-1] if o_ctx else None
        except Exception:
            context.setdefault("fund", None)
            context.setdefault("oi", None)

    # indicators on the trigger TF
    rsi_s = ind.rsi(trig, cfg.rsi_length)
    macd_df = ind.macd(trig)
    obv_s = ind.obv(trig)
    cvd_s = ind.cvd_proxy(trig)
    ma_fast = float(ind.sma(trig["close"], cfg.ma_fast).iloc[-1])
    ma_slow = float(ind.sma(trig["close"], cfg.ma_slow).iloc[-1])
    vol_regime, _ = ind.volatility_regime(trig, settings.structure.atr_length)
    ext = float(ind.extension_atr(trig, settings.screener.ext_ma_len, settings.structure.atr_length).iloc[-1])
    freshness = "fresh" if abs(ext) < 1.5 else "stretched" if abs(ext) < 2.5 else "extended"

    divergence = detect_rsi_divergence(struct.swings, rsi_s)
    location = classify_location(struct, last, atr_val, cfg)

    # market context (the tide + RS vs BTC) on the bias TF
    posture = mc.NEUTRAL
    rs_btc = float("nan")
    btc_sym = mc.reference_symbol("BTC", market, settings.screener.quote)
    if symbol != btc_sym:
        try:
            btc_bias = fetch(btc_sym, cfg.tf_bias)
            posture = mc.assess_market(btc_bias, btc_sym, settings.screener.adx_length,
                                       settings.screener.adx_trending, settings.screener.adx_choppy).posture
            rs_btc = mc.relative_strength(bias_df["close"], btc_bias["close"], cfg.rs_window).rel_strength_pct
        except dfetch.DataError:
            pass
    else:
        rs_btc = 0.0
        posture = mc.assess_market(bias_df, btc_sym, settings.screener.adx_length,
                                   settings.screener.adx_trending, settings.screener.adx_choppy).posture

    mkt = make_market(market, ex, settings.markets)
    funding = oi = None
    if market is Market.USDM:
        funding = mkt.funding(symbol)
        oi = mkt.open_interest(symbol)

    # --- lenses ---
    lenses = [
        _lens_trend(struct, ma_fast, ma_slow),
        _lens_momentum(float(rsi_s.iloc[-1]), float(macd_df["hist"].iloc[-1]), divergence, cfg),
        _lens_volatility(vol_regime, freshness, ext),
        _lens_volume(_slope(obv_s), _slope(cvd_s)),
        _lens_levels(location),
        _lens_context(rs_btc, posture, funding, market),
    ]

    net, bias, confluence, alignment, agree = aggregate(lenses, cfg, htf_struct.state.trend)
    invalidation, inval_note = _invalidation(struct, bias)
    conflicts = _collect_conflicts(lenses, bias, location, divergence, oi, ext,
                                   posture, htf_struct.state.trend)

    # illustrative trade scaffold (structure-anchored)
    side = "long" if bias == BULL else "short" if bias == BEAR else "long"
    entry = last
    stop = invalidation
    target = struct.resistance_above(last) if bias == BULL else struct.support_below(last) if bias == BEAR else None

    # named, backtestable setups detected on the trigger bar (look-ahead-safe)
    setup_ctx = su.SetupContext(htf_trend=htf_struct.state.trend, tide=posture)
    detected = su.detect_all(trig, struct, setup_ctx, settings.setups, market)

    # a validated, net-cost trade plan for the best setup, sized by the EFFECTIVE-RISK pipeline
    plan = None
    risk_pct_eff = settings.risk.risk_pct
    risk_notes: list[str] = []
    risk_inputs: dict = {}
    if detected:
        top = detected[0]
        mgmt = settings.risk_mgmt
        regime = struct.state.trend if struct.state else ""
        si = None
        try:                                          # always read the sizing evidence (for guidance);
            from . import edge_score as _es           # the scaling STAGES stay opt-in inside effective_risk_pct
            si = _es.sizing_inputs(market, tf_trigger, top.setup, regime, settings.edge, symbol=symbol,
                                   context=_es.resolve_context(context, top.direction))
        except Exception:
            si = None
        adj = rk.effective_risk_pct(
            settings.risk.risk_pct, mgmt, drawdown_pct=drawdown_pct,
            edge_r=(si or {}).get("edge_r"), confidence=(si or {}).get("confidence"),
            sigma_r=(si or {}).get("sigma_r"),
            win_rate=(si or {}).get("win_rate"), avg_win_r=(si or {}).get("avg_win_r"),
            avg_loss_r=(si or {}).get("avg_loss_r"))
        risk_pct_eff, risk_notes = adj.effective_pct, adj.notes
        risk_inputs = dict(si or {}, base_pct=settings.risk.risk_pct,
                           effective_pct=risk_pct_eff, drawdown_pct=drawdown_pct)
        try:
            plan = rk.plan_trade(
                mkt, symbol=symbol, side=top.direction, entry=top.entry, stop=top.stop,
                targets=top.targets, account_equity=settings.risk.account_equity,
                risk_pct=risk_pct_eff, mgmt=settings.risk_mgmt,
                funding_rate=(funding.rate if funding else None),
            )
        except Exception:
            plan = None

    sizing = None
    if bias != NEUTRAL and stop is not None:
        risk_amount = settings.risk.account_equity * settings.risk.risk_pct / 100.0
        try:
            sizing = make_market(market, ex, settings.markets).size_from_risk(
                symbol, entry, stop, risk_amount, side
            )
        except Exception:
            sizing = None

    return AnalysisResult(
        symbol=symbol, market=market, tf_bias=cfg.tf_bias, tf_trigger=tf_trigger,
        last_price=last, htf_trend=htf_struct.state.trend, bias=bias, net_score=net,
        confluence=confluence, alignment=alignment, agree_count=agree, lenses=lenses,
        conflicts=conflicts, location=location, freshness=freshness, invalidation=invalidation,
        invalidation_note=inval_note, rs_btc=rs_btc, posture=posture, funding=funding,
        oi=oi, structure=struct, htf_structure=htf_struct, side=side, entry=entry,
        stop=stop, target=target, sizing=sizing, setups=detected, plan=plan,
        risk_pct=risk_pct_eff, risk_notes=risk_notes, context=context, risk_inputs=risk_inputs,
    )


_GRADE_ORDER = {"A": 0, "B": 1, "C": 2, "D": 3}


def rank_board(results, setups_only: bool = False) -> list:
    """Order a batch of AnalysisResults for the `analyze --all` board.

    Coins with a VALID, net-cost plan come first (best grade, then best net R:R),
    then coins with a setup but no valid plan, then NO-TRADE coins. `setups_only`
    drops the NO-TRADE rows entirely.
    """
    def has_plan(r) -> bool:
        return getattr(r, "plan", None) is not None and bool(getattr(r.plan, "valid", False))

    def key(r):
        setups = getattr(r, "setups", None) or []
        if setups and has_plan(r):
            return (0, _GRADE_ORDER.get(setups[0].grade, 9), -(getattr(r.plan, "net_rr", 0.0) or 0.0))
        if setups:
            return (1, _GRADE_ORDER.get(setups[0].grade, 9), 0.0)
        return (2, 9, 0.0)

    rows = [r for r in results if (getattr(r, "setups", None) or [])] if setups_only else list(results)
    return sorted(rows, key=key)
