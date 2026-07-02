"""Layer 3 — setups: named, backtestable trade patterns.

A setup is a complete CONTRACT; detecting one emits a :class:`Signal` that stage,
backtest, and edge_score consume IDENTICALLY.

DETECTION CONTRACT — every detector is a PURE function over CLOSED bars:
    detect(bars, structure, ctx, cfg, market) -> Signal | None
where ``bars`` is OHLCV up to and including the decision bar (the last row is the
just-closed bar) and ``structure`` is computed on that SAME slice. It uses only
data through the decision bar — no look-ahead — so the live detector and the
backtest run the same code. The backtest loop calls detect(bars.iloc[:i+1], ...).

Parameters are FIXED defaults (SetupConfig) — no per-coin curve-fitting here;
optimisation belongs to the walk-forward engine (Layer 5).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import indicators as ind
from . import market_context as mc
from . import structure as st
from .config import Market, SetupConfig
from .markets import LONG, SHORT

# entry types (drive fill + slippage modelling in the backtest)
MARKET = "market"   # fill at the decision-bar close (on confirmation)
LIMIT = "limit"     # rest a limit at a level (may not fill if price gaps through)
STOP = "stop"       # stop-entry on a break (fills with slippage)


@dataclass
class SetupContext:
    """External gates the bars alone can't provide (from higher-TF / market)."""

    htf_trend: str = st.RANGE          # up / down / range
    tide: str = mc.NEUTRAL             # risk-on / risk-off / neutral


@dataclass
class Signal:
    setup: str
    direction: str                     # long / short
    market: Market
    entry_type: str                    # MARKET / LIMIT / STOP
    entry: float
    stop: float
    targets: list[float]               # TP1, TP2
    invalidation: float                # pre-entry level that voids the setup
    expiry_bars: int                   # time-stop for a pending setup
    rr: float                          # reward(TP1):risk, gross
    regime: str                        # regime it is valid in
    grade: str                         # A / B / C (provisional until backtest)
    reasons: list[str] = field(default_factory=list)
    price_ref: float = float("nan")    # decision-bar close


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _atr(bars: pd.DataFrame, cfg: SetupConfig) -> float:
    return float(ind.atr(bars, cfg.atr_length).iloc[-1])


def _ema(bars: pd.DataFrame, cfg: SetupConfig) -> float:
    return float(ind.ema(bars["close"], cfg.ema_len).iloc[-1])


def _counter_tide(direction: str, tide: str) -> bool:
    return (direction == LONG and tide == mc.RISK_OFF) or \
           (direction == SHORT and tide == mc.RISK_ON)


def _grade(rr: float, n_reasons: int, counter_tide: bool, cfg: SetupConfig) -> str:
    """PROVISIONAL grade (confluence + R:R). Backtested expectancy folds in at
    Layer 5/edge_score; thin R:R / counter-tide can never be A."""
    if rr < cfg.min_rr:
        return "C"
    if counter_tide:
        return "B" if (rr >= 2.0 and n_reasons >= 3) else "C"
    if rr >= 2.0 and n_reasons >= 3:
        return "A"
    if rr >= 1.5 and n_reasons >= 2:
        return "B"
    return "C"


def _build(
    name: str, direction: str, market: Market, entry_type: str,
    entry: float, stop: float, tp1: float | None, invalidation: float,
    structure: st.Structure, cfg: SetupConfig, ctx: SetupContext,
    reasons: list[str], regime: str,
) -> Signal | None:
    risk = abs(entry - stop)
    if risk <= 0:
        return None
    if tp1 is None or (direction == LONG and tp1 <= entry) or (direction == SHORT and tp1 >= entry):
        tp1 = entry + cfg.rr_target_default * risk if direction == LONG \
            else entry - cfg.rr_target_default * risk
    if direction == LONG:
        tp2 = max(structure.range_high, tp1 + risk) if not np.isnan(structure.range_high) else tp1 + risk
        reward = tp1 - entry
    else:
        tp2 = min(structure.range_low, tp1 - risk) if not np.isnan(structure.range_low) else tp1 - risk
        reward = entry - tp1
    rr = reward / risk

    ct = _counter_tide(direction, ctx.tide)
    reasons = list(reasons)
    reasons.append("counter-tide" if ct else f"with-tide ({ctx.tide})")
    if (direction == LONG and ctx.htf_trend == st.UP) or (direction == SHORT and ctx.htf_trend == st.DOWN):
        reasons.append(f"HTF aligned ({ctx.htf_trend})")

    grade = _grade(rr, len(reasons), ct, cfg)
    return Signal(
        setup=name, direction=direction, market=market, entry_type=entry_type,
        entry=entry, stop=stop, targets=[tp1, tp2], invalidation=invalidation,
        expiry_bars=cfg.expiry_bars, rr=rr, regime=regime, grade=grade,
        reasons=reasons, price_ref=entry,
    )


def _ohlc(bars: pd.DataFrame):
    last = bars.iloc[-1]
    return float(last["open"]), float(last["high"]), float(last["low"]), float(last["close"])


# --------------------------------------------------------------------------- #
# Setups
# --------------------------------------------------------------------------- #
def detect_trend_pullback(bars, structure, ctx, cfg, market) -> Signal | None:
    """Trend regime: buy a pullback to a holding level/MA in an uptrend (mirror short)."""
    state = structure.state
    if state is None:
        return None
    o, h, l, c = _ohlc(bars)
    atr = _atr(bars, cfg)
    if not atr > 0:
        return None
    ema_v = _ema(bars, cfg)

    if state.trend == st.UP:
        sup = structure.support_below(c)
        ref = sup if (sup is not None and sup < c) else ema_v
        if not (l <= ref + cfg.zone_atr * atr and c >= ref and c > o):
            return None
        anchor = state.last_low if (not np.isnan(state.last_low) and state.last_low < c) else min(ref, l)
        stop = min(anchor, l) - cfg.stop_buffer_atr * atr
        reasons = [f"uptrend pullback to {'support' if sup else 'EMA'} {ref:,.4g}"]
        return _build("trend_pullback", LONG, market, MARKET, c, stop,
                      structure.resistance_above(c), ref, structure, cfg, ctx, reasons, "trend")

    if state.trend == st.DOWN and market is Market.USDM:
        res = structure.resistance_above(c)
        ref = res if (res is not None and res > c) else ema_v
        if not (h >= ref - cfg.zone_atr * atr and c <= ref and c < o):
            return None
        anchor = state.last_high if (not np.isnan(state.last_high) and state.last_high > c) else max(ref, h)
        stop = max(anchor, h) + cfg.stop_buffer_atr * atr
        reasons = [f"downtrend pullback to {'resistance' if res else 'EMA'} {ref:,.4g}"]
        return _build("trend_pullback", SHORT, market, MARKET, c, stop,
                      structure.support_below(c), ref, structure, cfg, ctx, reasons, "trend")
    return None


def detect_breakout_retest(bars, structure, ctx, cfg, market) -> Signal | None:
    """Trend/expansion: enter the successful retest of a broken level (mirror short)."""
    state = structure.state
    o, h, l, c = _ohlc(bars)
    atr = _atr(bars, cfg)
    if not atr > 0:
        return None
    window = bars["close"].iloc[-cfg.breakout_lookback:]

    # LONG: a level now acting as support that price recently broke above.
    below = [lv.price for lv in structure.levels if lv.price < c]
    if below:
        level = max(below)
        broke = float(window.min()) < level
        retest = l <= level + cfg.zone_atr * atr and c >= level and c > o
        if broke and retest and (state is None or state.trend in (st.UP, st.RANGE)):
            stop = level - cfg.stop_buffer_atr * atr
            measured = level + (level - float(window.min()))
            tp1 = structure.resistance_above(c) or measured
            reasons = [f"break+retest of {level:,.4g} as support"]
            return _build("breakout_retest", LONG, market, LIMIT, level, stop, tp1,
                          level, structure, cfg, ctx, reasons, "trend")

    # SHORT (usdm): a level now acting as resistance that price recently broke below.
    if market is Market.USDM:
        above = [lv.price for lv in structure.levels if lv.price > c]
        if above:
            level = min(above)
            broke = float(window.max()) > level
            retest = h >= level - cfg.zone_atr * atr and c <= level and c < o
            if broke and retest and (state is None or state.trend in (st.DOWN, st.RANGE)):
                stop = level + cfg.stop_buffer_atr * atr
                measured = level - (float(window.max()) - level)
                tp1 = structure.support_below(c) or measured
                reasons = [f"break+retest of {level:,.4g} as resistance"]
                return _build("breakout_retest", SHORT, market, LIMIT, level, stop, tp1,
                              level, structure, cfg, ctx, reasons, "trend")
    return None


def detect_failed_breakout(bars, structure, ctx, cfg, market) -> Signal | None:
    """Range/liquidity: a sweep beyond the range that reclaims (spring / upthrust)."""
    state = structure.state
    o, h, l, c = _ohlc(bars)
    atr = _atr(bars, cfg)
    if not atr > 0:
        return None
    rl, rh = structure.range_low, structure.range_high

    if not np.isnan(rl) and l < rl and c > rl and c > o:        # swept below, reclaimed -> long
        stop = l - cfg.stop_buffer_atr * atr
        tp1 = rh if not np.isnan(rh) else structure.resistance_above(c)
        reasons = [f"failed breakdown: swept {rl:,.4g} and reclaimed (spring)"]
        return _build("failed_breakout", LONG, market, MARKET, c, stop, tp1, rl,
                      structure, cfg, ctx, reasons, "range")

    if market is Market.USDM and not np.isnan(rh) and h > rh and c < rh and c < o:  # upthrust -> short
        stop = h + cfg.stop_buffer_atr * atr
        tp1 = rl if not np.isnan(rl) else structure.support_below(c)
        reasons = [f"failed breakout: swept {rh:,.4g} and rejected (upthrust)"]
        return _build("failed_breakout", SHORT, market, MARKET, c, stop, tp1, rh,
                      structure, cfg, ctx, reasons, "range")
    return None


def detect_range_fade(bars, structure, ctx, cfg, market) -> Signal | None:
    """Non-trending: fade a clean range extreme back toward the middle (mirror short)."""
    state = structure.state
    if state is None or state.trend != st.RANGE:
        return None
    o, h, l, c = _ohlc(bars)
    atr = _atr(bars, cfg)
    if not atr > 0 or np.isnan(structure.range_low) or np.isnan(structure.range_high):
        return None
    rl, rh = structure.range_low, structure.range_high
    mid = (rl + rh) / 2.0

    if l <= rl + cfg.zone_atr * atr and c > rl and c > o:       # bounce off range low -> long
        stop = rl - cfg.stop_buffer_atr * atr
        reasons = [f"range fade off the low {rl:,.4g}"]
        return _build("range_fade", LONG, market, LIMIT, rl, stop, mid, rl,
                      structure, cfg, ctx, reasons, "range")

    if market is Market.USDM and h >= rh - cfg.zone_atr * atr and c < rh and c < o:  # fade the high -> short
        stop = rh + cfg.stop_buffer_atr * atr
        reasons = [f"range fade off the high {rh:,.4g}"]
        return _build("range_fade", SHORT, market, LIMIT, rh, stop, mid, rh,
                      structure, cfg, ctx, reasons, "range")
    return None


def detect_breakout_momentum(bars, structure, ctx, cfg, market) -> Signal | None:
    """Breakout/expansion: enter ON a decisive break of the recent range (no retest).

    Catches the move that runs without looking back. STOP entry beyond the breakout
    bar (only fills on continuation); ADX-gated so we never chase a break in chop."""
    o, h, l, c = _ohlc(bars)
    atr = _atr(bars, cfg)
    if not atr > 0 or len(bars) <= cfg.breakout_lookback + 1:
        return None
    adx_val = float(ind.adx(bars, cfg.atr_length)["adx"].iloc[-1])
    if not adx_val >= cfg.momentum_adx_floor:          # only when the regime can sustain a run
        return None
    prior = bars.iloc[-(cfg.breakout_lookback + 1):-1]  # the range BEFORE the breakout bar
    prior_high = float(prior["high"].max())
    prior_low = float(prior["low"].min())
    rng = prior_high - prior_low
    pad = cfg.momentum_break_atr * atr

    if c > prior_high + pad and c > o:                  # LONG: decisive break of the prior high
        stop = prior_high - cfg.stop_buffer_atr * atr
        tp1 = structure.resistance_above(h) or (h + rng)
        reasons = [f"momentum breakout > {prior_high:,.4g} (ADX {adx_val:.0f}, +{(c - prior_high) / atr:.1f} ATR)"]
        return _build("breakout_momentum", LONG, market, STOP, h, stop, tp1, prior_high,
                      structure, cfg, ctx, reasons, "trend")

    if market is Market.USDM and c < prior_low - pad and c < o:   # SHORT: decisive breakdown
        stop = prior_low + cfg.stop_buffer_atr * atr
        tp1 = structure.support_below(l) or (l - rng)
        reasons = [f"momentum breakdown < {prior_low:,.4g} (ADX {adx_val:.0f}, -{(prior_low - c) / atr:.1f} ATR)"]
        return _build("breakout_momentum", SHORT, market, STOP, l, stop, tp1, prior_low,
                      structure, cfg, ctx, reasons, "trend")
    return None


def detect_liquidity_sweep_reversal(bars, structure, ctx, cfg, market) -> Signal | None:
    """Reversal/trap: a wick beyond an equal-highs/lows liquidity POOL that reclaims.

    The stop-run that marks a top/bottom before a big move — tight stop beyond the
    sweep wick. Sharper than failed_breakout (pool-based, wick-confirmed)."""
    o, h, l, c = _ohlc(bars)
    atr = _atr(bars, cfg)
    rng = h - l
    if not atr > 0 or rng <= 0:
        return None
    upper_wick, lower_wick = h - max(o, c), min(o, c) - l

    if market is Market.USDM and structure.buy_side_liquidity:    # SHORT: swept stops above, rejected
        swept = [p for p in structure.buy_side_liquidity if l < p < h and c < p]
        if swept and c < o and upper_wick >= cfg.sweep_wick_frac * rng:
            pool = min(swept)
            stop = h + cfg.stop_buffer_atr * atr
            tp1 = structure.support_below(c)
            if tp1 is None and not np.isnan(structure.range_low):
                tp1 = structure.range_low
            reasons = [f"liquidity sweep: wicked {pool:,.4g} (stops) and rejected"]
            return _build("liquidity_sweep_reversal", SHORT, market, MARKET, c, stop, tp1, h,
                          structure, cfg, ctx, reasons, "reversal")

    if structure.sell_side_liquidity:                             # LONG: swept stops below, reclaimed
        swept = [p for p in structure.sell_side_liquidity if l < p < h and c > p]
        if swept and c > o and lower_wick >= cfg.sweep_wick_frac * rng:
            pool = max(swept)
            stop = l - cfg.stop_buffer_atr * atr
            tp1 = structure.resistance_above(c)
            if tp1 is None and not np.isnan(structure.range_high):
                tp1 = structure.range_high
            reasons = [f"liquidity sweep: wicked {pool:,.4g} (stops) and reclaimed"]
            return _build("liquidity_sweep_reversal", LONG, market, MARKET, c, stop, tp1, l,
                          structure, cfg, ctx, reasons, "reversal")
    return None


def detect_momentum_flag(bars, structure, ctx, cfg, market) -> Signal | None:
    """Continuation: a major impulse, then a tight shallow coil -> STOP break onward.

    Catches the meat of a strong trend that never pulls back to the MA."""
    state = structure.state
    swings = structure.swings
    if state is None or len(swings) < 2:
        return None
    o, h, l, c = _ohlc(bars)
    atr = _atr(bars, cfg)
    if not atr > 0:
        return None
    last, prev = swings[-1], swings[-2]
    if last.leg_atr < cfg.flag_impulse_atr:             # need a genuine impulse leg
        return None
    consol = bars.iloc[last.index + 1:]                 # bars AFTER the impulse swing
    if not (2 <= len(consol) <= cfg.flag_max_bars):
        return None
    chi, clo = float(consol["high"].max()), float(consol["low"].min())
    if (chi - clo) > cfg.flag_max_range_atr * atr:      # must be a TIGHT coil
        return None

    if last.kind == st.HIGH and state.trend == st.UP:   # BULL flag
        height = last.price - prev.price
        if height <= 0 or (last.price - clo) > cfg.flag_max_retrace * height:
            return None
        stop = clo - cfg.stop_buffer_atr * atr
        tp1 = structure.resistance_above(chi) or (chi + height)
        reasons = [f"bull flag: {last.leg_atr:.1f}-ATR impulse, {len(consol)}-bar coil"]
        return _build("momentum_flag", LONG, market, STOP, chi, stop, tp1, clo,
                      structure, cfg, ctx, reasons, "trend")

    if market is Market.USDM and last.kind == st.LOW and state.trend == st.DOWN:   # BEAR flag
        height = prev.price - last.price
        if height <= 0 or (chi - last.price) > cfg.flag_max_retrace * height:
            return None
        stop = chi + cfg.stop_buffer_atr * atr
        tp1 = structure.support_below(clo) or (clo - height)
        reasons = [f"bear flag: {last.leg_atr:.1f}-ATR impulse, {len(consol)}-bar coil"]
        return _build("momentum_flag", SHORT, market, STOP, clo, stop, tp1, chi,
                      structure, cfg, ctx, reasons, "trend")
    return None


def detect_squeeze_breakout(bars, structure, ctx, cfg, market) -> Signal | None:
    """Breakout/expansion: a volatility coil (low ATR%ile) that releases — the
    IGNITION of a move, which the ADX-gated breakout_momentum would skip while quiet."""
    o, h, l, c = _ohlc(bars)
    atr = _atr(bars, cfg)
    if not atr > 0 or len(bars) <= cfg.breakout_lookback + 1:
        return None
    pctl = ind.atr_percentile(bars.iloc[:-1], cfg.atr_length, 100)   # compression BEFORE the break
    if not (pctl == pctl) or pctl > cfg.squeeze_pctl:
        return None
    prior = bars.iloc[-(cfg.breakout_lookback + 1):-1]
    sq_high, sq_low = float(prior["high"].max()), float(prior["low"].min())
    sq_range = sq_high - sq_low
    if sq_range <= 0:
        return None

    if c > sq_high and c > o:                                        # LONG: release upward
        stop = sq_low - cfg.stop_buffer_atr * atr
        tp1 = structure.resistance_above(c) or (c + sq_range)
        reasons = [f"squeeze breakout (ATR%ile {pctl:.0f}) out of the {sq_low:,.4g}-{sq_high:,.4g} coil"]
        return _build("squeeze_breakout", LONG, market, MARKET, c, stop, tp1, sq_low,
                      structure, cfg, ctx, reasons, "squeeze")

    if market is Market.USDM and c < sq_low and c < o:               # SHORT: release downward
        stop = sq_high + cfg.stop_buffer_atr * atr
        tp1 = structure.support_below(c) or (c - sq_range)
        reasons = [f"squeeze breakdown (ATR%ile {pctl:.0f}) out of the {sq_low:,.4g}-{sq_high:,.4g} coil"]
        return _build("squeeze_breakout", SHORT, market, MARKET, c, stop, tp1, sq_high,
                      structure, cfg, ctx, reasons, "squeeze")
    return None


def detect_choch_reversal(bars, structure, ctx, cfg, market) -> Signal | None:
    """Reversal/trap: a Change of Character — the prior trend's structure breaks
    (the swing engine flags CHoCH_up/down) — trade the new direction on confirmation."""
    state = structure.state
    if state is None or np.isnan(state.last_high) or np.isnan(state.last_low):
        return None
    o, h, l, c = _ohlc(bars)
    atr = _atr(bars, cfg)
    if not atr > 0:
        return None

    if state.event == "CHoCH_up":                                    # downtrend's structure broke up
        stop = state.last_low - cfg.stop_buffer_atr * atr
        tp1 = structure.resistance_above(c)
        if tp1 is None and not np.isnan(structure.range_high):
            tp1 = structure.range_high
        reasons = [f"CHoCH up: broke the prior lower-high {state.last_high:,.4g}"]
        return _build("choch_reversal", LONG, market, MARKET, c, stop, tp1, state.last_low,
                      structure, cfg, ctx, reasons, "reversal")

    if market is Market.USDM and state.event == "CHoCH_down":         # uptrend's structure broke down
        stop = state.last_high + cfg.stop_buffer_atr * atr
        tp1 = structure.support_below(c)
        if tp1 is None and not np.isnan(structure.range_low):
            tp1 = structure.range_low
        reasons = [f"CHoCH down: broke the prior higher-low {state.last_low:,.4g}"]
        return _build("choch_reversal", SHORT, market, MARKET, c, stop, tp1, state.last_high,
                      structure, cfg, ctx, reasons, "reversal")
    return None


def _rsi_at(rsi_s: pd.Series, i: int) -> float:
    return float(rsi_s.iloc[i]) if 0 <= i < len(rsi_s) else float("nan")


def detect_divergence_reversal(bars, structure, ctx, cfg, market) -> Signal | None:
    """Reversal/trap: regular RSI divergence at a structural extreme (price makes a
    new extreme, momentum does not) — fade the exhausted move on confirmation."""
    o, h, l, c = _ohlc(bars)
    atr = _atr(bars, cfg)
    if not atr > 0:
        return None
    rsi_s = ind.rsi(bars, cfg.rsi_length)
    highs = [s for s in structure.swings if s.kind == st.HIGH][-2:]
    lows = [s for s in structure.swings if s.kind == st.LOW][-2:]

    if market is Market.USDM and len(highs) == 2:                    # bearish divergence -> short
        a, b = highs
        ra, rb = _rsi_at(rsi_s, a.index), _rsi_at(rsi_s, b.index)
        if b.price > a.price and ra == ra and rb == rb and rb < ra and c < o:
            stop = b.price + cfg.stop_buffer_atr * atr
            tp1 = structure.support_below(c)
            if tp1 is None and not np.isnan(structure.range_low):
                tp1 = structure.range_low
            reasons = [f"bearish divergence: HH {b.price:,.4g} but RSI {rb:.0f} < {ra:.0f}"]
            return _build("divergence_reversal", SHORT, market, MARKET, c, stop, tp1, b.price,
                          structure, cfg, ctx, reasons, "reversal")

    if len(lows) == 2:                                               # bullish divergence -> long
        a, b = lows
        ra, rb = _rsi_at(rsi_s, a.index), _rsi_at(rsi_s, b.index)
        if b.price < a.price and ra == ra and rb == rb and rb > ra and c > o:
            stop = b.price - cfg.stop_buffer_atr * atr
            tp1 = structure.resistance_above(c)
            if tp1 is None and not np.isnan(structure.range_high):
                tp1 = structure.range_high
            reasons = [f"bullish divergence: LL {b.price:,.4g} but RSI {rb:.0f} > {ra:.0f}"]
            return _build("divergence_reversal", LONG, market, MARKET, c, stop, tp1, b.price,
                          structure, cfg, ctx, reasons, "reversal")
    return None


DETECTORS = (
    detect_trend_pullback,
    detect_breakout_retest,
    detect_breakout_momentum,
    detect_momentum_flag,
    detect_squeeze_breakout,
    detect_failed_breakout,
    detect_liquidity_sweep_reversal,
    detect_choch_reversal,
    detect_divergence_reversal,
    detect_range_fade,
)
SETUP_NAMES = (
    "trend_pullback", "breakout_retest", "breakout_momentum", "momentum_flag",
    "squeeze_breakout", "failed_breakout", "liquidity_sweep_reversal",
    "choch_reversal", "divergence_reversal", "range_fade",
)
_GRADE_ORDER = {"A": 0, "B": 1, "C": 2}


def detect_all(bars, structure, ctx, cfg, market) -> list[Signal]:
    """Run every applicable setup on the decision bar; return all, ranked best-first."""
    signals: list[Signal] = []
    for fn in DETECTORS:
        try:
            sig = fn(bars, structure, ctx, cfg, market)
        except Exception:
            sig = None
        if sig is not None:
            signals.append(sig)
    signals.sort(key=lambda s: (_GRADE_ORDER.get(s.grade, 3), -s.rr))
    return signals
