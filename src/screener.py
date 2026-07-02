"""Layer 0 — the screener (context-aware).

Scans the exchange, rejects untradeable coins, and returns a RANKED shortlist of
candidates worthy of deep analysis. The USER reads the table and picks one; the
rest of the tool then runs only on that coin.

Filters in the brief's priority order:
  1. LIQUIDITY        — top-N by 24h quote volume, above a floor, tight spread, thick book.
  2. HISTORY          — enough closed candles to backtest later.
  3. VOLATILITY REGIME— ATR% inside a band AND its regime (squeeze/normal/expanding).
  4. REGIME CLARITY   — trending / ranging / choppy (ADX); flag choppy/avoid.
  5. CORRELATION      — cluster movers-together so the user sees fake diversification.
  6. MARKET CONTEXT   — score against BTC: relative strength, beta, risk posture.

RANKING is a transparent COMPOSITE (trend clarity + relative strength + volatility
regime + liquidity), NOT ADX alone. Everything is descriptive, not predictive, and
the "why it passed" line restates the measured facts. Majors dominating is correct.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from . import data_fetch as dfetch
from . import market_context as mc
from .config import Market, ScreenerConfig
from .indicators import (
    CHOPPY,
    EXPANDING,
    EXTENDED,
    FRESH,
    NORMAL,
    RANGING,
    SQUEEZE,
    STRETCHED,
    TRENDING,
    adx,
    atr_percent,
    extension_atr,
    regime_label,
    volatility_regime,
)

ProgressCb = Callable[[str, int, int], None]

_VOL_SCORE = {SQUEEZE: 1.0, NORMAL: 0.7, EXPANDING: 0.4}
_FRESH_SCORE = {FRESH: 1.0, STRETCHED: 0.6, EXTENDED: 0.2}


@dataclass
class Candidate:
    symbol: str
    quote_volume: float
    last_price: float
    spread_bps: float
    depth_usd: float
    atr_pct: float
    adx: float
    plus_di: float
    minus_di: float
    regime: str
    trend: str          # "up" / "down" / "flat"
    n_candles: int
    # market context / volatility regime (defaulted so older callers still work)
    rel_strength: float = float("nan")   # % vs BTC over rs_window
    btc_beta: float = float("nan")
    btc_corr: float = float("nan")
    vol_regime: str = NORMAL
    atr_pctl: float = float("nan")
    extension: float = float("nan")   # directional ATRs from reference MA (+ = into the move)
    freshness: str = FRESH
    group: str = "-"                  # influence group, named by its leader (BTC/ETH/alt/indep)
    driver: str = ""                  # the leader symbol this coin tracks most
    driver_corr: float = float("nan") # correlation to that driver
    driver_rs: float = float("nan")   # relative strength vs the driver (%)
    driver_beta: float = float("nan") # beta to the driver
    clustered: bool = False           # group has >1 member (same group = one bet)
    fighting_tide: bool = False
    score: float = 0.0
    why: str = ""
    # internal only (not displayed)
    _close: pd.Series | None = field(default=None, repr=False)


@dataclass
class ScreenResult:
    market: Market
    quote: str
    screen_tf: str
    candidates: list[Candidate]      # ranked, passed all filters
    examined: int                    # symbols that reached the deep checks
    eligible_universe: int           # eligible symbols before top-N gate
    rejections: dict[str, int]       # stage -> count rejected
    context: mc.MarketContext


def _regime_from_adx(adx_val: float, cfg: ScreenerConfig) -> str:
    return regime_label(adx_val, cfg.adx_trending, cfg.adx_choppy)


def _trend_dir(plus_di: float, minus_di: float) -> str:
    if pd.isna(plus_di) or pd.isna(minus_di):
        return "flat"
    if plus_di > minus_di:
        return "up"
    if minus_di > plus_di:
        return "down"
    return "flat"


def _freshness_label(dir_extension: float, cfg: ScreenerConfig) -> str:
    """Classify how far price is into the move (directional ATRs from its MA)."""
    if pd.isna(dir_extension):
        return FRESH
    if dir_extension >= cfg.extended_atr:
        return EXTENDED
    if dir_extension >= cfg.fresh_atr:
        return STRETCHED
    return FRESH


def eligible_symbols(ex, cfg: ScreenerConfig, market: Market) -> list[str]:
    """Symbols tradeable on this market, in the target quote, minus stables/fiat."""
    out: list[str] = []
    for symbol, m in ex.markets.items():
        if not m.get("active", True):
            continue
        if m.get("quote") != cfg.quote:
            continue
        base = (m.get("base") or "").upper()
        if base in cfg.excluded_bases:
            continue

        if market is Market.SPOT:
            if not m.get("spot"):
                continue
            if base.endswith(("UP", "DOWN")) or base.startswith(("BULL", "BEAR")):
                continue
        else:  # USD-M perpetuals only (linear, no dated expiry)
            if not (m.get("swap") and m.get("linear")):
                continue
            if m.get("expiry"):
                continue

        out.append(symbol)
    return out


def _rank_by_volume(
    tickers: dict, symbols: list[str], cfg: ScreenerConfig
) -> tuple[list[tuple[str, float, float]], int]:
    """Return [(symbol, quote_volume, last_price)] passing the volume+top-N gate."""
    rows: list[tuple[str, float, float]] = []
    for sym in symbols:
        t = tickers.get(sym)
        if not t:
            continue
        qv = t.get("quoteVolume")
        last = t.get("last")
        if qv is None or last is None:
            continue
        rows.append((sym, float(qv), float(last)))

    rows.sort(key=lambda r: r[1], reverse=True)

    kept: list[tuple[str, float, float]] = []
    for sym, qv, last in rows:
        if qv < cfg.min_quote_volume:
            continue
        kept.append((sym, qv, last))
        if len(kept) >= cfg.top_n:
            break

    rejected = len(rows) - len(kept)
    return kept, rejected


def _assign_groups(
    cands: list[Candidate],
    anchor_closes: dict[str, pd.Series],
    cfg: ScreenerConfig,
) -> None:
    """Assign each candidate to an influence group (named by its leader/driver).

    Injects the macro anchors (BTC, ETH) so coins tracking them land in the
    "BTC"/"ETH" group; also computes each coin's relative strength/beta vs its
    own driver — "leading or lagging" judged against the RIGHT leader.
    """
    closes = {c.symbol: c._close for c in cands if c._close is not None}
    for sym, series in anchor_closes.items():
        if series is not None:
            closes[sym] = series

    if len(closes) < 2:
        for c in cands:
            c.group = "indep"
        return

    liquidity = {c.symbol: c.quote_volume for c in cands}
    leader_priority = [s for s, ser in anchor_closes.items() if ser is not None]
    groups = mc.assign_groups(
        closes, leader_priority, liquidity, cfg.corr_window, cfg.corr_threshold
    )

    for c in cands:
        g = groups.get(c.symbol)
        if g is None:
            c.group = "indep"
            continue
        c.driver = g.leader
        c.driver_corr = g.corr
        c.clustered = g.size > 1
        if g.leader == c.symbol and g.size == 1:
            c.group = "indep"
        else:
            c.group = g.leader.split("/")[0]

        # relative strength vs the driver (the right leader), not only BTC
        if g.leader == c.symbol:
            c.driver_rs = 0.0
            c.driver_beta = 1.0
        else:
            leader_close = closes.get(g.leader)
            if leader_close is not None and c._close is not None:
                rs = mc.relative_strength(c._close, leader_close, cfg.rs_window)
                c.driver_rs = rs.rel_strength_pct
                c.driver_beta = rs.beta


def _score(
    c: Candidate,
    cfg: ScreenerConfig,
    ctx: "mc.MarketContext | None" = None,
    market: Market = Market.SPOT,
) -> float:
    """Composite ranking score — surfaces the cleanest, leading, well-priced setups.

    components (each ~[0,1]): trend clarity (ADX), directional relative strength
    vs BTC, volatility regime (favour coiling), liquidity. Choppy regimes and
    tide-fighting longs are penalised so they sink.
    """
    trend_clarity = 0.0 if pd.isna(c.adx) else min(c.adx / 50.0, 1.0)

    # Relative strength aligned to the coin's OWN direction: a leader in an
    # uptrend and a heavy laggard in a downtrend are both "strong" for that side.
    if pd.isna(c.rel_strength):
        rs_component = 0.5
    else:
        rs_dir = c.rel_strength if c.trend != "down" else -c.rel_strength
        rs_norm = max(-1.0, min(1.0, rs_dir / 20.0))  # +/-20% vs BTC saturates
        rs_component = (rs_norm + 1.0) / 2.0

    vol_component = _VOL_SCORE.get(c.vol_regime, 0.5)
    fresh_component = _FRESH_SCORE.get(c.freshness, 0.7)
    liq_component = min(c.quote_volume / 1e9, 1.0)

    score = (
        cfg.rank_w_trend * trend_clarity
        + cfg.rank_w_rs * rs_component
        + cfg.rank_w_vol * vol_component
        + cfg.rank_w_fresh * fresh_component
        + cfg.rank_w_liq * liq_component
    )

    if c.regime == CHOPPY:
        score -= cfg.choppy_penalty

    # Don't fight the tide: long-only SPOT into a risk-off BTC. Penalty scales by
    # EFFECTIVE BTC exposure = beta x correlation — a coin decoupled from BTC
    # (low correlation) that is leading is NOT fighting the tide, it's defying it.
    if market is Market.SPOT and ctx is not None and ctx.posture == mc.RISK_OFF:
        beta = 0.5 if pd.isna(c.btc_beta) else max(0.0, min(c.btc_beta, 2.0)) / 2.0
        corr = 0.5 if pd.isna(c.btc_corr) else max(0.0, min(c.btc_corr, 1.0))
        exposure = beta * corr
        score -= cfg.context_penalty * exposure
        if exposure > 0.25:
            c.fighting_tide = True

    # SPOT downtrends aren't long-tradeable -> deprioritise.
    if market is Market.SPOT and c.trend == "down":
        score -= 0.5

    return score


def _build_why(c: Candidate, rank: int, cfg: ScreenerConfig) -> str:
    vol_m = c.quote_volume / 1e6
    depth_k = c.depth_usd / 1e3
    parts = [
        f"#{rank} by composite",
        f"${vol_m:,.0f}M 24h vol",
        f"{c.spread_bps:.1f}bps spread",
        f"depth ${depth_k:,.0f}k(±{cfg.depth_band_pct:g}%)",
        f"ATR% {c.atr_pct:.2f} ({c.vol_regime}, {c.freshness} {c.extension:+.1f}σ)",
        f"ADX {c.adx:.0f}→{c.regime} {c.trend}",
    ]
    if not pd.isna(c.rel_strength):
        parts.append(f"RS {c.rel_strength:+.1f}% vs BTC (β{c.btc_beta:.2f})")
    why = "; ".join(parts)
    if c.clustered:
        extra = f"corr {c.driver_corr:.2f}" if not pd.isna(c.driver_corr) else ""
        if not pd.isna(c.driver_rs):
            extra += f", RS {c.driver_rs:+.1f}% vs {c.group}"
        why += f"; group {c.group} ({extra}) — same group = one bet"
    elif c.group not in ("-", "indep", ""):
        why += f"; group {c.group}"
    if c.fighting_tide:
        why += "; fighting risk-off BTC"
    if c.regime == CHOPPY:
        why += "; CHOPPY — avoid"
    return why


def run_screen(
    market: Market,
    cfg: ScreenerConfig,
    api_key: str | None = None,
    api_secret: str | None = None,
    progress: ProgressCb | None = None,
) -> ScreenResult:
    """Execute the full screen against real Binance data and return a ranking."""
    ex = dfetch.make_exchange(market, api_key, api_secret)
    dfetch.load_markets(ex)
    tickers = dfetch.fetch_all_tickers(ex)

    # --- Anchors (the tide + influence-group leaders): BTC, ETH, ... ---
    anchor_closes: dict[str, pd.Series] = {}
    btc_ohlc = None
    for base in cfg.anchor_bases:
        sym = mc.reference_symbol(base, market, cfg.quote)
        try:
            raw = dfetch.fetch_ohlcv(ex, sym, cfg.screen_tf, cfg.candle_limit)
            ohlc = dfetch.drop_unclosed(ex, raw, cfg.screen_tf)
        except dfetch.DataError:
            continue
        anchor_closes[sym] = ohlc["close"]
        if base == "BTC":
            btc_ohlc = ohlc

    btc_sym = mc.reference_symbol("BTC", market, cfg.quote)
    if btc_ohlc is not None:
        context = mc.assess_market(btc_ohlc, btc_sym, cfg.adx_length, cfg.adx_trending, cfg.adx_choppy)
    else:
        context = mc.unavailable_context(btc_sym, "BTC proxy unavailable")
    btc_close = anchor_closes.get(btc_sym)

    universe = eligible_symbols(ex, cfg, market)
    volume_kept, vol_rejected = _rank_by_volume(tickers, universe, cfg)

    # concurrent OHLCV prefetch (thread pool) — the screener's main latency cost
    ohlcv_cache: dict[str, pd.DataFrame] = {}
    if cfg.fetch_workers and cfg.fetch_workers > 1 and volume_kept:
        ohlcv_cache = dfetch.fetch_ohlcv_concurrent(
            market, [s for s, _, _ in volume_kept], cfg.screen_tf, cfg.candle_limit,
            api_key, api_secret, cfg.fetch_workers)

    rejections = {
        "below_volume_or_top_n": vol_rejected,
        "insufficient_history": 0,
        "atr_band": 0,
        "spread_or_depth": 0,
        "stale_or_error": 0,
    }

    candidates: list[Candidate] = []
    total = len(volume_kept)
    for i, (sym, qv, last) in enumerate(volume_kept, start=1):
        if progress:
            progress(sym, i, total)

        # --- fetch + history gate (use the concurrent prefetch when available) ---
        raw = ohlcv_cache.get(sym)
        if raw is None:
            try:
                raw = dfetch.fetch_ohlcv(ex, sym, cfg.screen_tf, cfg.candle_limit)
            except dfetch.DataError:
                rejections["stale_or_error"] += 1
                continue
        if dfetch.is_stale(ex, raw, cfg.screen_tf):
            rejections["stale_or_error"] += 1
            continue

        ohlc = dfetch.drop_unclosed(ex, raw, cfg.screen_tf)
        n = len(ohlc)
        if n < cfg.min_candles:
            rejections["insufficient_history"] += 1
            continue

        # --- volatility band + regime gate (on the last closed candle) ---
        atr_pct = float(atr_percent(ohlc, cfg.atr_length).iloc[-1])
        if not (cfg.atr_pct_min <= atr_pct <= cfg.atr_pct_max):
            rejections["atr_band"] += 1
            continue
        vol_reg, atr_pctl = volatility_regime(
            ohlc, cfg.atr_length, cfg.atr_pctl_window, cfg.squeeze_pctl, cfg.expand_pctl
        )

        # --- regime / direction ---
        adx_df = adx(ohlc, cfg.adx_length)
        adx_val = float(adx_df["adx"].iloc[-1])
        plus_di = float(adx_df["plus_di"].iloc[-1])
        minus_di = float(adx_df["minus_di"].iloc[-1])
        trend = _trend_dir(plus_di, minus_di)

        # --- extension / freshness: how far is price into the move? ---
        ext = float(extension_atr(ohlc, cfg.ext_ma_len, cfg.atr_length).iloc[-1])
        dir_ext = ext if trend != "down" else -ext   # align to the trade direction
        freshness = _freshness_label(dir_ext, cfg)

        # --- market context: relative strength / beta vs BTC ---
        if btc_close is not None:
            rs = mc.relative_strength(ohlc["close"], btc_close, cfg.rs_window)
        else:
            rs = mc.RelStrength(float("nan"), float("nan"), float("nan"))

        # --- liquidity: real spread + book depth gate ---
        try:
            book = dfetch.fetch_book_liquidity(ex, sym, cfg.depth_band_pct)
        except dfetch.DataError:
            rejections["stale_or_error"] += 1
            continue
        if book.spread_bps > cfg.max_spread_bps or book.depth_usd < cfg.min_depth_usd:
            rejections["spread_or_depth"] += 1
            continue

        candidates.append(
            Candidate(
                symbol=sym,
                quote_volume=qv,
                last_price=last,
                spread_bps=book.spread_bps,
                depth_usd=book.depth_usd,
                atr_pct=atr_pct,
                adx=adx_val,
                plus_di=plus_di,
                minus_di=minus_di,
                regime=_regime_from_adx(adx_val, cfg),
                trend=trend,
                n_candles=n,
                rel_strength=rs.rel_strength_pct,
                btc_beta=rs.beta,
                btc_corr=rs.corr,
                vol_regime=vol_reg,
                atr_pctl=atr_pctl,
                extension=dir_ext,
                freshness=freshness,
                _close=ohlc["close"].copy(),
            )
        )

    _assign_groups(candidates, anchor_closes, cfg)
    for c in candidates:
        c.score = _score(c, cfg, context, market)
    candidates.sort(key=lambda c: c.score, reverse=True)
    for rank, c in enumerate(candidates, start=1):
        c.why = _build_why(c, rank, cfg)

    return ScreenResult(
        market=market,
        quote=cfg.quote,
        screen_tf=cfg.screen_tf,
        candidates=candidates,
        examined=total,
        eligible_universe=len(universe),
        rejections=rejections,
        context=context,
    )


def relative_strength_scan(
    market: Market,
    cfg: ScreenerConfig,
    limit: int,
    api_key: str | None = None,
    api_secret: str | None = None,
    progress: ProgressCb | None = None,
) -> tuple[mc.MarketContext, list[tuple[str, mc.RelStrength, str]]]:
    """Light scan for the `context` command: BTC posture + leaders/laggards.

    Returns the market context and a list of (symbol, RelStrength, trend) for the
    top-`limit` symbols by volume, sorted by relative strength (leaders first).
    No order-book / full-filter work — just enough to show the tide and who's
    leading it.
    """
    ex = dfetch.make_exchange(market, api_key, api_secret)
    dfetch.load_markets(ex)
    tickers = dfetch.fetch_all_tickers(ex)

    proxy = mc.btc_reference_symbol(market, cfg.quote)
    btc_close: pd.Series | None = None
    try:
        btc_raw = dfetch.fetch_ohlcv(ex, proxy, cfg.screen_tf, cfg.candle_limit)
        btc_ohlc = dfetch.drop_unclosed(ex, btc_raw, cfg.screen_tf)
        context = mc.assess_market(btc_ohlc, proxy, cfg.adx_length, cfg.adx_trending, cfg.adx_choppy)
        btc_close = btc_ohlc["close"]
    except dfetch.DataError as exc:
        return mc.unavailable_context(proxy, f"BTC proxy unavailable: {exc}"), []

    universe = eligible_symbols(ex, cfg, market)
    volume_kept, _ = _rank_by_volume(tickers, universe, cfg)
    volume_kept = volume_kept[:limit]

    rows: list[tuple[str, mc.RelStrength, str]] = []
    total = len(volume_kept)
    for i, (sym, _qv, _last) in enumerate(volume_kept, start=1):
        if progress:
            progress(sym, i, total)
        try:
            raw = dfetch.fetch_ohlcv(ex, sym, cfg.screen_tf, cfg.candle_limit)
        except dfetch.DataError:
            continue
        ohlc = dfetch.drop_unclosed(ex, raw, cfg.screen_tf)
        if len(ohlc) < cfg.rs_window + 2:
            continue
        rs = mc.relative_strength(ohlc["close"], btc_close, cfg.rs_window)
        adx_df = adx(ohlc, cfg.adx_length)
        trend = _trend_dir(float(adx_df["plus_di"].iloc[-1]), float(adx_df["minus_di"].iloc[-1]))
        rows.append((sym, rs, trend))

    rows.sort(key=lambda r: (r[1].rel_strength_pct if not pd.isna(r[1].rel_strength_pct) else -1e9), reverse=True)
    return context, rows
