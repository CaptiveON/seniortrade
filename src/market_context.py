"""Layer 1 — Market context (the tide).

Before judging any coin, judge the market. In crypto, BTC is the dominant driver,
so it is used as the market proxy: its regime sets the risk-on/off posture, and
each coin is measured by its RELATIVE STRENGTH and BETA versus BTC.

The rule this enforces: do not fight the tide. A long-only spot trade into a
risk-off BTC is deprioritised; coins leading BTC (positive relative strength) are
surfaced over laggards.

NOTE (scope): total-market-cap / BTC-dominance feeds require a non-Binance data
source (e.g. CoinGecko) and are a later enhancement; BTC is the proxy for now.
These functions are pure (operate on DataFrames/Series) so they unit-test without
the network — the screener does the fetching.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Market
from .indicators import TRENDING, adx, regime_label

RISK_ON = "risk-on"
RISK_OFF = "risk-off"
NEUTRAL = "neutral"


@dataclass
class MarketContext:
    proxy_symbol: str
    regime: str          # trending / ranging / choppy / unknown
    trend: str           # up / down / flat
    posture: str         # risk-on / risk-off / neutral
    adx: float
    last_price: float
    available: bool = True
    note: str = ""


def reference_symbol(base: str, market: Market, quote: str) -> str:
    """Market-proxy symbol for any base on the given market/quote."""
    if market is Market.SPOT:
        return f"{base}/{quote}"
    return f"{base}/{quote}:{quote}"


def btc_reference_symbol(market: Market, quote: str) -> str:
    """The BTC market-proxy symbol (the tide). Kept for convenience."""
    return reference_symbol("BTC", market, quote)


def unavailable_context(symbol: str, reason: str) -> MarketContext:
    """A neutral placeholder when the BTC proxy can't be read (don't block the run)."""
    return MarketContext(
        proxy_symbol=symbol, regime="unknown", trend="flat", posture=NEUTRAL,
        adx=float("nan"), last_price=float("nan"), available=False, note=reason,
    )


def assess_market(
    btc_ohlc: pd.DataFrame,
    proxy_symbol: str,
    adx_length: int,
    trending_th: float,
    choppy_th: float,
) -> MarketContext:
    """Classify BTC's regime/trend into a risk posture."""
    adx_df = adx(btc_ohlc, adx_length)
    adx_val = float(adx_df["adx"].iloc[-1])
    plus_di = float(adx_df["plus_di"].iloc[-1])
    minus_di = float(adx_df["minus_di"].iloc[-1])

    regime = regime_label(adx_val, trending_th, choppy_th)
    if plus_di > minus_di:
        trend = "up"
    elif minus_di > plus_di:
        trend = "down"
    else:
        trend = "flat"

    if regime == TRENDING and trend == "up":
        posture, note = RISK_ON, "BTC in a clean uptrend — favour longs, alts can run."
    elif regime == TRENDING and trend == "down":
        posture, note = RISK_OFF, "BTC in a clean downtrend — be cautious with alt longs."
    else:
        posture, note = NEUTRAL, "BTC has no clear trend — pick your spots, expect chop."

    return MarketContext(
        proxy_symbol=proxy_symbol,
        regime=regime,
        trend=trend,
        posture=posture,
        adx=adx_val,
        last_price=float(btc_ohlc["close"].iloc[-1]),
        available=True,
        note=note,
    )


@dataclass
class RelStrength:
    rel_strength_pct: float   # coin %-return minus BTC %-return over the window
    beta: float               # sensitivity of coin returns to BTC returns
    corr: float               # correlation of returns to BTC


def relative_strength(
    coin_close: pd.Series,
    ref_close: pd.Series,
    window: int,
) -> RelStrength:
    """Relative strength, beta, and correlation of a coin versus the BTC proxy.

    Positive relative strength => the coin outperformed BTC over the window
    (a leader); negative => a laggard. Beta ~1 moves with BTC; >1 amplifies it.
    """
    joined = pd.concat(
        [coin_close.rename("c"), ref_close.rename("b")], axis=1
    ).dropna()
    if len(joined) < 3:
        return RelStrength(float("nan"), float("nan"), float("nan"))

    joined = joined.tail(window + 1)
    perf_c = joined["c"].iloc[-1] / joined["c"].iloc[0] - 1.0
    perf_b = joined["b"].iloc[-1] / joined["b"].iloc[0] - 1.0
    rs_pct = (perf_c - perf_b) * 100.0

    cr = joined["c"].pct_change(fill_method=None).dropna()
    br = joined["b"].pct_change(fill_method=None).dropna()
    if len(br) >= 2 and float(br.var()) > 0:
        beta = float(np.cov(cr, br)[0, 1] / br.var())
        corr = float(cr.corr(br))
    else:
        beta = float("nan")
        corr = float("nan")

    return RelStrength(rel_strength_pct=rs_pct, beta=beta, corr=corr)


@dataclass
class GroupInfo:
    leader: str        # symbol of the cluster leader (the coin's "driver")
    corr: float        # member's correlation to that leader (1.0 for the leader)
    size: int          # members in the cluster (candidates + anchors)


def _pick_leader(members: set[str], priority: list[str], liquidity: dict[str, float]) -> str:
    """Anchor leaders (BTC, ETH, ...) win if present; else the most-liquid member."""
    for p in priority:
        if p in members:
            return p
    return max(members, key=lambda m: liquidity.get(m, 0.0))


def assign_groups(
    closes: dict[str, pd.Series],
    leader_priority: list[str],
    liquidity: dict[str, float],
    window: int,
    threshold: float,
) -> dict[str, GroupInfo]:
    """Cluster symbols that move together; name each group by its leader.

    Macro anchors (BTC, ETH) are injected as members so a coin that tracks them
    lands in the "BTC" / "ETH" group; clusters with no anchor are led by their
    most-liquid member (a "relative-chain" leader). Coins in one group move
    together — they are effectively ONE bet.
    """
    syms = list(closes)
    if len(syms) < 2:
        return {s: GroupInfo(leader=s, corr=1.0, size=1) for s in syms}

    rets = (
        pd.DataFrame({s: closes[s].pct_change(fill_method=None) for s in syms})
        .dropna(how="all")
        .tail(window)
    )
    corr = rets.corr()

    assigned: dict[str, int] = {}
    clusters: list[set[str]] = []
    for s in corr.index:
        if s in assigned:
            continue
        members = {s}
        for other in corr.index:
            if other == s or other in assigned:
                continue
            v = corr.loc[s, other]
            if pd.notna(v) and v >= threshold:
                members.add(other)
        for m in members:
            assigned[m] = len(clusters)
        clusters.append(members)

    out: dict[str, GroupInfo] = {}
    for members in clusters:
        leader = _pick_leader(members, leader_priority, liquidity)
        size = len(members)
        for m in members:
            if m == leader:
                cval = 1.0
            else:
                v = corr.loc[m, leader] if leader in corr.columns else float("nan")
                cval = float(v) if pd.notna(v) else float("nan")
            out[m] = GroupInfo(leader=leader, corr=cval, size=size)
    return out
