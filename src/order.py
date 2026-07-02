"""Layer 9 — order staging (the only layer that can touch money).

DRY-RUN by default: build the EXACT order ticket that live would send, run every
safety check, log STAGED to the journal on typed CONFIRM — and transmit NOTHING.
`--live` (built last) re-fetches and re-confirms before any real order.

This module holds the pure, testable pieces (ticket, drift/validity/duplicate
checks); the interactive flow (re-fetch, CONFIRM prompt, journal write) lives in
the CLI so these stay network/IO-free.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import guards as gd
from . import journal as jn
from .config import Market
from .markets import LONG, SHORT

MARKET = "market"
LIMIT = "limit"
STOP = "stop"


@dataclass
class OrderLeg:
    kind: str            # entry / stop / tp1 / tp2
    order_type: str      # market / limit / stop_market / take_profit_market
    side: str            # buy / sell
    price: float
    reduce_only: bool
    qty_fraction: float  # 1.0 = full size, 0.5 = half, ...


@dataclass
class OrderTicket:
    symbol: str
    market: str
    leverage: float | None
    margin_mode: str | None      # "isolated" for usdm
    legs: list = field(default_factory=list)


def build_proposed(plan, signal, group: str, risk_pct: float) -> gd.ProposedTrade:
    return gd.ProposedTrade(
        symbol=plan.symbol, group=group, side=signal.direction, risk_pct=risk_pct,
        entry=plan.entry, stop=plan.stop, liquidation=plan.liquidation_price)


def build_ticket(plan, signal, market: Market, tp1_fraction: float) -> OrderTicket:
    """The exact orders that would be sent — identical for dry-run and live."""
    long = signal.direction == LONG
    entry_side = "buy" if long else "sell"
    exit_side = "sell" if long else "buy"
    legs = [OrderLeg("entry", signal.entry_type, entry_side, plan.entry, False, 1.0),
            OrderLeg("stop", "stop_market", exit_side, plan.stop, True, 1.0)]
    targets = plan.targets or []
    if len(targets) >= 1:
        legs.append(OrderLeg("tp1", "take_profit_market", exit_side, targets[0], True, tp1_fraction))
    if len(targets) >= 2:
        legs.append(OrderLeg("tp2", "take_profit_market", exit_side, targets[-1], True, round(1.0 - tp1_fraction, 4)))
    is_usdm = market is Market.USDM
    return OrderTicket(symbol=plan.symbol, market=market.value,
                       leverage=plan.leverage if is_usdm else None,
                       margin_mode="isolated" if is_usdm else None, legs=legs)


def check_drift(entry_type: str, planned_entry: float, current_price: float,
                drift_pct: float) -> tuple[bool, str]:
    """Market entries fill at ~current price → abort if it drifted from the analysis.
    Limit/stop entries rest at their level, so there's no drift race."""
    if entry_type != MARKET:
        return True, f"{entry_type} entry rests at its level — no drift race"
    if planned_entry <= 0:
        return False, "invalid planned entry"
    drift = abs(current_price - planned_entry) / planned_entry * 100.0
    if drift > drift_pct:
        return False, f"price drifted {drift:.2f}% from analysed entry (> {drift_pct:g}%) — read is stale"
    return True, f"price within {drift:.2f}% of the analysed entry"


def check_validity(signal, current_price: float) -> tuple[bool, str]:
    """Abort if the pre-entry invalidation has ALREADY been hit since the signal."""
    inv = getattr(signal, "invalidation", None)
    if inv in (None, 0):
        return True, "no pre-entry invalidation"
    if signal.direction == LONG and current_price <= inv:
        return False, f"invalidation hit: price {current_price:,.6g} ≤ {inv:,.6g} — setup void"
    if signal.direction == SHORT and current_price >= inv:
        return False, f"invalidation hit: price {current_price:,.6g} ≥ {inv:,.6g} — setup void"
    return True, "invalidation intact"


def has_open_or_staged(records, symbol: str) -> bool:
    return any(r.symbol == symbol and r.status in (jn.OPEN, jn.STAGED) for r in records)
