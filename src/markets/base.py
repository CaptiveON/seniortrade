"""Market-mechanics abstraction (base + SPOT behaviour).

`BaseMarket` is the interface the rest of the tool calls so it stays
market-agnostic. SPOT behaviour lives here (long-only, no leverage/funding/OI);
:mod:`src.markets.usdm` overrides the leverage/liquidation/funding/OI hooks.

Pure functions (``liquidation_price``, ``safe_leverage``) are network-free and
unit-tested directly.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..config import Market, MarketsConfig

LONG = "long"
SHORT = "short"


# --------------------------------------------------------------------------- #
# Value objects
# --------------------------------------------------------------------------- #
@dataclass
class InstrumentSpec:
    symbol: str
    min_qty: float
    min_notional: float
    contract_size: float
    max_leverage: float


@dataclass
class Fees:
    maker: float
    taker: float


@dataclass
class FundingRead:
    rate: float              # per-interval funding rate (fraction)
    next_time: int | None
    annualized_pct: float
    signal: str              # neutral / crowded_long / crowded_short


@dataclass
class OIRead:
    open_interest: float
    change_pct: float        # vs oi_history_len bars ago
    trend: str               # rising / falling / flat


@dataclass
class SizingResult:
    symbol: str
    side: str
    entry: float
    stop: float
    per_unit_risk: float
    risk_amount: float
    size: float              # base units / contracts (rounded to lot)
    notional: float
    leverage: float
    margin: float
    liquidation_price: float | None
    liq_distance_pct: float | None
    stop_distance_pct: float
    fee_cost: float          # round-trip taker fees (approx)
    valid: bool
    notes: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Pure mechanics (testable without network)
# --------------------------------------------------------------------------- #
def liquidation_price(entry: float, leverage: float, mmr: float, side: str) -> float | None:
    """Isolated-margin linear liquidation estimate (uses the tier MMR).

    Standard approximation; a safety buffer is enforced separately. Not exact to
    the cent (ignores the maintenance-amount constant and fees) but correct for
    "is the stop comfortably inside liquidation?".
    """
    if leverage <= 0:
        return None
    if side == LONG:
        return entry * (1.0 - 1.0 / leverage + mmr)
    return entry * (1.0 + 1.0 / leverage - mmr)


def safe_leverage(entry: float, stop: float, mmr: float, buffer_mult: float, cap: float) -> float:
    """Highest integer leverage that keeps liquidation >= buffer_mult * stop-distance away.

    liq distance ~= 1/lev - mmr ; require it >= stop_dist * buffer_mult
    => lev <= 1 / (stop_dist * buffer_mult + mmr).
    """
    if entry <= 0:
        return 1.0
    stop_dist = abs(entry - stop) / entry
    denom = stop_dist * buffer_mult + mmr
    if denom <= 0:
        return cap
    return max(1.0, min(float(math.floor(1.0 / denom)), cap))


# --------------------------------------------------------------------------- #
# Base market (SPOT behaviour)
# --------------------------------------------------------------------------- #
class BaseMarket:
    market: Market = Market.SPOT
    long_only: bool = True

    def __init__(self, ex, cfg: MarketsConfig):
        self.ex = ex
        self.cfg = cfg

    # --- instrument metadata / rounding (source of truth) ---
    def spec(self, symbol: str) -> InstrumentSpec:
        m = self.ex.markets[symbol]
        limits = m.get("limits", {}) or {}
        amt = limits.get("amount", {}) or {}
        cost = limits.get("cost", {}) or {}
        lev = limits.get("leverage", {}) or {}
        return InstrumentSpec(
            symbol=symbol,
            min_qty=float(amt.get("min") or 0.0),
            min_notional=float(cost.get("min") or 0.0),
            contract_size=float(m.get("contractSize") or 1.0),
            max_leverage=float(lev.get("max") or self.cfg.max_leverage_cap),
        )

    def round_price(self, symbol: str, price: float) -> float:
        return float(self.ex.price_to_precision(symbol, price))

    def round_amount(self, symbol: str, amount: float) -> float:
        return float(self.ex.amount_to_precision(symbol, amount))

    def fees(self) -> Fees:
        return Fees(self.cfg.spot_maker, self.cfg.spot_taker)

    def min_notional_ok(self, symbol: str, amount: float, price: float) -> bool:
        sp = self.spec(symbol)
        return amount > 0 and amount >= sp.min_qty and amount * price >= sp.min_notional

    # --- usdm-only hooks (no-ops for spot) ---
    def max_leverage(self, symbol: str) -> float:
        return 1.0

    def funding(self, symbol: str) -> FundingRead | None:
        return None

    def open_interest(self, symbol: str) -> OIRead | None:
        return None

    def _leverage_block(self, symbol, entry, stop, side, notional, requested_leverage):
        """SPOT: full margin, no leverage, no liquidation."""
        return 1.0, notional, None, None, []

    # --- risk-first sizing (shared) ---
    def size_from_risk(
        self,
        symbol: str,
        entry: float,
        stop: float,
        risk_amount: float,
        side: str = LONG,
        leverage: float | None = None,
    ) -> SizingResult:
        side = side.lower()
        notes: list[str] = []
        valid = True

        if self.long_only and side == SHORT:
            notes.append("SPOT is long-only — short rejected")
            valid = False
        if side == LONG and not stop < entry:
            notes.append("long requires stop < entry")
            valid = False
        if side == SHORT and not stop > entry:
            notes.append("short requires stop > entry")
            valid = False

        per_unit = abs(entry - stop)
        if per_unit <= 0:
            notes.append("zero stop distance")
            valid = False

        size = 0.0
        notional = 0.0
        if valid:
            raw = risk_amount / per_unit
            size = self.round_amount(symbol, raw)
            notional = size * entry
            if not self.min_notional_ok(symbol, size, entry):
                notes.append(f"below min notional/qty (notional {notional:.2f})")
                valid = False

        lev, margin, liq, liq_dist, lev_notes = self._leverage_block(
            symbol, entry, stop, side, notional, leverage
        )
        notes += lev_notes

        fee_cost = notional * self.fees().taker * 2.0  # entry + exit, taker
        stop_dist_pct = (per_unit / entry * 100.0) if entry > 0 else float("nan")

        return SizingResult(
            symbol=symbol, side=side, entry=entry, stop=stop, per_unit_risk=per_unit,
            risk_amount=risk_amount, size=size, notional=notional, leverage=lev,
            margin=margin, liquidation_price=liq, liq_distance_pct=liq_dist,
            stop_distance_pct=stop_dist_pct, fee_cost=fee_cost, valid=valid, notes=notes,
        )
