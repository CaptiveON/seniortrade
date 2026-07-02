"""USD-M perpetual mechanics — leverage, margin, liquidation, funding, OI.

Adds the futures-specific machinery on top of BaseMarket:
  - leverage chosen so liquidation sits safely beyond the structural stop,
  - liquidation from the maintenance-margin tier (notional-tiered MMR),
  - funding-rate read (cost estimate + contrarian signal),
  - open-interest read (current + trend).
All exchange calls are defensive — on any failure we fall back to a safe default
rather than raising, so analysis never crashes on a flaky endpoint.
"""
from __future__ import annotations

from ..config import Market
from .base import (
    LONG,
    BaseMarket,
    Fees,
    FundingRead,
    OIRead,
    liquidation_price,
    safe_leverage,
)


# Bundled, notional-tiered maintenance-margin rates for liquid USDT perps — used
# when live tiers aren't available (the public/no-key path). Live Binance tiers
# (fetched when an API key is present) override this with exact per-symbol values.
GENERIC_USDT_PERP_MMR: list[tuple[float, float]] = [
    (50_000.0, 0.005),
    (250_000.0, 0.01),
    (1_000_000.0, 0.025),
    (5_000_000.0, 0.05),
    (float("inf"), 0.10),
]


def generic_mmr(notional: float) -> float:
    for ceiling, mmr in GENERIC_USDT_PERP_MMR:
        if notional <= ceiling:
            return mmr
    return GENERIC_USDT_PERP_MMR[-1][1]


class UsdmMarket(BaseMarket):
    market = Market.USDM
    long_only = False

    def __init__(self, ex, cfg):
        super().__init__(ex, cfg)
        self._tier_cache: dict[str, list] = {}

    def fees(self) -> Fees:
        return Fees(self.cfg.usdm_maker, self.cfg.usdm_taker)

    def max_leverage(self, symbol: str) -> float:
        tiers = self._tiers(symbol)
        try:
            mx = max((float(t.get("maxLeverage") or 0) for t in tiers), default=0.0)
            if mx > 0:
                return mx
        except Exception:
            pass
        sp = self.spec(symbol)
        return sp.max_leverage if sp.max_leverage > 1 else self.cfg.max_leverage_cap

    # --- maintenance-margin tiers ---
    def _tiers(self, symbol: str) -> list:
        if symbol in self._tier_cache:
            return self._tier_cache[symbol]
        tiers: list = []
        try:
            tiers = self.ex.fetch_market_leverage_tiers(symbol) or []
        except Exception:
            try:
                tiers = (self.ex.fetch_leverage_tiers([symbol]) or {}).get(symbol, []) or []
            except Exception:
                tiers = []
        self._tier_cache[symbol] = tiers
        return tiers

    def maintenance_margin_rate(self, symbol: str, notional: float) -> float:
        tiers = self._tiers(symbol)
        try:
            for t in tiers:
                lo = float(t.get("minNotional") or 0.0)
                hi = t.get("maxNotional")
                hi = float(hi) if hi else float("inf")
                if lo <= notional <= hi:
                    return float(t["maintenanceMarginRate"])
            if tiers:
                return float(tiers[-1]["maintenanceMarginRate"])
        except Exception:
            pass
        return generic_mmr(notional)   # no live tiers (public/no-key) -> bundled estimate

    # --- leverage / margin / liquidation ---
    def _leverage_block(self, symbol, entry, stop, side, notional, requested_leverage):
        notes: list[str] = []
        mmr = self.maintenance_margin_rate(symbol, notional)
        cap = min(self.max_leverage(symbol), self.cfg.max_leverage_cap)
        requested = requested_leverage or self.cfg.default_leverage
        safe = safe_leverage(entry, stop, mmr, self.cfg.liq_buffer_mult, cap)
        lev = max(1.0, min(float(requested), safe, cap))
        if lev < requested:
            notes.append(f"leverage capped {requested:g}x→{lev:g}x to keep liquidation beyond the stop")

        margin = notional / lev if lev > 0 else notional
        liq = liquidation_price(entry, lev, mmr, side)
        liq_dist = abs(entry - liq) / entry * 100.0 if (liq and entry > 0) else None
        stop_dist = abs(entry - stop) / entry * 100.0 if entry > 0 else float("inf")
        if liq_dist is not None and liq_dist <= stop_dist:
            notes.append("WARNING: liquidation sits inside the stop — unsafe, do not trade")
        return lev, margin, liq, liq_dist, notes

    # --- funding ---
    def funding(self, symbol: str) -> FundingRead | None:
        try:
            fr = self.ex.fetch_funding_rate(symbol)
        except Exception:
            return None
        rate = float(fr.get("fundingRate") or 0.0)
        nxt = fr.get("fundingTimestamp") or fr.get("nextFundingTime")
        annualized = rate * self.cfg.funding_periods_per_day * 365 * 100.0
        if rate >= 0.0005:
            signal = "crowded_long"
        elif rate <= -0.0005:
            signal = "crowded_short"
        else:
            signal = "neutral"
        return FundingRead(rate=rate, next_time=nxt, annualized_pct=annualized, signal=signal)

    def funding_cost(self, notional: float, side: str, rate: float) -> float:
        """Estimated funding paid (>0) / earned (<0) over the expected hold, in quote."""
        periods = self.cfg.expected_hold_days * self.cfg.funding_periods_per_day
        # longs pay when rate > 0; shorts pay when rate < 0
        sign = 1.0 if side == LONG else -1.0
        return sign * rate * notional * periods

    # --- open interest ---
    def open_interest(self, symbol: str) -> OIRead | None:
        # Use the HISTORY series for both current and past so units match (the
        # snapshot endpoint reports base units, history reports quote value).
        hist: list = []
        try:
            hist = self.ex.fetch_open_interest_history(
                symbol, self.cfg.oi_history_tf, limit=self.cfg.oi_history_len
            ) or []
        except Exception:
            hist = []

        if len(hist) >= 2:
            cur = _oi_value(hist[-1])
            past = _oi_value(hist[0])
            if cur is not None and past:
                change_pct = (cur - past) / past * 100.0
                if change_pct >= self.cfg.oi_trend_pct:
                    trend = "rising"
                elif change_pct <= -self.cfg.oi_trend_pct:
                    trend = "falling"
                else:
                    trend = "flat"
                return OIRead(open_interest=cur, change_pct=change_pct, trend=trend)

        # Fallback: a current snapshot with no trend.
        try:
            snap = _oi_value(self.ex.fetch_open_interest(symbol))
            if snap is not None:
                return OIRead(open_interest=snap, change_pct=0.0, trend="flat")
        except Exception:
            pass
        return None


def _oi_value(d: dict) -> float | None:
    for key in ("openInterestValue", "openInterestAmount", "openInterest"):
        v = d.get(key)
        if v:
            try:
                return float(v)
            except (TypeError, ValueError):
                continue
    info = d.get("info") or {}
    for key in ("openInterest", "sumOpenInterest", "sumOpenInterestValue"):
        v = info.get(key)
        if v:
            try:
                return float(v)
            except (TypeError, ValueError):
                continue
    return None
