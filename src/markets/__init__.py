"""Market-mechanics package: a market-agnostic interface over SPOT / USD-M."""
from __future__ import annotations

from ..config import Market, MarketsConfig
from .base import (
    LONG,
    SHORT,
    BaseMarket,
    Fees,
    FundingRead,
    InstrumentSpec,
    OIRead,
    SizingResult,
    liquidation_price,
    safe_leverage,
)
from .spot import SpotMarket
from .usdm import UsdmMarket


def make_market(market: Market, ex, cfg: MarketsConfig) -> BaseMarket:
    """Construct the right Market implementation for the given market type."""
    if market is Market.USDM:
        return UsdmMarket(ex, cfg)
    return SpotMarket(ex, cfg)


__all__ = [
    "LONG", "SHORT", "BaseMarket", "SpotMarket", "UsdmMarket", "make_market",
    "Fees", "FundingRead", "OIRead", "InstrumentSpec", "SizingResult",
    "liquidation_price", "safe_leverage",
]
