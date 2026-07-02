"""SPOT market mechanics — long-only, no leverage/funding/OI.

Behaviour is the BaseMarket default; this subclass just pins the market type and
spot fee schedule.
"""
from __future__ import annotations

from ..config import Market
from .base import BaseMarket, Fees


class SpotMarket(BaseMarket):
    market = Market.SPOT
    long_only = True

    def fees(self) -> Fees:
        return Fees(self.cfg.spot_maker, self.cfg.spot_taker)
