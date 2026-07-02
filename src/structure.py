"""Layer 2 — market structure & key levels (expert grade).

Auto-detects what a seasoned trader marks on a chart and what a quant measures,
so the USER never draws a line:

  - SWINGS        ATR-thresholded ZigZag legs (noise-filtered), tagged minor/major.
  - MARKET STATE  HH/HL vs LH/LL, plus Break of Structure (BOS) and Change of
                  Character (CHoCH).
  - LEVELS        swing clusters scored by touches + recency + reaction + confluence,
                  pruned to the significant few.
  - RETRACEMENT   last-leg 38/50/61.8/79% (+ golden pocket) and measured move.
  - VOLUME        volume profile: POC + value-area high/low.
  - LIQUIDITY     equal highs/lows = stop pools (buy-side above / sell-side below).
  - CONTEXT       round numbers (magnitude-scaled) + prior-range (Donchian) edges.

These feed entries (trade AT a level), stops (BEYOND a level + liquidity, ATR as a
min-distance sanity check), targets (NEXT level / measured move), and the setups.
It is descriptive geometry on realised price — it does not predict.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import StructureConfig
from .indicators import atr

HIGH = "high"
LOW = "low"
SUPPORT = "support"
RESISTANCE = "resistance"
BUY_LIQ = "buy_liq"     # made of swing highs -> buy-side liquidity (stops above)
SELL_LIQ = "sell_liq"   # made of swing lows  -> sell-side liquidity (stops below)
MIXED = "mixed"

UP = "up"
DOWN = "down"
RANGE = "range"


@dataclass
class Swing:
    index: int
    timestamp: pd.Timestamp | None
    price: float
    kind: str             # HIGH or LOW
    leg_atr: float = 0.0  # size of the leg INTO this swing, in ATRs
    major: bool = False


@dataclass
class Level:
    price: float
    touches: int
    kind: str             # SUPPORT / RESISTANCE (relative to last price)
    side: str             # BUY_LIQ / SELL_LIQ / MIXED (by swing composition)
    last_touch_index: int
    strength: float
    reaction_atr: float = 0.0
    confluence: list[str] = field(default_factory=list)


@dataclass
class StructureState:
    trend: str            # UP / DOWN / RANGE (from the swing sequence)
    event: str            # BOS_up / BOS_down / CHoCH_up / CHoCH_down / none
    last_high: float
    last_low: float
    sequence: list[str]   # recent HH/HL/LH/LL labels


@dataclass
class VolumeProfile:
    poc: float            # point of control (highest-volume price)
    vah: float            # value-area high
    val: float            # value-area low


@dataclass
class Structure:
    last_price: float
    swings: list[Swing] = field(default_factory=list)
    levels: list[Level] = field(default_factory=list)
    state: StructureState | None = None
    round_levels: list[float] = field(default_factory=list)
    range_high: float = float("nan")
    range_low: float = float("nan")
    retracements: dict[str, float] = field(default_factory=dict)
    measured_move: float | None = None
    volume_profile: VolumeProfile | None = None
    buy_side_liquidity: list[float] = field(default_factory=list)
    sell_side_liquidity: list[float] = field(default_factory=list)

    # --- clustered S/R helpers (used by risk / setups) ---
    def supports(self) -> list[Level]:
        out = [lv for lv in self.levels if lv.price < self.last_price]
        return sorted(out, key=lambda lv: lv.price, reverse=True)

    def resistances(self) -> list[Level]:
        out = [lv for lv in self.levels if lv.price > self.last_price]
        return sorted(out, key=lambda lv: lv.price)

    def nearest_support(self) -> Level | None:
        s = self.supports()
        return s[0] if s else None

    def nearest_resistance(self) -> Level | None:
        r = self.resistances()
        return r[0] if r else None

    def _all_below(self, price: float) -> list[float]:
        cands = [lv.price for lv in self.levels if lv.price < price]
        cands += [r for r in self.round_levels if r < price]
        if not math.isnan(self.range_low) and self.range_low < price:
            cands.append(self.range_low)
        return sorted(cands, reverse=True)

    def _all_above(self, price: float) -> list[float]:
        cands = [lv.price for lv in self.levels if lv.price > price]
        cands += [r for r in self.round_levels if r > price]
        if not math.isnan(self.range_high) and self.range_high > price:
            cands.append(self.range_high)
        return sorted(cands)

    def support_below(self, price: float) -> float | None:
        below = self._all_below(price)
        return below[0] if below else None

    def resistance_above(self, price: float) -> float | None:
        above = self._all_above(price)
        return above[0] if above else None


# --------------------------------------------------------------------------- #
# Swings (ATR-thresholded ZigZag)
# --------------------------------------------------------------------------- #
def find_swings(df: pd.DataFrame, atr_values, reversal_atr: float = 2.0) -> list[Swing]:
    """ZigZag swings: a leg reverses only when price retraces >= reversal_atr * ATR.

    `atr_values` may be a scalar or a per-bar array/Series. Confirmed swings only
    (the in-progress final leg is intentionally not emitted).
    """
    n = len(df)
    if n < 2:
        return []
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    idx = df.index

    if np.isscalar(atr_values):
        atr_arr = np.full(n, float(atr_values))
    else:
        atr_arr = np.asarray(atr_values, dtype=float)

    swings: list[Swing] = []
    mode = 0                      # 0 undetermined, +1 seeking HIGH, -1 seeking LOW
    hi, hi_i = high[0], 0
    lo, lo_i = low[0], 0

    for i in range(1, n):
        a = atr_arr[i]
        if high[i] > hi:
            hi, hi_i = high[i], i
        if low[i] < lo:
            lo, lo_i = low[i], i
        if not (a == a) or a <= 0:   # NaN/zero ATR — can't threshold yet
            continue
        thr = reversal_atr * a

        if mode >= 0 and (hi - low[i]) >= thr:        # reversal down confirms a HIGH
            swings.append(Swing(hi_i, idx[hi_i], float(hi), HIGH))
            mode = -1
            lo, lo_i = low[i], i
            hi, hi_i = high[i], i
            continue
        if mode <= 0 and (high[i] - lo) >= thr:        # reversal up confirms a LOW
            swings.append(Swing(lo_i, idx[lo_i], float(lo), LOW))
            mode = 1
            hi, hi_i = high[i], i
            lo, lo_i = low[i], i

    _tag_legs(swings, atr_arr)
    return swings


def _tag_legs(swings: list[Swing], atr_arr: np.ndarray) -> None:
    """Annotate each swing with its leg size (in ATRs). `major` is set by caller."""
    for j in range(1, len(swings)):
        prev, cur = swings[j - 1], swings[j]
        a = atr_arr[cur.index]
        leg = abs(cur.price - prev.price)
        cur.leg_atr = float(leg / a) if a == a and a > 0 else 0.0


# --------------------------------------------------------------------------- #
# Market structure state (HH/HL, BOS/CHoCH)
# --------------------------------------------------------------------------- #
def classify_structure(swings: list[Swing], last_price: float) -> StructureState:
    highs = [s for s in swings if s.kind == HIGH]
    lows = [s for s in swings if s.kind == LOW]
    last_high = highs[-1].price if highs else float("nan")
    last_low = lows[-1].price if lows else float("nan")

    sequence: list[str] = []
    for s in swings[-6:]:
        same = [p for p in swings if p.kind == s.kind and p.index < s.index]
        if not same:
            continue
        prior = same[-1].price
        if s.kind == HIGH:
            sequence.append("HH" if s.price > prior else "LH")
        else:
            sequence.append("HL" if s.price > prior else "LL")

    trend = RANGE
    if len(highs) >= 2 and len(lows) >= 2:
        hh = highs[-1].price > highs[-2].price
        hl = lows[-1].price > lows[-2].price
        lh = highs[-1].price < highs[-2].price
        ll = lows[-1].price < lows[-2].price
        if hh and hl:
            trend = UP
        elif lh and ll:
            trend = DOWN

    event = "none"
    if not math.isnan(last_high) and not math.isnan(last_low):
        if trend == UP:
            if last_price > last_high:
                event = "BOS_up"
            elif last_price < last_low:
                event = "CHoCH_down"
        elif trend == DOWN:
            if last_price < last_low:
                event = "BOS_down"
            elif last_price > last_high:
                event = "CHoCH_up"
        else:
            if last_price > last_high:
                event = "BOS_up"
            elif last_price < last_low:
                event = "BOS_down"

    return StructureState(trend=trend, event=event, last_high=last_high,
                          last_low=last_low, sequence=sequence)


# --------------------------------------------------------------------------- #
# Levels: cluster, score, prune
# --------------------------------------------------------------------------- #
def cluster_levels(
    swings: list[Swing],
    tolerance: float,
    last_price: float,
    n_bars: int,
    recency_window: int,
) -> list[Level]:
    """Single-linkage cluster swing prices into scored S/R levels."""
    if not swings:
        return []
    pts = sorted(swings, key=lambda s: s.price)
    clusters: list[list[Swing]] = [[pts[0]]]
    for s in pts[1:]:
        if s.price - clusters[-1][-1].price <= tolerance:
            clusters[-1].append(s)
        else:
            clusters.append([s])

    levels: list[Level] = []
    for cl in clusters:
        price = sum(s.price for s in cl) / len(cl)
        last_touch = max(s.index for s in cl)
        recency = max(0.0, 1.0 - (n_bars - 1 - last_touch) / recency_window) if recency_window > 0 else 0.0
        reaction = sum(s.leg_atr for s in cl) / len(cl)
        n_high = sum(1 for s in cl if s.kind == HIGH)
        n_low = len(cl) - n_high
        side = BUY_LIQ if n_high > n_low else SELL_LIQ if n_low > n_high else MIXED
        strength = len(cl) + recency + min(reaction, 3.0) * 0.5
        levels.append(
            Level(
                price=price,
                touches=len(cl),
                kind=RESISTANCE if price >= last_price else SUPPORT,
                side=side,
                last_touch_index=last_touch,
                strength=strength,
                reaction_atr=reaction,
            )
        )
    return levels


def _add_confluence(levels: list[Level], round_lvls: list[float],
                    range_hi: float, range_lo: float, tol: float) -> None:
    """Bump level strength where a round number / range edge agrees."""
    for lv in levels:
        for r in round_lvls:
            if abs(lv.price - r) <= tol:
                lv.confluence.append("round")
                lv.strength += 0.5
                break
        for edge, name in ((range_hi, "range_high"), (range_lo, "range_low")):
            if not math.isnan(edge) and abs(lv.price - edge) <= tol:
                lv.confluence.append(name)
                lv.strength += 0.5


def prune_levels(levels: list[Level], min_separation: float, max_levels: int) -> list[Level]:
    """Keep the strongest levels, enforcing a minimum price separation."""
    kept: list[Level] = []
    for lv in sorted(levels, key=lambda x: x.strength, reverse=True):
        if all(abs(lv.price - k.price) >= min_separation for k in kept):
            kept.append(lv)
        if len(kept) >= max_levels:
            break
    return sorted(kept, key=lambda x: x.price)


# --------------------------------------------------------------------------- #
# Context levels
# --------------------------------------------------------------------------- #
def round_levels(price: float, span_pct: float = 15.0, divisions=(1.0, 0.5)) -> list[float]:
    if price <= 0:
        return []
    magnitude = 10.0 ** math.floor(math.log10(price))
    lo = price * (1.0 - span_pct / 100.0)
    hi = price * (1.0 + span_pct / 100.0)
    levels: set[float] = set()
    for div in divisions:
        step = magnitude * div
        if step <= 0:
            continue
        v = math.floor(lo / step) * step
        while v <= hi + step * 1e-9:
            if lo <= v <= hi and v > 0:
                levels.add(round(v, 12))
            v += step
    return sorted(levels)


def range_edges(df: pd.DataFrame, window: int) -> tuple[float, float]:
    # The PRIOR range (exclude the current/decision bar) so the current bar can
    # genuinely sweep or break it — otherwise `low < range_low` is impossible
    # (the current low is part of the min) and e.g. failed_breakout never fires.
    seg = df.iloc[-(window + 1):-1]
    if len(seg) == 0:
        return float("nan"), float("nan")
    return float(seg["high"].max()), float(seg["low"].min())


# --------------------------------------------------------------------------- #
# Retracement / measured move
# --------------------------------------------------------------------------- #
def retracement_levels(swings: list[Swing]) -> tuple[dict[str, float], float | None]:
    """Retracements (38/50/61.8/79%) of the last leg + a measured-move projection."""
    if len(swings) < 2:
        return {}, None
    a, b = swings[-2], swings[-1]      # last completed leg a -> b
    lo, hi = min(a.price, b.price), max(a.price, b.price)
    span = hi - lo
    if span <= 0:
        return {}, None
    if b.kind == HIGH:                 # up leg -> retraces down from hi
        retr = {f"{r:.3f}": hi - r * span for r in (0.382, 0.5, 0.618, 0.786)}
        measured = hi + span           # measured move up
    else:                              # down leg -> retraces up from lo
        retr = {f"{r:.3f}": lo + r * span for r in (0.382, 0.5, 0.618, 0.786)}
        measured = lo - span           # measured move down
    return retr, measured


# --------------------------------------------------------------------------- #
# Volume profile
# --------------------------------------------------------------------------- #
def volume_profile(df: pd.DataFrame, window: int, bins: int, value_area_pct: float) -> VolumeProfile | None:
    seg = df.tail(window)
    lo = float(seg["low"].min())
    hi = float(seg["high"].max())
    if not (hi > lo) or bins < 2:
        return None
    edges = np.linspace(lo, hi, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    typical = ((seg["high"] + seg["low"] + seg["close"]) / 3.0).to_numpy()
    vol = seg["volume"].to_numpy(dtype=float)
    bin_idx = np.clip(np.digitize(typical, edges) - 1, 0, bins - 1)
    hist = np.zeros(bins)
    np.add.at(hist, bin_idx, vol)

    total = hist.sum()
    if total <= 0:
        return None
    poc_bin = int(hist.argmax())
    target = total * value_area_pct / 100.0
    lo_b = hi_b = poc_bin
    acc = hist[poc_bin]
    while acc < target and (lo_b > 0 or hi_b < bins - 1):
        left = hist[lo_b - 1] if lo_b > 0 else -1.0
        right = hist[hi_b + 1] if hi_b < bins - 1 else -1.0
        if right >= left:
            hi_b += 1
            acc += hist[hi_b]
        else:
            lo_b -= 1
            acc += hist[lo_b]
    return VolumeProfile(poc=float(centers[poc_bin]),
                         vah=float(centers[hi_b]),
                         val=float(centers[lo_b]))


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def analyze_structure(
    df: pd.DataFrame,
    cfg: StructureConfig | None = None,
    atr_value: float | None = None,
    with_volume_profile: bool = True,
) -> Structure:
    """Build the full expert structure read for a coin's OHLC frame.

    `with_volume_profile=False` skips the (relatively expensive) volume-profile
    histogram — used by the backtest hot loop, since the setups don't read it.
    """
    cfg = cfg or StructureConfig()
    last_price = float(df["close"].iloc[-1])

    atr_series = atr(df, cfg.atr_length)
    if atr_value is None:
        atr_value = float(atr_series.iloc[-1])
    ref_atr = atr_value if (atr_value == atr_value and atr_value > 0) else last_price * 0.01
    tolerance = cfg.level_atr_fraction * ref_atr

    swings = find_swings(df, atr_series.to_numpy(), cfg.reversal_atr)
    # major flag uses major_leg_atr, applied after leg sizing
    for s in swings:
        s.major = s.leg_atr >= cfg.major_leg_atr

    levels = cluster_levels(swings, tolerance, last_price, len(df), cfg.recency_window)
    rlevels = round_levels(last_price, cfg.round_span_pct)
    range_high, range_low = range_edges(df, cfg.range_window)
    _add_confluence(levels, rlevels, range_high, range_low, tolerance)
    levels = prune_levels(levels, cfg.min_separation_atr * ref_atr, cfg.max_levels)

    state = classify_structure(swings, last_price)
    retr, measured = retracement_levels(swings)
    vp = volume_profile(df, cfg.vp_window, cfg.vp_bins, cfg.value_area_pct) if with_volume_profile else None

    # liquidity pools: levels with >= eq_touches same-kind swings
    buy_side = sorted(
        lv.price for lv in levels
        if lv.side == BUY_LIQ and lv.touches >= cfg.eq_touches and lv.price > last_price
    )
    sell_side = sorted(
        (lv.price for lv in levels
         if lv.side == SELL_LIQ and lv.touches >= cfg.eq_touches and lv.price < last_price),
        reverse=True,
    )

    return Structure(
        last_price=last_price,
        swings=swings,
        levels=levels,
        state=state,
        round_levels=rlevels,
        range_high=range_high,
        range_low=range_low,
        retracements=retr,
        measured_move=measured,
        volume_profile=vp,
        buy_side_liquidity=buy_side,
        sell_side_liquidity=sell_side,
    )
