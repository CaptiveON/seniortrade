"""Layer 10 — trade management (manage an OPEN position by the PLAN, never emotion).

PROPOSE-AND-CONFIRM, never autonomous. The management rules ARE the backtest's
state machine: both consume :func:`risk.walk_management`, so what you backtested
is exactly how you manage — no divergence. A dry-run STAGED trade is PAPER-FILLED
on real prices when price reaches its entry, so the whole stage→manage→review
loop (and the edge) runs WITHOUT real money — and so manage is exercisable today.

This module holds the pure, testable pieces (paper-fill detection, the management
step, USD-M warnings); the interactive flow (fetch, CONFIRM prompt, journal
writes) lives in the CLI so this stays network/IO-free — mirroring order.py.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd

from . import risk as rk
from .config import MarketsConfig, RiskConfig
from .markets import LONG, SHORT
from .order import LIMIT, MARKET, STOP

# Proposed-action kinds.
HOLD = "hold"
PAPER_FILL = "paper_fill"
SCALE_TP1 = "scale_tp1"
CLOSE_TP2 = "close_tp2"
STOP_EXIT = "stop_exit"
TRAIL = "trail"
CANCEL = "cancel"


@dataclass
class ManageAction:
    """One PROPOSED management action — the user confirms before anything happens."""

    kind: str
    reason: str
    closes: bool = False                    # does confirming this fully close the position?
    new_stop: float | None = None
    fill_price: float | None = None
    entry_actual: float | None = None       # for PAPER_FILL
    realized_delta_r: float | None = None   # R booked by THIS action
    total_realized_r: float | None = None   # locked_r + this, when it closes
    remaining_after: float | None = None
    tp1_filled_after: bool | None = None
    warnings: list = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _bars_after(bars: pd.DataFrame, marker: str) -> pd.DataFrame:
    """The closed bars that occurred strictly after the `marker` ISO timestamp."""
    if not marker or len(bars) == 0:
        return bars
    try:
        ts = pd.Timestamp(marker)
    except (ValueError, TypeError):
        return bars
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return bars[bars.index > ts]


def _with_live_bar(bars: pd.DataFrame, current_price: float | None) -> pd.DataFrame:
    """Append the live price as a one-tick forming bar so an intrabar touch counts now."""
    if current_price is None or current_price != current_price:  # None / NaN
        return bars
    row = {"open": current_price, "high": current_price, "low": current_price,
           "close": current_price, "volume": 0.0}
    if len(bars) >= 2:
        idx = bars.index[-1] + (bars.index[-1] - bars.index[-2])
    else:
        idx = pd.Timestamp.now(tz="UTC")
    live = pd.DataFrame([row], index=[idx])
    return pd.concat([bars, live]) if len(bars) else live


def _is_tighter(side: str, cur_stop: float, new_stop: float) -> bool:
    """A trail may only move the stop in the PROFIT direction (never looser)."""
    return new_stop > cur_stop if side == LONG else new_stop < cur_stop


# --------------------------------------------------------------------------- #
# Paper-fill: a STAGED dry-run trade reaches its entry on REAL prices → OPEN
# --------------------------------------------------------------------------- #
def detect_paper_fill(record, bars: pd.DataFrame, current_price: float,
                      expiry_bars: int | None = None) -> tuple[bool, float | None, str]:
    """Has a STAGED trade reached its entry? Returns (filled, entry_price, why).

    A pre-entry invalidation hit before the fill VOIDS the setup (caller cancels).
    Market entries fill at the live price; limit/stop entries fill when their
    level is reached over the bars since the trade was staged. A resting entry
    still unfilled after ``expiry_bars`` closed bars is STALE → voided too (the
    setup was priced for THEN, not indefinitely — audit-walk finding 5).
    """
    et = (record.entry_type or MARKET).lower()
    long = record.side == LONG
    entry = record.entry_planned
    inv = record.invalidation

    def _voided() -> bool:  # the thesis broke (price beyond the invalidation level)
        return bool(inv) and ((long and current_price <= inv) or (not long and current_price >= inv))

    if et == MARKET:  # entered now, at market — don't enter a setup that's already void
        if _voided():
            return False, None, f"invalidation {inv:,.6g} hit — setup void"
        return True, current_price, f"market entry → paper-fill at live {current_price:,.6g}"

    # A resting limit/stop FILLS when its level is reached SINCE staging (plus the live
    # tick) — never pre-stage history. Reaching the entry fills it; the protective stop
    # then takes over. The pre-entry invalidation only cancels an order that is NOT yet filled.
    window = _bars_after(bars, record.managed_at or record.timestamp)
    hi = max([float(h) for h in window["high"]] + [current_price])
    lo = min([float(low) for low in window["low"]] + [current_price])
    if et == STOP:   # stop entry triggers when price trades THROUGH it in the trade direction
        reached, verb = ((long and hi >= entry) or (not long and lo <= entry)), "triggered"
    else:            # limit: buy fills on a dip to entry, sell on a pop to entry
        reached, verb = ((long and lo <= entry) or (not long and hi >= entry)), "touched"
    if reached:
        return True, entry, f"{et} entry {entry:,.6g} {verb}"
    if _voided():
        return False, None, f"invalidation {inv:,.6g} hit before entry — setup void"
    if expiry_bars and len(window) > expiry_bars:
        return False, None, (f"unfilled for {len(window)} bars (> expiry {expiry_bars}) — "
                             "stale, setup void")
    return False, None, f"{et} entry {entry:,.6g} not reached yet"


def replay_after_fill(record, bars: pd.DataFrame, cfg: RiskConfig) -> dict | None:
    """A resting entry TOUCHED somewhere in the bars since staging — what happened NEXT?
    (Audit-walk finding 6: proposing a naive OPEN when the stop was already breached after
    the touch mis-books history.) Replays the SHARED state machine (walk_management — the
    same one the backtest and live manager use) from the fill bar forward:

    returns {"closed", "r", "exit", "held", "fill_ts"} — closed=True means the position
    already completed in history (record the outcome, don't 'open' it); closed=False means
    it is genuinely still open (open it retroactively AT fill_ts so `manage` catches up the
    since-fill bars with the same machine). None when there is no locatable fill bar
    (market entries fill 'now'; nothing to replay)."""
    et = (record.entry_type or MARKET).lower()
    if et == MARKET:
        return None
    long = record.side == LONG
    entry = record.entry_planned
    window = _bars_after(bars, record.managed_at or record.timestamp)
    j = None
    for i in range(len(window)):
        hi, lo = float(window["high"].iloc[i]), float(window["low"].iloc[i])
        if et == STOP:
            reached = (long and hi >= entry) or (not long and lo <= entry)
        else:
            reached = (long and lo <= entry) or (not long and hi >= entry)
        if reached:
            j = i
            break
    if j is None:
        return None
    # Same anti-look-ahead convention as the backtest: the fill bar opens at the entry and,
    # for an intra-bar fill, the FAVOURABLE extreme is clamped to entry (a same-bar target
    # that may have printed BEFORE the fill must never be credited; the stop side stays).
    after = window.iloc[j:].copy()
    after.iloc[0, after.columns.get_loc("open")] = entry
    fav = "high" if long else "low"
    after.iloc[0, after.columns.get_loc(fav)] = entry
    realized, remaining, closed, exit_price, held = 0.0, 1.0, False, None, len(after)
    for ev in rk.walk_management(record.side, entry, record.stop_planned,
                                 list(record.targets), after, cfg):
        realized += ev.realized_delta
        remaining = ev.remaining_after
        if ev.closes:
            closed, exit_price, held = True, ev.fill_price, ev.bar_index + 1
            break
    if not closed:
        risk = abs(entry - record.stop_planned)
        last = float(after["close"].iloc[-1])
        sgn = 1.0 if long else -1.0
        mark = realized + (remaining * sgn * (last - entry) / risk if risk > 0 else 0.0)
        return {"closed": False, "r": mark, "exit": None, "held": len(after),
                "fill_ts": after.index[0].isoformat()}
    return {"closed": True, "r": realized, "exit": exit_price, "held": held,
            "fill_ts": after.index[0].isoformat()}


# --------------------------------------------------------------------------- #
# The management step — the SAME state machine the backtest walks
# --------------------------------------------------------------------------- #
def manage_step(record, bars: pd.DataFrame, current_price: float, cfg: RiskConfig,
                protective_stop: float | None = None) -> ManageAction:
    """Propose the ONE highest-priority management action for an OPEN position.

    Walks :func:`risk.walk_management` over the bars since the position was last
    managed (plus the live price), SEEDED with the position's persisted state —
    worst-case fills, stop-first, identical to the backtest. Returns the first
    transition (or a trail / hold when nothing has triggered).
    """
    side = record.side
    entry = record.entry_actual if record.entry_actual is not None else record.entry_planned
    orig_stop = record.stop_planned                      # fixes the 1R unit
    cur_stop = record.current_stop if record.current_stop is not None else orig_stop
    targets = list(record.targets)
    locked_r = record.locked_r or 0.0

    window = _bars_after(bars, record.managed_at)
    walk_bars = _with_live_bar(window, current_price)

    for ev in rk.walk_management(side, entry, orig_stop, targets, walk_bars, cfg,
                                 remaining=record.remaining_fraction, cur_stop=cur_stop,
                                 hit_tp1=record.tp1_filled):
        if ev.kind == rk.EV_STOP:
            total = locked_r + ev.realized_delta
            where = "breakeven stop" if abs(cur_stop - entry) < 1e-12 else "protective stop"
            return ManageAction(
                STOP_EXIT, f"price hit the {where} {ev.fill_price:,.6g} → EXIT ({total:+.2f}R realised)",
                closes=True, fill_price=ev.fill_price, new_stop=cur_stop,
                realized_delta_r=ev.realized_delta, total_realized_r=total, remaining_after=0.0)
        if ev.kind == rk.EV_TP1:
            return ManageAction(
                SCALE_TP1,
                f"TP1 {targets[0]:,.6g} reached → scale out {cfg.tp1_fraction:g} (+{ev.realized_delta:.2f}R "
                f"booked) and move stop to breakeven {ev.new_stop:,.6g}",
                new_stop=ev.new_stop, fill_price=targets[0], realized_delta_r=ev.realized_delta,
                remaining_after=ev.remaining_after, tp1_filled_after=True)
        if ev.kind == rk.EV_TP2:
            total = locked_r + ev.realized_delta
            return ManageAction(
                CLOSE_TP2, f"TP2 {targets[-1]:,.6g} reached → close the runner ({total:+.2f}R realised)",
                closes=True, fill_price=targets[-1], new_stop=cur_stop,
                realized_delta_r=ev.realized_delta, total_realized_r=total, remaining_after=0.0)

    # No state transition. Offer an optional trail to a tighter protective level.
    if (cfg.trail_to_structure and protective_stop is not None
            and protective_stop == protective_stop and _is_tighter(side, cur_stop, protective_stop)):
        return ManageAction(
            TRAIL, f"trail the stop {cur_stop:,.6g} → {protective_stop:,.6g} (lock more in; never looser)",
            new_stop=protective_stop)
    return ManageAction(
        HOLD, f"holding by the plan — stop {cur_stop:,.6g}, {record.remaining_fraction:g} open, "
              f"no trigger reached (live {current_price:,.6g})")


def unrealized_r(record, current_price: float) -> float:
    """Open R on the remaining fraction, marked to the live price."""
    entry = record.entry_actual if record.entry_actual is not None else record.entry_planned
    risk = abs(entry - record.stop_planned)
    if risk <= 0:
        return 0.0
    sgn = 1.0 if record.side == LONG else -1.0
    return (record.locked_r or 0.0) + record.remaining_fraction * sgn * (current_price - entry) / risk


# --------------------------------------------------------------------------- #
# USD-M watch — liquidation distance + funding window (as price moves)
# --------------------------------------------------------------------------- #
def _minutes_to_funding(now: datetime) -> float:
    """Minutes until the next Binance funding time (00 / 08 / 16 UTC)."""
    mins = now.hour * 60 + now.minute + now.second / 60.0
    period = 8 * 60
    nxt = math.ceil((mins + 1e-6) / period) * period
    return nxt - mins


def usdm_warnings(record, current_price: float, mcfg: MarketsConfig, *,
                  now: datetime | None = None, funding_window_minutes: float = 15.0,
                  funding_rate: float | None = None, funding_signal: str | None = None,
                  oi_trend: str | None = None, oi_change_pct: float | None = None) -> list[str]:
    """Liquidation distance + live funding/OI watch for a USD-M position (as price moves)."""
    warns: list[str] = []
    liq = record.liquidation_price
    if liq:
        cur_stop = record.current_stop if record.current_stop is not None else record.stop_planned
        stop_dist = abs(current_price - cur_stop)
        liq_dist = abs(current_price - liq)
        if stop_dist > 0 and liq_dist <= stop_dist * mcfg.liq_buffer_mult:
            warns.append(f"liquidation {liq:,.6g} is near: {liq_dist:,.6g} away vs your stop's "
                         f"{stop_dist:,.6g} (< {mcfg.liq_buffer_mult:g}× buffer) — size/leverage too hot")
    now = now or datetime.now(timezone.utc)
    mins = _minutes_to_funding(now)
    if mins <= funding_window_minutes:
        msg = f"funding settles in ~{mins:.0f} min"
        if funding_rate is not None:
            side_pays = ((record.side == LONG and funding_rate > 0)
                         or (record.side == SHORT and funding_rate < 0))
            msg += f" (rate {funding_rate * 100:+.4f}% — you {'PAY' if side_pays else 'receive'})"
        warns.append(msg)
    if funding_signal in ("crowded_long", "crowded_short"):
        crowded_side = LONG if funding_signal == "crowded_long" else SHORT
        if record.side == crowded_side:
            warns.append(f"funding shows a CROWDED {record.side} — you're with the crowd; a move against "
                         "a one-sided book can cascade")
    if oi_trend == "rising" and oi_change_pct is not None:
        warns.append(f"open interest rising ({oi_change_pct:+.1f}%) — leverage building; expect sharper moves/squeezes")
    elif oi_trend == "falling" and oi_change_pct is not None:
        warns.append(f"open interest falling ({oi_change_pct:+.1f}%) — positions unwinding; the move may be losing fuel")
    return warns
