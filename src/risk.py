"""Risk & trade management (Layer 4).

Two halves:
  - the EXPLAIN-AS-YOU-GO worked-example formatter (build_worked_example/...), and
  - the trade ENGINE: risk-first sizing reconciled to lot rounding, honest NET
    R:R after fees + slippage + funding (which can flip a setup to NO TRADE), the
    deterministic look-ahead-safe trade-management state machine (shared with the
    backtest), drawdown-scaled sizing, and a fractional-Kelly cap.

Nothing here predicts price.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from rich.panel import Panel
from rich.text import Text

from .config import RiskConfig
from .markets import LONG, SHORT


@dataclass
class WorkedExample:
    side: str                # "long" or "short"
    account_equity: float
    risk_pct: float
    entry: float
    stop: float
    take_profit: float
    risk_amount: float       # dollars risked = equity * risk_pct%
    per_unit_risk: float     # |entry - stop| per coin/contract
    position_size: float     # coins/contracts
    notional: float          # position_size * entry
    reward_amount: float     # dollars gained if TP hit
    rr: float                # reward : risk multiple


def build_worked_example(
    *,
    account_equity: float,
    risk_pct: float,
    entry: float,
    stop: float,
    take_profit: float,
    side: str = "long",
) -> WorkedExample:
    """Risk-first position sizing.

    Step 1: decide the dollars at risk = equity * risk_pct%.
    Step 2: per-unit risk = distance from entry to stop.
    Step 3: size = dollars-at-risk / per-unit risk. Size is an *output*.
    """
    side = side.lower()
    if side not in ("long", "short"):
        raise ValueError("side must be 'long' or 'short'")
    if account_equity <= 0:
        raise ValueError("account_equity must be positive")
    if risk_pct <= 0:
        raise ValueError("risk_pct must be positive")

    # Sanity: stop/TP must sit on the correct side of entry for the direction.
    if side == "long" and not (stop < entry < take_profit):
        raise ValueError("long requires stop < entry < take_profit")
    if side == "short" and not (take_profit < entry < stop):
        raise ValueError("short requires take_profit < entry < stop")

    risk_amount = account_equity * (risk_pct / 100.0)
    per_unit_risk = abs(entry - stop)
    if per_unit_risk <= 0:
        raise ValueError("entry and stop cannot be equal (zero risk distance)")

    position_size = risk_amount / per_unit_risk
    notional = position_size * entry

    if side == "long":
        reward_amount = (take_profit - entry) * position_size
    else:
        reward_amount = (entry - take_profit) * position_size

    rr = abs(take_profit - entry) / per_unit_risk

    return WorkedExample(
        side=side,
        account_equity=account_equity,
        risk_pct=risk_pct,
        entry=entry,
        stop=stop,
        take_profit=take_profit,
        risk_amount=risk_amount,
        per_unit_risk=per_unit_risk,
        position_size=position_size,
        notional=notional,
        reward_amount=reward_amount,
        rr=rr,
    )


def _fmt_price(value: float) -> str:
    """Format a price with precision scaled to its magnitude."""
    if value == 0:
        return "0"
    a = abs(value)
    if a >= 1000:
        return f"{value:,.2f}"
    if a >= 1:
        return f"{value:,.4f}"
    if a >= 0.01:
        return f"{value:.5f}"
    return f"{value:.8f}"


def _fmt_usd(value: float) -> str:
    return f"${value:,.2f}"


def _fmt_size(value: float) -> str:
    if abs(value) >= 1:
        return f"{value:,.4f}"
    return f"{value:.6f}"


def worked_example_lines(we: WorkedExample, symbol: str) -> list[str]:
    """The brief's dollar lines, in order. Returned as plain strings (testable)."""
    base = symbol.split("/")[0]
    return [
        f"With {_fmt_usd(we.account_equity)} account and {we.risk_pct:g}% risk: "
        f"you risk {_fmt_usd(we.risk_amount)} on this trade.",
        f"Entry {_fmt_price(we.entry)}, stop {_fmt_price(we.stop)} → "
        f"if stopped you lose ~{_fmt_usd(we.risk_amount)} (your {we.risk_pct:g}%).",
        f"Take-profit {_fmt_price(we.take_profit)} → "
        f"if hit you make ~{_fmt_usd(we.reward_amount)} ({we.rr:.2f}R).",
        f"Ideal position size: {_fmt_size(we.position_size)} {base} "
        f"(notional {_fmt_usd(we.notional)}).",
        f"Worst case (stop): -{_fmt_usd(we.risk_amount)}.  "
        f"Planned best case (TP): +{_fmt_usd(we.reward_amount)}.  "
        f"R:R 1:{we.rr:.2f}.",
    ]


def render_worked_example(
    we: WorkedExample,
    symbol: str,
    market_label: str,
    *,
    illustrative: bool = True,
) -> Panel:
    """Render the worked example as a rich Panel for the CLI."""
    body = Text()
    for line in worked_example_lines(we, symbol):
        body.append("  • ", style="bold cyan")
        body.append(line + "\n")

    title = f"Worked dollar example — {symbol} ({market_label}, {we.side})"
    if illustrative:
        note = Text(
            "\nIllustrative ONLY (not a trade signal): stop derived from ATR, "
            "TP set to the default reward:risk. It shows the mechanics on real "
            "numbers — it is not a prediction or a recommendation to trade.",
            style="italic dim",
        )
        body.append(note)

    return Panel(body, title=title, border_style="green", title_align="left")


# =========================================================================== #
# Trade engine: sizing reconciliation, net costs, management state machine
# =========================================================================== #
@dataclass
class TradePlan:
    symbol: str
    side: str
    entry: float
    stop: float
    targets: list[float]
    size: float
    notional: float
    leverage: float
    margin: float
    liquidation_price: float | None
    risk_budget: float           # equity * risk%
    risk_actual: float           # size * per-unit risk (AFTER lot rounding)
    per_unit_risk: float
    gross_rr: float              # to TP1, before costs
    net_rr: float                # to TP1, after costs
    fee_cost: float
    slippage_cost: float
    funding_cost: float
    total_cost: float
    valid: bool
    scenarios: dict[str, float]  # realized-R: stopped / tp1_then_breakeven / tp1_then_tp2
    notes: list[str] = field(default_factory=list)


def _scenarios(side: str, entry: float, stop: float, targets: list[float], cfg: RiskConfig) -> dict[str, float]:
    """Realized-R outcomes for the partial-TP plan (for the blended worked example)."""
    risk = abs(entry - stop)
    if risk <= 0:
        return {}
    sgn = 1.0 if side == LONG else -1.0
    rr1 = sgn * (targets[0] - entry) / risk
    rr2 = sgn * (targets[-1] - entry) / risk
    f1 = cfg.tp1_fraction
    return {
        "stopped": -1.0,
        "tp1_then_breakeven": f1 * rr1,                 # runner stopped at breakeven
        "tp1_then_tp2": f1 * rr1 + (1.0 - f1) * rr2,
    }


def plan_trade(
    market_obj,
    *,
    symbol: str,
    side: str,
    entry: float,
    stop: float,
    targets: list[float],
    account_equity: float,
    risk_pct: float,
    mgmt: RiskConfig,
    funding_rate: float | None = None,
) -> TradePlan:
    """Build a validated, R-native trade plan with honest NET costs."""
    notes: list[str] = []
    risk_budget = account_equity * risk_pct / 100.0
    sizing = market_obj.size_from_risk(symbol, entry, stop, risk_budget, side)
    notes += list(sizing.notes)

    size = sizing.size
    per_unit = sizing.per_unit_risk
    notional = sizing.notional
    risk_actual = size * per_unit
    valid = sizing.valid

    # post-rounding reconciliation: min-notional must not force us over budget
    if valid and risk_actual > risk_budget * 1.05:
        notes.append(f"min-notional forces actual risk ${risk_actual:,.2f} > budget "
                     f"${risk_budget:,.2f} → NO TRADE (or raise risk%)")
        valid = False

    # costs (both sides)
    taker = market_obj.fees().taker
    slippage_bps = getattr(market_obj.cfg, "slippage_bps", 0.0)
    fee_cost = notional * taker * 2.0
    slippage_cost = notional * (slippage_bps / 1e4) * 2.0
    funding_cost = 0.0
    if funding_rate is not None and hasattr(market_obj, "funding_cost"):
        funding_cost = market_obj.funding_cost(notional, side, funding_rate)  # signed: + = you pay
    total_cost = fee_cost + slippage_cost + funding_cost

    tp1 = targets[0]
    gross_reward = abs(tp1 - entry) * size
    gross_rr = abs(tp1 - entry) / per_unit if per_unit > 0 else 0.0
    net_reward = gross_reward - total_cost
    net_loss = risk_actual + total_cost
    net_rr = net_reward / net_loss if net_loss > 0 else 0.0
    if valid and net_rr < mgmt.min_net_rr:
        notes.append(f"net R:R {net_rr:.2f} < min {mgmt.min_net_rr} after costs → not worth taking")
        valid = False

    return TradePlan(
        symbol=symbol, side=side, entry=entry, stop=stop, targets=list(targets),
        size=size, notional=notional, leverage=sizing.leverage, margin=sizing.margin,
        liquidation_price=sizing.liquidation_price, risk_budget=risk_budget,
        risk_actual=risk_actual, per_unit_risk=per_unit, gross_rr=gross_rr, net_rr=net_rr,
        fee_cost=fee_cost, slippage_cost=slippage_cost, funding_cost=funding_cost,
        total_cost=total_cost, valid=valid, scenarios=_scenarios(side, entry, stop, targets, mgmt),
        notes=notes,
    )


# Management-event kinds — the transitions of the shared state machine (Layer 10).
EV_STOP = "stop"
EV_TP1 = "tp1"
EV_TP2 = "tp2"


@dataclass
class MgmtEvent:
    """One deterministic transition of the trade-management state machine."""

    kind: str               # EV_STOP / EV_TP1 / EV_TP2
    bar_index: int          # 0-based index into the bars walked
    fill_price: float
    realized_delta: float   # R booked by this event (for the fraction it affects)
    new_stop: float         # stop AFTER this event (breakeven on TP1)
    remaining_after: float
    closes: bool            # does this fully close the remaining position?


def walk_management(side: str, entry: float, stop: float, targets: list[float],
                    bars: pd.DataFrame, cfg: RiskConfig, *, remaining: float = 1.0,
                    cur_stop: float | None = None, hit_tp1: bool = False):
    """Yield the management transitions, in order, for `bars` walked after entry.

    THE single source of truth for trade management: the backtest
    (``simulate_detailed``) and the live manager (``monitor.manage_step``) both
    consume this generator, so what you backtest is exactly how you manage — no
    divergence. Conservative fills: a gap past the stop fills at the gap; a bar
    spanning both stop and target counts as the STOP (worst case).

    ``remaining``/``cur_stop``/``hit_tp1`` seed the walk from a position's CURRENT
    state — the live manager passes its persisted state; the backtest starts fresh.
    """
    risk = abs(entry - stop)
    if risk <= 0 or len(bars) == 0:
        return
    tp1, tp2 = targets[0], targets[-1]
    f1 = cfg.tp1_fraction
    sgn = 1.0 if side == LONG else -1.0
    cur_stop = stop if cur_stop is None else cur_stop
    for i, (_, b) in enumerate(bars.iterrows()):
        o, h, l = float(b["open"]), float(b["high"]), float(b["low"])
        # gap through the stop at the open -> fill at the gap (worst case)
        if (side == LONG and o <= cur_stop) or (side == SHORT and o >= cur_stop):
            yield MgmtEvent(EV_STOP, i, o, remaining * sgn * (o - entry) / risk, cur_stop, 0.0, True)
            return
        # same-bar stop touch: assume the STOP filled first (worst case)
        if (side == LONG and l <= cur_stop) or (side == SHORT and h >= cur_stop):
            yield MgmtEvent(EV_STOP, i, cur_stop, remaining * sgn * (cur_stop - entry) / risk, cur_stop, 0.0, True)
            return
        # TP1 -> scale out + move to breakeven
        if not hit_tp1 and ((side == LONG and h >= tp1) or (side == SHORT and l <= tp1)):
            delta = f1 * sgn * (tp1 - entry) / risk
            remaining = remaining - f1
            hit_tp1 = True
            if cfg.move_to_breakeven_on_tp1:
                cur_stop = entry
            yield MgmtEvent(EV_TP1, i, tp1, delta, cur_stop, remaining, False)
        # TP2 -> close the runner
        if hit_tp1 and remaining > 0 and ((side == LONG and h >= tp2) or (side == SHORT and l <= tp2)):
            yield MgmtEvent(EV_TP2, i, tp2, remaining * sgn * (tp2 - entry) / risk, cur_stop, 0.0, True)
            return


def simulate_detailed(side: str, entry: float, stop: float, targets: list[float],
                      bars: pd.DataFrame, cfg: RiskConfig) -> tuple[float, int, float]:
    """Deterministic, look-ahead-safe trade management → (realized_R, bars_held, exit_price).

    A thin reduction over :func:`walk_management` (the shared state machine): book
    each event's R; if one closes the position, stop there; otherwise mark the
    runner to the last close. Bars are the candles AFTER entry, walked in order.
    """
    risk = abs(entry - stop)
    if risk <= 0 or len(bars) == 0:
        return 0.0, 0, entry
    sgn = 1.0 if side == LONG else -1.0
    realized = 0.0
    remaining = 1.0
    for ev in walk_management(side, entry, stop, targets, bars, cfg):
        realized += ev.realized_delta
        remaining = ev.remaining_after
        if ev.closes:
            return realized, ev.bar_index + 1, ev.fill_price
    last = float(bars["close"].iloc[-1])
    return realized + remaining * sgn * (last - entry) / risk, len(bars), last


def simulate(side: str, entry: float, stop: float, targets: list[float],
             bars: pd.DataFrame, cfg: RiskConfig) -> float:
    """Realized R from the trade-management state machine (see simulate_detailed)."""
    return simulate_detailed(side, entry, stop, targets, bars, cfg)[0]


def drawdown_scaled_risk(base_risk_pct: float, drawdown_pct: float, cfg: RiskConfig) -> float:
    """Reduce risk% as drawdown deepens (capital protection in losing streaks)."""
    dd = abs(drawdown_pct)
    if dd <= cfg.dd_scale_start:
        return base_risk_pct
    if dd >= cfg.dd_scale_full:
        return base_risk_pct * cfg.dd_scale_floor
    frac = (dd - cfg.dd_scale_start) / (cfg.dd_scale_full - cfg.dd_scale_start)
    return base_risk_pct * (1.0 - frac * (1.0 - cfg.dd_scale_floor))


def kelly_fraction(win_rate: float, avg_win_r: float, avg_loss_r: float) -> float:
    """Full-Kelly fraction f* = W − (1−W)/R, floored at 0 (R = avg win / avg loss)."""
    if avg_loss_r <= 0:
        return 0.0
    R = avg_win_r / avg_loss_r
    if R <= 0:
        return 0.0
    return max(0.0, win_rate - (1.0 - win_rate) / R)


def kelly_capped_risk(base_risk_pct: float, win_rate: float, avg_win_r: float,
                      avg_loss_r: float, cfg: RiskConfig) -> float:
    """Fractional-Kelly cap — only ever REDUCES risk vs base; off by default."""
    if not cfg.kelly_enabled:
        return base_risk_pct
    kelly_pct = kelly_fraction(win_rate, avg_win_r, avg_loss_r) * cfg.kelly_fraction * 100.0
    return min(base_risk_pct, kelly_pct) if kelly_pct > 0 else base_risk_pct


@dataclass
class RiskAdjustment:
    """The result of the effective-risk pipeline: base % → effective %, and which stages fired."""

    base_pct: float
    effective_pct: float
    notes: list = field(default_factory=list)


def effective_risk_pct(base_pct: float, cfg: RiskConfig, *, drawdown_pct: float = 0.0,
                       edge_r: float | None = None, confidence: float | None = None,
                       sigma_r: float | None = None,
                       win_rate: float | None = None, avg_win_r: float | None = None,
                       avg_loss_r: float | None = None) -> RiskAdjustment:
    """Compose the per-trade risk % from the base, gated by each (opt-in) stage:
    base → ×drawdown-scale → ×edge-scale → ×vol-target(σ_R) → Kelly cap → clamp to max_risk_pct.
    With every stage off (defaults) this returns the flat base %. Pure + testable."""
    pct = base_pct
    notes: list[str] = []

    if cfg.dd_scale_enabled and drawdown_pct:
        scaled = drawdown_scaled_risk(pct, drawdown_pct, cfg)
        if scaled < pct - 1e-12:
            notes.append(f"drawdown {abs(drawdown_pct):.1f}% → ×{(scaled / pct if pct else 0):.2f} = {scaled:.2f}%")
            pct = scaled

    if cfg.edge_scaled and edge_r is not None and cfg.edge_ref_r > 0:
        mult = max(0.0, edge_r) / cfg.edge_ref_r
        if confidence is not None:
            mult *= max(0.0, min(1.0, confidence))
        mult = max(cfg.edge_min_mult, min(cfg.edge_max_mult, mult))
        notes.append(f"edge {edge_r:+.2f}R×conf → ×{mult:.2f} = {pct * mult:.2f}%")
        pct = pct * mult

    # VOLATILITY TARGET (P12): equalise per-trade dollar-volatility — size ∝ target_σ / σ_R,
    # clamped. Default vol_max_mult=1.0 → only ever REDUCES (a wilder setup is sized smaller).
    if cfg.vol_target_enabled and sigma_r and sigma_r > 0 and cfg.vol_target_sigma_r > 0:
        mult = max(cfg.vol_min_mult, min(cfg.vol_max_mult, cfg.vol_target_sigma_r / sigma_r))
        if abs(mult - 1.0) > 1e-12:
            notes.append(f"vol-target σ_R {sigma_r:.2f} (ref {cfg.vol_target_sigma_r:.2f}) → ×{mult:.2f} = {pct * mult:.2f}%")
            pct = pct * mult

    if cfg.kelly_enabled and None not in (win_rate, avg_win_r, avg_loss_r):
        capped = kelly_capped_risk(pct, win_rate, avg_win_r, avg_loss_r, cfg)
        if capped < pct - 1e-12:
            notes.append(f"Kelly cap → {capped:.2f}%")
            pct = capped

    if cfg.max_risk_pct > 0 and pct > cfg.max_risk_pct:
        notes.append(f"ceiling {cfg.max_risk_pct:.2f}% applied (was {pct:.2f}%)")
        pct = cfg.max_risk_pct

    return RiskAdjustment(base_pct=base_pct, effective_pct=max(0.0, pct), notes=notes)


def plan_worked_lines(plan: TradePlan) -> list[str]:
    """Net + blended dollar lines for a TradePlan."""
    s = plan.scenarios
    base = plan.symbol.split("/")[0]
    risk_amt = plan.risk_actual + plan.total_cost
    return [
        f"Risk {_fmt_usd(plan.risk_budget)} budget → actual {_fmt_usd(plan.risk_actual)} "
        f"+ {_fmt_usd(plan.total_cost)} costs = {_fmt_usd(risk_amt)} at risk.",
        f"Entry {_fmt_price(plan.entry)}, stop {_fmt_price(plan.stop)} → "
        f"size {_fmt_size(plan.size)} {base} (notional {_fmt_usd(plan.notional)}).",
        f"Gross R:R {plan.gross_rr:.2f} → NET {plan.net_rr:.2f} after fees+slippage"
        + ("+funding" if plan.funding_cost else "") + ".",
        f"Outcomes: stopped {s.get('stopped', 0):+.2f}R · "
        f"TP1→breakeven {s.get('tp1_then_breakeven', 0):+.2f}R · "
        f"TP1→TP2 {s.get('tp1_then_tp2', 0):+.2f}R.",
    ]
