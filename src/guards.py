"""Layer 6 — guards: the pre-trade portfolio & behaviour gate.

`check_trade(state, proposed, cfg)` returns a GuardResult — ALLOW / BLOCK (hard) /
WARN (soft) + the guard(s) that fired. It protects the account from the trader:
daily-loss lockout, post-loss cooldown, group-aware portfolio heat (scale-to-fit
or block), max positions / per-group cap, correlation, funding-window, and sanity
bounds. State (open positions, today's realised R, equity, loss streak) is passed
in — the journal populates it; it's empty in dry-run.

Boundary: risk.py builds the per-trade plan; guards is the portfolio gate over it.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .config import GuardsConfig
from .markets import LONG, SHORT


@dataclass
class OpenPosition:
    symbol: str
    group: str
    risk_amount: float       # $ at risk to the stop
    side: str


@dataclass
class PortfolioState:
    equity: float
    open_positions: list = field(default_factory=list)
    realized_r_today: float = 0.0          # since UTC midnight
    consecutive_losses: int = 0
    last_loss_time: datetime | None = None
    drawdown_pct: float = 0.0              # current equity drawdown from peak (for risk-scaling)
    trades_today: int = 0                 # new trades opened since UTC midnight (over-trading guard)


@dataclass
class ProposedTrade:
    symbol: str
    group: str
    side: str
    risk_pct: float
    entry: float
    stop: float
    liquidation: float | None = None


@dataclass
class GuardResult:
    allowed: bool
    hard_blocks: list = field(default_factory=list)
    soft_warns: list = field(default_factory=list)
    adjusted_risk_pct: float | None = None    # set if heat scaled the trade down
    heat_pct: float = 0.0


def group_heat_usd(positions, corr_factor: float) -> float:
    """Group-aware open risk in $: per group, the largest position counts fully and
    the rest at corr_factor (so correlated same-group bets aren't double-counted)."""
    groups: dict[str, list[float]] = defaultdict(list)
    for p in positions:
        groups[p.group].append(p.risk_amount)
    total = 0.0
    for risks in groups.values():
        risks.sort(reverse=True)
        total += risks[0] + corr_factor * sum(risks[1:])
    return total


@dataclass
class Allocation:
    symbol: str
    group: str
    side: str
    score: float
    risk_pct: float
    risk_amount: float


def allocate_basket(items, equity: float, cfg: GuardsConfig, base_risk_pct: float,
                    tilt: bool = False) -> list:
    """Divide the risk budget across a basket of PROVEN trades like a desk would.

    A basket succeeds over MANY trades, so spread risk across DIVERSE edges — but the
    BASKET's total risk stays under the heat cap (N trades are NOT N× the risk).
    Group-aware (correlated coins share a budget), capped by max positions / per group.
    Default EQUAL-risk per chosen bet; `tilt` weights by edge×confidence (each still
    capped at base risk, total still under heat). Returns ranked Allocations.
    """
    ranked = sorted(items, key=lambda x: x.get("score", 0.0), reverse=True)
    chosen: list = []
    per_group: dict = defaultdict(int)
    for it in ranked:
        if len(chosen) >= cfg.max_positions:
            break
        g = it.get("group", "indep")
        if g not in ("indep", "-", "") and per_group[g] >= cfg.max_per_group:
            continue                                   # diversify: cap correlated names
        chosen.append(dict(it))
        per_group[g] += 1
    if not chosen:
        return []

    if tilt:
        total = sum(c.get("score", 0.0) for c in chosen) or 1.0
        for c in chosen:                               # weight by edge×confidence, capped at base
            c["risk_pct"] = min(base_risk_pct, cfg.heat_cap_pct * c.get("score", 0.0) / total)
    else:
        for c in chosen:
            c["risk_pct"] = base_risk_pct

    def _heat() -> float:
        pos = [OpenPosition(c["symbol"], c.get("group", "indep"),
                            equity * c["risk_pct"] / 100.0, c.get("side", LONG)) for c in chosen]
        return basket_heat_usd(pos, cfg.group_corr_factor)

    cap_usd = equity * cfg.heat_cap_pct / 100.0
    heat = _heat()
    if heat > cap_usd and heat > 0:                    # scale the whole basket to fit the cap
        scale = cap_usd / heat
        for c in chosen:
            c["risk_pct"] *= scale

    return [Allocation(symbol=c["symbol"], group=c.get("group", "indep"), side=c.get("side", LONG),
                       score=c.get("score", 0.0), risk_pct=c["risk_pct"],
                       risk_amount=equity * c["risk_pct"] / 100.0) for c in chosen]


def basket_heat_usd(positions, corr_factor: float) -> float:
    """Group-aware heat, but 'indep' names are each their OWN bet (fully additive) —
    only genuinely-correlated groups get the correlation discount."""
    remapped = [OpenPosition(p.symbol, (p.symbol if p.group in ("indep", "-", "") else p.group),
                             p.risk_amount, p.side) for p in positions]
    return group_heat_usd(remapped, corr_factor)


def _in_funding_window(now: datetime, minutes: float) -> bool:
    """Within `minutes` of a Binance funding time (00 / 08 / 16 UTC)."""
    cur = now.hour + now.minute / 60.0 + now.second / 3600.0
    nearest = min((0, 8, 16, 24), key=lambda fh: abs(fh - cur))
    return abs(nearest - cur) * 60.0 <= minutes


def check_trade(state: PortfolioState, proposed: ProposedTrade, cfg: GuardsConfig,
                now: datetime | None = None) -> GuardResult:
    now = now or datetime.now(timezone.utc)
    hard: list[str] = []
    soft: list[str] = []

    # --- sanity bounds (hard) ---
    if proposed.entry <= 0 or proposed.risk_pct <= 0 or proposed.stop == proposed.entry:
        hard.append("sanity: invalid entry / stop / risk%")
    if proposed.side == LONG and not proposed.stop < proposed.entry:
        hard.append("sanity: long needs stop < entry")
    if proposed.side == SHORT and not proposed.stop > proposed.entry:
        hard.append("sanity: short needs stop > entry")
    if proposed.liquidation is not None:
        if proposed.side == LONG and proposed.liquidation >= proposed.stop:
            hard.append("sanity: liquidation is not beyond the stop")
        if proposed.side == SHORT and proposed.liquidation <= proposed.stop:
            hard.append("sanity: liquidation is not beyond the stop")

    # --- daily-loss lockout (hard) ---
    if state.realized_r_today <= -cfg.daily_loss_limit_r:
        hard.append(f"daily-loss lockout: {state.realized_r_today:+.1f}R today "
                    f"≤ −{cfg.daily_loss_limit_r:g}R — no new trades until UTC reset")

    # --- cooldown (hard) ---
    if state.consecutive_losses >= cfg.cooldown_consecutive_losses:
        hard.append(f"cooldown: {state.consecutive_losses} consecutive losses → session lock")
    elif state.last_loss_time is not None:
        mins = (now - state.last_loss_time).total_seconds() / 60.0
        if 0 <= mins < cfg.cooldown_minutes:
            hard.append(f"cooldown: {mins:.0f}min since last loss < {cfg.cooldown_minutes:g}min")

    # --- max positions (hard) ---
    if len(state.open_positions) >= cfg.max_positions:
        hard.append(f"max positions: {len(state.open_positions)}/{cfg.max_positions} open")

    # --- max trades per day (hard, over-trading guard; 0 = unlimited) ---
    if cfg.max_trades_per_day > 0 and state.trades_today >= cfg.max_trades_per_day:
        hard.append(f"max trades/day: {state.trades_today}/{cfg.max_trades_per_day} taken today "
                    f"— over-trading guard, no new trades until UTC reset")

    # --- correlation / per-group (soft) ---
    if proposed.group not in ("indep", "-", ""):
        same = [p for p in state.open_positions if p.group == proposed.group]
        if same:
            soft.append(f"correlation: {len(same)} open position(s) in group "
                        f"{proposed.group} — really one bet")
            if len(same) >= cfg.max_per_group:
                soft.append(f"per-group cap: {len(same)}/{cfg.max_per_group} in {proposed.group}")

    # --- event guard: funding window (soft) ---
    if _in_funding_window(now, cfg.funding_window_minutes):
        soft.append("event guard: within a funding-time window (00/08/16 UTC)")

    # --- portfolio heat (hard, group-aware) with scale-to-fit ---
    risk_amt = state.equity * proposed.risk_pct / 100.0
    cap_usd = state.equity * cfg.heat_cap_pct / 100.0
    current = group_heat_usd(state.open_positions, cfg.group_corr_factor)
    new_pos = OpenPosition(proposed.symbol, proposed.group, risk_amt, proposed.side)
    with_new = group_heat_usd(state.open_positions + [new_pos], cfg.group_corr_factor)
    adjusted = None
    heat_pct = with_new / state.equity * 100.0 if state.equity > 0 else 0.0
    if with_new > cap_usd:
        marginal = with_new - current
        headroom = cap_usd - current
        if headroom <= 0 or marginal <= 0:
            hard.append(f"portfolio heat: already {current / state.equity * 100:.1f}% at/over "
                        f"the {cfg.heat_cap_pct:g}% cap — no room")
        else:
            adjusted = (risk_amt * headroom / marginal) / state.equity * 100.0
            soft.append(f"heat scale-to-fit: risk {proposed.risk_pct:.2f}%→{adjusted:.2f}% "
                        f"to stay under the {cfg.heat_cap_pct:g}% heat cap")
            heat_pct = cfg.heat_cap_pct

    return GuardResult(allowed=len(hard) == 0, hard_blocks=hard, soft_warns=soft,
                       adjusted_risk_pct=adjusted, heat_pct=heat_pct)
