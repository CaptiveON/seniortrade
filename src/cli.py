"""Command-line interface.

Live now (build-order steps 1–4): `screen` (context-aware ranked shortlist with a
worked dollar example) and `context` (BTC/market posture + leaders/laggards). The
remaining subcommands are intentional stubs that name their build-order step —
the brief mandates building in order, proving each step before the next.

Run:
    python -m src.cli context --usdm
    python -m src.cli screen --spot
    python -m src.cli screen --usdm --top 15 --equity 2000 --risk 0.5
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text

from . import market_context as mc
from .alerts import add_level_alert, add_setup_alert, check_alerts, list_alerts, remove_alert
from .analysis import BEAR, BULL, NEUTRAL
from .analysis import analyze as run_analyze
from .analysis import rank_board
from .backtest import run_backtest
from .config import TIERS, Settings, apply_tier
from . import order as od
from . import monitor as mon
from . import live as lv
from . import optimize as opt
from .setups import SETUP_NAMES
from . import edge_score as es
from .edge_score import diff_boards
from .edge_score import scan as run_scan
from . import guards as gd
from .guards import check_trade
from .journal import (
    OPEN,
    STAGED,
    build_review,
    load_records,
    portfolio_state,
    record_cancel,
    record_close,
    record_manage,
    record_open,
    record_staged,
)
from .expectancy import VERDICT_EV, VERDICT_FEW, VERDICT_INCONCLUSIVE, VERDICT_NEG
from .config import DATA_DIR, PROJECT_ROOT, Market, load_settings, with_overrides
from .data_fetch import DataError, drop_unclosed, fetch_ohlcv, load_markets, make_exchange
from .markets import make_market
from .indicators import (
    CHOPPY,
    EXPANDING,
    EXTENDED,
    FRESH,
    RANGING,
    SQUEEZE,
    STRETCHED,
    TRENDING,
)
from .risk import (
    build_worked_example,
    effective_risk_pct,
    plan_trade,
    plan_worked_lines,
    render_worked_example,
    worked_example_lines,
)
from .screener import Candidate, ScreenResult, relative_strength_scan, run_screen

_BIAS_STYLE = {BULL: "bold green", BEAR: "bold red", NEUTRAL: "yellow"}
_SIDE_STYLE = {"long": "green", "short": "red"}
_GRADE_STYLE = {"A": "bold green", "B": "yellow", "C": "dim"}
_VERDICT_STYLE = {VERDICT_EV: "bold green", VERDICT_INCONCLUSIVE: "yellow",
                  VERDICT_NEG: "red", VERDICT_FEW: "dim"}
_VERDICT_BORDER = {VERDICT_EV: "green", VERDICT_INCONCLUSIVE: "yellow",
                   VERDICT_NEG: "red", VERDICT_FEW: "grey50"}

console = Console()

DISCLAIMER = (
    "Technical analysis DESCRIBES market structure — it does NOT predict price. "
    "'NO TRADE' is always a valid choice. Read-only data; this tool places no "
    "orders. Not financial advice."
)

_REGIME_STYLE = {
    TRENDING: "bold green",
    RANGING: "yellow",
    CHOPPY: "dim red",
}
_TREND_GLYPH = {"up": "↑ up", "down": "↓ down", "flat": "→ flat"}
_VOL_ABBR = {SQUEEZE: "sqz", "normal": "nrm", EXPANDING: "exp"}
_FRESH_STYLE = {FRESH: "green", STRETCHED: "yellow", EXTENDED: "red"}
_POSTURE_STYLE = {mc.RISK_ON: "bold green", mc.RISK_OFF: "bold red", mc.NEUTRAL: "yellow"}


def _fmt_compact_usd(value: float) -> str:
    if value >= 1e9:
        return f"${value / 1e9:.1f}B"
    if value >= 1e6:
        return f"${value / 1e6:.0f}M"
    if value >= 1e3:
        return f"${value / 1e3:.0f}k"
    return f"${value:.0f}"


def _market_from_args(args: argparse.Namespace) -> Market:
    if args.spot and args.usdm:
        console.print("[red]Choose only one of --spot / --usdm.[/red]")
        raise SystemExit(2)
    if args.usdm:
        return Market.USDM
    return Market.SPOT  # default


def _load_tiered(args: argparse.Namespace):
    # PRECEDENCE: defaults → tier PRESET → env overrides on top (explicit env always wins over
    # the tier), so e.g. `--tier intraday` + BACKTEST_CANDLE_LIMIT=2000 keeps 1h AND honours 2000.
    base = apply_tier(Settings(), getattr(args, "tier", None))
    return load_settings(base=base)


def _settings_with_screen_overrides(args: argparse.Namespace):
    settings = _load_tiered(args)
    settings = with_overrides(
        settings,
        top_n=getattr(args, "top", None),
        quote=getattr(args, "quote", None),
        screen_tf=getattr(args, "tf", None),
        min_quote_volume=getattr(args, "min_volume", None),
    )
    equity = getattr(args, "equity", None)
    risk = getattr(args, "risk", None)
    if equity is not None or risk is not None:
        from dataclasses import replace
        settings = replace(
            settings,
            risk=replace(
                settings.risk,
                account_equity=equity if equity is not None else settings.risk.account_equity,
                risk_pct=risk if risk is not None else settings.risk.risk_pct,
            ),
        )
    return settings


def _render_context_banner(ctx: mc.MarketContext) -> Panel:
    body = Text()
    if not ctx.available:
        body.append(f"BTC proxy {ctx.proxy_symbol}: unavailable.\n", style="dim")
        body.append(ctx.note, style="dim")
        return Panel(body, title="Market context (the tide)", border_style="grey50", title_align="left")

    style = _POSTURE_STYLE.get(ctx.posture, "white")
    body.append(f"BTC ({ctx.proxy_symbol}): ", style="bold")
    body.append(f"{ctx.regime} {ctx.trend}", style=_REGIME_STYLE.get(ctx.regime, "white"))
    body.append("  →  posture ")
    body.append(ctx.posture.upper(), style=style)
    body.append(f"   (ADX {ctx.adx:.0f}, last {ctx.last_price:,.0f})\n")
    body.append(ctx.note, style="italic dim")
    return Panel(body, title="Market context (the tide) — read before any coin",
                 border_style=style, title_align="left")


def _print_gates(market: Market, cfg) -> None:
    body = Text()
    body.append(f"Market      : {market.label}  (quote {cfg.quote})\n")
    body.append(f"Liquidity   : top {cfg.top_n} by 24h vol, ≥ {_fmt_compact_usd(cfg.min_quote_volume)}, "
                f"spread ≤ {cfg.max_spread_bps:g}bps, depth ≥ {_fmt_compact_usd(cfg.min_depth_usd)} (±{cfg.depth_band_pct:g}%)\n")
    body.append(f"History     : ≥ {cfg.min_candles} closed {cfg.screen_tf} candles\n")
    body.append(f"Volatility  : ATR% in [{cfg.atr_pct_min:g}, {cfg.atr_pct_max:g}]; regime by ATR%ile "
                f"(≤{cfg.squeeze_pctl:g} squeeze, ≥{cfg.expand_pctl:g} expanding)\n")
    body.append(f"Regime      : ADX ≥ {cfg.adx_trending:g} trending · < {cfg.adx_choppy:g} choppy/avoid\n")
    body.append(f"Context     : relative strength + beta vs BTC over {cfg.rs_window} candles; "
                f"composite rank (trend+RS+vol+liq)\n")
    body.append(f"Correlation : flag pairs with |corr| ≥ {cfg.corr_threshold:g} over {cfg.corr_window} candles")
    console.print(Panel(body, title="Screener gates (tunable in config / env)", border_style="blue", title_align="left"))


def _rs_text(rel_strength: float) -> Text:
    if rel_strength != rel_strength:  # NaN
        return Text("—")
    style = "green" if rel_strength >= 0 else "red"
    return Text(f"{rel_strength:+.1f}%", style=style)


def _ext_text(extension: float, freshness: str) -> Text:
    """Directional extension in ATRs from the MA, coloured by freshness."""
    if extension != extension:  # NaN
        return Text("—")
    return Text(f"{extension:+.1f}σ", style=_FRESH_STYLE.get(freshness, ""))


def _render_table(result: ScreenResult) -> Table:
    table = Table(
        title=f"Ranked candidates — {result.market.label} / {result.quote} "
              f"(screen TF {result.screen_tf})",
        title_style="bold",
        header_style="bold",
        expand=True,
    )
    table.add_column("#", justify="right", no_wrap=True)
    table.add_column("Symbol", no_wrap=True)
    table.add_column("24h Vol", justify="right", no_wrap=True)
    table.add_column("Spread", justify="right", no_wrap=True)
    table.add_column("ATR%", justify="right", no_wrap=True)
    table.add_column("Vol", no_wrap=True)
    table.add_column("Ext", justify="right", no_wrap=True)
    table.add_column("ADX", justify="right", no_wrap=True)
    table.add_column("Regime", no_wrap=True)
    table.add_column("Trend", no_wrap=True)
    table.add_column("RS·BTC", justify="right", no_wrap=True)
    table.add_column("β", justify="right", no_wrap=True)
    table.add_column("Group", no_wrap=True)
    table.add_column("Why it passed", overflow="fold")

    for rank, c in enumerate(result.candidates, start=1):
        regime_style = _REGIME_STYLE.get(c.regime, "")
        group = f"[red]{c.group}*[/red]" if c.clustered else c.group
        beta = "—" if c.btc_beta != c.btc_beta else f"{c.btc_beta:.2f}"
        table.add_row(
            str(rank),
            c.symbol,
            _fmt_compact_usd(c.quote_volume),
            f"{c.spread_bps:.1f}bp",
            f"{c.atr_pct:.2f}",
            _VOL_ABBR.get(c.vol_regime, c.vol_regime),
            _ext_text(c.extension, c.freshness),
            f"{c.adx:.0f}",
            Text(c.regime, style=regime_style),
            _TREND_GLYPH.get(c.trend, c.trend),
            _rs_text(c.rel_strength),
            beta,
            group,
            c.why,
        )
    return table


def _render_rejections(result: ScreenResult) -> Panel:
    r = result.rejections
    body = Text()
    body.append(f"Eligible {result.market.label}/{result.quote} symbols : {result.eligible_universe}\n")
    body.append(f"Reached deep checks (top-N)        : {result.examined}\n")
    body.append(f"Passed all filters                 : {len(result.candidates)}\n")
    body.append("Rejected by stage:\n", style="bold")
    body.append(f"  • below volume / outside top-N : {r['below_volume_or_top_n']}\n")
    body.append(f"  • insufficient history         : {r['insufficient_history']}\n")
    body.append(f"  • ATR% outside band            : {r['atr_band']}\n")
    body.append(f"  • wide spread / thin book      : {r['spread_or_depth']}\n")
    body.append(f"  • stale feed / fetch error     : {r['stale_or_error']}")
    return Panel(body, title="How the universe narrowed", border_style="grey50", title_align="left")


def _illustrative_example_for(top: Candidate, market: Market, cfg_risk) -> Panel:
    """Worked dollar example for the top candidate from real ATR (teaching only)."""
    entry = top.last_price
    atr_abs = (top.atr_pct / 100.0) * entry
    stop_dist = cfg_risk.atr_stop_mult * atr_abs

    side = "long"
    if market is Market.USDM and top.trend == "down":
        side = "short"

    if side == "long":
        stop = entry - stop_dist
        take_profit = entry + cfg_risk.rr_target * stop_dist
    else:
        stop = entry + stop_dist
        take_profit = entry - cfg_risk.rr_target * stop_dist

    we = build_worked_example(
        account_equity=cfg_risk.account_equity,
        risk_pct=cfg_risk.risk_pct,
        entry=entry,
        stop=stop,
        take_profit=take_profit,
        side=side,
    )
    return render_worked_example(we, top.symbol, market.label, illustrative=True)


def _progress():
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    )


def _data_error_panel(exc: DataError) -> Panel:
    return Panel(
        Text(
            f"Could not reach Binance public market data.\n\n{exc}\n\n"
            "These commands need no API key — this is a network/region issue. "
            "Check connectivity and that api.binance.com is reachable.",
            style="red",
        ),
        title="Data error",
        border_style="red",
    )


def cmd_screen(args: argparse.Namespace) -> int:
    settings = _settings_with_screen_overrides(args)
    market = _market_from_args(args)
    cfg = settings.screener

    _print_gates(market, cfg)

    try:
        with _progress() as progress:
            task = progress.add_task("Screening Binance…", total=cfg.top_n)

            def cb(symbol: str, i: int, total: int) -> None:
                progress.update(task, completed=i, total=total, description=f"Analysing {symbol}")

            result = run_screen(market, cfg, settings.api_key, settings.api_secret, progress=cb)
    except DataError as exc:
        console.print(_data_error_panel(exc))
        return 1

    console.print(_render_context_banner(result.context))

    if not result.candidates:
        console.print(_render_rejections(result))
        console.print(Panel(
            Text("No candidates passed every filter. Loosen a gate (e.g. lower "
                 "--min-volume or widen ATR% band) or try the other market.",
                 style="yellow"),
            title="Empty shortlist",
            border_style="yellow",
        ))
        console.print(Panel(Text(DISCLAIMER, style="dim"), border_style="dim"))
        return 0

    console.print(_render_table(result))
    console.print(_render_rejections(result))
    console.print(_illustrative_example_for(result.candidates[0], market, settings.risk))
    console.print(Text(
        "→ You pick. Next (build-order step 7+): "
        f"analyze {result.candidates[0].symbol} --{market.value}",
        style="bold",
    ))
    console.print(Panel(Text(DISCLAIMER, style="dim"), border_style="dim"))
    return 0


def _render_leaderboard(rows, context, market: Market) -> None:
    if not rows:
        return
    table = Table(title=f"Relative strength vs BTC — {market.label} (top by volume)",
                  title_style="bold", header_style="bold")
    table.add_column("Symbol", no_wrap=True)
    table.add_column("RS·BTC", justify="right")
    table.add_column("β", justify="right")
    table.add_column("corr", justify="right")
    table.add_column("Trend", no_wrap=True)

    def add(sym, rs, trend):
        beta = "—" if rs.beta != rs.beta else f"{rs.beta:.2f}"
        corr = "—" if rs.corr != rs.corr else f"{rs.corr:.2f}"
        table.add_row(sym, _rs_text(rs.rel_strength_pct), beta, corr,
                      _TREND_GLYPH.get(trend, trend))

    leaders = rows[:5]
    laggards = rows[-5:] if len(rows) > 5 else []
    for sym, rs, trend in leaders:
        add(sym, rs, trend)
    if laggards:
        table.add_section()
        for sym, rs, trend in laggards:
            add(sym, rs, trend)
    console.print(table)
    console.print(Text("Top rows lead BTC (relative strength); bottom rows lag it. "
                       "In a risk-off tide, leaders hold up best; laggards fall fastest.",
                       style="dim"))


def cmd_context(args: argparse.Namespace) -> int:
    settings = _settings_with_screen_overrides(args)
    market = _market_from_args(args)
    cfg = settings.screener
    limit = args.top if args.top is not None else cfg.top_n

    try:
        with _progress() as progress:
            task = progress.add_task("Reading the tide…", total=limit)

            def cb(symbol: str, i: int, total: int) -> None:
                progress.update(task, completed=i, total=total, description=f"RS {symbol}")

            context, rows = relative_strength_scan(
                market, cfg, limit, settings.api_key, settings.api_secret, progress=cb
            )
    except DataError as exc:
        console.print(_data_error_panel(exc))
        return 1

    console.print(_render_context_banner(context))
    _render_leaderboard(rows, context, market)
    console.print(Panel(Text(DISCLAIMER, style="dim"), border_style="dim"))
    return 0


def _analysis_trade_panel(result, settings) -> Panel:
    plan = result.plan
    if plan is not None and result.setups:
        top = result.setups[0]
        body = Text()
        body.append(f"Setup: {top.setup} {top.direction} (grade {top.grade})\n", style="bold")
        for line in plan_worked_lines(plan):
            body.append("  • ", style="bold cyan")
            body.append(line + "\n")
        if result.market is Market.USDM and plan.liquidation_price:
            body.append(f"  USD-M: {plan.leverage:g}x · margin ${plan.margin:,.2f} · "
                        f"liquidation {plan.liquidation_price:,.4g}\n", style="dim")
        if plan.valid:
            body.append("\nIllustrative NET plan — NOT a signal. Prove edge (backtest) first.",
                        style="italic dim")
        else:
            body.append("\n✗ NOT WORTH TAKING:\n", style="bold yellow")
            for n in plan.notes:
                body.append(f"  − {n}\n", style="yellow")
        border = "green" if plan.valid else "yellow"
        mark = "✓" if plan.valid else "✗ NO TRADE"
        return Panel(body, title=f"Trade plan — {result.symbol} ({top.direction}) {mark}",
                     border_style=border, title_align="left")

    if result.bias == NEUTRAL or result.stop is None:
        return Panel(
            Text("NO clean directional read → stand aside (NO TRADE). "
                 "Absence of a setup is a valid, frequent outcome.", style="yellow"),
            title="Trade scaffold", border_style="yellow", title_align="left")
    entry, stop, side = result.entry, result.stop, result.side
    risk_dist = abs(entry - stop)
    if result.target and ((side == "long" and result.target > entry) or
                          (side == "short" and result.target < entry)):
        tp, tnote = result.target, "structure target"
    else:
        mult = settings.risk.rr_target
        tp = entry + mult * risk_dist if side == "long" else entry - mult * risk_dist
        tnote = "no structure target — default R:R"
    try:
        we = build_worked_example(account_equity=settings.risk.account_equity,
                                  risk_pct=settings.risk.risk_pct, entry=entry,
                                  stop=stop, take_profit=tp, side=side)
    except ValueError as exc:
        return Panel(Text(f"scaffold invalid: {exc}", style="red"), border_style="red")
    body = Text()
    for line in worked_example_lines(we, result.symbol):
        body.append("  • ", style="bold cyan")
        body.append(line + "\n")
    body.append(f"  (target: {tnote})\n", style="dim")
    sz = result.sizing
    if result.market is Market.USDM and sz is not None and sz.liquidation_price:
        body.append(f"  USD-M: {sz.leverage:g}x · margin ${sz.margin:,.2f} · "
                    f"liquidation {sz.liquidation_price:,.4g} "
                    f"(liq {sz.liq_distance_pct:.1f}% vs stop {sz.stop_distance_pct:.1f}%)\n", style="dim")
    body.append("\nIllustrative — structure-anchored stop, NOT a signal. "
                "Prove edge (backtest) before risking.", style="italic dim")
    return Panel(body, title=f"Worked dollar example — {result.symbol} ({side})",
                 border_style="green", title_align="left")


def _render_setups(result) -> None:
    if not result.setups:
        console.print(Panel(
            Text("No named setup present on this bar → NO TRADE (stand aside).", style="yellow"),
            title="Detected setups", border_style="yellow", title_align="left"))
        return
    table = Table(title="Detected setups (named, backtestable candidates)",
                  header_style="bold", expand=True)
    table.add_column("Setup", no_wrap=True)
    table.add_column("Dir", no_wrap=True)
    table.add_column("Grade", justify="center", no_wrap=True)
    table.add_column("Entry", justify="right", no_wrap=True)
    table.add_column("Stop", justify="right", no_wrap=True)
    table.add_column("TP1", justify="right", no_wrap=True)
    table.add_column("R:R", justify="right", no_wrap=True)
    table.add_column("Why", overflow="fold")
    for s in result.setups:
        table.add_row(
            s.setup,
            Text(s.direction, style=_SIDE_STYLE.get(s.direction, "")),
            Text(s.grade, style=_GRADE_STYLE.get(s.grade, "")),
            f"{s.entry:,.6g} ({s.entry_type})",
            f"{s.stop:,.6g}",
            f"{s.targets[0]:,.6g}",
            f"{s.rr:.2f}",
            "; ".join(s.reasons),
        )
    console.print(table)
    console.print(Text("Candidates, NOT signals — grade is provisional until edge is "
                       "backtested (steps 10–11).", style="dim"))


def _render_analysis(result, settings) -> None:
    head = Text()
    head.append(f"{result.symbol} ({result.market.label})  ", style="bold")
    head.append(f"{result.tf_trigger} read · {result.tf_bias} bias\n")
    head.append("Bias: ")
    head.append(result.bias.upper(), style=_BIAS_STYLE.get(result.bias, ""))
    head.append(f"   confluence {result.confluence*100:.0f}%  ({result.agree_count}/5 lenses)  "
                f"alignment={result.alignment}\n")
    head.append(f"HTF trend: {result.htf_trend}   Location: ", style="")
    head.append(result.location, style="bold")
    head.append(f"   Last {result.last_price:,.6g}\n")
    if result.invalidation:
        head.append(f"Invalidation: {result.invalidation_note}", style="red")
    else:
        head.append(result.invalidation_note, style="yellow")
    console.print(Panel(head, title="Analysis — a READ, not a prediction",
                        border_style=_BIAS_STYLE.get(result.bias, "blue"), title_align="left"))

    table = Table(title="Six lenses", header_style="bold", expand=True)
    table.add_column("Lens", no_wrap=True)
    table.add_column("Read", no_wrap=True)
    table.add_column("Score", justify="right", no_wrap=True)
    table.add_column("Observation", overflow="fold")
    for l in result.lenses:
        table.add_row(l.name, Text(l.direction or "—", style=_BIAS_STYLE.get(l.direction, "")),
                      f"{l.score:+.2f}", l.note)
    console.print(table)

    # the bear case
    cbody = Text()
    if result.conflicts:
        for c in result.conflicts:
            cbody.append("  ⚠ ", style="bold yellow")
            cbody.append(c + "\n")
    else:
        cbody.append("No major lens conflicts — but absence of conflict is NOT a signal.", style="dim")
    console.print(Panel(cbody, title="Conflict / the bear case", border_style="yellow", title_align="left"))

    _render_setups(result)

    if result.market is Market.USDM and (result.funding or result.oi):
        fb = Text()
        if result.funding:
            fb.append(f"Funding {result.funding.rate*100:.4f}%/8h ({result.funding.signal}); ")
        if result.oi:
            fb.append(f"OI {result.oi.change_pct:+.1f}% → {result.oi.trend}")
        console.print(Panel(fb, title="Futures (funding / OI)", border_style="grey50", title_align="left"))

    console.print(_analysis_trade_panel(result, settings))
    console.print(Text("→ This is analysis, not a signal. Next (step 8/10): "
                       "define & backtest the setup before risking anything.", style="bold"))
    console.print(Panel(Text(DISCLAIMER, style="dim"), border_style="dim"))


def cmd_analyze(args: argparse.Namespace) -> int:
    settings = _load_tiered(args)
    market = _market_from_args(args)
    if getattr(args, "run_all", False):
        return _analyze_board(args, settings, market)
    if not args.symbol:
        console.print(Text("Give a SYMBOL (e.g. `analyze BTC/USDT:USDT --usdm`) "
                           "or use `--all` to scan the whole shortlist.", style="yellow"))
        return 2
    try:
        with _progress() as progress:
            progress.add_task(f"Analysing {args.symbol}…", total=None)
            result = run_analyze(market, args.symbol, settings, tf_trigger=args.tf)
    except DataError as exc:
        console.print(_data_error_panel(exc))
        return 1
    _render_analysis(result, settings)
    if result.setups and result.structure and result.structure.state:
        top = result.setups[0]
        regime = result.structure.state.trend
        verdict = es.lookup_pooled_verdict(market, result.tf_trigger, top.setup, regime, symbol=result.symbol,
                                           context=es.resolve_context(result.context, top.direction))
        level, msg = es.edge_gate(verdict, settings.edge)
        style = _EDGE_STYLE.get(level, "yellow")
        panel = Text(f"{top.setup} in a {regime} regime — {msg}", style=style)
        rb = (verdict or {}).get("robustness")
        if rb:                                            # P6 — profitable vs consistently profitable
            panel.append(f"\nRobustness {rb['score']:.2f} — {rb['label']} "
                         f"(OOS folds {rb['cv']:.0%}+ · bootstrap {'✓' if rb['bootstrap']==1.0 else '~' if rb['bootstrap']==0.5 else '✗'} · "
                         f"walk-forward {rb['walk_forward']:.2f}).", style="dim")
        console.print(Panel(panel, title="Proven edge (pooled across the universe)",
                            border_style=style, title_align="left"))
        _render_archetype(result, top, settings, market)
        _render_explain(result, top, level, msg, regime, verdict, settings)
        _render_sizing_guidance(result, settings, market)
    return 0


def _render_sizing_guidance(result, settings, market: Market) -> None:
    """P12 — translate the statistical edge into position-sizing GUIDANCE (not automation):
    the evidence behind the size + which conservative stages adjusted it. Correlation and
    portfolio exposure are managed at the basket level (roar), surfaced here."""
    ri = result.risk_inputs or {}
    mgmt = settings.risk_mgmt
    base = ri.get("base_pct", settings.risk.risk_pct)
    eff = ri.get("effective_pct", result.risk_pct)
    body = Text()
    body.append(f"Base risk {base:.2f}% → effective {eff:.2f}%\n", style="bold")
    # the evidence (shown even when the auto-stages are off — it's discretionary guidance)
    ev = []
    if ri.get("edge_r") is not None:
        ev.append(f"edge {ri['edge_r']:+.2f}R")
    if ri.get("confidence") is not None:
        both = ""
        if ri.get("p_beats_null") is not None:
            both = (f" [P(beats null) {ri['p_beats_null']*100:.0f}% · "
                    f"P(>0) {ri.get('p_positive', 0)*100:.0f}%]")
        ev.append(f"confidence {ri['confidence']*100:.0f}%{both} (n{ri.get('regime_n', 0)}, "
                  f"sample-quality {ri.get('sample_quality', 0):.2f})")
    if ri.get("sigma_r"):
        ev.append(f"σ_R {ri['sigma_r']:.2f}")
    if ri.get("drawdown_pct"):
        ev.append(f"drawdown {ri['drawdown_pct']:.1f}%")
    if ev:
        body.append("evidence: " + " · ".join(ev) + "\n", style="dim")
    if result.risk_notes:
        body.append("adjustments: " + "; ".join(result.risk_notes) + "\n", style="cyan")
    else:
        on = [n for n, f in (("drawdown", mgmt.dd_scale_enabled), ("edge", mgmt.edge_scaled),
                             ("vol-target", mgmt.vol_target_enabled), ("Kelly", mgmt.kelly_enabled)) if f]
        body.append(("flat base — adaptive stages off (" + ", ".join(on) + " on)\n") if on
                    else "flat base — adaptive sizing stages off (set RISK_MGMT_* to size from edge/σ_R/drawdown)\n",
                    style="dim")
    body.append(f"correlation & portfolio exposure: managed at basket level (roar) — correlated coins "
                f"share a budget; total basket heat capped at {settings.guards.heat_cap_pct:.1f}%.",
                style="dim")
    console.print(Panel(body, title="Position-sizing guidance (discretionary, conservative)",
                        border_style="grey50", title_align="left"))


def _render_explain(result, top, level, msg, regime, verdict, settings) -> None:
    """P8 — the auditable WHY: factors pushing FOR vs AGAINST this recommendation."""
    reg = (verdict or {}).get("regime") if verdict else None
    tier = None
    if reg and reg.get("n"):
        sc = es.statistical_confidence(reg, (verdict or {}).get("null"),
                                       (verdict or {}).get("fold_consistency", 0.0), settings.edge)
        tier = es._conf_tier(sc["confidence"], settings.edge)
    ex = es.explain_signal(
        direction=top.direction, lenses=result.lenses, edge_level=level, edge_msg=msg, regime=regime,
        freshness=result.freshness, posture=result.posture,
        context_label=(reg.get("context_label") if reg else None), confidence_tier=tier)
    body = Text()
    body.append("Supports ▲\n", style="bold green")
    for s in ex["positive"]:
        body.append(f"  + {s}\n", style="green")
    if not ex["positive"]:
        body.append("  (nothing notable)\n", style="dim")
    body.append("Against ▼\n", style="bold red")
    for s in ex["negative"]:
        body.append(f"  − {s}\n", style="red")
    if not ex["negative"]:
        body.append("  (nothing notable)\n", style="dim")
    console.print(Panel(body, title="Why — evidence for & against (auditable)",
                        border_style="grey50", title_align="left"))


_ARCH_DESC = {
    "panic": "downtrend + volatility expansion + sharp down thrust",
    "euphoria": "uptrend + volatility expansion + sharp up thrust",
    "vol_expansion": "trending into a volatility expansion",
    "vol_compression": "coiling — volatility compressed (squeeze)",
    "accumulation": "ranging after a decline (basing)",
    "distribution": "ranging after an advance (topping)",
    "trending_up": "orderly uptrend, normal volatility",
    "trending_down": "orderly downtrend, normal volatility",
    "range": "neutral range, normal volatility",
}


def _render_archetype(result, top, settings, market: Market) -> None:
    """P3/P14: name the CURRENT market archetype and show this setup's historical conditional
    expectancy there (shrunk) — evidence, whether or not it has earned a gated promotion."""
    arch = (result.context or {}).get("arch")
    if not arch:
        return
    ev = es.archetype_evidence(market, result.tf_trigger, top.setup, arch)
    head = f"Archetype: {arch.upper()} — {_ARCH_DESC.get(arch, '')}"
    if not ev:
        body = Text(f"{head}\n  No pooled history for {top.setup} in this archetype yet "
                    "(evidence accrues as the cache rebuilds).", style="grey70")
    else:
        tag = "PROVEN edge here (drives sizing)" if ev["proven"] else "evidence only (not yet a gated edge)"
        sign = "green" if ev["exp"] > 0 else "red"
        body = Text.assemble(
            (head + "\n", "bold"),
            (f"  {top.setup} historically ", ""),
            (f"{ev['exp']:+.2f}R", sign),
            (f" in {arch} (shrunk; raw {ev['raw_exp']:+.2f}R over {ev['n']} pooled trades) — {tag}.", ""),
        )
    console.print(Panel(body, title="Conditional expectancy (this environment)",
                        border_style="grey50", title_align="left"))


def _analyze_many(market: Market, symbols, settings, tf):
    """Run the six-lens analyze across many coins concurrently (per-thread exchange)."""
    import threading
    from concurrent.futures import ThreadPoolExecutor

    local = threading.local()

    def thread_ex():
        ex = getattr(local, "ex", None)
        if ex is None:
            ex = make_exchange(market, settings.api_key, settings.api_secret)
            load_markets(ex)
            local.ex = ex
        return ex

    def work(sym):
        try:
            return sym, run_analyze(market, sym, settings, tf_trigger=tf, ex=thread_ex()), None
        except Exception as exc:  # noqa: BLE001 — one bad coin must not sink the board
            return sym, None, str(exc)

    results, failed = [], []
    workers = max(1, settings.screener.fetch_workers)
    with _progress() as progress:
        task = progress.add_task(f"Analysing {len(symbols)} coins…", total=len(symbols))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for sym, res, err in pool.map(work, symbols):
                progress.advance(task)
                (results if res is not None else failed).append(res if res is not None else (sym, err))
    return results, failed


def _short_flag(conflicts) -> str:
    if not conflicts:
        return ""
    first = conflicts[0].split(" — ")[0].split(" (")[0]
    return (first[:34] + "…") if len(first) > 35 else first


_LOC_ABBR = {"at_support": "support", "at_resistance": "resist", "mid_range": "mid"}


def _analyze_board(args: argparse.Namespace, settings, market: Market) -> int:
    if args.top:
        from dataclasses import replace
        settings = replace(settings, screener=replace(settings.screener, top_n=args.top))
    try:
        with _progress() as progress:
            progress.add_task("Screening the universe…", total=None)
            screen = run_screen(market, settings.screener, settings.api_key, settings.api_secret)
    except DataError as exc:
        console.print(_data_error_panel(exc))
        return 1
    symbols = [c.symbol for c in screen.candidates]
    if not symbols:
        console.print(Panel(Text("Screener returned no candidates to analyze.", style="yellow"),
                            title="Analyze board", border_style="yellow", title_align="left"))
        return 0

    results, failed = _analyze_many(market, symbols, settings, args.tf)
    rows = rank_board(results, setups_only=getattr(args, "setups_only", False))
    n_setups = sum(1 for r in results if r.setups)
    tf = args.tf or settings.analysis.tf_trigger

    posture = results[0].posture if results else mc.NEUTRAL
    console.print(Text(f"Tide (BTC posture, measured during the scan): {posture.upper()}",
                       style=_POSTURE_STYLE.get(posture, "dim")))
    table = Table(title=f"Analyze board — {market.label} · {tf} read · {len(results)} coins, {n_setups} with a setup",
                  title_style="bold", header_style="bold", expand=True)
    for col, just in (("#", "right"), ("Symbol", "left"), ("Bias", "left"), ("Conf", "right"),
                      ("Loc", "left"), ("HTF", "left"), ("Setup", "left"), ("net R:R", "right"),
                      ("Flag", "left")):
        table.add_column(col, justify=just, no_wrap=(col != "Flag"), overflow="fold")
    for i, r in enumerate(rows, 1):
        has_plan = r.plan is not None and getattr(r.plan, "valid", False)
        if r.setups:
            top = r.setups[0]
            setup_txt = Text(f"{top.setup} {top.direction[:1].upper()} ({top.grade})",
                             style=_SIDE_STYLE.get(top.direction, ""))
            rr = f"{r.plan.net_rr:.2f}" if has_plan else "—"
        else:
            setup_txt, rr = Text("— no setup", style="dim"), "—"
        table.add_row(
            str(i), r.symbol,
            Text(r.bias, style=_BIAS_STYLE.get(r.bias, "")), f"{r.confluence * 100:.0f}%",
            _LOC_ABBR.get(r.location, r.location), r.alignment, setup_txt, rr,
            Text(_short_flag(r.conflicts), style="yellow"))
    console.print(table)
    foot = "→ deep-dive any row: " + f"analyze SYMBOL {'--usdm' if market is Market.USDM else '--spot'} --tf {tf}"
    if failed:
        foot += f"   ({len(failed)} skipped: data error)"
    console.print(Text(foot, style="dim"))
    console.print(Panel(Text(DISCLAIMER, style="dim"), border_style="dim"))
    return 0


def _pf(value: float) -> str:
    return "∞" if value == float("inf") else f"{value:.2f}"


def _render_backtest(symbol, market, tf, profiles, settings, full: bool = False,
                     n_candles: int | None = None, quality: dict | None = None) -> None:
    bt = settings.backtest
    # HONEST candle count: the ACTUAL closed candles used, not the requested limit. If they differ
    # (young symbol, or the ±1 forming candle), show both so the number is never misleading.
    if n_candles is None:
        candles_txt = f"{bt.candle_limit} candles requested"
    elif n_candles == bt.candle_limit:
        candles_txt = f"{n_candles} candles"
    else:
        candles_txt = f"{n_candles} candles (of {bt.candle_limit} requested)"
    head = Text(f"{symbol} ({market.label})  TF {tf} · {candles_txt} · "
                f"min sample {bt.min_sample} · {bt.folds} folds", style="bold")
    q = quality or {}
    issues = {k: q.get(k, 0) for k in ("dropped", "gaps", "zero_volume", "suspect_spikes") if q.get(k, 0)}
    if issues:                                    # audit finding 6: never hide a dirty tape
        head.append("\ndata quality: " + " · ".join(f"{k.replace('_', ' ')} {v}" for k, v in issues.items())
                    + ("  (broken rows DROPPED; anomalies flagged, never 'fixed')"), style="yellow")
    console.print(Panel(
        head,
        title="Backtest — proving edge on real history (same code as live)",
        border_style="blue", title_align="left"))

    if not profiles or all(p.overall.n == 0 for p in profiles.values()):
        console.print(Panel(Text("No setups triggered on this history (try a longer TF or more candles).",
                                 style="yellow"), border_style="yellow"))
        console.print(Panel(Text(DISCLAIMER, style="dim"), border_style="dim"))
        return

    table = Table(header_style="bold", expand=True)
    for col, just in (("Setup", "left"), ("Verdict", "left"), ("N", "right"), ("Win%", "right"),
                      ("Expectancy [CI]", "right"), ("vs Null [low]", "right"), ("PF", "right"),
                      ("SQ", "right"), ("MaxDD", "right"), ("Streak", "right"), ("Folds+", "right"),
                      ("Ruin", "right")):
        table.add_column(col, justify=just, no_wrap=True)
    for name, p in profiles.items():
        o = p.overall
        mc = p.monte_carlo
        vs_null = (f"{p.edge_vs_null:+.2f} [{p.edge_vs_null_ci_low:+.2f}]" if p.null_n else "—")
        table.add_row(
            name, Text(p.verdict, style=_VERDICT_STYLE.get(p.verdict, "")), str(o.n),
            f"{o.win_rate*100:.0f}%",
            f"{o.expectancy:+.2f} [{o.ci_low:+.2f},{o.ci_high:+.2f}]", vs_null,
            _pf(o.profit_factor), f"{o.system_quality:.2f}",
            f"{o.max_drawdown_r:.1f}R", str(o.longest_loss_streak),
            f"{p.fold_consistency:.0%}",
            f"{mc.risk_of_ruin*100:.1f}%" if mc else "—",
        )
    console.print(table)

    for name, p in profiles.items():
        if p.overall.n == 0:
            continue
        o = p.overall
        body = Text()
        body.append("verdict: ")
        body.append(p.verdict, style=_VERDICT_STYLE.get(p.verdict, ""))
        body.append(" — " + "; ".join(p.notes) + "\n")
        if p.null_n:
            body.append(f"null baseline (random entries, same exit+geometry): {p.null_expectancy:+.2f}R "
                        f"over {p.null_n} shadows → real edge over null {p.edge_vs_null:+.2f}R "
                        f"(lower bound {p.edge_vs_null_ci_low:+.2f}R)\n", style="dim")
        # P10 — separate the raw statistical edge from execution costs.
        body.append(f"edge vs execution: gross {p.gross_expectancy:+.3f}R − fees/slippage {p.exec_cost_r:.3f}R")
        if market is Market.USDM:
            body.append(f" − funding {p.funding_cost_r:.3f}R")
        body.append(f" = NET {o.expectancy:+.3f}R\n", style="cyan")
        # P9 — distribution shape (what the average hides).
        body.append(f"distribution: median {o.median_r:+.2f}R · avg win {o.avg_win_r:+.2f} / "
                    f"avg loss {o.avg_loss_r:+.2f} · σ {o.std_r:.2f} · skew {o.skew_r:+.2f} · "
                    f"p10 {o.p10_r:+.2f} / p90 {o.p90_r:+.2f}\n", style="dim")
        # audit finding 2 — cross-coin correlation: how much INDEPENDENT evidence is really here
        if getattr(p, "deff", 1.0) > 1.0:
            body.append(f"correlation: ρ {p.corr_rho:.2f} within same-day clusters → design effect "
                        f"{p.deff:.2f} → effective n ≈ {o.n_eff:.0f} of {o.n} (SE widened ×{p.deff**0.5:.2f})\n",
                        style="dim")
        body.append("by regime: ")
        for reg, s in p.by_regime.items():
            body.append(f"{reg} {s.expectancy:+.2f}R (n{s.n})   ")
        # P11 — risk intelligence: the drawdown DISTRIBUTION, not just the historical max.
        if p.monte_carlo:
            m = p.monte_carlo
            recov = (m.p95_max_drawdown_r / o.expectancy) if o.expectancy > 0 else None
            recov_s = f"~{recov:.0f} trades" if recov is not None else "n/a (−EV)"
            body.append(f"\nrisk (Monte-Carlo {m.runs}×): median DD {m.median_max_drawdown_r:.1f}R · "
                        f"95th-pctile DD {m.p95_max_drawdown_r:.1f}R · risk-of-ruin(≥{bt.ruin_drawdown_r:.0f}R) "
                        f"{m.risk_of_ruin*100:.1f}% · expected losing streak {m.expected_loss_streak:.1f} · "
                        f"recovery {recov_s}\n", style="dim")
            body.append(f"projection: median total {m.median_total_r:+.1f}R · 5th-pctile {m.p5_total_r:+.1f}R")
        # P6 — VALIDATION DEPTH: profitable vs CONSISTENTLY profitable (report-only composite).
        rb = es.robustness(o.expectancy, p.fold_consistency, p.bootstrap_low, p.bootstrap_high,
                           p.recent_expectancy)
        boot_s = ">0" if rb["bootstrap"] == 1.0 else "<0" if rb["bootstrap"] == 0.0 else "straddles 0"
        rb_style = "green" if rb["score"] >= 0.6 else "yellow" if rb["score"] >= 0.3 else "red"
        body.append(f"\nrobustness {rb['score']:.2f} — {rb['label']}: OOS folds {p.fold_consistency:.0%}+ · "
                    f"bootstrap CI {boot_s} · walk-forward {rb['walk_forward']:.2f} "
                    f"(recent {p.recent_expectancy:+.2f}R vs overall {o.expectancy:+.2f}R)", style=rb_style)
        if full:
            body.append(f"\n— full — bootstrap 95% CI [{p.bootstrap_low:+.3f}, {p.bootstrap_high:+.3f}]R · "
                        f"recent (last {p.recent_n}) {p.recent_expectancy:+.3f}R vs overall {o.expectancy:+.3f}R · "
                        "folds " + " / ".join(f"{f:+.2f}" for f in p.fold_expectancies), style="cyan")
        console.print(Panel(body, title=name, border_style=_VERDICT_BORDER.get(p.verdict, "grey50"),
                            title_align="left"))

    console.print(Panel(Text(
        f"Caveats: Binance shows SURVIVORS only (delistings absent → results flattered); "
        f"universe = TODAY'S most-liquid names (point-in-time selection bias — read as "
        f"'edge on currently-liquid survivors'); "
        f"{len(profiles)} setups judged (multiple-testing — a lone +EV can be luck); "
        f"pooled coins are CROSS-CORRELATED — SEs/gates use the measured effective sample "
        f"(design effect), not raw pooled n; Monte-Carlo uses block bootstrap (streaks survive); "
        f"costs modeled (maker/taker + slippage{', funding' if market is Market.USDM else ''}). "
        "Historical edge with stated confidence — NOT a prediction.", style="dim"),
        border_style="dim"))
    console.print(Panel(Text(DISCLAIMER, style="dim"), border_style="dim"))


_EDGE_GRADE_STYLE = {"A": "bold green", "B": "green", "C": "yellow", "D": "dim", "F": "dim red"}


def _render_board(market, context, opps, watch, settings) -> None:
    console.print(_render_context_banner(context))
    if not opps:
        body = Text("NO PROVEN EDGE on the board right now → stand aside.\n", style="bold yellow")
        body.append("A valid, frequent result — the tool will not manufacture an opportunity.\n",
                    style="dim")
        if watch:
            body.append(f"Watchlist (live setup, but no proven regime-matched edge): "
                        f"{', '.join(watch[:12])}", style="dim")
        console.print(Panel(body, title="Opportunity Board", border_style="yellow", title_align="left"))
        console.print(Panel(Text(DISCLAIMER, style="dim"), border_style="dim"))
        return

    table = Table(title="Opportunity Board — ranked by conservative Edge Score",
                  header_style="bold", expand=True)
    for col, just in (("#", "right"), ("Symbol", "left"), ("Group", "left"), ("Side", "left"),
                      ("Setup", "left"), ("Edge", "right"), ("Grade", "center"),
                      ("Confidence", "left"), ("Fresh", "left"), ("Regime", "left")):
        table.add_column(col, justify=just, no_wrap=True)
    for rank, o in enumerate(opps, start=1):
        setup_cell = o.setup if not o.context_label else f"{o.setup} ·{o.context_label}"
        table.add_row(
            str(rank), o.symbol, o.group, Text(o.side, style=_SIDE_STYLE.get(o.side, "")), setup_cell,
            f"{o.edge_score_r:+.2f}R", Text(o.grade, style=_EDGE_GRADE_STYLE.get(o.grade, "")),
            f"{o.confidence_tier} ({o.confidence:.0%}, n{o.regime_n})", o.fresh, o.regime)
    console.print(table)
    console.print(Text("Confidence = P(edge genuinely beats random entries) × fold-consistency — "
                       "'how certain we are the edge exists,' not how many signals agree.", style="dim"))
    if any(o.context_label for o in opps):
        console.print(Text("·label = scored on a PROVEN context refinement (single feature, or an 'A & B' "
                           "feature INTERACTION that beats either alone); shrunk toward the base setup×regime.",
                           style="dim"))
    console.print(Text("#1 is the MOST likely fluke (board-level multiple testing) — confidence is "
                       "shown for a reason; verify with `analyze` and don't chase blindly.", style="dim"))
    console.print(Text("Pooled coins are cross-correlated: gates/confidence use the measured EFFECTIVE "
                       "sample size (design effect), not raw pooled n. Universe = today's most-liquid "
                       "survivors (selection bias) — read as 'edge on currently-liquid names.'", style="dim"))
    if watch:
        console.print(Text(f"Watchlist (no proven edge yet): {', '.join(watch[:12])}", style="dim"))
    console.print(Panel(Text(DISCLAIMER, style="dim"), border_style="dim"))


def cmd_scan(args: argparse.Namespace) -> int:
    settings = _load_tiered(args)
    market = _market_from_args(args)
    try:
        with _progress() as progress:
            task = progress.add_task("Scanning…", total=None)

            def cb(symbol, i, total):
                progress.update(task, description=f"Edge-scoring {symbol} ({i}/{total})")

            context, opps, watch = run_scan(market, settings, top=args.top,
                                            refresh=args.refresh, progress=cb,
                                            hard=getattr(args, "hard", False))
    except DataError as exc:
        console.print(_data_error_panel(exc))
        return 1
    _render_board(market, context, opps, watch, settings)
    return 0


_TREND_STYLE = {"rising": "green", "decaying": "red", "flat": "dim", "new": "cyan"}
_TREND_MARK = {"rising": "▲ rising", "decaying": "▼ decaying", "flat": "· flat", "new": "＊ new"}


def cmd_features(args: argparse.Namespace) -> int:
    """P13 — Continuous Feature Evaluation: which context features carry predictive value now,
    and how it's shifting (rising/decaying vs the previous rebuild). Rebuilt by `scan --refresh`."""
    settings = _load_tiered(args)
    market = _market_from_args(args)
    tf = args.tf or settings.edge.tf
    fi = es.load_feature_importance(market, tf)
    if not fi:
        console.print(Panel(Text("No feature-importance snapshot yet — run `scan --refresh` "
                                 "to build/refresh the universe edge cache first.", style="yellow"),
                            border_style="yellow"))
        return 1
    rows = sorted(fi.items(), key=lambda kv: -kv[1]["importance"])
    table = Table(title=f"Continuous feature evaluation — {market.label} {tf} (predictive value of each context feature)",
                  header_style="bold", expand=True)
    for col, just in (("Feature", "left"), ("Importance", "right"), ("# significant", "right"),
                      ("mean lift", "right"), ("best lift_low", "right"), ("proven", "right"), ("Trend", "left")):
        table.add_column(col, justify=just, no_wrap=True)
    active = {f for f, _ in rows[: settings.edge.feature_top_k]}
    for f, d in rows:
        star = " ●" if f in active else ""
        table.add_row(
            f + star, f"{d['importance']:.2f}", str(d["n_significant"]), f"{d['mean_lift']:+.3f}R",
            f"{d['best_lift_low']:+.3f}R", str(d.get("n_proven", 0)),
            Text(_TREND_MARK.get(d.get("trend", "flat"), d.get("trend", "")),
                 style=_TREND_STYLE.get(d.get("trend", "flat"), "")))
    console.print(table)
    console.print(Text(f"● = one of the top-{settings.edge.feature_top_k} features that may seed P4 interaction "
                       "pairs (important features carry greater influence; decayed ones decline).", style="dim"))
    console.print(Text("Importance = # raw-significant positive directions + capped mean lift; trend vs the "
                       "previous `scan --refresh`. Recompute by rebuilding the cache.", style="dim"))
    console.print(Panel(Text(DISCLAIMER, style="dim"), border_style="dim"))
    return 0


def _notify(title: str, message: str, enabled: bool = True) -> None:
    """Desktop notification (macOS osascript) + terminal bell. Never raises."""
    console.bell()
    if not enabled:
        return
    try:
        subprocess.run(["osascript", "-e",
                        f"display notification {json.dumps(message)} with title {json.dumps(title)}"],
                       capture_output=True, timeout=5)
    except Exception:  # noqa: BLE001 — notifications are best-effort
        pass


def _watch_paths():
    return DATA_DIR / "watch.pid", DATA_DIR / "watch.log", DATA_DIR / "watch_status.json"


def _pid_alive(pid) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, ValueError, TypeError):
        return False
    except PermissionError:
        return True


def _read_pid(pidf) -> int | None:
    try:
        return int(pidf.read_text().strip())
    except (ValueError, OSError):
        return None


def _watch_running() -> int | None:
    """The single source of truth: a registered pidfile whose process is alive."""
    pidf, _, _ = _watch_paths()
    pid = _read_pid(pidf) if pidf.exists() else None
    return pid if (pid is not None and _pid_alive(pid)) else None


def _watch_start(args, market: Market) -> int:
    running = _watch_running()
    if running is not None:
        console.print(Text(f"watch already running (pid {running}). `watch --stop` first.", style="yellow"))
        return 0
    pidf, logf, statusf = _watch_paths()
    statusf.unlink(missing_ok=True)                         # clear any stale status before a fresh run
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "src.cli", "watch", "--_run",
           "--usdm" if market is Market.USDM else "--spot", "--interval", str(args.interval)]
    if getattr(args, "tier", None):
        cmd += ["--tier", args.tier]
    if getattr(args, "top", None):
        cmd += ["--top", str(args.top)]
    log = open(logf, "a")
    proc = subprocess.Popen(cmd, stdout=log, stderr=log, start_new_session=True, cwd=str(PROJECT_ROOT))
    pidf.write_text(str(proc.pid))
    console.print(Panel(Text(f"watch started in the background (pid {proc.pid}).\n"
                             f"Log: {logf}\nStatus: `watch --status`   ·   Stop: `watch --stop`",
                             style="green"), title="watch — running", border_style="green", title_align="left"))
    return 0


def _watch_stop() -> int:
    pidf, _, statusf = _watch_paths()
    pid = _read_pid(pidf) if pidf.exists() else None
    if pid is None or not _pid_alive(pid):
        pidf.unlink(missing_ok=True)
        statusf.unlink(missing_ok=True)
        console.print(Text("watch is not running.", style="yellow"))
        return 0
    for sig in (signal.SIGTERM, signal.SIGCONT):           # graceful; SIGCONT resumes a Ctrl-Z'd process
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            break
    deadline = time.time() + 3.0                           # the loop may be mid-scan — give it a moment, then force
    while time.time() < deadline and _pid_alive(pid):
        time.sleep(0.2)
    forced = False
    if _pid_alive(pid):
        forced = True
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            forced = False
    pidf.unlink(missing_ok=True)
    statusf.unlink(missing_ok=True)
    console.print(Text(f"watch stopped (pid {pid}){' — force-killed mid-scan' if forced else ''}.", style="yellow"))
    return 0


def _watch_status_show() -> int:
    pidf, logf, statusf = _watch_paths()
    pid = _watch_running()
    if pid is None:                                        # not running -> clean stale state, say so plainly
        pidf.unlink(missing_ok=True)
        statusf.unlink(missing_ok=True)
        console.print(Text("watch is not running.", style="yellow"))
        return 0
    if not statusf.exists():
        console.print(Text(f"watch RUNNING (pid {pid}) — first scan in progress…", style="cyan"))
        return 0
    try:
        s = json.loads(statusf.read_text())
    except (ValueError, OSError):
        console.print(Text(f"watch RUNNING (pid {pid}) — status pending…", style="cyan"))
        return 0
    body = Text()
    body.append(f"RUNNING (pid {pid})\n", style="green")
    body.append(f"last pass: {s.get('ts','?')} UTC (#{s.get('passes','?')})\n")
    body.append(f"monitoring {s.get('monitoring',0)} coins · {s.get('proven',0)} PROVEN on board\n")
    body.append(f"board: {', '.join(s.get('board') or []) or '—'}\n", style="green")
    body.append(f"watchlist: {', '.join(s.get('watchlist') or []) or '—'}\n", style="dim")
    body.append(f"log: {logf}", style="dim")
    console.print(Panel(body, title="watch — status", border_style="cyan", title_align="left"))
    return 0


def _tf_seconds(tf: str) -> float:
    tf = (tf or "4h").strip().lower()
    try:
        val = float(tf[:-1])
    except ValueError:
        return 14400.0
    return {"m": 60.0, "h": 3600.0, "d": 86400.0, "w": 604800.0}.get(tf[-1], 3600.0) * val


def _next_bar_close_ts(tf_s: float) -> float:
    """The next candle boundary (Binance sub-day candles are UTC-epoch-aligned)."""
    return (int(time.time() // tf_s) + 1) * tf_s


def _has_price_watch(market: Market, settings) -> bool:
    """Anything PRICE-driven to check between bar closes — an active alert, or an
    open/staged position (a limit waiting to fill / a stop in play)?"""
    try:
        if any(a.market == market.value for a in list_alerts(active_only=True)):
            return True
    except Exception:  # noqa: BLE001
        pass
    try:
        return any(r.market == market.value and r.status in (OPEN, STAGED) for r in load_records())
    except Exception:  # noqa: BLE001
        return False


def _sleep_until(target_ts: float, stop: dict) -> None:
    while not stop["flag"] and time.time() < target_ts:
        time.sleep(min(2.0, max(0.1, target_ts - time.time())))


def _watch_loop(args, settings, market: Market, spawned: bool = False) -> int:
    pidf, _, statusf = _watch_paths()
    if not spawned:                                        # foreground: register ourselves (so --status/--stop work)
        running = _watch_running()
        if running is not None:
            console.print(Text(f"watch already running (pid {running}). `watch --stop` first.", style="yellow"))
            return 0
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        statusf.unlink(missing_ok=True)
        pidf.write_text(str(os.getpid()))

    wcfg = settings.watch
    tf = settings.edge.tf
    tf_s = _tf_seconds(tf)
    price_poll = max(60.0, args.interval or wcfg.poll_seconds)   # light between-bar check cadence (≥60s)
    cooldown = wcfg.cooldown_minutes * 60.0
    stop = {"flag": False}

    def _handler(*_):
        stop["flag"] = True

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _handler)
        except Exception:  # noqa: BLE001
            pass

    if not spawned:
        console.print(Panel(Text(
            f"watch running in the FOREGROUND (pid {os.getpid()}) · {market.label}.\n"
            f"The board only changes when a {tf} candle closes, so it's re-scanned right AFTER each {tf} "
            "close; between closes it does only light alert checks (no pointless re-scans). "
            "Press Ctrl+C to stop.  (Background radar: `watch --start`.)", style="cyan"),
            title="watch", border_style="cyan", title_align="left"))

    prev: set[str] = set()
    notified: dict = {}
    last_digest = 0.0
    board_passes = 0
    next_board = 0.0                                             # 0 → scan immediately on the first iteration
    last_board = {"mon": 0, "proven": 0, "watch": []}
    try:
        while not stop["flag"]:
            if time.time() >= next_board:
                # ---- BOARD SCAN: a trigger-TF candle just closed (or first run) ----
                board_passes += 1
                try:
                    with _progress() as progress:
                        task = progress.add_task("Scanning the board…", total=None)

                        def _cb(sym, i, total):
                            progress.update(task, description=f"Edge-scoring {sym} ({i}/{total})")

                        context, opps, watch = run_scan(market, settings, top=args.top, progress=_cb)
                except DataError as exc:
                    console.print(_data_error_panel(exc))
                    next_board = time.time() + price_poll       # retry shortly on a fetch error
                    _sleep_until(next_board, stop)
                    continue
                now = time.time()
                ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
                new, dropped = diff_boards(prev, opps)
                console.rule(f"watch · board pass {board_passes} · {ts} UTC · {len(opps)} proven / {len(watch)} watched")

                for o in opps:                                  # INSTANT: a NEW proven edge (per-coin cooldown)
                    if o.symbol in new and (now - notified.get(o.symbol, 0.0)) > cooldown:
                        _notify(f"ROAR: proven edge — {o.symbol}",
                                f"{o.setup} {o.side} ({o.grade}); edge {o.edge_score_r:+.2f}R — run `roar`", wcfg.notify)
                        notified[o.symbol] = now
                        console.print(Text(f"🆕 NEW PROVEN: {o.symbol} — {o.setup} {o.side} {o.edge_score_r:+.2f}R "
                                           "→ auto-analysing…", style="bold green"))
                        try:
                            _render_analysis(run_analyze(market, o.symbol, settings, tf_trigger=tf), settings)
                        except DataError:
                            pass
                if dropped:
                    console.print(Text(f"⬇ dropped from board: {', '.join(dropped)}", style="dim"))

                if now - last_digest >= wcfg.digest_seconds or board_passes == 1:
                    _notify("watch — status",
                            f"{len({o.symbol for o in opps} | set(watch))} coins · {len(watch)} live setups · "
                            f"{len(opps)} PROVEN", wcfg.notify)
                    _render_board(market, context, opps, watch, settings)
                    last_digest = now

                last_board = {"mon": len({o.symbol for o in opps} | set(watch)),
                              "proven": len(opps), "watch": list(watch[:12])}
                try:
                    statusf.write_text(json.dumps({
                        "ts": ts, "passes": board_passes, "monitoring": last_board["mon"], "proven": len(opps),
                        "board": [o.symbol for o in opps], "watchlist": last_board["watch"]}))
                except OSError:
                    pass
                prev = {o.symbol for o in opps}
                next_board = _next_bar_close_ts(tf_s) + wcfg.bar_settle_seconds
                nxt = datetime.fromtimestamp(next_board, tz=timezone.utc).strftime("%H:%M")
                console.print(Text(f"board fresh until the next {tf} close → next scan {nxt} UTC "
                                   f"(~{max(0, (next_board - time.time()) / 60):.0f} min); light alert checks meanwhile.",
                                   style="dim"))
                if args.iterations is not None and board_passes >= args.iterations:
                    break
                continue

            # ---- BETWEEN BARS: light, price-driven only (the board cannot change here) ----
            pending = _has_price_watch(market, settings)
            if pending:
                try:
                    _ping_fired(check_alerts(market, settings))
                except DataError:
                    pass
            if time.time() - last_digest >= wcfg.digest_seconds:
                _notify("watch — status",
                        f"{last_board['mon']} coins · {last_board['proven']} PROVEN (last board)", wcfg.notify)
                console.print(Text(f"· status {datetime.now(timezone.utc).strftime('%H:%M')} UTC: "
                                   f"{last_board['mon']} coins · {last_board['proven']} proven · "
                                   f"watchlist {', '.join(last_board['watch']) or '—'}", style="cyan"))
                last_digest = time.time()
            now = time.time()
            target = min(next_board, last_digest + wcfg.digest_seconds)
            if pending:
                target = min(target, now + price_poll)
            _sleep_until(max(now + 1.0, target), stop)
    finally:
        if pidf.exists() and _read_pid(pidf) == os.getpid():    # clean up only OUR own pidfile/status
            pidf.unlink(missing_ok=True)
            statusf.unlink(missing_ok=True)

    console.print(Text(f"watch stopped cleanly after {board_passes} board scan(s).", style="yellow"))
    return 0


def cmd_watch(args: argparse.Namespace) -> int:
    if getattr(args, "watch_stop", False):
        return _watch_stop()
    if getattr(args, "watch_status", False):
        return _watch_status_show()
    market = _market_from_args(args)
    if getattr(args, "watch_start", False):
        return _watch_start(args, market)
    return _watch_loop(args, _load_tiered(args), market, spawned=getattr(args, "watch_run", False))


def cmd_backtest(args: argparse.Namespace) -> int:
    settings = _load_tiered(args)
    market = _market_from_args(args)
    tf = args.tf or settings.analysis.tf_trigger
    try:
        with _progress() as progress:
            progress.add_task(f"Backtesting {args.symbol} {tf}…", total=None)
            profiles, n_candles, quality = run_backtest(market, args.symbol, settings, tf, setup_name=args.setup)
    except DataError as exc:
        console.print(_data_error_panel(exc))
        return 1
    _render_backtest(args.symbol, market, tf, profiles, settings, full=getattr(args, "full", False),
                     n_candles=n_candles, quality=quality)
    return 0


def _pf2(value: float) -> str:
    return "∞" if value == float("inf") else f"{value:.2f}"


def _render_review(rep, settings) -> None:
    if rep.n_closed == 0:
        console.print(Panel(
            Text("No closed trades yet — the journal fills as you stage and close trades.",
                 style="yellow"), title="Review", border_style="yellow", title_align="left"))
        return
    o = rep.overall
    head = Text()
    head.append(f"Equity {rep.starting_equity:,.0f} → {rep.equity:,.0f}  "
                f"({(rep.equity/rep.starting_equity-1)*100:+.1f}%)   over {rep.n_closed} closed trades\n",
                style="bold")
    head.append(f"Expectancy {o.expectancy:+.2f}R [{o.ci_low:+.2f},{o.ci_high:+.2f}]  ·  "
                f"win {o.win_rate*100:.0f}%  ·  PF {_pf2(o.profit_factor)}  ·  "
                f"max DD {rep.max_drawdown_r:.1f}R  ·  longest losing streak {o.longest_loss_streak}")
    console.print(Panel(head, title="Review — realised performance",
                        border_style="blue", title_align="left"))

    table = Table(header_style="bold", expand=True)
    for col in ("Bucket", "Key", "N", "Expectancy", "Win%", "PF"):
        table.add_column(col, justify="left" if col in ("Bucket", "Key") else "right", no_wrap=True)
    for label, bucket in (("setup", rep.by_setup), ("regime", rep.by_regime), ("grade", rep.by_grade)):
        for key, s in sorted(bucket.items(), key=lambda kv: kv[1].expectancy, reverse=True):
            table.add_row(label, str(key), str(s.n), f"{s.expectancy:+.2f}R",
                          f"{s.win_rate*100:.0f}%", _pf2(s.profit_factor))
    console.print(table)

    a = rep.adherence
    ab = Text()
    ab.append(f"Adherence {a.adherence_pct:.0f}%  ({a.followed}/{a.n} clean)\n",
              style="bold green" if a.adherence_pct >= 90 else "bold yellow")
    ab.append(f"stop-moved-against {a.stop_moved_against} · oversized {a.oversized} · "
              f"overrides {a.overrides} · revenge {a.revenge}", style="dim")
    console.print(Panel(ab, title="Adherence (discipline is measured)", border_style="grey50", title_align="left"))

    vb = Text()
    for v in rep.verdicts:
        vb.append("  • " + v + "\n")
    console.print(Panel(vb or Text("—"), title="Verdicts", border_style="green", title_align="left"))
    console.print(Panel(Text(DISCLAIMER, style="dim"), border_style="dim"))


def cmd_review(args: argparse.Namespace) -> int:
    settings = load_settings()
    _render_review(build_review(cfg=settings.journal), settings)
    return 0


def _ping_fired(fired) -> None:
    if not fired:
        console.print(Text("No alerts triggered.", style="dim"))
        return
    console.bell()
    for alert, msg in fired:
        console.print(Text("🔔 ALERT  ", style="bold yellow") + Text(msg, style="bold"))
    console.print(Text("(notification only — no order placed. Run `analyze` to act.)", style="dim"))


def cmd_alert(args: argparse.Namespace) -> int:
    settings = load_settings()
    if args.list:
        alerts = list_alerts(active_only=not args.all)
        if not alerts:
            console.print(Text("No alerts.", style="dim"))
            return 0
        table = Table(title="Alerts", header_style="bold")
        for c in ("ID", "Symbol", "Kind", "Level/TF", "Dir", "Status"):
            table.add_column(c, no_wrap=True)
        for a in alerts:
            table.add_row(a.id, a.symbol, a.kind,
                          f"{a.level:,.6g}" if a.kind == "level" else a.tf,
                          a.direction or "—", a.status)
        console.print(table)
        return 0
    if args.remove:
        console.print("removed." if remove_alert(args.remove) else "no such alert id.")
        return 0
    if args.check:
        market = _market_from_args(args)
        try:
            fired = check_alerts(market, settings)
        except DataError as exc:
            console.print(_data_error_panel(exc))
            return 1
        _ping_fired(fired)
        return 0

    # create
    if not args.symbol:
        console.print("[red]Specify SYMBOL --level X | --setup, or --list / --check / --remove ID[/red]")
        return 2
    market = _market_from_args(args)
    if args.setup:
        tf = args.tf or settings.alerts.tf
        a = add_setup_alert(market, args.symbol, tf, note=args.note)
        console.print(f"alert {a.id}: SETUP on {a.symbol} ({tf}) — fires when a setup triggers")
        return 0
    if args.level is not None:
        try:
            ex = make_exchange(market, settings.api_key, settings.api_secret)
            load_markets(ex)
            cur = float(ex.fetch_ticker(args.symbol)["last"])
        except DataError as exc:
            console.print(_data_error_panel(exc))
            return 1
        a = add_level_alert(market, args.symbol, args.level, cur, note=args.note)
        arrow = "↑" if a.direction == "up" else "↓"
        console.print(f"alert {a.id}: LEVEL {a.symbol} {arrow} {a.level:,.6g} (now {cur:,.6g})")
        return 0
    console.print("[red]Provide --level X or --setup[/red]")
    return 2


_EDGE_STYLE = {es.EDGE_PROVEN: "green", es.EDGE_UNPROVEN: "yellow", es.EDGE_NEGATIVE: "red"}


def _render_stage(symbol, market, signal, plan, ticket, guard, last, drift_msg, edge=None,
                  risk_pct=None, risk_notes=None) -> None:
    cb = Text()
    cb.append("Guards: PASS (no hard blocks)\n", style="green")
    for w in guard.soft_warns:
        cb.append(f"  ⚠ {w}\n", style="yellow")
    if edge is not None:
        level, msg = edge
        glyph = "✓" if level == es.EDGE_PROVEN else "⚠"
        cb.append(f"Proven edge: {glyph} {msg}\n", style=_EDGE_STYLE.get(level, "yellow"))
    if risk_notes:
        cb.append(f"Risk sizing: {risk_pct:.2f}% — " + "; ".join(risk_notes) + "\n", style="cyan")
    cb.append(f"Re-fetch: last {last:,.6g} — {drift_msg}", style="dim")
    console.print(Panel(cb, title="Pre-trade checklist", border_style="blue", title_align="left"))

    body = Text()
    body.append(f"Setup: {signal.setup} {signal.direction} (grade {signal.grade})\n", style="bold")
    for line in plan_worked_lines(plan):
        body.append("  • ", style="bold cyan")
        body.append(line + "\n")
    console.print(Panel(body, title=f"Plan — {symbol}", border_style="green", title_align="left"))

    tb = Text()
    if ticket.leverage:
        tb.append(f"(USD-M: set {ticket.leverage:g}x {ticket.margin_mode} margin first)\n", style="dim")
    for leg in ticket.legs:
        ro = " reduceOnly" if leg.reduce_only else ""
        frac = f" ×{leg.qty_fraction:g}" if leg.qty_fraction != 1.0 else ""
        tb.append(f"  {leg.kind.upper():5s} {leg.side.upper():4s} {leg.order_type} @ {leg.price:,.6g}{frac}{ro}\n")
    console.print(Panel(tb, title="Order ticket (DRY-RUN — not sent)", border_style="grey50", title_align="left"))


def cmd_stage(args: argparse.Namespace) -> int:
    settings = _load_tiered(args)
    market = _market_from_args(args)
    symbol = args.symbol
    # portfolio state FIRST — its drawdown feeds the effective-risk pipeline at sizing time.
    records = load_records()
    state = portfolio_state(records, settings.journal)
    try:
        with _progress() as progress:
            progress.add_task(f"Analysing {symbol}…", total=None)
            result = run_analyze(market, symbol, settings, tf_trigger=args.tf,
                                 drawdown_pct=state.drawdown_pct)
    except DataError as exc:
        console.print(_data_error_panel(exc))
        return 1

    if not result.setups or result.plan is None:
        console.print(Panel(Text("Nothing to stage — no live setup right now → NO TRADE.",
                                 style="yellow"), title="Stage", border_style="yellow", title_align="left"))
        return 0
    signal, plan = result.setups[0], result.plan
    if not plan.valid:
        console.print(Panel(Text("Setup present, but the plan is NOT worth taking:\n  "
                                 + "\n  ".join(plan.notes), style="yellow"),
                            title="Stage — NO TRADE", border_style="yellow", title_align="left"))
        return 0

    if od.has_open_or_staged(records, symbol):
        console.print(Panel(Text(f"Already an OPEN/STAGED position on {symbol} — refusing a duplicate.",
                                 style="red"), title="Stage — blocked", border_style="red", title_align="left"))
        return 0

    proposed = od.build_proposed(plan, signal, group="indep", risk_pct=result.risk_pct)
    guard = check_trade(state, proposed, settings.guards)
    if not guard.allowed:
        console.print(Panel(Text("GUARD BLOCK (hard) — cannot stage:\n  "
                                 + "\n  ".join("• " + h for h in guard.hard_blocks), style="red"),
                            title="Stage — blocked", border_style="red", title_align="left"))
        return 0

    # PROVEN-EDGE gate (pooled across the universe): never stage a proven loser.
    regime = result.structure.state.trend if (result.structure and result.structure.state) else ""
    edge_verdict = es.lookup_pooled_verdict(market, result.tf_trigger, signal.setup, regime, symbol=symbol,
                                            context=es.resolve_context(result.context, signal.direction))
    edge_level, edge_msg = es.edge_gate(edge_verdict, settings.edge)
    if edge_level == es.EDGE_NEGATIVE:
        console.print(Panel(Text(f"EDGE BLOCK — this setup is a proven loser here:\n  • {edge_msg}\n"
                                 "The provisional grade reflects confluence/R:R, but the backtested edge is "
                                 "negative. Refusing to stage.", style="red"),
                            title="Stage — blocked (proven −EV)", border_style="red", title_align="left"))
        return 0

    try:
        ex = make_exchange(market, settings.api_key, settings.api_secret)
        load_markets(ex)
        last = float(ex.fetch_ticker(symbol)["last"])
    except DataError as exc:
        console.print(_data_error_panel(exc))
        return 1
    drift_ok, dmsg = od.check_drift(signal.entry_type, plan.entry, last, settings.order.drift_pct)
    val_ok, vmsg = od.check_validity(signal, last)
    if not (drift_ok and val_ok):
        reasons = [m for ok, m in ((drift_ok, dmsg), (val_ok, vmsg)) if not ok]
        console.print(Panel(Text("RE-FETCH ABORT (stale read):\n  " + "\n  ".join("• " + r for r in reasons),
                                 style="red"), title="Stage — aborted", border_style="red", title_align="left"))
        return 0

    ticket = od.build_ticket(plan, signal, market, settings.risk_mgmt.tp1_fraction)
    _render_stage(symbol, market, signal, plan, ticket, guard, last, dmsg, edge=(edge_level, edge_msg),
                  risk_pct=result.risk_pct, risk_notes=result.risk_notes)

    word = settings.order.confirm_word
    try:
        typed = input(f"\nType {word} to stage (DRY-RUN — nothing is sent), anything else to abort: ").strip()
    except EOFError:
        typed = ""
    if typed != word:
        console.print(Text("Aborted — no CONFIRM.", style="yellow"))
        return 0

    regime = result.structure.state.trend if (result.structure and result.structure.state) else ""
    tid = record_staged(
        market=market.value, symbol=symbol, side=signal.direction, setup=signal.setup,
        grade=signal.grade, group=proposed.group, entry=plan.entry, stop=plan.stop,
        targets=list(plan.targets), risk_pct=result.risk_pct, risk_amount=plan.risk_actual,
        edge_r=getattr(plan, "net_rr", None), regime=regime, bias=result.bias, tide=result.posture,
        tf=result.tf_trigger, entry_type=signal.entry_type, invalidation=signal.invalidation,
        leverage=plan.leverage, liquidation_price=plan.liquidation_price, mode="dry-run")
    console.print(Panel(Text(f"DRY-RUN: staged to the journal (id {tid}). NO order sent.",
                             style="bold green"), title="Staged (dry-run)", border_style="green", title_align="left"))
    if getattr(args, "live", False):
        _live_place(settings, market, symbol, signal, plan, tid, testnet=getattr(args, "testnet", False))
    return 0


def _confirm(word: str) -> bool:
    try:
        typed = input(f"\nType {word} to apply (DRY-RUN — nothing is sent), anything else to skip: ").strip()
    except EOFError:
        typed = ""
    if typed != word:
        console.print(Text("Skipped — no CONFIRM.", style="yellow"))
        return False
    return True


_ACTION_STYLE = {
    mon.HOLD: "dim", mon.SCALE_TP1: "bold green", mon.CLOSE_TP2: "bold green",
    mon.STOP_EXIT: "bold red", mon.TRAIL: "yellow", mon.PAPER_FILL: "bold cyan", mon.CANCEL: "red",
}


def _render_position(rec, last: float, staged: bool = False) -> None:
    side_style = _SIDE_STYLE.get(rec.side, "white")
    entry = rec.entry_actual if rec.entry_actual is not None else rec.entry_planned
    cur_stop = rec.current_stop if rec.current_stop is not None else rec.stop_planned
    body = Text()
    body.append(f"{rec.symbol}  ", style="bold")
    body.append(f"{rec.side.upper()} ", style=side_style)
    status = "STAGED (dry-run)" if staged else f"OPEN ({rec.remaining_fraction:g} left)"
    body.append(f"{rec.setup} grade {rec.grade}  [{status}]\n")
    body.append(f"entry {entry:,.6g}  ·  stop {cur_stop:,.6g}", style="dim")
    if rec.tp1_filled:
        body.append("  ·  TP1 ✓", style="green")
    body.append(f"  ·  live {last:,.6g}", style="dim")
    if not staged:
        body.append(f"  ·  open {mon.unrealized_r(rec, last):+.2f}R", style="dim")
    console.print(Panel(body, title="Position", border_style=side_style, title_align="left"))


def _render_action(action, warnings) -> None:
    style = _ACTION_STYLE.get(action.kind, "white")
    body = Text()
    body.append(f"Proposed: {action.kind.upper().replace('_', ' ')}\n", style=style)
    body.append(action.reason)
    for w in warnings:
        body.append(f"\n⚠ {w}", style="yellow")
    console.print(Panel(body, title="Manage — proposed action", border_style=style, title_align="left"))


def _manage_staged(rec, bars, last: float, word: str) -> None:
    filled, entry_actual, why = mon.detect_paper_fill(rec, bars, last)
    _render_position(rec, last, staged=True)
    if not filled and "void" in why:
        console.print(Panel(Text(f"{why} → propose CANCEL.", style="red"),
                            title="Paper-fill — voided", border_style="red", title_align="left"))
        console.bell()
        if _confirm(word):
            record_cancel(rec.id, reason=why)
            console.print(Text(f"Cancelled {rec.symbol} (dry-run).", style="yellow"))
        return
    if not filled:
        console.print(Panel(Text(f"Pending — {why}. Nothing to do yet.", style="dim"),
                            title="Paper-fill — pending", border_style="grey50", title_align="left"))
        return
    tgts = ", ".join(f"{t:,.6g}" for t in rec.targets)
    console.print(Panel(Text(f"PAPER-FILL: {why}.\nWould OPEN {rec.side} {rec.symbol}: "
                             f"entry {entry_actual:,.6g}, stop {rec.stop_planned:,.6g}, targets {tgts}.",
                             style="bold cyan"), title="Paper-fill (dry-run)",
                        border_style="cyan", title_align="left"))
    console.bell()
    if not _confirm(word):
        return
    record_open(rec.id, entry_actual=entry_actual, current_stop=rec.stop_planned, mode="dry-run")
    console.print(Text(f"OPENED {rec.symbol} (paper, dry-run). Run `manage` again to manage it on real prices.",
                       style="green"))


def _manage_open(rec, bars, last: float, market: Market, settings, word: str, mkt=None) -> None:
    warns = []
    if market is Market.USDM and mkt is not None:
        fr = mkt.funding(rec.symbol)
        oi = mkt.open_interest(rec.symbol)
        warns = mon.usdm_warnings(
            rec, last, settings.markets,
            funding_window_minutes=settings.guards.funding_window_minutes,
            funding_rate=(fr.rate if fr else None), funding_signal=(fr.signal if fr else None),
            oi_trend=(oi.trend if oi else None), oi_change_pct=(oi.change_pct if oi else None))
    action = mon.manage_step(rec, bars, last, settings.risk_mgmt)
    _render_position(rec, last)
    _render_action(action, warns)
    if action.kind == mon.HOLD:
        return
    console.bell()
    if not _confirm(word):
        return
    now = datetime.now(timezone.utc).isoformat()
    if action.kind in (mon.STOP_EXIT, mon.CLOSE_TP2):
        pnl = (action.total_realized_r or 0.0) * (rec.risk_amount or 0.0)
        record_close(rec.id, exit_price=action.fill_price, realized_r=action.total_realized_r,
                     pnl_usd=pnl, stop_actual=action.new_stop)
        console.print(Panel(Text(f"CLOSED {rec.symbol} (dry-run) at {action.fill_price:,.6g} → "
                                 f"{action.total_realized_r:+.2f}R booked. Review & guards updated.",
                                 style="bold green"), title="Closed (dry-run)",
                            border_style="green", title_align="left"))
    elif action.kind == mon.SCALE_TP1:
        locked = (rec.locked_r or 0.0) + (action.realized_delta_r or 0.0)
        record_manage(rec.id, current_stop=action.new_stop, remaining_fraction=action.remaining_after,
                      tp1_filled=True, locked_r=locked, managed_at=now,
                      note=f"TP1 scale-out +{action.realized_delta_r:.2f}R; stop→breakeven")
        console.print(Panel(Text(f"Scaled out at TP1 (dry-run): booked {action.realized_delta_r:+.2f}R; "
                                 f"{action.remaining_after:g} runs with stop at breakeven {action.new_stop:,.6g}.",
                                 style="bold green"), title="Partial taken (dry-run)",
                            border_style="green", title_align="left"))
    elif action.kind == mon.TRAIL:
        record_manage(rec.id, current_stop=action.new_stop, managed_at=now,
                      note=f"trailed stop → {action.new_stop:,.6g}")
        console.print(Text(f"Trailed stop → {action.new_stop:,.6g} (dry-run).", style="green"))


def _render_preflight(pf) -> None:
    body = Text()
    for name, ok, detail in pf.checks:
        glyph, style = ("✓", "green") if ok is True else (("—", "yellow") if ok is None else ("✗", "red"))
        body.append(f"  {glyph} {name}", style=style)
        if detail:
            body.append(f"  ({detail})", style="dim")
        body.append("\n")
    for b in pf.blocks:
        body.append(f"  ✗ {b}\n", style="bold red")
    console.print(Panel(body, title=("Live preflight — TESTNET" if pf.testnet else "Live preflight — REAL MONEY"),
                        border_style=("green" if pf.ok else "red"), title_align="left"))


def _live_confirm(label: str, phrase: str, what: str) -> bool:
    warn_style = "bold yellow" if "TESTNET" in label else "bold red"
    console.print(Text(f"\n⚠ {what}", style=warn_style))
    try:
        typed = input(f"Type '{phrase}' to proceed on {label}, anything else to abort: ").strip()
    except EOFError:
        typed = ""
    if typed != phrase:
        console.print(Text("Aborted — no CONFIRM LIVE.", style="yellow"))
        return False
    return True


def _arm_live(settings, market: Market, testnet: bool):
    """Build the live/testnet exchange and run preflight. Returns (ex, mkt) or None if blocked."""
    label = "TESTNET" if testnet else "LIVE (REAL MONEY)"
    try:
        ex = lv.make_live_exchange(settings, market, testnet)
    except lv.LiveError as exc:
        console.print(Panel(Text(str(exc), style="red"), title=f"{label} — cannot arm",
                            border_style="red", title_align="left"))
        return None
    pf = lv.preflight(settings, market, testnet, ex)
    _render_preflight(pf)
    if not pf.ok:
        console.print(Text("Preflight FAILED — nothing sent.", style="yellow"))
        return None
    return ex, make_market(market, ex, settings.markets)


def _live_place(settings, market: Market, symbol, signal, plan, tid, testnet: bool) -> None:
    label = "TESTNET" if testnet else "LIVE (REAL MONEY)"
    armed = _arm_live(settings, market, testnet)
    if armed is None:
        console.print(Text("The trade stays STAGED (dry-run).", style="dim"))
        return
    ex, mkt = armed
    try:                                                    # SECOND, independent re-fetch
        last = float(ex.fetch_ticker(symbol)["last"])
    except Exception as exc:  # noqa: BLE001
        console.print(Panel(Text(f"Could not re-fetch price: {exc}", style="red"),
                            title=f"{label} — aborted", border_style="red", title_align="left"))
        return
    drift_ok, dmsg = od.check_drift(signal.entry_type, plan.entry, last, settings.order.drift_pct)
    val_ok, vmsg = od.check_validity(signal, last)
    if not (drift_ok and val_ok):
        reasons = [m for ok, m in ((drift_ok, dmsg), (val_ok, vmsg)) if not ok]
        console.print(Panel(Text("SECOND re-fetch ABORT (read went stale):\n  " + "\n  ".join("• " + r for r in reasons),
                                 style="red"), title=f"{label} — aborted", border_style="red", title_align="left"))
        return
    ticket = od.build_ticket(plan, signal, market, settings.risk_mgmt.tp1_fraction)
    what = (f"About to place a {label} order: {signal.direction} {plan.size:g} {symbol} "
            f"(entry {plan.entry:,.6g}, stop {plan.stop:,.6g}, live {last:,.6g}).")
    if not _live_confirm(label, settings.live.confirm_phrase, what):
        console.print(Text("The trade stays STAGED.", style="dim"))
        return
    console.bell()
    try:
        res = lv.place_entry_with_protection(ex, mkt, symbol, ticket, plan.size, settings.live)
    except lv.LiveError as exc:
        console.print(Panel(Text(str(exc), style="bold red"), title=f"{label} — EXECUTION ABORTED",
                            border_style="red", title_align="left"))
        return
    record_open(tid, entry_actual=res.avg_price, current_stop=plan.stop,
                mode=("testnet" if testnet else "live"), broker_order_ids=res.order_ids)
    extra = ("\n⚠ TP issues (position IS protected by the stop): " + "; ".join(res.tp_errors)) if res.tp_errors else ""
    console.print(Panel(Text(f"{label}: filled {res.filled_qty:g} @ {res.avg_price} — protective stop placed.\n"
                             f"broker ids: {res.order_ids}{extra}\n"
                             "Manage with `manage --live`; kill switch `manage --flatten --live`.",
                             style="bold green"), title=f"{label} — ORDER LIVE", border_style="green", title_align="left"))


def _cmd_flatten(settings, market: Market, testnet: bool) -> int:
    label = "TESTNET" if testnet else "LIVE (REAL MONEY)"
    armed = _arm_live(settings, market, testnet)
    if armed is None:
        return 0
    ex, mkt = armed
    if not _live_confirm(label, settings.live.confirm_phrase,
                         f"KILL SWITCH: market-CLOSE ALL {label} positions and CANCEL ALL orders."):
        return 0
    console.bell()
    try:
        summary = lv.flatten_all(ex, mkt)
    except lv.LiveError as exc:
        console.print(Panel(Text(str(exc), style="bold red"), title=f"{label} — flatten failed",
                            border_style="red", title_align="left"))
        return 1
    console.print(Panel(Text(f"Flattened {label}. Closed: {summary['closed'] or '—'}; "
                             f"cancelled {len(summary['cancelled'])} order(s).", style="bold green"),
                        title=f"{label} — flattened", border_style="green", title_align="left"))
    return 0


def _cmd_manage_live(args, settings, market: Market) -> int:
    testnet = getattr(args, "testnet", False)
    armed = _arm_live(settings, market, testnet)
    if armed is None:
        return 0
    ex, mkt = armed
    records = load_records()
    live_recs = [r for r in records if r.market == market.value and r.status == OPEN
                 and r.mode in ("live", "testnet")]
    if args.symbol:
        live_recs = [r for r in live_recs if r.symbol == args.symbol]
    if not live_recs:
        console.print(Panel(Text("No live/testnet OPEN positions to manage.", style="yellow"),
                            title="Manage (live)", border_style="yellow", title_align="left"))
        return 0
    # 1. RECONCILE — the exchange is the source of truth.
    try:
        findings = lv.reconcile_live(ex, live_recs)
    except lv.LiveError as exc:
        console.print(Panel(Text(str(exc), style="red"), title="Reconcile failed",
                            border_style="red", title_align="left"))
        return 1
    finding_kind = {f.symbol: f for f in findings}
    for f in findings:
        style = {"ok": "green", "naked": "bold red", "closed": "yellow"}.get(f.kind, "white")
        console.print(Text(f"  • {f.symbol}: {f.kind.upper()} — {f.detail}", style=style))
        if f.kind == "closed":
            rec = next(r for r in live_recs if r.symbol == f.symbol)
            exit_price = rec.current_stop if rec.current_stop is not None else rec.stop_planned
            realized = mon.unrealized_r(rec, exit_price)
            record_close(rec.id, exit_price=exit_price, realized_r=realized,
                         pnl_usd=realized * (rec.risk_amount or 0.0), stop_actual=exit_price)
            console.print(Text(f"      → journal marked CLOSED ({realized:+.2f}R, reconciled).", style="dim"))
    # 2. Manage the positions still genuinely open + protected.
    label = "TESTNET" if testnet else "LIVE (REAL MONEY)"
    for rec in [r for r in live_recs if finding_kind.get(r.symbol, None) and finding_kind[r.symbol].kind == "ok"]:
        tf = rec.tf or settings.analysis.tf_trigger
        try:
            bars = drop_unclosed(ex, fetch_ohlcv(ex, rec.symbol, tf, settings.analysis.candle_limit), tf)
            last = float(ex.fetch_ticker(rec.symbol)["last"])
        except DataError as exc:
            console.print(_data_error_panel(exc))
            continue
        action = mon.manage_step(rec, bars, last, settings.risk_mgmt)
        _render_position(rec, last)
        _render_action(action, [])
        if action.kind == mon.HOLD:
            continue
        try:
            pos = lv._open_positions(ex).get(rec.symbol, {})
            qty = abs(float(pos.get("contracts") or 0.0))
        except lv.LiveError:
            qty = 0.0
        exit_side = "sell" if rec.side == lv.LONG else "buy"
        if not _live_confirm(label, settings.live.confirm_phrase, f"Execute {action.kind.upper()} on {rec.symbol}."):
            continue
        console.bell()
        now = datetime.now(timezone.utc).isoformat()
        if action.kind == mon.SCALE_TP1:                    # the TP1 order fills itself; move the stop to BE
            new_id = lv.move_stop(ex, mkt, rec.symbol, rec.broker_order_ids.get("stop"),
                                  exit_side, action.new_stop, qty)
            ids = dict(rec.broker_order_ids); ids["stop"] = new_id
            record_manage(rec.id, current_stop=action.new_stop, remaining_fraction=action.remaining_after,
                          tp1_filled=True, locked_r=(rec.locked_r or 0.0) + (action.realized_delta_r or 0.0),
                          managed_at=now, broker_order_ids=ids, note="live: stop→breakeven after TP1")
            console.print(Text(f"Moved stop → breakeven {action.new_stop:,.6g} (live).", style="green"))
        elif action.kind in (mon.STOP_EXIT, mon.CLOSE_TP2):
            lv.close_position(ex, mkt, rec.symbol, exit_side, qty, rec.broker_order_ids)
            record_close(rec.id, exit_price=action.fill_price, realized_r=action.total_realized_r,
                         pnl_usd=(action.total_realized_r or 0.0) * (rec.risk_amount or 0.0),
                         stop_actual=action.new_stop)
            console.print(Text(f"CLOSED {rec.symbol} (live) {action.total_realized_r:+.2f}R.", style="bold green"))
    return 0


def cmd_manage(args: argparse.Namespace) -> int:
    settings = _load_tiered(args)
    market = _market_from_args(args)
    if getattr(args, "flatten", False):
        if not (getattr(args, "live", False) or getattr(args, "testnet", False)):
            console.print(Text("--flatten needs --live or --testnet (it touches real orders).", style="red"))
            return 2
        return _cmd_flatten(settings, market, testnet=getattr(args, "testnet", False))
    if getattr(args, "live", False) or getattr(args, "testnet", False):
        return _cmd_manage_live(args, settings, market)
    word = settings.order.confirm_word
    records = load_records()
    manageable = [r for r in records if r.market == market.value and r.status in (OPEN, STAGED)]
    if args.symbol:
        manageable = [r for r in manageable if r.symbol == args.symbol]
    if not manageable:
        scope = f" for {args.symbol}" if args.symbol else ""
        console.print(Panel(Text(f"No OPEN or STAGED {market.label} positions to manage{scope}.\n"
                                 "Stage one first: `stage SYMBOL` (dry-run).", style="yellow"),
                            title="Manage", border_style="yellow", title_align="left"))
        return 0
    try:
        ex = make_exchange(market, settings.api_key, settings.api_secret)
        load_markets(ex)
    except DataError as exc:
        console.print(_data_error_panel(exc))
        return 1
    mkt = make_market(market, ex, settings.markets)

    console.print(Panel(Text("DRY-RUN — manage PROPOSES; you CONFIRM each action; nothing is sent. "
                             "It uses the SAME deterministic state machine as the backtest.", style="dim"),
                        title=f"Manage {market.label} — {len(manageable)} position(s)",
                        border_style="blue", title_align="left"))
    for rec in manageable:
        tf = rec.tf or settings.analysis.tf_trigger
        try:
            bars = drop_unclosed(ex, fetch_ohlcv(ex, rec.symbol, tf, settings.analysis.candle_limit), tf)
            last = float(ex.fetch_ticker(rec.symbol)["last"])
        except DataError as exc:
            console.print(_data_error_panel(exc))
            continue
        if rec.status == STAGED:
            _manage_staged(rec, bars, last, word)
        else:
            _manage_open(rec, bars, last, market, settings, word, mkt)
    return 0


def _render_optimize_report(rep) -> bool:
    d = rep.decision
    border = "green" if d.adopt else "grey50"
    head = Text()
    head.append(f"{rep.setup}  ·  {rep.param}\n", style="bold")
    head.append(f"grid {', '.join(f'{g:g}' for g in rep.grid)}  ·  pooled over {len(rep.coins)} coins\n", style="dim")
    head.append(f"default {rep.default_value:g} → OOS {rep.default_test_exp:+.3f}R   |   "
                f"best {rep.best_value:g} → OOS {rep.best_test_exp:+.3f}R\n", style="white")
    head.append("gates:\n", style="bold")
    for note in d.notes:
        head.append(f"  {note}\n", style=("green" if note.startswith("✓") else "red"))
    verdict = "ADOPT (proposed)" if d.adopt else "KEEP DEFAULT"
    head.append(f"DECISION: {verdict}", style=("bold green" if d.adopt else "bold yellow"))
    console.print(Panel(head, title=f"optimize — {rep.setup}", border_style=border, title_align="left"))
    return d.adopt


def cmd_optimize(args: argparse.Namespace) -> int:
    settings = _load_tiered(args)
    market = _market_from_args(args)
    ocfg = settings.optimize
    tf = args.tf or settings.analysis.tf_trigger
    param = args.param or "stop_buffer_atr"
    n_coins = args.coins or ocfg.coins
    setups = [args.setup] if args.setup else list(SETUP_NAMES)

    console.print(Panel(Text(
        "Walk-forward config optimization — the PROCESS is the proof.\n"
        "Select on TRAIN, validate on out-of-sample TEST, confirm on the LOCK-BOX, correct for "
        "multiple-testing, demand a plateau + cost survival. Proposes a diff — applies NOTHING.",
        style="dim"), title=f"optimize {market.label} · {tf} · param {param}",
        border_style="blue", title_align="left"))

    try:
        with _progress() as p:
            p.add_task("Screening for liquid coins…", total=None)
            screen = run_screen(market, settings.screener, settings.api_key, settings.api_secret)
        symbols = [c.symbol for c in screen.candidates][:n_coins]
        with _progress() as p:
            p.add_task(f"Fetching history for {len(symbols)} coins…", total=None)
            bars_by_coin = opt.fetch_bars(market, symbols, settings, tf)
    except DataError as exc:
        console.print(_data_error_panel(exc))
        return 1
    if not bars_by_coin:
        console.print(Text("No history fetched — cannot optimize.", style="red"))
        return 1

    proposed: list = []
    with _progress() as p:
        task = p.add_task(f"Optimizing {len(setups)} setup(s) × grid…", total=len(setups))
        for name in setups:
            rep = opt.optimize_setup(market, bars_by_coin, settings, tf, name, param, ocfg)
            p.advance(task)
            p.stop()
            if _render_optimize_report(rep):
                proposed.append(rep)
            p.start()

    if proposed:
        body = Text()
        body.append("Proposed config changes (NOT applied — review, then edit config.py / .env):\n\n", style="bold")
        for rep in proposed:
            body.append(f"  {rep.setup}: {rep.param} {rep.default_value:g} → {rep.best_value:g}  "
                        f"(OOS {rep.default_test_exp:+.3f}R → {rep.best_test_exp:+.3f}R)\n", style="green")
        body.append("\nNote: per-setup `stop_buffer_atr` overrides would need a per-setup config "
                    "(currently one shared default) — adopt deliberately.", style="dim")
        console.print(Panel(body, title="PROPOSED DIFF", border_style="green", title_align="left"))
    else:
        console.print(Panel(Text("No parameter beat its default out-of-sample with proof — KEEP all defaults. "
                                 "(The honest, common outcome.)", style="yellow"),
                            title="Result", border_style="yellow", title_align="left"))
    console.print(Panel(Text(DISCLAIMER, style="dim"), border_style="dim"))
    return 0


def _render_basket(basket, equity, gcfg, tilt: bool) -> None:
    table = Table(title=f"ROAR basket — {len(basket)} proven trade(s) · "
                        f"{'edge-weighted' if tilt else 'equal-risk'} allocation",
                  title_style="bold", header_style="bold", expand=True)
    for col, j in (("#", "right"), ("Symbol", "left"), ("Side", "left"), ("Setup", "left"),
                   ("Risk $", "right"), ("Risk %", "right"), ("Size", "right"), ("Notional", "right"),
                   ("net R:R", "right"), ("Group", "left"), ("Edge", "right")):
        table.add_column(col, justify=j, no_wrap=True)
    pos = []
    for i, (a, sig, plan, ar) in enumerate(basket, 1):
        table.add_row(str(i), a.symbol, Text(sig.direction, style=_SIDE_STYLE.get(sig.direction, "")),
                      f"{sig.setup} {sig.grade}", f"${plan.risk_actual:,.2f}", f"{a.risk_pct:.2f}%",
                      f"{plan.size:g}", f"${plan.notional:,.0f}", f"{plan.net_rr:.2f}",
                      f"{a.group}{'*' if a.group not in ('indep','-','') else ''}", f"{a.score:+.2f}")
        pos.append(gd.OpenPosition(a.symbol, a.group, plan.risk_actual, sig.direction))
    console.print(table)
    heat = gd.basket_heat_usd(pos, gcfg.group_corr_factor)
    heat_pct = heat / equity * 100.0 if equity > 0 else 0.0
    groups = sorted({a.group for a, *_ in basket})
    console.print(Text(f"Portfolio heat: ${heat:,.2f} ({heat_pct:.2f}% of ${equity:,.0f}; cap {gcfg.heat_cap_pct:g}%) "
                       f"· groups: {', '.join(groups)} · positions {len(basket)}/{gcfg.max_positions}",
                       style="dim"))


def _match_leg_setup(setups, setup: str, side: str):
    """Find THIS basket leg's setup among all currently-detected ones. The board ranks by
    EDGE SCORE while analyze ranks by provisional GRADE, so the leg's setup may not be #1 —
    matching by (setup, direction) avoids a false 'setup changed' skip."""
    return next((s for s in setups if s.setup == setup and s.direction == side), None)


def _leg_ok_to_send(fsig, expected_setup: str, expected_side: str, plan_valid: bool,
                    edge_level: str, drift_ok: bool, val_ok: bool) -> tuple[bool, str]:
    """Re-validate ONE basket leg at the moment of the live send (mirrors `stage`): the
    proven setup must still be present + the same, the plan still valid, the edge still
    PROVEN in the CURRENT regime, and no drift / pre-entry invalidation. Pure + testable."""
    if fsig is None:
        return False, "setup gone at send time — skipped"
    if fsig.setup != expected_setup or fsig.direction != expected_side:
        return False, f"setup changed to {fsig.setup} {fsig.direction} since the board — skipped"
    if not plan_valid:
        return False, "plan no longer valid (R:R/sizing) — skipped"
    if edge_level == es.EDGE_NEGATIVE:
        return False, "edge flipped −EV in the current regime — skipped"
    if edge_level != es.EDGE_PROVEN:
        return False, "edge no longer PROVEN in the current regime — skipped"
    if not drift_ok:
        return False, "price drifted past tolerance since the board — skipped"
    if not val_ok:
        return False, "pre-entry invalidation already hit — skipped"
    return True, "re-validated (proven edge · plan valid · no drift)"


def _roar_live(basket, settings, market: Market, testnet: bool) -> None:
    label = "TESTNET" if testnet else "LIVE (REAL MONEY)"
    armed = _arm_live(settings, market, testnet)
    if armed is None:
        console.print(Text("Basket stays STAGED (dry-run).", style="dim"))
        return
    ex, mkt = armed
    # margin overview — so the user sees BEFORE confirming whether the basket fits the balance
    quote = lv._quote_of(basket[0][0].symbol) if basket else "USDT"
    free = lv._free_margin(ex, quote)
    need = sum(p.notional / max(p.leverage or 1.0, 1.0) for (_, _, p, _) in basket) * 1.05
    if free is not None:
        short = need > free
        console.print(Text(f"Margin: basket needs ~{need:,.2f} {quote}, {free:,.2f} free"
                           + (" — some legs will be SKIPPED (insufficient margin)." if short else ""),
                           style=("yellow" if short else "dim")))
    if not _live_confirm(label, settings.live.confirm_phrase,
                         f"Place ALL {len(basket)} {label} orders (each arms its own stop-or-bail)."):
        return
    console.bell()
    placed = 0
    tf = settings.edge.tf
    for (a, sig, plan, ar) in basket:
        # RE-VALIDATE on FRESH data right before the send — a regime can flip (proven→−EV), price
        # can drift, or the invalidation can hit while you were confirming. `stage` already does this;
        # the master strike must too, or it could place a leg `stage` would have refused.
        try:
            fresh = run_analyze(market, a.symbol, settings, tf_trigger=tf, ex=ex)
        except DataError as exc:
            console.print(Text(f"  ✗ {a.symbol}: re-fetch failed ({exc}) — skipped", style="yellow"))
            continue
        fsig = _match_leg_setup(fresh.setups, sig.setup, sig.direction)
        regime = fresh.structure.state.trend if (fresh.structure and fresh.structure.state) else ""
        edge_level = es.EDGE_UNPROVEN
        fresh_plan = None
        drift_ok = val_ok = True
        if fsig is not None:
            edge_level, _ = es.edge_gate(
                es.lookup_pooled_verdict(market, tf, fsig.setup, regime, symbol=a.symbol,
                                         context=es.resolve_context(fresh.context, fsig.direction)), settings.edge)
            fresh_plan = plan_trade(mkt, symbol=a.symbol, side=fsig.direction, entry=fsig.entry,
                                    stop=fsig.stop, targets=fsig.targets,
                                    account_equity=settings.risk.account_equity, risk_pct=a.risk_pct,
                                    mgmt=settings.risk_mgmt,
                                    funding_rate=(fresh.funding.rate if fresh.funding else None))
            drift_ok, _ = od.check_drift(fsig.entry_type, fresh_plan.entry, fresh.last_price, settings.order.drift_pct)
            val_ok, _ = od.check_validity(fsig, fresh.last_price)
        ok, reason = _leg_ok_to_send(fsig, sig.setup, sig.direction,
                                     bool(fresh_plan and fresh_plan.valid), edge_level, drift_ok, val_ok)
        if not ok:
            console.print(Text(f"  ✗ {a.symbol}: {reason}", style="yellow"))
            continue
        ticket = od.build_ticket(fresh_plan, fsig, market, settings.risk_mgmt.tp1_fraction)
        try:
            res = lv.place_entry_with_protection(ex, mkt, a.symbol, ticket, fresh_plan.size, settings.live)
        except lv.LiveError as exc:               # expected, safe failure — skip this leg, keep the basket
            console.print(Text(f"  ✗ {a.symbol}: {exc}", style="red"))
            continue
        except Exception as exc:                  # unexpected — a leg may be half-placed; reconcile
            console.print(Text(f"  ⚠ {a.symbol}: unexpected error after send ({exc}) — run "
                               f"`manage --{market.value} --live` to reconcile a possible unprotected position.",
                               style="bold red"))
            continue
        placed += 1
        console.print(Text(f"  ✓ {a.symbol}: filled {res.filled_qty:g} @ {res.avg_price} — stop armed "
                           f"({reason})", style="green"))
    if placed == 0:
        console.print(Text("No legs placed — each failed re-validation at send time (regime flip / drift / "
                           "stale). The board is a candidate list, not a signal.", style="yellow"))


def cmd_roar(args: argparse.Namespace) -> int:
    settings = _load_tiered(args)
    market = _market_from_args(args)
    tilt = getattr(args, "tilt", False)
    console.print(Panel(Text("ROAR — one-shot strike on the whole PROVEN basket.\n"
                             "tide → proven board → allocate the risk budget → CONFIRM. "
                             "Inherits every gate; an empty board means nothing to do.", style="dim"),
                        title=f"roar {market.label}", border_style="blue", title_align="left"))
    try:
        with _progress() as p:
            p.add_task("Scanning for proven edge…", total=None)
            ctx, opportunities, watchlist = run_scan(market, settings, refresh=getattr(args, "refresh", False))
    except DataError as exc:
        console.print(_data_error_panel(exc))
        return 1
    if ctx is not None:
        console.print(_render_context_banner(ctx))
    if not opportunities:
        wl = ", ".join(watchlist[:10]) if watchlist else "—"
        console.print(Panel(Text("No PROVEN-edge opportunities right now — nothing to ROAR.\n"
                                 f"Watchlist (live setup, edge not yet proven): {wl}\n"
                                 "Leave `watch` running and it'll ping you the moment one appears.",
                                 style="yellow"), title="ROAR — stand aside", border_style="yellow",
                            title_align="left"))
        return 0

    items = [dict(symbol=o.symbol, group=o.group, side=o.side,
                  score=max(o.edge_score_r, 0.0) * max(o.confidence, 0.01)) for o in opportunities]
    # account-level risk stages (drawdown-scale + hard ceiling) shrink the WHOLE basket budget;
    # per-trade edge weighting is handled by allocate_basket's --tilt.
    state = portfolio_state(load_records(), settings.journal)
    radj = effective_risk_pct(settings.risk.risk_pct, settings.risk_mgmt, drawdown_pct=state.drawdown_pct)
    if radj.notes:
        console.print(Text("Basket risk: base {:.2f}% → {:.2f}% ({})".format(
            radj.base_pct, radj.effective_pct, "; ".join(radj.notes)), style="cyan"))
    allocs = gd.allocate_basket(items, settings.risk.account_equity, settings.guards,
                                radj.effective_pct, tilt=tilt)
    try:
        ex = make_exchange(market, settings.api_key, settings.api_secret)
        load_markets(ex)
    except DataError as exc:
        console.print(_data_error_panel(exc))
        return 1
    mkt = make_market(market, ex, settings.markets)

    basket = []
    for a in allocs:
        try:
            ar = run_analyze(market, a.symbol, settings, tf_trigger=settings.edge.tf, ex=ex)
        except DataError:
            continue
        if not ar.setups:
            continue
        sig = ar.setups[0]
        plan = plan_trade(mkt, symbol=a.symbol, side=sig.direction, entry=sig.entry, stop=sig.stop,
                          targets=sig.targets, account_equity=settings.risk.account_equity,
                          risk_pct=a.risk_pct, mgmt=settings.risk_mgmt,
                          funding_rate=(ar.funding.rate if ar.funding else None))
        if plan.valid:
            basket.append((a, sig, plan, ar))
    if not basket:
        console.print(Text("Proven edges found, but none produced a valid R:R plan right now — stand aside.",
                           style="yellow"))
        return 0

    _render_basket(basket, settings.risk.account_equity, settings.guards, tilt)
    word = settings.order.confirm_word
    try:
        typed = input(f"\nType {word} to STAGE the whole basket (DRY-RUN — sends nothing): ").strip()
    except EOFError:
        typed = ""
    if typed != word:
        console.print(Text("Aborted — no CONFIRM.", style="yellow"))
        return 0

    tids = []
    for (a, sig, plan, ar) in basket:
        regime = ar.structure.state.trend if (ar.structure and ar.structure.state) else ""
        tid = record_staged(
            market=market.value, symbol=a.symbol, side=sig.direction, setup=sig.setup, grade=sig.grade,
            group=a.group, entry=plan.entry, stop=plan.stop, targets=list(plan.targets),
            risk_pct=a.risk_pct, risk_amount=plan.risk_actual, edge_r=getattr(plan, "net_rr", None),
            regime=regime, bias=ar.bias, tide=ar.posture, tf=ar.tf_trigger, entry_type=sig.entry_type,
            invalidation=sig.invalidation, leverage=plan.leverage, liquidation_price=plan.liquidation_price,
            mode="dry-run")
        tids.append(tid)
    console.print(Panel(Text(f"DRY-RUN: staged {len(tids)} trade(s) to the journal. NO orders sent.",
                             style="bold green"), title="ROAR — basket staged (dry-run)",
                        border_style="green", title_align="left"))
    if getattr(args, "live", False):
        _roar_live(basket, settings, market, testnet=getattr(args, "testnet", False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trading-ai",
        description="Screener-first crypto trading analysis & risk tool (Binance SPOT/USD-M). "
                    "Not financial advice; does not predict price.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_market_flags(p: argparse.ArgumentParser) -> None:
        p.add_argument("--spot", action="store_true", help="SPOT market (long-only) [default]")
        p.add_argument("--usdm", action="store_true", help="USD-M perpetual futures (long/short)")

    def add_tier_flag(p: argparse.ArgumentParser) -> None:
        p.add_argument("--tier", choices=sorted(TIERS), default=None,
                       help="speed tier: swing (1d/4h) | intraday (4h/1h)")

    # context
    p_ctx = sub.add_parser("context", help="BTC/market regime + leaders/laggards")
    add_market_flags(p_ctx)
    add_tier_flag(p_ctx)
    p_ctx.add_argument("--top", type=int, default=None, help="How many top-volume symbols to rank")
    p_ctx.add_argument("--quote", type=str, default=None)
    p_ctx.add_argument("--tf", type=str, default=None)
    p_ctx.add_argument("--min-volume", dest="min_volume", type=float, default=None)
    p_ctx.set_defaults(func=cmd_context)

    # scan (the Opportunity Board)
    p_scan = sub.add_parser("scan", help="Opportunity Board: survivors ranked by conservative Edge Score")
    add_market_flags(p_scan)
    add_tier_flag(p_scan)
    p_scan.add_argument("--top", type=int, default=None, help="cap how many survivors to deep-scan (default: ALL that passed)")
    p_scan.add_argument("--refresh", action="store_true", help="Re-backtest (ignore cached edge profiles)")
    p_scan.add_argument("--hard", action="store_true",
                        help="P4: exhaustive feature-PAIR (interaction) search on rebuild (else greedy)")
    p_scan.set_defaults(func=cmd_scan)

    # features (P13 — continuous feature evaluation)
    p_feat = sub.add_parser("features", help="P13: predictive value of each context feature + rising/decaying trend")
    add_market_flags(p_feat)
    add_tier_flag(p_feat)
    p_feat.add_argument("--tf", type=str, default=None)
    p_feat.set_defaults(func=cmd_features)

    # watch (continuous Opportunity Board)
    p_watch = sub.add_parser("watch", help="Continuous Opportunity Board — re-scan on an interval")
    add_market_flags(p_watch)
    add_tier_flag(p_watch)
    p_watch.add_argument("--top", type=int, default=None)
    p_watch.add_argument("--interval", type=float, default=900.0, help="seconds between polls (default 15 min)")
    p_watch.add_argument("--iterations", type=int, default=None, help="stop after N passes (default: until stopped)")
    p_watch.add_argument("--start", dest="watch_start", action="store_true",
                         help="run in the BACKGROUND (desktop notifications + hourly digest)")
    p_watch.add_argument("--stop", dest="watch_stop", action="store_true", help="stop the background watch")
    p_watch.add_argument("--status", dest="watch_status", action="store_true", help="show what watch is monitoring")
    p_watch.add_argument("--_run", dest="watch_run", action="store_true", help=argparse.SUPPRESS)  # internal: the spawned loop
    p_watch.set_defaults(func=cmd_watch)

    # screen
    p_screen = sub.add_parser("screen", help="Ranked candidate shortlist (the user picks from it)")
    add_market_flags(p_screen)
    add_tier_flag(p_screen)
    p_screen.add_argument("--top", type=int, default=None, help="Top-N by 24h volume (default 30)")
    p_screen.add_argument("--quote", type=str, default=None, help="Quote currency (default USDT)")
    p_screen.add_argument("--tf", type=str, default=None, help="Screening timeframe (default 1d)")
    p_screen.add_argument("--min-volume", dest="min_volume", type=float, default=None,
                          help="Min 24h quote volume floor")
    p_screen.add_argument("--equity", type=float, default=None, help="Account equity for worked example")
    p_screen.add_argument("--risk", type=float, default=None, help="Risk %% per trade for worked example")
    p_screen.set_defaults(func=cmd_screen)

    # downstream stubs (build order)
    p_analyze = sub.add_parser("analyze", help="Six-lens multi-TF read of one coin; or --all for a shortlist board")
    add_market_flags(p_analyze)
    add_tier_flag(p_analyze)
    p_analyze.add_argument("symbol", nargs="?", help="coin to read deeply; omit with --all")
    p_analyze.add_argument("--tf", default=None, help="Trigger timeframe (default 4h)")
    p_analyze.add_argument("--all", dest="run_all", action="store_true",
                           help="analyze the WHOLE screener shortlist as one compact board")
    p_analyze.add_argument("--top", type=int, default=None,
                           help="with --all: how many shortlisted coins to analyze (default 30)")
    p_analyze.add_argument("--setups-only", dest="setups_only", action="store_true",
                           help="with --all: show only coins that have a live setup")
    p_analyze.set_defaults(func=cmd_analyze)

    p_backtest = sub.add_parser("backtest", help="Per-setup edge report on real history (walk-forward, MC)")
    add_market_flags(p_backtest)
    add_tier_flag(p_backtest)
    p_backtest.add_argument("symbol")
    p_backtest.add_argument("--tf", default=None, help="Timeframe (default 4h)")
    p_backtest.add_argument("--setup", default=None, help="Limit to one setup")
    p_backtest.add_argument("--full", action="store_true",
                            help="Show the full research view (bootstrap CI, per-fold, recent-vs-overall)")
    p_backtest.set_defaults(func=cmd_backtest)

    p_stage = sub.add_parser("stage", help="Stage a trade: guards→plan→re-fetch→CONFIRM (DRY-RUN; sends nothing)")
    add_market_flags(p_stage)
    add_tier_flag(p_stage)
    p_stage.add_argument("symbol")
    p_stage.add_argument("--tf", default=None, help="Trigger timeframe")
    p_stage.add_argument("--live", action="store_true",
                         help="after the dry-run gate, place a REAL order (preflight + CONFIRM LIVE)")
    p_stage.add_argument("--testnet", action="store_true",
                         help="route --live through the Binance TESTNET sandbox (do this FIRST)")
    p_stage.set_defaults(func=cmd_stage)

    p_manage = sub.add_parser("manage", help="Manage OPEN positions by the plan (TP1→breakeven→close); "
                                             "paper-fills staged dry-runs. Propose-and-CONFIRM; DRY-RUN.")
    add_market_flags(p_manage)
    add_tier_flag(p_manage)
    p_manage.add_argument("symbol", nargs="?", help="manage one symbol; omit to manage all open/staged")
    p_manage.add_argument("--tf", default=None, help="Override the management timeframe")
    p_manage.add_argument("--live", action="store_true",
                          help="manage REAL positions: reconcile with the exchange, then execute on CONFIRM LIVE")
    p_manage.add_argument("--testnet", action="store_true", help="run --live against the TESTNET sandbox")
    p_manage.add_argument("--flatten", action="store_true",
                          help="KILL SWITCH: market-close ALL positions + cancel ALL orders (needs --live/--testnet)")
    p_manage.set_defaults(func=cmd_manage)

    p_roar = sub.add_parser("roar", help="MASTER: tide → proven board → allocate risk across the whole basket "
                                         "→ CONFIRM (dry-run; --live for real, behind CONFIRM LIVE)")
    add_market_flags(p_roar)
    add_tier_flag(p_roar)
    p_roar.add_argument("--tilt", action="store_true", help="edge-weight the allocation (default: equal-risk)")
    p_roar.add_argument("--refresh", action="store_true", help="rebuild the pooled edge cache first")
    p_roar.add_argument("--live", action="store_true", help="place the basket for REAL (preflight + CONFIRM LIVE)")
    p_roar.add_argument("--testnet", action="store_true", help="route --live through the TESTNET sandbox")
    p_roar.set_defaults(func=cmd_roar)

    p_opt = sub.add_parser("optimize", help="Walk-forward config optimization (offline, slow): "
                                            "OOS-validated, propose-a-diff, never auto-applies")
    add_market_flags(p_opt)
    add_tier_flag(p_opt)
    p_opt.add_argument("--setup", default=None, help="optimize one setup (default: all)")
    p_opt.add_argument("--param", default=None, help="parameter to optimize (default: stop_buffer_atr)")
    p_opt.add_argument("--coins", type=int, default=None, help="liquid coins to pool across (default 6)")
    p_opt.add_argument("--tf", default=None, help="timeframe (default trigger tf)")
    p_opt.set_defaults(func=cmd_optimize)

    p_review = sub.add_parser("review", help="Realised expectancy / adherence / drawdown / verdicts")
    p_review.set_defaults(func=cmd_review)

    p_alert = sub.add_parser("alert", help="Level & setup alerts (ping when reached; no auto-exec)")
    add_market_flags(p_alert)
    p_alert.add_argument("symbol", nargs="?")
    p_alert.add_argument("--level", type=float, default=None, help="price level to alert on")
    p_alert.add_argument("--setup", action="store_true", help="alert when a setup triggers")
    p_alert.add_argument("--tf", default=None, help="timeframe for a setup alert")
    p_alert.add_argument("--note", default="")
    p_alert.add_argument("--list", action="store_true", help="list active alerts")
    p_alert.add_argument("--all", action="store_true", help="with --list, include triggered")
    p_alert.add_argument("--check", action="store_true", help="evaluate alerts now")
    p_alert.add_argument("--remove", default=None, metavar="ID", help="remove an alert by id")
    p_alert.set_defaults(func=cmd_alert)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
