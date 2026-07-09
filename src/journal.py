"""Layer 7 — journal & review.

Append-only persistence (JSONL in /data), the SOURCE OF TRUTH for:
  - the PortfolioState the guards consume (`portfolio_state`), and
  - the planned-vs-executed pairs adherence scores,
plus the equity curve and the `review` analytics.

A TradeRecord spans STAGED → OPEN → CLOSED/CANCELLED, tagged dry-run vs live. Each
state change appends a full snapshot; `load_records` keeps the latest per id. Only
CLOSED trades feed realised statistics.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone

import numpy as np

from . import adherence as adh
from . import expectancy as ex
from . import guards as gd
from .config import DATA_DIR, JournalConfig

STAGED = "staged"
OPEN = "open"
CLOSED = "closed"
CANCELLED = "cancelled"

_JOURNAL_PATH = DATA_DIR / "journal.jsonl"


@dataclass
class TradeRecord:
    id: str
    timestamp: str                      # ISO, when staged
    market: str                         # spot / usdm
    symbol: str
    side: str                           # long / short
    setup: str = ""
    grade: str = "B"
    status: str = STAGED
    mode: str = "dry-run"               # dry-run / live
    group: str = "indep"
    # plan
    entry_planned: float = 0.0
    stop_planned: float = 0.0
    targets: list = field(default_factory=list)
    risk_pct_planned: float = 0.0
    risk_amount: float = 0.0
    edge_r: float | None = None
    # setup mechanics (needed to paper-fill and to manage by the plan, Layer 10)
    tf: str = ""                        # trigger timeframe the setup lives on
    entry_type: str = "market"          # market / limit / stop
    invalidation: float = 0.0           # pre-entry level that voids a STAGED setup
    leverage: float = 1.0
    liquidation_price: float | None = None
    # management state (Layer 10) — the OPEN position, run by the SAME machine as the backtest
    current_stop: float | None = None   # live stop (→ breakeven after TP1, may trail); never looser
    remaining_fraction: float = 1.0     # position fraction still open
    tp1_filled: bool = False
    locked_r: float = 0.0               # R already booked from partial exits
    managed_at: str = ""                # ISO; bars after this are unprocessed by manage
    broker_order_ids: dict = field(default_factory=dict)  # (live, step 18) kind→broker order id
    # context snapshot
    regime: str = ""
    bias: str = ""
    tide: str = ""
    # outcome
    entry_actual: float | None = None
    stop_actual: float | None = None    # FINAL stop after any moves
    exit_price: float | None = None
    realized_r: float | None = None
    pnl_usd: float | None = None
    fees_usd: float | None = None
    closed_at: str | None = None
    # discipline flags
    in_cooldown: bool = False
    guard_blocked: bool = False
    notes: list = field(default_factory=list)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def journal_path():
    return _JOURNAL_PATH


# --------------------------------------------------------------------------- #
# Storage
# --------------------------------------------------------------------------- #
def append(record: TradeRecord, path=None) -> None:
    p = path or _JOURNAL_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(json.dumps(asdict(record)) + "\n")


def load_records(path=None) -> list[TradeRecord]:
    """Load all records, keeping the LATEST snapshot per id (append-only history)."""
    p = path or _JOURNAL_PATH
    if not p.exists():
        return []
    known = {f.name for f in fields(TradeRecord)}
    latest: dict[str, TradeRecord] = {}
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except ValueError:
            continue
        rec = TradeRecord(**{k: v for k, v in d.items() if k in known})
        latest[rec.id] = rec
    # preserve first-seen order
    order: list[str] = []
    seen = set()
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            i = json.loads(line)["id"]
        except (ValueError, KeyError):
            continue
        if i not in seen:
            seen.add(i)
            order.append(i)
    return [latest[i] for i in order if i in latest]


# --------------------------------------------------------------------------- #
# Lifecycle helpers (called by stage / manage later)
# --------------------------------------------------------------------------- #
def record_staged(*, market, symbol, side, setup, grade, group, entry, stop, targets,
                  risk_pct, risk_amount, edge_r=None, regime="", bias="", tide="",
                  tf="", entry_type="market", invalidation=0.0, leverage=1.0,
                  liquidation_price=None, mode="dry-run", in_cooldown=False,
                  guard_blocked=False, profile=None, path=None) -> str:
    rec = TradeRecord(
        id=uuid.uuid4().hex[:12], timestamp=_now(), market=market, symbol=symbol, side=side,
        setup=setup, grade=grade, status=STAGED, mode=mode, group=group,
        entry_planned=entry, stop_planned=stop, targets=list(targets),
        risk_pct_planned=risk_pct, risk_amount=risk_amount, edge_r=edge_r,
        tf=tf, entry_type=entry_type, invalidation=invalidation, leverage=leverage,
        liquidation_price=liquidation_price,
        regime=regime, bias=bias, tide=tide, in_cooldown=in_cooldown, guard_blocked=guard_blocked)
    if profile:                                  # attribution: which risk posture staged this
        rec.notes.append(f"risk profile: {profile}")
    append(rec, path)
    return rec.id


def _update(trade_id: str, path=None, **changes) -> TradeRecord | None:
    recs = {r.id: r for r in load_records(path)}
    rec = recs.get(trade_id)
    if rec is None:
        return None
    for k, v in changes.items():
        setattr(rec, k, v)
    append(rec, path)
    return rec


def record_fill(trade_id, *, entry_actual, path=None):
    return _update(trade_id, path=path, status=OPEN, entry_actual=entry_actual)


def record_open(trade_id, *, entry_actual, current_stop, managed_at=None, mode=None,
                broker_order_ids=None, path=None):
    """STAGED → OPEN (a paper-fill in dry-run, or a real fill under --live).

    Initialises the management state the SAME machine then runs against."""
    changes = dict(status=OPEN, entry_actual=entry_actual, current_stop=current_stop,
                   remaining_fraction=1.0, tp1_filled=False, locked_r=0.0,
                   managed_at=managed_at or _now())
    if mode is not None:
        changes["mode"] = mode
    if broker_order_ids is not None:
        changes["broker_order_ids"] = dict(broker_order_ids)
    return _update(trade_id, path=path, **changes)


def record_manage(trade_id, *, current_stop=None, remaining_fraction=None, tp1_filled=None,
                  locked_r=None, managed_at=None, broker_order_ids=None, note=None, path=None):
    """Persist an OPEN position's evolving management state (stop move / partial)."""
    changes: dict = {}
    if current_stop is not None:
        changes["current_stop"] = current_stop
    if remaining_fraction is not None:
        changes["remaining_fraction"] = remaining_fraction
    if tp1_filled is not None:
        changes["tp1_filled"] = tp1_filled
    if locked_r is not None:
        changes["locked_r"] = locked_r
    if managed_at is not None:
        changes["managed_at"] = managed_at
    if broker_order_ids is not None:
        changes["broker_order_ids"] = dict(broker_order_ids)
    rec = _update(trade_id, path=path, **changes)
    if rec is not None and note:
        rec.notes.append(note)
        append(rec, path)
    return rec


def record_cancel(trade_id, *, reason=None, path=None):
    rec = _update(trade_id, path=path, status=CANCELLED)
    if rec is not None and reason:
        rec.notes.append(reason)
        append(rec, path)
    return rec


def record_close(trade_id, *, exit_price, realized_r, pnl_usd=None, fees_usd=None,
                 stop_actual=None, path=None):
    changes = dict(status=CLOSED, exit_price=exit_price, realized_r=realized_r,
                   pnl_usd=pnl_usd, fees_usd=fees_usd, closed_at=_now())
    if stop_actual is not None:
        changes["stop_actual"] = stop_actual
    return _update(trade_id, path=path, **changes)


# --------------------------------------------------------------------------- #
# Account state (feeds guards)
# --------------------------------------------------------------------------- #
def _pnl(rec: TradeRecord) -> float:
    if rec.pnl_usd is not None:
        return rec.pnl_usd
    if rec.realized_r is not None and rec.risk_amount:
        return rec.realized_r * rec.risk_amount
    return 0.0


def portfolio_state(records=None, cfg: JournalConfig | None = None, now=None) -> "gd.PortfolioState":
    cfg = cfg or JournalConfig()
    records = records if records is not None else load_records()
    closed = [r for r in records if r.status == CLOSED]
    equity = cfg.starting_equity + sum(_pnl(r) for r in closed)
    open_positions = [gd.OpenPosition(r.symbol, r.group, r.risk_amount, r.side)
                      for r in records if r.status == OPEN]

    today = (now or datetime.now(timezone.utc)).date()
    realized_r_today = 0.0
    for r in closed:
        if r.closed_at and r.realized_r is not None:
            try:
                if datetime.fromisoformat(r.closed_at).date() == today:
                    realized_r_today += r.realized_r
            except ValueError:
                pass

    closed_sorted = sorted([r for r in closed if r.closed_at],
                           key=lambda r: r.closed_at)
    streak = 0
    for r in reversed(closed_sorted):
        if (r.realized_r or 0.0) < 0:
            streak += 1
        else:
            break
    last_loss = None
    for r in closed_sorted:
        if (r.realized_r or 0.0) < 0:
            try:
                last_loss = datetime.fromisoformat(r.closed_at)
            except ValueError:
                pass

    # current drawdown from the peak of the chronological $ equity curve (for risk-scaling)
    eq = cfg.starting_equity
    peak = eq
    for r in closed_sorted:
        eq += _pnl(r)
        peak = max(peak, eq)
    drawdown_pct = (peak - eq) / peak * 100.0 if peak > 0 else 0.0

    # new trades taken today (by staged timestamp; excludes cancelled) — over-trading guard
    trades_today = 0
    for r in records:
        if r.status == CANCELLED or not r.timestamp:
            continue
        try:
            if datetime.fromisoformat(r.timestamp).date() == today:
                trades_today += 1
        except ValueError:
            pass

    return gd.PortfolioState(equity=equity, open_positions=open_positions,
                             realized_r_today=realized_r_today,
                             consecutive_losses=streak, last_loss_time=last_loss,
                             drawdown_pct=drawdown_pct, trades_today=trades_today)


def equity_curve_r(records) -> list[float]:
    closed = sorted([r for r in records if r.status == CLOSED and r.closed_at and r.realized_r is not None],
                    key=lambda r: r.closed_at)
    out, cum = [], 0.0
    for r in closed:
        cum += r.realized_r
        out.append(cum)
    return out


# --------------------------------------------------------------------------- #
# Review
# --------------------------------------------------------------------------- #
@dataclass
class ReviewReport:
    n_closed: int
    starting_equity: float
    equity: float
    overall: ex.Stats
    by_setup: dict
    by_regime: dict
    by_grade: dict
    adherence: adh.AdherenceReport
    max_drawdown_r: float
    enough_sample: bool
    verdicts: list = field(default_factory=list)


def _bucket(closed, key) -> dict:
    groups: dict = {}
    for r in closed:
        groups.setdefault(key(r), []).append(r.realized_r)
    return {k: ex.summarize(v) for k, v in groups.items()}


def build_review(records=None, cfg: JournalConfig | None = None) -> ReviewReport:
    cfg = cfg or JournalConfig()
    records = records if records is not None else load_records()
    closed = [r for r in records if r.status == CLOSED and r.realized_r is not None]
    rs = [r.realized_r for r in closed]
    overall = ex.summarize(rs)
    by_setup = _bucket(closed, lambda r: r.setup or "?")
    by_regime = _bucket(closed, lambda r: r.regime or "?")
    by_grade = _bucket(closed, lambda r: r.grade or "?")

    adh_records = [adh.AdherenceRecord(
        planned_risk_pct=r.risk_pct_planned, actual_risk_pct=r.risk_pct_planned,
        planned_stop=r.stop_planned, actual_stop=(r.stop_actual if r.stop_actual is not None else r.stop_planned),
        entry=(r.entry_actual if r.entry_actual is not None else r.entry_planned),
        side=r.side, grade=r.grade, guard_blocked=r.guard_blocked, in_cooldown=r.in_cooldown)
        for r in closed]
    adherence = adh.evaluate_adherence(adh_records)

    rs_chrono = [r.realized_r for r in sorted(closed, key=lambda r: r.closed_at or "")]
    max_dd = ex._max_drawdown_r(np.asarray(rs_chrono)) if rs_chrono else 0.0

    enough = len(closed) >= cfg.min_sample
    verdicts: list[str] = []
    if not enough:
        verdicts.append(f"too few closed trades ({len(closed)} < {cfg.min_sample}) — no verdicts yet")
    else:
        proven = {k: s for k, s in by_setup.items() if s.n >= cfg.min_bucket_sample}
        if proven:
            best = max(proven.items(), key=lambda kv: kv[1].expectancy)
            worst = min(proven.items(), key=lambda kv: kv[1].expectancy)
            verdicts.append(f"your edge is in '{best[0]}' ({best[1].expectancy:+.2f}R over {best[1].n})")
            if worst[1].expectancy < 0:
                verdicts.append(f"stop taking '{worst[0]}' ({worst[1].expectancy:+.2f}R over {worst[1].n})")
        if adherence.adherence_pct < 90 and adherence.n:
            verdicts.append(f"discipline: adherence {adherence.adherence_pct:.0f}% — tighten it")

    start = cfg.starting_equity
    equity = start + sum(_pnl(r) for r in closed)
    return ReviewReport(n_closed=len(closed), starting_equity=start, equity=equity,
                        overall=overall, by_setup=by_setup, by_regime=by_regime,
                        by_grade=by_grade, adherence=adherence, max_drawdown_r=max_dd,
                        enough_sample=enough, verdicts=verdicts)
