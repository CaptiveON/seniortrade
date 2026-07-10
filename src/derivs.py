"""Derivatives history — funding (A1) and OI / long-short snapshots (A3).

ANALYTICSAUDIT roadmap items A1+A3. Two very different data situations, one module:

  FUNDING (A1): Binance serves the FULL settled-funding history → fetch once
  (paginated), cache to disk, top-up incrementally. Immediately backtestable.

  OI / LONG-SHORT RATIOS (A3): Binance retains only ~30 DAYS → every uncollected
  day is lost forever. `collect_snapshots` appends-dedupes the current window into
  a local store on every scan; the history GROWS from today. Features built on it
  stay honest: bars older than the store simply carry None (never fabricated).

PIT DISCIPLINE (non-negotiable): a bar may only see events with ts <= its own
open — `align_*` enforces it. Bucket thresholds use TRAILING windows only.
Everything downstream goes through the SAME gates as every other context feature
(Bonferroni + OOS + effective-n); nothing here touches a verdict directly.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

from .config import DATA_DIR, Market

_DIR = DATA_DIR / "derivs"

# funding z-score buckets (z of the CURRENT settled rate vs the trailing window)
_FUND_WINDOW = 90            # ~30 days of 8h settlements
_FUND_MIN = 20               # need this many trailing events before bucketing (else None)
# OI 24h-change buckets
_OI_EXPAND_PCT = 5.0
_OI_PERIODS = {"4h": 6, "1h": 24, "1d": 1}   # snapshots per 24h at the stored period


def _path(market: Market, symbol: str, metric: str) -> Path:
    safe = symbol.replace("/", "_").replace(":", "-")
    return _DIR / f"{market.value}_{safe}_{metric}.json"


def _load(path: Path) -> list:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return []


def _save(path: Path, rows: list) -> None:
    _DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows))


def _merge(old: list, new: list) -> list:
    """Append-dedupe by timestamp (ms), ascending — the collector's one invariant."""
    by_ts = {int(r[0]): r for r in old}
    for r in new:
        by_ts[int(r[0])] = r
    return [by_ts[t] for t in sorted(by_ts)]


# ---------------------------------------------------------------------------- #
# A1 — FUNDING: full history, cached + incrementally topped up
# ---------------------------------------------------------------------------- #
def funding_series(ex, market: Market, symbol: str, since_ms: int | None = None) -> list:
    """[[ts_ms, rate], ...] ascending — settled funding events. Fetches the full history
    the first time (paginated), then only the tail beyond the cache. Failures degrade to
    whatever is cached (features go None, never invented)."""
    path = _path(market, symbol, "funding")
    rows = _load(path)
    if since_ms and rows and int(rows[0][0]) > since_ms:
        start = since_ms                          # BACKFILL: cache starts later than needed
    else:
        start = (int(rows[-1][0]) + 1) if rows else (since_ms or 0)
    try:
        while True:
            batch = ex.fetch_funding_rate_history(symbol, since=start or None, limit=1000)
            if not batch:
                break
            new = [[int(b["timestamp"]), float(b["fundingRate"])] for b in batch
                   if b.get("fundingRate") is not None]
            rows = _merge(rows, new)
            nxt = int(batch[-1]["timestamp"]) + 1
            if nxt <= start or len(batch) < 100:
                break
            start = nxt
        _save(path, rows)
    except Exception:  # noqa: BLE001 — offline/rate-limited → serve the cache
        pass
    return rows


def fund_bucket(z: float) -> str:
    if z >= 2.0:
        return "ext+"
    if z >= 1.0:
        return "hi+"
    if z <= -2.0:
        return "ext-"
    if z <= -1.0:
        return "hi-"
    return "mid"


def align_funding(bar_ts: list, events: list) -> tuple[list, list]:
    """Per-bar (bucket, latest_settled_rate) with strict PIT: a bar sees only events
    settled AT OR BEFORE its own timestamp. Bucket = z of the latest rate vs the
    TRAILING _FUND_WINDOW events (None until _FUND_MIN have accrued)."""
    ctx: list = []
    rates: list = []
    j = 0
    window: list = []
    for ts in bar_ts:
        while j < len(events) and events[j][0] <= ts:
            window.append(float(events[j][1]))
            if len(window) > _FUND_WINDOW:
                window.pop(0)
            j += 1
        if not window:
            ctx.append(None)
            rates.append(None)
            continue
        rate = window[-1]
        rates.append(rate)
        if len(window) < _FUND_MIN:
            ctx.append(None)
            continue
        mean = sum(window) / len(window)
        var = sum((x - mean) ** 2 for x in window) / max(1, len(window) - 1)
        sd = math.sqrt(var)
        ctx.append(fund_bucket((rate - mean) / sd) if sd > 0 else "mid")
    return ctx, rates


# ---------------------------------------------------------------------------- #
# A3 — OI / LONG-SHORT: 30-day exchange retention → collect NOW, grow forever
# ---------------------------------------------------------------------------- #
_SNAPSHOT_ENDPOINTS = {
    "oi": ("fapiDataGetOpenInterestHist", "sumOpenInterest"),
    "ls_top_pos": ("fapiDataGetTopLongShortPositionRatio", "longShortRatio"),
    "ls_top_acct": ("fapiDataGetTopLongShortAccountRatio", "longShortRatio"),
    "ls_global": ("fapiDataGetGlobalLongShortAccountRatio", "longShortRatio"),
    "taker_ratio": ("fapiDataGetTakerlongshortRatio", "buySellRatio"),
}


def collect_snapshots(ex, market: Market, symbols: list, period: str = "4h") -> dict:
    """Persist the exchange's current ~30-day window for OI + all L/S ratios, per symbol.
    Append-dedupe: calling this on every scan builds an ever-growing local history.
    Returns {metric: rows_added}. USD-M only; best-effort per endpoint."""
    if market is not Market.USDM:
        return {}
    added: dict = {}
    for symbol in symbols:
        try:
            raw_id = ex.market(symbol)["id"]
        except Exception:  # noqa: BLE001
            continue
        for metric, (method, field) in _SNAPSHOT_ENDPOINTS.items():
            try:
                fn = getattr(ex, method, None)
                if fn is None:
                    continue
                batch = fn({"symbol": raw_id, "period": period, "limit": 500})
                new = [[int(r["timestamp"]), float(r[field])] for r in batch if r.get(field) is not None]
                path = _path(market, symbol, f"{metric}_{period}")
                rows = _load(path)
                merged = _merge(rows, new)
                _save(path, merged)
                added[metric] = added.get(metric, 0) + (len(merged) - len(rows))
            except Exception:  # noqa: BLE001 — one endpoint failing must not stop collection
                continue
    return added


def oi_series(market: Market, symbol: str, period: str = "4h") -> list:
    """Collected OI rows [[ts_ms, sum_oi], ...] — only what we have persisted locally."""
    return _load(_path(market, symbol, f"oi_{period}"))


def align_oi(bar_ts: list, rows: list, period: str = "4h") -> list:
    """Per-bar OI bucket vs ~24h earlier (PIT: snapshots at/before the bar only).
    'expand' / 'contract' beyond ±_OI_EXPAND_PCT, else 'flat'; None without enough history —
    honest: bars older than the local store simply have no OI feature."""
    lookback = _OI_PERIODS.get(period, 6)
    out: list = []
    j = 0
    seen: list = []
    for ts in bar_ts:
        while j < len(rows) and rows[j][0] <= ts:
            seen.append(rows[j])
            j += 1
        if len(seen) <= lookback:
            out.append(None)
            continue
        cur, prev = float(seen[-1][1]), float(seen[-1 - lookback][1])
        if prev <= 0:
            out.append(None)
            continue
        chg = (cur / prev - 1.0) * 100.0
        out.append("expand" if chg >= _OI_EXPAND_PCT else "contract" if chg <= -_OI_EXPAND_PCT else "flat")
    return out
