"""Universe Studio — the USER chooses WHAT to scan; the gates decide what's true.

Commercial-cycle principle: the truth is the only permitted friction. Everything
around it — which coins, which slice of the market — is the user's to command.
A lens ONLY selects candidates; every honesty gate downstream (liquidity/history/
volatility screens, the null baseline, significance, effective-n) runs unchanged
inside whatever universe the user picks. A lens can surface a coin; it can never
manufacture a verdict for it.

Lenses (all Binance-native except mcap_band, which uses a cached free CoinGecko map):
  top_volume   {top_n}                      — the classic top-N by 24h quote volume (default)
  volume_band  {min_usd, max_usd, top_n}    — mid/low-liquidity slices
  movers       {direction: up|down|both, top_n} — biggest 24h % movers (volume floor still applies)
  new_listings {days, top_n}                — USD-M contracts onboarded within N days
  custom       {symbols: [...]}             — the user's own list, verbatim
  mcap_band    {min_usd, max_usd, top_n}    — market-cap slice (CoinGecko, cached 24h)
"""
from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

from .config import DATA_DIR, Market

LENSES = ("top_volume", "volume_band", "movers", "new_listings", "custom", "mcap_band")

_MCAP_CACHE = DATA_DIR / "mcap_cache.json"
_MCAP_TTL = 24 * 3600.0
_MCAP_URL = ("https://api.coingecko.com/api/v3/coins/markets"
             "?vs_currency=usd&order=market_cap_desc&per_page=250&page={page}")


def describe(lens: dict | None) -> str:
    """One-line human label for echo/board headers."""
    k = (lens or {}).get("kind", "top_volume")
    p = lens or {}
    if k == "top_volume":
        return f"top {p.get('top_n', 100)} by 24h volume"
    if k == "volume_band":
        return f"24h volume ${p.get('min_usd', 0):,.0f}–${p.get('max_usd', 0):,.0f}"
    if k == "movers":
        return f"top {p.get('top_n', 50)} movers ({p.get('direction', 'both')}, 24h)"
    if k == "new_listings":
        return f"listed within {p.get('days', 30)} days"
    if k == "custom":
        return f"custom list ({len(p.get('symbols') or [])} coins)"
    if k == "mcap_band":
        return f"market cap ${p.get('min_usd', 0):,.0f}–${p.get('max_usd', 0):,.0f}"
    return k


def _rows_from(tickers: dict, symbols: list) -> list:
    rows = []
    for sym in symbols:
        t = tickers.get(sym)
        if not t or t.get("quoteVolume") is None or t.get("last") is None:
            continue
        rows.append((sym, float(t["quoteVolume"]), float(t["last"])))
    return rows


def _mcap_map() -> dict:
    """symbol(base, upper) -> market cap USD. Cached 24h; top ~1000 coins. Best-effort:
    an offline/rate-limited fetch degrades to the last cache (or {}), never a fake."""
    try:
        cached = json.loads(_MCAP_CACHE.read_text())
        if time.time() - cached.get("ts", 0) < _MCAP_TTL:
            return cached.get("map", {})
    except (OSError, ValueError):
        cached = {}
    out: dict = {}
    try:
        for page in (1, 2, 3, 4):
            req = urllib.request.Request(_MCAP_URL.format(page=page),
                                         headers={"User-Agent": "seniortrade/1.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                for row in json.loads(resp.read().decode()):
                    symu = str(row.get("symbol", "")).upper()
                    mcap = row.get("market_cap")
                    if symu and mcap and symu not in out:      # first (largest) wins on collisions
                        out[symu] = float(mcap)
        _MCAP_CACHE.parent.mkdir(parents=True, exist_ok=True)
        _MCAP_CACHE.write_text(json.dumps({"ts": time.time(), "map": out}))
        return out
    except Exception:  # noqa: BLE001
        return cached.get("map", {})


def select(ex, tickers: dict, eligible: list, cfg, lens: dict | None,
           market: Market) -> tuple[list, str]:
    """Resolve a lens to [(symbol, quote_volume, last)] candidates + a note. The MINIMUM
    volume floor always applies (an untradeable coin is a disservice, not freedom) —
    except for `custom`, where the user's explicit pick is honored and the downstream
    screens/verdicts tell the honest story."""
    lens = lens or {"kind": "top_volume"}
    kind = lens.get("kind", "top_volume")
    if kind not in LENSES:
        raise ValueError(f"unknown universe lens {kind!r} — choose from {LENSES}")
    floor = cfg.min_quote_volume

    if kind == "custom":
        want = [s.strip() for s in (lens.get("symbols") or []) if s.strip()]
        known = [s for s in want if s in tickers or s in eligible]
        rows = _rows_from(tickers, known)
        note = (f"custom list ({len(rows)} of {len(want)} resolved)" if len(rows) < len(want)
                else describe(lens))                   # be explicit when a symbol didn't resolve
        return rows, note

    rows = _rows_from(tickers, eligible)

    if kind == "top_volume":
        rows = [r for r in rows if r[1] >= floor]
        rows.sort(key=lambda r: r[1], reverse=True)
        return rows[: int(lens.get("top_n", cfg.top_n))], describe(lens)

    if kind == "volume_band":
        lo = float(lens.get("min_usd", floor))
        hi = float(lens.get("max_usd", float("inf")))
        rows = [r for r in rows if lo <= r[1] <= hi and r[1] >= min(floor, lo)]
        rows.sort(key=lambda r: r[1], reverse=True)
        return rows[: int(lens.get("top_n", cfg.top_n))], describe(lens)

    if kind == "movers":
        direction = lens.get("direction", "both")
        scored = []
        for sym, qv, last in rows:
            if qv < floor:
                continue
            pct = tickers.get(sym, {}).get("percentage")
            if pct is None:
                continue
            pct = float(pct)
            if direction == "up" and pct <= 0:
                continue
            if direction == "down" and pct >= 0:
                continue
            scored.append(((sym, qv, last), abs(pct)))
        scored.sort(key=lambda x: -x[1])
        return [r for r, _ in scored[: int(lens.get("top_n", 50))]], describe(lens)

    if kind == "new_listings":
        days = float(lens.get("days", 30))
        cutoff = (time.time() - days * 86400) * 1000
        picked = []
        for sym, qv, last in rows:
            if qv < floor:
                continue
            onboard = (ex.markets.get(sym, {}).get("info") or {}).get("onboardDate")
            if onboard and float(onboard) >= cutoff:
                picked.append((sym, qv, last, float(onboard)))
        picked.sort(key=lambda r: -r[3])                      # newest first
        return [(s, q, l) for s, q, l, _ in picked[: int(lens.get("top_n", 50))]], describe(lens)

    # mcap_band
    mmap = _mcap_map()
    if not mmap:
        raise ValueError("market-cap data unavailable right now (CoinGecko unreachable) — "
                         "try another lens")
    lo = float(lens.get("min_usd", 0))
    hi = float(lens.get("max_usd", float("inf")))
    picked = []
    for sym, qv, last in rows:
        if qv < floor:
            continue
        base = sym.split("/")[0]
        mcap = mmap.get(base.upper())
        if mcap is not None and lo <= mcap <= hi:
            picked.append((sym, qv, last, mcap))
    picked.sort(key=lambda r: -r[3])
    return [(s, q, l) for s, q, l, _ in picked[: int(lens.get("top_n", cfg.top_n))]], describe(lens)
