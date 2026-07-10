"""SENIORTRADE cockpit — local, read-only web API (UI Phase A).

A thin FastAPI layer over the SAME engine the CLI uses (zero statistical logic here —
every number the UI shows comes from the identical gated functions). Design contract:

  - LOCALHOST ONLY (hard-coded 127.0.0.1): API keys and data never leave this machine.
  - READ-ONLY: Phase A exposes no order/stage/manage endpoint at all. The typed-CONFIRM
    dry-run flow stays in the CLI until Phase B; live stays CLI-only (Phase D policy).
  - The slow board scan runs in a BACKGROUND thread; the UI polls its state. Everything
    else (analyze/backtest) is synchronous with a spinner client-side.

Run:  python3 -m src.api          (or: python3 -m src.cli ui)
Then open http://127.0.0.1:8484
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from . import analysis as an
from . import backtest as bt
from . import edge_score as es
from . import journal as jn
from . import risk as rk
from .config import DATA_DIR, Market, PROFILE_LABELS, Settings, apply_profile, apply_tier, load_settings, strictness_fluke_note

app = FastAPI(title="SENIORTRADE cockpit", docs_url=None, redoc_url=None)

_WEBUI = Path(__file__).resolve().parent / "webui" / "index.html"

# ---------------------------------------------------------------------------- #
# server state — the board scan is slow (1–2 min), so it runs in a background
# thread and the UI polls. One scan at a time; state survives between polls.
# ---------------------------------------------------------------------------- #
_STATE: dict = {"board": None, "board_ts": 0.0, "scanning": False, "error": None}
_LOCK = threading.Lock()


def _settings():
    """Same precedence as the CLI: defaults → tier(env) → profile(env) → env overrides."""
    import os
    base = apply_tier(Settings(), os.getenv("TIER") or None)
    base = apply_profile(base, os.getenv("RISK_PROFILE") or None)
    return load_settings(base=base)


def _market(name: str | None) -> Market:
    return Market.SPOT if (name or "").lower() == "spot" else Market.USDM


def _scan_worker(market: Market, refresh: bool) -> None:
    try:
        s = _settings()
        context, opps, watch, beta = es.scan(market, s, refresh=refresh)
        with _LOCK:
            _STATE["board"] = {
                "posture": getattr(context, "posture", None),
                "context_note": getattr(context, "note", ""),
                "alpha": [asdict(o) for o in opps],
                "beta": beta,
                "watchlist": watch,
                "market": market.value,
            }
            _STATE["board_ts"] = time.time()
            _STATE["error"] = None
    except Exception as exc:  # noqa: BLE001 — surface, never crash the server
        with _LOCK:
            _STATE["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        with _LOCK:
            _STATE["scanning"] = False


@app.get("/")
def index():
    return FileResponse(_WEBUI, media_type="text/html")


@app.get("/api/status")
def status():
    s = _settings()
    watch = {"running": False}
    try:
        ws = json.loads((DATA_DIR / "watch_status.json").read_text())
        pid = (DATA_DIR / "watch.pid")
        watch = {"running": pid.exists(), **({k: ws[k] for k in ("last_scan", "board") if k in ws})}
    except Exception:  # noqa: BLE001
        pass
    cache = es.load_universe(Market.USDM, s.edge.tf, ttl_hours=24 * 365) or {}
    m = s.risk_mgmt
    return {
        "profile": s.profile,
        "profile_label": PROFILE_LABELS.get(s.profile or "", ""),
        "tier": s.edge.tf,
        "equity": s.risk.account_equity,
        "risk_pct": s.risk.risk_pct,
        "stages": {"dd_scale": m.dd_scale_enabled, "vol_target": m.vol_target_enabled,
                   "kelly": m.kelly_enabled, "edge_scaled": m.edge_scaled},
        "strictness_note": strictness_fluke_note(s.edge.sig_z),
        "sig_z": s.edge.sig_z,
        "watch": watch,
        "cache": {"ts": cache.get("timestamp"), "coins": len(cache.get("coins") or []),
                  "tf": cache.get("tf")},
        "mode": "DRY-RUN (read-only UI — orders only via CLI)",
    }


@app.get("/api/board")
def board():
    with _LOCK:
        return {"scanning": _STATE["scanning"], "ts": _STATE["board_ts"],
                "error": _STATE["error"], "board": _STATE["board"]}


@app.post("/api/board/refresh")
def board_refresh(market: str | None = None, refresh: bool = False):
    with _LOCK:
        if _STATE["scanning"]:
            return JSONResponse({"started": False, "reason": "scan already running"}, status_code=409)
        _STATE["scanning"] = True
    t = threading.Thread(target=_scan_worker, args=(_market(market), refresh), daemon=True)
    t.start()
    return {"started": True}


@app.get("/api/analyze/{symbol:path}")
def analyze(symbol: str, market: str | None = None):
    s = _settings()
    mkt = _market(market)
    try:
        r = an.analyze(mkt, symbol, s, tf_trigger=s.analysis.tf_trigger)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"{type(exc).__name__}: {exc}")
    regime = r.structure.state.trend if (r.structure and r.structure.state) else ""
    payload = {
        "symbol": r.symbol, "last": r.last_price, "bias": r.bias, "confluence": r.confluence,
        "htf_trend": r.htf_trend, "location": r.location, "freshness": r.freshness,
        "posture": r.posture, "regime": regime, "archetype": (r.context or {}).get("arch"),
        "funding": ({"rate": r.funding.rate, "signal": r.funding.signal} if r.funding else None),
        "oi": ({"trend": r.oi.trend} if r.oi else None),
        "lenses": [{"name": L.name, "read": L.direction, "score": L.score, "note": L.note}
                   for L in (r.lenses or [])],
        "conflicts": list(r.conflicts or []),
        "setups": [{"setup": t.setup, "side": t.direction, "grade": t.grade, "entry": t.entry,
                    "stop": t.stop, "targets": list(t.targets), "why": "; ".join(t.reasons or [])}
                   for t in (r.setups or [])],
        "plan": None, "edge": None, "archetype_evidence": None, "why": None,
        "sizing": {"risk_pct": r.risk_pct, "notes": list(r.risk_notes or []),
                   "inputs": dict(r.risk_inputs or {})},
    }
    if r.plan is not None:
        p = r.plan
        payload["plan"] = {"risk_budget": p.risk_budget, "risk_actual": p.risk_actual,
                           "total_cost": p.total_cost, "size": p.size, "notional": p.notional,
                           "gross_rr": p.gross_rr, "net_rr": p.net_rr,
                           "leverage": getattr(p, "leverage", None),
                           "liquidation": getattr(p, "liquidation_price", None)}
    if r.setups:
        top = r.setups[0]
        verdict = es.lookup_pooled_verdict(mkt, s.analysis.tf_trigger, top.setup, regime,
                                           symbol=r.symbol,
                                           context=es.resolve_context(r.context, top.direction))
        level, msg = es.edge_gate(verdict, s.edge)
        rb = (verdict or {}).get("robustness")
        payload["edge"] = {"level": level, "msg": msg, "robustness": rb,
                           "setup": top.setup, "regime": regime}
        payload["archetype_evidence"] = es.archetype_evidence(
            mkt, s.analysis.tf_trigger, top.setup, (r.context or {}).get("arch"))
        reg = (verdict or {}).get("regime") if verdict else None
        tier = None
        if reg and reg.get("n"):
            sc = es.statistical_confidence(reg, (verdict or {}).get("null"),
                                           (verdict or {}).get("fold_consistency", 0.0), s.edge)
            tier = es._conf_tier(sc["confidence"], s.edge)
        payload["why"] = es.explain_signal(
            direction=top.direction, lenses=r.lenses, edge_level=level, edge_msg=msg,
            regime=regime, freshness=r.freshness, posture=r.posture,
            context_label=(reg.get("context_label") if reg else None), confidence_tier=tier)
    return payload


@app.get("/api/backtest/{symbol:path}")
def backtest(symbol: str, market: str | None = None):
    s = _settings()
    try:
        profiles, n_candles, quality = bt.run_backtest(_market(market), symbol, s,
                                                       s.analysis.tf_trigger)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"{type(exc).__name__}: {exc}")
    out = {"symbol": symbol, "n_candles": n_candles, "requested": s.backtest.candle_limit,
           "quality": quality, "setups": {}}
    for name, p in profiles.items():
        d = asdict(p)
        rb = es.robustness(p.overall.expectancy, p.fold_consistency, p.bootstrap_low,
                           p.bootstrap_high, p.recent_expectancy)
        d["robustness"] = rb
        out["setups"][name] = d
    return out


@app.get("/api/features")
def features(market: str | None = None):
    s = _settings()
    return es.load_feature_importance(_market(market), s.edge.tf) or {}


@app.get("/api/positions")
def positions():
    recs = jn.load_records()
    open_like = [r for r in recs if r.status in ("staged", "open", "pending", "STAGED", "OPEN", "PENDING")]
    return {"records": [asdict(r) for r in open_like[-40:]]}


@app.get("/api/journal")
def journal_view():
    recs = jn.load_records()
    closed = [r for r in recs if (r.realized_r is not None)]
    rs = [r.realized_r for r in closed]
    wins = [x for x in rs if x > 0]
    return {
        "closed": [asdict(r) for r in closed[-60:]],
        "stats": {"n": len(rs), "win_rate": (len(wins) / len(rs)) if rs else 0.0,
                  "avg_r": (sum(rs) / len(rs)) if rs else 0.0, "total_r": sum(rs)},
    }


def main() -> None:
    import uvicorn
    # LOCALHOST ONLY — the cockpit (and your keys) never leave this machine.
    uvicorn.run(app, host="127.0.0.1", port=8484, log_level="warning")


if __name__ == "__main__":
    main()
