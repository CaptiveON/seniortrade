"""SENIORTRADE cockpit — local, read-only web API (UI Phase A).

A thin FastAPI layer over the SAME engine the CLI uses (zero statistical logic here —
every number the UI shows comes from the identical gated functions). Design contract:

  - LOCALHOST ONLY (hard-coded 127.0.0.1): API keys and data never leave this machine.
  - DRY-RUN ONLY: Phase B adds stage/manage — two-step (fresh-preview ticket → typed
    CONFIRM, enforced SERVER-side) and strictly paper (mode="dry-run"). There is NO live
    route and no import of the live module; live stays CLI-only (Phase D policy).
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

import secrets
from datetime import datetime, timezone

from . import analysis as an
from . import backtest as bt
from . import data_fetch as dfetch
from . import edge_score as es
from . import journal as jn
from . import monitor as mon
from . import order as od
from .config import DATA_DIR, Market, PROFILE_LABELS, Settings, apply_profile, apply_tier, load_settings, strictness_fluke_note
from .guards import check_trade

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
        "mode": "DRY-RUN (web stage/manage = paper + typed CONFIRM · LIVE only via CLI)",
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


# ---------------------------------------------------------------------------- #
# PHASE B — dry-run interactions. Two-step, typed-CONFIRM, server-enforced:
#   1. /preview runs the SAME checklist as the CLI (guards → edge gate → drift)
#      on FRESH data and returns a short-lived one-time TICKET.
#   2. /confirm requires that ticket + the EXACT typed phrase — then applies the
#      IDENTICAL journal mutation the CLI would. Nothing touches an exchange:
#      records are mode="dry-run"; there is still no live route in this app.
# ---------------------------------------------------------------------------- #
CONFIRM_PHRASE = "CONFIRM"
_TICKETS: dict = {}
_TICKET_TTL = 120.0          # a ticket is a FRESH read; stale → re-preview (re-fetch rail)


def _mint(payload: dict) -> str:
    now = time.time()
    for k in [k for k, v in _TICKETS.items() if now - v["ts"] > _TICKET_TTL]:
        _TICKETS.pop(k, None)
    tok = secrets.token_hex(8)
    _TICKETS[tok] = {"ts": now, **payload}
    return tok


def _redeem(token: str) -> dict | None:
    t = _TICKETS.pop(token or "", None)
    if not t or time.time() - t["ts"] > _TICKET_TTL:
        return None
    return t


def _require_phrase(body: dict) -> None:
    if (body or {}).get("phrase") != CONFIRM_PHRASE:
        raise HTTPException(status_code=403,
                            detail=f"type the exact phrase {CONFIRM_PHRASE!r} to proceed")


@app.post("/api/stage/preview")
def stage_preview(body: dict):
    symbol = (body or {}).get("symbol", "").strip()
    if not symbol:
        raise HTTPException(status_code=422, detail="symbol required")
    s = _settings()
    mkt = _market((body or {}).get("market"))
    records = jn.load_records()
    state = jn.portfolio_state(records, s.journal)
    try:
        r = an.analyze(mkt, symbol, s, tf_trigger=s.analysis.tf_trigger,
                       drawdown_pct=state.drawdown_pct)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"{type(exc).__name__}: {exc}")
    if not r.setups or r.plan is None:
        raise HTTPException(status_code=409, detail="Nothing to stage — no live setup right now → NO TRADE.")
    signal, plan = r.setups[0], r.plan
    if not getattr(plan, "valid", True):
        raise HTTPException(status_code=409, detail="Plan not worth taking: " + "; ".join(plan.notes))
    if od.has_open_or_staged(records, symbol):
        raise HTTPException(status_code=409, detail=f"Already OPEN/STAGED on {symbol} — refusing a duplicate.")
    proposed = od.build_proposed(plan, signal, group="indep", risk_pct=r.risk_pct)
    guard = check_trade(state, proposed, s.guards)
    if not guard.allowed:
        raise HTTPException(status_code=409, detail="GUARD BLOCK: " + "; ".join(guard.hard_blocks))
    regime = r.structure.state.trend if (r.structure and r.structure.state) else ""
    verdict = es.lookup_pooled_verdict(mkt, s.analysis.tf_trigger, signal.setup, regime,
                                       symbol=symbol,
                                       context=es.resolve_context(r.context, signal.direction))
    level, msg = es.edge_gate(verdict, s.edge)
    if level == es.EDGE_NEGATIVE:
        raise HTTPException(status_code=409, detail=f"EDGE BLOCK: {msg}")
    drift_ok, dmsg = od.check_drift(signal.entry_type, plan.entry, r.last_price, s.order.drift_pct)
    val_ok, vmsg = od.check_validity(signal, r.last_price)
    if not (drift_ok and val_ok):
        raise HTTPException(status_code=409, detail="STALE READ: " +
                            "; ".join(m for ok, m in ((drift_ok, dmsg), (val_ok, vmsg)) if not ok))
    stage_kwargs = dict(
        market=mkt, symbol=symbol, side=signal.direction, setup=signal.setup, grade=signal.grade,
        group="indep", entry=plan.entry, stop=plan.stop, targets=list(plan.targets),
        risk_pct=r.risk_pct, risk_amount=plan.risk_actual, regime=regime, bias=r.bias,
        tide=r.posture, tf=s.analysis.tf_trigger, entry_type=signal.entry_type,
        invalidation=getattr(r, "invalidation", 0.0) or 0.0,
        leverage=getattr(plan, "leverage", 1.0) or 1.0,
        liquidation_price=getattr(plan, "liquidation_price", None),
        mode="dry-run", profile=s.profile)
    token = _mint({"kind": "stage", "kwargs": stage_kwargs})
    return {"token": token, "ttl": _TICKET_TTL, "ticket": {
        "symbol": symbol, "setup": signal.setup, "side": signal.direction, "grade": signal.grade,
        "guards": "PASS", "edge": {"level": level, "msg": msg},
        "refetch": f"fresh · {dmsg}; {vmsg}",
        "entry": plan.entry, "stop": plan.stop, "targets": list(plan.targets),
        "entry_type": signal.entry_type, "risk_pct": r.risk_pct,
        "risk_actual": plan.risk_actual, "total_cost": plan.total_cost,
        "notional": plan.notional, "net_rr": plan.net_rr,
        "leverage": getattr(plan, "leverage", None),
        "liquidation": getattr(plan, "liquidation_price", None),
        "profile": s.profile,
    }}


@app.post("/api/stage/confirm")
def stage_confirm(body: dict):
    _require_phrase(body)
    t = _redeem((body or {}).get("token", ""))
    if not t or t.get("kind") != "stage":
        raise HTTPException(status_code=410, detail="ticket unknown or expired — preview again (fresh read required)")
    tid = jn.record_staged(**t["kwargs"])
    return {"staged": True, "id": tid, "mode": "dry-run",
            "note": "recorded to the journal — NO order sent to any exchange"}


def _bars_for(rec, s, exchanges: dict):
    mkt = Market(rec.market) if not isinstance(rec.market, Market) else rec.market
    ex = exchanges.get(mkt.value)
    if ex is None:
        ex = dfetch.make_exchange(mkt, s.api_key, s.api_secret)
        dfetch.load_markets(ex)
        exchanges[mkt.value] = ex
    tf = rec.tf or s.analysis.tf_trigger
    bars = dfetch.drop_unclosed(ex, dfetch.fetch_ohlcv(ex, rec.symbol, tf, s.analysis.candle_limit), tf)
    return bars, float(bars["close"].iloc[-1])


@app.get("/api/manage/proposals")
def manage_proposals():
    s = _settings()
    out = []
    exchanges: dict = {}
    for rec in jn.load_records():
        if rec.status not in ("staged", "open"):
            continue
        row = {"id": rec.id, "symbol": rec.symbol, "side": rec.side, "setup": rec.setup,
               "status": rec.status, "entry": rec.entry_planned, "stop": rec.stop_planned,
               "targets": list(rec.targets or []), "notes": list(rec.notes or []),
               "kind": "hold", "detail": "", "token": None}
        try:
            bars, last = _bars_for(rec, s, exchanges)
            row["last"] = last
            if rec.status == "staged":
                filled, entry_actual, why = mon.detect_paper_fill(
                    rec, bars, last, expiry_bars=s.setups.expiry_bars)
                if not filled and ("void" in why or "stale" in why or "expir" in why):
                    row.update(kind="cancel", detail=why,
                               token=_mint({"kind": "cancel", "rec_id": rec.id, "why": why}))
                elif not filled:
                    row.update(kind="pending", detail=why)
                else:
                    replay = mon.replay_after_fill(rec, bars, s.risk_mgmt)
                    if replay and replay["closed"]:
                        row.update(kind="outcome",
                                   detail=(f"{why} — bars since the touch already CLOSED it at "
                                           f"{replay['exit']:,.6g} ({replay['r']:+.2f}R). Record the "
                                           f"COMPLETED outcome, never a naive open."),
                                   token=_mint({"kind": "outcome", "rec_id": rec.id,
                                                "entry_actual": entry_actual, "stop": rec.stop_planned,
                                                "exit": replay["exit"], "r": replay["r"],
                                                "fill_ts": replay["fill_ts"]}))
                    else:
                        mark = f" · marks {replay['r']:+.2f}R since fill" if replay else ""
                        row.update(kind="open", detail=f"{why}{mark}",
                                   token=_mint({"kind": "open", "rec_id": rec.id,
                                                "entry_actual": entry_actual, "stop": rec.stop_planned,
                                                "fill_ts": (replay or {}).get("fill_ts")}))
            else:  # open — the SAME state machine as the backtest proposes the next step
                action = mon.manage_step(rec, bars, last, s.risk_mgmt)
                row["unrealized_r"] = mon.unrealized_r(rec, last)
                if action.kind == mon.HOLD:
                    row.update(kind="hold", detail=action.reason)
                elif action.kind in (mon.STOP_EXIT, mon.CLOSE_TP2):
                    row.update(kind="close", detail=action.reason,
                               token=_mint({"kind": "close", "rec_id": rec.id,
                                            "fill_price": action.fill_price,
                                            "total_r": action.total_realized_r,
                                            "risk_amount": rec.risk_amount,
                                            "new_stop": action.new_stop}))
                elif action.kind == mon.SCALE_TP1:
                    row.update(kind="scale", detail=action.reason,
                               token=_mint({"kind": "scale", "rec_id": rec.id,
                                            "new_stop": action.new_stop,
                                            "remaining": action.remaining_after,
                                            "delta_r": action.realized_delta_r,
                                            "locked": (rec.locked_r or 0.0) + (action.realized_delta_r or 0.0)}))
                elif action.kind == mon.TRAIL:
                    row.update(kind="trail", detail=action.reason,
                               token=_mint({"kind": "trail", "rec_id": rec.id,
                                            "new_stop": action.new_stop}))
        except Exception as exc:  # noqa: BLE001 — one bad symbol must not sink the book
            row.update(kind="error", detail=f"{type(exc).__name__}: {exc}")
        out.append(row)
    return {"proposals": out, "phrase": CONFIRM_PHRASE, "ttl": _TICKET_TTL}


@app.post("/api/manage/confirm")
def manage_confirm(body: dict):
    _require_phrase(body)
    t = _redeem((body or {}).get("token", ""))
    if not t:
        raise HTTPException(status_code=410, detail="ticket unknown or expired — re-check the book")
    now = datetime.now(timezone.utc).isoformat()
    k = t["kind"]
    if k == "cancel":
        jn.record_cancel(t["rec_id"], reason=t["why"])
    elif k == "open":
        jn.record_open(t["rec_id"], entry_actual=t["entry_actual"], current_stop=t["stop"],
                       mode="dry-run", managed_at=t.get("fill_ts"))
    elif k == "outcome":
        jn.record_open(t["rec_id"], entry_actual=t["entry_actual"], current_stop=t["stop"],
                       managed_at=t["fill_ts"], mode="dry-run")
        jn.record_close(t["rec_id"], exit_price=t["exit"], realized_r=t["r"])
    elif k == "close":
        pnl = (t["total_r"] or 0.0) * (t["risk_amount"] or 0.0)
        jn.record_close(t["rec_id"], exit_price=t["fill_price"], realized_r=t["total_r"],
                        pnl_usd=pnl, stop_actual=t["new_stop"])
    elif k == "scale":
        jn.record_manage(t["rec_id"], current_stop=t["new_stop"], remaining_fraction=t["remaining"],
                         tp1_filled=True, locked_r=t["locked"], managed_at=now,
                         note=f"TP1 scale-out +{(t['delta_r'] or 0):.2f}R; stop→breakeven")
    elif k == "trail":
        jn.record_manage(t["rec_id"], current_stop=t["new_stop"], managed_at=now,
                         note=f"trailed stop → {t['new_stop']:,.6g}")
    else:
        raise HTTPException(status_code=400, detail=f"unknown action {k!r}")
    return {"applied": k, "id": t.get("rec_id"), "mode": "dry-run"}


def main() -> None:
    import uvicorn
    # LOCALHOST ONLY — the cockpit (and your keys) never leave this machine.
    uvicorn.run(app, host="127.0.0.1", port=8484, log_level="warning")


if __name__ == "__main__":
    main()
