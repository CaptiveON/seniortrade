"""UI Phase A — the local read-only API (offline: engine calls monkeypatched)."""
from __future__ import annotations

import time
from types import SimpleNamespace

from fastapi.testclient import TestClient

import src.api as api


def _client():
    return TestClient(api.app)


def test_index_serves_the_cockpit():
    r = _client().get("/")
    assert r.status_code == 200
    assert "SENIORTRADE" in r.text and "PROOF BEFORE POSITION" in r.text


def test_status_shape_and_safety_banner():
    r = _client().get("/api/status")
    assert r.status_code == 200
    d = r.json()
    for k in ("profile", "tier", "equity", "sig_z", "watch", "cache", "mode", "stages"):
        assert k in d
    assert "DRY-RUN" in d["mode"] and "LIVE only via CLI" in d["mode"]


def test_board_refresh_runs_scan_in_background(monkeypatch):
    ctx = SimpleNamespace(posture="risk-off", note="test tide")
    fake_beta = {"posture": "risk-off", "tide_regime": "down",
                 "drift": {"proven": False, "mean": -0.1, "lower": -0.15, "n": 100, "cells": 3},
                 "candidates": [], "near_misses": []}
    def fake_scan(market, settings, top=None, refresh=False, progress=None, hard=False):
        return ctx, [], ["ETH/USDT:USDT"], fake_beta
    monkeypatch.setattr(api.es, "scan", fake_scan)
    with api._LOCK:                                       # reset server state
        api._STATE.update({"board": None, "board_ts": 0.0, "scanning": False, "error": None})
    c = _client()
    r = c.post("/api/board/refresh")
    assert r.status_code == 200 and r.json()["started"]
    for _ in range(50):                                   # wait for the worker thread
        d = c.get("/api/board").json()
        if not d["scanning"] and d["board"]:
            break
        time.sleep(0.05)
    assert d["error"] is None
    assert d["board"]["posture"] == "risk-off"
    assert d["board"]["watchlist"] == ["ETH/USDT:USDT"]
    assert d["board"]["beta"]["drift"]["proven"] is False  # honest refusal survives serialization


def test_board_refresh_refuses_concurrent_scans(monkeypatch):
    with api._LOCK:
        api._STATE["scanning"] = True
    try:
        r = _client().post("/api/board/refresh")
        assert r.status_code == 409
    finally:
        with api._LOCK:
            api._STATE["scanning"] = False


def test_phase_b_contract_no_live_surface():
    # Phase B: stage/manage exist as DRY-RUN + typed-CONFIRM. LIVE must have NO surface:
    paths = {r.path for r in api.app.routes}
    assert not any("live" in p or "roar" in p for p in paths), paths
    import inspect
    src = inspect.getsource(api)
    assert "from . import live" not in src and "import live" not in src   # never even imported


def test_stage_confirm_requires_exact_phrase_and_fresh_ticket():
    c = _client()
    # wrong phrase → 403 (server-enforced, the UI cannot bypass)
    r = c.post("/api/stage/confirm", json={"token": "whatever", "phrase": "confirm"})
    assert r.status_code == 403
    # right phrase but unknown/expired ticket → 410 (fresh preview required)
    r = c.post("/api/stage/confirm", json={"token": "deadbeef", "phrase": "CONFIRM"})
    assert r.status_code == 410


def test_stage_preview_and_confirm_happy_path(monkeypatch):
    plan = SimpleNamespace(valid=True, notes=[], entry=1.0, stop=1.1, targets=[0.9, 0.8],
                           risk_actual=10.0, total_cost=0.5, notional=400.0, net_rr=2.5,
                           risk_budget=10.0, size=100.0, gross_rr=2.7, leverage=3.0,
                           liquidation_price=1.4, symbol="X/USDT:USDT", side="short")
    sig = SimpleNamespace(setup="range_fade", direction="short", grade="A",
                          entry_type="limit", entry=1.0, stop=1.1, targets=[0.9, 0.8])
    res = SimpleNamespace(setups=[sig], plan=plan, last_price=1.0, bias="neutral",
                          posture="risk-off", freshness="fresh", context={"arch": "range"},
                          risk_pct=1.0, structure=SimpleNamespace(state=SimpleNamespace(trend="range")),
                          invalidation=1.2)
    monkeypatch.setattr(api.an, "analyze", lambda *a, **k: res)
    monkeypatch.setattr(api.jn, "load_records", lambda path=None: [])
    monkeypatch.setattr(api.od, "has_open_or_staged", lambda recs, sym: False)
    monkeypatch.setattr(api.es, "lookup_pooled_verdict", lambda *a, **k: None)  # hermetic: no cache
    staged = {}
    monkeypatch.setattr(api.jn, "record_staged", lambda **kw: staged.update(kw) or "tid123")
    c = _client()
    r = c.post("/api/stage/preview", json={"symbol": "X/USDT:USDT"})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["ticket"]["guards"] == "PASS" and d["ticket"]["entry_type"] == "limit"
    r2 = c.post("/api/stage/confirm", json={"token": d["token"], "phrase": "CONFIRM"})
    assert r2.status_code == 200 and r2.json()["staged"] and r2.json()["mode"] == "dry-run"
    assert staged["mode"] == "dry-run" and staged["symbol"] == "X/USDT:USDT"
    # the ticket is ONE-TIME: replay refused
    r3 = c.post("/api/stage/confirm", json={"token": d["token"], "phrase": "CONFIRM"})
    assert r3.status_code == 410


def test_manage_confirm_applies_the_same_journal_mutations(monkeypatch):
    calls = []
    monkeypatch.setattr(api.jn, "record_open",
                        lambda rid, **kw: calls.append(("open", rid, kw)))
    monkeypatch.setattr(api.jn, "record_close",
                        lambda rid, **kw: calls.append(("close", rid, kw)))
    tok = api._mint({"kind": "outcome", "rec_id": "r1", "entry_actual": 1.0, "stop": 1.1,
                     "exit": 1.1, "r": -1.0, "fill_ts": "t"})
    c = _client()
    r = c.post("/api/manage/confirm", json={"token": tok, "phrase": "CONFIRM"})
    assert r.status_code == 200 and r.json()["applied"] == "outcome"
    kinds = [k for k, *_ in calls]
    assert kinds == ["open", "close"]                 # finding #6 semantics: fill THEN close, never naive open
    assert calls[0][2]["mode"] == "dry-run"


def test_manage_confirm_phrase_gate():
    tok = api._mint({"kind": "cancel", "rec_id": "r2", "why": "stale"})
    r = _client().post("/api/manage/confirm", json={"token": tok, "phrase": "yes"})
    assert r.status_code == 403


def test_positions_and_journal_read_endpoints():
    c = _client()
    assert c.get("/api/positions").status_code == 200
    d = c.get("/api/journal").json()
    assert "stats" in d and "closed" in d
