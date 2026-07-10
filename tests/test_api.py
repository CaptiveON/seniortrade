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
    assert "READ-ONLY" in d["mode"].upper() or "read-only" in d["mode"]


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


def test_no_order_endpoints_exist_in_phase_a():
    # READ-ONLY contract: no stage/order/live route is registered at all
    paths = {r.path for r in api.app.routes}
    for forbidden in ("stage", "order", "live", "manage", "roar"):
        assert not any(forbidden in p for p in paths), paths


def test_positions_and_journal_read_endpoints():
    c = _client()
    assert c.get("/api/positions").status_code == 200
    d = c.get("/api/journal").json()
    assert "stats" in d and "closed" in d
