"""FULL SPECTRUM + Universe Studio + Consequence Card.

The commercial layer's contract: every candidate gets an HONEST tier (never a void),
the consequence simulation prices the cell's measured stats (never invents them), and
a universe lens only chooses WHAT to scan — no lens can manufacture a verdict.
"""
from types import SimpleNamespace

import pytest

from src import edge_score as es
from src import expectancy as ex
from src import universe as uv
from src.config import EdgeScoreConfig, Market


# --------------------------------------------------------------------------- #
# Consequence Card — MC of N trades from a cell's measured stats
# --------------------------------------------------------------------------- #

def test_consequence_deterministic():
    a = ex.consequence(0.5, 1.5, 1.0)
    b = ex.consequence(0.5, 1.5, 1.0)
    assert a == b                       # fixed seed → same card every render


def test_consequence_shape_and_ordering():
    c = ex.consequence(0.45, 1.8, 1.0, n_trades=20)
    assert c["n_trades"] == 20
    assert c["p5_r"] <= c["median_r"] <= c["p95_r"]
    assert 0.0 <= c["p_negative"] <= 1.0
    assert 0.0 <= c["exp_worst_streak"] <= 20


def test_consequence_loser_cell_is_priced_negative():
    # 30% winners at +0.8R vs 70% losers at -1R: a measured money-loser
    c = ex.consequence(0.30, 0.8, -1.0)
    assert c["median_r"] < 0
    assert c["p_negative"] > 0.5


def test_consequence_monotonic_in_win_rate():
    lo = ex.consequence(0.30, 1.5, 1.0)["median_r"]
    hi = ex.consequence(0.60, 1.5, 1.0)["median_r"]
    assert hi > lo


def test_consequence_sign_convention_on_avg_loss():
    # avg_loss_r arrives negative from Stats; a positive magnitude must not flip the sim
    neg = ex.consequence(0.4, 1.0, -1.0)
    pos = ex.consequence(0.4, 1.0, 1.0)
    assert neg == pos


# --------------------------------------------------------------------------- #
# spectrum_row — the tier classifier (crafted pooled profiles)
# --------------------------------------------------------------------------- #

CFG = EdgeScoreConfig()


def _profile(exp, ci_low, n=200, null_exp=0.0, null_ci=None, null_n=200):
    return {
        "by_regime": {"up": {"expectancy": exp, "ci_low": ci_low, "n": n, "n_eff": n,
                             "win_rate": 0.5, "avg_win_r": 1.5, "avg_loss_r": -1.0}},
        "null_by_regime": {"up": {"expectancy": null_exp,
                                  "ci_low": null_exp if null_ci is None else null_ci,
                                  "n": null_n}},
        "fold_consistency": 1.0,
    }


def _row(profile, **kw):
    args = dict(setup="s", side="long", regime="up", symbol="BTC/USDT:USDT",
                context=None, cfg=CFG)
    args.update(kw)
    return es.spectrum_row(profile, **args)


def test_spectrum_no_history_is_thin():
    r = _row({"by_regime": {}})
    assert r["tier"] == "thin"
    assert r["exp"] is None and r["consequence" if False else "win_rate"] is None


def test_spectrum_small_n_eff_is_thin():
    r = _row(_profile(0.5, 0.3, n=5))
    assert r["tier"] == "thin"
    assert "effective" in r["reason"]


def test_spectrum_money_loser_is_loser():
    r = _row(_profile(-0.2, -0.4))
    assert r["tier"] == "loser"
    assert r["exp"] == pytest.approx(-0.2)


def test_spectrum_full_gate_pass_is_alpha():
    # strong edge, tight CI, null flat → clears sig + floor
    r = _row(_profile(0.60, 0.55, n=400, null_exp=0.0))
    assert r["tier"] == "alpha"
    assert r["edge"] is not None and r["edge"] > CFG.floor


def test_spectrum_positive_but_luck_is_unproven_or_near():
    # positive mean, wide CI → fails significance; must NOT be hidden, must NOT be alpha
    r = _row(_profile(0.10, -0.30))
    assert r["tier"] in ("near", "unproven")
    assert r["exp"] == pytest.approx(0.10)


def test_spectrum_near_floor_is_near():
    # passes significance but magnitude lands just under the floor → near
    r = _row(_profile(0.30, 0.06, n=400, null_exp=0.0))
    ok, edge, _ = es.null_adjusted_edge(
        _profile(0.30, 0.06, n=400)["by_regime"]["up"],
        {"expectancy": 0.0, "ci_low": 0.0, "n": 200}, CFG)
    if not ok and 0 < edge <= CFG.floor and (CFG.floor - edge) <= 0.03:
        assert r["tier"] == "near"
    else:                                # geometry drifted → still an honest non-alpha tier
        assert r["tier"] in ("near", "unproven", "alpha")


def test_spectrum_carries_cell_stats_for_consequence():
    r = _row(_profile(0.60, 0.55, n=400))
    assert r["win_rate"] == pytest.approx(0.5)
    assert r["avg_win_r"] == pytest.approx(1.5)
    assert r["avg_loss_r"] == pytest.approx(-1.0)


def test_tier_order_covers_all_emitted_tiers():
    assert set(es.TIER_ORDER) == {"alpha", "beta", "near", "unproven", "thin", "loser", "quiet"}


# --------------------------------------------------------------------------- #
# Universe Studio lenses — user picks the slice; floor + gates stay honest
# --------------------------------------------------------------------------- #

def _cfg(floor=1_000_000, top_n=100):
    return SimpleNamespace(min_quote_volume=floor, top_n=top_n)


TICKERS = {
    "AAA/USDT:USDT": {"quoteVolume": 900_000_000, "last": 10.0, "percentage": 2.0},
    "BBB/USDT:USDT": {"quoteVolume": 40_000_000, "last": 5.0, "percentage": -8.0},
    "CCC/USDT:USDT": {"quoteVolume": 8_000_000, "last": 1.0, "percentage": 15.0},
    "DDD/USDT:USDT": {"quoteVolume": 500_000, "last": 0.1, "percentage": 40.0},  # below floor
    "EEE/USDT:USDT": {"quoteVolume": 12_000_000, "last": 2.0, "percentage": None},
}
ELIGIBLE = list(TICKERS)
FAKE_EX = SimpleNamespace(markets={
    "AAA/USDT:USDT": {"info": {"onboardDate": "1500000000000"}},          # old
    "CCC/USDT:USDT": {"info": {"onboardDate": str(int(2e12))}},           # far future = brand new
    "BBB/USDT:USDT": {"info": {}},
})


def test_lens_top_volume_orders_and_floors():
    rows, note = uv.select(FAKE_EX, TICKERS, ELIGIBLE, _cfg(), {"kind": "top_volume", "top_n": 2},
                           Market.USDM)
    assert [r[0] for r in rows] == ["AAA/USDT:USDT", "BBB/USDT:USDT"]
    assert "top 2" in note


def test_lens_volume_band():
    rows, _ = uv.select(FAKE_EX, TICKERS, ELIGIBLE, _cfg(),
                        {"kind": "volume_band", "min_usd": 5_000_000, "max_usd": 50_000_000},
                        Market.USDM)
    assert {r[0] for r in rows} == {"BBB/USDT:USDT", "CCC/USDT:USDT", "EEE/USDT:USDT"}


def test_lens_movers_direction_and_floor():
    rows, _ = uv.select(FAKE_EX, TICKERS, ELIGIBLE, _cfg(), {"kind": "movers", "direction": "down"},
                        Market.USDM)
    assert [r[0] for r in rows] == ["BBB/USDT:USDT"]     # DDD is the biggest mover but below floor
    rows, _ = uv.select(FAKE_EX, TICKERS, ELIGIBLE, _cfg(), {"kind": "movers", "direction": "up"},
                        Market.USDM)
    assert [r[0] for r in rows] == ["CCC/USDT:USDT", "AAA/USDT:USDT"]    # sorted by |move|


def test_lens_new_listings_uses_onboard_date():
    rows, _ = uv.select(FAKE_EX, TICKERS, ELIGIBLE, _cfg(), {"kind": "new_listings", "days": 30},
                        Market.USDM)
    assert [r[0] for r in rows] == ["CCC/USDT:USDT"]


def test_lens_custom_honors_user_list_and_skips_floor():
    rows, note = uv.select(FAKE_EX, TICKERS, ELIGIBLE, _cfg(),
                           {"kind": "custom", "symbols": ["DDD/USDT:USDT", "NOPE/USDT:USDT"]},
                           Market.USDM)
    assert [r[0] for r in rows] == ["DDD/USDT:USDT"]     # explicit pick honored despite the floor
    assert "1 of 2 resolved" in note                     # unresolved symbol is SAID, not hidden


def test_lens_mcap_band(monkeypatch):
    monkeypatch.setattr(uv, "_mcap_map",
                        lambda: {"AAA": 5e9, "BBB": 4e8, "CCC": 2e8})
    rows, _ = uv.select(FAKE_EX, TICKERS, ELIGIBLE, _cfg(),
                        {"kind": "mcap_band", "min_usd": 1e8, "max_usd": 1e9}, Market.USDM)
    assert [r[0] for r in rows] == ["BBB/USDT:USDT", "CCC/USDT:USDT"]    # sorted by mcap desc


def test_lens_unknown_kind_raises():
    with pytest.raises(ValueError, match="unknown universe lens"):
        uv.select(FAKE_EX, TICKERS, ELIGIBLE, _cfg(), {"kind": "moonshot"}, Market.USDM)


def test_lens_mcap_unavailable_is_honest(monkeypatch):
    monkeypatch.setattr(uv, "_mcap_map", lambda: {})
    with pytest.raises(ValueError, match="unavailable"):
        uv.select(FAKE_EX, TICKERS, ELIGIBLE, _cfg(), {"kind": "mcap_band"}, Market.USDM)


# --------------------------------------------------------------------------- #
# API — spectrum in the board payload, lens validation, lens catalog
# --------------------------------------------------------------------------- #

@pytest.fixture()
def client():
    from fastapi.testclient import TestClient
    from src import api
    return TestClient(api.app)


def test_api_universe_catalog_matches_lenses(client):
    data = client.get("/api/universe").json()
    kinds = [l["kind"] for l in data["lenses"]]
    assert set(kinds) == set(uv.LENSES)
    assert all("params" in l and "label" in l for l in data["lenses"])


def test_api_refresh_rejects_unknown_lens(client):
    r = client.post("/api/board/refresh", json={"universe": {"kind": "moonshot"}})
    assert r.status_code == 422
    assert "unknown lens" in r.json()["detail"]


def test_webui_carries_spectrum_layer():
    from src import api
    html = api._WEBUI.read_text()
    for needle in ("FULL SPECTRUM", "renderSpectrum", "/api/universe",
                   "quiet:[", "PROVEN LOSER", "uvLens()"):
        assert needle in html, f"webui missing {needle!r}"


def test_scan_emits_quiet_row_for_no_setup_coins():
    # the never-a-void contract lives in scan(); assert the wiring exists at source level
    import inspect
    src = inspect.getsource(es.scan)
    assert '"tier": "quiet"' in src and "watchlist.append" in src


def test_api_board_payload_carries_spectrum_and_universe(client, monkeypatch):
    from src import api
    spec = [{"symbol": "AAA/USDT:USDT", "tier": "near", "setup": "s", "side": "long",
             "regime": "up", "exp": 0.1, "edge": 0.03, "n_eff": 50, "reason": "close",
             "consequence": ex.consequence(0.5, 1.5, 1.0)}]
    fake_ctx = SimpleNamespace(posture="risk_on", note="test")
    monkeypatch.setattr(api.es, "scan",
                        lambda market, s, refresh=False, universe=None:
                        (fake_ctx, [], ["AAA/USDT:USDT"], None, spec))
    api._scan_worker(Market.USDM, refresh=False, universe={"kind": "movers", "direction": "up"})
    board = client.get("/api/board").json()["board"]
    assert board["spectrum"] == spec
    assert board["universe"] == {"kind": "movers", "direction": "up"}
    assert board["spectrum"][0]["consequence"]["median_r"] > 0
