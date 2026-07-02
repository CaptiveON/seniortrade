"""Tests for FULL .env control — the generic {SECTION}_{FIELD} override loader."""
from __future__ import annotations

from src.config import load_settings


def test_generic_override_across_sections_and_types(monkeypatch):
    monkeypatch.setenv("GUARDS_HEAT_CAP_PCT", "9")          # float
    monkeypatch.setenv("GUARDS_MAX_TRADES_PER_DAY", "3")    # int
    monkeypatch.setenv("MGMT_KELLY_ENABLED", "true")        # bool
    monkeypatch.setenv("MGMT_EDGE_SCALED", "on")            # bool (alt truthy)
    monkeypatch.setenv("MGMT_VOL_TARGET_ENABLED", "true")   # P12 vol-target bool
    monkeypatch.setenv("MGMT_VOL_TARGET_SIGMA_R", "2.0")    # P12 vol-target float
    monkeypatch.setenv("EDGE_FLOOR", "0.08")               # float
    monkeypatch.setenv("EDGE_SCAN_TOP", "25")              # Optional[int] from None
    monkeypatch.setenv("MARKETS_USDM_TAKER", "0.0004")     # small float
    monkeypatch.setenv("SCREEN_EXCLUDED_BASES", "USDC,EUR")  # tuple[str]
    s = load_settings()
    assert s.guards.heat_cap_pct == 9.0
    assert s.guards.max_trades_per_day == 3
    assert s.risk_mgmt.kelly_enabled is True
    assert s.risk_mgmt.edge_scaled is True
    assert s.risk_mgmt.vol_target_enabled is True
    assert s.risk_mgmt.vol_target_sigma_r == 2.0
    assert s.edge.floor == 0.08
    assert s.edge.scan_top == 25
    assert s.markets.usdm_taker == 0.0004
    assert s.screener.excluded_bases == ("USDC", "EUR")


def test_friendly_aliases(monkeypatch):
    monkeypatch.setenv("RISK_PCT", "2.5")
    monkeypatch.setenv("ACCOUNT_EQUITY", "5000")
    monkeypatch.setenv("SCREEN_TF", "1h")
    s = load_settings()
    assert s.risk.risk_pct == 2.5
    assert s.risk.account_equity == 5000.0
    assert s.screener.screen_tf == "1h"


def test_safety_rails_are_not_overridable(monkeypatch):
    monkeypatch.setenv("LIVE_ENABLE_VALUE", "HACKED")
    monkeypatch.setenv("LIVE_CONFIRM_PHRASE", "go")
    monkeypatch.setenv("LIVE_ENABLE_ENV", "ANYTHING")
    s = load_settings()
    assert s.live.enable_value == "I_UNDERSTAND_THE_RISK"   # protected
    assert s.live.confirm_phrase == "CONFIRM LIVE"          # protected
    assert s.live.enable_env == "LIVE_TRADING_ENABLED"      # protected
    # but a non-rail LIVE field IS tunable
    monkeypatch.setenv("LIVE_FILL_TIMEOUT_S", "30")
    assert load_settings().live.fill_timeout_s == 30.0


def test_unset_fields_keep_defaults(monkeypatch):
    monkeypatch.delenv("GUARDS_MAX_POSITIONS", raising=False)
    assert load_settings().guards.max_positions == 5        # untouched default
