"""Risk-profile system (Phase 3): appetite/selectivity move; honesty gates never do."""
from __future__ import annotations

import pytest

from src.config import (
    _HONESTY_LOCKED,
    RISK_PROFILES,
    Settings,
    apply_profile,
    apply_tier,
    load_settings,
    strictness_fluke_note,
)


def test_no_profile_is_todays_defaults():
    s = Settings()
    assert apply_profile(s, None) is s
    assert s.profile is None


def test_profiles_are_a_monotonic_appetite_ladder():
    order = ["L0", "L1", "L2", "L3"]
    applied = [apply_profile(Settings(), n) for n in order]
    risks = [a.risk.risk_pct for a in applied]
    heats = [a.guards.heat_cap_pct for a in applied]
    floors = [a.edge.floor for a in applied]
    assert risks == sorted(risks) and len(set(risks)) == 4          # appetite strictly rises
    assert heats == sorted(heats) and len(set(heats)) == 4
    assert floors == sorted(floors, reverse=True)                   # selectivity loosens
    assert applied[0].risk_mgmt.dd_scale_enabled and not applied[3].risk_mgmt.dd_scale_enabled
    for a in applied:                                               # coherence: base ≤ ceiling
        assert a.risk.risk_pct <= a.risk_mgmt.max_risk_pct


def test_profiles_never_touch_honesty_gates_or_rails():
    # registry-level: no locked (section, field) appears in any profile
    for name, sections in RISK_PROFILES.items():
        for sec, fields_ in sections.items():
            for f in fields_:
                assert (sec, f) not in _HONESTY_LOCKED, f"{name} touches {sec}.{f}"
    # applied-level: the truth bar is bit-identical across every profile
    base = Settings()
    for name in RISK_PROFILES:
        a = apply_profile(base, name)
        assert a.edge.sig_z == base.edge.sig_z
        assert a.backtest.sig_z == base.backtest.sig_z
        assert a.backtest.null_k == base.backtest.null_k
        assert a.backtest.deff_enabled == base.backtest.deff_enabled
        assert a.edge.min_regime_n == base.edge.min_regime_n
        assert a.live.confirm_phrase == base.live.confirm_phrase


def test_precedence_tier_then_profile_then_env(monkeypatch):
    # profile applies over the tier base; explicit env beats the profile
    monkeypatch.setenv("RISK_PCT", "2.0")
    base = apply_profile(apply_tier(Settings(), "intraday"), "L3")
    s = load_settings(base=base)
    assert s.edge.tf == "1h"                        # tier survived
    assert s.guards.heat_cap_pct == 25.0            # profile applied
    assert s.risk.risk_pct == 2.0                   # env WON over L3's 5%
    assert s.profile == "L3"                        # attribution preserved


def test_unknown_profile_refused():
    with pytest.raises(ValueError, match="unknown risk profile"):
        apply_profile(Settings(), "L9")


def test_strictness_fluke_note_quantifies_the_cost():
    assert strictness_fluke_note(1.65) is None                      # default bar → silent
    assert "~12%" in strictness_fluke_note(1.28)
    assert "~16%" in strictness_fluke_note(1.00)
    mid = strictness_fluke_note(1.50)                               # interpolated, monotone
    assert mid and any(f"~{v}%" in mid for v in (7, 8))


def test_record_staged_carries_profile_attribution(tmp_path):
    from src.config import Market
    from src.journal import record_staged
    path = tmp_path / "journal.jsonl"
    record_staged(market=Market.USDM, symbol="X/USDT:USDT", side="short", setup="range_fade",
                  grade="A", group="indep", entry=1.0, stop=1.1, targets=[0.9],
                  risk_pct=3.0, risk_amount=30.0, profile="L2", path=path)
    assert "risk profile: L2" in path.read_text()
