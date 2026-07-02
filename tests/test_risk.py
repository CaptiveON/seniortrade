"""Tests for the dollar worked-example formatter (risk-first sizing)."""
from __future__ import annotations

import pytest

from src.risk import build_worked_example, worked_example_lines


def test_long_sizing_math():
    we = build_worked_example(
        account_equity=1000.0, risk_pct=1.0,
        entry=100.0, stop=95.0, take_profit=107.5, side="long",
    )
    assert we.risk_amount == pytest.approx(10.0)        # risk 1% of $1000
    assert we.per_unit_risk == pytest.approx(5.0)       # 100 -> 95
    assert we.position_size == pytest.approx(2.0)       # $10 / $5
    assert we.notional == pytest.approx(200.0)          # 2 * 100
    assert we.reward_amount == pytest.approx(15.0)      # (107.5-100) * 2
    assert we.rr == pytest.approx(1.5)


def test_short_sizing_math():
    we = build_worked_example(
        account_equity=2000.0, risk_pct=0.5,
        entry=100.0, stop=104.0, take_profit=92.0, side="short",
    )
    assert we.risk_amount == pytest.approx(10.0)        # 0.5% of $2000
    assert we.per_unit_risk == pytest.approx(4.0)
    assert we.position_size == pytest.approx(2.5)       # $10 / $4
    assert we.reward_amount == pytest.approx(20.0)      # (100-92) * 2.5
    assert we.rr == pytest.approx(2.0)


def test_worked_example_lines_speak_dollars():
    we = build_worked_example(
        account_equity=1000.0, risk_pct=1.0,
        entry=100.0, stop=95.0, take_profit=107.5, side="long",
    )
    lines = worked_example_lines(we, "BTC/USDT")
    assert len(lines) == 5
    assert "$1,000.00 account and 1% risk" in lines[0]
    assert "you risk $10.00" in lines[0]
    assert "you make ~$15.00 (1.50R)" in lines[2]
    assert "R:R 1:1.50" in lines[4]


def test_rejects_wrong_side_geometry():
    # long needs stop < entry < take_profit
    with pytest.raises(ValueError):
        build_worked_example(
            account_equity=1000.0, risk_pct=1.0,
            entry=100.0, stop=105.0, take_profit=110.0, side="long",
        )


def test_rejects_zero_risk_distance():
    with pytest.raises(ValueError):
        build_worked_example(
            account_equity=1000.0, risk_pct=1.0,
            entry=100.0, stop=100.0, take_profit=110.0, side="long",
        )


# --------------------------------------------------------------------------- #
# Effective-risk pipeline (drawdown-scale × edge-scale × Kelly cap × ceiling)
# --------------------------------------------------------------------------- #
from dataclasses import replace                       # noqa: E402

from src.config import RiskConfig                      # noqa: E402
from src.risk import effective_risk_pct                # noqa: E402


def test_effective_risk_flat_by_default():
    adj = effective_risk_pct(1.0, RiskConfig())        # all stages off
    assert adj.effective_pct == pytest.approx(1.0) and adj.notes == []


def test_effective_risk_hard_ceiling_always_applies():
    adj = effective_risk_pct(8.0, RiskConfig())        # default max_risk_pct = 5
    assert adj.effective_pct == pytest.approx(5.0)
    assert any("ceiling" in n for n in adj.notes)


def test_effective_risk_drawdown_scaling():
    cfg = replace(RiskConfig(), dd_scale_enabled=True)     # start 5%, full 20%, floor 0.5
    assert effective_risk_pct(2.0, cfg, drawdown_pct=0.0).effective_pct == pytest.approx(2.0)
    deep = effective_risk_pct(2.0, cfg, drawdown_pct=20.0)  # ≥ full → ×floor
    assert deep.effective_pct == pytest.approx(1.0)         # 2.0 × 0.5


def test_effective_risk_edge_scaling_clamped():
    cfg = replace(RiskConfig(), edge_scaled=True, edge_ref_r=0.15,
                  edge_min_mult=0.5, edge_max_mult=2.0, max_risk_pct=100.0)
    strong = effective_risk_pct(1.0, cfg, edge_r=0.45, confidence=1.0)   # 3× → cap 2×
    assert strong.effective_pct == pytest.approx(2.0)
    weak = effective_risk_pct(1.0, cfg, edge_r=0.05, confidence=0.5)     # 0.17× → floor 0.5×
    assert weak.effective_pct == pytest.approx(0.5)


def test_effective_risk_kelly_only_reduces():
    cfg = replace(RiskConfig(), kelly_enabled=True, kelly_fraction=0.25, max_risk_pct=100.0)
    # W=0.52, R=1 → f*=0.04 → ¼-Kelly = 1.0% < base 2% → capped to 1%
    adj = effective_risk_pct(2.0, cfg, win_rate=0.52, avg_win_r=1.0, avg_loss_r=1.0)
    assert adj.effective_pct == pytest.approx(1.0, abs=1e-6)
    # strong edge → Kelly above base → no reduction
    adj2 = effective_risk_pct(1.0, cfg, win_rate=0.7, avg_win_r=1.5, avg_loss_r=1.0)
    assert adj2.effective_pct == pytest.approx(1.0)


# --- P12 volatility-target (variance) sizing -------------------------------- #
def test_vol_target_off_by_default_is_flat():
    assert effective_risk_pct(1.0, RiskConfig(), sigma_r=3.0).effective_pct == pytest.approx(1.0)


def test_vol_target_shrinks_high_variance_setups():
    cfg = replace(RiskConfig(), vol_target_enabled=True, vol_target_sigma_r=1.5)
    assert effective_risk_pct(1.0, cfg, sigma_r=3.0).effective_pct == pytest.approx(0.5)   # 1.5/3 → ×0.5
    assert effective_risk_pct(1.0, cfg, sigma_r=2.0).effective_pct == pytest.approx(0.75)  # 1.5/2 → ×0.75


def test_vol_target_is_conservative_only_reduces_by_default():
    cfg = replace(RiskConfig(), vol_target_enabled=True, vol_target_sigma_r=1.5)
    # low σ_R would imply ×1.0+, but vol_max_mult=1.0 keeps it at base (never sizes UP by default)
    assert effective_risk_pct(1.0, cfg, sigma_r=0.5).effective_pct == pytest.approx(1.0)


def test_vol_target_clamped_to_floor():
    cfg = replace(RiskConfig(), vol_target_enabled=True, vol_target_sigma_r=1.5, vol_min_mult=0.5)
    assert effective_risk_pct(1.0, cfg, sigma_r=100.0).effective_pct == pytest.approx(0.5)  # floor


def test_vol_target_noop_without_sigma():
    cfg = replace(RiskConfig(), vol_target_enabled=True)
    assert effective_risk_pct(1.0, cfg, sigma_r=None).effective_pct == pytest.approx(1.0)
