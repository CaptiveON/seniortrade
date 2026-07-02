"""Tests for the step-3 optimizer's pure logic — objective, significance, plateau, decision."""
from __future__ import annotations

from src.optimize import bootstrap_p_positive, decide, plateau_ok, robust_score


def test_robust_score_requires_min_sample():
    assert robust_score([0.5] * 5, [0.5], min_trades=20, penalty_lambda=0.5) == float("-inf")


def test_robust_score_penalises_inconsistency():
    consistent = robust_score([0.5] * 30, [0.5, 0.5, 0.5], min_trades=20, penalty_lambda=0.5)
    scattered = robust_score([0.5] * 30, [1.5, -0.5], min_trades=20, penalty_lambda=0.5)
    assert consistent == 0.5                       # zero cross-coin spread → no penalty
    assert scattered < consistent                  # same expectancy, but inconsistent → penalised


def test_bootstrap_p_positive_separates_signal_from_noise():
    assert bootstrap_p_positive([1.0] * 40, runs=1000, seed=1) < 0.05      # clearly +EV
    assert bootstrap_p_positive([-1.0] * 40, runs=1000, seed=1) > 0.95     # clearly −EV
    mixed = bootstrap_p_positive([1.0, -1.0] * 20, runs=2000, seed=1)      # mean ~0
    assert 0.2 < mixed < 0.8


def test_plateau_ok_rejects_lone_spike():
    values = [0.25, 0.5, 0.75, 1.0, 1.5]
    exps = [-0.1, 0.3, 0.4, 0.2, -0.2]
    assert plateau_ok(values, exps, 0.75)          # neighbours 0.5 and 1.0 both ≥ 0
    assert not plateau_ok(values, exps, 0.5)       # neighbour 0.25 is negative → spike


def _kw(**over):
    base = dict(best_value=0.75, default_value=0.5, best_test_exp=0.40, default_test_exp=0.20,
                p_value=0.005, alpha_adj=0.01, plateau=True, cost_stress_exp=0.30, lockbox_exp=0.10)
    base.update(over)
    return base


def test_decide_adopts_only_when_all_gates_pass():
    assert decide(**_kw()).adopt
    assert not decide(**_kw(p_value=0.5)).adopt              # fails significance
    assert not decide(**_kw(best_test_exp=0.10)).adopt        # doesn't beat the default OOS
    assert not decide(**_kw(plateau=False)).adopt             # lone spike
    assert not decide(**_kw(cost_stress_exp=-0.05)).adopt     # dies under realistic costs
    assert not decide(**_kw(lockbox_exp=-0.20)).adopt         # lock-box disagrees
    assert not decide(**_kw(best_value=0.5)).adopt            # identical to default
