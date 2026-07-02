"""Tests for expectancy statistics, Monte-Carlo, and the verdict."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.config import BacktestConfig
from src.expectancy import (
    VERDICT_EV,
    VERDICT_FEW,
    VERDICT_INCONCLUSIVE,
    VERDICT_NEG,
    evaluate,
    monte_carlo,
    summarize,
    verdict,
)


def _t(r, regime="up"):
    return SimpleNamespace(r=r, regime=regime)


# --- P5 hierarchical shrinkage ---------------------------------------------- #
from src.expectancy import between_group_var, shrink  # noqa: E402


def test_shrink_no_own_no_scatter_returns_prior_exactly():
    m, se, w = shrink(0, 0.0, 0.0, prior_mean=0.10, prior_se=0.03, tau2=0.0)
    assert (m, se, w) == (0.10, 0.03, 0.0)           # τ²=0 → exactly the pool (thin coin unchanged)


def test_shrink_no_own_but_scatter_widens_se():
    # never-traded coin: point estimate is the prior, but SE widens by the between-coin scatter
    # → a coin we've never observed is NEVER scored more permissively than the pool
    m, se, w = shrink(0, 0.0, 0.0, prior_mean=0.10, prior_se=0.03, tau2=0.05)
    assert m == 0.10 and w == 0.0 and se > 0.03


def test_shrink_tau2_zero_ignores_own_data():
    # groups don't differ beyond noise → trust the prior mean, not the loud own sample
    m, se, w = shrink(100, 0.50, 0.20, prior_mean=0.10, prior_se=0.03, tau2=0.0)
    assert m == 0.10 and w == 0.0


def test_shrink_large_own_trusts_own():
    m, se, w = shrink(500, 0.40, 0.30, prior_mean=0.10, prior_se=0.02, tau2=0.05)
    assert w > 0.8 and abs(m - 0.40) < 0.05


def test_shrink_thin_own_pulls_to_prior():
    m, se, w = shrink(3, 0.40, 0.40, prior_mean=0.10, prior_se=0.02, tau2=0.01)
    assert w < 0.5 and 0.10 <= m < 0.40           # between, closer to the prior


def test_between_group_var_identical_means_zero():
    stats = [(50, 0.10, 0.5), (60, 0.10, 0.5), (40, 0.10, 0.5)]
    assert between_group_var(stats, 0.10) == 0.0  # no spread beyond sampling noise


def test_between_group_var_real_spread_positive():
    stats = [(50, -0.30, 0.2), (50, 0.0, 0.2), (50, 0.30, 0.2)]
    assert between_group_var(stats, 0.0) > 0.0


def test_summarize_basic_math():
    s = summarize([1, 1, -1, 1, -1])
    assert s.n == 5
    assert s.win_rate == pytest.approx(0.6)
    assert s.expectancy == pytest.approx(0.2)
    assert s.avg_win_r == pytest.approx(1.0)
    assert s.avg_loss_r == pytest.approx(-1.0)
    assert s.profit_factor == pytest.approx(1.5)   # 3 won / 2 lost


def test_monte_carlo_positive_edge_low_ruin():
    mc = monte_carlo([0.5] * 50, runs=2000, ruin_drawdown_r=15.0)
    assert mc.risk_of_ruin < 0.05
    assert mc.median_total_r > 0


def test_verdict_too_few():
    s = summarize([0.5] * 10)
    v, _ = verdict(s, [0.5, 0.5, 0.5], min_sample=30)
    assert v == VERDICT_FEW


def test_verdict_negative():
    s = summarize([-0.5] * 40)
    v, _ = verdict(s, [-0.5, -0.5, -0.5], min_sample=30)
    assert v == VERDICT_NEG


def test_verdict_inconclusive_when_lower_bound_not_above_zero():
    rs = [5.0] * 8 + [-1.0] * 32          # positive mean, high variance -> ci_low <= 0
    s = summarize(rs)
    assert s.expectancy > 0 and s.ci_low <= 0
    v, _ = verdict(s, [0.5, -1.0, -1.0], min_sample=30)
    assert v == VERDICT_INCONCLUSIVE


def test_verdict_positive_consistent():
    s = summarize([0.5] * 40)
    v, _ = verdict(s, [0.5, 0.5, 0.5], min_sample=30)
    assert v == VERDICT_EV


def test_verdict_fallback_flags_no_null():
    s = summarize([0.5] * 40)
    v, notes = verdict(s, [0.5, 0.5, 0.5], min_sample=30)   # no null kwargs
    assert v == VERDICT_EV
    assert "no null baseline" in notes[0]


def test_verdict_null_rejects_when_real_below_null():
    # real expectancy positive but BELOW the null (the exit artifact) -> not +EV
    s = summarize([0.3] * 30 + [-0.1] * 30)            # mean +0.10R
    v, _ = verdict(s, [0.1, 0.1, 0.1], min_sample=30,
                   null_exp=0.12, null_excess_ci_low=-0.05)
    assert v == VERDICT_NEG                              # no edge beyond the artifact


def test_verdict_null_inconclusive_when_excess_not_significant():
    s = summarize([1.0] * 30 + [-1.0] * 30)            # mean 0, but force a small +excess scenario
    v, _ = verdict(summarize([0.6] * 30 + [-0.2] * 30), [0.2, 0.2, -0.1], min_sample=30,
                   null_exp=0.15, null_excess_ci_low=-0.02)
    assert v == VERDICT_INCONCLUSIVE


def test_evaluate_null_baseline_kills_exit_artifact():
    # The exit produces the SAME positive expectancy on real and on the null
    # (random-entry shadows) -> the setup has NO real edge -> must NOT be +EV.
    trades = [_t(0.25)] * 40 + [_t(-0.05)] * 20         # mean +0.15R
    nulls = ([_t(0.25)] * 400 + [_t(-0.05)] * 200)      # identical distribution
    prof = evaluate(trades, setup="x", n_combos_tested=1, cfg=BacktestConfig(), null_trades=nulls)
    assert prof.null_n == 600
    assert prof.verdict != VERDICT_EV
    assert prof.edge_vs_null <= 0.001


def test_evaluate_null_baseline_passes_real_edge():
    # Real entries clearly beat the random-entry baseline -> +EV.
    trades = [_t(1.0)] * 50 + [_t(-1.0)] * 10           # mean ~+0.67R
    nulls = [_t(0.1)] * 600                             # weak artifact only
    prof = evaluate(trades, setup="x", n_combos_tested=1, cfg=BacktestConfig(), null_trades=nulls)
    assert prof.verdict == VERDICT_EV
    assert prof.edge_vs_null > 0.4 and prof.edge_vs_null_ci_low > 0


def test_evaluate_builds_profile_with_regimes():
    trades = [SimpleNamespace(r=0.5, regime="up") for _ in range(20)] + \
             [SimpleNamespace(r=-0.3, regime="range") for _ in range(20)]
    prof = evaluate(trades, setup="trend_pullback", n_combos_tested=4, cfg=BacktestConfig())
    assert prof.setup == "trend_pullback"
    assert "up" in prof.by_regime and "range" in prof.by_regime
    assert prof.by_regime["up"].expectancy > 0
    assert prof.monte_carlo is not None


# --- P9/P10/P11 reporting layer -------------------------------------------- #
import numpy as np  # noqa: E402
from src.expectancy import bootstrap_ci  # noqa: E402


def test_summarize_distribution_shape_fields():
    rs = [-1.0, -1.0, -1.0, 3.0, 0.5]      # right-skewed (few big winners)
    s = summarize(rs)
    assert s.median_r == -1.0
    assert s.std_r > 0
    assert s.skew_r > 0                      # positive skew (the +3R tail)
    assert s.p10_r <= s.median_r <= s.p90_r


def test_bootstrap_ci_brackets_the_mean():
    rng = np.random.default_rng(0)
    rs = list(rng.normal(0.2, 1.0, 500))
    lo, hi = bootstrap_ci(rs, runs=1000)
    assert lo < 0.2 < hi
    assert bootstrap_ci([]) == (0.0, 0.0)


def test_monte_carlo_risk_distribution_fields():
    rng = np.random.default_rng(1)
    rs = list(rng.normal(0.1, 1.0, 200))
    mc = monte_carlo(rs, runs=500, ruin_drawdown_r=15.0)
    assert mc.median_max_drawdown_r >= 0
    assert mc.median_max_drawdown_r <= mc.p95_max_drawdown_r     # median DD ≤ 95th-pctile DD
    assert mc.expected_loss_streak >= 1


def test_evaluate_decomposes_gross_minus_cost_equals_net():
    # trades carrying gross_r + funding_r: NET = gross − exec − funding must reconcile.
    trades = [SimpleNamespace(r=0.30, gross_r=0.50, funding_r=0.05, regime="up") for _ in range(40)]
    prof = evaluate(trades, setup="x", n_combos_tested=1, cfg=BacktestConfig())
    assert prof.gross_expectancy == pytest.approx(0.50)
    assert prof.funding_cost_r == pytest.approx(0.05)
    assert prof.exec_cost_r == pytest.approx(0.15)               # 0.50 − 0.30 − 0.05
    assert prof.gross_expectancy - prof.exec_cost_r - prof.funding_cost_r == pytest.approx(0.30)
    assert prof.recent_n > 0 and prof.bootstrap_low <= prof.bootstrap_high


def test_evaluate_cost_decomposition_backward_compatible_without_gross():
    # bare trades (no gross_r/funding_r) → gross defaults to net, costs 0 (no crash).
    trades = [SimpleNamespace(r=0.10, regime="up") for _ in range(40)]
    prof = evaluate(trades, setup="x", n_combos_tested=1, cfg=BacktestConfig())
    assert prof.gross_expectancy == pytest.approx(0.10)
    assert prof.exec_cost_r == pytest.approx(0.0) and prof.funding_cost_r == pytest.approx(0.0)


# --- P7 time-weighted learning --------------------------------------------- #
from dataclasses import replace as _replace  # noqa: E402
from src.expectancy import decay_weights  # noqa: E402


def test_summarize_unweighted_matches_when_weights_none():
    rs = [1.0, -1.0, 2.0, -0.5, 0.3, -1.0, 1.5, -0.2]
    s = summarize(rs)
    assert s.expectancy == pytest.approx(float(np.mean(rs)))
    assert s.std_r == pytest.approx(float(np.std(rs, ddof=1)))
    assert s.expectancy_se == pytest.approx(float(np.std(rs, ddof=1)) / np.sqrt(len(rs)))


def test_decay_weights_favour_recent_and_off_when_disabled():
    ts = [1000.0, 1000.0 + 90 * 86400, 1000.0 + 180 * 86400]   # 180d, 90d, 0d old (ref=newest)
    w = decay_weights(ts, half_life_days=90)
    assert w[2] == pytest.approx(1.0) and w[1] == pytest.approx(0.5) and w[0] == pytest.approx(0.25)
    assert decay_weights(ts, half_life_days=0) is None          # disabled → no weighting
    assert decay_weights([0.0, 0.0], half_life_days=90) is None  # no timestamps → no weighting


def test_weighted_summarize_pulls_toward_recent_and_widens_se():
    rs = [-1.0, -1.0, 1.0, 1.0]
    base = 1_000_000.0
    ts = [base, base + 50 * 86400, base + 300 * 86400, base + 320 * 86400]   # losers old, winners recent
    w = decay_weights(ts, half_life_days=90)
    weighted, plain = summarize(rs, weights=w), summarize(rs)
    assert weighted.expectancy > plain.expectancy                # recent winners dominate
    assert weighted.win_rate > plain.win_rate                    # weighted win-rate also shifts


def _t_ts(r, regime, ts):
    return SimpleNamespace(r=r, regime=regime, entry_ts=ts, gross_r=r, funding_r=0.0)


def test_evaluate_time_decay_shifts_edge_toward_recent():
    base = 1_000_000.0
    # 40 old losers, 40 recent winners → unweighted ~0; time-decayed → positive
    trades = ([_t_ts(-0.5, "up", base + i * 86400) for i in range(40)]
              + [_t_ts(+0.5, "up", base + (200 + i) * 86400) for i in range(40)])
    flat = evaluate(trades, setup="x", n_combos_tested=1, cfg=BacktestConfig())
    decayed = evaluate(trades, setup="x", n_combos_tested=1,
                       cfg=_replace(BacktestConfig(), time_decay_enabled=True, time_decay_half_life_days=60.0))
    assert flat.overall.expectancy == pytest.approx(0.0, abs=1e-9)
    assert decayed.overall.expectancy > 0.10                     # recent winners now dominate the estimate
    assert decayed.by_regime["up"].expectancy > 0.10


# --- AUDIT FINDING 1: pooled order-dependence of sequence statistics ---------- #
def _t_full(r, ts):
    return SimpleNamespace(r=r, regime="up", entry_ts=ts, gross_r=r, funding_r=0.0)


def test_evaluate_is_order_invariant_and_chronological():
    # The audit's exact proof, now asserted as a fix: coin A = OLD winners, coin B = RECENT
    # losers. Coin-blocked A+B vs B+A must yield IDENTICAL results, and "recent" must mean
    # RECENT IN TIME (the losers), not "last coins appended".
    A = [_t_full(+0.5, 1_000_000 + i * 3600) for i in range(40)]      # old
    B = [_t_full(-0.5, 2_000_000 + i * 3600) for i in range(40)]      # recent
    ab = evaluate(A + B, setup="x", n_combos_tested=1, cfg=BacktestConfig())
    ba = evaluate(B + A, setup="x", n_combos_tested=1, cfg=BacktestConfig())
    assert ab.recent_expectancy == pytest.approx(ba.recent_expectancy)          # order-invariant
    assert ab.fold_expectancies == pytest.approx(ba.fold_expectancies)
    assert ab.recent_expectancy == pytest.approx(-0.5)                          # time-recent = losers
    assert ab.fold_expectancies[0] > 0 > ab.fold_expectancies[-1]               # chronological folds
    assert ab.overall.max_drawdown_r == pytest.approx(ba.overall.max_drawdown_r)  # equity-curve stats too


def test_evaluate_without_timestamps_keeps_input_order():
    # trades with no entry_ts (legacy/synthetic) → stable sort is a no-op; behaviour unchanged
    trades = [SimpleNamespace(r=r, regime="up") for r in (+1.0, -1.0, +1.0, -1.0, +0.5, -0.5)]
    p = evaluate(trades, setup="x", n_combos_tested=1, cfg=BacktestConfig())
    assert p.overall.n == 6 and p.recent_expectancy == pytest.approx(0.0)       # last third = (+0.5, -0.5)


# --- AUDIT FINDING 2: cross-coin correlation → measured design effect ---------- #
from src.expectancy import design_effect  # noqa: E402

_DAY = 86400.0


def test_design_effect_independent_is_near_one():
    rng = np.random.default_rng(7)
    # 12 'coins' × 60 days, INDEPENDENT r's within each day-bucket → ρ≈0 → DEFF≈1
    ts = [d * _DAY + c for d in range(60) for c in range(12)]
    rs = list(rng.normal(0, 1, len(ts)))
    deff, rho = design_effect(ts, rs, _DAY)
    assert rho < 0.15 and deff < 1.0 + 0.15 * 11 + 0.2


def test_design_effect_perfectly_clustered_deflates_to_bucket_count():
    rng = np.random.default_rng(8)
    # 12 coins share the SAME r each day (perfect within-day correlation) → DEFF ≈ m̄ = 12
    ts, rs = [], []
    for d in range(60):
        day_r = float(rng.normal(0, 1))
        for c in range(12):
            ts.append(d * _DAY + c)
            rs.append(day_r)
    deff, rho = design_effect(ts, rs, _DAY)
    assert rho > 0.95 and deff == pytest.approx(12.0, rel=0.05)


def test_design_effect_degenerate_inputs_are_independent():
    assert design_effect([0.0, 0.0, 0.0], [1, -1, 1], _DAY) == (1.0, 0.0)     # no timestamps
    assert design_effect([1e6], [0.5], _DAY) == (1.0, 0.0)                     # single trade
    assert design_effect([1e6, 1e6 + 10], [1, -1], 0) == (1.0, 0.0)            # bucket disabled
    assert design_effect([d * _DAY for d in range(10)], list(range(10)), _DAY) == (1.0, 0.0)  # all singletons


def test_evaluate_widens_ci_and_deflates_n_under_clustering():
    rng = np.random.default_rng(9)
    trades = []
    for d in range(50):
        day_r = float(rng.normal(0.2, 1.0))
        for c in range(10):                                     # 10 coins echo the day's move
            trades.append(SimpleNamespace(r=day_r + float(rng.normal(0, 0.05)), regime="up",
                                          entry_ts=d * _DAY + c * 60, gross_r=day_r, funding_r=0.0))
    on = evaluate(trades, setup="x", n_combos_tested=1, cfg=BacktestConfig())
    off = evaluate(trades, setup="x", n_combos_tested=1,
                   cfg=_replace(BacktestConfig(), deff_enabled=False))
    assert on.deff > 5.0                                        # heavy clustering detected
    assert on.overall.n_eff < on.overall.n / 5                  # far fewer independent obs
    assert on.overall.ci_low < off.overall.ci_low               # CI honestly wider
    assert off.overall.n_eff == pytest.approx(off.overall.n)    # disabled → independence


# --- AUDIT FINDING 4: block-bootstrap Monte-Carlo (honest streak/drawdown tails) ---- #
def test_block_mc_widens_drawdown_tails_under_persistent_clustering():
    # a LOSING REGIME (40 consecutive losses) — the autocorrelation that kills accounts.
    # iid permutation scatters it (p95 DD ~5R, ruin 0% — dangerously optimistic); the block
    # bootstrap preserves and can concatenate loss blocks (p95 DD ~40R, ruin >50% — honest).
    rs = [-1.0] * 40 + [0.7] * 160
    iid = monte_carlo(rs, runs=800, ruin_drawdown_r=15.0, seed=3, block=1)
    blk = monte_carlo(rs, runs=800, ruin_drawdown_r=15.0, seed=3)          # auto ≈ √200
    assert blk.p95_max_drawdown_r > 2 * iid.p95_max_drawdown_r              # honest tails
    assert blk.expected_loss_streak > iid.expected_loss_streak              # streaks survive
    assert blk.risk_of_ruin > iid.risk_of_ruin                              # ruin no longer hidden


def test_block_mc_legacy_iid_keeps_total_fixed_and_is_deterministic():
    rs = list(np.random.default_rng(1).normal(0.05, 1.0, 150))
    iid = monte_carlo(rs, runs=200, ruin_drawdown_r=15.0, seed=7, block=1)
    assert iid.median_total_r == pytest.approx(sum(rs))                     # permutation: total invariant
    a = monte_carlo(rs, runs=200, ruin_drawdown_r=15.0, seed=9)
    b = monte_carlo(rs, runs=200, ruin_drawdown_r=15.0, seed=9)
    assert a.p95_max_drawdown_r == b.p95_max_drawdown_r                     # seeded → reproducible
