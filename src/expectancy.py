"""Layer 5 — expectancy & the edge verdict.

Turns a list of backtested trades (realized R) into honest statistics and a
verdict. "Proven" (+EV) is earned only when expectancy is net-positive, its
LOWER confidence bound is > 0, the sample is big enough, AND the edge is
consistent across non-overlapping folds. Everything else is INCONCLUSIVE, not +EV.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

VERDICT_EV = "+EV"
VERDICT_INCONCLUSIVE = "INCONCLUSIVE"
VERDICT_NEG = "-EV"
VERDICT_FEW = "TOO FEW TRADES"


@dataclass
class Stats:
    n: int
    win_rate: float
    avg_win_r: float
    avg_loss_r: float
    expectancy: float
    expectancy_se: float
    ci_low: float
    ci_high: float
    profit_factor: float
    system_quality: float      # expectancy / SD(R)
    max_drawdown_r: float
    longest_loss_streak: int
    n_eff: float = 0.0         # EFFECTIVE independent sample size (n / design effect); = n when independent
    # P9 — distribution shape (supplements, doesn't replace, the above)
    median_r: float = 0.0
    std_r: float = 0.0         # SD of per-trade R (variance = std_r²)
    skew_r: float = 0.0        # Fisher-Pearson sample skew of R
    p10_r: float = 0.0         # 10th-percentile (downside) per-trade outcome
    p90_r: float = 0.0         # 90th-percentile (upside) per-trade outcome


@dataclass
class MonteCarlo:
    runs: int
    risk_of_ruin: float        # P(max drawdown >= ruin threshold)
    median_total_r: float
    p5_total_r: float
    p95_max_drawdown_r: float
    # P11 — risk intelligence (drawdown distribution, not just the historical max)
    median_max_drawdown_r: float = 0.0
    expected_loss_streak: float = 0.0   # mean longest losing streak across shuffles


@dataclass
class EdgeProfile:
    setup: str
    n_combos_tested: int
    overall: Stats
    by_regime: dict            # regime -> Stats
    fold_expectancies: list    # per-fold expectancy
    fold_consistency: float    # fraction of folds with positive expectancy
    monte_carlo: MonteCarlo | None
    verdict: str
    notes: list = field(default_factory=list)
    # NULL BASELINE (matched-geometry random entries) — the honest bar, not zero.
    null_n: int = 0
    null_expectancy: float = 0.0           # what the exit + drift earn with no edge
    edge_vs_null: float = 0.0              # overall.expectancy − null_expectancy
    edge_vs_null_ci_low: float = 0.0       # lower bound of that excess (the real test)
    null_by_regime: dict = field(default_factory=dict)   # regime -> null expectancy
    # P10 — separate EDGE from EXECUTION: raw statistical edge → costs → net tradable edge
    gross_expectancy: float = 0.0          # mean R BEFORE costs (the raw statistical edge)
    exec_cost_r: float = 0.0               # fees + slippage per trade (R)
    funding_cost_r: float = 0.0            # funding/carry per trade (R)
    # P9/P6 — robustness + time-awareness of the estimate
    bootstrap_low: float = 0.0             # non-parametric bootstrap CI on the expectancy
    bootstrap_high: float = 0.0
    recent_expectancy: float = 0.0         # mean R of the most-recent third (chronological)
    recent_n: int = 0
    # AUDIT FINDING 2 — measured cross-coin correlation of the pooled sample
    deff: float = 1.0                      # Kish design effect (1 = independent)
    corr_rho: float = 0.0                  # intra-time-bucket correlation of R


def _max_drawdown_r(rs: np.ndarray) -> float:
    if len(rs) == 0:
        return 0.0
    eq = np.cumsum(rs)
    peak = np.maximum.accumulate(eq)
    return float(np.max(peak - eq))


def _longest_loss_streak(rs) -> int:
    streak = best = 0
    for r in rs:
        if r < 0:
            streak += 1
            best = max(best, streak)
        else:
            streak = 0
    return best


def _skew(arr: np.ndarray) -> float:
    """Fisher-Pearson sample skewness (0 when n<3 or no spread)."""
    n = len(arr)
    if n < 3:
        return 0.0
    sd = float(arr.std(ddof=1))
    if sd <= 0:
        return 0.0
    return float((((arr - arr.mean()) / sd) ** 3).sum() * n / ((n - 1) * (n - 2)))


def bootstrap_ci(rs, runs: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Non-parametric bootstrap CI for the expectancy: resample trades WITH replacement and
    take percentiles of the resampled means. Robust to skew/fat tails (no normality assumed)."""
    arr = np.asarray(rs, dtype=float)
    n = len(arr)
    if n == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    means = arr[rng.integers(0, n, size=(runs, n))].mean(axis=1)
    return (float(np.percentile(means, 100 * alpha / 2)),
            float(np.percentile(means, 100 * (1 - alpha / 2))))


def summarize(rs, z: float = 1.0, weights=None) -> Stats:
    """Per-trade R → honest stats. With `weights` (P7 time-decay) the first/second MOMENTS are
    weighted (mean, SE via Kish effective-N, win-rate, avg win/loss, σ); order statistics
    (median, percentiles, max-DD, streak) stay on the realized path. weights=None reproduces the
    plain unweighted result exactly (ones → W=n, n_eff=n → ddof-1 SE)."""
    n = len(rs)
    if n == 0:
        return Stats(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    arr = np.asarray(rs, dtype=float)
    w = np.ones(n) if weights is None else np.clip(np.asarray(weights, dtype=float), 0.0, None)
    W = float(w.sum())
    if W <= 0:                                          # degenerate weights → fall back to unweighted
        w = np.ones(n); W = float(n)
    expectancy = float((w * arr).sum() / W)
    n_eff = float(W * W / float((w * w).sum())) if float((w * w).sum()) > 0 else float(n)
    if n_eff > 1:
        var_w = float((w * (arr - expectancy) ** 2).sum() / W) * n_eff / (n_eff - 1)
        sd = math.sqrt(max(0.0, var_w))
        se = sd / math.sqrt(n_eff)
    else:
        sd = se = 0.0
    wmask, lmask = arr > 0, arr < 0
    win_w, loss_w = float(w[wmask].sum()), float(w[lmask].sum())
    gross_win = float((w[wmask] * arr[wmask]).sum())
    gross_loss = float(-(w[lmask] * arr[lmask]).sum())
    pf = gross_win / gross_loss if gross_loss > 0 else (math.inf if gross_win > 0 else 0.0)
    return Stats(
        n=n,
        win_rate=win_w / W,
        avg_win_r=(gross_win / win_w) if win_w > 0 else 0.0,
        avg_loss_r=(-gross_loss / loss_w) if loss_w > 0 else 0.0,
        expectancy=expectancy,
        expectancy_se=se,
        ci_low=expectancy - z * se,
        ci_high=expectancy + z * se,
        profit_factor=pf,
        system_quality=expectancy / sd if sd > 0 else 0.0,
        max_drawdown_r=_max_drawdown_r(arr),
        longest_loss_streak=_longest_loss_streak(arr),
        n_eff=float(n),
        median_r=float(np.median(arr)),
        std_r=sd,
        skew_r=_skew(arr),
        p10_r=float(np.percentile(arr, 10)),
        p90_r=float(np.percentile(arr, 90)),
    )


def design_effect(entry_ts, rs, bucket_seconds: float) -> tuple[float, float]:
    """AUDIT FINDING 2 — cross-coin correlation. Pooled trades entered in the same time
    bucket (default: same UTC day) ride the same market move, so N pooled trades are FEWER
    than N independent observations. MEASURE it (never assume): one-way ANOVA intra-cluster
    correlation ρ over time buckets → Kish design effect DEFF = 1 + (m̄−1)·ρ.

    Returns (deff, rho). SEs are widened by √DEFF and gates use n_eff = n / DEFF.
    Degenerate inputs (no timestamps, <2 buckets, all-singleton buckets) → (1.0, 0.0):
    independence assumed only when clustering cannot be estimated."""
    ts = np.asarray(entry_ts, dtype=float)
    y = np.asarray(rs, dtype=float)
    if len(y) < 2 or bucket_seconds <= 0 or not np.any(ts > 0):
        return 1.0, 0.0
    buckets: dict = {}
    for t, r in zip(ts, y):
        buckets.setdefault(int(t // bucket_seconds), []).append(r)
    k, N = len(buckets), len(y)
    if k < 2 or k == N:                          # one bucket, or all singletons → no estimate
        return 1.0, 0.0
    grand = float(y.mean())
    groups = [np.asarray(v) for v in buckets.values()]
    msb = sum(len(g) * (float(g.mean()) - grand) ** 2 for g in groups) / (k - 1)
    ssw = sum(float(((g - g.mean()) ** 2).sum()) for g in groups)
    msw = ssw / (N - k) if N > k else 0.0
    m0 = (N - sum(len(g) ** 2 for g in groups) / N) / (k - 1)   # ANOVA average cluster size
    denom = msb + (m0 - 1) * msw
    rho = (msb - msw) / denom if denom > 0 else 0.0
    rho = min(1.0, max(0.0, rho))
    deff = 1.0 + (N / k - 1.0) * rho             # Kish: mean cluster size m̄ = N/k
    return max(1.0, deff), rho


def _inflate_stats(s: Stats, deff: float, z: float) -> None:
    """Widen a Stats' SE/CI by √DEFF and deflate its effective n (in place). No-op at deff≤1."""
    if deff <= 1.0 or s.n == 0:
        return
    s.expectancy_se *= math.sqrt(deff)
    s.ci_low = s.expectancy - z * s.expectancy_se
    s.ci_high = s.expectancy + z * s.expectancy_se
    s.n_eff = s.n / deff


def decay_weights(entry_ts, half_life_days: float, ref_ts: float | None = None):
    """P7 exponential time-decay weights from entry timestamps (epoch seconds): a trade
    `half_life_days` old weighs 0.5. ref_ts defaults to the most recent trade (= 'now')."""
    ts = np.asarray(entry_ts, dtype=float)
    if half_life_days <= 0 or ts.size == 0 or not np.any(ts > 0):
        return None
    ref = float(ts.max()) if ref_ts is None else float(ref_ts)
    age_days = np.clip((ref - ts) / 86400.0, 0.0, None)
    return 0.5 ** (age_days / half_life_days)


def _block_resample(arr: np.ndarray, block: int, rng) -> np.ndarray:
    """Circular block bootstrap: sample whole CONSECUTIVE blocks (wrapping) so local
    autocorrelation — loss streaks, clustered volatility — survives into the resample.
    An i.i.d. permutation scatters streaks and understates drawdown tails (audit finding 4)."""
    n = len(arr)
    if block <= 1 or block >= n:
        return rng.permutation(arr)
    k = -(-n // block)                                  # blocks needed to cover n
    starts = rng.integers(0, n, size=k)
    idx = (starts[:, None] + np.arange(block)[None, :]) % n
    return arr[idx.reshape(-1)[:n]]


def monte_carlo(rs, runs: int, ruin_drawdown_r: float, seed: int = 0, block: int = 0) -> MonteCarlo:
    """Drawdown/ruin distribution under resampling. ``block`` = 0 → AUTO circular block
    bootstrap (length ≈ √n, min 5) preserving autocorrelation (honest tails); 1 → legacy
    i.i.d. permutation (total R fixed, streaks scattered); ≥2 → explicit block length."""
    arr = np.asarray(rs, dtype=float)
    n = len(arr)
    if n == 0:
        return MonteCarlo(0, 0.0, 0.0, 0.0, 0.0)
    if block <= 0:
        block = max(5, int(round(math.sqrt(n))))
    rng = np.random.default_rng(seed)
    totals = np.empty(runs)
    maxdds = np.empty(runs)
    streaks = np.empty(runs)
    for k in range(runs):
        perm = _block_resample(arr, block, rng)
        eq = np.cumsum(perm)
        peak = np.maximum.accumulate(eq)
        maxdds[k] = float(np.max(peak - eq))
        totals[k] = float(eq[-1])
        streaks[k] = _longest_loss_streak(perm)
    return MonteCarlo(
        runs=runs,
        risk_of_ruin=float(np.mean(maxdds >= ruin_drawdown_r)),
        median_total_r=float(np.median(totals)),
        p5_total_r=float(np.percentile(totals, 5)),
        p95_max_drawdown_r=float(np.percentile(maxdds, 95)),
        median_max_drawdown_r=float(np.percentile(maxdds, 50)),
        expected_loss_streak=float(np.mean(streaks)),
    )


def consequence(win_rate: float, avg_win_r: float, avg_loss_r: float,
                n_trades: int = 20, sims: int = 2000, seed: int = 0) -> dict:
    """THE CONSEQUENCE CARD — what taking N trades LIKE THIS CELL does, simulated from its
    measured stats (win rate, avg win, avg loss). Bernoulli wins/losses; honest about
    dispersion, silent about nothing: median & 5th-percentile total R, P(you end negative),
    and the expected worst losing streak. Display converts R → $ at the user's risk%."""
    wr = min(1.0, max(0.0, float(win_rate or 0.0)))
    aw = float(avg_win_r or 0.0)
    al = -abs(float(avg_loss_r or 0.0))
    rng = np.random.default_rng(seed)
    wins = rng.random((sims, n_trades)) < wr
    rs = np.where(wins, aw, al)
    totals = rs.sum(axis=1)
    # worst losing streak per sim
    streaks = np.zeros(sims)
    run = np.zeros(sims)
    for j in range(n_trades):
        run = np.where(wins[:, j], 0.0, run + 1.0)
        streaks = np.maximum(streaks, run)
    return {"n_trades": n_trades,
            "median_r": float(np.median(totals)),
            "p5_r": float(np.percentile(totals, 5)),
            "p95_r": float(np.percentile(totals, 95)),
            "p_negative": float(np.mean(totals < 0)),
            "exp_worst_streak": float(np.mean(streaks))}


def fold_expectancies(rs, k: int) -> list[float]:
    arr = np.asarray(rs, dtype=float)
    if len(arr) == 0:
        return []
    if len(arr) < k:
        return [float(arr.mean())]
    return [float(f.mean()) for f in np.array_split(arr, k) if len(f)]


def between_group_var(stats, grand_mean: float, min_n: int = 2) -> float:
    """τ² — the BETWEEN-group variance of true means, net of within-group sampling noise
    (method-of-moments: observed spread of group means − the expected sampling spread).
    `stats` = iterable of (n, mean, sd) per group. τ²≈0 ⇒ groups don't differ beyond noise
    ⇒ shrink fully to the prior. Only groups with n ≥ min_n count (a handful of 2-trade means
    would otherwise fabricate a huge spurious τ²); needs ≥2 such groups, else 0."""
    valid = [(n, m, sd) for (n, m, sd) in stats
             if n is not None and n >= max(2, min_n) and sd is not None]
    if len(valid) < 2:
        return 0.0
    obs_var = float(np.mean([(m - grand_mean) ** 2 for (_, m, _) in valid]))
    samp_var = float(np.mean([(sd ** 2 / n) for (n, _, sd) in valid]))
    return max(0.0, obs_var - samp_var)


def shrink(own_n: int, own_mean: float, own_sd: float, prior_mean: float,
           prior_se: float, tau2: float) -> tuple[float, float, float]:
    """Empirical-Bayes posterior for ONE group's mean, shrunk toward a prior.

    prior ~ N(prior_mean, τ²) [τ² = between-group scatter]; likelihood ~ N(own_mean, σ²=own_sd²/own_n).
    Returns (posterior_mean, posterior_se, weight_on_own). The reported SE adds the prior-mean
    uncertainty (prior_se) so a group we've never observed is never scored MORE precisely than the
    prior. Recovers the prior EXACTLY when there's no own data AND no between-group signal (τ²=0)."""
    own_var = (own_sd ** 2 / own_n) if (own_n and own_n > 0 and own_sd and own_sd > 0) else math.inf
    own_prec = 0.0 if math.isinf(own_var) else 1.0 / own_var
    prior_prec = (1.0 / tau2) if tau2 > 0 else math.inf
    if own_prec == 0.0 and math.isinf(prior_prec):       # no own data, no scatter → exactly the prior
        return prior_mean, prior_se, 0.0
    if math.isinf(prior_prec):                            # τ²=0: groups identical → fully trust the prior mean
        return prior_mean, prior_se, 0.0
    total_prec = own_prec + prior_prec
    post_mean = (own_mean * own_prec + prior_mean * prior_prec) / total_prec
    post_var = 1.0 / total_prec
    post_se = math.sqrt(post_var + prior_se ** 2)         # + uncertainty in the prior mean itself
    weight = own_prec / total_prec
    return post_mean, post_se, weight


def verdict(overall: Stats, fold_exps, min_sample: int, *,
            null_exp: float | None = None, null_excess_ci_low: float | None = None) -> tuple[str, list]:
    """The edge verdict. When a NULL baseline is supplied (null_exp / the lower bound
    of the excess over it), "proven" means the edge beats the matched-geometry
    random-entry baseline with significance — NOT merely beats zero (which a
    path-dependent exit can do on pure noise). Without a null, falls back to the
    lower-bound-vs-zero test, flagged in the notes."""
    if overall.n < min_sample:
        return VERDICT_FEW, [f"{overall.n} trades < {min_sample} minimum — no verdict"]
    if overall.expectancy <= 0:
        return VERDICT_NEG, [f"expectancy {overall.expectancy:+.2f}R ≤ 0 — rejected"]
    consistency = (sum(1 for f in fold_exps if f > 0) / len(fold_exps)) if fold_exps else 0.0

    if null_excess_ci_low is not None:                      # NULL-baseline test (the honest bar)
        excess = overall.expectancy - (null_exp or 0.0)
        if excess <= 0:
            return VERDICT_NEG, [
                f"expectancy {overall.expectancy:+.2f}R ≤ null baseline {null_exp:+.2f}R "
                "— no edge beyond the exit artifact / directional drift"]
        if null_excess_ci_low <= 0:
            return VERDICT_INCONCLUSIVE, [
                f"beats null by {excess:+.2f}R but lower-bound {null_excess_ci_low:+.2f}R ≤ 0 "
                f"(not distinguishable from random entries; null {null_exp:+.2f}R)"]
        if consistency < 0.5:
            return VERDICT_INCONCLUSIVE, [f"only {consistency:.0%} of folds positive (not consistent)"]
        return VERDICT_EV, [
            f"edge over null {excess:+.2f}R, lower-bound {null_excess_ci_low:+.2f}R > 0 "
            f"(null {null_exp:+.2f}R); {consistency:.0%} of folds positive"]

    # fallback: no null available (e.g. a thin single-coin slice) -> zero baseline
    if overall.ci_low <= 0:
        return VERDICT_INCONCLUSIVE, [
            f"expectancy {overall.expectancy:+.2f}R but lower-bound {overall.ci_low:+.2f}R ≤ 0 "
            "(not distinguishable from zero; no null baseline)"]
    if consistency < 0.5:
        return VERDICT_INCONCLUSIVE, [f"only {consistency:.0%} of folds positive (not consistent)"]
    return VERDICT_EV, [
        f"lower-bound {overall.ci_low:+.2f}R > 0 (no null baseline); {consistency:.0%} of folds positive"]


def evaluate(trades, *, setup: str, n_combos_tested: int, cfg, null_trades=None) -> EdgeProfile:
    # Chronological order is a CONTRACT here: folds (OOS consistency), the recent-third
    # expectancy, and the equity-curve stats (max drawdown / streak) are all sequence
    # statistics. A STABLE sort by entry_ts enforces it for any caller (no-op when trades
    # are already time-ordered or carry no timestamps — e.g. synthetic test trades).
    trades = sorted(trades, key=lambda t: getattr(t, "entry_ts", 0.0) or 0.0)
    rs = [t.r for t in trades]
    # P7 — TIME-DECAY weights (opt-in). Reference 'now' = the most recent trade across ALL of
    # them, so a regime that hasn't occurred lately is downweighted as a whole. None → unweighted.
    weights = None
    if getattr(cfg, "time_decay_enabled", False) and trades:
        ref = max((getattr(t, "entry_ts", 0.0) or 0.0) for t in trades)
        weights = decay_weights([getattr(t, "entry_ts", 0.0) or 0.0 for t in trades],
                                cfg.time_decay_half_life_days, ref_ts=ref)
    overall = summarize(rs, cfg.z, weights)
    # P10 — gross (pre-cost) edge and the cost decomposition (getattr fallback keeps tests /
    # legacy callers that pass bare trades working: gross defaults to net, funding to 0).
    gross_rs = [getattr(t, "gross_r", t.r) for t in trades]
    fund_rs = [getattr(t, "funding_r", 0.0) for t in trades]
    _wnorm = weights if weights is not None else None
    gross_expectancy = float(np.average(gross_rs, weights=_wnorm)) if gross_rs else 0.0
    funding_cost_r = float(np.average(fund_rs, weights=_wnorm)) if fund_rs else 0.0
    exec_cost_r = (gross_expectancy - overall.expectancy) - funding_cost_r
    # P9/P6: bootstrap CI (robust to skew) + recent (last-third) expectancy for time-stability.
    boot_runs = getattr(cfg, "bootstrap_runs", 2000)
    boot_low, boot_high = bootstrap_ci(rs, boot_runs) if rs else (0.0, 0.0)
    rk = max(1, len(rs) // 3)
    recent = rs[-rk:] if rs else []
    recent_expectancy = float(np.mean(recent)) if recent else 0.0
    by_regime: dict = {}
    grouped: dict = {}
    for t in trades:
        grouped.setdefault(t.regime, []).append(t)
    for regime, ts in grouped.items():
        wsub = (decay_weights([getattr(t, "entry_ts", 0.0) or 0.0 for t in ts],
                              cfg.time_decay_half_life_days, ref_ts=ref) if weights is not None else None)
        by_regime[regime] = summarize([t.r for t in ts], cfg.z, wsub)
    folds = fold_expectancies(rs, cfg.folds)
    consistency = (sum(1 for f in folds if f > 0) / len(folds)) if folds else 0.0
    mc = monte_carlo(rs, cfg.mc_runs, cfg.ruin_drawdown_r,
                     block=getattr(cfg, "mc_block", 0)) if rs else None

    # AUDIT FINDING 2 — cross-coin correlation. Pooled trades are NOT independent (coins ride
    # the same market move); MEASURE the clustering per time bucket and widen every SE/CI by
    # √DEFF (n_eff = n/DEFF feeds the sample gates). Per-regime cells get their OWN estimate;
    # the null (no timestamps: shadows live on the same tapes/windows) inherits the setup's.
    deff, corr_rho = 1.0, 0.0
    if getattr(cfg, "deff_enabled", True):
        bucket_s = getattr(cfg, "deff_bucket_hours", 24.0) * 3600.0
        all_ts = [getattr(t, "entry_ts", 0.0) or 0.0 for t in trades]
        deff, corr_rho = design_effect(all_ts, rs, bucket_s)
        _inflate_stats(overall, deff, cfg.z)
        for regime, ts in grouped.items():
            d_reg, _ = design_effect([getattr(t, "entry_ts", 0.0) or 0.0 for t in ts],
                                     [t.r for t in ts], bucket_s)
            _inflate_stats(by_regime[regime], d_reg, cfg.z)

    # NULL baseline (matched-geometry random entries). The edge is the EXCESS over it.
    null_trades = null_trades or []
    null_rs = [t.r for t in null_trades]
    null_overall = summarize(null_rs, cfg.z) if null_rs else None
    null_grouped: dict = {}
    for t in null_trades:
        null_grouped.setdefault(t.regime, []).append(t.r)
    null_by_regime = {reg: summarize(v, cfg.z) for reg, v in null_grouped.items() if v}
    if deff > 1.0:                                 # nulls share the tapes → same correlation
        if null_overall is not None:
            _inflate_stats(null_overall, deff, cfg.z)
        for s_ in null_by_regime.values():
            _inflate_stats(s_, deff, cfg.z)
    null_min = getattr(cfg, "null_min", 0)
    use_null = null_overall is not None and null_overall.n >= null_min
    null_exp = null_overall.expectancy if null_overall else 0.0
    null_se = null_overall.expectancy_se if null_overall else 0.0
    edge_vs_null = overall.expectancy - null_exp
    se_excess = math.sqrt(overall.expectancy_se ** 2 + null_se ** 2)
    # SIGNIFICANCE bound at the stricter sig_z (multiple-testing / small-sample guard),
    # not the 1-SE display z — so a noisy small sample can't fluke "+EV".
    sig_z = getattr(cfg, "sig_z", cfg.z)
    edge_vs_null_ci_low = edge_vs_null - sig_z * se_excess

    v, vnotes = verdict(overall, folds, cfg.min_sample,
                        null_exp=(null_exp if use_null else None),
                        null_excess_ci_low=(edge_vs_null_ci_low if use_null else None))
    notes = list(vnotes)
    if n_combos_tested > 1 and v == VERDICT_EV:
        notes.append(f"multiple-testing: {n_combos_tested} setups tested — a lone +EV can be luck")
    return EdgeProfile(
        setup=setup, n_combos_tested=n_combos_tested, overall=overall, by_regime=by_regime,
        fold_expectancies=folds, fold_consistency=consistency, monte_carlo=mc,
        verdict=v, notes=notes,
        null_n=(null_overall.n if null_overall else 0), null_expectancy=null_exp,
        edge_vs_null=edge_vs_null, edge_vs_null_ci_low=edge_vs_null_ci_low,
        null_by_regime=null_by_regime,
        gross_expectancy=gross_expectancy, exec_cost_r=exec_cost_r, funding_cost_r=funding_cost_r,
        bootstrap_low=boot_low, bootstrap_high=boot_high,
        recent_expectancy=recent_expectancy, recent_n=len(recent),
        deff=deff, corr_rho=corr_rho,
    )
