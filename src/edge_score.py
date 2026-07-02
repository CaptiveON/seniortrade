"""Edge Score & Opportunity Board (the headline UX).

Fuses a coin's BACKTESTED edge (regime-matched, conservative lower bound) with its
LIVE present conditions into one honest score, and ranks the screener's survivors
into a board the user picks from. Only coins with a LIVE setup AND a proven edge
for it appear; an EMPTY board ("no proven edge — stand aside") is a valid,
frequent result.

  edge_score = lower-bound(current-regime expectancy) × present_multiplier
  present_multiplier = grade × freshness × tide-alignment × regime-match

The heavy backtest (EdgeProfile) is cached per (market, symbol, setup, tf) so cold
scans amortise into fast warm scans.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field

import numpy as np

from . import analysis as an
from . import backtest as bt
from . import expectancy as ex
from . import market_context as mc
from . import screener as scr
from . import setups as su
from .config import DATA_DIR, EdgeScoreConfig, Market

_CACHE_DIR = DATA_DIR / "edge_cache"


@dataclass
class Opportunity:
    symbol: str
    group: str
    side: str
    setup: str
    edge_score_r: float
    grade: str               # A–F from the edge score
    confidence: float        # 0..1
    confidence_tier: str     # high / mod / low
    fresh: str
    regime: str
    trustworthy_edge_r: float
    present_multiplier: float
    regime_n: int
    context_label: str | None = None    # P1: which proven context cell scored it (else None)
    p_positive: float = 0.0             # P2: P(expectancy>0) — shown alongside confidence (=P(beats null)×fold)


# --------------------------------------------------------------------------- #
# EdgeProfile cache (per market/symbol/tf)
# --------------------------------------------------------------------------- #
def _safe(sym: str) -> str:
    return sym.replace("/", "_").replace(":", "-")


def _cache_path(market: Market, symbol: str, tf: str):
    return _CACHE_DIR / f"{market.value}_{_safe(symbol)}_{tf}.json"


def _compact(profile) -> dict:
    o = profile.overall
    return {
        "setup": profile.setup,
        "verdict": profile.verdict,
        "fold_consistency": profile.fold_consistency,
        # P6 robustness inputs (report-only): bootstrap stability + walk-forward recency
        "bootstrap_low": profile.bootstrap_low,
        "bootstrap_high": profile.bootstrap_high,
        "recent_expectancy": profile.recent_expectancy,
        "recent_n": profile.recent_n,
        # audit finding 2: measured cross-coin correlation (SEs in the CIs below are √DEFF-widened)
        "deff": profile.deff,
        "corr_rho": profile.corr_rho,
        "overall": {"n": o.n, "n_eff": o.n_eff, "expectancy": o.expectancy, "ci_low": o.ci_low,
                    "win_rate": o.win_rate, "avg_win_r": o.avg_win_r, "avg_loss_r": o.avg_loss_r,
                    "std_r": o.std_r},
        "by_regime": {r: {"n": s.n, "n_eff": s.n_eff, "expectancy": s.expectancy, "ci_low": s.ci_low,
                          "win_rate": s.win_rate, "avg_win_r": s.avg_win_r, "avg_loss_r": s.avg_loss_r,
                          "std_r": s.std_r}
                      for r, s in profile.by_regime.items()},
        # null baseline (matched-geometry random entries) — gated by the gate/scorer
        "null_n": profile.null_n,
        "null_expectancy": profile.null_expectancy,
        "edge_vs_null": profile.edge_vs_null,
        "edge_vs_null_ci_low": profile.edge_vs_null_ci_low,
        "null_by_regime": {r: {"n": s.n, "expectancy": s.expectancy, "ci_low": s.ci_low}
                           for r, s in profile.null_by_regime.items()},
    }


def load_cached(market: Market, symbol: str, tf: str, ttl_hours: float) -> dict | None:
    path = _cache_path(market, symbol, tf)
    if not path.exists():
        return None
    if (time.time() - path.stat().st_mtime) > ttl_hours * 3600:
        return None
    try:
        return json.loads(path.read_text())["profiles"]
    except (ValueError, KeyError, OSError):
        return None


def save_cache(market: Market, symbol: str, tf: str, profiles: dict, coverage: int) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    compact = {name: _compact(p) for name, p in profiles.items()}
    payload = {"timestamp": time.time(), "tf": tf, "coverage": coverage, "profiles": compact}
    _cache_path(market, symbol, tf).write_text(json.dumps(payload, indent=2))


def ensure_profiles(market: Market, symbol: str, settings, tf: str, refresh: bool) -> dict:
    """Return compact EdgeProfiles for a coin, backtesting (and caching) if stale."""
    if not refresh:
        cached = load_cached(market, symbol, tf, settings.edge.cache_ttl_hours)
        if cached is not None:
            return cached
    profiles, _n, _q = bt.run_backtest(market, symbol, settings, tf)
    save_cache(market, symbol, tf, profiles, settings.backtest.candle_limit)
    return {name: _compact(p) for name, p in profiles.items()}


# --------------------------------------------------------------------------- #
# POOLED (universe) edge profiles — a setup's edge is a property of its RULES,
# proven on trades pooled ACROSS the liquid universe. This beats the per-coin
# "too few trades" wall: on one coin most setups never reach the sample minimum,
# but pooled across ~12 coins they do — and pooling is also less overfit.
# --------------------------------------------------------------------------- #
def _universe_path(market: Market, tf: str):
    return _CACHE_DIR / f"{market.value}_{tf}_UNIVERSE.json"


def load_universe(market: Market, tf: str, ttl_hours: float) -> dict | None:
    path = _universe_path(market, tf)
    if not path.exists():
        return None
    if (time.time() - path.stat().st_mtime) > ttl_hours * 3600:
        return None
    try:
        return json.loads(path.read_text())
    except (ValueError, OSError):
        return None


def _attach_shrinkage(compact: dict, setup: str, per_symbol_trades: dict, min_n: int) -> None:
    """P5: store per-symbol setup×regime stats {n,mean,sd} + the between-symbol variance τ²
    per regime (estimated only from coins with ≥ min_n own trades), so the scorer can shrink
    each coin's own edge toward the pooled prior. `min_n` is baked in so the board AND the
    stage gate (both read the cache) use the SAME activation floor — coherent by construction."""
    import numpy as _np
    by_sym: dict = {}
    for sym, trades_by_setup in per_symbol_trades.items():
        for t in trades_by_setup.get(setup, []):
            by_sym.setdefault(sym, {}).setdefault(t.regime, []).append(t.r)
    symbols_block: dict = {}
    for sym, regs in by_sym.items():
        symbols_block[sym] = {
            reg: {"n": len(rs), "mean": float(_np.mean(rs)),
                  "sd": float(_np.std(rs, ddof=1)) if len(rs) > 1 else 0.0}
            for reg, rs in regs.items()}
    compact["symbols"] = symbols_block
    compact["shrink_min_n"] = int(min_n)
    tau2: dict = {}
    for reg, pool_stats in compact.get("by_regime", {}).items():
        stats = [(s[reg]["n"], s[reg]["mean"], s[reg]["sd"]) for s in symbols_block.values() if reg in s]
        tau2[reg] = ex.between_group_var(stats, pool_stats["expectancy"], min_n)
    compact["tau2_by_regime"] = tau2


def build_pooled_profiles(market: Market, settings, tf: str, symbols: list,
                          hard: bool = False) -> tuple[dict, list, dict]:
    """Backtest each coin, POOL the trades per setup across coins, evaluate to one
    EdgeProfile per setup (overall + by-regime) — and retain per-symbol stats + τ² for
    hierarchical shrinkage. `hard` = exhaustive P4 pair search. Returns (compact_profiles,
    coins_used, feature_importance_trend [P13])."""
    bars_by = bt.fetch_history(market, symbols, settings, tf)
    from . import data_fetch as _df
    htf_ex = None
    try:                                            # one shared client for the bias-TF (HTF) fetches
        htf_ex = _df.make_exchange(market, settings.api_key, settings.api_secret)
        _df.load_markets(htf_ex)
    except Exception:
        htf_ex = None
    pooled: dict = {}
    pooled_null: dict = {}
    per_symbol_trades: dict = {}
    for seed, (sym, bars) in enumerate(bars_by.items()):
        htf = bt.htf_trend_aligned(market, sym, settings, bars.index, exch=htf_ex) if htf_ex else None
        res = bt.run_setups(bars, market, settings, tf, seed=seed, htf_trend=htf)
        per_symbol_trades[sym] = res.trades
        for name, trades in res.trades.items():
            pooled.setdefault(name, []).extend(trades)
        for name, nl in res.nulls.items():
            pooled_null.setdefault(name, []).extend(nl)
    # AUDIT FINDING 1 FIX: the pool is built coin-by-coin (coin-BLOCKED order). Every
    # order-sensitive statistic downstream (chronological folds → fold_consistency/OOS,
    # recent-third expectancy, walk-forward, pooled equity-curve drawdown/streak — in
    # evaluate AND evaluate_conditioning) must see TIME order, not screener order.
    for trades in pooled.values():
        trades.sort(key=lambda t: getattr(t, "entry_ts", 0.0) or 0.0)
    n_combos = len(su.DETECTORS)
    profiles: dict = {}
    for name, trades in pooled.items():
        compact = _compact(ex.evaluate(trades, setup=name, n_combos_tested=n_combos,
                                       cfg=settings.backtest, null_trades=pooled_null.get(name, [])))
        _attach_shrinkage(compact, name, per_symbol_trades, settings.edge.shrink_min_n)
        compact["archetype_table"] = _archetype_table(trades, compact["overall"], settings.edge.shrink_min_n)
        profiles[name] = compact
    # P13: reassess feature importance (singles-only), persist a snapshot + trend, and let the
    # TOP-K important features carry greater influence — only they seed the P4 pair search.
    fi = feature_importance(pooled, settings.edge, settings.edge.context_min_n)
    fi_trend = record_feature_importance(market, tf, fi)
    active = set(sorted(CONTEXT_FEATURES, key=lambda f: -fi[f]["importance"])[: settings.edge.feature_top_k])
    # P1/P3/P4: cache PROVEN context refinements (strict gate: money-maker + Bonferroni + OOS;
    # pairs must also beat their best single parent), each with the τ² to shrink it toward its
    # setup×regime parent at score time. `hard` switches pair search from greedy → exhaustive.
    refs = evaluate_conditioning(pooled, settings.edge, min_child_n=settings.edge.context_min_n,
                                 hard=hard, active_features=active)
    for r in refs:
        if not r["proven"]:
            continue
        prof = profiles.get(r["setup"])
        if prof is None:
            continue
        r = dict(r)
        feats = [c[0] for c in r["conditions"]]
        r["tau2"] = _context_tau2(pooled[r["setup"]], r["regime"], feats,
                                  prof["by_regime"][r["regime"]]["expectancy"], settings.edge.context_min_n)
        prof.setdefault("context_refinements", {}).setdefault(r["regime"], []).append(
            {k: r[k] for k in ("feature", "value", "conditions", "label", "interaction",
                               "n", "exp", "ci_low", "lift", "lift_low", "fold_consistency", "tau2")})
    return profiles, list(bars_by), fi_trend


def _archetype_table(trades, parent: dict, min_n: int) -> dict:
    """P3/P14 transparency (ALWAYS computed — evidence, not a gate): per-archetype conditional
    expectancy for a setup, each SHRUNK toward the setup's overall edge (P5 on the archetype
    axis). Lets the report say 'this setup earns +X only in <archetype>' even when no cell is
    strong enough to change sizing (that promotion runs through evaluate_conditioning's gate)."""
    cells: dict = {}
    for t in trades:
        a = t.context.get("arch")
        if a is not None:
            cells.setdefault(a, []).append(t.r)
    if not cells:
        return {}
    grand = float(parent.get("expectancy", 0.0))
    prior_se = max(0.0, grand - float(parent.get("ci_low", grand)))
    stats = [(len(rs), float(np.mean(rs)), float(np.std(rs, ddof=1)) if len(rs) > 1 else 0.0)
             for rs in cells.values()]
    tau2 = ex.between_group_var(stats, grand, min_n)
    out: dict = {}
    for arch, rs in cells.items():
        n = len(rs)
        mean = float(np.mean(rs))
        sd = float(np.std(rs, ddof=1)) if n > 1 else 0.0
        post_mean, post_se, w = ex.shrink(n, mean, sd, grand, prior_se, tau2)
        out[arch] = {"n": n, "raw_exp": mean, "exp": post_mean,
                     "ci_low": post_mean - post_se, "weight": w}
    return out


def _context_tau2(trades, regime: str, features, grand_mean: float, min_n: int) -> float:
    """Between-cell variance for a setup×regime conditioned on `features` (one feature, or a
    tuple for a P4 pair — cells are the JOINT value-combos) — the prior scatter used to shrink a
    proven cell toward its setup×regime parent (P5 on the context axis)."""
    feats = (features,) if isinstance(features, str) else tuple(features)
    cells: dict = {}
    for t in trades:
        if t.regime != regime:
            continue
        key = tuple(t.context.get(f) for f in feats)
        if all(k is not None for k in key):
            cells.setdefault(key, []).append(t.r)
    stats = [(len(rs), float(np.mean(rs)), float(np.std(rs, ddof=1)) if len(rs) > 1 else 0.0)
             for rs in cells.values()]
    return ex.between_group_var(stats, grand_mean, min_n)


def ensure_pooled_profiles(market: Market, settings, tf: str, symbols: list,
                           refresh: bool = False, hard: bool = False) -> dict:
    """Compact pooled profiles for the universe, (re)built and cached on staleness."""
    if not refresh:
        cached = load_universe(market, tf, settings.edge.cache_ttl_hours)
        if cached is not None:
            return cached.get("profiles", {})
    profiles, coins, fi_trend = build_pooled_profiles(market, settings, tf, symbols, hard=hard)
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _universe_path(market, tf).write_text(json.dumps(
        {"timestamp": time.time(), "tf": tf, "coins": coins, "profiles": profiles,
         "feature_importance": fi_trend}, indent=2))
    return profiles


def _shrunk_reg(profile: dict, symbol: str | None, regime: str) -> dict | None:
    """The regime stats to gate on: the symbol's OWN setup×regime edge SHRUNK toward the
    pooled prior (empirical-Bayes, P5). Falls back to the flat pool reg when there's no
    symbol data / on an old cache — so thin coins behave exactly as before. Keeps n = pool n
    (the estimate is anchored to the pool's evidence); carries own_n/weight for explainability."""
    pool = (profile.get("by_regime") or {}).get(regime)
    if not pool:
        return None
    min_n = int(profile.get("shrink_min_n", 5))
    own = ((profile.get("symbols") or {}).get(symbol) or {}).get(regime) if symbol else None
    if not own or own.get("n", 0) < min_n:               # thin / no own data / old cache → pool reg
        return dict(pool)                                  # (below the floor a noisy mean must NOT be trusted)
    prior_mean = float(pool["expectancy"])
    prior_se = max(0.0, prior_mean - float(pool["ci_low"]))
    tau2 = float((profile.get("tau2_by_regime") or {}).get(regime, 0.0))
    post_mean, post_se, w = ex.shrink(int(own["n"]), float(own["mean"]), float(own.get("sd", 0.0)),
                                      prior_mean, prior_se, tau2)
    return {"n": pool["n"], "n_eff": pool.get("n_eff"), "expectancy": post_mean, "ci_low": post_mean - post_se,
            "win_rate": pool.get("win_rate"), "avg_win_r": pool.get("avg_win_r"),
            "avg_loss_r": pool.get("avg_loss_r"), "std_r": pool.get("std_r"),
            "own_n": int(own["n"]), "own_mean": float(own["mean"]), "weight": w, "prior": prior_mean}


def resolve_context(bar_context: dict | None, direction: str) -> dict | None:
    """Turn the analyze bar-context (raw htf) into the direction-resolved context the cached
    refinements are keyed on (htf → htf_align for THIS trade direction)."""
    if not bar_context:
        return None
    from . import backtest as _bt
    out = {f: bar_context.get(f) for f in ("vol", "mom", "loc", "div", "arch")}
    out["htf_align"] = _bt._htf_align(bar_context.get("htf"), direction)
    return out


def _context_cell_reg(profile: dict, regime: str, context: dict | None, sig_z: float = 1.65) -> dict | None:
    """P1 (wired): if the coin's CURRENT context matches a PROVEN context refinement for this
    setup×regime, return that cell's reg SHRUNK toward the setup×regime parent (P5 on the context
    axis), tagged for explainability. Else None → fall back to the parent / per-symbol estimate.
    With the strict gate (build time) this only fires when a refinement genuinely earned it."""
    if not context:
        return None
    refs = (profile.get("context_refinements") or {}).get(regime)
    pool = (profile.get("by_regime") or {}).get(regime)
    if not refs or not pool:
        return None
    def _match(r) -> bool:                                # match ALL conditions (single or pair, P4)
        conds = r.get("conditions") or [[r["feature"], r["value"]]]
        return all(context.get(f) == v for f, v in conds)
    matches = [r for r in refs if _match(r)]
    if not matches:
        return None
    best = max(matches, key=lambda r: r["ci_low"])       # most-specific proven edge wins (pair > single if higher)
    prior_mean = float(pool["expectancy"])
    prior_se = max(0.0, prior_mean - float(pool["ci_low"]))
    cell_se = max(0.0, (float(best["exp"]) - float(best["ci_low"])) / sig_z) if sig_z > 0 else 0.0
    cell_sd = cell_se * math.sqrt(best["n"]) if best["n"] > 0 else 0.0
    post_mean, post_se, _ = ex.shrink(int(best["n"]), float(best["exp"]), cell_sd,
                                      prior_mean, prior_se, float(best.get("tau2", 0.0)))
    label = best.get("label") or (best["value"] if best["feature"] == "arch"
                                  else f"{best['feature']}={best['value']}")
    return {"n": pool["n"], "n_eff": pool.get("n_eff"), "expectancy": post_mean, "ci_low": post_mean - post_se,
            "win_rate": pool.get("win_rate"), "avg_win_r": pool.get("avg_win_r"),
            "avg_loss_r": pool.get("avg_loss_r"), "std_r": pool.get("std_r"),
            "context_label": label, "context_lift": float(best["lift"])}


def lookup_pooled_verdict(market: Market, tf: str, setup: str, regime: str,
                          symbol: str | None = None, context: dict | None = None) -> dict | None:
    """Read the cached universe verdict for a setup × regime (stale-tolerant). Regime stats =
    the most-specific PROVEN CONTEXT cell matching `context` (P1), else the PER-SYMBOL shrunk
    estimate (P5) for `symbol`, else the flat pool."""
    cached = load_universe(market, tf, ttl_hours=24 * 365)
    if not cached:
        return None
    prof = (cached.get("profiles") or {}).get(setup)
    if not prof:
        return None
    null_reg = (prof.get("null_by_regime") or {}).get(regime)
    reg = _context_cell_reg(prof, regime, context) or _shrunk_reg(prof, symbol, regime)
    rob = robustness((prof.get("overall") or {}).get("expectancy"), prof.get("fold_consistency", 0.0),
                     prof.get("bootstrap_low"), prof.get("bootstrap_high"), prof.get("recent_expectancy"))
    return {"overall": prof.get("overall"), "regime": reg,
            "verdict": prof.get("verdict"), "null": null_reg,
            "fold_consistency": prof.get("fold_consistency", 0.0), "robustness": rob}


def archetype_evidence(market: Market, tf: str, setup: str, arch: str | None) -> dict | None:
    """P3/P14 reporting: the setup's historical conditional expectancy in the coin's CURRENT
    archetype (shrunk toward the setup's overall edge), plus whether that archetype has earned
    a gated promotion (a PROVEN refinement). Pure read of the cache — None if unavailable."""
    if not arch:
        return None
    cached = load_universe(market, tf, ttl_hours=24 * 365)
    if not cached:
        return None
    prof = (cached.get("profiles") or {}).get(setup)
    if not prof:
        return None
    row = (prof.get("archetype_table") or {}).get(arch)
    if not row:
        return None
    proven = any(rf.get("feature") == "arch" and rf.get("value") == arch
                 for refs in (prof.get("context_refinements") or {}).values() for rf in refs)
    return {**row, "arch": arch, "proven": proven}


def sizing_inputs(market: Market, tf: str, setup: str, regime: str,
                  cfg: EdgeScoreConfig, symbol: str | None = None, context: dict | None = None) -> dict | None:
    """Inputs the effective-risk pipeline needs for a setup in the current regime (P12):
    the null-net edge (R), confidence in [0,1] (sample quality × fold-consistency), the per-trade
    σ_R (for the volatility target), and the Kelly win/avg-win/avg-loss. Reads the cached pooled
    verdict (context cell / shrunk to `symbol`) — None if no profile yet."""
    v = lookup_pooled_verdict(market, tf, setup, regime, symbol, context)
    if not v:
        return None
    reg = v.get("regime") or v.get("overall")
    if not reg:
        return None
    ok, edge, _ = null_adjusted_edge(reg, v.get("null"), cfg)
    n = int(reg.get("n", 0) or 0)
    ne = _n_eff(reg)                                               # audit finding 2: independent evidence
    sample_quality = ne / (ne + cfg.shrink_k) if ne else 0.0       # → 1 as the EFFECTIVE sample grows
    sc = statistical_confidence(reg, v.get("null"), float(v.get("fold_consistency", 0.0)), cfg)
    avg_loss = reg.get("avg_loss_r")
    return {
        "edge_r": edge if ok else 0.0,
        "confidence": sc["confidence"],
        "p_positive": sc["p_positive"],
        "p_beats_null": sc["p_beats_null"],
        "sample_quality": sample_quality,
        "regime_n": n,
        "sigma_r": reg.get("std_r"),
        "win_rate": reg.get("win_rate"),
        "avg_win_r": reg.get("avg_win_r"),
        "avg_loss_r": abs(avg_loss) if avg_loss is not None else None,   # Kelly wants the magnitude
    }


# edge gate (pure) — used by stage/analyze to keep discipline coherent end-to-end
EDGE_PROVEN = "proven"
EDGE_UNPROVEN = "unproven"
EDGE_NEGATIVE = "negative"


def null_adjusted_edge(reg: dict, null: dict | None, cfg: EdgeScoreConfig) -> tuple[bool, float, str]:
    """THE single coherent rule for "is this a real, money-making edge over random?"
    Used by BOTH the board scorer and the stage gate so they can never disagree.

    The null is a GATE plus an ARTIFACT subtractor — it can NEVER promote a money-loser
    ("loses less than a random trader" is still a loser) nor inflate the shown edge above
    what the setup actually earns. Returns (eligible, edge_R, reason)."""
    e = float(reg["expectancy"])
    ci = float(reg["ci_low"])
    if e <= 0:                                          # must MAKE money in this regime
        return False, e, f"loses money in this regime ({e:+.2f}R) — not an edge"
    ne = float((null or {}).get("expectancy", 0.0))
    nci = float((null or {}).get("ci_low", ne))
    informative = int((null or {}).get("n", 0)) >= cfg.min_regime_n
    if informative:                                     # must BEAT random with significance
        sig_z = getattr(cfg, "sig_z", 1.0)              # stricter than the 1-SE display bound
        se = math.sqrt(max(0.0, e - ci) ** 2 + max(0.0, ne - nci) ** 2)
        excess_low = (e - ne) - sig_z * se              # lower bound of the excess over the null
        if excess_low <= 0:
            return False, excess_low, (f"not distinguishable from random entries "
                                       f"(excess {e - ne:+.2f}R, lower bound {excess_low:+.2f}R)")
    # MAGNITUDE: conservative absolute edge (the lower bound when positive, else the point
    # estimate the significance test vouched for), minus only a POSITIVE (artifact) null —
    # never a negative one, and never above the point expectancy.
    base = ci if ci > 0 else e
    edge = min(e, base - max(ne, 0.0))
    if edge <= cfg.floor:
        return False, edge, f"edge {edge:+.2f}R ≤ floor {cfg.floor:+.2f}R after the null"
    return True, edge, f"proven edge {edge:+.2f}R over the random-entry baseline"


def _n_eff(reg: dict) -> float:
    """EFFECTIVE independent sample size (audit finding 2) — falls back to raw n on old caches."""
    return float(reg.get("n_eff") or reg.get("n", 0) or 0)


def edge_gate(verdict: dict | None, cfg: EdgeScoreConfig) -> tuple[str, str]:
    """Classify a setup's pooled edge for the current regime: proven / unproven /
    negative — so stage can BLOCK a money-loser and WARN on the unproven."""
    if verdict is None:
        return EDGE_UNPROVEN, "no universe edge profile yet — run `scan`; treat as discretionary"
    reg = verdict.get("regime") or verdict.get("overall")
    if not reg or _n_eff(reg) < cfg.min_regime_n:
        n = reg.get("n", 0) if reg else 0
        ne = _n_eff(reg) if reg else 0
        return EDGE_UNPROVEN, (f"only ~{ne:.0f} EFFECTIVE independent trades in this regime "
                               f"({n} pooled, cross-coin correlated; < {cfg.min_regime_n}) — unproven, discretionary")
    if reg["expectancy"] <= 0:                          # absolute money-loser — block (never on the null's account)
        return EDGE_NEGATIVE, (f"PROVEN −EV across the universe ({reg['expectancy']:+.2f}R over {reg['n']}) "
                               "— refusing to stage")
    ok, edge, reason = null_adjusted_edge(reg, verdict.get("null"), cfg)
    if ok:
        return EDGE_PROVEN, f"{reason} (over {reg['n']} pooled trades)"
    return EDGE_UNPROVEN, f"{reason} (over {reg['n']}) — discretionary"


# --------------------------------------------------------------------------- #
# P8 — EXPLAINABILITY. Assemble the WHY behind a recommendation from signals the
# pipeline ALREADY computed (lenses + the edge gate + freshness/tide/context) —
# no new heuristic, no new score. Every recommendation becomes auditable.
# --------------------------------------------------------------------------- #
def explain_signal(*, direction: str, lenses, edge_level: str, edge_msg: str, regime: str,
                   freshness: str, posture: str, context_label: str | None = None,
                   confidence_tier: str | None = None) -> dict:
    """Return {'positive': [...], 'negative': [...]} — the factors pushing FOR vs AGAINST the
    trade, strongest first. Pure: takes already-computed reads, invents nothing."""
    from .markets import LONG
    want = "bull" if direction == LONG else "bear"          # lens directions are lowercase
    pos: list[str] = []
    neg: list[str] = []
    for L in sorted(lenses or [], key=lambda x: -abs(getattr(x, "score", 0.0))):
        if abs(getattr(L, "score", 0.0)) < 0.10:
            continue
        line = f"{L.name}: {L.note}"
        d = (getattr(L, "direction", "") or "").lower()
        if d == want:
            pos.append(line)
        elif d not in ("neutral", want, ""):
            neg.append(line)
    # the edge verdict is the DECISIVE contributor — lead with it
    if edge_level == EDGE_PROVEN:
        pos.insert(0, f"Proven edge in the {regime} regime — {edge_msg}")
    elif edge_level == EDGE_NEGATIVE:
        neg.insert(0, f"Backtests −EV in the {regime} regime — {edge_msg}")
    else:
        neg.append(f"No proven edge yet in the {regime} regime — {edge_msg}")
    if context_label:
        pos.append(f"Context cell '{context_label}' earned its edge (money + Bonferroni + OOS)")
    if freshness == "fresh":
        pos.append("Fresh entry — not extended into the move")
    elif freshness in ("stretched", "extended"):
        neg.append(f"Entry {freshness} — chasing risk")
    if _tide_aligned(direction, posture):
        pos.append("Market tide (BTC regime / relative strength) supports this side")
    else:
        neg.append("Counter-tide — the broader market leans the other way")
    if confidence_tier == "high":
        pos.append("Statistical confidence: high (deep, fold-consistent sample)")
    elif confidence_tier == "low":
        neg.append("Statistical confidence: low (thin or inconsistent sample)")
    return {"positive": pos, "negative": neg}


# --------------------------------------------------------------------------- #
# The Edge Score
# --------------------------------------------------------------------------- #
_GRADE_FACTOR = {"A": 1.0, "B": 0.7, "C": 0.4}
_FRESH_FACTOR = {"fresh": 1.0, "stretched": 0.6, "extended": 0.2}


def statistical_confidence(reg: dict, null: dict | None, fold_consistency: float,
                           cfg: EdgeScoreConfig) -> dict:
    """P2 — EVIDENCE-DRIVEN confidence: 'how certain are we this edge actually EXISTS?'
    = P(edge beats the random-entry null) × fold-consistency. A single [0,1] number that folds in
    historical SAMPLE QUALITY + VARIANCE + the CONFIDENCE INTERVAL (all via the SE recovered from
    the cached 1-SE CI: SE = expectancy − ci_low, cfg.z=1.0), the PROBABILITY the edge is genuinely
    positive, and ROBUSTNESS across folds. Parametric Φ(mean/SE); also returns P(expectancy>0) for
    display (the brief's literal metric) and P(beats null). No null sample → falls back to P(>0)."""
    from statistics import NormalDist
    nd = NormalDist()
    e = float(reg.get("expectancy", 0.0))
    ci_low = float(reg.get("ci_low", e))
    se = max(1e-9, e - ci_low)                              # 1-SE band ⇒ SE = mean − lower bound
    p_positive = nd.cdf(e / se)
    ne = float((null or {}).get("expectancy", 0.0))
    nci = float((null or {}).get("ci_low", ne))
    if int((null or {}).get("n", 0)) >= cfg.min_regime_n:   # informative null → beats-random probability
        se_null = max(0.0, ne - nci)
        se_excess = math.sqrt(se * se + se_null * se_null) or 1e-9
        p_beats_null = nd.cdf((e - ne) / se_excess)
    else:
        p_beats_null = p_positive                           # no null evidence → beats-zero
    fold = max(0.0, min(1.0, float(fold_consistency or 0.0)))
    return {"confidence": p_beats_null * fold, "p_beats_null": p_beats_null,
            "p_positive": p_positive, "fold_consistency": fold}


def robustness(overall_exp, fold_consistency, bootstrap_low, bootstrap_high, recent_exp) -> dict:
    """P6 — 'profitable vs CONSISTENTLY profitable' in [0,1] (REPORT-ONLY — does NOT change the
    score/confidence/ranking). A product of three independent robustness axes, each ∈ [0,1]:
      cv           = OOS cross-validation (fraction of chronological folds positive)
      bootstrap    = distribution-free stability (resampled CI lower bound > 0 → 1; straddles 0 → ½; <0 → 0)
      walk_forward = recent-third vs overall edge held up (no decay)
    Any weak axis drags the score down — exactly 'reward robustness, not isolated performance.'"""
    cv = max(0.0, min(1.0, float(fold_consistency or 0.0)))
    if bootstrap_low is not None and bootstrap_low > 0:
        boot = 1.0
    elif bootstrap_high is not None and bootstrap_high <= 0:
        boot = 0.0
    else:
        boot = 0.5                                          # bootstrap CI straddles zero
    o = float(overall_exp or 0.0)
    if o > 0 and recent_exp is not None:
        wf = max(0.0, min(1.0, float(recent_exp) / o))      # recent collapse → low; holds up → ~1
    else:
        wf = 1.0                                            # non-positive overall → robustness moot
    score = cv * boot * wf
    label = ("consistently profitable" if (score >= 0.6 and o > 0)
             else "profitable but not robust" if o > 0 else "not an edge")
    return {"score": score, "cv": cv, "bootstrap": boot, "walk_forward": wf, "label": label}


def _conf_tier(c: float, cfg: EdgeScoreConfig) -> str:
    if c >= cfg.conf_high:
        return "high"
    if c >= cfg.conf_mod:
        return "mod"
    return "low"


def _letter(edge_r: float, cfg: EdgeScoreConfig) -> str:
    for thr, g in cfg.grade_bands:
        if edge_r >= thr:
            return g
    return "F"


def score_opportunity(
    *, setup: str, side: str, provisional_grade: str, freshness: str,
    tide_aligned: bool, regime: str, profile: dict, cfg: EdgeScoreConfig,
    symbol: str | None = None, context: dict | None = None,
) -> dict | None:
    """Score one live setup against its cached edge profile. None if no proven edge. Regime edge =
    most-specific PROVEN CONTEXT cell matching `context` (P1), else the PER-SYMBOL shrunk estimate
    for `symbol` (P5), else the flat pool."""
    reg = _context_cell_reg(profile, regime, context, cfg.sig_z) or _shrunk_reg(profile, symbol, regime)
    if not reg or _n_eff(reg) < cfg.min_regime_n:
        return None                                  # unproven in the current regime (effective n)
    null_reg = (profile.get("null_by_regime") or {}).get(regime)
    ok, trustworthy, _ = null_adjusted_edge(reg, null_reg, cfg)
    if not ok:
        return None                                  # money-loser, or not distinguishable from random

    grade_factor = _GRADE_FACTOR.get(provisional_grade, 0.5)
    fresh_factor = _FRESH_FACTOR.get(freshness, 0.6)
    tide_factor = 1.0 if tide_aligned else 0.5
    present = grade_factor * fresh_factor * tide_factor * 1.0   # regime-match = 1 (regime had edge)
    edge_score = trustworthy * present
    sc = statistical_confidence(reg, null_reg, float(profile.get("fold_consistency", 0.0)), cfg)
    return {
        "setup": setup, "side": side, "edge_score_r": edge_score,
        "grade": _letter(edge_score, cfg), "confidence": sc["confidence"],
        "confidence_tier": _conf_tier(sc["confidence"], cfg), "trustworthy_edge_r": trustworthy,
        "p_positive": sc["p_positive"], "p_beats_null": sc["p_beats_null"],
        "present_multiplier": present, "fresh": freshness, "regime": regime,
        "regime_n": reg["n"], "own_n": reg.get("own_n"), "shrink_weight": reg.get("weight"),
        "context_label": reg.get("context_label"),
    }


# --------------------------------------------------------------------------- #
# P1 — contextual conditioning EVALUATION. For each setup×regime, test every
# single-feature split (F=v vs the REST of the same cell) and keep only the
# refinements that EARN their specificity: F=v makes money AND beats the rest with
# significance AFTER a Bonferroni penalty for the number of splits tried. This is
# the honest "does context add edge?" measurement — run before any wiring.
# --------------------------------------------------------------------------- #
CONTEXT_FEATURES = ("vol", "mom", "loc", "div", "htf_align", "arch")


def _cell_label(conditions) -> str:
    """Human label for a refinement cell: bare archetype name, else feature=value, &-joined."""
    return " & ".join(v if f == "arch" else f"{f}={v}" for f, v in conditions)


def evaluate_conditioning(trades_by_setup: dict, cfg: EdgeScoreConfig, *,
                          min_child_n: int = 20, folds: int = 3, hard: bool = False,
                          singles_only: bool = False, active_features=None) -> list:
    """Test context refinements of each setup×regime — SINGLE features (P1/P3) AND FEATURE PAIRS
    (P4 'interactions'). A cell (one or more feature=value conditions) is PROVEN only when ALL
    hold: it MAKES MONEY (exp>0), BEATS THE REST with significance after a BONFERRONI penalty over
    every split tried, and is FOLD-CONSISTENT (OOS). A PAIR must ALSO beat its best single parent
    (genuine INTERACTION, not inheriting one feature's edge). Pair candidates come from a GREEDY
    screen — only pairs built on a raw-significant single — to avoid the combinatorial explosion
    the brief warns of; ``hard=True`` instead tests EVERY value-pair (exhaustive, heavier bar)."""
    from statistics import NormalDist
    from itertools import combinations

    def _split(ts, conditions):
        a = [t.r for t in ts if all(t.context.get(f) == v for f, v in conditions)]
        b = [t.r for t in ts if not all(t.context.get(f) == v for f, v in conditions)]
        return a, b

    candidates = []                  # (setup, regime, conditions, a_array, b_array)
    single_exp: dict = {}            # (setup, regime, feat, val) -> cell expectancy (parent lookup)
    promising: dict = {}             # (setup, regime) -> set of raw-significant (feat, val)
    for setup, trades in trades_by_setup.items():
        by_regime: dict = {}
        # stable time-sort so each cell's fold-consistency (the OOS gate on refinements)
        # is CHRONOLOGICAL even if a caller passes a coin-blocked pool (audit finding 1).
        for t in sorted(trades, key=lambda t: getattr(t, "entry_ts", 0.0) or 0.0):
            by_regime.setdefault(t.regime, []).append(t)
        for regime, ts in by_regime.items():
            if len(ts) < 2 * min_child_n:
                continue
            feat_vals = {f: {t.context.get(f) for t in ts if t.context.get(f) is not None}
                         for f in CONTEXT_FEATURES}
            # --- single-feature cells ---
            for feat, values in feat_vals.items():
                for v in values:
                    a, b = _split(ts, [(feat, v)])
                    if len(a) >= min_child_n and len(b) >= min_child_n:
                        candidates.append((setup, regime, ((feat, v),), np.asarray(a), np.asarray(b)))
                        ma = float(np.mean(a))
                        single_exp[(setup, regime, feat, v)] = ma
                        # raw-significant (pre-Bonferroni) → eligible to seed a greedy pair
                        sa, sb = np.asarray(a), np.asarray(b)
                        se = math.sqrt(sa.var(ddof=1) / len(sa) + sb.var(ddof=1) / len(sb)) \
                            if (len(sa) > 1 and len(sb) > 1) else math.inf
                        if ma > 0 and (ma - float(sb.mean())) - cfg.sig_z * se > 0:
                            promising.setdefault((setup, regime), set()).add((feat, v))
            # --- feature PAIRS (interactions) ---  (skipped for a singles-only importance pass)
            pair_feats = ([f for f in CONTEXT_FEATURES if active_features is None or f in active_features]
                          if not singles_only else [])
            for f1, f2 in combinations(pair_feats, 2):     # P13: only pair the IMPORTANT features
                for v1 in feat_vals[f1]:
                    for v2 in feat_vals[f2]:
                        seed = ((f1, v1) in promising.get((setup, regime), set())
                                or (f2, v2) in promising.get((setup, regime), set()))
                        if not hard and not seed:
                            continue                 # greedy: only pairs built on a promising single
                        conds = ((f1, v1), (f2, v2))
                        a, b = _split(ts, conds)
                        if len(a) >= min_child_n and len(b) >= min_child_n:
                            candidates.append((setup, regime, conds, np.asarray(a), np.asarray(b)))
    m = max(1, len(candidates))
    z_adj = NormalDist().inv_cdf(1 - 0.05 / m)                # one-sided Bonferroni over splits tried
    # pass 1 — metrics for every candidate
    recs = []
    single_lift_low: dict = {}        # (setup, regime, feat, val) -> the single cell's lift_low
    for setup, regime, conds, a, b in candidates:
        ma, mb = float(a.mean()), float(b.mean())
        se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)) if (len(a) > 1 and len(b) > 1) else math.inf
        cell_se = float(a.std(ddof=1) / math.sqrt(len(a))) if len(a) > 1 else math.inf
        lift = ma - mb
        lift_low = lift - z_adj * se
        fe = ex.fold_expectancies(a, folds)
        fold_consistency = (sum(1 for f in fe if f > 0) / len(fe)) if fe else 0.0
        rec = {"setup": setup, "regime": regime, "conds": conds, "ma": ma, "mb": mb, "cell_se": cell_se,
               "lift": lift, "lift_low": lift_low, "lift_low_raw": lift - cfg.sig_z * se, "n": int(len(a)),
               "fold_consistency": fold_consistency, "is_pair": len(conds) > 1}
        recs.append(rec)
        if len(conds) == 1:
            single_lift_low[(setup, regime, conds[0][0], conds[0][1])] = lift_low
    # pass 2 — proven gate. A PAIR must, beyond the single-cell gate, beat the BETTER of its two
    # single parents' significant edge-over-rest (lift_low) — genuine synergy, not inherited edge.
    out = []
    for r in recs:
        conds = r["conds"]
        parent_exp = parent_lift_low = float("-inf")
        if r["is_pair"]:
            parent_exp = max((single_exp.get((r["setup"], r["regime"], f, v), float("-inf")) for f, v in conds),
                             default=float("-inf"))
            parent_lift_low = max((single_lift_low.get((r["setup"], r["regime"], f, v), float("-inf"))
                                   for f, v in conds), default=float("-inf"))
        beats_parent = (not r["is_pair"]) or (r["lift_low"] > parent_lift_low)
        proven = (r["ma"] > 0 and r["lift_low"] > 0 and r["fold_consistency"] >= 0.5 and beats_parent)
        out.append({"setup": r["setup"], "regime": r["regime"],
                    "feature": conds[0][0], "value": conds[0][1],   # back-compat (1st condition)
                    "conditions": [list(c) for c in conds], "label": _cell_label(conds),
                    "interaction": r["is_pair"], "parent_exp": (parent_exp if r["is_pair"] else None),
                    "n": r["n"], "exp": r["ma"], "ci_low": r["ma"] - cfg.sig_z * r["cell_se"], "rest_exp": r["mb"],
                    "lift": r["lift"], "lift_low": r["lift_low"], "lift_low_raw": r["lift_low_raw"],
                    "fold_consistency": r["fold_consistency"], "proven": proven, "splits_tested": m})
    return sorted(out, key=lambda d: -d["lift_low"])


# --------------------------------------------------------------------------- #
# P13 — CONTINUOUS FEATURE EVALUATION. Periodically reassess which context
# features carry predictive value (from their single-feature splits), persist a
# snapshot each rebuild to track rise/decay, and let the IMPORTANT features carry
# greater influence (they seed the P4 pair search; decayed ones decline).
# --------------------------------------------------------------------------- #
def feature_importance(trades_by_setup: dict, cfg: EdgeScoreConfig, min_child_n: int) -> dict:
    """Per-context-feature predictive value from its SINGLE-feature splits across all setup×regime
    cells: n_significant (raw-significant positive directions = breadth), mean positive lift
    (depth), best Bonferroni-significant lift_low, n_proven. The `importance` scalar = breadth +
    capped depth — robust even when nothing clears the strict bar. (Singles-only; cheap.)"""
    refs = evaluate_conditioning(trades_by_setup, cfg, min_child_n=min_child_n, singles_only=True)
    out: dict = {}
    for f in CONTEXT_FEATURES:
        fr = [r for r in refs if r["feature"] == f]
        n_sig = sum(1 for r in fr if r["lift_low_raw"] > 0 and r["exp"] > 0)
        mean_lift = float(np.mean([max(0.0, r["lift"]) for r in fr])) if fr else 0.0
        best = max((r["lift_low"] for r in fr), default=0.0)
        n_proven = sum(1 for r in fr if r["proven"])
        out[f] = {"importance": float(n_sig) + min(mean_lift, 1.0), "n_significant": n_sig,
                  "mean_lift": mean_lift, "best_lift_low": float(best), "n_proven": n_proven,
                  "splits": len(fr)}
    return out


def load_feature_importance(market: Market, tf: str) -> dict | None:
    """P13: the feature-importance + trend readout stored in the universe cache (None if absent)."""
    cached = load_universe(market, tf, ttl_hours=24 * 365)
    return (cached or {}).get("feature_importance")


def _fi_history_path():
    return _CACHE_DIR.parent / "feature_importance_history.json"


def record_feature_importance(market: Market, tf: str, importance: dict) -> dict:
    """Append this build's importance snapshot to the history file and return a per-feature trend
    (rising / flat / decaying / new) vs the PREVIOUS snapshot for this market×tf — the continual
    'features gain and lose predictive power' signal. Side-effecting (writes once per rebuild)."""
    path = _fi_history_path()
    hist: list = []
    if path.exists():
        try:
            hist = json.loads(path.read_text())
        except (ValueError, OSError):
            hist = []
    prev = next((s for s in reversed(hist) if s.get("market") == market.value and s.get("tf") == tf), None)
    hist.append({"ts": time.time(), "market": market.value, "tf": tf,
                 "importance": {f: importance[f]["importance"] for f in importance}})
    try:
        _CACHE_DIR.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(hist[-200:], indent=2))
    except OSError:
        pass
    trend: dict = {}
    for f, d in importance.items():
        p = (prev or {}).get("importance", {}).get(f) if prev else None
        cur = d["importance"]
        t = ("new" if p is None else "rising" if cur > p + 0.5 else "decaying" if cur < p - 0.5 else "flat")
        trend[f] = {**d, "prev": p, "trend": t}
    return trend


def diff_boards(prev_symbols, opportunities) -> tuple[list[str], list[str]]:
    """New and dropped opportunity symbols vs the previous `watch` pass."""
    cur = {o.symbol for o in opportunities}
    new = [o.symbol for o in opportunities if o.symbol not in prev_symbols]
    dropped = sorted(s for s in prev_symbols if s not in cur)
    return new, dropped


def _tide_aligned(side: str, posture: str) -> bool:
    if posture == mc.NEUTRAL:
        return True
    return (side == "long" and posture == mc.RISK_ON) or \
           (side == "short" and posture == mc.RISK_OFF)


# --------------------------------------------------------------------------- #
# scan: screener -> edge-ranked Opportunity Board
# --------------------------------------------------------------------------- #
def _analyze_many(market: Market, symbols, settings, tf, workers: int, progress=None) -> dict:
    """Deep-analyze many coins CONCURRENTLY (per-thread exchange) — so a wide
    universe (all screen survivors) stays usable instead of dozens of serial reads."""
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from . import data_fetch as dfetch

    local = threading.local()

    def thread_ex():
        e = getattr(local, "ex", None)
        if e is None:
            e = dfetch.make_exchange(market, settings.api_key, settings.api_secret)
            dfetch.load_markets(e)
            local.ex = e
        return e

    def work(sym):
        try:
            return sym, an.analyze(market, sym, settings, tf_trigger=tf, ex=thread_ex())
        except Exception:
            return sym, None

    out: dict = {}
    done = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for sym, ar in pool.map(work, symbols):
            done += 1
            if progress:
                progress(sym, done, len(symbols))
            if ar is not None:
                out[sym] = ar
    return out


def scan(market: Market, settings, top: int | None = None, refresh: bool = False,
         progress=None, hard: bool = False) -> tuple:
    """Returns (market_context, [Opportunity ranked], [watchlist symbols]). `hard` = exhaustive
    P4 interaction (pair) search on rebuild (else greedy)."""
    cfg = settings.edge
    tf = cfg.tf
    result = scr.run_screen(market, settings.screener, settings.api_key, settings.api_secret)
    # POOLED universe edge profile (built once across the liquid survivors, cached) —
    # the verdict each coin's live setup is scored against.
    pool_syms = [c.symbol for c in result.candidates][: cfg.pool_coins]
    profiles = ensure_pooled_profiles(market, settings, tf, pool_syms, refresh, hard=hard)
    # Deep-scan ALL survivors by default (cfg.scan_top / `top` can cap it).
    limit = top or cfg.scan_top
    survivors = result.candidates[:limit] if limit else result.candidates
    analyses = _analyze_many(market, [c.symbol for c in survivors], settings, tf,
                             settings.screener.fetch_workers, progress)

    opportunities: list[Opportunity] = []
    watchlist: list[str] = []
    for c in survivors:
        ar = analyses.get(c.symbol)
        if ar is None or not ar.setups or ar.structure is None or ar.structure.state is None:
            watchlist.append(c.symbol)
            continue
        regime = ar.structure.state.trend
        best = None
        for sig in ar.setups:
            prof = profiles.get(sig.setup)
            if not prof:
                continue
            scored = score_opportunity(
                setup=sig.setup, side=sig.direction, provisional_grade=sig.grade,
                freshness=ar.freshness, tide_aligned=_tide_aligned(sig.direction, ar.posture),
                regime=regime, profile=prof, cfg=cfg, symbol=c.symbol,
                context=resolve_context(ar.context, sig.direction))
            if scored and (best is None or scored["edge_score_r"] > best["edge_score_r"]):
                best = scored
        if best is None:
            watchlist.append(c.symbol)
            continue
        opportunities.append(Opportunity(
            symbol=c.symbol, group=c.group, side=best["side"], setup=best["setup"],
            edge_score_r=best["edge_score_r"], grade=best["grade"], confidence=best["confidence"],
            confidence_tier=best["confidence_tier"], fresh=best["fresh"], regime=best["regime"],
            trustworthy_edge_r=best["trustworthy_edge_r"], present_multiplier=best["present_multiplier"],
            regime_n=best["regime_n"], context_label=best.get("context_label"),
            p_positive=best.get("p_positive", 0.0)))

    opportunities.sort(key=lambda o: o.edge_score_r, reverse=True)
    return result.context, opportunities, watchlist
