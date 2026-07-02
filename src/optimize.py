"""Step 3 — config optimization, done like a quant (the PROCESS is the proof).

"Ideal" does not mean "best numbers on history" (that is curve-fit, and it dies
live). It means: a parameter whose edge SURVIVED data it was never tuned on, with
the degradation honestly measured. So every choice here is built to make
overfitting impossible to hide, and the most common honest output is KEEP DEFAULT.

The method:
  1. Objective fixed up front: robust OOS expectancy = test expectancy penalised
     for cross-coin inconsistency (consistency beats peak; hardest to overfit).
  2. Out-of-sample wall: split each coin chronologically into TRAIN / TEST /
     LOCK-BOX. SELECT the parameter ONLY on TRAIN; VALIDATE on TEST the optimiser
     never saw; the LOCK-BOX is touched once, last, as the final honesty check.
  3. Pool trades ACROSS COINS per setup (a parameter must generalise; per-coin
     tuning is overfit-by-construction and starves the sample).
  4. Multiple-testing: the OOS edge must beat chance — a bootstrap significance
     test, Bonferroni-divided by the number of grid cells tried.
  5. Robustness: a stable PLATEAU (neighbours also +EV, not a lone spike) and
     survival under PESSIMISTIC costs.
  6. Decision: ADOPT only if TEST beats the default AND clears every gate; else
     KEEP DEFAULT. Output is a PROPOSED DIFF — never auto-applied.

Offline and deliberate (run via the `optimize` command); never in the live path.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from . import backtest as bt
from . import data_fetch as dfetch
from .config import Market


# --------------------------------------------------------------------------- #
# Pure, testable pieces
# --------------------------------------------------------------------------- #
def robust_score(rs, per_coin_expectancies, min_trades: int, penalty_lambda: float) -> float:
    """Objective: expectancy penalised for cross-coin inconsistency.

    Ineligible (−inf) below the minimum sample, so thin cells can never 'win'."""
    if len(rs) < min_trades:
        return float("-inf")
    exp = float(np.mean(rs))
    spread = float(np.std(per_coin_expectancies, ddof=1)) if len(per_coin_expectancies) > 1 else 0.0
    return exp - penalty_lambda * spread


def bootstrap_p_positive(rs, runs: int = 2000, seed: int = 0) -> float:
    """One-sided p-value that the mean is NOT > 0 (resampling the trade R's).

    p small  => the OOS edge is unlikely to be luck. Empty/degenerate => p = 1."""
    arr = np.asarray(rs, dtype=float)
    if len(arr) < 2:
        return 1.0
    rng = np.random.default_rng(seed)
    means = arr[rng.integers(0, len(arr), size=(runs, len(arr)))].mean(axis=1)
    return float(np.mean(means <= 0.0))


def plateau_ok(values, test_expectancies, best_value) -> bool:
    """A robust optimum is a PLATEAU: the chosen cell's grid neighbours are also
    non-negative. A lone +EV spike surrounded by losses is curve-fit → reject."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    pos = next((k for k, i in enumerate(order) if values[i] == best_value), None)
    if pos is None:
        return False
    neighbours = [order[k] for k in (pos - 1, pos + 1) if 0 <= k < len(order)]
    if not neighbours:
        return False
    return all(test_expectancies[i] >= 0.0 for i in neighbours)


@dataclass
class Decision:
    adopt: bool
    notes: list = field(default_factory=list)


def decide(*, best_value, default_value, best_test_exp, default_test_exp, p_value,
           alpha_adj, plateau, cost_stress_exp, lockbox_exp) -> Decision:
    """ADOPT only if OOS beats the default AND every honesty gate passes."""
    notes, gates = [], []

    def gate(ok, msg):
        gates.append(ok)
        notes.append(("✓ " if ok else "✗ ") + msg)

    gate(best_value != default_value, f"differs from default ({default_value:g})")
    gate(best_test_exp > default_test_exp,
         f"OOS test {best_test_exp:+.3f}R > default {default_test_exp:+.3f}R")
    gate(p_value < alpha_adj, f"OOS edge beats chance (p={p_value:.3f} < {alpha_adj:.4f}, multiple-testing adj.)")
    gate(plateau, "sits on a stable plateau (neighbours also ≥ 0)")
    gate(cost_stress_exp > 0.0, f"survives pessimistic costs ({cost_stress_exp:+.3f}R)")
    gate(lockbox_exp >= 0.0, f"lock-box not negative ({lockbox_exp:+.3f}R)")
    return Decision(adopt=all(gates), notes=notes)


# --------------------------------------------------------------------------- #
# Orchestration (network + compute; run offline via the `optimize` command)
# --------------------------------------------------------------------------- #
@dataclass
class Cell:
    value: float
    train_rs: list
    test_rs: list
    lock_rs: list
    train_score: float
    test_exp: float


@dataclass
class OptimizeReport:
    setup: str
    param: str
    coins: list
    grid: list
    default_value: float
    default_test_exp: float
    best_value: float
    best_test_exp: float
    p_value: float
    alpha_adj: float
    plateau: bool
    cost_stress_exp: float
    lockbox_exp: float
    decision: Decision
    cells: list = field(default_factory=list)


def _splits(n: int, train_frac: float, test_frac: float) -> tuple[float, float]:
    return n * train_frac, n * (train_frac + test_frac)


def _split_rs(trades, t_end: float, v_end: float):
    tr = [t.r for t in trades if t.entry_index < t_end]
    te = [t.r for t in trades if t_end <= t.entry_index < v_end]
    lk = [t.r for t in trades if t.entry_index >= v_end]
    return tr, te, lk


def _trades_for(bars_by_coin, market, settings, tf, setup_name, param, value):
    """Backtest every coin at this parameter value; return {coin: [Trade] for the setup}."""
    sv = replace(settings, setups=replace(settings.setups, **{param: value}))
    out = {}
    for sym, bars in bars_by_coin.items():
        out[sym] = bt.backtest_all(bars, market, sv, tf).get(setup_name, [])
    return out


def _grid_for(param: str, current: float, ocfg) -> list:
    if param == "stop_buffer_atr":
        return list(ocfg.stop_buffer_grid)
    return sorted({round(current * m, 6) for m in (0.5, 0.75, 1.0, 1.5, 2.0)})


def optimize_setup(market: Market, bars_by_coin: dict, settings, tf: str, setup_name: str,
                   param: str, ocfg) -> OptimizeReport:
    """Run the full honest optimisation for one setup × one parameter (pooled across coins)."""
    default_value = getattr(settings.setups, param)
    grid = _grid_for(param, default_value, ocfg)
    t_end_v = {sym: _splits(len(b), ocfg.train_frac, ocfg.test_frac) for sym, b in bars_by_coin.items()}

    cells: list[Cell] = []
    for value in grid:
        per_coin = _trades_for(bars_by_coin, market, settings, tf, setup_name, param, value)
        train_rs, test_rs, lock_rs, coin_train_exps = [], [], [], []
        for sym, trades in per_coin.items():
            t_end, v_end = t_end_v[sym]
            tr, te, lk = _split_rs(trades, t_end, v_end)
            train_rs += tr
            test_rs += te
            lock_rs += lk
            if tr:
                coin_train_exps.append(float(np.mean(tr)))
        cells.append(Cell(
            value=value, train_rs=train_rs, test_rs=test_rs, lock_rs=lock_rs,
            train_score=robust_score(train_rs, coin_train_exps, ocfg.min_trades, ocfg.penalty_lambda),
            test_exp=float(np.mean(test_rs)) if test_rs else float("nan")))

    # SELECT on TRAIN only (keeps TEST genuinely out-of-sample)
    eligible = [c for c in cells if c.train_score > float("-inf")]
    best = max(eligible, key=lambda c: c.train_score) if eligible else max(cells, key=lambda c: len(c.train_rs))
    default_cell = next((c for c in cells if c.value == default_value), None)
    default_test_exp = float(np.mean(default_cell.test_rs)) if (default_cell and default_cell.test_rs) else 0.0

    # VALIDATE the selected cell out-of-sample
    p_value = bootstrap_p_positive(best.test_rs, ocfg.bootstrap)
    alpha_adj = ocfg.alpha / max(1, len(grid))
    plat = plateau_ok([c.value for c in cells], [c.test_exp if c.test_exp == c.test_exp else -1.0 for c in cells], best.value)
    lockbox_exp = float(np.mean(best.lock_rs)) if best.lock_rs else 0.0

    # COST STRESS: re-validate the winner under pessimistic fees + slippage
    m = settings.markets
    stressed = replace(settings, markets=replace(
        m, slippage_bps=m.slippage_bps * ocfg.cost_mult,
        spot_taker=m.spot_taker * ocfg.cost_mult, usdm_taker=m.usdm_taker * ocfg.cost_mult))
    per_coin_s = _trades_for(bars_by_coin, market, stressed, tf, setup_name, param, best.value)
    cost_test = []
    for sym, trades in per_coin_s.items():
        t_end, v_end = t_end_v[sym]
        cost_test += _split_rs(trades, t_end, v_end)[1]
    cost_stress_exp = float(np.mean(cost_test)) if cost_test else 0.0

    decision = decide(
        best_value=best.value, default_value=default_value,
        best_test_exp=(best.test_exp if best.test_exp == best.test_exp else -1.0),
        default_test_exp=default_test_exp, p_value=p_value, alpha_adj=alpha_adj,
        plateau=plat, cost_stress_exp=cost_stress_exp, lockbox_exp=lockbox_exp)

    return OptimizeReport(
        setup=setup_name, param=param, coins=list(bars_by_coin), grid=grid,
        default_value=default_value, default_test_exp=default_test_exp,
        best_value=best.value, best_test_exp=(best.test_exp if best.test_exp == best.test_exp else float("nan")),
        p_value=p_value, alpha_adj=alpha_adj, plateau=plat, cost_stress_exp=cost_stress_exp,
        lockbox_exp=lockbox_exp, decision=decision, cells=cells)


def fetch_bars(market: Market, symbols, settings, tf: str) -> dict:
    """Fetch + clean history once per coin (reused across all grid cells)."""
    return bt.fetch_history(market, symbols, settings, tf)
