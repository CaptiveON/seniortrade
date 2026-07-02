"""Regression: the engine RECOVERS a planted edge and REJECTS pure noise.

A fast, single-tape version of proofs/planted_edge_lab.py — locks in that the null
baseline does not just suppress fabrication (test_backtest covers that) but also
RECOVERS a real, conditional, planted edge.
"""
from __future__ import annotations

from dataclasses import replace

from proofs.planted_edge_lab import control_pure_noise, make_noise, plant_setup, recover
from src.config import Market, Settings
from src.expectancy import VERDICT_EV

_S = replace(Settings(), backtest=replace(
    Settings().backtest, null_k=6, null_horizon=150, min_sample=20, null_min=30))


def test_recovers_planted_trend_pullback_edge():
    df = make_noise(1600, seed=11)
    df, planted = plant_setup(df, "trend_pullback", Market.USDM, _S, seed=11, edge_R=2.5)
    assert planted >= 20
    prof = recover(df, Market.USDM, _S, seed=11)["trend_pullback"]
    assert prof.verdict == VERDICT_EV, prof.notes
    assert prof.edge_vs_null_ci_low > 0          # real edge over the random-entry baseline


def test_pure_noise_pooled_is_not_proven():
    # The real system POOLS across the universe (large n) — that is what makes a
    # true-zero edge reliably fail the bound. Judged on a single tiny tape, the
    # z=1 lower bound can fluke +EV ~16% of the time; pooled it cannot.
    control = control_pure_noise(Market.USDM, _S, n_series=4, n_bars=1500)
    proven = [nm for nm, p in control.items() if p.verdict == VERDICT_EV]
    assert proven == [], f"fabricated edge on pure noise: {proven}"
