"""Layer 6 — adherence: discipline is MEASURED, not assumed.

Compares what was PLANNED to what was EXECUTED (from the journal) and scores
behaviour: stop-moved-against (the cardinal sin), oversizing, overrides (taking a
guard-blocked / sub-grade trade), and revenge trades (entering during cooldown).
adherence % = followed / total. Surfaced in `review`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .markets import LONG


@dataclass
class AdherenceRecord:
    planned_risk_pct: float
    actual_risk_pct: float
    planned_stop: float
    actual_stop: float          # the FINAL stop (after any moves)
    entry: float
    side: str
    grade: str = "B"
    guard_blocked: bool = False  # guards blocked it, but it was taken anyway
    in_cooldown: bool = False    # taken during a post-loss cooldown


@dataclass
class AdherenceReport:
    n: int
    followed: int
    adherence_pct: float
    stop_moved_against: int
    oversized: int
    overrides: int
    revenge: int
    violations: list = field(default_factory=list)


def _stop_moved_against(r: AdherenceRecord) -> bool:
    """Stop widened AWAY from entry (more risk) — the cardinal discipline sin."""
    if r.side == LONG:
        return r.actual_stop < r.planned_stop - 1e-12
    return r.actual_stop > r.planned_stop + 1e-12


def evaluate_adherence(records, oversize_tol: float = 0.10) -> AdherenceReport:
    n = len(records)
    sma = ov = orr = rev = followed = 0
    violations: list[str] = []
    for i, r in enumerate(records):
        flags = []
        if _stop_moved_against(r):
            sma += 1
            flags.append("stop-moved-against")
        if r.actual_risk_pct > r.planned_risk_pct * (1.0 + oversize_tol):
            ov += 1
            flags.append("oversized")
        if r.guard_blocked:
            orr += 1
            flags.append("override")
        if r.in_cooldown:
            rev += 1
            flags.append("revenge")
        if flags:
            violations.append(f"trade {i}: " + ", ".join(flags))
        else:
            followed += 1
    return AdherenceReport(
        n=n, followed=followed, adherence_pct=(followed / n * 100.0 if n else 0.0),
        stop_moved_against=sma, oversized=ov, overrides=orr, revenge=rev,
        violations=violations,
    )
