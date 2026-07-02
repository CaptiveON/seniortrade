"""Tests for adherence scoring (planned vs executed)."""
from __future__ import annotations

from src.adherence import AdherenceRecord, evaluate_adherence
from src.markets import LONG, SHORT


def _ok(**kw):
    base = dict(planned_risk_pct=1.0, actual_risk_pct=1.0, planned_stop=95.0,
                actual_stop=95.0, entry=100.0, side=LONG, grade="A",
                guard_blocked=False, in_cooldown=False)
    base.update(kw)
    return AdherenceRecord(**base)


def test_clean_run_full_adherence():
    rep = evaluate_adherence([_ok(), _ok(), _ok()])
    assert rep.adherence_pct == 100.0
    assert rep.stop_moved_against == 0


def test_stop_moved_against_long_and_short():
    rep = evaluate_adherence([
        _ok(actual_stop=92.0),                      # long: stop lowered = wider = against
        _ok(side=SHORT, planned_stop=105.0, actual_stop=108.0),  # short: stop raised = against
    ])
    assert rep.stop_moved_against == 2
    assert rep.adherence_pct == 0.0


def test_oversized_override_revenge_counted():
    rep = evaluate_adherence([
        _ok(actual_risk_pct=1.5),     # oversized (>1.0*1.1)
        _ok(guard_blocked=True),      # override
        _ok(in_cooldown=True),        # revenge
        _ok(),                        # clean
    ])
    assert rep.oversized == 1 and rep.overrides == 1 and rep.revenge == 1
    assert rep.followed == 1
    assert rep.adherence_pct == 25.0


def test_within_tolerance_not_oversized():
    rep = evaluate_adherence([_ok(actual_risk_pct=1.05)])   # within 10% tolerance
    assert rep.oversized == 0 and rep.adherence_pct == 100.0
