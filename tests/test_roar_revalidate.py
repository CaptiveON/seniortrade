"""roar --live re-validates each leg at send time (the regime-flip gap closed)."""
from __future__ import annotations

from types import SimpleNamespace

from src import edge_score as es
from src.cli import _leg_ok_to_send, _match_leg_setup


def _sig(setup="range_fade", direction="long"):
    return SimpleNamespace(setup=setup, direction=direction)


def test_match_leg_setup_finds_board_setup_not_just_first():
    # the board's setup (by edge score) may not be analyze's #1 (by provisional grade) —
    # the leg must still be matched, not falsely skipped as "setup changed" (live-caught bug)
    detected = [_sig("breakout_retest", "short"), _sig("trend_pullback", "short")]
    found = _match_leg_setup(detected, "trend_pullback", "short")
    assert found is not None and found.setup == "trend_pullback"
    assert _match_leg_setup(detected, "range_fade", "long") is None      # genuinely absent


def test_places_only_when_everything_revalidates():
    ok, reason = _leg_ok_to_send(_sig(), "range_fade", "long", True, es.EDGE_PROVEN, True, True)
    assert ok and "re-validated" in reason


def test_skips_when_edge_flipped_negative():
    # the live finding: a setup proven in `range` goes −EV when the coin tips into `down`
    ok, reason = _leg_ok_to_send(_sig(), "range_fade", "long", True, es.EDGE_NEGATIVE, True, True)
    assert not ok and "flipped" in reason


def test_skips_when_no_longer_proven():
    ok, reason = _leg_ok_to_send(_sig(), "range_fade", "long", True, es.EDGE_UNPROVEN, True, True)
    assert not ok and "PROVEN" in reason


def test_skips_when_setup_changed_or_gone():
    ok, reason = _leg_ok_to_send(_sig("breakout_retest", "short"), "range_fade", "long",
                                 True, es.EDGE_PROVEN, True, True)
    assert not ok and "changed" in reason
    assert not _leg_ok_to_send(None, "range_fade", "long", True, es.EDGE_PROVEN, True, True)[0]


def test_skips_on_drift_invalidation_or_invalid_plan():
    assert not _leg_ok_to_send(_sig(), "range_fade", "long", True, es.EDGE_PROVEN, False, True)[0]   # drift
    assert not _leg_ok_to_send(_sig(), "range_fade", "long", True, es.EDGE_PROVEN, True, False)[0]   # invalidation hit
    assert not _leg_ok_to_send(_sig(), "range_fade", "long", False, es.EDGE_PROVEN, True, True)[0]   # plan invalid
