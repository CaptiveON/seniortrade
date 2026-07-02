"""Tests for the intraday runtime: speed tiers and board diffing (no network)."""
from __future__ import annotations

from types import SimpleNamespace

from src.config import Settings, apply_tier, load_settings
from src.edge_score import diff_boards


def test_apply_tier_intraday():
    s = apply_tier(Settings(), "intraday")
    assert s.screener.screen_tf == "4h"
    assert s.analysis.tf_bias == "4h"
    assert s.analysis.tf_trigger == "1h"
    assert s.edge.tf == "1h"
    assert s.backtest.candle_limit == 1500


def test_apply_tier_swing():
    s = apply_tier(Settings(), "swing")
    assert s.screener.screen_tf == "1d"
    assert s.analysis.tf_trigger == "4h"


def test_apply_tier_none_or_unknown_unchanged():
    base = Settings()
    assert apply_tier(base, None) is base
    assert apply_tier(base, "nope") is base


def test_env_overrides_layer_on_top_of_tier(monkeypatch):
    # PRECEDENCE: defaults → tier preset → env on top. Explicit env WINS over the tier.
    monkeypatch.setenv("BACKTEST_CANDLE_LIMIT", "2000")
    s = load_settings(base=apply_tier(Settings(), "intraday"))
    assert s.edge.tf == "1h"                    # tier's 1h preset preserved
    assert s.backtest.candle_limit == 2000      # env 2000 beats the tier's 1500 (the fix)
    # env can even override the tier's timeframe
    monkeypatch.setenv("EDGE_TF", "15m")
    assert load_settings(base=apply_tier(Settings(), "intraday")).edge.tf == "15m"


def test_load_settings_base_none_equals_defaults_plus_env(monkeypatch):
    monkeypatch.delenv("BACKTEST_CANDLE_LIMIT", raising=False)
    assert load_settings().backtest.candle_limit == load_settings(base=Settings()).backtest.candle_limit == 1000


def test_diff_boards_new_and_dropped():
    opps = [SimpleNamespace(symbol="A"), SimpleNamespace(symbol="B")]
    new, dropped = diff_boards({"B", "C"}, opps)
    assert new == ["A"]          # A wasn't on the previous board
    assert dropped == ["C"]      # C was, and is gone now


def test_diff_boards_first_pass_all_new():
    opps = [SimpleNamespace(symbol="X"), SimpleNamespace(symbol="Y")]
    new, dropped = diff_boards(set(), opps)
    assert new == ["X", "Y"] and dropped == []
