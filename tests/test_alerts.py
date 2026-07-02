"""Tests for alerts: level logic + persistence (no network)."""
from __future__ import annotations

from src.alerts import (
    LEVEL,
    SETUP,
    Alert,
    add_level_alert,
    add_setup_alert,
    level_triggered,
    list_alerts,
    remove_alert,
)
from src.config import Market


def test_level_triggered_up_and_down():
    up = Alert(id="1", created_at="t", market="usdm", symbol="X", kind=LEVEL, level=100.0, direction="up")
    assert level_triggered(up, 101.0) and not level_triggered(up, 99.0)
    down = Alert(id="2", created_at="t", market="usdm", symbol="X", kind=LEVEL, level=100.0, direction="down")
    assert level_triggered(down, 99.0) and not level_triggered(down, 101.0)


def test_add_level_alert_infers_direction(tmp_path):
    p = tmp_path / "a.jsonl"
    above = add_level_alert(Market.USDM, "BTC/USDT:USDT", 70000, current_price=64000, path=p)
    below = add_level_alert(Market.USDM, "BTC/USDT:USDT", 60000, current_price=64000, path=p)
    assert above.direction == "up"      # level above current -> wait for a move up
    assert below.direction == "down"
    assert len(list_alerts(path=p)) == 2


def test_setup_alert_and_remove(tmp_path):
    p = tmp_path / "a.jsonl"
    a = add_setup_alert(Market.USDM, "ETH/USDT:USDT", "4h", path=p)
    assert a.kind == SETUP and len(list_alerts(path=p)) == 1
    assert remove_alert(a.id, path=p)
    assert list_alerts(path=p) == []
    assert not remove_alert("missing", path=p)
