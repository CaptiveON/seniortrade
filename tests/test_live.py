"""Tests for step-18 live execution — against a FAKE exchange. No network, no real orders.

Proves the safety-critical logic: the placement SEQUENCE, the never-naked
stop-or-bail, partial-fill stop sizing, preflight gating, reconcile, and flatten.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.config import LiveConfig, Market, Settings
from src.live import (
    LiveError,
    flatten_all,
    place_entry_with_protection,
    preflight,
    reconcile_live,
)
from src.order import OrderLeg, OrderTicket

FAST = LiveConfig(fill_timeout_s=0.0, fill_poll_s=0.0)


class FakeMarket:
    def round_amount(self, s, a):
        return round(a, 6)

    def round_price(self, s, p):
        return round(p, 2)


class FakeExchange:
    def __init__(self, fill="full", stop_fails=False, positions=None,
                 open_orders=None, withdrawals=False, free_balance=None, entry_fails=False):
        self.fill = fill
        self.stop_fails = stop_fails
        self.calls = []           # ordered method log
        self.created = []         # create_order payloads
        self.cancelled = []
        self._oid = 0
        self._req_amt = 0.0
        self.positions = positions or []
        self.open_orders = open_orders or {}
        self.restrictions = {"enableWithdrawals": withdrawals}
        self.free_balance = free_balance     # None → no fetch_balance support
        self.entry_fails = entry_fails       # simulate an exchange rejection on the entry send

    def set_margin_mode(self, mode, symbol):
        self.calls.append(("set_margin_mode", mode, symbol))

    def set_leverage(self, lev, symbol):
        self.calls.append(("set_leverage", lev, symbol))

    def create_order(self, symbol, type, side, amount, price=None, params=None):
        params = params or {}
        if self.entry_fails and not params.get("reduceOnly"):    # the ENTRY send is rejected
            raise RuntimeError('binanceusdm {"code":-2019,"msg":"Margin is insufficient."}')
        self._oid += 1
        oid = f"o{self._oid}"
        self.calls.append(("create_order", type, side, amount, bool(params.get("reduceOnly"))))
        self.created.append(dict(id=oid, type=type, side=side, amount=amount, params=params))
        if type.lower().startswith("stop") and params.get("reduceOnly") and self.stop_fails:
            raise RuntimeError("stop rejected by exchange")
        if not params.get("reduceOnly") and type in ("market", "limit"):
            self._req_amt = amount
        return {"id": oid}

    def fetch_balance(self):
        if self.free_balance is None:
            raise RuntimeError("balance unavailable")
        return {"USDT": {"free": self.free_balance}, "free": {"USDT": self.free_balance},
                "info": {"availableBalance": str(self.free_balance)}}

    def fetch_order(self, oid, symbol):
        amt = self._req_amt
        if self.fill == "full":
            return {"id": oid, "status": "closed", "filled": amt, "amount": amt, "average": 100.0}
        if self.fill == "partial":
            return {"id": oid, "status": "open", "filled": amt * 0.4, "amount": amt, "average": 100.0}
        return {"id": oid, "status": "open", "filled": 0.0, "amount": amt, "average": None}

    def cancel_order(self, oid, symbol):
        self.cancelled.append(oid)

    def fetch_positions(self):
        return self.positions

    def fetch_open_orders(self, symbol):
        return self.open_orders.get(symbol, [])

    def sapi_get_account_apirestrictions(self):
        return self.restrictions


def _ticket(leverage=3.0):
    legs = [OrderLeg("entry", "market", "sell", 72.0, False, 1.0),
            OrderLeg("stop", "stop_market", "buy", 73.0, True, 1.0),
            OrderLeg("tp1", "take_profit_market", "buy", 70.0, True, 0.5),
            OrderLeg("tp2", "take_profit_market", "buy", 68.0, True, 0.5)]
    return OrderTicket(symbol="SOL/USDT:USDT", market="usdm", leverage=leverage,
                       margin_mode="isolated", legs=legs)


def _create_kinds(ex):
    return [(c["type"], c["side"], c["amount"], bool(c["params"].get("reduceOnly"))) for c in ex.created]


# --- the placement SEQUENCE ------------------------------------------------- #
def test_sequence_leverage_then_entry_then_stop_then_tps():
    ex = FakeExchange(fill="full")
    res = place_entry_with_protection(ex, FakeMarket(), "SOL/USDT:USDT", _ticket(), size=2.0, live_cfg=FAST)
    methods = [c[0] for c in ex.calls]
    # leverage + isolated come BEFORE the first order
    assert methods.index("set_leverage") < methods.index("create_order")
    assert "set_margin_mode" in methods
    # entry (not reduceOnly) precedes the protective stop (reduceOnly)
    creates = _create_kinds(ex)
    assert creates[0] == ("market", "sell", 2.0, False)         # entry
    assert creates[1] == ("STOP_MARKET", "buy", 2.0, True)      # stop sized to filled
    assert creates[2][0] == "TAKE_PROFIT_MARKET" and creates[3][0] == "TAKE_PROFIT_MARKET"
    assert set(res.order_ids) == {"entry", "stop", "tp1", "tp2"}
    assert res.filled_qty == 2.0 and res.avg_price == 100.0


def test_never_naked_stop_failure_emergency_closes():
    ex = FakeExchange(fill="full", stop_fails=True)
    with pytest.raises(LiveError, match="STOP FAILED"):
        place_entry_with_protection(ex, FakeMarket(), "SOL/USDT:USDT", _ticket(), size=2.0, live_cfg=FAST)
    # last action must be an emergency market reduceOnly close — no position left naked
    last = ex.created[-1]
    assert last["type"] == "market" and last["params"].get("reduceOnly") and last["side"] == "buy"


def test_partial_fill_sizes_stop_to_filled_qty():
    ex = FakeExchange(fill="partial")
    res = place_entry_with_protection(ex, FakeMarket(), "SOL/USDT:USDT", _ticket(), size=2.0, live_cfg=FAST)
    assert res.filled_qty == pytest.approx(0.8)                  # 40% of 2.0
    stop = next(c for c in ex.created if c["type"] == "STOP_MARKET")
    assert stop["amount"] == pytest.approx(0.8)                  # stop matches what filled
    assert ex.cancelled                                          # resting remainder cancelled


def test_entry_not_filled_cancels_and_aborts():
    ex = FakeExchange(fill="none")
    with pytest.raises(LiveError, match="did not fill"):
        place_entry_with_protection(ex, FakeMarket(), "SOL/USDT:USDT", _ticket(), size=2.0, live_cfg=FAST)
    assert ex.cancelled and len(ex.created) == 1                 # only the entry was attempted


def test_margin_preflight_blocks_before_sending():
    # notional 72×2=144, /3x ×1.05 ≈ 50.4 needed; only 40 free → blocked, NOTHING sent
    ex = FakeExchange(fill="full", free_balance=40.0)
    with pytest.raises(LiveError, match="insufficient margin"):
        place_entry_with_protection(ex, FakeMarket(), "SOL/USDT:USDT", _ticket(), size=2.0, live_cfg=FAST)
    assert ex.created == []                                       # no order ever reached the exchange


def test_margin_preflight_passes_when_funded():
    ex = FakeExchange(fill="full", free_balance=1000.0)
    res = place_entry_with_protection(ex, FakeMarket(), "SOL/USDT:USDT", _ticket(), size=2.0, live_cfg=FAST)
    assert res.filled_qty == 2.0 and set(res.order_ids) == {"entry", "stop", "tp1", "tp2"}


def test_entry_rejection_is_wrapped_not_raw():
    # the exact failure the user hit: -2019 Margin insufficient on the ENTRY send
    ex = FakeExchange(fill="full", entry_fails=True)             # no fetch_balance → preflight skipped
    with pytest.raises(LiveError, match="entry order REJECTED"):
        place_entry_with_protection(ex, FakeMarket(), "SOL/USDT:USDT", _ticket(), size=2.0, live_cfg=FAST)
    assert ex.created == []                                       # nothing filled → nothing to protect, nothing naked


# --- preflight gating ------------------------------------------------------- #
def _settings(**kw):
    return Settings(api_key=kw.get("key", "k"), api_secret=kw.get("secret", "s"))


def test_preflight_blocks_unarmed_real_money(monkeypatch):
    monkeypatch.delenv("LIVE_TRADING_ENABLED", raising=False)
    pf = preflight(_settings(), Market.USDM, testnet=False)
    assert not pf.ok and any("armed" in b for b in pf.blocks)


def test_preflight_testnet_ok_without_arming(monkeypatch):
    monkeypatch.delenv("LIVE_TRADING_ENABLED", raising=False)
    pf = preflight(_settings(), Market.USDM, testnet=True)
    assert pf.ok and pf.testnet


def test_preflight_armed_real_money_ok(monkeypatch):
    monkeypatch.setenv("LIVE_TRADING_ENABLED", "I_UNDERSTAND_THE_RISK")
    pf = preflight(_settings(), Market.USDM, testnet=False)
    assert pf.ok


def test_preflight_blocks_withdrawal_enabled_key(monkeypatch):
    monkeypatch.setenv("LIVE_TRADING_ENABLED", "I_UNDERSTAND_THE_RISK")
    ex = FakeExchange(withdrawals=True)
    pf = preflight(_settings(), Market.USDM, testnet=False, ex=ex)
    assert not pf.ok and any("withdrawals" in b.lower() for b in pf.blocks)


def test_preflight_no_keys_blocks():
    pf = preflight(Settings(), Market.USDM, testnet=True)
    assert not pf.ok and any("no API key" in b for b in pf.blocks)


# --- reconcile (exchange is the source of truth) ---------------------------- #
def _rec(symbol, mode="live"):
    return SimpleNamespace(symbol=symbol, status="open", mode=mode)


def test_reconcile_flags_closed_naked_and_ok():
    ex = FakeExchange(
        positions=[{"symbol": "BTC/USDT:USDT", "contracts": 0.5, "side": "long"},
                   {"symbol": "ETH/USDT:USDT", "contracts": 2.0, "side": "long"}],
        open_orders={"BTC/USDT:USDT": [{"id": "s1", "type": "STOP_MARKET", "reduceOnly": True}]})
    recs = [_rec("BTC/USDT:USDT"), _rec("ETH/USDT:USDT"), _rec("SOL/USDT:USDT"),
            _rec("XRP/USDT:USDT", mode="dry-run")]
    found = {f.symbol: f.kind for f in reconcile_live(ex, recs)}
    assert found["BTC/USDT:USDT"] == "ok"        # position + resting stop
    assert found["ETH/USDT:USDT"] == "naked"     # position, no stop
    assert found["SOL/USDT:USDT"] == "closed"    # no position on the exchange
    assert "XRP/USDT:USDT" not in found          # dry-run isn't reconciled live


# --- kill switch ------------------------------------------------------------ #
def test_flatten_closes_positions_and_cancels_orders():
    ex = FakeExchange(
        positions=[{"symbol": "BTC/USDT:USDT", "contracts": 0.5, "side": "long"}],
        open_orders={"BTC/USDT:USDT": [{"id": "s1"}, {"id": "t1"}]})
    summary = flatten_all(ex, FakeMarket())
    assert summary["closed"] == ["BTC/USDT:USDT"]
    assert set(summary["cancelled"]) == {"s1", "t1"}
    close = ex.created[-1]
    assert close["type"] == "market" and close["side"] == "sell" and close["params"].get("reduceOnly")
