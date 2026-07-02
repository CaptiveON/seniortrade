"""Step 18 — LIVE EXECUTION (the only code that can move real money). Built DEAD LAST.

Safe by default. Nothing here sends an order unless:
  1. an explicit env flag arms real-money mode (or it's running on TESTNET), AND
  2. the caller has already passed the full `stage` gate, AND
  3. a SECOND, distinct ``CONFIRM LIVE`` is typed at the CLI.

CARDINAL RULE: never a naked position. The entry is armed with its protective
stop the instant it fills; if the stop cannot be placed, the position is
IMMEDIATELY closed. The EXCHANGE is the source of truth — `reconcile_live`
re-reads positions/orders so a stop-out that happened while away is caught.

These primitives are unit-tested against a fake exchange; no real or testnet
order is ever placed from the test suite. The secret is never read or logged here.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

from .data_fetch import load_markets, make_exchange
from .markets import LONG, SHORT, make_market  # noqa: F401  (re-exported for callers)
from .order import OrderLeg


class LiveError(Exception):
    """Any failure on the live-execution path. Caller aborts loudly — never silently."""


# --------------------------------------------------------------------------- #
# Arming + preflight
# --------------------------------------------------------------------------- #
def live_armed(live_cfg) -> bool:
    """True only when the explicit opt-in env flag is set to the exact value."""
    return os.getenv(live_cfg.enable_env, "") == live_cfg.enable_value


@dataclass
class PreflightResult:
    ok: bool
    testnet: bool
    checks: list = field(default_factory=list)   # (name, passed: bool|None, detail)
    blocks: list = field(default_factory=list)


def verify_permissions(ex) -> tuple[bool | None, str]:
    """Best-effort: confirm the key cannot withdraw. None = couldn't verify (warn)."""
    try:
        r = ex.sapi_get_account_apirestrictions()
    except Exception:
        return None, "could not verify key restrictions — ENSURE withdrawals are OFF on this key"
    if r.get("enableWithdrawals") is True:
        return False, "this key has WITHDRAWALS ENABLED — refusing; create a trade-only key"
    return True, "withdrawals disabled (trade-only key)"


def preflight(settings, market, testnet: bool, ex=None) -> PreflightResult:
    """Gate every real-money send: keys, the arm flag, testnet-first, key permissions."""
    live = settings.live
    checks: list = []
    blocks: list = []

    has_keys = bool(settings.api_key and settings.api_secret)
    checks.append(("API key present in .env", has_keys, "trade-only · no-withdrawal · IP-locked"))
    if not has_keys:
        blocks.append("no API key configured in .env")

    armed = live_armed(live)
    checks.append((f"Real-money armed ({live.enable_env})", armed if not testnet else None,
                   "explicit opt-in" if not testnet else "not required on testnet"))
    if not testnet and not armed:
        blocks.append(f"real-money mode is NOT armed — export {live.enable_env}={live.enable_value}")

    checks.append(("Target", True, "TESTNET sandbox (safe)" if testnet else "LIVE exchange — REAL MONEY"))

    if ex is not None and has_keys:
        ok, detail = verify_permissions(ex)
        checks.append(("Withdrawals disabled", ok, detail))
        if ok is False:
            blocks.append("withdrawals are ENABLED on this key")

    return PreflightResult(ok=(has_keys and not blocks), testnet=testnet, checks=checks, blocks=blocks)


def make_live_exchange(settings, market, testnet: bool):
    """A keyed ccxt client; on testnet it runs against Binance's sandbox first."""
    if not (settings.api_key and settings.api_secret):
        raise LiveError("no API key — cannot trade live")
    ex = make_exchange(market, settings.api_key, settings.api_secret)
    if testnet:
        try:
            ex.set_sandbox_mode(True)
        except Exception as exc:
            raise LiveError(f"could not enable testnet sandbox: {exc}") from exc
    load_markets(ex)
    return ex


# --------------------------------------------------------------------------- #
# Order primitives
# --------------------------------------------------------------------------- #
def _leg(ticket, kind: str):
    return next((leg for leg in ticket.legs if leg.kind == kind), None)


def _ccxt_create(ex, mkt, symbol: str, leg: OrderLeg, amount: float, reduce_only: bool):
    """Map an OrderLeg to a ccxt create_order call (rounded to the instrument)."""
    amt = mkt.round_amount(symbol, amount)
    if amt <= 0:
        raise LiveError(f"{leg.kind}: rounded amount is zero — cannot place")
    params: dict = {}
    if reduce_only:
        params["reduceOnly"] = True
    ot = leg.order_type
    if ot == "market":
        return ex.create_order(symbol, "market", leg.side, amt, None, params)
    if ot == "limit":
        return ex.create_order(symbol, "limit", leg.side, amt, mkt.round_price(symbol, leg.price), params)
    if ot == "stop":  # a STOP entry (trigger into the trade) — not reduceOnly
        params["stopPrice"] = mkt.round_price(symbol, leg.price)
        return ex.create_order(symbol, "STOP_MARKET", leg.side, amt, None, params)
    if ot == "stop_market":
        params["stopPrice"] = mkt.round_price(symbol, leg.price)
        return ex.create_order(symbol, "STOP_MARKET", leg.side, amt, None, params)
    if ot == "take_profit_market":
        params["stopPrice"] = mkt.round_price(symbol, leg.price)
        return ex.create_order(symbol, "TAKE_PROFIT_MARKET", leg.side, amt, None, params)
    raise LiveError(f"{leg.kind}: unknown order type {ot!r}")


def _safe_cancel(ex, symbol: str, order_id) -> None:
    if not order_id:
        return
    try:
        ex.cancel_order(order_id, symbol)
    except Exception:
        pass  # already filled/cancelled — nothing to undo


def _set_leverage_isolated(ex, symbol: str, leverage: float, margin_mode: str | None) -> None:
    if margin_mode:
        try:
            ex.set_margin_mode(margin_mode, symbol)
        except Exception:
            pass  # "no need to change margin type" is benign
    ex.set_leverage(int(round(leverage)), symbol)


def _await_fill(ex, symbol: str, order_id, live_cfg) -> float:
    """Wait for the entry to fill; cancel any resting remainder. Returns FILLED qty."""
    deadline = time.time() + max(live_cfg.fill_timeout_s, 0.0)
    o = ex.fetch_order(order_id, symbol)
    while (o.get("status") == "open" and float(o.get("filled") or 0.0) <= 0.0
           and time.time() < deadline):
        time.sleep(max(live_cfg.fill_poll_s, 0.0))
        o = ex.fetch_order(order_id, symbol)
    filled = float(o.get("filled") or 0.0)
    amount = float(o.get("amount") or filled)
    if o.get("status") == "open" and filled < amount:
        _safe_cancel(ex, symbol, order_id)  # protect only what filled — no surprise later fill
    return filled


def _avg_price(ex, symbol: str, order_id) -> float | None:
    try:
        o = ex.fetch_order(order_id, symbol)
    except Exception:
        return None
    return o.get("average") or o.get("price")


def emergency_close(ex, mkt, symbol: str, exit_side: str, qty: float):
    """Market-close `qty` reduceOnly — the never-naked safety action."""
    return ex.create_order(symbol, "market", exit_side, mkt.round_amount(symbol, qty),
                           None, {"reduceOnly": True})


def _quote_of(symbol: str) -> str:
    return symbol.split(":")[-1] if ":" in symbol else symbol.split("/")[-1]


def _free_margin(ex, quote: str) -> float | None:
    """Best-effort FREE (available) balance in the settle/quote currency. None if unreadable."""
    fb = getattr(ex, "fetch_balance", None)
    if fb is None:
        return None
    try:
        bal = fb()
    except Exception:
        return None
    cur = bal.get(quote)
    if isinstance(cur, dict) and cur.get("free") is not None:
        return float(cur["free"])
    free_map = bal.get("free")
    if isinstance(free_map, dict) and free_map.get(quote) is not None:
        return float(free_map[quote])
    info = bal.get("info") or {}
    for key in ("availableBalance", "availableMargin", "maxWithdrawAmount"):
        try:
            return float(info.get(key))
        except (TypeError, ValueError):
            continue
    return None


def check_entry_margin(ex, symbol: str, entry_price: float, size: float, leverage: float,
                       buffer: float = 1.05) -> tuple[bool | None, str]:
    """Pre-send margin check: required initial margin (+buffer for fees) vs FREE balance.
    Returns (True, ok) / (False, why) / (None, unreadable → let the exchange enforce).
    This stops `-2019 Margin is insufficient` from ever reaching the order endpoint."""
    quote = _quote_of(symbol)
    if not entry_price or size <= 0:
        return None, "no entry price/size to estimate margin — exchange will enforce"
    notional = abs(entry_price * size)
    needed = notional / max(leverage, 1.0) * buffer
    free = _free_margin(ex, quote)
    if free is None:
        return None, f"could not read free {quote} — exchange will enforce margin"
    if free < needed:
        return False, (f"insufficient margin: need ~{needed:,.2f} {quote} (notional {notional:,.2f} "
                       f"÷ {leverage:g}x + fees) but only {free:,.2f} {quote} free — lower risk %, "
                       "raise leverage, drop a basket leg, or fund the account")
    return True, f"margin ok (~{needed:,.2f} {quote} of {free:,.2f} free)"


@dataclass
class LiveResult:
    filled_qty: float
    avg_price: float | None
    order_ids: dict = field(default_factory=dict)   # kind -> broker order id
    tp_errors: list = field(default_factory=list)


def place_entry_with_protection(ex, mkt, symbol: str, ticket, size: float, live_cfg) -> LiveResult:
    """The SEQUENCE: (usdm) leverage+isolated → entry → ON FILL the protective stop
    (sized to the FILLED qty) → TPs. If the stop can't be placed, EMERGENCY-close
    immediately. Never returns with an unprotected position."""
    entry = _leg(ticket, "entry")
    stop = _leg(ticket, "stop")
    if entry is None or stop is None:
        raise LiveError("ticket lacks an entry or a protective stop — refusing (never naked)")

    if ticket.leverage:  # USD-M: leverage + isolated margin FIRST
        _set_leverage_isolated(ex, symbol, ticket.leverage, ticket.margin_mode)

    # MARGIN PREFLIGHT — never send an order the account can't fund (clean skip, not a crash).
    ok, mdetail = check_entry_margin(ex, symbol, entry.price, mkt.round_amount(symbol, size),
                                     ticket.leverage or 1.0)
    if ok is False:
        raise LiveError(mdetail)

    # The ENTRY send — wrap every exchange rejection (insufficient margin, min-notional,
    # precision, …) as a clean LiveError. Entry rejected = nothing filled = no naked risk.
    try:
        eo = _ccxt_create(ex, mkt, symbol, entry, size, reduce_only=False)
    except LiveError:
        raise
    except Exception as exc:  # noqa: BLE001 — turn a raw ccxt error into a clean, non-fatal abort
        raise LiveError(f"entry order REJECTED by the exchange — no position opened ({exc})") from exc
    order_ids: dict = {"entry": eo.get("id")}

    filled = _await_fill(ex, symbol, eo.get("id"), live_cfg)
    if filled <= 0:
        _safe_cancel(ex, symbol, eo.get("id"))
        raise LiveError("entry did not fill within the timeout — cancelled it; no position, no naked risk")
    avg = _avg_price(ex, symbol, eo.get("id")) or entry.price

    # CARDINAL: protective stop sized to the FILLED quantity — or bail out.
    try:
        so = _ccxt_create(ex, mkt, symbol, stop, filled, reduce_only=True)
        order_ids["stop"] = so.get("id")
    except Exception as exc:
        emergency_close(ex, mkt, symbol, stop.side, filled)
        raise LiveError(f"PROTECTIVE STOP FAILED → emergency-closed the position (never naked): {exc}")

    # Take-profits (sized to the filled qty). Non-fatal: the stop is the protection.
    tp_errors: list = []
    for kind in ("tp1", "tp2"):
        leg = _leg(ticket, kind)
        if leg is None:
            continue
        try:
            o = _ccxt_create(ex, mkt, symbol, leg, filled * leg.qty_fraction, reduce_only=True)
            order_ids[kind] = o.get("id")
        except Exception as exc:  # noqa: BLE001 — keep the protected position; just report
            tp_errors.append(f"{kind}: {exc}")
    return LiveResult(filled_qty=filled, avg_price=avg, order_ids=order_ids, tp_errors=tp_errors)


def move_stop(ex, mkt, symbol: str, old_stop_id, exit_side: str, new_price: float, qty: float):
    """Cancel the resting stop and place a new protective stop (e.g. to breakeven)."""
    _safe_cancel(ex, symbol, old_stop_id)
    leg = OrderLeg("stop", "stop_market", exit_side, new_price, True, 1.0)
    return _ccxt_create(ex, mkt, symbol, leg, qty, reduce_only=True).get("id")


def close_position(ex, mkt, symbol: str, exit_side: str, qty: float, order_ids: dict | None = None):
    """Cancel the resting protective orders, then market-close the position."""
    for oid in (order_ids or {}).values():
        _safe_cancel(ex, symbol, oid)
    return emergency_close(ex, mkt, symbol, exit_side, qty)


# --------------------------------------------------------------------------- #
# Reconcile (the exchange is the source of truth) + kill switch
# --------------------------------------------------------------------------- #
@dataclass
class ReconcileFinding:
    symbol: str
    kind: str        # "closed" / "naked" / "ok"
    detail: str


def _open_positions(ex) -> dict:
    try:
        positions = ex.fetch_positions()
    except Exception as exc:
        raise LiveError(f"could not fetch positions to reconcile: {exc}") from exc
    out = {}
    for p in positions or []:
        contracts = abs(float(p.get("contracts") or 0.0))
        if contracts > 0:
            out[p.get("symbol")] = p
    return out


def _has_resting_stop(ex, symbol: str) -> bool:
    try:
        opens = ex.fetch_open_orders(symbol)
    except Exception:
        return False
    for o in opens or []:
        ro = o.get("reduceOnly") or (o.get("info") or {}).get("reduceOnly")
        if ro and "stop" in (o.get("type") or "").lower():
            return True
    return False


def reconcile_live(ex, records) -> list[ReconcileFinding]:
    """Compare journal OPEN live positions to the exchange. Catch stop-outs and naked positions."""
    positions = _open_positions(ex)
    findings: list[ReconcileFinding] = []
    for r in records:
        if getattr(r, "status", None) != "open" or getattr(r, "mode", None) not in ("live", "testnet"):
            continue
        if r.symbol not in positions:
            findings.append(ReconcileFinding(r.symbol, "closed",
                                             "no position on the exchange — closed/stopped-out while away"))
        elif not _has_resting_stop(ex, r.symbol):
            findings.append(ReconcileFinding(r.symbol, "naked",
                                             "position OPEN but NO protective stop resting — re-arm or --flatten NOW"))
        else:
            findings.append(ReconcileFinding(r.symbol, "ok", "position and protective stop present"))
    return findings


def flatten_all(ex, mkt) -> dict:
    """KILL SWITCH: cancel all resting orders and market-close every open position."""
    positions = _open_positions(ex)
    closed: list = []
    cancelled: list = []
    for symbol, p in positions.items():
        side = (p.get("side") or "").lower()
        exit_side = "sell" if side == LONG else "buy"
        qty = abs(float(p.get("contracts") or 0.0))
        try:
            for o in ex.fetch_open_orders(symbol) or []:
                _safe_cancel(ex, symbol, o.get("id"))
                cancelled.append(o.get("id"))
        except Exception:
            pass
        emergency_close(ex, mkt, symbol, exit_side, qty)
        closed.append(symbol)
    return {"closed": closed, "cancelled": cancelled}
