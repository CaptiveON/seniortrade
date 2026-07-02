"""Layer 8 — alerting.

Ping when a watched coin reaches an actionable zone, so you act deliberately
instead of watching charts. Two kinds:
  - LEVEL  — price crosses/reaches a price (a setup entry or structure level),
  - SETUP  — a named setup triggers on the latest closed bar.

Alerts are persisted (JSONL in /data), one-shot (fire once → inactive), and
checked on demand (`alert --check`) or on every `watch` pass. An alert NOTIFIES
only — it never places an order.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone

from . import data_fetch as dfetch
from . import setups as su
from . import structure as st
from .config import DATA_DIR, Market

LEVEL = "level"
SETUP = "setup"
ACTIVE = "active"
TRIGGERED = "triggered"

_ALERTS_PATH = DATA_DIR / "alerts.jsonl"


@dataclass
class Alert:
    id: str
    created_at: str
    market: str               # spot / usdm
    symbol: str
    kind: str                 # LEVEL / SETUP
    level: float | None = None
    direction: str = ""       # level: "up" (fire when price >= level) / "down" (<= level)
    tf: str = "4h"            # setup alerts
    note: str = ""
    status: str = ACTIVE


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def alerts_path():
    return _ALERTS_PATH


def _load(path=None) -> list[Alert]:
    p = path or _ALERTS_PATH
    if not p.exists():
        return []
    known = {f.name for f in fields(Alert)}
    out = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except ValueError:
            continue
        out.append(Alert(**{k: v for k, v in d.items() if k in known}))
    return out


def _save(alerts, path=None) -> None:
    p = path or _ALERTS_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(asdict(a)) + "\n" for a in alerts))


def list_alerts(active_only: bool = True, path=None) -> list[Alert]:
    alerts = _load(path)
    return [a for a in alerts if a.status == ACTIVE] if active_only else alerts


def remove_alert(alert_id: str, path=None) -> bool:
    alerts = _load(path)
    kept = [a for a in alerts if a.id != alert_id]
    if len(kept) == len(alerts):
        return False
    _save(kept, path)
    return True


def add_level_alert(market: Market, symbol: str, level: float, current_price: float,
                    note: str = "", path=None) -> Alert:
    direction = "up" if level >= current_price else "down"
    a = Alert(id=uuid.uuid4().hex[:8], created_at=_now(), market=market.value, symbol=symbol,
              kind=LEVEL, level=level, direction=direction, note=note)
    _save(_load(path) + [a], path)
    return a


def add_setup_alert(market: Market, symbol: str, tf: str, note: str = "", path=None) -> Alert:
    a = Alert(id=uuid.uuid4().hex[:8], created_at=_now(), market=market.value, symbol=symbol,
              kind=SETUP, tf=tf, note=note)
    _save(_load(path) + [a], path)
    return a


def level_triggered(alert: Alert, price: float) -> bool:
    if alert.level is None:
        return False
    return price >= alert.level if alert.direction == "up" else price <= alert.level


# --------------------------------------------------------------------------- #
# Checking
# --------------------------------------------------------------------------- #
def check_alerts(market: Market, settings, path=None) -> list[tuple[Alert, str]]:
    """Evaluate active alerts for `market`; mark + return the ones that fired."""
    alerts = _load(path)
    active = [a for a in alerts if a.status == ACTIVE and a.market == market.value]
    if not active:
        return []

    ex = dfetch.make_exchange(market, settings.api_key, settings.api_secret)
    dfetch.load_markets(ex)
    fired: list[tuple[Alert, str]] = []

    for a in active:
        msg = None
        try:
            if a.kind == LEVEL:
                last = float(ex.fetch_ticker(a.symbol)["last"])
                if level_triggered(a, last):
                    arrow = "↑" if a.direction == "up" else "↓"
                    msg = f"{a.symbol}: price {last:,.6g} {arrow} reached level {a.level:,.6g}"
            elif a.kind == SETUP:
                raw = dfetch.fetch_ohlcv(ex, a.symbol, a.tf, settings.alerts.candle_limit)
                bars = dfetch.drop_unclosed(ex, raw, a.tf)
                struct = st.analyze_structure(bars, settings.structure)
                if struct.state is not None:
                    ctx = su.SetupContext(htf_trend=struct.state.trend, tide="neutral")
                    sigs = su.detect_all(bars, struct, ctx, settings.setups, market)
                    if sigs:
                        names = ", ".join(f"{s.setup} {s.direction} (grade {s.grade})" for s in sigs[:3])
                        msg = f"{a.symbol}: setup fired on {a.tf} — {names}"
        except dfetch.DataError:
            continue
        if msg:
            a.status = TRIGGERED
            fired.append((a, msg))

    if fired:
        _save(alerts, path)   # persist the status changes
    return fired
