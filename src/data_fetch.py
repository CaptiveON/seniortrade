"""Real market data from Binance via ccxt.

Public endpoints only (tickers, OHLCV, order book) — no API key required. All
failures are wrapped in :class:`DataError` with a clear message so the CLI can
explain what went wrong instead of dumping a traceback.

Includes a stale-data guard: the screener must never reason about a feed that
has silently stopped updating.
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import ccxt
import pandas as pd

from .config import Market

OHLCV_COLS = ["timestamp", "open", "high", "low", "close", "volume"]
_NUMERIC = ["open", "high", "low", "close", "volume"]


class DataError(Exception):
    """Any data-layer failure (network, exchange, empty/stale feed)."""


def make_exchange(
    market: Market,
    api_key: str | None = None,
    api_secret: str | None = None,
) -> ccxt.Exchange:
    """Construct a ccxt client for the given market.

    Keys are optional and unused for public data; the screener never sends them.
    """
    params: dict = {
        "enableRateLimit": True,
        "timeout": 20_000,
        # adjustForTimeDifference syncs to server time; a wide recvWindow tolerates
        # residual local-clock drift (avoids -1021 "outside recvWindow" on signed calls).
        "options": {"adjustForTimeDifference": True, "recvWindow": 60_000},
    }
    if api_key and api_secret:
        params["apiKey"] = api_key
        params["secret"] = api_secret

    if market is Market.SPOT:
        ex = ccxt.binance(params)
        ex.options["defaultType"] = "spot"
    else:
        ex = ccxt.binanceusdm(params)
    return ex


def load_markets(ex: ccxt.Exchange) -> None:
    try:
        ex.load_markets()
    except ccxt.BaseError as exc:  # network, geo-block, etc.
        raise DataError(f"Could not load Binance markets: {exc}") from exc


def fetch_all_tickers(ex: ccxt.Exchange) -> dict:
    """One call returns 24h stats (incl. quoteVolume) for every symbol."""
    try:
        return ex.fetch_tickers()
    except ccxt.BaseError as exc:
        raise DataError(f"Could not fetch tickers: {exc}") from exc


# Binance caps klines at ~1000 per request (spot 1000 / USD-M ~1500; 1000 is safe for both);
# a single call cannot exceed this, so larger `limit`s are PAGINATED (see _fetch_ohlcv_paginated).
_MAX_CANDLES_PER_CALL = 1000

# ---------------------------------------------------------------------------- #
# CANDLE SANITATION (audit finding 6). One bad tape silently contaminates every
# pooled statistic downstream, so raw exchange candles are screened here at the
# single choke point. Policy: DROP only PROVABLY broken rows; FLAG (never "fix")
# anomalies — filling gaps or clipping spikes would fabricate market data and
# corrupt the edge measurement.
# ---------------------------------------------------------------------------- #
_SPIKE_RANGE_MULT = 12.0    # bar range > this × median range, fully reverted next bar → suspect print


def _tf_seconds(tf: str) -> float:
    try:
        val = float(tf[:-1])
    except (ValueError, IndexError):
        return 0.0
    return val * {"m": 60.0, "h": 3600.0, "d": 86400.0, "w": 604800.0}.get(tf[-1].lower(), 0.0)


def sanitize_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Screen raw exchange candles; returns the screened frame with a quality report in
    ``df.attrs['quality']``.

    DROPPED (provably broken — impossible market states): NaN/non-positive prices,
    high < low, high below the body, low above the body, NaN/negative volume.
    FLAGGED ONLY (real-world anomalies the engine must SEE, not invent around):
    gaps (missing bars vs the timeframe grid), zero-volume bars, suspect single-bar
    spikes (range > 12× median that fully reverts on the next bar — bad-print shaped)."""
    n0 = len(df)
    body_hi = df[["open", "close"]].max(axis=1)
    body_lo = df[["open", "close"]].min(axis=1)
    prices = df[["open", "high", "low", "close"]]
    broken = (prices.isna().any(axis=1) | (prices <= 0).any(axis=1)
              | (df["high"] < df["low"]) | (df["high"] < body_hi) | (df["low"] > body_lo)
              | df["volume"].isna() | (df["volume"] < 0))
    if bool(broken.any()):
        df = df.loc[~broken].copy()
    gaps = 0
    step = _tf_seconds(timeframe)
    if len(df) > 1 and step > 0:
        deltas = df.index.to_series().diff().dropna().dt.total_seconds()
        gaps = int(((deltas / step).round() - 1).clip(lower=0).sum())
    zero_vol = int((df["volume"] == 0).sum()) if len(df) else 0
    spikes = 0
    if len(df) > 2:
        rng = df["high"] - df["low"]
        med = float(rng.median())
        if med > 0:
            big = rng > _SPIKE_RANGE_MULT * med
            reverted = (df["close"].shift(-1) - df["close"].shift(1)).abs() <= 2 * med
            spikes = int((big & reverted).fillna(False).sum())
    df.attrs["quality"] = {"dropped": int(broken.sum()), "gaps": gaps, "zero_volume": zero_vol,
                           "suspect_spikes": spikes, "rows": len(df), "raw_rows": n0}
    return df


def _fetch_ohlcv_paginated(ex: ccxt.Exchange, symbol: str, timeframe: str, limit: int) -> list:
    """Collect up to `limit` of the MOST-RECENT candles by walking BACKWARD in cap-sized batches
    (via the ``endTime`` param), stitching until we have enough or history is exhausted. Dedupes by
    timestamp and returns ascending rows. Each iteration's oldest strictly decreases (loop-safe)."""
    cap = _MAX_CANDLES_PER_CALL
    collected: dict[int, list] = {}
    end = ex.milliseconds()
    while len(collected) < limit:
        batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=cap, params={"endTime": end})
        if not batch:
            break
        for row in batch:
            collected[int(row[0])] = row
        oldest = int(batch[0][0])
        if oldest >= end:                      # no backward progress → stop (defensive)
            break
        end = oldest - 1                        # next page ends strictly before this one's oldest
        if len(batch) < cap:                   # short page → history exhausted
            break
    rows = [collected[ts] for ts in sorted(collected)]
    return rows[-limit:]                        # keep the most-recent `limit`


def fetch_ohlcv(
    ex: ccxt.Exchange,
    symbol: str,
    timeframe: str,
    limit: int,
) -> pd.DataFrame:
    """Fetch OHLCV as a UTC-indexed float DataFrame.

    `limit` is honoured for ANY size: a single request when it fits under the exchange's per-call
    cap (~1000), otherwise transparent backward PAGINATION so a `limit` of 1500/2000/… actually
    delivers that many candles (subject to the symbol's available history).

    The most recent row is typically the *currently forming* candle; use
    :func:`drop_unclosed` before computing indicators if you need closed bars.
    """
    try:
        if limit <= _MAX_CANDLES_PER_CALL:
            raw = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        else:
            raw = _fetch_ohlcv_paginated(ex, symbol, timeframe, limit)
    except ccxt.BaseError as exc:
        raise DataError(f"OHLCV fetch failed for {symbol} {timeframe}: {exc}") from exc

    if not raw:
        raise DataError(f"No OHLCV returned for {symbol} {timeframe}")

    df = pd.DataFrame(raw, columns=OHLCV_COLS)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("datetime")
    df[_NUMERIC] = df[_NUMERIC].astype(float)
    df = sanitize_ohlcv(df, timeframe)           # audit finding 6: screen at the choke point
    if df.empty:
        raise DataError(f"All candles for {symbol} {timeframe} failed sanitation (broken tape)")
    return df


def fetch_ohlcv_concurrent(
    market: Market,
    symbols: list[str],
    timeframe: str,
    limit: int,
    api_key: str | None = None,
    api_secret: str | None = None,
    workers: int = 4,
) -> dict[str, pd.DataFrame]:
    """Fetch OHLCV for many symbols in parallel via a thread pool.

    Each worker thread gets its OWN ccxt exchange (requests.Session is not
    thread-safe), with its own rate limiter — so concurrency stays safe. Failures
    for a symbol are skipped (absent from the result), not raised.
    """
    if workers <= 1 or len(symbols) <= 1:
        ex = make_exchange(market, api_key, api_secret)
        load_markets(ex)
        out: dict[str, pd.DataFrame] = {}
        for s in symbols:
            try:
                out[s] = fetch_ohlcv(ex, s, timeframe, limit)
            except DataError:
                pass
        return out

    local = threading.local()

    def thread_exchange() -> ccxt.Exchange:
        ex = getattr(local, "ex", None)
        if ex is None:
            ex = make_exchange(market, api_key, api_secret)
            load_markets(ex)
            local.ex = ex
        return ex

    def work(sym: str):
        try:
            return sym, fetch_ohlcv(thread_exchange(), sym, timeframe, limit)
        except DataError:
            return sym, None

    out = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for sym, df in pool.map(work, symbols):
            if df is not None:
                out[sym] = df
    return out


def fetch_two_timeframes(
    ex: ccxt.Exchange,
    symbol: str,
    tf_high: str,
    tf_low: str,
    limit: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch two timeframes for the same symbol (higher TF first).

    Build-order step 1 requires real data on two timeframes; multi-timeframe
    alignment (Layer 1) builds on this.
    """
    return (
        fetch_ohlcv(ex, symbol, tf_high, limit),
        fetch_ohlcv(ex, symbol, tf_low, limit),
    )


def timeframe_seconds(ex: ccxt.Exchange, timeframe: str) -> int:
    """Length of one candle in seconds (ccxt parses '1d', '4h', '15m'...)."""
    return int(ex.parse_timeframe(timeframe))


def is_unclosed(ex: ccxt.Exchange, df: pd.DataFrame, timeframe: str) -> bool:
    """True if the last row is still forming (its period hasn't elapsed)."""
    if df.empty:
        return False
    last_open = df.index[-1].timestamp()
    return time.time() < (last_open + timeframe_seconds(ex, timeframe))


def drop_unclosed(ex: ccxt.Exchange, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Drop a trailing not-yet-closed candle so indicators use settled data."""
    if is_unclosed(ex, df, timeframe) and len(df) > 1:
        return df.iloc[:-1]
    return df


def is_stale(ex: ccxt.Exchange, df: pd.DataFrame, timeframe: str, max_age_factor: float = 3.0) -> bool:
    """True if even the latest candle is older than ``max_age_factor`` periods.

    A healthy feed's last candle is at most ~1 period behind 'now'. If it's
    several periods behind, the symbol/feed is stale and must not be traded on.
    """
    if df.empty:
        return True
    last_open = df.index[-1].timestamp()
    age = time.time() - last_open
    return age > max_age_factor * timeframe_seconds(ex, timeframe)


@dataclass
class BookLiquidity:
    """Top-of-book spread and resting depth near mid price."""

    spread_bps: float       # (ask - bid) / mid, in basis points
    depth_usd: float        # bid+ask quote-volume within +/- band_pct of mid
    mid: float


def fetch_book_liquidity(
    ex: ccxt.Exchange,
    symbol: str,
    band_pct: float,
    limit: int = 100,
) -> BookLiquidity:
    """Measure real spread and book thickness from the live order book."""
    try:
        ob = ex.fetch_order_book(symbol, limit=limit)
    except ccxt.BaseError as exc:
        raise DataError(f"Order book fetch failed for {symbol}: {exc}") from exc

    bids = ob.get("bids") or []
    asks = ob.get("asks") or []
    if not bids or not asks:
        raise DataError(f"Empty order book for {symbol}")

    best_bid = float(bids[0][0])
    best_ask = float(asks[0][0])
    if best_bid <= 0 or best_ask <= 0:
        raise DataError(f"Invalid top-of-book for {symbol}")

    mid = (best_bid + best_ask) / 2.0
    spread_bps = (best_ask - best_bid) / mid * 1e4

    lo = mid * (1.0 - band_pct / 100.0)
    hi = mid * (1.0 + band_pct / 100.0)
    depth = 0.0
    for price, qty in bids:
        price = float(price)
        if price >= lo:
            depth += price * float(qty)
    for price, qty in asks:
        price = float(price)
        if price <= hi:
            depth += price * float(qty)

    return BookLiquidity(spread_bps=spread_bps, depth_usd=depth, mid=mid)
