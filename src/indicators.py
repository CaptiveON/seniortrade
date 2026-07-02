"""Technical indicators used by the screener.

Implemented natively (Wilder's smoothing) so Layer 0 has no hard dependency on
pandas-ta's shifting packaging. These are descriptive measures of *realised*
market structure — they do not forecast price.

All functions take an OHLCV DataFrame with float columns: open, high, low,
close, volume. They return pandas Series/DataFrames aligned to the input index.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def wilder_rma(series: pd.Series, length: int) -> pd.Series:
    """Wilder's running moving average (a.k.a. RMA / SMMA).

    Equivalent to an EWM with alpha = 1/length; ``min_periods`` keeps the head
    NaN until we have a full window, so early values aren't misleadingly stable.
    """
    if length <= 0:
        raise ValueError("length must be positive")
    return series.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    """Wilder's True Range: max of the three classic ranges."""
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)
    ranges = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """Average True Range (absolute price units)."""
    return wilder_rma(true_range(df), length)


def atr_percent(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """ATR expressed as a percent of close — comparable across symbols/prices."""
    return atr(df, length) / df["close"] * 100.0


def adx(df: pd.DataFrame, length: int = 14) -> pd.DataFrame:
    """Average Directional Index with +DI / -DI (Wilder).

    Returns a DataFrame with columns: ``plus_di``, ``minus_di``, ``adx``.
    High ADX => a strong (clean) trend; low ADX => choppy/rangey. Direction is
    read from +DI vs -DI.
    """
    high = df["high"]
    low = df["low"]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0.0), up_move, 0.0),
        index=df.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0.0), down_move, 0.0),
        index=df.index,
    )

    atr_ = wilder_rma(true_range(df), length)
    # Guard against division by zero on perfectly flat segments.
    safe_atr = atr_.replace(0.0, np.nan)

    plus_di = 100.0 * wilder_rma(plus_dm, length) / safe_atr
    minus_di = 100.0 * wilder_rma(minus_dm, length) / safe_atr

    di_sum = (plus_di + minus_di).replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / di_sum
    adx_ = wilder_rma(dx, length)

    return pd.DataFrame(
        {"plus_di": plus_di, "minus_di": minus_di, "adx": adx_},
        index=df.index,
    )


# --- Regime classification (shared by screener + market_context) ---
TRENDING = "trending"
RANGING = "ranging"
CHOPPY = "choppy"


def regime_label(adx_value: float, trending_th: float, choppy_th: float) -> str:
    """Map an ADX reading to a regime label.

    >= trending_th -> trending; < choppy_th -> choppy; otherwise ranging.
    NaN (not enough data) is treated as choppy/avoid — the safe default.
    """
    if pd.isna(adx_value):
        return CHOPPY
    if adx_value >= trending_th:
        return TRENDING
    if adx_value < choppy_th:
        return CHOPPY
    return RANGING


# --- Volatility regime (is volatility coiling or expanding?) ---
SQUEEZE = "squeeze"
NORMAL = "normal"
EXPANDING = "expanding"


def atr_percentile(df: pd.DataFrame, length: int = 14, window: int = 100) -> float:
    """Percentile rank (0..100) of the latest ATR% within its trailing window.

    A seasoned trader cares less about the raw ATR than about whether volatility
    is *low for this coin* (coiling -> potential expansion) or *high* (a move may
    already be extended/late). We rank the latest ATR% against its own recent
    history rather than an absolute threshold.
    """
    ap = atr_percent(df, length).dropna()
    if ap.empty:
        return float("nan")
    recent = ap.tail(window)
    latest = recent.iloc[-1]
    return float((recent <= latest).mean() * 100.0)


def volatility_regime(
    df: pd.DataFrame,
    length: int = 14,
    window: int = 100,
    squeeze_pctl: float = 25.0,
    expand_pctl: float = 75.0,
) -> tuple[str, float]:
    """Classify volatility regime from the ATR% percentile.

    Returns (label, percentile). squeeze = coiling (low percentile),
    expanding = elevated (high percentile), normal = in between.
    """
    pctl = atr_percentile(df, length, window)
    if pd.isna(pctl):
        return NORMAL, pctl
    if pctl <= squeeze_pctl:
        return SQUEEZE, pctl
    if pctl >= expand_pctl:
        return EXPANDING, pctl
    return NORMAL, pctl


# --- Extension / freshness (how far is price into the move?) ---
FRESH = "fresh"
STRETCHED = "stretched"
EXTENDED = "extended"


def sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length).mean()


def extension_atr(df: pd.DataFrame, ma_len: int = 20, atr_len: int = 14) -> pd.Series:
    """How far price sits from its reference MA, measured in ATRs.

    Positive = above the MA (stretched up), negative = below (stretched down).
    Expressed in ATR units so it's comparable across coins/prices: +3 means price
    is three average-true-ranges above its mean — a leader that has run a long way
    and is LATE for a fresh entry.
    """
    ma = df["close"].rolling(ma_len).mean()
    return (df["close"] - ma) / atr(df, atr_len)


# --- Momentum ---
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """Wilder's RSI (0..100). >50 = up-momentum, <50 = down-momentum."""
    delta = df["close"].diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = wilder_rma(gain, length)
    avg_loss = wilder_rma(loss, length)
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    # all-gains windows -> RSI 100; all-losses -> 0
    out = out.where(avg_loss != 0, 100.0)
    out = out.where(avg_gain != 0, out.where(avg_loss == 0, 0.0))
    return out


def macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD line, signal line, and histogram (momentum confirming vs waning)."""
    macd_line = ema(df["close"], fast) - ema(df["close"], slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return pd.DataFrame({"macd": macd_line, "signal": signal_line, "hist": hist}, index=df.index)


# --- Volume / participation ---
def obv(df: pd.DataFrame) -> pd.Series:
    """On-Balance Volume: cumulative volume signed by the close direction."""
    direction = np.sign(df["close"].diff().fillna(0.0))
    return (direction * df["volume"]).cumsum()


def cvd_proxy(df: pd.DataFrame) -> pd.Series:
    """Cumulative volume-delta PROXY from OHLCV (the Accumulation/Distribution line).

    True CVD needs taker buy/sell volume; this approximates buying vs selling
    pressure from where the close sits in each candle's range. Rising = net
    accumulation, falling = net distribution.
    """
    rng = (df["high"] - df["low"]).replace(0.0, np.nan)
    clv = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / rng
    clv = clv.fillna(0.0)
    return (clv * df["volume"]).cumsum()
