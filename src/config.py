"""Central configuration.

Holds the market enum, screener thresholds, worked-example/risk defaults, and
loads optional API keys + env overrides. Public market data (the screener,
analysis, backtest) needs no API key; keys are only consulted by the much-later
order layer.

Every threshold here is a *gate the user can see and tune* — the screener prints
them so the shortlist is never a black box.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, is_dataclass, replace
from enum import Enum
from pathlib import Path

from dotenv import load_dotenv

# Load .env if present. Safe no-op when the file is absent.
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"


class Market(str, Enum):
    """Which Binance market we operate on."""

    SPOT = "spot"      # long-only
    USDM = "usdm"      # USD-margined perpetual futures (long/short + leverage)

    @property
    def label(self) -> str:
        return "SPOT" if self is Market.SPOT else "USD-M"


@dataclass(frozen=True)
class ScreenerConfig:
    """Thresholds for the Layer-0 screener, applied in the brief's priority order.

    1. LIQUIDITY  -> top_n / min_quote_volume / max_spread_bps / min_depth_usd
    2. HISTORY    -> min_candles (need enough to backtest later)
    3. VOLATILITY -> atr_pct_min..atr_pct_max band (per screen_tf candle)
    4. REGIME     -> adx_trending / adx_choppy thresholds
    5. CORRELATION-> corr_window / corr_threshold (flag fake diversification)
    """

    quote: str = "USDT"

    # 1. Liquidity gatekeeper
    top_n: int = 100                        # top-N by 24h volume to deep-check (the universe width)
    min_quote_volume: float = 5_000_000.0   # 24h quote volume floor (USDT)
    max_spread_bps: float = 8.0             # best bid/ask spread, basis points
    min_depth_usd: float = 50_000.0         # resting book within +/- depth_band_pct of mid
    depth_band_pct: float = 0.5

    # 2. History
    min_candles: int = 200                  # closed candles required on screen_tf
    screen_tf: str = "1d"                   # timeframe used for regime/vol/history
    candle_limit: int = 250                 # candles to request per symbol

    # 3. Volatility band (ATR as % of price, on screen_tf)
    atr_length: int = 14
    atr_pct_min: float = 1.5
    atr_pct_max: float = 15.0

    # 4. Regime clarity (ADX)
    adx_length: int = 14
    adx_trending: float = 25.0              # ADX >= this -> trending
    adx_choppy: float = 20.0                # ADX <  this -> choppy/avoid

    # 5. Correlation clustering (also used for BTC beta/correlation window)
    corr_window: int = 60                   # candles of returns to correlate
    corr_threshold: float = 0.85            # >= this -> same cluster

    # 3b. Volatility regime (is volatility coiling or extended?)
    atr_pctl_window: int = 100              # trailing window for ATR% percentile
    squeeze_pctl: float = 25.0              # <= -> "squeeze" (coiling)
    expand_pctl: float = 75.0               # >= -> "expanding" (possibly late)

    # 3c. Extension / freshness (how far into the move is price, in ATRs vs its MA?)
    ext_ma_len: int = 20                    # reference MA for extension
    fresh_atr: float = 1.5                  # |dir. extension| <  this -> "fresh"
    extended_atr: float = 2.5               # |dir. extension| >= this -> "extended"

    # 6. Market context & influence groups
    rs_window: int = 30                     # candles for relative-strength vs a leader
    anchor_bases: tuple[str, ...] = ("BTC", "ETH")   # always-present macro leaders

    fetch_workers: int = 4                  # thread-pool workers for concurrent OHLCV (1 = sequential)

    # Composite ranking weights (trend clarity + rel-strength + vol + liquidity);
    # NOT ADX alone. Penalties push choppy / tide-fighting candidates down.
    rank_w_trend: float = 1.0
    rank_w_rs: float = 1.0
    rank_w_vol: float = 0.5
    rank_w_fresh: float = 0.5               # reward fresh entries, penalise extended/late
    rank_w_liq: float = 0.5
    choppy_penalty: float = 2.0             # subtracted when regime is choppy
    context_penalty: float = 1.0            # spot longs into a risk-off BTC (x beta)

    # Symbols we never want as candidates (stable/stable pairs, fiat).
    excluded_bases: tuple[str, ...] = (
        "USDC", "FDUSD", "TUSD", "DAI", "USDP", "BUSD", "USD1",
        "EUR", "GBP", "TRY", "BRL", "AEUR", "EURI",
    )


@dataclass(frozen=True)
class StructureConfig:
    """Parameters for expert-grade structure detection (Layer 2)."""

    atr_length: int = 14
    # ZigZag swings
    reversal_atr: float = 2.0          # confirm a swing on a >= this * ATR reversal
    major_leg_atr: float = 4.0         # a leg >= this * ATR is a "major" swing
    # Level clustering / scoring / pruning
    level_atr_fraction: float = 0.5    # cluster swings within this * ATR into one level
    min_separation_atr: float = 1.0    # pruned levels must be >= this * ATR apart
    max_levels: int = 8                # report at most this many S/R levels
    recency_window: int = 120          # recency component of level strength
    eq_touches: int = 2                # >= this many same-kind touches = a liquidity pool
    # Context levels
    range_window: int = 60             # candles for prior-range (Donchian) edges
    round_span_pct: float = 15.0       # round numbers within +/- this % of price
    # Volume profile
    vp_window: int = 120               # lookback for the volume profile
    vp_bins: int = 40
    value_area_pct: float = 70.0


@dataclass(frozen=True)
class SetupConfig:
    """FIXED, robust setup parameters — no per-coin curve-fitting (Layer 3).

    Any optimisation happens only in the walk-forward engine (Layer 5) with
    out-of-sample validation; these defaults stay fixed across coins.
    """

    atr_length: int = 14
    ema_len: int = 20              # pullback reference MA
    zone_atr: float = 0.6          # "near a level" tolerance, in ATR
    stop_buffer_atr: float = 0.5   # stop placed this far beyond the level/swing
    breakout_lookback: int = 20    # bars to confirm a genuine prior break
    min_rr: float = 1.2            # below this -> cannot grade above C
    rr_target_default: float = 1.5 # fallback target when no structure level
    expiry_bars: int = 8           # pending-setup time-stop

    # breakout_momentum (enter ON the break, no retest — rides moves that run)
    momentum_adx_floor: float = 22.0    # only in a trending/expanding regime (ADX >= this)
    momentum_break_atr: float = 0.5     # close must clear the prior range by this * ATR

    # momentum_flag (impulse -> tight coil -> continuation break)
    flag_impulse_atr: float = 2.5       # the leg into the last swing must be >= this * ATR
    flag_max_bars: int = 6              # consolidation no longer than this many bars
    flag_max_retrace: float = 0.5       # consolidation retraces <= this fraction of the impulse
    flag_max_range_atr: float = 2.0     # consolidation total range <= this * ATR (tight)

    # liquidity_sweep_reversal (stop-run beyond an equal-highs/lows pool, then reclaim)
    sweep_wick_frac: float = 0.5        # rejection wick >= this fraction of the bar's range

    # squeeze_breakout (volatility compression -> expansion; the ignition of a move)
    squeeze_pctl: float = 30.0          # prior ATR%ile <= this = a coil worth a breakout
    # divergence_reversal (momentum divergence at a structural extreme)
    rsi_length: int = 14

    # P14 regime ARCHETYPE labelling (descriptive name for the trend×vol grid cell; the
    # statistical edge still comes from the gated conditional expectancy, not these labels).
    arch_thrust_bars: int = 5           # lookback (bars) for the recent directional thrust
    arch_thrust_atr: float = 1.5        # |thrust| >= this * ATR + vol expanding -> panic / euphoria
    arch_leg_bars: int = 30             # lookback for the leg INTO a range (accum vs distrib)
    arch_leg_atr: float = 2.0           # |leg| >= this * ATR decides accumulation / distribution


@dataclass(frozen=True)
class AnalysisConfig:
    """Parameters for the six-lens multi-timeframe synthesis (Layer 2)."""

    tf_bias: str = "1d"                # higher TF sets the bias
    tf_trigger: str = "4h"             # lower TF provides the read/trigger
    candle_limit: int = 300
    rsi_length: int = 14
    rsi_bull: float = 55.0
    rsi_bear: float = 45.0
    ma_fast: int = 20
    ma_slow: int = 50
    rs_window: int = 30
    bias_threshold: float = 0.15       # |net score| >= this -> a directional bias
    location_atr_frac: float = 0.6     # within this * ATR of a level -> "at" it

    # Directional lens weights (volatility is non-directional -> weight 0)
    w_trend: float = 0.30
    w_momentum: float = 0.20
    w_volume: float = 0.15
    w_levels: float = 0.15
    w_context: float = 0.20


@dataclass(frozen=True)
class MarketsConfig:
    """Market-mechanics parameters (fees, leverage, liquidation, funding/OI)."""

    # Fees (fraction of notional, per side)
    spot_taker: float = 0.001
    spot_maker: float = 0.001
    usdm_taker: float = 0.0005
    usdm_maker: float = 0.0002
    slippage_bps: float = 3.0          # modeled slippage per side (basis points)

    # Leverage / liquidation (USD-M)
    default_leverage: float = 3.0
    max_leverage_cap: float = 20.0     # our cap, regardless of what Binance allows
    liq_buffer_mult: float = 1.5       # liquidation distance must be >= stop distance * this
    fallback_mmr: float = 0.005        # maintenance-margin rate if tiers unavailable

    # Funding / open interest (USD-M)
    funding_periods_per_day: int = 3   # Binance funds every 8h
    expected_hold_days: float = 2.0    # horizon for the funding-cost estimate
    oi_history_tf: str = "4h"          # Binance OI-history period (NOT 8h — unsupported)
    oi_history_len: int = 42           # ~7 days of 4h bars
    oi_trend_pct: float = 5.0          # |change| >= this % => rising/falling


@dataclass(frozen=True)
class RiskDefaults:
    """Defaults for the dollar worked-example formatter.

    These are *teaching* defaults; the real risk engine (Layer 2) refines them
    with fees, funding, tick/lot rounding and drawdown scaling.
    """

    account_equity: float = 1_000.0
    risk_pct: float = 1.0          # percent of equity risked per trade
    atr_stop_mult: float = 1.5     # stop distance = mult * ATR (illustrative)
    rr_target: float = 1.5         # reward:risk used for the illustrative TP


@dataclass(frozen=True)
class RiskConfig:
    """Trade-management, cost, drawdown-scaling and Kelly parameters (Layer 4)."""

    tp1_fraction: float = 0.5          # scale out this fraction at TP1
    move_to_breakeven_on_tp1: bool = True
    trail_to_structure: bool = False   # optional trail on the runner
    min_net_rr: float = 1.2            # net R:R below this -> NO TRADE
    # --- EFFECTIVE-RISK PIPELINE (composed in risk.effective_risk_pct; each stage opt-in) ---
    # base risk% -> ×drawdown-scale -> ×edge-scale -> Kelly cap -> clamp to max_risk_pct.
    # drawdown-scaled sizing (reduce risk as the account draws down)
    dd_scale_enabled: bool = False     # OFF by default: default behaviour stays flat base %
    dd_scale_start: float = 5.0        # begin reducing risk past this % drawdown
    dd_scale_full: float = 20.0        # drawdown % at which the multiplier hits the floor
    dd_scale_floor: float = 0.5        # risk multiplier never drops below this
    # edge-scaled sizing (size from the EDGE×confidence, not the appetite; only on proven setups)
    edge_scaled: bool = False          # OFF by default
    edge_ref_r: float = 0.15           # edge (R) that maps to ×1.0; more → larger, less → smaller
    edge_min_mult: float = 0.5         # clamp the edge multiplier to [min, max]
    edge_max_mult: float = 2.0
    # variance / VOLATILITY-TARGET sizing (P12): equalise per-trade dollar-volatility — a
    # setup whose trades swing wider (higher σ_R) is sized smaller for the SAME edge.
    vol_target_enabled: bool = False   # OFF by default: default behaviour stays flat base %
    vol_target_sigma_r: float = 1.5    # the σ_R that maps to ×1.0; setups wilder than this shrink
    vol_min_mult: float = 0.5          # clamp the vol multiplier to [min, max]
    vol_max_mult: float = 1.0          # default 1.0 = only ever REDUCES (conservative); raise to scale up
    # fractional-Kelly cap (only ever REDUCES risk; off by default)
    kelly_enabled: bool = False
    kelly_fraction: float = 0.25
    # hard ceiling the whole pipeline can never exceed (a backstop for risk-up testing)
    max_risk_pct: float = 5.0


@dataclass(frozen=True)
class BacktestConfig:
    """Edge-engine parameters (Layer 5)."""

    candle_limit: int = 1000       # history fetched for the backtest
    warmup: int = 60               # bars before the first signal is allowed
    struct_window: int = 150       # rolling window for point-in-time structure
    min_sample: int = 30           # trades required before a verdict
    folds: int = 3                 # walk-forward consistency folds
    mc_runs: int = 5000            # Monte-Carlo reshuffles
    mc_block: int = 0              # MC resampling: 0 = AUTO circular block (≈√n; honest streak/DD
    #                                tails, audit finding 4) · 1 = legacy iid permutation · ≥2 = block len
    bootstrap_runs: int = 2000     # P9: bootstrap resamples for the robust expectancy CI
    ruin_drawdown_r: float = 15.0  # "ruin" = equity draws down this many R
    # TIME-WEIGHTED LEARNING (P7): recent trades weigh more; older fade gradually (never dropped).
    time_decay_enabled: bool = False        # OFF by default → unweighted (today's behaviour exactly)
    time_decay_half_life_days: float = 90.0 # a trade this many days old counts half — gradual decay
    # CROSS-COIN CORRELATION (audit finding 2): pooled trades in the same time bucket ride the
    # same market move → MEASURED design effect widens SEs/CIs (√DEFF) and gates use n_eff=n/DEFF.
    # ON by default — this is statistical HONESTY, not risk appetite (disable only for comparison).
    deff_enabled: bool = True
    deff_bucket_hours: float = 24.0         # correlation cluster = same UTC day
    assumed_funding_per_8h: float = 0.0001  # flat funding assumption for usdm carry
    z: float = 1.0                 # expectancy CI half-width = z * standard error (display)
    sig_z: float = 1.65            # STRICTER z for the null-excess SIGNIFICANCE gate (~95% one-sided):
    #                                a setup must beat the random-entry null by this many SE, not just 1.
    #                                Controls the small-sample false-positive rate (z=1 → ~16%; 1.65 → ~5%).
    # NULL BASELINE — the honest bar is NOT zero. For every real trade, spawn this
    # many SHADOW trades (same direction + same risk geometry, RANDOM entry bar) and
    # manage them with the same state machine; the edge must beat that distribution.
    null_k: int = 10               # shadow trades spawned per real trade (0 = disable)
    null_horizon: int = 200        # bars a shadow is managed over (caps sim cost)
    null_cap: int = 5000           # max shadows kept per setup per coin (bounds cost)
    null_min: int = 50             # need this many shadows before the null gates a verdict


@dataclass(frozen=True)
class OrderConfig:
    """Order-staging parameters (Layer 9)."""

    drift_pct: float = 1.0         # market entry: abort if price drifted > this % from analysed entry
    confirm_word: str = "CONFIRM"  # the literal word required to stage


@dataclass(frozen=True)
class LiveConfig:
    """Real-money execution gates (Layer 9 `--live`; built DEAD LAST).

    Safe by default: live sending is REFUSED unless the env flag is explicitly
    armed. Even then it demands a second, distinct typed confirmation. The secret
    is never read here — only the opt-in flag — and never logged anywhere.
    """

    enable_env: str = "LIVE_TRADING_ENABLED"      # this env var must equal enable_value
    enable_value: str = "I_UNDERSTAND_THE_RISK"   # belt-and-suspenders opt-in
    confirm_phrase: str = "CONFIRM LIVE"           # the SECOND, distinct confirmation
    fill_timeout_s: float = 60.0                   # how long to wait for an entry fill
    fill_poll_s: float = 2.0                       # poll cadence while awaiting the fill


@dataclass(frozen=True)
class OptimizeConfig:
    """Step-3 config optimizer (the `optimize` engine).

    Robust, OUT-OF-SAMPLE, propose-a-diff. The process is the proof: select on
    TRAIN, validate on untouched TEST, confirm on a once-touched LOCK-BOX, correct
    for multiple-testing, demand a stable plateau + cost survival. NEVER auto-applies.
    """

    train_frac: float = 0.60       # chronological split: optimize ONLY on this
    test_frac: float = 0.25        # validate on this (OOS); lock-box = the rest (~0.15)
    min_trades: int = 20           # per pooled train/test sample, else the cell is ineligible
    penalty_lambda: float = 0.5    # robust score = expectancy − λ·(cross-coin spread)
    alpha: float = 0.05            # significance; Bonferroni-divided by the number of grid cells
    bootstrap: int = 2000          # resamples for the OOS significance test
    cost_mult: float = 2.0         # pessimistic cost stress (× fees + slippage)
    coins: int = 6                 # liquid coins to POOL across (cross-coin generalization)
    # default coarse grid for the universally-impactful stop buffer (in ATRs)
    stop_buffer_grid: tuple = (0.25, 0.5, 0.75, 1.0, 1.5)


@dataclass(frozen=True)
class AlertsConfig:
    """Alerting parameters (Layer 8)."""

    tf: str = "4h"                 # timeframe for setup-trigger alerts
    candle_limit: int = 300


@dataclass(frozen=True)
class WatchConfig:
    """Always-on radar (`watch`): poll for new PROVEN edge, digest hourly, stay diverse."""

    poll_seconds: float = 60.0         # LIGHT between-bar price/alert check cadence (clamped ≥60s)
    digest_seconds: float = 3600.0     # hourly "here's what I'm monitoring" status
    cooldown_minutes: float = 60.0     # don't re-ping the same coin within this window (no nagging)
    bar_settle_seconds: float = 20.0   # wait this long AFTER a trigger-TF candle closes before scanning
    notify: bool = True                # desktop notifications (macOS osascript)


@dataclass(frozen=True)
class JournalConfig:
    """Journal & review parameters (Layer 7)."""

    starting_equity: float = 1000.0
    min_sample: int = 20           # closed trades before review draws verdicts
    min_bucket_sample: int = 5     # per setup/regime/grade bucket before a verdict


@dataclass(frozen=True)
class GuardsConfig:
    """Portfolio/behaviour gate parameters (Layer 6)."""

    daily_loss_limit_r: float = 3.0        # realised ≤ −this R today → hard lockout
    daily_loss_limit_pct: float = 3.0      # (alt, if equity-% P&L is tracked)
    max_trades_per_day: int = 0            # ≤ N new trades opened since UTC midnight (0 = unlimited; over-trading guard)
    cooldown_minutes: float = 60.0         # block new entries this long after a loss
    cooldown_consecutive_losses: int = 3   # this many in a row → session lock
    heat_cap_pct: float = 6.0              # max total open risk as % equity (group-aware)
    group_corr_factor: float = 0.8         # same-group risk credited at this fraction beyond the largest
    max_positions: int = 5
    max_per_group: int = 2
    funding_window_minutes: float = 15.0   # warn within this of a funding time (00/08/16 UTC)
    oversize_tol: float = 0.10             # adherence: actual risk > planned*(1+tol) = oversized


@dataclass(frozen=True)
class EdgeScoreConfig:
    """Edge Score + Opportunity Board parameters (Layer: edge_score)."""

    tf: str = "4h"                 # timeframe for setups/backtest on the board
    floor: float = 0.05            # min trustworthy edge (R) to count as an edge
    min_regime_n: int = 20         # min trades in the CURRENT regime to trust it (raised 15→20: small-sample guard)
    sig_z: float = 1.65            # stricter z for the null-excess SIGNIFICANCE gate on the board (~95% one-sided)
    shrink_min_n: int = 5          # P5: min OWN trades before a coin personalizes vs the pool / contributes to τ²
    #                                (below this → pure pooled prior; stops a noisy 2-trade mean from being trusted)
    context_min_n: int = 25        # P1: min trades in a context cell (F=v) before it can be a proven refinement
    feature_top_k: int = 4         # P13: only the top-K most-IMPORTANT context features may seed P4 pairs
    shrink_k: int = 20             # confidence shrink prior: n/(n+k)
    conf_high: float = 0.60        # confidence tier cutoffs
    conf_mod: float = 0.35
    cache_ttl_hours: float = 24.0  # re-backtest if the cached profile is older
    scan_top: int | None = None    # deep-scan only the top-N survivors (None = ALL that passed the screen)
    pool_coins: int = 12           # coins POOLED for the universe edge verdict (beats per-coin "too few trades")
    # edge_score -> letter grade bands (descending)
    grade_bands: tuple = ((0.50, "A"), (0.30, "B"), (0.15, "C"), (0.05, "D"))


@dataclass(frozen=True)
class Settings:
    screener: ScreenerConfig = field(default_factory=ScreenerConfig)
    structure: StructureConfig = field(default_factory=StructureConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    setups: SetupConfig = field(default_factory=SetupConfig)
    markets: MarketsConfig = field(default_factory=MarketsConfig)
    risk: RiskDefaults = field(default_factory=RiskDefaults)
    risk_mgmt: RiskConfig = field(default_factory=RiskConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    edge: EdgeScoreConfig = field(default_factory=EdgeScoreConfig)
    guards: GuardsConfig = field(default_factory=GuardsConfig)
    journal: JournalConfig = field(default_factory=JournalConfig)
    alerts: AlertsConfig = field(default_factory=AlertsConfig)
    watch: WatchConfig = field(default_factory=WatchConfig)
    order: OrderConfig = field(default_factory=OrderConfig)
    optimize: OptimizeConfig = field(default_factory=OptimizeConfig)
    live: LiveConfig = field(default_factory=LiveConfig)
    profile: str | None = None     # active RISK PROFILE name (None = raw config, today's defaults)
    api_key: str | None = None
    api_secret: str | None = None

    @property
    def has_keys(self) -> bool:
        return bool(self.api_key and self.api_secret)


# --------------------------------------------------------------------------- #
# FULL .env CONTROL — every config field is overridable via {SECTION}_{FIELD}.
# --------------------------------------------------------------------------- #
# Settings attribute -> env prefix. EVERY field of each section is exposed.
_SECTION_PREFIX: dict[str, str] = {
    "screener": "SCREEN", "structure": "STRUCT", "analysis": "ANALYSIS", "setups": "SETUP",
    "markets": "MARKETS", "risk": "RISK", "risk_mgmt": "MGMT", "backtest": "BACKTEST",
    "edge": "EDGE", "guards": "GUARDS", "journal": "JOURNAL", "alerts": "ALERTS",
    "watch": "WATCH", "order": "ORDER", "optimize": "OPTIMIZE", "live": "LIVE",
}
# SAFETY RAILS — never env-overridable (tuning thresholds must not disable a rail).
_PROTECTED: set[tuple[str, str]] = {
    ("live", "enable_env"), ("live", "enable_value"), ("live", "confirm_phrase"),
}
# Friendly aliases (canonical env name -> (section, field)). The alias WINS over the
# generic name so the long-standing knobs keep working.
_ALIASES: dict[str, tuple[str, str]] = {
    "ACCOUNT_EQUITY": ("risk", "account_equity"),
    "RISK_PCT": ("risk", "risk_pct"),
    "SCREEN_TF": ("screener", "screen_tf"),
}


def _coerce(current, raw: str):
    """Coerce an env string to the type of the field's current value."""
    raw = raw.strip()
    if isinstance(current, bool):
        return raw.lower() in ("1", "true", "yes", "on")
    if isinstance(current, int) and not isinstance(current, bool):
        try:
            return int(raw)
        except ValueError:
            try:
                return int(float(raw))
            except ValueError:
                return current
    if isinstance(current, float):
        try:
            return float(raw)
        except ValueError:
            return current
    if isinstance(current, tuple):
        if current and isinstance(current[0], tuple):
            return current                       # nested tuples (e.g. grade_bands) — not env-parseable
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        if current and isinstance(current[0], (int, float)) and not isinstance(current[0], bool):
            conv = float if isinstance(current[0], float) else int
            out = []
            for p in parts:
                try:
                    out.append(conv(p))
                except ValueError:
                    pass
            return tuple(out)
        return tuple(parts)
    if current is None:                          # Optional[...] — best-effort: int, float, else str
        if raw.lower() in ("none", "null", ""):
            return None
        for conv in (int, float):
            try:
                return conv(raw)
            except ValueError:
                pass
        return raw
    return raw                                   # str


def _override_from_env(instance, prefix: str, section: str):
    """Return a copy of a config dataclass with every field overridden from
    {prefix}_{FIELD} env vars (type-coerced). Safety-rail fields are skipped."""
    if instance is None or not is_dataclass(instance):
        return instance
    changes: dict = {}
    for f in fields(instance):
        if (section, f.name) in _PROTECTED:
            continue
        raw = os.getenv(f"{prefix}_{f.name.upper()}")
        if raw is None or raw.strip() == "":
            continue
        changes[f.name] = _coerce(getattr(instance, f.name), raw)
    return replace(instance, **changes) if changes else instance


def load_settings(base: Settings | None = None) -> Settings:
    """Build Settings from defaults overlaid with environment variables.

    EVERY field of EVERY section is overridable via {SECTION}_{FIELD} (see
    _SECTION_PREFIX), plus a few friendly aliases — so the whole tool is .env-tunable
    for wide testability. Safety rails (the live arm flag / confirm phrase) are never
    overridable. The active config is rendered back by the CLI so nothing is opaque.

    `base` sets the starting Settings (defaults when None). A speed tier is applied as a
    base PRESET this way so explicit env overrides layer ON TOP of it (env always wins over
    the tier — e.g. `--tier intraday` + BACKTEST_CANDLE_LIMIT=2000 keeps the tier's 1h but
    honours your 2000 candles).
    """
    s = base if base is not None else Settings()
    overrides: dict = {}
    for attr, prefix in _SECTION_PREFIX.items():
        new = _override_from_env(getattr(s, attr), prefix, attr)
        if new is not getattr(s, attr):
            overrides[attr] = new
    if overrides:
        s = replace(s, **overrides)

    # friendly aliases win over the generic names
    alias_changes: dict[str, dict] = {}
    for env_name, (section, fieldname) in _ALIASES.items():
        raw = os.getenv(env_name)
        if raw is None or raw.strip() == "":
            continue
        inst = getattr(s, section)
        alias_changes.setdefault(section, {})[fieldname] = _coerce(getattr(inst, fieldname), raw)
    for section, ch in alias_changes.items():
        s = replace(s, **{section: replace(getattr(s, section), **ch)})

    return replace(s, api_key=os.getenv("BINANCE_API_KEY") or None,
                   api_secret=os.getenv("BINANCE_API_SECRET") or None)


@dataclass(frozen=True)
class SpeedTier:
    """A coherent set of timeframes for a speed tier (swing vs intraday)."""

    name: str
    screen_tf: str
    tf_bias: str
    tf_trigger: str
    backtest_candle_limit: int


TIERS: dict[str, SpeedTier] = {
    "swing": SpeedTier("swing", screen_tf="1d", tf_bias="1d", tf_trigger="4h", backtest_candle_limit=1000),
    "intraday": SpeedTier("intraday", screen_tf="4h", tf_bias="4h", tf_trigger="1h", backtest_candle_limit=1500),
}


def apply_tier(settings: Settings, tier_name: str | None) -> Settings:
    """Return settings with TFs/limits set for the named speed tier (None = unchanged)."""
    tier = TIERS.get(tier_name) if tier_name else None
    if tier is None:
        return settings
    return replace(
        settings,
        screener=replace(settings.screener, screen_tf=tier.screen_tf),
        analysis=replace(settings.analysis, tf_bias=tier.tf_bias, tf_trigger=tier.tf_trigger),
        edge=replace(settings.edge, tf=tier.tf_trigger),
        backtest=replace(settings.backtest, candle_limit=tier.backtest_candle_limit),
    )


# ---------------------------------------------------------------------------- #
# RISK PROFILES (Phase 3 — the operator's command authority). A profile moves
# CAPITAL/EXPOSURE (Category A) and SELECTIVITY (Category B) ONLY — how hard you
# press on proven edges, never what counts as proven. Statistical honesty
# (null/sig_z/Bonferroni/effective-n/sample floors) and live safety rails are
# HONESTY-LOCKED below and enforced at import; a profile that touches them
# cannot exist. Precedence: defaults → tier → PROFILE → explicit env (env wins).
# No profile set → raw defaults: exactly today's behaviour.
# ---------------------------------------------------------------------------- #
_HONESTY_LOCKED: set[tuple[str, str]] = {
    ("edge", "sig_z"), ("edge", "min_regime_n"), ("edge", "shrink_min_n"), ("edge", "context_min_n"),
    ("backtest", "sig_z"), ("backtest", "null_k"), ("backtest", "null_min"), ("backtest", "min_sample"),
    ("backtest", "deff_enabled"), ("backtest", "deff_bucket_hours"),
} | _PROTECTED

RISK_PROFILES: dict[str, dict[str, dict]] = {
    # L0 — the safest posture: base appetite, every self-throttle ON.
    "L0": {"risk": {"risk_pct": 1.0},
           "risk_mgmt": {"dd_scale_enabled": True, "vol_target_enabled": True, "kelly_enabled": True,
                          "kelly_fraction": 0.25, "edge_scaled": True, "max_risk_pct": 5.0, "min_net_rr": 1.2},
           "guards": {"heat_cap_pct": 6.0, "max_positions": 5, "max_per_group": 2, "daily_loss_limit_r": 3.0},
           "edge": {"floor": 0.05}},
    "L1": {"risk": {"risk_pct": 1.5},
           "risk_mgmt": {"dd_scale_enabled": True, "vol_target_enabled": True, "kelly_enabled": True,
                          "kelly_fraction": 0.25, "edge_scaled": True, "max_risk_pct": 6.0, "min_net_rr": 1.2},
           "guards": {"heat_cap_pct": 9.0, "max_positions": 6, "max_per_group": 2, "daily_loss_limit_r": 4.0},
           "edge": {"floor": 0.04}},
    "L2": {"risk": {"risk_pct": 3.0},
           "risk_mgmt": {"dd_scale_enabled": True, "vol_target_enabled": True, "kelly_enabled": True,
                          "kelly_fraction": 0.334, "edge_scaled": True, "edge_max_mult": 2.5,
                          "max_risk_pct": 10.0, "min_net_rr": 1.1},
           "guards": {"heat_cap_pct": 15.0, "max_positions": 8, "max_per_group": 3, "daily_loss_limit_r": 6.0},
           "edge": {"floor": 0.03}},
    # L3 — presses hardest AND stops self-throttling (dd-scale/vol-target OFF): eyes open.
    "L3": {"risk": {"risk_pct": 5.0},
           "risk_mgmt": {"dd_scale_enabled": False, "vol_target_enabled": False, "kelly_enabled": True,
                          "kelly_fraction": 0.5, "edge_scaled": True, "edge_max_mult": 3.0,
                          "max_risk_pct": 15.0, "min_net_rr": 1.0},
           "guards": {"heat_cap_pct": 25.0, "max_positions": 12, "max_per_group": 4, "daily_loss_limit_r": 10.0},
           "edge": {"floor": 0.02}},
}
PROFILE_LABELS = {"L0": "CONSERVATIVE", "L1": "CAUTIOUS", "L2": "MEDIUM", "L3": "HIGH"}


def _validate_profile_registry() -> None:
    """Import-time guarantee: no profile can touch an honesty-locked or protected field."""
    for name, sections in RISK_PROFILES.items():
        for sec, fields in sections.items():
            for f in fields:
                if (sec, f) in _HONESTY_LOCKED:
                    raise AssertionError(f"risk profile {name} illegally touches locked {sec}.{f}")


_validate_profile_registry()


def apply_profile(settings: Settings, name: str | None) -> Settings:
    """Overlay a named RISK PROFILE (None → unchanged). Applied UNDER env overrides, so an
    explicit env var always beats the profile. Refuses unknown names and incoherent results."""
    if not name:
        return settings
    key = name.strip().upper()
    if key not in RISK_PROFILES:
        raise ValueError(f"unknown risk profile {name!r} — choose from {sorted(RISK_PROFILES)}")
    s = settings
    for sec, fields in RISK_PROFILES[key].items():
        s = replace(s, **{sec: replace(getattr(s, sec), **fields)})
    s = replace(s, profile=key)
    if s.risk.risk_pct > s.risk_mgmt.max_risk_pct:
        raise ValueError(f"profile {key} incoherent: base risk {s.risk.risk_pct}% exceeds ceiling")
    return s


# Measured on the planted-edge lab (see AUDITREPORT): approximate small-sample fluke
# rate at the board's real sample sizes for each significance bar.
_FLUKE_POINTS = ((1.65, 5.0), (1.28, 12.0), (1.00, 16.0))


def strictness_fluke_note(sig_z: float) -> str | None:
    """None when sig_z is at the default bar; else an honest cost estimate for the echo."""
    if abs(sig_z - 1.65) < 1e-9:
        return None
    pts = sorted(_FLUKE_POINTS)
    if sig_z <= pts[0][0]:
        est = pts[0][1]
    elif sig_z >= pts[-1][0]:
        est = pts[-1][1]
    else:
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            if x0 <= sig_z <= x1:
                est = y1 + (y0 - y1) * (sig_z - x1) / (x0 - x1)
                break
    return f"sig_z {sig_z:g} → expected fluke rate ~{est:.0f}% (default 1.65 ≈ 5%)"


def with_overrides(settings: Settings, **screener_overrides) -> Settings:
    """Return a copy of settings with screener fields overridden (CLI flags)."""
    clean = {k: v for k, v in screener_overrides.items() if v is not None}
    if not clean:
        return settings
    return replace(settings, screener=replace(settings.screener, **clean))
