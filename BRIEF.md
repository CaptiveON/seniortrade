# PROJECT BRIEF: Crypto Trading SYSTEM
### Binance SPOT + USD-M — Screener → Context → Analysis → Setup → Risk → Edge → Staged Order
A command-line tool that reads the market and individual coins the way a seasoned,
highly competitive trader does: top-down, structure-aware, context-first, setup-driven,
risk-first, edge-proven, and disciplined — and that TEACHES the user at every step.
It analyses and stages; it does NOT predict price and never trades autonomously.

---

## 0. PHILOSOPHY
Success = a measured edge, executed with discipline, over many trades.
No single trade matters. The tool exists to:
  1. find clean, NAMED setups in liquid, tradeable markets,
  2. prove each setup has a real edge on real history,
  3. size every trade by risk first,
  4. enforce discipline and protect the account, and
  5. make every number mean something in DOLLARS, not just appear on screen.
It is a teaching tool, not a black box, and not financial advice. "NO TRADE" is
always a valid — and frequent — output.

---

## 1. THE METHOD — how a seasoned trader reads a market (the spine of the tool)
Always top-down. Each stage gates the next; failure at any stage ends in NO TRADE.

    MARKET    → What is BTC / the whole market doing? Risk-on or risk-off?
    UNIVERSE  → Which liquid coins are even tradeable right now? (screener)
    COIN      → What is THIS coin's regime, structure, momentum, participation?
    LEVEL     → WHERE is price relative to meaningful structure? (trade at a level)
    SETUP     → Is a specific, pre-defined, backtested pattern present? (else NO TRADE)
    RISK      → Size from the STOP; stop lives beyond structure, not at a guess.
    EDGE      → Has this setup proven +EV on history? If not, don't risk money.
    EXECUTE   → Stage only; dry-run by default; typed CONFIRM; re-fetch before send.
    REVIEW    → Journal everything; measure expectancy and adherence; adapt.

Core rules this encodes:
  - CONTEXT FIRST: never long an alt into a breaking-down BTC; respect the tide.
  - TRADE AT A LEVEL: entries happen at structure, never in the middle of nowhere.
  - STOPS BEYOND STRUCTURE: the stop sits past a swing/level; ATR is only a
    MINIMUM-distance sanity check so the stop is not parked inside the noise.
  - NAMED SETUP OR NO TRADE: only act on a defined, graded, backtested setup.
  - PROVE BEFORE RISKING: edge is demonstrated on history before any live order.
  - RISK IS THE CONSTANT: the dollars risked are chosen first; size is derived.

---

## 2. MARKET TYPES (first-class throughout)
  --spot : long-only. Spot pairs (e.g. BTC/USDT).
  --usdm : long AND short. USD-M linear perpetual swaps (e.g. BTC/USDT:USDT),
           with leverage, margin, funding, open interest, and liquidation.
Every layer behaves correctly for the chosen market; USD-M adds funding/OI reads,
liquidation-distance validation, and margin checks.

---

## 3. EXPLAIN-AS-YOU-GO (cross-cutting requirement, applies to ALL outputs)
Every stage prints a plain-language worked example in REAL DOLLARS, e.g.:
  - "With $1,000 account and 1% risk: you risk $10 on this trade."
  - "Entry 0.0860, stop 0.0885 → if stopped you lose ~$10 (your 1%)."
  - "Take-profit 0.0820 → if hit you make ~$15 (1.5R)."
  - "Ideal position size: X contracts / Y coins, costing $Z margin."
  - "Worst case (stop): -$10. Planned best case (TP): +$15. R:R 1:1.5."
The user must always understand what each number MEANS for their money.

---

## 4. PRESENTATION PRINCIPLE — compute like a quant, speak like a coach
The engine does seasoned-trader + quant work (structure, levels, patterns, setups,
backtested edge); the USER never draws a line or marks a level. Every output leads
with the DECISION (Edge Score, setup, grade, the dollar example), keeps the heavy
math hidden by default, and exposes it on demand (`--explain`). No black box, no
homework: the tool detects and ranks; the user reads and picks. (Later nicety: an
optional annotated chart image — auto-drawn levels / zones / entry-stop-target —
gives the TradingView-style visual WITHOUT manual drawing.)

---

## 5. SPEED TIERS & RUNTIME MODES
The analytical core is TIMEFRAME-AGNOSTIC and serves multiple speed tiers. Faster
≠ better: more trades = more cost drag → a HIGHER edge bar. A speed-tier config
selects timeframes + on-demand vs continuous + REST vs websocket.
  - SWING (4h–1d): run-on-demand, REST data, limit-at-level entries.  [built]
  - ACTIVE INTRADAY (5m–1h): the SAME engine on lower TFs + a RUNTIME extension —
    THREAD-POOL concurrent fetching (per-thread exchange; sync, not an async
    rewrite), a continuous `watch` mode (interval re-scan + flag new/dropped
    opportunities), tier-appropriate backtest candle limits + a light hot-loop
    optimisation. A data-source seam keeps WEBSOCKETS a drop-in later — but REST
    polling is on-time for 5m+ (a 5-min bar dwarfs ~30s poll latency; limit-at-level
    entries have no latency race), so websockets are DEFERRED to the scalping tier.
    [building now]
  - SCALPING / SUB-MINUTE: a SEPARATE engine (websocket L2/tape, microstructure
    signals, an order-management system, tick-level backtest) sharing only the
    risk / guards / journal / markets spine. DEFERRED — and gated on the edge engine
    first proving a sub-minute setup survives costs (most won't).

---

## LAYER 0 — SCREENER  (universe → ranked shortlist; runs first, gates everything)
screener.py
  Purpose: scan the whole exchange, reject untradeable coins, and output a RANKED
  shortlist of candidates worthy of deep analysis. The USER reads the table and
  picks; the rest of the tool then runs only on that coin.

  Filters (in priority order — each prints its threshold so it is never a black box):
    1. LIQUIDITY (gatekeeper): 24h quote volume above a floor; tight top-of-book
       spread; thick resting book near mid. Reject anything outside top-N by
       volume (default top 30). Illiquid = no fair fill, no real stop.
    2. HISTORY: require >= M closed candles of clean history (else we can't
       backtest it — and unbacktestable = untradeable here).
    3. VOLATILITY REGIME & EXTENSION: ATR% inside a sane band [min, max] AND its
       regime — ATR percentile vs the coin's OWN history and squeeze state
       (contraction = coiling, expansion = possibly late). ALSO measure how far
       price is into the move (FRESHNESS): distance from the move's origin / a
       reference MA, position in the recent range, run length. A leader that has
       already gone parabolic is LATE — flag it "extended", don't reward it.
    4. TREND/REGIME CLARITY: classify trending / ranging / choppy (e.g. ADX +
       structure). Surface the cleanest; flag "choppy / avoid."
    5. CORRELATION: cluster coins that move together so the user is not fooled by
       fake diversification (five correlated longs is one big bet).
    6. MARKET CONTEXT: gate/score against BTC — BTC regime, each coin's beta and
       correlation to BTC, and RELATIVE STRENGTH vs BTC. Deprioritize alts
       fighting a hostile BTC; surface leaders (outperforming BTC) over laggards.

  RANKING: a transparent COMPOSITE score — trend clarity + relative strength +
  volatility regime + FRESHNESS (penalise extended/late price) + liquidity. NOT
  ADX alone (ADX is lagging and says nothing about where in the move price is, nor
  about leadership). A high rank means "worth a deep look", NOT "a signal" — an
  extended leader can rank well yet offer no fresh entry.

  OUTPUT: a table the user reads and selects from. For each candidate show:
    symbol · driver/influence-group (which leader it tracks) · 24h volume · spread ·
    book depth · ATR% + vol-regime (squeeze/expanding) · extension/freshness flag ·
    regime (trending/ranging/choppy) · trend direction · relative strength vs its
    DRIVER and vs BTC · beta · and a one-line "why it passed."
  Also print HOW THE UNIVERSE NARROWED (count rejected at each stage) and the
  current MARKET CONTEXT banner (what BTC/the market is doing).
  Then print ONE dollar worked example for the top candidate (clearly labelled
  ILLUSTRATIVE — not a signal) so the user sees the teaching output end-to-end.

  NOTE: expect the shortlist to be dominated by BTC/ETH/top alts — that is
  CORRECT; they are the most liquid, deepest, longest-history markets.

  TWO TIERS: this screener is the CHEAP wide gate (≈600 → top-N) using structural
  proxies — its rank means "worth a deep look", NOT "a signal". The edge-ranked
  OPPORTUNITY BOARD (see "Edge Score & Opportunity Board") runs the deep engine
  only on these survivors and ranks them by PROVEN edge — that is what the user
  ultimately picks from, once setups + backtest exist.

---

## LAYER 1 — MARKET CONTEXT & INFLUENCE GROUPS  (the tide + the packs)
market_context.py
  Before judging any coin, judge the market AND the pack it moves with.

  THE TIDE (macro):
    - BTC regime (trend/range/choppy) on higher timeframes; risk-on vs risk-off.
    - BTC dominance direction (alts-favourable or not); total-market proxy.

  MACRO ANCHORS: BTC and ETH are always-present market factors (BTC = the tide,
  ETH = the alt complex). Each coin gets correlation + beta + RELATIVE STRENGTH to
  BOTH — and to its own group leader (below).

  INFLUENCE GROUPS (the packs):
    - Data-driven clustering: group coins that move together; NAME each group by
      its most-liquid member (its de-facto leader), plus the BTC and ETH anchors.
    - Each coin reports its PRIMARY DRIVER (the anchor/leader it tracks most) and
      its RS/beta to that driver — so "leading or lagging" is judged against the
      RIGHT leader, not only BTC.
    - FAKE DIVERSIFICATION made explicit: five coins from one group is ONE bet.
      The tool shows the group; the USER decides whether to stack same-group coins.
    - Correlation-spike warning: in a sell-off everything → 1 (no diversification
      anywhere) — flag it.
  Curated sector/narrative labels (CoinGecko categories) are a later external-data
  enhancement; the price-only grouping above carries most of the value.

  Used to gate/rank the screener, drive GROUP-AWARE portfolio heat (Layer 6), and
  frame every coin's analysis. The rule it enforces: do not fight the tide — and
  know which pack you are really betting on.

---

## LAYER 2 — ANALYSIS  (deep read of the user-selected coin)
analysis.py  (+ structure.py)
  Market-type first-class. The SYNTHESIS layer: runs the six lenses across a
  TIMEFRAME STACK and fuses them into one seasoned-trader read. It DESCRIBES,
  never predicts; readings are OBSERVATIONS, not signals (only backtested setups
  become signals). "No clean read" is a frequent, valid output.

  MULTI-TIMEFRAME (mechanism, not just intent):
    - A configurable TF stack: BIAS (e.g. 1d) → TRIGGER (e.g. 4h/1h).
    - HTF bias GATES the LTF: no bullish trigger read into a bearish HTF bias.
    - ALIGNMENT score = how much the stack agrees (aligned / mixed / opposed).
    - Per-TF stale-data + unclosed-candle guards.

  THE SIX LENSES — each emits a directional read (bull/bear/neutral) + a strength
  score, so they can be FUSED (not merely listed):
    1. TREND & STRUCTURE — HH/HL vs LH/LL + BOS/CHoCH (from structure); MA stack & slope.
    2. MOMENTUM — RSI / MACD (confirming vs waning); DIVERGENCE (price vs RSI on the
       swings — regular & hidden).
    3. VOLATILITY REGIME — ATR percentile; squeeze (coiling) vs expansion (late).
    4. VOLUME & PARTICIPATION — volume trend; OBV/CVD; volume profile / value area
       (is the move backed by real participation?).
    5. KEY LEVELS & LIQUIDITY — S/R, round numbers, value-area edges, retracement
       zones, liquidity pools (from structure).
    6. RELATIVE STRENGTH & MARKET CONTEXT — coin vs its DRIVER/group leader AND vs
       BTC (market_context); the tide; (usdm) funding cost+signal & OI trend (markets).

  SYNTHESIS — confluence AND conflict (the part that makes it expert):
    - Aggregate the per-lens reads (weighted) into an overall BIAS + a CONFLUENCE
      score (how many lenses agree, weighted).
    - SURFACE CONFLICT explicitly — the BEAR case: divergences, price into
      resistance on weak volume, falling OI into a move, counter-tide. A pro weighs
      what could go WRONG, not just what agrees.
    - LOCATION: classify WHERE price is now — at support / at resistance / mid-range /
      in the retracement (golden-pocket) zone / at a value-area edge / breaking out —
      because location decides whether a setup is even possible.
    - INVALIDATION as a concrete LEVEL (the structure swing / BOS/CHoCH that must
      hold), not prose — the exact price that proves the thesis wrong.

  OUTPUT (presentation principle — decision first):
    HEADLINE: bias · location · confluence-vs-conflict · key levels · invalidation;
    then the per-lens breakdown; then the dollar worked example. `--explain` reveals
    the math. Hands a structured READ to Layer 3 (setups) for named-pattern detection.

  STRUCTURE & KEY LEVELS (structure.py) — expert grade:
    - SWINGS: ATR-thresholded ZigZag legs (not fixed fractals), tagged minor/major
      by leg size, so swings are meaningful and noise is filtered out.
    - MARKET STRUCTURE: the swing SEQUENCE → HH/HL (bullish) vs LH/LL (bearish),
      plus BREAK OF STRUCTURE (BOS, continuation) and CHANGE OF CHARACTER
      (CHoCH, the first break against the trend → possible reversal).
    - KEY LEVELS: cluster swings into S/R zones, SCORED by touches + recency +
      reaction size + confluence (round number / range edge / MA agreeing), then
      PRUNED to the significant few (min separation, top-K) — not 20 noisy lines.
    - RETRACEMENT / MEASURED MOVE: of the last leg (38/50/61.8/79% + golden
      pocket) for pullback entries; measured-move & extension projections for targets.
    - VOLUME PROFILE: POC and value-area high/low (where price actually traded).
    - LIQUIDITY POOLS: equal highs/lows = stop clusters (buy-side above /
      sell-side below) that price hunts — lens 5's "where stops cluster."
    - ROUND NUMBERS (magnitude-scaled) and PRIOR-RANGE (Donchian) edges.
  Consumed by:
    - entries → trade AT a level / retracement zone,
    - stops   → BEYOND a level + liquidity (ATR as a min-distance sanity check),
    - targets → the NEXT level / measured move / opposite liquidity,
    - setups  → the rule definitions in Layer 3.

---

## LAYER 3 — SETUPS  (the strategy layer; defined BEFORE any backtest)
setups.py
  A backtest must test NAMED rules — you cannot prove "edge" on a vague idea. A
  setup is a complete, fixed CONTRACT, and detecting one emits a structured SIGNAL
  that `stage`, `backtest`, and `edge_score` all consume IDENTICALLY.

  DETECTION CONTRACT (the rule that makes edge real, not a backtest illusion):
    detect(bars_up_to_i, structure, context) -> Signal | None
    A PURE function over CLOSED bars that decides at bar i using ONLY data through
    i (no look-ahead). The LIVE detector and the BACKTEST run the SAME code — so
    what you backtest is exactly what you trade.

  SETUP CONTRACT (every setup specifies, as exact conditions):
    - direction (long / short / both) + market (spot = long-only);
    - PRECONDITIONS: regime + analysis-state gates (only evaluated when they hold);
    - ENTRY TRIGGER + ENTRY TYPE (limit at a level / stop on a break / market on
      confirmation) — the type drives fill + slippage modelling in the backtest;
    - STOP: structure-anchored (beyond a swing/level), ATR as a min-distance check;
    - TARGET(S): TP1/TP2 + partial sizing (next level / measured move / opposite edge);
    - PRE-ENTRY INVALIDATION: voids the setup BEFORE entry — distinct from the stop;
    - TIME-STOP / EXPIRY: a pending setup is cancelled after N bars or on invalidation.

  PARAMETERS — robustness over fit (the quant-honesty rule):
    Setups use FIXED, sane DEFAULT parameters (which MA, retrace depth, buffer ATR,
    confirmation rule, expiry bars). NO per-coin curve-fitting at this layer — any
    optimisation happens ONLY in the walk-forward engine (Layer 5) with out-of-
    sample validation.

  VOCABULARY — COVER THE MECHANISMS so "NO TRADE" is authentic (not blindness):
    The set spans the FOUR ways an edge arises — continuation, breakout/expansion,
    mean-reversion, and reversal/trap — long + mirrored short (usdm), regime-gated.
    Completeness of RECOGNITION is the goal; the EDGE BAR does not move — each setup
    must still INDEPENDENTLY prove +EV before it fires live. The cost of a wider
    vocabulary is multiple-testing: more setups = more chances a junk one looks +EV
    by luck, so the edge gate MUST tighten in step (per-setup sample minimums, a
    multiple-testing correction, "enough trades in the CURRENT regime"). Vocabulary
    and rigor scale TOGETHER — a wide set without rigor is worse than a small one.

    A. CONTINUATION  — trend_pullback (pullback to MA/level, resume);
       momentum_flag (major impulse -> tight coil -> STOP break in trend direction).
    B. BREAKOUT/EXPANSION — breakout_retest (enter the retest, limit); breakout_momentum
       (enter ON the break via STOP, no retest — rides moves that run; ADX-gated);
       squeeze_breakout (a low-ATR%ile COIL that releases — the ignition momentum's
       ADX gate would skip while quiet).
    C. MEAN-REVERSION — range_fade (fade a clean range extreme; range regime only).
    D. REVERSAL/TRAP — failed_breakout (break of a range edge that reclaims: spring/
       upthrust); liquidity_sweep_reversal (wick beyond an equal-highs/lows POOL then
       reclaims — sharper, tighter stop, the top/bottom before a big move);
       choch_reversal (Change-of-Character: the prior trend's structure breaks);
       divergence_reversal (regular RSI divergence at a structural extreme).
    Each: structure-anchored stop, TP1/TP2, pre-entry invalidation, regime gate.
    The set is now COMPLETE (10 setups across the 4 mechanisms, long + mirrored short).
    Overlapping reversals dedup by grade (sweep > failed_breakout). Per-coin×TF most show
    "too few trades" — which is why the `optimize` engine (below) POOLS across coins.

  CONFIG OPTIMIZATION (`optimize` — the step-3 rigor that makes a wide vocabulary trustworthy):
    The PROCESS is the proof, engineered so overfitting cannot hide. Per setup × parameter:
    pool trades ACROSS COINS; split each coin chronologically into TRAIN / TEST / LOCK-BOX;
    SELECT only on train; VALIDATE on the out-of-sample test; correct for multiple-testing
    (bootstrap significance, Bonferroni-divided by grid size); demand a stable PLATEAU + survival
    under pessimistic COSTS + a non-negative LOCK-BOX. Objective = robust OOS expectancy
    (penalised for cross-coin inconsistency). ADOPT only if all gates pass, else KEEP DEFAULT
    (the common, honest outcome). Output is a PROPOSED DIFF — never auto-applied. Offline/slow.

  GATING & OUTPUT: only setups whose regime/direction/market match the current
  analysis state are evaluated; detect returns ALL matched setups, RANKED — the
  Edge Score picks the one-per-coin best.

  GRADING (A/B/C) — present-conditions quality that FEEDS the Edge Score:
    (a) CONFLUENCE: trend alignment, level strength, momentum, relative strength,
        participation, market context, freshness (not extended);
    (b) net REWARD:RISK after fees + slippage;
    (c) the setup's BACKTESTED expectancy on this coin / regime (cached edge profile).
  Hard caps: a thin R:R (≲ 1.3R), or unproven / −EV, or counter-tide CANNOT be
  grade A. Low grades are reported but discouraged. The grade is the present-quality
  input to the Edge Score (which adds backtested edge × confidence). DEFAULT IS NO
  SETUP = NO TRADE.

---

## MARKETS — SPOT / USD-M mechanics  (markets/ : base.py, spot.py, usdm.py)
The market-mechanics abstraction so every other layer (risk, analysis, backtest,
order) stays market-agnostic. base.py defines the interface; spot.py / usdm.py
implement it.

  INSTRUMENT METADATA (source of truth, from ccxt precision/limits):
    - tick size, lot/step size, MIN-NOTIONAL, min/max qty, contract size.
    - All sizing ROUNDS to these and rejects sub-min-notional orders — wrong
      rounding = rejected / garbage orders.

  SIZING:
    - SPOT (long-only): size(base) = risk$ / stop-distance; cost = size × entry
      (needs that much quote balance); no leverage.
    - USD-M (long/short): size is leverage-INDEPENDENT (still risk$ / stop-dist).
      Leverage only sets MARGIN (= notional / leverage) and LIQUIDATION distance.

  LEVERAGE ↔ MARGIN ↔ LIQUIDATION (usdm, done right):
    - Choose leverage so the LIQUIDATION price sits well BEYOND the structural stop
      (+ a buffer) — never size up just because leverage allows it.
    - Liquidation from Binance MAINTENANCE-MARGIN TIERS (notional-tiered MMR), not
      a naive 1/leverage formula; ISOLATED margin by default.
    - Respect each symbol's MAX-LEVERAGE cap (Binance notional brackets).
    - HARD RULE: if no leverage keeps liq beyond stop+buffer at a sane size → NO TRADE.

  FUNDING (usdm):
    - Read current funding rate + next funding time; estimate funding COST over the
      expected hold (nets into R:R / expectancy for held perps).
    - Extreme funding = crowded positioning = a CONTRARIAN signal (lens 6).

  OPEN INTEREST (usdm):
    - Current OI + its TREND/delta: rising into a move = new money (real);
      falling = short-covering (hollow). Feeds lens 6 participation.

  FEES: spot vs USD-M maker/taker schedule (config-tunable) — the source the risk
  and backtest layers use for net-cost math.

  base.py INTERFACE (what the rest of the tool calls):
    round_price · round_amount · min_notional_ok · size_from_risk ·
    margin_required · liquidation_price · max_leverage · funding() ·
    open_interest() · fees().  SPOT = simple (long-only, no leverage/funding/OI);
    USD-M = full.

---

## LAYER 4 — RISK & TRADE MANAGEMENT  (+ worked examples on every output)
risk.py
  Orchestrates a Signal (entry/stop/targets) + the markets layer + account state
  into a complete, validated, R-native trade PLAN — and owns the trade-management
  RULES (shared with the backtest). Boundary: risk.py = the PER-TRADE plan +
  management; guards (Layer 6) = the PORTFOLIO gate (heat / daily-loss / cooldown).

  SIZING (risk-first, reconciled):
    - Choose dollars at risk (equity × risk%) FIRST; derive size from the stop
      distance. Size is an OUTPUT, never an input.
    - STOPS are STRUCTURE-ANCHORED (beyond a swing/level), ATR only as a MINIMUM-
      distance sanity check.
    - POST-ROUNDING RECONCILIATION: after lot/tick rounding (markets layer), the
      DISPLAYED risk is the ACTUAL risk. If min-notional forces a size whose risk
      exceeds the budget → NO TRADE (or flag); never silently over-risk.

  COSTS (honest — can flip a gross-OK setup to NO TRADE):
    - NET R:R after FEES + SLIPPAGE (both sides) + (usdm) FUNDING CARRY over the
      expected hold. If NET R:R < min → not worth taking → flag / NO TRADE.
    - The dollar worked example shows NET (after-cost) dollars, not gross.
    - (usdm) liquidation-distance validation (stop safely inside liq); margin/balance check.

  TRADE MANAGEMENT — a deterministic, look-ahead-safe STATE MACHINE, run IDENTICALLY
  live (Layer 10) and in the BACKTEST so the proven edge matches reality:
    manage(position, bar) -> actions
    - partial TP1 (scale out a fixed fraction, e.g. 50%) → move stop to BREAKEVEN,
      runner to TP2; optional TRAIL-to-structure (last swing) on the runner.
    - FILL ASSUMPTIONS: a bar spanning both stop and target → assume the STOP filled
      first (worst case); a GAP past the stop fills at the GAP (extra slippage).
    - Outputs realized R; the worked example shows the BLENDED scenarios
      (stopped −1R / TP1→breakeven / TP1→TP2).

  CAPITAL PROTECTION — the EFFECTIVE-RISK PIPELINE (one composable function,
  effective_risk_pct(), WIRED into analyze/stage/roar so the size you see is the size
  the rules permit — not just the base %):
    base risk% → ×DRAWDOWN-SCALE → ×EDGE-SCALE → ×VOL-TARGET(σ_R) → cap by FRACTIONAL-KELLY
    → clamp to a hard MAX. Each stage is independently toggleable + tunable from .env; with
    all off it is exactly the flat base %.
    - DRAWDOWN-SCALED sizing: reduce risk% as account drawdown deepens (halve past −X%,
      restore on recovery) — reads live drawdown from the journal's equity curve.
    - EDGE-SCALED sizing (size from the EDGE, not the appetite): scale risk by the
      proven edge × confidence relative to a reference edge, CLAMPED to [min_mult,
      max_mult]. High-conviction proven setups get more, marginal ones less. DEFAULT
      OFF (flat). Only ever applied to a setup that already PASSED the null-proven gate.
    - VOLATILITY-TARGET sizing (ENHANCEMENTBRIEF P12 — the 'variance' input): scale risk by
      target_σ_R / σ_R (the setup's per-trade R standard deviation, from the cache), CLAMPED;
      a wilder setup is sized SMALLER for the same edge so each trade carries similar dollar-
      volatility ("consistency of discretionary execution"). DEFAULT max_mult=1.0 → only ever
      REDUCES (conservative). Live: divergence_reversal σ_R 10.8 → ×0.50, breakout_retest
      σ_R 2.0 → ×0.75, trend_pullback σ_R 0.78 → flat. DEFAULT OFF.
    - FRACTIONAL-KELLY CAP: f* = W − (1−W)/R; take a fraction (¼–½); only ever REDUCES
      vs what came before (never inflates); DEFAULT OFF; uses the setup's backtested
      W / avg-win / avg-loss (no proven edge → skipped).
    - HARD CEILING: a final max_risk_pct clamp the pipeline can never exceed — the last
      backstop when a user dials risk up for testing.
    The trade still passes the PORTFOLIO guards (heat, daily-loss, max-positions, the
    new per-session trade cap) AFTER sizing — sizing and the portfolio gate are distinct.
    P12 INPUTS (all 7 the brief asks for): edge + confidence + variance(σ_R) + drawdown +
    sample-quality (n/(n+k), surfaced) are per-trade; CORRELATION + PORTFOLIO EXPOSURE are
    managed at the BASKET level (roar: correlated groups share a budget, total heat capped).

  EVERY risk output is accompanied by the dollar worked-example (now net + blended), and a
  POSITION-SIZING GUIDANCE panel (P12) that lists the evidence (edge/confidence/sample-quality/
  σ_R/drawdown) and which stages fired (e.g. "base 1.0% → edge ×0.5 → vol-target ×0.75 → 0.38%")
  — sizing made auditable + discretionary, never automated.

---

## LAYER 5 — EDGE ENGINE  (prove edge on real history BEFORE any live wiring)
backtest.py + expectancy.py
  Runs the SAME detect() + simulate() code as live across real history, with the
  rigor that separates a measured edge from a backtest mirage. Output is an
  EDGE PROFILE per (coin × setup), reported BY REGIME, that the Edge Score caches.

  ENGINE (the simulation contract — honest fills):
    - Walk closed bars; at each, compute structure on bars-up-to-i (no look-ahead)
      and run detect(). ONE position at a time per coin/setup (no pyramiding).
    - FILL BY ENTRY TYPE: market → next bar's OPEN (taker); LIMIT → fills only if a
      later bar trades to the level (maker), else EXPIRES (time-stop) or is
      INVALIDATED first; STOP → fills on the break (taker + slippage). A pending
      order that never fills is NOT a trade.
    - Management via simulate() (TP1→breakeven→TP2/trail; stop-first / gap fills).

  COSTS (tied to entry type):
    - LIMIT fills = MAKER fee; market/stop = TAKER + slippage (+ half-spread).
    - (usdm) FUNDING CARRY netted across funding windows for held perps.

  NO LOOK-AHEAD anywhere: the regime for "by regime" is labelled POINT-IN-TIME at
  the entry bar; structure/indicators use only data through the signal bar.

  TIME-WEIGHTED LEARNING (ENHANCEMENTBRIEF P7 — markets evolve, so recent trades carry more
    weight; older fade GRADUALLY, never discarded): every Trade carries entry_ts; when enabled,
    expectancy/SE/win-rate/avg-win-loss/σ become a WEIGHTED estimate, weight = 0.5^(age_days /
    half_life) (exponential, default half-life 90d). SE uses the KISH effective-N (n_eff =
    (Σw)²/Σw²) so heavy decay → fewer effective samples → wider CI → lower confidence (self-
    honest). Order statistics (median/percentiles/max-DD/streak) stay on the realized path; the
    NULL stays UNWEIGHTED (random-entry CONTROL, not a learned edge). DEFAULT OFF → identical to
    today; opt-in (BACKTEST_TIME_DECAY_*); `scan --refresh` bakes it into the cache so board/gate/
    sizing/confidence all adapt. Live: reweights each setup by whether its RECENT trades out/under-
    performed (breakout_retest +0.026R, trend_pullback −0.032R — they move independently).

  STATISTICS (is the edge real, or noise?):
    - EXPECTANCY with a CONFIDENCE INTERVAL / bootstrap; the verdict uses the
      LOWER BOUND, not the point estimate.
    - Win rate, avg win/loss R, profit factor, system quality (expectancy ÷ SD of R).
    - MULTIPLE-TESTING honesty: report how many (coin×setup) combos were tested —
      "+EV on 1 of N by luck" — and lean on the lower bound + fold-consistency.
    - NULL BASELINE (the bar is NOT zero): a path-dependent exit (scale-out at TP1
      → move to breakeven → runner) yields POSITIVE expectancy even on no-edge
      entries — the "free option" artifact — so comparing expectancy to ZERO falsely
      stamps junk as +EV. The honest null is the SAME exit + the SAME risk geometry
      on entries with NO predictive content: for every real trade, spawn k SHADOW
      trades — same direction, same stop distance, same TP R-multiples — entered at
      RANDOM bars in the same coin and managed by the same state machine. The
      setup's edge is the EXCESS of its expectancy over this shadow distribution
      (subtracting the exit artifact AND generic directional drift, e.g. "it was a
      bull market"); proven requires the LOWER BOUND of that excess > 0. (Considered
      and rejected: return-shuffle nulls also destroy the structure the setup keys
      on — apples-to-oranges; an analytic exit-artifact correction is distribution-
      dependent and brittle. The matched-geometry random-entry null assumes nothing.)
    - WALK-FORWARD: with FIXED-parameter setups (no fitting) this is an out-of-sample
      CONSISTENCY check across non-overlapping folds — does the edge persist, or
      live in one lucky window? (Strict train/test only if any param is optimised.)

  EQUITY / RISK:
    - Max drawdown, longest losing streak, time-to-recovery.
    - MONTE-CARLO reshuffle → RISK OF RUIN and the DRAWDOWN DISTRIBUTION (the
      95th-pctile drawdown to brace for), so the user sees the range, not one path.

  SAMPLE & CAVEATS:
    - Minimum SAMPLE SIZE before any verdict; below it → "too few trades".
    - SURVIVORSHIP caveat stated (Binance shows survivors only — absent delistings
      flatter results).

  VERDICT (per setup × regime) — "proven" is earned only when ALL hold: net-positive
  expectancy AND the edge SIGNIFICANTLY beats the NULL BASELINE — the excess over
  matched-geometry random entries must clear a STRICTER significance bound (sig_z·SE,
  default sig_z=1.65 ≈ 95% one-sided, not the 1-SE display bound), NOT merely > 0 vs
  zero — AND >= min sample (regime sample >= min_regime_n) AND consistent across folds;
  otherwise INCONCLUSIVE, not +EV. −EV setups are rejected. (No null available — e.g. a
  thin single-coin slice — falls back to the lower-bound-vs-zero test, flagged as such.)
  WHY sig_z + min_regime_n: empirically (planted-edge lab, proofs/), the fluke rate is a
  SAMPLE-SIZE phenomenon — pooling across the universe is the main cure (single-tape ~20%
  → pooled ~10%), sig_z=1.65 trims it further (→~5%), and at the live board's real sample
  sizes (hundreds) it approaches ~2%. The residual thin-regime fluke is caught by defense-
  in-depth: you CONFIRM every trade, size at a fixed % risk, and `review` flags realised-
  vs-backtested decay. Hardening the bar trades a little recall (faint real edges) for
  reliability — by design.

  VALIDATION DEPTH (ENHANCEMENTBRIEF P6 — distinguish PROFITABLE from CONSISTENTLY profitable;
    REPORT-ONLY, does not change the score/verdict/sizing): a ROBUSTNESS read ∈ [0,1] =
    OOS-fold-consistency × bootstrap-stability (resampled CI lower bound > 0 → 1, straddles 0 → ½,
    < 0 → 0) × walk-forward non-decay (recent-third vs overall edge held up). Any weak axis drags
    it down — so a setup that looks great overall but COLLAPSED recently scores ~0. Shown in
    `backtest` per setup ('robustness 0.33 — profitable but not robust: OOS folds 67%+ · bootstrap
    straddles 0 · walk-forward …') and on the analyze proven-edge panel; rides along on the cached
    verdict (es.robustness + lookup_pooled_verdict). Live: a setup with overall +1.02R but recent
    −1.23R correctly reads walk-forward 0.00 → robustness 0.00 (the brief's exact 'profitable ≠
    consistently profitable'). REWARD path was kept report-only by choice — robustness already
    enters scoring indirectly via P2 confidence's fold-consistency term; this surfaces the DEEPER
    evidence (bootstrap + walk-forward) without double-counting or changing the tested scorer.

  EDGE PROFILE (the cached contract the Edge Score consumes): expectancy + CI by
  setup and by regime, sample size, OOS fold-consistency, cost assumptions, verdict.

---

## EDGE SCORE & OPPORTUNITY BOARD  (the headline UX — what the user picks from)
  CONTEXTUAL STRATEGIES (ENHANCEMENTBRIEF P1 — the context IS part of the setup's identity):
    A setup's edge is conditioned not only on regime but on the MARKET CONTEXT at the signal bar —
    `trend_pullback · down · squeeze · htf-aligned · no-divergence` is a DIFFERENT strategy from the same
    setup in an expanding, htf-opposed, divergent context. Each historical Trade captures a CONTEXT tag
    (point-in-time, no look-ahead) computed from the same structure/indicators the live read uses:
      vol_regime (squeeze/normal/expanding) · momentum (RSI/MACD bucket) · location (at-level vs mid) ·
      htf_align (vs the BIAS TF — backtest now fetches+aligns it) · divergence (none/bull/bear).
    AVOID COMBINATORIAL EXPLOSION (the brief's own warning): NOT the full cross-product. Evaluate
    SINGLE-FEATURE splits of each setup×regime and KEEP a context cell only if it EARNS its specificity —
    its edge BEATS THE PARENT (setup×regime) with significance, gated by the SAME null + sig_z, SHRUNK
    toward the parent (P5 extended to the context axis — this is what activates the deeper hierarchy),
    with a multiple-testing penalty for the number of features tried + OOS fold-consistency. Greedy,
    depth-capped. The live board (analyze already computes the current context) scores a coin's setup on
    the MOST-SPECIFIC PROVEN context cell matching its conditions, FALLING BACK to setup×regime when no
    context cell is proven. CONTRACT: default = the parent (today's behaviour); conditioning only ever
    REFINES when the data proves the context matters — never lowers the edge bar. Built in provable order:
    capture → MEASURE whether any refinement beats its parent on live data → wire only the winners.

  FEATURE INTERACTIONS (ENHANCEMENTBRIEF P4 — "edge emerges from COMBINATIONS"): evaluate_conditioning
    now also tests FEATURE PAIRS (F1=v1 & F2=v2), generalised to multi-condition cells (conditions list;
    _context_cell_reg matches ALL conditions; the board row shows an 'A & B' label). To honour the brief's
    explicit warning against combinatorial explosion, pair candidates come from a GREEDY screen (default) —
    only pairs built on a RAW-significant single feature — keeping the Bonferroni count sane; `scan --hard`
    instead tests EVERY value-pair (exhaustive, far larger m → far stricter bar). A pair is PROVEN only with
    the full single-cell gate (money + Bonferroni-over-all-splits + OOS) AND it must BEAT THE BETTER OF ITS
    TWO SINGLE PARENTS' significant edge-over-rest (lift_low) — a GENUINE interaction, not inherited edge
    (this lift_low test is noise-robust where a naive exp>parent_exp is not). Live: greedy surfaced a single
    marginal candidate (breakout_momentum/down · vol=expanding & mom=bear, +0.55R, lift_low +0.03) that did
    NOT survive a fresh pool re-screen NOR `--hard` (m 189→577) → 0 robustly proven → board unchanged, the
    correct conservative result (a +0.03 'edge' that flips across snapshots is not real). Auto-activates when
    a STABLE interaction earns it; the synthetic tests prove it detects true synergy + rejects inherited edge.

  CONTINUOUS FEATURE EVALUATION (ENHANCEMENTBRIEF P13 — features gain and lose predictive power;
    reassess periodically + let the important ones carry more influence): at every cache rebuild,
    es.feature_importance (a cheap SINGLES-ONLY pass) scores each context feature by its predictive
    value — n_significant (raw-significant positive directions = breadth) + capped mean lift (depth)
    → an `importance` scalar; robust even when nothing clears the strict bar. es.record_feature_
    importance APPENDS a snapshot to data/feature_importance_history.json and returns a per-feature
    TREND (rising / decaying / flat / new) vs the PREVIOUS snapshot for this market×tf. The TOP-K
    (EdgeScoreConfig.feature_top_k=4) important features become the ACTIVE set that may seed the P4
    greedy pair search — important features carry greater INFLUENCE, decayed ones DECLINE (the
    Bonferroni gate + beat-parents test are UNCHANGED; only candidate ordering adapts). The readout
    is stored in the universe cache + shown by the `features` command (ranked table + ● active +
    trend). Live: vol is the most predictive feature (importance 3.22, 3 significant directions);
    top-4 = vol/loc/mom/arch seed pairs, htf_align/div decline; trend flips new→flat across a
    same-data rebuild (real drift shows over calendar time). Recompute = `scan --refresh`.

  REGIME ARCHETYPES + REGIME-SPECIFIC LEARNING (ENHANCEMENTBRIEF P14 + P3 — recognise the
  ENVIRONMENT, learn an edge conditional on it):
    Every signal bar is tagged with a named, disjoint ARCHETYPE derived ONLY from primitives the
    engine already computes — structural trend (up/down/range) × ATR-%ile volatility (squeeze/normal/
    expanding) × recent thrust (N-bar return / ATR) × the leg into a range — via a priority-ordered
    decision tree (most-specific first, one label per bar):
      panic (down+expanding+sharp-down) · euphoria (up+expanding+sharp-up) · accumulation (range,
      fell in) · distribution (range, rose in) · vol_expansion · vol_compression · trending_up ·
      trending_down · range.  The label is DESCRIPTIVE — the EDGE comes from the measured conditional
      expectancy of the cell, not the name (P14 names the trend×vol GRID cell; P3 measures it).
    P3 = each setup accumulates a CONDITIONAL EXPECTANCY per archetype, each shrunk toward the setup's
    overall edge (P5 on the archetype axis), surfaced ALWAYS as evidence (the report can say "this setup
    earns +0.42R only in panic, −0.44R in euphoria" — structure a single pooled −0.08R number hides).
    GATED PROMOTION (the discipline): the archetype is just another single-feature context axis fed to
    the SAME evaluate_conditioning gate — it only changes a live score/size when its cell MAKES money AND
    beats the rest with BONFERRONI significance AND is OOS fold-consistent (the named archetype becomes
    the board's `·label`). Else → the setup×regime parent + P5 shrinkage. Adding the archetype RAISES the
    Bonferroni bar (more splits) — "scale rigor with recognition," never a looser path. Live-measured:
    sane 9-way distribution; breakout_retest +0.42R in panic vs −0.08R pooled SURFACED as evidence; 0
    cells clear the strict gate yet → board unchanged, auto-activates when one earns it. Knobs in
    SetupConfig (arch_thrust_bars/atr, arch_leg_bars/atr), env-overridable.

  HIERARCHICAL SHRINKAGE (ENHANCEMENTBRIEF P5 — borrow statistical strength, don't discard it):
  the flat pool applies ONE setup×regime edge to every coin identically (ENA gets the universe
  number; its own 13 trades are ignored). Instead, estimate a PER-SYMBOL edge as an EMPIRICAL-BAYES
  blend of the symbol's OWN setup×regime trades shrunk TOWARD the pooled prior:
    prior ~ N(μ_pool, τ²)  ·  likelihood ~ N(m_symbol, σ²=s²/n)  ·  posterior = precision-weighted blend,
    where τ² (between-symbol variance) is estimated method-of-moments = max(0, spread of symbol means − mean sampling var).
    weight on own data w = (1/σ²)/(1/σ² + 1/τ²): thin coin (small n) or no symbol signal (τ²≈0) → shrink to μ_pool
    (today's behaviour, exactly when τ²=0); rich, genuinely-different coin → trust its own data. The shown SE adds
    the grand-mean uncertainty so a coin we've NEVER traded is never scored MORE permissively than the pool.
  The shrunk per-symbol estimate then passes the SAME gates (null baseline + sig_z significance on the POSTERIOR
  variance + floor) — a thin coin legitimately borrows the universe's evidence; a coin whose own data contradicts
  the pool pulls away and can correctly flip to unproven. Stored transparently (μ, τ², per-symbol n/mean/sd) for
  explainability. This is the FOUNDATION that makes context-conditioning (P1) safe: more conditioning → thinner
  cells → shrinkage stabilises them. Falls back to the flat pool when a coin has no own trades / on an old cache.

  PROFESSIONAL REPORTING (ENHANCEMENTBRIEF P8/P9/P10/P11 + P15 — output BETTER information,
  not more numbers; every metric must change a decision). All built on stats ALREADY computed
  (Stats/MonteCarlo) — supplement, don't replace; surfaced in `backtest` (and `analyze`):
    P10 EDGE vs EXECUTION — decompose the raw statistical edge from its costs: every Trade now
      carries gross_r (pre-cost) + funding_r; evaluate reports gross → fees/slippage → funding →
      NET. (Live: breakout_retest gross +0.37R, costs eat ~40% → NET +0.22R — you SEE where edge
      leaks.) _cost_breakdown splits exec (fees+slippage, both sides) from funding/carry.
    P9 DISTRIBUTION — what the average hides: median R, avg win / avg loss, σ (variance), Fisher-
      Pearson skew, p10 / p90 outcomes (Stats), plus a non-parametric BOOTSTRAP CI (robust to fat
      tails) and RECENT (last-third) vs overall expectancy on EdgeProfile. (Live: breakout_retest
      mean +0.22R sits on a −1.10R MEDIAN with skew +2.8 — a few big winners carry it.)
    P11 RISK INTELLIGENCE — the drawdown DISTRIBUTION, not just the max: monte_carlo now returns
      median + 95th-pctile max-DD, risk-of-ruin, and EXPECTED LOSING STREAK (mean longest run of
      losers across shuffles); the report adds a recovery estimate (p95 DD ÷ expectancy). Answers
      "what drawdown / losing streak should a trader realistically expect?"
    P8 EXPLAINABILITY — every recommendation auditable: es.explain_signal assembles, from signals
      ALREADY computed (the six lenses + the edge gate + freshness/tide/context cell + confidence
      tier), the factors pushing FOR vs AGAINST the trade — decisive contributor first. No new
      heuristic, no new score. (`analyze` prints a "Why — evidence for & against" panel.)
    Knob: BacktestConfig.bootstrap_runs. `--full` reveals bootstrap CI + recent-vs-overall + folds.

  POOLED EDGE (the fix for per-coin "too few trades"): a setup's edge is a property of
  its RULES, proven on trades POOLED ACROSS the liquid universe (per setup × regime), not
  re-proven on one coin's thin slice (where most setups never reach the sample minimum and
  are wrongly auto-failed). Pooling is also less overfit. scan scores each coin's LIVE setup
  against the pooled verdict; `analyze`/`stage` consult it too — stage BLOCKS a proven −EV
  setup and WARNS on the unproven, so the board, analyze, and stage stay coherent (no more
  "grade-A but backtests −EV"). The universe profile is cached (rebuild via `scan --refresh`).

edge_score.py  ( `scan` )
  Turns the screener's survivors into a board of ranked OPPORTUNITIES — each
  pre-analysed, backtested, graded — so the user picks from PROVEN edges. It ranks
  only coins that have a LIVE setup now AND a proven edge for it. An EMPTY board
  ("no proven edge anywhere right now → stand aside") is a VALID, FREQUENT result;
  the tool never manufactures an opportunity to fill the board.

  EDGE vs CONFIDENCE (a quant never conflates +0.6R over 12 trades with 400):
    EDGE       = backtested expectancy (R) for this setup IN THE CURRENT REGIME.
    CONFIDENCE = STATISTICAL, evidence-driven (ENHANCEMENTBRIEF P2 — "how certain are we
                 the edge actually EXISTS?", not "how many signals agree?"):
                 P(edge beats the random-entry null) × fold-consistency. The probability is
                 parametric Φ(mean/SE) where SE is recovered from the cached 1-SE CI — so it
                 folds in historical SAMPLE QUALITY + VARIANCE + the CONFIDENCE INTERVAL in one
                 number, × OOS fold-robustness. P(expectancy>0) is shown ALONGSIDE (the brief's
                 literal metric) so the user sees the distinction: a setup can be 96% likely +EV
                 yet only 88% likely to beat RANDOM (live: trend_pullback/down). No informative
                 null → falls back to P(>0). Thin sample → wide SE → low confidence (self-gating).
    Rank by the CONSERVATIVE edge, GATED by the null (the null is a GATE, not a free
    magnitude bonus): a setup qualifies only if it (a) MAKES money in the current regime
    (expectancy > 0 — "loses less than a random trader" is STILL a loser, never an edge),
    AND (b) BEATS the random-entry baseline with SIGNIFICANCE (excess over the regime-
    matched shadows clears sig_z·SE, default sig_z=1.65). The shown edge subtracts only a
    POSITIVE (artifact) null and NEVER exceeds the setup's own expectancy. Fail either
    gate, or edge ≤ a floor → NO EDGE. (The board scorer and the stage gate share ONE
    function so they can never disagree — `scan` and `stage` always tell the same story.)

  EDGE SCORE (exact composition):
    trustworthy_edge_R = lower-bound(current-regime expectancy)   [≤ floor → NO EDGE]
    present_multiplier = grade × freshness × tide-alignment × regime-match (each ∈ [0,1])
    edge_score         = trustworthy_edge_R × present_multiplier
    HEADLINE = expected R + LETTER grade (A–F from edge_score) + CONFIDENCE TIER
               (high/mod/low). e.g. "Edge +0.34R (B+), moderate confidence, fresh, with-tide".
    The Edge-Score grade SUPERSEDES the setup's provisional grade once a profile is
    cached (the provisional grade is only the fallback when there's no backtest yet).

  THE OPPORTUNITY BOARD ( `scan --spot|--usdm` ): ONE row per coin — its best
  (setup, direction) by Edge Score (usdm evaluates long AND short; spot long-only).
  Columns: symbol · driver/group · side · setup · grade · Edge(R) · CONFIDENCE ·
  fresh? · regime. Sorted by conservative Edge Score, with confidence shown
  PROMINENTLY. The user reads top-down and picks; `analyze SYMBOL` drills in.
  BOARD-LEVEL MULTIPLE TESTING: ranking the best of many survivors means the #1 row
  is the MOST likely fluke — the conservative bound + confidence guard it, and the
  board states the caveat. Don't blindly chase #1.

  PIPELINE: scan = screener (Tier-1 gate) → for each survivor ensure its EdgeProfile
  is CACHED (backtest if stale) → compute LIVE present-conditions → Edge Score → rank.
  CACHE: an EdgeProfile per (market, symbol, setup, tf) in /data, with timestamp +
  candle coverage; refresh if stale (older than TTL / new candles) or on `--refresh`.
  COLD scan backtests survivors (slow — progress, cap top-N); WARM scan combines
  cached edge × live present-conditions (fast). Rigor amortised.

  Honesty rail: historical edge with STATED CONFIDENCE, NOT a prediction.

---

## LAYER 6 — PORTFOLIO & BEHAVIOR  (protect the account from the trader)
guards.py — a PRE-TRADE GATE invoked by `stage`; lockout/cooldown state persists
across the session. Boundary: risk.py = the per-trade plan; guards = the
portfolio/behaviour gate over it.

  STATE: guards consume a PortfolioState — open positions (each with $ risk +
  group), today's realised R (since UTC midnight), current equity, consecutive-loss
  streak + last-loss time. (Populated by the journal; passed in, empty in dry-run.)

  RESULT: a GuardResult = ALLOW / BLOCK (hard) / WARN (soft) + the guard(s) that
  fired + reasons. Hard guards block; soft guards warn-and-confirm.

  GUARDS (concrete):
    - DAILY-LOSS LOCKOUT (hard): realised ≤ −X R / −X% equity today → no new trades
      until UTC reset.
    - MAX TRADES PER DAY (hard): no more than N new trades opened since UTC midnight
      (over-trading guard) — counts STAGED+OPEN entries today; 0/unset = unlimited.
    - COOLDOWN (hard): after a loss, block for N minutes/bars; after K consecutive
      losses, lock for the session.
    - PORTFOLIO HEAT (hard, GROUP-AWARE): total open risk %equity, same-group
      positions scaled by intra-group correlation (3 ETH-group longs ≈ ~1 bet);
      incremental check on the NEW trade. SCALE-TO-FIT (reduce size) or block if
      even minimum size won't fit.
    - MAX POSITIONS (hard) + PER-GROUP CAP (soft): ≤ N total, ≤ M per influence group.
    - CORRELATION CAP (soft): warn on stacking correlated bets.
    - EVENT GUARD: FUNDING-WINDOW (computable — Binance funds 00/08/16 UTC) = soft
      warn; token-unlocks / macro (CPI/FOMC) need EXTERNAL data → optional manual
      date list / deferred (don't pretend to guard what we can't see).
    - SANITY BOUNDS (hard): price>0, stop≠entry, finite size, (usdm) stop inside liq.

adherence.py — discipline is MEASURED, not assumed (consumes the journal: planned
  vs executed). Metrics: STOP-MOVED-AGAINST (the cardinal sin), OVERSIZED vs plan,
  OVERRIDE (took a guard-blocked / sub-grade trade), REVENGE (entered in cooldown),
  SKIPPED A-grade setups. adherence % = followed / total; surfaced in `review`.

---

## LAYER 7 — JOURNAL & REVIEW
journal.py — the persistence + feedback layer, and the SOURCE OF TRUTH for the
PortfolioState guards consume and the planned-vs-executed pairs adherence scores.

  STORAGE: append-only JSONL in /data. Each TradeRecord spans the lifecycle
  (STAGED → OPEN → CLOSED / CANCELLED), tagged dry-run vs live:
    - PLAN: symbol, market, side, setup, grade, Edge(R) at entry, entry/stop/targets,
      risk% + $risk, context snapshot (bias, regime, tide, group);
    - OUTCOME: actual entry, FINAL stop (after any moves), exit, realized R, P&L,
      fees/funding, closed_at;
    - DISCIPLINE flags: in-cooldown, guard-overridden.

  ACCOUNT STATE: portfolio_state() derives the live PortfolioState guards need —
  equity (start + realised P&L), OPEN positions (with $risk + group), realised R
  today (UTC), consecutive-loss streak + last-loss time. (Closes the guards
  dependency.) Plus the equity curve for drawdown.

  REVIEW (`review`): realised expectancy OVERALL + by SETUP / REGIME / GRADE; win
  rate, profit factor, max/current drawdown; the ADHERENCE report (stop-moved-against,
  oversized, override, revenge); REALISED-vs-BACKTESTED (is live matching the proven
  edge, or decaying?); actionable verdicts ("your edge is in setup X / coin Y; stop
  taking Z"). Only CLOSED trades feed realised stats; below a min sample → "too few
  closed trades to conclude". Closes the feedback loop that makes a trader competitive.

---

## LAYER 8 — ALERTING
alerts.py — ping when a watched coin reaches an actionable zone, so you act
deliberately (not glued to charts). NO auto-execution, ever.
  - TYPES: LEVEL (price crosses/reaches a price — e.g. a setup entry or structure
    level) and SETUP (a named setup triggers on the latest closed bar).
  - STORAGE: persisted (JSONL in /data); one-shot (fires once → inactive; re-arm to
    keep watching).
  - DELIVERY: prominent console ping + terminal bell; richer channels
    (desktop / Telegram / email) are a config-driven extension.
  - CHECKED on demand (`alert --check`) and on every `watch` pass — the continuous
    board doubles as the alert engine.
  - An alert NOTIFIES only; the user then runs `analyze` / `stage`.

---

## LAYER 9 — ORDER  (last; the only layer that can touch money)
order.py ( `stage` ) — a manual-confirm STATE MACHINE; every step can ABORT.

  FLOW: derive the current setup → GUARDS (hard block → ABORT) → build net plan +
  checklist + worked example → RE-FETCH live data → drift/validity/liq re-check
  (ABORT) → typed CONFIRM → DRY-RUN (default): log STAGED to the journal, print the
  order ticket, SEND NOTHING.

  RE-FETCH + DRIFT/VALIDITY ABORT: at confirm time, abort if (market entry) price
  drifted > X% / X·ATR from the analysed entry, the pre-entry INVALIDATION already
  hit, or (usdm) liquidation no longer sits beyond the stop. (Limit/stop entries
  rest at their level — no drift race.)

  GUARD-OVERRIDE POLICY: HARD guards (daily-loss lockout, cooldown, sanity,
  liq-inside-stop) are ABSOLUTE — never overridable into a live trade. SOFT warns
  may proceed with an explicit ack, LOGGED to the journal as an override (adherence
  catches it).

  DUPLICATE PROTECTION: refuse to stage if the journal already has an OPEN or
  unresolved STAGED position for that symbol/setup.

  ORDER TICKET (dry-run IDENTICAL to live): entry (limit/stop/market per the setup's
  entry type) + stop-market reduceOnly + TP1/TP2 reduceOnly; (usdm) set leverage +
  ISOLATED margin-mode first. Dry-run builds the same ticket, just doesn't transmit.

  CONFIRM (two-tier): typed `CONFIRM` (the literal word, not y/n). DRY-RUN logs the
  plan, no send. `--live` requires key present + an explicit --live flag + a SECOND
  re-fetch + a SECOND CONFIRM before any real order.

  CARDINAL RULE (live): if the stop cannot be placed, the position must NOT stand —
  entry filled but stop failed → immediately close/cancel + alert. Never a naked position.

  JOURNAL/KEY: reads journal.portfolio_state() to feed guards; writes record_staged
  on CONFIRM. `--live` needs a trade-only / no-withdrawal / IP-locked key; the secret
  is never logged.

  LIVE EXECUTION (`--live`, built DEAD LAST; a mode of stage/manage):
    - PREFLIGHT: key present + TRADE-ONLY permission verified (withdrawal OFF) +
      IP-locked + an explicit LIVE-enable env flag. TEST ON BINANCE TESTNET
      (set_sandbox_mode) FIRST — validate the whole send path before real funds.
    - SEQUENCE: (usdm) set leverage + isolated → place entry → ON FILL place the
      protective STOP (reduceOnly); if the stop can't be placed → EMERGENCY
      close/cancel + loud alert (never naked) → then TP1/TP2. Track broker order ids.
    - PARTIAL FILLS: size the stop to the FILLED quantity.
    - RECONCILE: the EXCHANGE is the source of truth — on each manage/watch,
      reconcile journal OPEN positions with fetch_positions / open orders (catch a
      stop-out that happened while away).
    - KILL SWITCH: `--flatten` market-closes all positions + cancels all orders.
    - GATE: the full stage gate PLUS a second re-fetch + a distinct second confirm
      (`CONFIRM LIVE`) before any real order.

---

## LAYER 10 — TRADE MANAGEMENT  (managing the OPEN position by the plan)
monitor.py  ( `manage` )
  Manage an OPEN position by the PLAN, never by emotion. The management rules are
  the SAME deterministic state machine as the backtest's simulate() — so what you
  backtested is exactly how you manage live (no divergence).

  STATE (in the journal record): current_stop, remaining_fraction, tp1_filled,
  broker order ids. manage reads OPEN positions; `manage SYMBOL` one, `manage` all.

  manage_step(position, current price/bar, recomputed structure) → PROPOSED action:
    - price hit TP1 → scale out tp1_fraction + move stop to BREAKEVEN;
    - price hit TP2 → close the runner;
    - price hit the CURRENT stop → exit (stopped);
    - TRAIL the stop to a new protective swing (optional);
    - re-validate INVALIDATION (a close beyond the structural stop → exit);
    - (usdm) re-check liquidation distance + funding/OI as price moves; warn.
  PROPOSE-AND-CONFIRM, never autonomous: it shows the action; the user types
  CONFIRM. On confirm it updates the journal (stop/partial) or CLOSES the trade →
  realised R → feeds review + the guards' PortfolioState.

  PAPER-FILL: a dry-run STAGED trade can be paper-filled when price reaches its
  entry → OPEN → managed/closed on REAL prices, so the whole stage→manage→review
  loop (and the edge) runs WITHOUT real money — and so manage is exercisable in
  dry-run. Emits alerts (Layer 8).

---

## SAFETY (non-negotiable, every layer)
  - No autonomous trading: type-to-CONFIRM, re-fetch before send, dry-run default;
    `--live` last, behind a confirmation gate.
  - No predictive claims: outputs DESCRIBE structure; NO TRADE is valid + frequent.
  - Named, backtested setups only: "no setup" = NO TRADE.
  - Context first: do not long alts into a breaking-down BTC.
  - Risk-first sizing; (usdm) stop must sit inside liquidation distance.
  - API key: trade-only, NO withdrawal, IP-locked, gitignored. Never logged.
  - Live: never a NAKED position (stop-or-bail); exchange is source of truth
    (reconcile); TEST on TESTNET first; explicit LIVE-enable flag + `--flatten` kill switch.

## DATA INTEGRITY
  Stale-feed guard (reject feeds that stopped updating); drop the unclosed
  forming candle before computing indicators; gap/outlier sanity checks;
  rate-limit-respecting fetches; clear, wrapped errors instead of tracebacks.

## TECH STACK
  Python 3.11+, ccxt (incl. funding-rate / open-interest endpoints), pandas +
  pandas-ta, numpy (Monte-Carlo), python-dotenv, rich, pytest.

## ARCHITECTURE
/src
  config.py              — thresholds, market enum, env/.env overrides (all gates visible)
  data_fetch.py          — real Binance data (2+ TFs), order book, funding/OI, integrity guards
  indicators.py          — ATR/ATR%, ADX, RSI/MACD, OBV/CVD, squeeze (+ vol-regime)
  market_context.py      — tide (BTC/ETH anchors, dominance) + influence groups (driver, RS/beta to leader)
  screener.py            — Layer 0: ranked shortlist; incl. extension/freshness flag + group column
  structure.py           — ZigZag swings, market structure (HH/HL, BOS/CHoCH), scored/pruned levels, retracement, volume profile, liquidity pools
  analysis.py            — six lenses + structure + multi-TF + regime; thesis + invalidation
  markets/ base.py spot.py usdm.py   — precision/limits, risk-first sizing, MMR-tier liquidation, max-lev, funding cost+signal, OI trend, fees; market-agnostic interface
  setups.py              — Setup contract + look-ahead-safe detect()→Signal (live≡backtest); entry type, invalidation vs stop, time-stop; fixed params; grade feeds Edge Score
  risk.py                — risk-first sizing (post-rounding reconciled), net-cost R:R, deterministic manage() state machine (shared w/ backtest), drawdown scaling, Kelly cap, net+blended worked example
  backtest.py            — engine: same detect()/simulate() as live; fills by entry type; pending lifecycle; point-in-time regime; walk-forward folds; Monte-Carlo
  expectancy.py          — expectancy + CI, profit factor, system quality, drawdown dist, multiple-testing, EdgeProfile + verdict (by setup × regime)
  edge_score.py          — Edge Score (regime-matched conservative edge × confidence × present) + cached EdgeProfile + Opportunity Board (`scan`); empty board valid
  guards.py              — GROUP-AWARE heat cap, daily-loss lockout, cooldown, event guard, checklist
  adherence.py           — discipline tracking + adherence %
  journal.py             — append-only trade log (lifecycle, dry-run/live); portfolio_state() feeds guards; equity curve; review (realised by setup/regime/grade, adherence, realised-vs-backtest, verdicts)
  alerts.py              — level & setup alerts (persisted, one-shot, console+bell); checked on `alert --check` and each `watch` pass; no auto-exec
  order.py               — stage state machine (guards→plan→re-fetch+drift abort→typed CONFIRM); dry-run ticket identical to live; duplicate protection; cardinal "stop-or-bail"; --live last
  monitor.py             — Layer 10: manage OPEN positions via the SAME state machine as backtest; propose-and-confirm; paper-fill; close→realised R; (live) reconcile + stop-or-bail + kill switch
  charts.py              — OPTIONAL annotated chart images (auto-drawn levels/zones/entry-stop-target)
  cli.py                 — command surface
/tests   /data   /logs

## CLI
  screen   --spot|--usdm                                  # cheap Tier-1 gate: liquid/structural shortlist
  scan     --spot|--usdm                                  # OPPORTUNITY BOARD: survivors ranked by conservative Edge Score
  context  --spot|--usdm                                  # tide (BTC/ETH) + influence groups + leaders/laggards
  analyze  SYMBOL --spot|--usdm --tf 1h [--explain]       # full read: six lenses + structure + setup + edge + worked example
  analyze  --all  --spot|--usdm [--top N] [--setups-only] # BOARD: same engine across the whole shortlist (concurrent), one compact row/coin, setups-first; = scan without the backtest gate
  backtest SYMBOL --spot|--usdm --tf 1h --setup NAME      # per-setup edge report (walk-forward)
  optimize --spot|--usdm [--setup N] [--param P] [--coins N]  # OOS-validated config optimization → PROPOSED DIFF (never auto-applies)
  roar     --spot|--usdm [--tilt] [--live]                 # MASTER: tide→proven board→allocate budget across the BASKET→CONFIRM
  watch    --usdm --start|--stop|--status                  # always-on radar: desktop notify on a NEW proven edge + hourly digest

## ROAR (master command) & WATCH (always-on radar) — grab opportunities at the right time
  roar = one-shot strike on the WHOLE proven basket: take every opportunity passing all gates,
  then DIVIDE THE RISK BUDGET across them like a desk (a basket wins over MANY trades). Equal-risk
  per uncorrelated bet by default; `--tilt` weights by edge×confidence — each capped at base risk,
  correlated coins share a budget, the BASKET total stays under the heat cap / max positions / per
  group (allocator in guards.allocate_basket). Stages all on one CONFIRM (dry-run); `--live` places
  each behind CONFIRM LIVE + arm flag, EACH independently stop-or-bail. Inherits every gate (empty
  board → nothing to do).
  RE-VALIDATE AT SEND TIME (live): the board is built once, but `--live` places AFTER the human types
  CONFIRM LIVE — a window in which a coin's REGIME can flip (a setup proven in `range` becomes −EV in
  `down`), price can drift, or the pre-entry invalidation can hit. So just like `stage`, roar `--live`
  RE-VALIDATES EACH LEG on FRESH data immediately before its send: re-analyze → the same setup/direction
  must still be present → the pooled edge must still be PROVEN in the CURRENT regime (skip on −EV /
  no-longer-proven) → re-fetch drift + pre-entry invalidation must pass → re-size on the fresh price with
  the leg's allocated risk. A leg that fails is SKIPPED and reported; only re-validated legs are sent.
  (Without this, the master strike could place a leg the `stage` gate would have refused — closed.)
  watch = the passive radar that solves "I'll miss it while busy". TWO-CADENCE, BAR-CLOSE-AWARE: the
  board only changes when a trigger-TF candle closes, so it re-scans right AFTER each close (not on a
  blind timer); between closes it does only light price/alert checks (~60s, only if a position/alert is
  pending). Fires a DESKTOP notification + auto-analyze the INSTANT a coin enters the proven board;
  HOURLY digest. Stays diverse (re-screen each cycle, alert on board ENTRY, group de-dup, per-coin
  cooldown). Background lifecycle: --start/--stop/--status (single live-pidfile source of truth; --stop
  is SIGTERM→grace→SIGKILL so it can't orphan). It ALERTS; you confirm via roar/stage — never autonomous.
  stage    SYMBOL --spot|--usdm ...                       # checklist → plan + example → dry-run → CONFIRM
  manage   SYMBOL --spot|--usdm                            # manage an OPEN position: TP1→breakeven→trail (confirm each)
  review                                                   # expectancy / adherence / drawdown
  alert    SYMBOL --level ..

## CONFIG — FULL .env CONTROL (every parameter, for wide testability)
  EVERY field of EVERY config section is overridable from .env via a SYSTEMATIC
  `{SECTION}_{FIELD}` convention (type-coerced int/float/bool/str/tuple), so the whole
  tool is tunable without touching code — built for wide testability. Sections + prefixes:
    SCREEN_ (screener) · STRUCT_ (structure) · ANALYSIS_ · SETUP_ · MARKETS_ ·
    RISK_ (account equity, base risk %, the effective-risk pipeline knobs) · MGMT_
    (trade management: tp fraction, drawdown-scale, edge-scale, Kelly, max risk %) ·
    BACKTEST_ (incl. null_k / sig_z) · EDGE_ (floor, min_regime_n, sig_z, …) · GUARDS_
    (heat cap, daily-loss, max-positions, max-trades/day, cooldown) · JOURNAL_ · ALERTS_ ·
    WATCH_ · ORDER_ · OPTIMIZE_ · LIVE_.
  Friendly ALIASES kept for the common knobs: ACCOUNT_EQUITY, RISK_PCT (== RISK_RISK_PCT).
  The rules NEVER change — .env tunes thresholds; it cannot disable a safety rail (live
  arm flag, stop-or-bail, never-naked, typed CONFIRM). `.env.example` documents every knob
  grouped + commented; the active config is printed back so nothing is opaque.

## BUILD ORDER (test each before the next; backtest BEFORE any live wiring)
  1. data_fetch (2 TFs, real) + data-integrity guards
  2. indicators + regime + VOLATILITY REGIME (+tests)
  3. market_context: tide (BTC + ETH anchors, dominance) + influence groups
     (driver, RS/beta to leader) (+tests)
  4. screener (ranked shortlist) + extension/freshness flag + group column +
     worked-example formatter   ← cheap Tier-1 gate
  5. structure: ZigZag swings + market structure (HH/HL, BOS/CHoCH) + scored/pruned
     levels + retracement + volume profile + liquidity pools (+tests)
  6. markets (base/spot/usdm): instrument precision & limits, risk-first sizing,
     leverage↔margin↔liquidation (MMR tiers, max-lev cap, isolated), funding
     cost+signal, OI trend, fee schedule, market-agnostic interface (+tests)
  7. analysis: add RSI/MACD/OBV/CVD indicators; six scored lenses; MTF stack
     (bias→trigger, alignment); confluence-AND-conflict synthesis; location +
     concrete invalidation; divergence; lens-6 funding/OI/driver (+tests)
  8. setups: Setup contract + look-ahead-safe detect()→Signal (live≡backtest); entry
     type, pre-entry invalidation vs stop, time-stop; fixed robust params; long+short,
     regime-gated; returns all ranked; grade feeds Edge Score (+tests)
  9. risk & trade management: risk-first + post-rounding reconciliation; net R:R
     after fees+slippage+funding (can say NO TRADE); deterministic manage() state
     machine (TP1→BE→trail, stop-first/gap fills) shared w/ backtest; drawdown
     scaling; Kelly cap; net+blended worked example (+tests)
 10. backtest + expectancy: engine (same detect/simulate, fills by entry type,
     pending lifecycle, point-in-time regime); expectancy+CI / lower-bound verdict,
     multiple-testing, fold-consistency, drawdown dist / risk-of-ruin; EdgeProfile
     (by setup × regime) (+tests)   ← prove edge
 11. EDGE SCORE + OPPORTUNITY BOARD (`scan`): regime-matched conservative edge ×
     confidence × present-conditions; cached EdgeProfile (per coin×setup×tf,
     --refresh); empty board valid; board-level multiple-testing caveat (+tests)   ← headline UX
 12. guards + adherence + EVENT GUARD; GROUP-AWARE heat (+tests)
 13. INTRADAY RUNTIME: speed-tier config (--tier); THREAD-POOL concurrent fetch;
     continuous `watch` mode (interval re-scan + new/dropped flags); light backtest
     hot-loop opt + tier candle limits; data-source seam for later websockets (+tests)
 14. journal + review: JSONL trade log (lifecycle, dry-run/live); portfolio_state()
     feeds guards; equity curve; review = realised expectancy by setup/regime/grade
     + adherence + realised-vs-backtest + verdicts (+tests)
 15. alerts: level & setup-trigger alerts (persisted, one-shot); ping (console+bell);
     checked on demand + each watch pass; no auto-exec (+tests)
 16. order DRY-RUN (stage): state machine (guards→plan→re-fetch+drift/validity abort→
     typed CONFIRM); hard/soft override policy; duplicate protection; live-identical
     ticket; journal STAGED log; sends NOTHING (+tests)
 17. trade management (`manage`): SAME state machine as backtest simulate(); open-position
     state in journal (current_stop/remaining/tp1/broker-ids); propose-and-confirm;
     PAPER-FILL so the loop runs in dry-run; close → realised R → review/guards (+tests)
 18. last: --live — preflight (trade-only key, IP-lock, LIVE flag, TESTNET first);
     set-lev/iso → entry → STOP-OR-BAIL → TPs; partial fills; broker-id tracking;
     exchange reconcile; --flatten kill switch; second `CONFIRM LIVE` gate
  (+ optional any time after analysis: annotated chart images — charts.py)
  (+ deferred: SCALPING engine — separate data/signal/execution, shares risk/guards/journal)

## START HERE  (first reviewable milestone — the context-aware screener)
  Scaffold repo + config. Build data_fetch (real, 2 TFs, integrity guards),
  indicators (incl. volatility regime), market_context (BTC regime + relative
  strength + beta), the SCREENER that outputs a ranked, context-aware candidate
  shortlist from real Binance data, AND the worked-example formatter.
  Prove the user can run `screen`, read the table (with market-context banner,
  relative strength, vol-regime, and "why it passed"), see how the universe
  narrowed, and see one worked dollar example for the top candidate.
  Stop and show output before continuing.

## README (to ship with the tool)
  Philosophy; the top-down method (market→coin→level→setup→risk→edge); how the
  screener selects (and why majors dominate); spot vs usdm behavior; market
  context, INFLUENCE GROUPS & relative strength; the NAMED setups and structure-
  based stops; the EDGE SCORE & Opportunity Board (and what "proven on history,
  with confidence" means — not a prediction); how to read the worked examples;
  honest backtest caveats (walk-forward, slippage, survivorship); trade-only /
  no-withdrawal / IP-locked / gitignored API key setup; disclaimer.

## DISCLAIMER
  For education and analysis only. NOT financial advice. Does not predict price
  and cannot guarantee profit. Trading crypto — especially leveraged USD-M
  futures — can lose your entire balance. The user owns every decision.
