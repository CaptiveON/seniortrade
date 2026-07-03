# Crypto Trading SYSTEM — Screener → Analysis → Edge → Staged Order

A Binance (SPOT + USD-M) trading **analysis and risk-management** CLI. It finds
clean setups, proves edge on real history, sizes by risk, enforces discipline,
and **teaches you at every step**.

> **It does NOT predict price.** Technical analysis describes market structure;
> it does not forecast it. "NO TRADE" is always a valid choice. **Not financial
> advice.**

## Philosophy

> Success = a measured edge, executed with discipline, over many trades.

No single trade matters. The tool exists to keep you trading a real, measured
edge with consistent risk — and to make every number mean something in dollars,
not just appear on a screen.

## Build status

This is built strictly in the brief's **build order**, proving each step before
the next. **The full pipeline is built** — the context-aware screener, six-lens
analysis, the edge engine + Opportunity Board, guards, journal, alerts, dry-run
order staging, managing an open position by the plan (paper-filled on real
prices), and — dead last, behind a hard env gate + a second `CONFIRM LIVE` —
real-money execution (`--live`) with a TESTNET-first path and a `--flatten` kill
switch. **It still places nothing without the explicit arm flag and your typed
confirmation.**

```
1. data_fetch (2 TFs, real) + integrity guards        ✅ done
2. indicators (ATR/ADX/RSI/MACD/OBV/CVD, regimes)     ✅ done
3. market_context (BTC+ETH anchors, influence groups) ✅ done
4. screener (context-aware shortlist, freshness, groups) ✅ done  ← gates everything
5. structure (ZigZag, HH/HL+BOS/CHoCH, levels, VP, liquidity) ✅ done
6. markets (precision, MMR-tier liquidation, funding/OI, fees) ✅ done
7. analysis (six scored lenses, MTF, confluence+conflict)  ✅ done  ← `analyze`
8. setups (look-ahead-safe detect→Signal, 10 setups / 4 mechanisms, graded) ✅ done  ← shown in `analyze`
9. risk & sizing (net-cost plan, shared walk_management state machine) ✅ done  ← in `analyze`
10. backtest + expectancy (same code as live; CI/verdict/MC; EdgeProfile) ✅ done  ← `backtest`
    └─ NULL BASELINE: "proven" means the edge beats matched-geometry RANDOM entries
       (subtracting the exit "free option" + directional drift), not merely beats zero.
11. Edge Score + Opportunity Board (cached, regime-matched, conservative) ✅ done  ← `scan`
    └─ HIERARCHICAL SHRINKAGE: each coin's edge = its own setup×regime trades shrunk
       (empirical-Bayes) toward the pooled prior — borrow strength, never trust a noisy mean.
    └─ CONTEXTUAL STRATEGIES: scores on a proven context refinement (vol/momentum/location/
       divergence/HTF-align) when one earns it (money + Bonferroni + OOS), else the base setup×regime.
    └─ FEATURE INTERACTIONS: also tests feature PAIRS that beat either alone (greedy default,
       `scan --hard` = exhaustive); a real higher-order state, gated like singles. (P4)
    └─ CONTINUOUS FEATURE EVALUATION: `features` ranks context features by predictive value +
       rising/decaying trend (persisted each rebuild); the top-4 seed the P4 pair search. (P13)
    └─ REGIME ARCHETYPES: every bar labelled (panic/euphoria/accumulation/distribution/vol-exp/
       vol-comp/trending/range); analyze shows the setup's conditional expectancy in the live
       archetype; it changes a verdict only through the same gate. (ENHANCEMENTBRIEF P3 + P14)
    └─ PRO REPORTING: backtest shows raw edge→costs→NET (P10), distribution median/skew/percentiles
       + bootstrap CI (P9), drawdown distribution / risk-of-ruin / expected streak (P11); analyze
       prints an auditable "why — for & against" panel (P8). `backtest --full` for the deep view.
    └─ STATISTICAL CONFIDENCE: confidence = P(edge beats random entries) × fold-consistency
       (P2) — "how certain the edge exists," with P(expectancy>0) shown alongside.
    └─ TIME-WEIGHTED LEARNING: opt-in exponential decay (half-life) so recent trades weigh
       more — the edge adapts as markets evolve; older data fades, never dropped. (P7)
    └─ VALIDATION DEPTH: a robustness read (OOS folds × bootstrap × walk-forward) distinguishes
       profitable from CONSISTENTLY profitable; report-only, shown in backtest/analyze. (P6)
    └─ CORRELATION HONESTY: cross-coin correlation is MEASURED (design effect) — SEs widened,
       gates use effective independent n, not raw pooled n. (audit finding 2)
    └─ CANDLE SANITATION: broken rows dropped at the fetch choke point; gaps/zero-volume/
       suspect prints flagged, never "fixed" — bad tapes can't contaminate the pool. (finding 6)
    └─ TWO-LANE BOARD: ALPHA (proven timing edge) kept pure; BETA = PROVEN market posture,
       held to the same sig_z bar on its own hypothesis (tide significantly +EV × vehicle cell
       significantly +EV + OOS) — fluke-proof on noise like alpha, honest refusal otherwise;
       near-misses shown as labelled hypotheses. Discretionary, never roar-struck.
12. guards + adherence (pre-trade gate: heat, lockout, cooldown, event)  ✅ done
13. intraday runtime (speed tiers, thread-pool fetch, `watch`, bt opt)   ✅ done  ← `watch`, `--tier`
14. journal + review (trade log, portfolio_state→guards, verdicts)       ✅ done  ← `review`
15. alerts (level & setup, one-shot, console+bell; no auto-exec)         ✅ done  ← `alert`
16. order DRY-RUN (`stage`): guards→plan→re-fetch→CONFIRM, sends nothing ✅ done  ← `stage`
17. trade-mgmt (`manage`): SAME machine as backtest; paper-fill; close→R    ✅ done  ← `manage`
18. --live: preflight + arm-flag, TESTNET-first, stop-or-bail, reconcile, --flatten ✅ done  ← `stage/manage --live`
```

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.11+. The screener uses **public** Binance data — **no API key needed.**

## Run it

```bash
python -m src.cli roar    --usdm                # MASTER: proven board → allocate budget across the basket → CONFIRM
python -m src.cli watch   --usdm --start        # always-on radar: desktop-notify on a new proven edge (--stop/--status)
python -m src.cli scan    --usdm                # Opportunity Board: edge-ranked, proven setups (the headline)
python -m src.cli scan    --usdm --tier intraday  # same, on the 4h/1h intraday tier
python -m src.cli watch   --usdm --interval 120 # continuous board — re-scan every 2 min, flag new/dropped
python -m src.cli context --spot                # the tide: BTC regime + leaders/laggards
python -m src.cli screen  --spot                # long-only ranked shortlist
python -m src.cli screen  --usdm                # USD-M perpetual futures
python -m src.cli analyze  BTC/USDT:USDT --usdm --tf 4h  # deep six-lens read of one coin
python -m src.cli analyze  --all --usdm                 # board: same read across the whole shortlist (setups-first)
python -m src.cli backtest BTC/USDT:USDT --usdm --tf 4h  # prove (or reject) each setup's edge
python -m src.cli optimize --usdm                       # OOS-validated config tuning → proposed diff (offline, slow)
python -m src.cli alert BTC/USDT:USDT --usdm --level 70000  # ping when price reaches a level
python -m src.cli review                                # realised performance + discipline + verdicts
python -m src.cli stage SOL/USDT:USDT --usdm            # guards→plan→re-fetch→CONFIRM (DRY-RUN; sends NOTHING)
python -m src.cli manage --usdm                         # manage open positions by the plan; paper-fills staged (DRY-RUN)
python -m src.cli manage SOL/USDT:USDT --usdm           # manage just one: TP1→breakeven→close, each step CONFIRMed

# --- real money (only after TESTNET + arming; see "Going live" below) ---
python -m src.cli stage  SOL/USDT:USDT --usdm --testnet # full gate → place on the Binance TESTNET sandbox
python -m src.cli stage  SOL/USDT:USDT --usdm --live    # full gate → preflight → CONFIRM LIVE → REAL order
python -m src.cli manage --usdm --live                  # reconcile with the exchange, then manage real positions
python -m src.cli manage --usdm --live --flatten        # KILL SWITCH: close everything + cancel all orders
```

Read the **market-context banner** (what BTC is doing) first, then the ranked
table — and **you pick** the coin. The rest of the tool (once built) runs only on
your choice.

### How the screener selects (and why majors dominate)

Filters are applied in priority order; the CLI prints the exact gates first.

1. **Liquidity (gatekeeper).** Keep the top-N by 24h quote volume, above a
   volume floor, with a tight top-of-book spread and a thick resting book near
   mid. Illiquid coins are rejected first — you can't get a fair fill or a real
   stop in them.
2. **History.** Require enough closed candles to backtest later. New listings
   are rejected: no history, no provable edge.
3. **Volatility regime & extension.** ATR% must sit inside a sane band, AND its
   *regime* is read from the ATR% percentile vs the coin's own history — `squeeze`
   (coiling), `normal`, or `expanding`. It also measures **extension** — how far
   price is from its reference MA in ATRs (the `Ext` column) — and flags a coin
   that has run too far as `extended`/late, so rank isn't fooled by a parabolic move.
4. **Regime clarity.** Classify trending / ranging / choppy (ADX). Choppy coins
   are listed but **flagged "avoid."**
5. **Correlation.** Candidates that move together share a cluster label and are
   flagged (`*`) so you don't mistake five correlated longs for diversification.
6. **Market context & influence groups.** Coins are measured against macro
   anchors **BTC and ETH** and grouped with the leader they track most (the
   `Group` column — `BTC`, `ETH`, an alt leader, or `indep`). Coins in one group
   move together, so the tool warns when several shortlisted coins are really
   **one bet** (`*`). In a risk-off BTC, spot longs that actually track a falling
   BTC are penalised/flagged — but a *decoupled* leader (low correlation,
   outperforming) is not, because it's defying the tide, not fighting it.

**Ranking is a transparent composite** — trend clarity + relative strength +
volatility regime + **freshness** (penalising extended/late price) + liquidity —
**not ADX alone.** The CLI prints a **market-context banner** (what BTC is doing)
above the table, and the per-row "why it passed" restates the measured facts.

**Majors (BTC/ETH/top alts) dominating the shortlist is correct, not a bug** —
they are the most liquid, deepest, longest-history markets, so they clear the
liquidity and history gates that thinner coins fail. (When BTC is risk-off and
strong alts decouple upward, those leaders can rank above the majors.)

The output also shows **how the universe narrowed** (how many were rejected at
each stage), so the shortlist is never a black box.

## SPOT vs USD-M

| | `--spot` | `--usdm` |
|---|---|---|
| Direction | long-only | long **and** short |
| Instruments | spot pairs (e.g. `BTC/USDT`) | linear perpetual swaps (e.g. `BTC/USDT:USDT`) |
| Worked example | always long | follows the measured trend (short if down) |
| Later layers add | — | leverage, margin, funding, liquidation-distance checks |

Stable/stable pairs, fiat pairs, and Binance leveraged tokens (`…UP/…DOWN`,
`BULL/BEAR`) are excluded from both.

## How to read the worked examples

Every stage ends with a plain-language dollar example so you always know what a
number means **for your money**. The screener shows one for the top candidate,
sized **risk-first**:

```
With $1,000.00 account and 1% risk: you risk $10.00 on this trade.
Entry 0.65290, stop 0.53115 → if stopped you lose ~$10.00 (your 1%).
Take-profit 0.83553 → if hit you make ~$15.00 (1.50R).
Ideal position size: 82.1329 WLD (notional $53.62).
Worst case (stop): -$10.00.  Planned best case (TP): +$15.00.  R:R 1:1.50.
```

Read it in this order:

1. **Risk first.** You decide the dollars at risk (equity × risk%) *before*
   anything else — here, $10.
2. **The stop sets the size.** Position size is an *output*: `$ at risk ÷
   (entry − stop)`. A wider stop ⇒ a smaller position, never more risk. (This is
   why a high-volatility coin like WLD gets a tiny notional — the wide
   ATR-based stop shrinks the size to keep the loss at $10.)
3. **R is the unit.** Reward is measured in multiples of the amount risked (R),
   so trades are comparable regardless of price.

The screener's example is **illustrative only** (stop from ATR, TP at a default
R:R) — it demonstrates the mechanics, it is **not** a trade signal.

## Configuration & risk control (`.env`)

**Every** config field is overridable from `.env` via a `{SECTION}_{FIELD}`
convention (type-coerced), so the whole tool is tunable for wide testing without
touching code. Copy `.env.example` (fully commented) and set what you need. A few
of the most useful:

```bash
ACCOUNT_EQUITY=5000
RISK_PCT=2                      # BASE % risked per trade (size is derived from the stop)
GUARDS_MAX_TRADES_PER_DAY=3    # over-trading guard (0 = unlimited)
GUARDS_HEAT_CAP_PCT=6          # max TOTAL open risk as % equity
```

### Risking more than the default 1% — the effective-risk pipeline

`RISK_PCT` is the **base**. Before sizing, it passes through an opt-in pipeline
(`base% → ×drawdown-scale → ×edge-scale → ×vol-target(σ_R) → Kelly cap → clamp to a hard ceiling`),
so the size you get is the size the **rules** permit, not just the number you typed.
`analyze` prints a **position-sizing guidance** panel (P12) — the evidence (edge, confidence,
sample-quality, σ_R) and which stages fired; correlation/exposure are handled at the basket level (`roar`):

| `.env` | Effect |
|---|---|
| `MGMT_MAX_RISK_PCT=5` | hard ceiling the pipeline can never exceed (default 5%) |
| `MGMT_DD_SCALE_ENABLED=true` | auto-reduce risk as the account draws down |
| `MGMT_EDGE_SCALED=true` | size from the **edge × confidence** — more on proven setups, less on marginal ones |
| `MGMT_VOL_TARGET_ENABLED=true` | size from **variance** — a wilder setup (high σ_R) is sized smaller (equal dollar-volatility) |
| `MGMT_KELLY_ENABLED=true` | fractional-Kelly cap (only ever *reduces* risk) |

Every stage that fires is printed (e.g. `base 8% → ceiling 5%`, or `drawdown 20% →
×0.50 = 1.50%`). With all stages off (the default) it is exactly the flat base %.
Raising risk is non-linear in pain — a 10-loss streak costs ~10% at 1%, ~26% at 3%,
~40% at 5% — so size from the edge, not the appetite. Safety rails (the `--live`
arm flag and confirm phrase) are **never** `.env`-overridable.

## API key setup (only needed much later, for order staging)

The screener, analysis, and backtest layers never need a key. When you reach the
order layer, create a Binance API key that is:

- **Trade-only** — enable Reading + Spot/Futures Trading; **Withdrawals OFF.**
- **IP-locked** — restrict to your IP address.
- **Dedicated** — ideally a sub-account/key used only by this tool.

Put it in `.env` (copy from `.env.example`). **`.env` is gitignored — never
commit keys.** Even then, order placement defaults to **dry-run** and requires a
typed `CONFIRM` with a fresh data re-fetch before anything goes live.

## Going live (`--live`) — the slow, deliberate path

Real-money execution is built **dead last** and is **off until you explicitly
arm it.** The full sequence, in order:

1. **Paper-trade first.** Run `stage` + `manage` in the default dry-run; trades
   are paper-filled on real prices and feed `review`. Confirm the loop and your
   discipline before risking a cent.
2. **TESTNET.** Run `stage --testnet` / `manage --testnet`. This routes through
   Binance's sandbox (`set_sandbox_mode`) so you validate the *entire send path*
   — leverage/margin, entry, the protective stop, TPs — with fake funds.
3. **Arm real money.** Export the opt-in flag (kept out of `.env` on purpose):
   ```bash
   export LIVE_TRADING_ENABLED=I_UNDERSTAND_THE_RISK
   ```
   Without it, `--live` runs preflight and **refuses to send.**
4. **Place.** `stage SYMBOL --usdm --live` runs the full dry-run gate, then a
   **preflight** (key present, arm flag, withdrawals-disabled check), then a
   **second** re-fetch + drift/validity check, then a distinct **`CONFIRM LIVE`**
   prompt — and only then sends. On fill it **immediately places the protective
   stop**; if the stop can't be placed it **emergency-closes the position** (never
   naked). Partial fills size the stop to what actually filled.
5. **Manage & reconcile.** `manage --live` treats the **exchange as the source of
   truth** — it reconciles open positions/orders first (catching a stop-out that
   happened while you were away), then proposes plan actions you `CONFIRM LIVE`.
6. **Kill switch.** `manage --live --flatten` market-closes every position and
   cancels every order, immediately.

## Safety (non-negotiable)

- **No autonomous trading.** Type-to-confirm, re-fetch before send, dry-run by
  default; `--live` is behind an env arm-flag **and** a distinct `CONFIRM LIVE`.
- **Never a naked position.** Live entries arm their stop on fill; if the stop
  fails, the position is emergency-closed. Test on **TESTNET first.**
- **Exchange is the source of truth.** `manage --live` reconciles before acting;
  `--flatten` is the kill switch.
- **Re-validated at send time.** `roar --live` re-checks every basket leg on fresh
  data right before its order (edge still proven in the current regime, no drift,
  invalidation intact) and **skips** any that went stale — plus a margin preflight so
  an unfundable leg is skipped cleanly, never a crash.
- **No predictive claims.** Outputs describe structure; `NO TRADE` is valid.
- **Risk-first sizing.** Risk the percentage first; derive size from the stop.
- **Keys** are trade-only, no-withdrawal, IP-locked, and gitignored; the secret
  is never logged.

## Tests

```bash
pytest
```

Covers the native indicators, the risk-first worked-example math, and the
screener's filtering/regime/clustering logic (all offline, no network).

## Disclaimer

This software is for education and analysis. It is **not financial advice**, does
not predict prices, and cannot guarantee profits. Trading crypto — especially
leveraged USD-M futures — can lose your entire balance. You are responsible for
your own decisions.
