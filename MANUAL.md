# TradingAIAssistant — Full Operating Manual

A complete, edge-to-edge command reference. Every flag below is exact (pulled from
the tool). Default behaviour is **dry-run** — nothing is sent to an exchange until
you explicitly arm `--live`.

> It does NOT predict price. **NO TRADE** (and an empty board) is always a valid,
> common answer — that's the discipline working, not a failure. Not financial advice.

---

## 0. Setup (one-time — already done on this machine)

```bash
cd /Users/captiveon/Documents/techproject/experimental/thmillion/practical/TradingAIAssistant
```

- Dependencies live in your system `python3`. Every command is `python3 -m src.cli <command> …`. No venv to activate.
- `.env` holds `BINANCE_API_KEY` / `BINANCE_API_SECRET` (trade-only, no-withdrawal, IP-locked). Public data (steps 1–6) needs **no** key.
- Run in a **real terminal** — `stage`/`manage` prompt for a typed `CONFIRM`.
- Tip: prefix with `COLUMNS=160` so wide tables don't wrap:
  `COLUMNS=160 python3 -m src.cli screen --usdm`

## Global flags (most commands accept these)

| Flag | Meaning |
|---|---|
| `--spot` | SPOT market, long-only, symbols like `BTC/USDT` (default if neither given) |
| `--usdm` | USD-M perps, long **and** short, symbols like `BTC/USDT:USDT` |
| `--tier swing` | timeframes 1d/4h |
| `--tier intraday` | timeframes 4h/1h |

---

## 1. `context` — read the tide (always first)

```bash
python3 -m src.cli context [--spot|--usdm] [--tier swing|intraday] [--top N] [--quote USDT] [--tf TF] [--min-volume FLOOR]
```
```bash
python3 -m src.cli context --usdm
python3 -m src.cli context --usdm --top 40
```
**Read:** the banner (BTC regime + RISK-ON/OFF) and the leaders/laggards table. Sets your directional bias.

## 2. `screen` — the candidate shortlist

```bash
python3 -m src.cli screen [--spot|--usdm] [--tier …] [--top N] [--quote USDT] [--tf TF] [--min-volume FLOOR] [--equity $] [--risk %]
```
```bash
python3 -m src.cli screen --usdm
python3 -m src.cli screen --usdm --top 20 --equity 2000 --risk 0.5
python3 -m src.cli screen --spot --tf 4h
```
**Read:** ranked table (trend, ADX, RS·BTC, β, regime, group, "why it passed") + how the universe narrowed. **You pick** the coin. By default it deep-checks the **top 100** by 24h volume (~40 typically pass); `--top N` narrows it.

## 3. `scan` — the Opportunity Board (the headline)

```bash
python3 -m src.cli scan [--spot|--usdm] [--tier …] [--top N] [--refresh]
```
```bash
python3 -m src.cli scan --usdm                  # survivors ranked by PROVEN edge (slow cold; cached after)
python3 -m src.cli scan --usdm --refresh        # ignore cache, re-backtest
python3 -m src.cli scan --usdm --tier intraday  # on the 4h/1h tier
```
**Read:** only setups with proven edge appear. **An empty board is valid** = no proven edge right now. By default `scan` now deep-scans **all survivors** from the top-100 screen (not a fixed 8), concurrently — wider net, but a cold/full scan takes ~1–2 min. `--top N` caps it if you want it faster.

> **STATISTICAL CONFIDENCE (P2):** the `Confidence` column is now evidence-driven — it answers *"how certain are we this edge actually exists?"*, **not** "how many signals agree." It is **P(edge genuinely beats random entries) × fold-consistency** (parametric, from the cached CI — so it folds in sample size, variance, and the confidence interval in one number). `analyze` shows it broken out: e.g. *confidence 59% [P(beats null) 88% · P(>0) 96%]* — a setup can be 96% likely +EV yet only 88% likely to beat **random**, and fold-inconsistency tempers it further. Higher confidence ⇒ the edge is more certain to be real (and, with `MGMT_EDGE_SCALED`, sized larger).

> **CONTEXTUAL STRATEGIES + INTERACTIONS (P1 / P4):** the board can score a coin's setup on a **proven context refinement** — a single feature (`trend_pullback ·vol=expanding`) **or a feature INTERACTION** (`breakout_momentum ·vol=expanding & mom=bear`) — when the coin's *current* context matches a cell that **earned it**: makes money AND beats the rest with **Bonferroni** significance AND is **OOS fold-consistent**; a **pair must also beat the better of its two single features** (genuine synergy, not inherited edge). Pair search is **greedy** by default (only combos built on a promising single — avoids combinatorial explosion); **`scan --hard`** tests every pair exhaustively (far stricter bar). The matched cell is shrunk toward its setup×regime parent (P5). No proven refinement / no match → plain setup×regime edge. A `·label` tag means a refinement scored it. *(On the current universe none clears the strict bar — a marginal greedy pair didn't survive a re-screen or `--hard`; the wiring is live and auto-activates when a stable one earns it.)*
>
> **REGIME ARCHETYPES + conditional expectancy (P14 + P3):** every bar is labelled with a named market **archetype** — `panic / euphoria / accumulation / distribution / vol_expansion / vol_compression / trending_up / trending_down / range` — derived from the structural trend × volatility × thrust the engine already computes. `analyze` names the coin's **current** archetype and shows that setup's **historical conditional expectancy** there (e.g. *"breakout_retest historically +0.42R in panic vs −0.08R pooled"*) — evidence even when it isn't strong enough to change sizing. An archetype only **changes a live verdict** through the *same* gate as any context refinement (money + Bonferroni + OOS); when it earns it, the board row shows the archetype as its `·label`. So the archetype answers *"does this setup have an edge **only** in this environment?"* without ever lowering the edge bar.
>
> **Per-coin edge via HIERARCHICAL SHRINKAGE (P5):** each coin's edge is its OWN setup×regime trades **shrunk toward** the pooled prior (empirical-Bayes) — a coin with no/thin history gets the universe number; a coin with ≥5 own trades that genuinely differs pulls toward its own. Loud small-sample means are damped (a coin needs real evidence before it's trusted over the pool), so this *reduces* overfitting while personalising. Transparent in the cache (own n/mean, τ², shrink weight).
>
> **Edge is proven on trades POOLED across the liquid universe** (per setup × regime), not on one coin's thin history — so the verdicts are real (hundreds of trades), not "too few trades." An empty board therefore means *"no proven edge in the current regime,"* not *"not enough data."* The board ranks by the **conservative edge NET OF THE NULL BASELINE** (see `backtest` — the edge must beat matched-geometry random entries, not just zero), so a setup riding only the exit artifact or directional drift never reaches the board. The pooled profile is cached; **`scan --refresh` rebuilds it** (do this after upgrading — older caches have no null and fall back to the zero-baseline until refreshed).

## 3b. `features` — which context features carry predictive value (P13)

```bash
python3 -m src.cli features [--spot|--usdm] [--tier …] [--tf TF]
```
**Read:** a ranked table of the context features (vol / momentum / location / divergence / HTF-align / archetype) by **importance** (# raw-significant positive directions + mean lift), with **best lift_low**, **# proven**, and a **trend** (▲ rising / ▼ decaying / · flat / ＊ new) vs the previous rebuild. A `●` marks the **top-4** important features — the only ones allowed to seed **P4 interaction pairs** (important features carry greater influence; decayed ones decline). Importance is recomputed and a snapshot persisted every **`scan --refresh`** (history in `data/feature_importance_history.json`), so running `features` periodically shows how each feature's predictive value **evolves** as markets change. *(Live: `vol` is the most predictive feature; `htf_align`/`div` sit lowest and don't seed pairs.)*

## 4. `watch` — always-on radar (background, desktop notifications)

```bash
python3 -m src.cli watch --usdm --start          # run in the BACKGROUND (notifications + hourly digest)
python3 -m src.cli watch --usdm --status         # what it's monitoring (coins, board, watchlist)
python3 -m src.cli watch --usdm --stop           # stop it cleanly
python3 -m src.cli watch --usdm --interval 600   # foreground, 10-min polls (Ctrl-C to stop)
```
**What it does (two-cadence, bar-close-aware):** the board can only change when a **trigger-TF candle closes** (4h, or 1h on `--tier intraday`), so watch **re-scans right after each close** — not on a blind timer (no pointless re-scans of a board that can't change). The instant a coin **enters the proven board** it fires a **desktop notification** and **auto-runs the deep `analyze`** so the read's ready — then you `roar`/`stage` (it never auto-trades). **Between** closes it only does **light, price-driven checks** (your `alert`s / open positions) every ~60s, *if* there's anything pending. **Hourly** it sends a status digest. Stays diverse — re-screens the universe each cycle, alerts on board *entry* (not every pass), group-de-dups, per-coin cooldown. Full control: `--start`/`--stop`/`--status` (pidfile + log under `data/`); **`Ctrl+C`** stops a foreground run cleanly (`Ctrl+Z` only suspends — but `--stop` will still kill it).

## 4b. `roar` — MASTER command: strike the whole proven basket

```bash
python3 -m src.cli roar --usdm                  # tide → proven board → allocate budget → CONFIRM (dry-run)
python3 -m src.cli roar --usdm --tilt           # edge-weight the allocation (default is equal-risk)
python3 -m src.cli roar --usdm --live           # place the basket for REAL (preflight + CONFIRM LIVE + arm flag)
python3 -m src.cli roar --usdm --testnet        # rehearse the basket on the sandbox
```
**What it does:** one command runs the whole pipeline and acts on **every** opportunity that clears all gates (proven edge + valid R:R + guards). It **divides your risk budget across the basket like a desk** — equal-risk per uncorrelated bet by default (`--tilt` = edge-weighted), correlated coins share a budget, capped by total **heat** / **max positions** / **per-group**. Shows the basket (per-trade risk $ / size, total portfolio heat, groups) → one typed `CONFIRM` stages them all (dry-run). With `--live` + the arm flag → `CONFIRM LIVE` → each placed **independently stop-or-bail**. **Inherits every gate** — an empty board means "nothing to ROAR." `watch` pings you when it's worth running.

> **Re-validation at send time (live):** the board is built once, but `--live` places *after* you type `CONFIRM LIVE` — a window in which a coin's **regime can flip** (a setup proven in `range` goes −EV in `down`), price can **drift**, or the pre-entry **invalidation** can hit. So `roar --live` **re-validates each leg on fresh data right before its send** — re-analyze → the same setup must still be present and still **PROVEN in the current regime** → drift + invalidation must pass → re-size on the fresh price — and **SKIPS** (reports) any leg that no longer qualifies. A `roar` basket is a *candidate list*, not a signal; the send-time gate is the arbiter (same discipline as `stage`). Also: a **margin preflight** + a basket margin overview mean an unfundable leg is skipped cleanly, never a crash.

## 5. `analyze` — deep read of one coin, OR the whole shortlist at once

```bash
python3 -m src.cli analyze SYMBOL [--spot|--usdm] [--tier …] [--tf TF]              # one coin, deep
python3 -m src.cli analyze --all  [--spot|--usdm] [--tier …] [--tf TF] [--top N] [--setups-only]   # the shortlist, as a board
```
```bash
python3 -m src.cli analyze WLD/USDT:USDT --usdm --tf 4h      # single deep read
python3 -m src.cli analyze BTC/USDT --spot --tf 1d
python3 -m src.cli analyze --all --usdm                       # board across the top-30 shortlist
python3 -m src.cli analyze --all --usdm --top 15 --setups-only  # only the coins with a live setup
```
**Single (`analyze SYMBOL`):** six scored lenses, structure, conflict/bear-case, the named setup (or NO TRADE), the risk-first dollar plan, and the concrete invalidation. When a setup is present it also prints: the **Proven edge** verdict, the **Conditional expectancy** for the current archetype (P3/P14), and a **Why — evidence for & against** panel (P8) — every factor pushing *for* vs *against* the trade (the aligned lenses, proven/−EV edge, earned context cell, freshness, tide, statistical confidence), decisive contributor first. The recommendation is **auditable**: you can see exactly *why* it's good or bad.

**Board (`analyze --all`):** runs the screener's top-N, analyzes each with the *same* engine concurrently, and prints **one compact row per coin** — `Symbol · Bias (conf%) · Loc · HTF align · Setup (dir/grade) · net R:R · key flag`. Sorted so coins with a **valid plan come first** (best grade, then best net R:R), then setup-without-plan, then NO-TRADE (dimmed). `--top N` widens/narrows how many shortlisted coins to read (default 100); `--setups-only` hides the NO-TRADE rows. Deep-dive any row with `analyze SYMBOL`. *(This is `scan` without the backtest/edge gate — fast, shows current setups across the whole shortlist.)*

### Setup vocabulary (what `analyze`/`backtest`/`scan` can recognize)

The tool checks each coin against a library spanning the **four ways an edge arises** —
so a NO TRADE means "checked against the full playbook, nothing qualifies," not blindness:

- **Continuation:** `trend_pullback` (dip to MA/level, resume) · `momentum_flag` (impulse → tight coil → break)
- **Breakout/expansion:** `breakout_retest` (enter the retest) · `breakout_momentum` (enter ON the break, no retest — catches moves that run; ADX-gated) · `squeeze_breakout` (a low-volatility coil that releases — the ignition of a move)
- **Mean-reversion:** `range_fade` (fade a clean range edge; range regime only)
- **Reversal/trap:** `failed_breakout` (spring/upthrust at a range edge) · `liquidity_sweep_reversal` (wick beyond an equal-highs/lows pool, then reclaim) · `choch_reversal` (the prior trend's structure breaks) · `divergence_reversal` (RSI divergence at a structural extreme)

Each is long + mirrored short (USD-M), structure-anchored stop, regime-gated, and **must independently prove edge in `backtest` before it's trusted**. Recognition is wide; the edge bar doesn't move.

## 6. `backtest` — prove the edge (the gate)

```bash
python3 -m src.cli backtest SYMBOL [--spot|--usdm] [--tier …] [--tf TF] [--setup NAME]
```
```bash
python3 -m src.cli backtest WLD/USDT:USDT --usdm --tf 4h
python3 -m src.cli backtest WLD/USDT:USDT --usdm --setup breakout_retest
python3 -m src.cli backtest WLD/USDT:USDT --usdm --tf 4h --full   # full research view
```
**Read:** per-setup verdict (PROVEN / INCONCLUSIVE / −EV / TOO FEW), expectancy + CI, the **vs Null [low]** column, **by-regime** breakdown, Monte-Carlo, caveats. Trust this over the chart.

> **Recommended research posture:** `BACKTEST_CANDLE_LIMIT=2000` **+** `BACKTEST_TIME_DECAY_ENABLED=true`, then `scan --refresh`. Deep history gives each setup×regime cell real evidence; time-decay keeps the old regimes from dominating the estimate. (Shipped defaults stay 1000/off — this posture is opt-in by owner choice.)
>
> **Deeper history — `BACKTEST_CANDLE_LIMIT` (any size).** The fetcher now **paginates** past the exchange's ~1000-candles-per-request cap, so `BACKTEST_CANDLE_LIMIT=2000` (or 1500, 3000…) really delivers that many (≈333 days at 4h for 2000) → **more trades per setup, more robust verdicts**. The header shows the **honest** count actually used (e.g. *"1999 candles (of 2000 requested)"* — the −1 is the still-forming candle). Deeper history reaches into **older regimes**, so pair it with `BACKTEST_TIME_DECAY_ENABLED=true` (P7) to down-weight the stale tail, and **`scan --refresh`** to rebuild the pooled cache on the deeper window.
>
> **Combining with a speed tier (e.g. 1h @ 2000 candles).** Env overrides now **layer on top of** a `--tier` preset (precedence: defaults → tier → env), so `scan --usdm --tier intraday` **and** `BACKTEST_CANDLE_LIMIT=2000` gives you the tier's **1h** trigger (+ 4h bias) *with* your **2000** candles — the tier no longer clobbers an explicit env value. Any tier field is env-overridable this way (`EDGE_TF`, `ANALYSIS_TF_BIAS`, `SCREEN_TF`, …). The 1h board builds a **separate** pooled cache (`{market}_1h_…`) from the 4h one, so `scan --refresh` once on the new tf.

> **The per-setup panel is now a quant research sheet** (ENHANCEMENTBRIEF P9/P10/P11):
> - **Edge vs execution (P10):** `gross −fees/slippage −funding = NET` — see exactly how much raw edge the costs eat (e.g. live `breakout_retest` gross +0.37R → NET +0.22R).
> - **Distribution (P9):** `median · avg win / avg loss · σ · skew · p10 / p90` — what the average hides (a +0.22R mean can sit on a −1.1R *median* with skew +2.8: a few big winners carry it).
> - **Risk (P11):** Monte-Carlo `median DD · 95th-pctile DD · risk-of-ruin · expected losing streak · recovery` — the drawdown *distribution*, not just the historical max; *"what should I realistically expect?"*
> - **`--full`:** adds the non-parametric **bootstrap 95% CI** (robust to skew), **recent (last-third) vs overall** expectancy (time-stability), and the **per-fold** expectancies.
> - **Statistical assumptions (stated, not hidden):** the Monte-Carlo uses a **circular block bootstrap** (≈√n blocks) so loss streaks and clustered volatility survive into the drawdown tails (`BACKTEST_MC_BLOCK=1` restores the legacy i.i.d. permutation — its tails are optimistic). Still i.i.d.-based and documented as such: the expectancy **bootstrap CI** (resamples trades independently) and the **null-excess SE** (shadow trades overlap the same tapes — mitigated by the shared design-effect widening). Probabilities like P(beats null) use a **normal approximation** on skewed R distributions — trustworthy at pooled sample sizes, softer in thin regime cells. **Universe caveats:** coins are selected by **today's** liquidity (point-in-time selection bias) and Binance lists **survivors** only — both flattering; read verdicts as *"edge on currently-liquid survivors."*
> - **Candle sanitation:** every fetched tape is screened at the source — provably-broken rows (NaN/≤0 prices, high<low, wick-inconsistent) are **dropped**; gaps, zero-volume bars and bad-print-shaped spikes are **flagged, never "fixed"** (a real crash must stay data). Issues appear as a yellow `data quality:` line in the backtest header; a fully-broken tape errors out instead of contaminating the pool.
> - **Correlation honesty (effective n):** pooled coins are cross-correlated (they ride the same market day), so each setup shows `correlation: ρ … → design effect … → effective n ≈ X of N` — the **measured** number of *independent* observations. All CIs are √DEFF-widened and the proven/unproven gates use **effective n**, not raw pooled n. (Live: ρ 0.06–0.66 by setup; SEs widened ×1.0–1.2.)
> - **Robustness (P6 — validation depth):** a `robustness 0.00–1.00` line per setup = **OOS folds × bootstrap stability × walk-forward non-decay**, labelled *consistently profitable / profitable but not robust / not an edge*. It distinguishes **profitable from CONSISTENTLY profitable** — e.g. a setup that's +1.0R overall but −1.2R *recently* scores ~0 (walk-forward collapse). Report-only; `analyze` shows it on the proven-edge panel too.
>
> **TIME-WEIGHTED LEARNING (P7):** markets evolve, so you can make recent trades count more — set `BACKTEST_TIME_DECAY_ENABLED=true` (+ `BACKTEST_TIME_DECAY_HALF_LIFE_DAYS=90`). The pooled expectancy/CI/confidence/sizing then use an exponential decay weight (a trade N days old counts half), so the **edge adapts** as conditions change; older data fades gradually, never dropped. The SE widens as fewer trades carry the weight (honest). The null baseline stays unweighted (it's a control). **Default off** → unweighted, exactly as before; `scan --refresh` bakes the weighting into the cache.

> **The bar is NOT zero — it's the NULL BASELINE.** A scale-out + breakeven exit earns *positive* R even on no-edge entries (the "free option" artifact), so "expectancy > 0" alone can stamp junk as proven. For every real trade the engine spawns *k* **shadow** trades — **same direction, same stop distance, same TP R-multiples**, but entered at **random bars** — and manages them identically. A setup is `+EV` only if its expectancy **beats that shadow distribution** with significance — the excess over the null must clear a **stricter bound** (`sig_z=1.65` ≈ 95% one-sided, not a 1-sigma bound), which holds the small-sample false-positive rate down (empirically ~5% pooled, ~2% at the board's real sample sizes; see `proofs/planted_edge_lab.py` and `REPORTPLANTEDEDGERESULTS.md`). This subtracts **both** the exit artifact **and** generic directional drift (e.g. "it was a bull market"), leaving only real entry-selection edge. *Live-proven: on a 12-coin USD-M 4h pool, `trend_pullback` went **INCONCLUSIVE → +EV** (random longs of its geometry lost −0.08R, so its entry timing adds a genuine +0.10R), while `divergence_reversal` — which beats random shorts by +0.42R but is still an absolute money-loser — stays −EV.* The pooled cache stores the null per setup × regime; **`scan --refresh`** rebuilds it.

## 7. `alert` — level & setup pings (optional)

```bash
python3 -m src.cli alert [SYMBOL] [--spot|--usdm] [--level PRICE] [--setup] [--tf TF] [--note "…"] [--list] [--all] [--check] [--remove ID]
```
```bash
python3 -m src.cli alert WLD/USDT:USDT --usdm --level 0.58 --note "retest short trigger"  # add a level alert
python3 -m src.cli alert WLD/USDT:USDT --usdm --setup --tf 4h                              # alert when a setup triggers
python3 -m src.cli alert --check          # evaluate all alerts now (also runs each `watch` pass)
python3 -m src.cli alert --list           # show active alerts
python3 -m src.cli alert --list --all     # include already-triggered
python3 -m src.cli alert --remove a1b2c3  # delete one by id
```

---

## 8. `stage` — the order gate (DRY-RUN by default — sends nothing)

```bash
python3 -m src.cli stage SYMBOL [--spot|--usdm] [--tier …] [--tf TF] [--live] [--testnet]
```
```bash
python3 -m src.cli stage WLD/USDT:USDT --usdm   # dry-run: guards→plan→re-fetch→ticket→type CONFIRM→logs STAGED
```
At the prompt, type the literal word **`CONFIRM`** to stage (anything else aborts). Nothing is transmitted in dry-run.

> **`stage` is edge-aware.** It consults the pooled universe verdict for the setup in the *current regime* and **BLOCKS a setup that is proven −EV** (so a high-confluence "grade-A" trade whose backtested edge is negative can't be armed), and **warns** when the edge is merely unproven (you may still proceed, discretionarily). Run `scan` first to build/refresh the universe edge cache the gate reads.

## 9. `manage` — manage open/staged positions (DRY-RUN by default)

```bash
python3 -m src.cli manage [SYMBOL] [--spot|--usdm] [--tier …] [--tf TF] [--live] [--testnet] [--flatten]
```
```bash
python3 -m src.cli manage --usdm                 # all open/staged; omit SYMBOL = manage everything
python3 -m src.cli manage WLD/USDT:USDT --usdm    # just one
```
**Read/act:** paper-fills a staged trade when price reaches entry → manages by the plan (TP1→scale+breakeven→TP2/stop→close). Each action is **proposed**, you type `CONFIRM`. USD-M shows live liquidation/funding/OI warnings.

## 9b. `optimize` — walk-forward config optimization (offline, slow; propose-a-diff)

```bash
python3 -m src.cli optimize [--spot|--usdm] [--tier …] [--setup NAME] [--param NAME] [--coins N] [--tf TF]
```
```bash
python3 -m src.cli optimize --usdm                                  # all setups, stop_buffer_atr, pooled over the liquid shortlist
python3 -m src.cli optimize --usdm --setup breakout_retest --coins 8  # one setup, wider pool
python3 -m src.cli optimize --usdm --param zone_atr --setup range_fade
```
**What it does:** for each setup it pools trades **across coins**, splits each coin's history into **TRAIN / TEST / LOCK-BOX**, **selects** the parameter only on train, **validates** it on the out-of-sample test, then demands it (a) beat the default OOS, (b) beat chance (bootstrap significance, Bonferroni-adjusted for the grid), (c) sit on a stable plateau, (d) survive pessimistic costs, and (e) not be contradicted by the lock-box. It **proposes a config diff** for the changes that pass — and **applies nothing**. The common, honest result is *KEEP DEFAULT*. Slow; run it deliberately.

## 10. `review` — realised performance (no flags)

```bash
python3 -m src.cli review
```
**Read:** equity curve, expectancy + CI, win%/PF/maxDD, buckets by setup/regime/grade, **adherence** (discipline), and verdicts (needs ≥20 closed trades).

---

## The full daily sequence (copy-paste, dry-run, USD-M)

```bash
cd /Users/captiveon/Documents/techproject/experimental/thmillion/practical/TradingAIAssistant

COLUMNS=160 python3 -m src.cli context  --usdm                          # 1. the tide
COLUMNS=160 python3 -m src.cli scan     --usdm                          # 2. proven-edge board (all survivors, pooled edge)
COLUMNS=160 python3 -m src.cli screen   --usdm                          # 3. (optional) broader shortlist (top 100)
COLUMNS=160 python3 -m src.cli analyze  --all --usdm                    # 3b. all shortlist coins for live setups (one board)
COLUMNS=160 python3 -m src.cli analyze  SOL/USDT:USDT --usdm --tf 4h     # 4. deep read of your pick
COLUMNS=160 python3 -m src.cli backtest SOL/USDT:USDT --usdm --tf 4h     # 5. prove the edge (per-coin slice)
COLUMNS=160 python3 -m src.cli stage    SOL/USDT:USDT --usdm             # 6. dry-run order gate (type CONFIRM)
COLUMNS=160 python3 -m src.cli manage   --usdm                          # 7. paper-fill + manage (type CONFIRM)
COLUMNS=160 python3 -m src.cli review                                   # 8. performance + discipline

# Or let the radar + the master command do it for you:
COLUMNS=160 python3 -m src.cli watch    --usdm --start                  # radar: pings you when a proven edge appears
COLUMNS=160 python3 -m src.cli roar     --usdm                          # MASTER: whole proven basket → CONFIRM (dry-run)
```

---

## The LIVE ladder (real money — only when ready; do A→B→C→D in order)

**A. Paper** — everything above in default dry-run. Prove the loop + your discipline first.

**B. Testnet** — validate the real send path with fake funds. Needs **separate testnet keys**
from testnet.binancefuture.com (swap them into `.env` temporarily; your mainnet key won't
authenticate on testnet):

```bash
python3 -m src.cli stage  SOL/USDT:USDT --usdm --testnet     # places on the sandbox
python3 -m src.cli manage --usdm --testnet                   # reconcile + manage on sandbox
python3 -m src.cli manage --usdm --testnet --flatten         # sandbox kill switch
```

**C. Arm real money** — export the flag per-session (deliberately NOT in `.env`):

```bash
export LIVE_TRADING_ENABLED=I_UNDERSTAND_THE_RISK
```
Without this exact value, `--live` runs preflight and **refuses to send.**

**D. Go live** (put your real mainnet key back in `.env`):

```bash
python3 -m src.cli stage  SOL/USDT:USDT --usdm --live        # full gate → preflight → 2nd re-fetch → type CONFIRM LIVE → REAL order
python3 -m src.cli manage --usdm --live                      # reconcile w/ exchange, then manage; type CONFIRM LIVE per action
python3 -m src.cli manage --usdm --live --flatten            # KILL SWITCH: close everything, cancel all orders
```
Live placement is **never naked**: on fill it immediately arms the protective stop; if the
stop fails it emergency-closes. Partial fills size the stop to what filled. The exchange is
the source of truth — `manage --live` reconciles before acting.

---

## Configuration & risk control (`.env` — full control for testing)

**Every** config field is `.env`-overridable via `{SECTION}_{FIELD}` (type-coerced).
Copy `.env.example` (fully commented). Prefixes: `SCREEN_ STRUCT_ ANALYSIS_ SETUP_
MARKETS_ RISK_ MGMT_ BACKTEST_ EDGE_ GUARDS_ JOURNAL_ ALERTS_ WATCH_ ORDER_
OPTIMIZE_ LIVE_`. Friendly aliases: `ACCOUNT_EQUITY`, `RISK_PCT`, `SCREEN_TF`.

**Risking more than 1% — the effective-risk pipeline (P12 intelligent sizing).** `RISK_PCT` is the *base*;
it flows into real sizing (`analyze`/`stage`/`roar`, proven live) through
`base% → ×drawdown-scale → ×edge-scale → ×vol-target(σ_R) → Kelly cap → clamp to MGMT_MAX_RISK_PCT`.
Each stage is opt-in (default off → flat base %). `analyze` prints a **Position-sizing guidance** panel
showing the evidence (edge · confidence/sample-quality · σ_R · drawdown) and exactly which stages fired —
sizing is **discretionary guidance, never automation**. Correlation & portfolio exposure are handled at the
**basket** level (`roar`: correlated coins share a budget, total heat capped).

```bash
RISK_PCT=3                        # base 3%/trade (size derived from the stop)
MGMT_MAX_RISK_PCT=5               # hard ceiling (e.g. RISK_PCT=8 → clamped to 5%)
MGMT_DD_SCALE_ENABLED=true        # auto-cut risk in a drawdown (3% → 1.5% at −20%)
MGMT_EDGE_SCALED=true             # size from edge×confidence (unproven setup → ×0.5)
MGMT_VOL_TARGET_ENABLED=true      # size from VARIANCE: wilder setup (high σ_R) → smaller (σ_R 2.0 → ×0.75)
MGMT_KELLY_ENABLED=true           # fractional-Kelly cap (only reduces)
GUARDS_MAX_TRADES_PER_DAY=3       # over-trading guard (0 = unlimited)
GUARDS_HEAT_CAP_PCT=6             # max TOTAL open risk as % equity
```

> Pain is non-linear: a 10-loss streak costs ~10% at 1%, ~26% at 3%, ~40% at 5%.
> Size from the edge, not the appetite. Safety rails (the `--live` arm flag + confirm
> phrase) are **never** `.env`-overridable.

## Housekeeping

```bash
rm -f data/journal.jsonl     # reset the trade journal (clears review/guards history)
ls data/edge_cache/          # cached backtest profiles; `scan --refresh` ignores them
pytest -q                    # run the test suite (offline)
```

## Safety quick-reference

- **Dry-run is the default.** `stage`/`manage` send nothing without `--live`.
- **`--live` needs BOTH** `LIVE_TRADING_ENABLED=I_UNDERSTAND_THE_RISK` **and** a typed `CONFIRM LIVE`.
- **NO TRADE / empty board is the most common honest answer** — the backtest gate and analyze layer refuse weak trades.
- **Risk-first:** you set risk %; size is derived from the stop. **Never a naked live position.** **`--flatten`** is the kill switch.
- **Keys** are trade-only, no-withdrawal, IP-locked, gitignored; the secret is never logged.

---

*Start with the dry-run daily sequence. Everything else is the same commands with `--testnet`,
then `--live`, once you've proven it to yourself.*
