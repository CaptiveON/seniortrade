# Advanced Analytics Audit — On-Chain, Derivatives, Order-Flow, Macro
**Date:** 2026-07-03 · **Auditor posture:** same as AUDITREPORT.md — evidence-only; every PRESENT claim verified in code this session; every EDGE claim graded honestly (mechanism + evidence quality + fit to our 4h swing horizon), never vendor marketing.

**The governing principle (non-negotiable, from this system's own architecture):** no external metric "creates a statistically proven edge" by citation. In this engine a metric earns influence exactly one way: **point-in-time capture → context feature → `evaluate_conditioning` (money + Bonferroni + OOS) → deff/effective-n → gated promotion.** That machinery (P1/P4/P13) is already built — new analytics are new *feature columns*, not new signal logic. Anything that can't be measured through those gates (macro metrics with ~4 independent cycles of history) is labeled **posture context**, never "proven edge."

---

## CATEGORY 1 — On-chain whale & institutional activity

### 1.1 Exchange Netflow
- **PRESENT: No.** (`grep -rniE "netflow|glassnode|cryptoquant"` → zero hits; the system consumes exchange market data only.)
- **EDGE VALUE (honest):** Mechanism is real: sustained coin outflows from exchanges shrink liquid sell-side supply (accumulation); inflows precede distribution. Evidence quality: **medium at weekly horizon, weak at 4h** — netflow is a slow variable; public studies are mostly vendor-published (selection bias). At our 4h swing horizon it is a *regime/context* candidate (e.g., `flow=accumulating/neutral/distributing` bucket on a 7–30d net z-score), not an entry timer. It must clear our conditioning gates before touching any verdict.
- **IMPLEMENTATION PLAN:** Data is the blocker — reliable netflow is **paid**:
  - CryptoQuant: `GET https://api.cryptoquant.com/v1/btc/exchange-flows/netflow?window=day&from=…` (JWT; Advanced plan).
  - Glassnode: `GET https://api.glassnode.com/v1/metrics/transactions/transfers_volume_exchanges_net?a=BTC&i=24h&api_key=…` (paid tier).
  - Ingestion: new `src/onchain.py` → daily fetch → `data/onchain/{asset}_netflow.json` (append-only, PIT); **lag 1 day** (on-chain dailies publish with delay — using same-day values is look-ahead). Feature: `flow` bucket from a 30d z-score of net flow / exchange balance, merged onto bars via `merge_asof`, added to `CONTEXT_FEATURES`. The gates decide the rest.
  - **Recommendation:** only if you hold a subscription; otherwise skip — no honest free source.

### 1.2 Whale Transaction Count (>$100k on-chain)
- **PRESENT: No.**
- **EDGE VALUE (honest):** Weakest of the ten for us. Large transfers are ambiguous (internal exchange shuffles, custody rotations dominate); count-of-large-tx has poor signed information, and at 4h it's noise. Published "whale alerts → price" results don't survive multiple-testing scrutiny. Plausible only as a volatility-imminent flag, which our ATR/vol features already proxy.
- **IMPLEMENTATION PLAN:** Whale Alert `GET https://api.whale-alert.io/v1/transactions?api_key=…&min_value=1000000&start=…` (free tier: 10 req/min, high min-value, 1h history — inadequate for backtests). Glassnode `transfers_volume_more_100k_usd` (paid). Same feature path if ever added. **Recommendation: do not build** until something upstream proves value.

### 1.3 Mean Coin Age / Dormancy (LTH behavior)
- **PRESENT: No.**
- **EDGE VALUE (honest):** Coin-days-destroyed / dormancy spikes mark long-term-holder distribution — genuinely informative at **cycle** scale (weeks–months). At our horizon it can only modulate posture. Critically: cycle-scale claims have n≈4 independent observations in BTC's history — **our own significance machinery can never stamp this "proven"**; honest role = macro context.
- **IMPLEMENTATION PLAN:** Glassnode `indicators/average_dormancy`, `indicators/cdd` (paid); CryptoQuant equivalents (paid). Free partial: CoinMetrics community has `SplyAct1yr%` style activity metrics (limited). Same `src/onchain.py` daily-PIT path, feeding a `lth=holding/distributing` posture flag. **Priority: low** (paid + macro + weakest horizon fit).

---

## CATEGORY 2 — Derivatives leverage & sentiment

### 2.1 Open Interest
- **PRESENT: Yes (live read), No (as a measured feature).** `src/markets/usdm.py::open_interest()` fetches per-coin OI level + trend; consumed in `src/analysis.py` (freshness/extension lens: "falling OI" note, line ~278) and the Futures panel (`cli`). **Gap:** no OI *history*, so OI never enters the backtest → its edge contribution is **unmeasured** (currently a heuristic lens nudge, which is below this system's own standard).
- **EDGE VALUE:** The strongest mechanical family for our horizon: OI expansion into a move = new leverage (fuel for cascades); OI flush + price stability = deleveraged base. Directly testable at 4h.
- **IMPLEMENTATION PLAN:** Binance free: `GET /futures/data/openInterestHist?symbol=BTCUSDT&period=4h&limit=500` (ccxt raw: `ex.fapiDataGetOpenInterestHist`). **⚠ Binance retains only ~30 days** → start a persistent collector NOW: `src/derivs.py` → `data/derivs/{sym}_oi_4h.json` append-on-every-scan/watch-cycle. Features: `oi_chg` bucket (rising/flat/falling, 24h z-score) + `oi_price_div` (OI up & price down = trapped longs). Aligned to bars by timestamp (`merge_asof`, lag 1 bar), added to `CONTEXT_FEATURES` → gates decide.

### 2.2 Funding Rates
- **PRESENT: Yes (three real uses), No (as a measured feature).** `src/markets/usdm.py::funding()` (rate + `crowded_long/short` signal); used (a) contrarian nudge in the context lens (`analysis.py` ~205-211), (b) **cost model** — real rate in `plan_trade`, assumed carry in backtest `_cost_breakdown`, (c) `guards` funding-window check; shown in the Futures panel; passed through roar re-validation. **Gap:** same as OI — no historical series as a backtest feature; the "contrarian signal" thresholds are heuristics that have never faced our gates.
- **EDGE VALUE:** Best evidence of the ten at our horizon: extreme funding = crowded positioning; mean-reversion after funding extremes is one of the few crypto effects that has held out-of-sample across venues. Fully testable here — **funding history is free and complete**, so this can be measured through the gates *immediately* (no data-accumulation wait).
- **IMPLEMENTATION PLAN:** `ex.fetch_funding_rate_history(symbol, since, limit=1000)` paginated (full history, free). Features: `fund` bucket from a rolling 30d z-score (extreme+/high/neutral/high−/extreme−) + funding momentum. Merge onto bars (funding is 8h — forward-fill, lag 1 bar), extend `bt._bar_context`, let `evaluate_conditioning` + `features` judge. Also replace the flat `assumed_funding_per_8h` backtest cost with the *actual* historical rate per trade (cost-model accuracy upgrade for free).

### 2.3 Long/Short Ratio (top traders)
- **PRESENT: No.**
- **EDGE VALUE:** Medium-good: top-trader *position* ratio at extremes is a positioning/contrarian input like funding but less arbed. Caveats: Binance-only view, definition changes historically, and **30-day retention** — cannot be backtested until we've collected our own history.
- **IMPLEMENTATION PLAN:** Free Binance endpoints (ccxt raw): `fapiDataGetTopLongShortPositionRatio`, `fapiDataGetTopLongShortAccountRatio`, `fapiDataGetGlobalLongShortAccountRatio`, `fapiDataGetTakerlongshortRatio` (`?symbol=BTCUSDT&period=4h&limit=500`). Same collector as OI (`src/derivs.py`, persist from day one). Interim: display in the Futures panel as *context (unmeasured)*; after ~90 days of collection, promote to `CONTEXT_FEATURES` and measure.

---

## CATEGORY 3 — Order-book structure & liquidity

### 3.1 Volume Profile Visible Range
- **PRESENT: Yes.** `src/structure.py::VolumeProfile` — volume-at-price over the analysis window → **POC / VAH / VAL** (lines 76-93); consumed in `analysis.classify_location` (price at VAH/VAL = location confluence, ~138-140) and level confluence. This *is* VPVR over the visible range.
- **Gap (minor):** only the three headline nodes; no HVN/LVN map (thin-zone traversal targets), and VP nodes aren't a backtest context feature. **Upgrade (cheap):** expose the full histogram, tag HVN/LVN, add `vp_loc` (at_hvn/at_lvn/mid) to `CONTEXT_FEATURES`. No new data needed.

### 3.2 Cumulative Volume Delta
- **PRESENT: Partial — honest proxy.** `src/indicators.py::cvd_proxy()` (A/D-line approximation; its own docstring states "True CVD needs taker buy/sell volume"); feeds the volume lens ("CVD accumulation"). **Gap:** not true taker delta, and no spot-vs-perp divergence.
- **EDGE VALUE:** True CVD divergence (price HH while delta LH; spot bid vs perp selling) is a genuine aggression read at exactly our timescale; the proxy loses most of it.
- **IMPLEMENTATION PLAN — free and immediate:** Binance klines carry **taker buy volume** (raw endpoint, array index 9): `ex.publicGetKlines` / `ex.fapiPublicGetKlines({symbol, interval, limit})` → `delta = 2·takerBuyVol − vol`; `CVD = cumsum(delta)`. Full history via the same pagination we built. Features: `cvd_div` (bull/bear/none — price-vs-CVD swing divergence, same pattern as the existing RSI `_divergence_flag`) and `basis_flow` (spot CVD slope − perp CVD slope z-score). Replaces the proxy in the volume lens *and* enters the gates as a feature.

---

## CATEGORY 4 — Macro network health & valuation

### 4.1 Hash Rate / Staking Participation
- **PRESENT: No.**
- **EDGE VALUE (honest):** Security per se ≈ zero trading edge at 4h. The defensible derivative is **miner capitulation** (hash-ribbon style: 30d/60d hashrate cross) as a bottom-context flag — cycle-scale, BTC-only, n≈handful of episodes → **cannot be "proven" under our gates; posture context only.**
- **IMPLEMENTATION PLAN:** Free: `GET https://mempool.space/api/v1/mining/hashrate/all` (or blockchain.info charts API). Daily PIT ingest → `hash_ribbon=capitulation/recovery/normal` flag feeding the **tide/posture layer** (`market_context`), explicitly labeled macro-context — and a natural input to the Phase-3 risk profiles (modulate posture, never verdicts).

### 4.2 MVRV Z-Score
- **PRESENT: No.**
- **EDGE VALUE (honest):** The best macro-valuation metric: MV−RV normalized by history flags cycle extremes (z>~7 tops, z<0 accumulation). Same hard truth: **~4 independent cycles** — statistically unprovable by our machinery; belongs in the tide/posture layer and risk-profile modulation ("macro extreme → conservative posture"), clearly labeled.
- **IMPLEMENTATION PLAN — free:** CoinMetrics community API (no key): `GET https://community-api.coinmetrics.io/v4/timeseries/asset-metrics?assets=btc&metrics=CapMrktCurUSD,CapRealUSD&frequency=1d&page_size=10000` → `MVRV = MV/RV`; **Z with an EXPANDING-window σ** (full-sample σ is look-ahead — the classic chart cheats; ours must not): `z_t = (MV_t − RV_t)/σ(MV_{≤t})`. Daily PIT, lag 1 day, → posture flag `macro=euphoric/neutral/deep_value` in `market_context`.

---

## Critical gaps — summary

1. **Derivatives history is the real hole (and it's free).** Funding/OI exist as *live reads and costs* but have never been **measured as features** through the gates — below the system's own standard. Long/short ratios are absent entirely, and Binance's 30-day retention means **every day we don't collect is history lost forever**.
2. **CVD is a proxy** when the true taker-delta is free in the same klines we already paginate.
3. **On-chain (Cat 1) is pay-walled** for the only honest sources; mechanism is real but slow; fit to 4h is context, not timing.
4. **Macro (Cat 4) is free but statistically unprovable at cycle scale** — its correct, honest home is the tide/posture layer and the upcoming risk profiles, never the edge gates.
5. **Architecture is NOT a gap:** the feature→conditioning→gate pipeline (P1/P13) was built for exactly this; no new statistical machinery is required for any of the ten.

## Prioritized integration roadmap (highest → lowest expected *testable* alpha)

| # | Item | Cost | History | Horizon fit | Why this rank |
|---|------|------|---------|-------------|---------------|
| **A1** | **Funding-rate history → gated features** (+ real funding in backtest costs) | free | **full** | ★★★ | Best-evidenced effect at our horizon; measurable through the gates *today* |
| **A2** | **True CVD + spot-vs-perp divergence** (klines taker-buy field) | free | full | ★★★ | Upgrades an existing proxy; genuine aggression read; immediate backtestability |
| **A3** | **OI-history features** (expansion/flush, OI-price divergence) + **start the 30-day-retention collector NOW** (OI + all L/S ratios) | free | 30d → self-collected | ★★★ | Mechanically linked to cascades; urgency = data loss is irreversible |
| **B1** | **Long/Short ratios → features** (after ~90d self-collected history; display-only meanwhile) | free | self-collected | ★★ | Positioning extreme complement to funding |
| **B2** | **MVRV-Z + hash-ribbon → tide/posture layer** (expanding-window, PIT, labeled macro-context; feeds Phase-3 risk profiles) | free | full (daily) | ★ (posture) | Real cycle information, honestly unprovable → posture only |
| **C1** | **Exchange netflow → context feature** | **paid** | vendor | ★★ (weekly) | Only with a CryptoQuant/Glassnode subscription |
| **C2** | Mean coin age / dormancy → posture | paid | vendor | ★ | Macro, paid, weakest marginal value over MVRV-Z |
| **C3** | Whale transaction count | free-thin/paid | inadequate | ☆ | Ambiguous signal; do not build |

**Non-negotiables carried into every item:** point-in-time discipline (lag published metrics; expanding-window normalizations), features enter only via `CONTEXT_FEATURES` → `evaluate_conditioning` → money+Bonferroni+OOS → deff-honest gates, macro items labeled context and excluded from "proven" language, and the `features` command tracks each addition's measured importance (P13) so anything that doesn't earn its place visibly decays.
