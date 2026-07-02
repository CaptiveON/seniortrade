# Independent Quant Audit — TradingAIAssistant
**Auditor posture:** senior quant/trading-systems auditor; adversarial, evidence-only. Every claim below was verified by reading code, running the 268-test suite, executing live commands against Binance, and running targeted adversarial experiments (fresh-noise fabrication probes, planted-edge recovery, order-dependence proofs). Nothing is taken from the docs on faith.
**Date:** 2026-07-02 · **Scope:** full codebase (`src/`, 25 test files, `proofs/`), BRIEF.md, ENHANCEMENTBRIEF_1.md, MANUAL.md, README.md, caches, live behavior.

---

## 1. Executive verdict

**This is a genuinely rigorous research-grade quantitative decision-support engine whose *statistical honesty machinery* meets or exceeds institutional practice, but whose *data foundation* and *engineering operations* remain pre-industrial, and one demonstrated statistical defect (pooled fold/recency metrics) needs fixing before its robustness/confidence numbers on pooled profiles can be fully trusted.**

Grade against the "industrial quantitative decision support system" bar:

| Dimension | Grade | One-line judgment |
|---|---|---|
| Statistical honesty (null baseline, multiple testing, gating) | **A** | Better than most retail and much “semi-pro” software; institutional-practice level |
| Validation discipline (adversarial testing, live proofs) | **A−** | Planted-edge lab + noise-fabrication probes are genuinely rare rigor |
| Statistical estimation correctness | **B−** | One demonstrated defect (pooled ordering); i.i.d. assumptions unexamined |
| Data foundation | **C** | Single venue, survivor universe, no PIT universe, no candle sanitation |
| Execution & risk safety | **A−** | Defense-in-depth is real: rails are code-enforced and unoverridable |
| Reporting & explainability | **A−** | Auditable "why", cost decomposition, distributions — professional |
| Engineering/ops maturity | **D+** | **No version control**, no CI, no lint/typecheck, deps not hard-pinned |
| Overall vs "industrial production desk system" | **B / 7.5–8 out of 10** | Industrial *methodology*, pre-industrial *infrastructure* |

**Bottom line for the stated aim** — *"find proven opportunities and guide trade decisions on rigorous statistical analysis and probability"*: the engine **does what it claims, honestly**. Its most convincing behavior is what it *refuses* to do: on the current risk-off tape it outputs an **empty board** rather than manufactured signals, and on adversarial pure noise it fabricates **zero** +EV verdicts while still recovering **4/4** genuinely planted edges. That asymmetry (specific *and* sensitive) is the core competence of a real edge-finding system, and it demonstrably has it.

---

## 2. What was independently verified (evidence, not claims)

### 2.1 The suite is real
- **268 tests, 0 skips/xfails/deselects** (checked with `-rsxX`), deterministic across consecutive runs, 592 assertions (≈2.3/test — not vacuous). One warning, third-party (pytz).

### 2.2 The honesty machinery works under adversarial attack
- **Fabrication probe (fresh seeds, never used by the tests):** 12 synthetic driftless GBM "coins" → pooled → full evaluation. **0 of 10 setups stamped +EV.** Setups with positive raw expectancy on noise (e.g. failed_breakout +0.47R) were correctly held at INCONCLUSIVE by the null-excess significance gate. **0** context refinements or interactions proven on noise; feature importance found ≤2 raw-significant directions, all killed by Bonferroni.
- **Sensitivity probe:** planted conditional edges in heavy noise (4 mechanisms) → **4/4 recovered as +EV** with large null-excess. The system is not "safe because it never fires."
- **The null baseline is architecturally correct:** matched-geometry random-entry shadows, per-regime, used as a *gate + artifact subtractor*, never a magnitude bonus — a money-loser that "beats a worse null" is rejected (unit-tested and verified in `null_adjusted_edge`).

### 2.3 Safety rails are code-enforced, not documentation
- `LIVE_ENABLE_VALUE` / `CONFIRM LIVE` / arm-flag name are in a `_PROTECTED` denylist — **environment cannot override them** (verified by attempting it). Dry-run is default; live requires env-arm **and** a distinct typed phrase; margin preflight; never-naked (stop-or-bail); testnet-first; `manage --flatten` kill switch; roar re-validates each leg on fresh data at send time. This is defense-in-depth as practiced on real desks.

### 2.4 Coherence invariants hold
- One edge rule (`null_adjusted_edge`) shared by board scorer and stage gate — they cannot disagree. One confidence function (`statistical_confidence`; the legacy formula is fully removed — verified by word-boundary search). One sizing pipeline. Backtest ≡ live management by construction (shared `walk_management` state machine, consumed by both — verified).
- Cache integrity: all layered fields (null-per-regime, P5 shrinkage + τ², P3/P14 archetype table, P12 σ_R, P6 bootstrap/recency, P13 importance) present, all values finite, `ci_low ≤ expectancy` invariant holds across every cell.

### 2.5 The enhancement brief is genuinely implemented
All 15 priorities are **live code paths**, not documentation (each was exercised end-to-end during this audit): contextual refinements with a strict money+Bonferroni+OOS gate (P1/P4), evidence-driven confidence = P(beats null)×fold (P2), archetype conditional expectancy (P3/P14), empirical-Bayes shrinkage (P5), robustness composite (P6), time-decay with Kish effective-N (P7), explainability/for-against (P8), distribution & cost decomposition (P9/P10), MC drawdown distributions (P11), σ_R vol-targeted sizing (P12), persisted feature-importance trends (P13), professional reporting (P15).

---

## 3. Findings (ranked by severity)

### FINDING 1 — **HIGH — DEMONSTRATED DEFECT**: pooled fold-consistency / "recent" / walk-forward metrics are coin-order-dependent, not chronological
`build_pooled_profiles` concatenates trades **coin-by-coin** (`pooled[...].extend(trades)` in screener order). `fold_expectancies` and the "recent last-third" both consume that raw list order. **Consequence, proven with a synthetic experiment:** identical trades with identical timestamps produce `recent_expectancy = −0.5` in one coin order and `+0.5` in the reversed order; fold values flip sign likewise.
- On the **pooled** path (the one the live board uses), "fold consistency" actually measures *consistency across coin blocks* (a defensible cross-sectional robustness statistic — but **not** the chronological OOS the docs and P6 label claim), and the P6 "walk-forward non-decay" axis + P9 "recent expectancy" measure *"the last ~4 coins in screener order,"* which is meaningless as a time signal.
- **Blast radius:** P2 confidence multiplies by this fold term → live confidence numbers on pooled profiles carry a mislabeled ingredient. P6 robustness `walk_forward` axis is unreliable on pooled profiles. Single-coin `backtest` output is **unaffected** (trades are time-ordered there). The core edge/null/sig_z gate is **unaffected** (order-independent means).
- **Fix (small):** sort pooled trades by `entry_ts` before `evaluate` (they carry timestamps since P7), or compute folds/recency on time-sorted copies. Then fold-consistency becomes genuinely chronological on the pooled path.

### FINDING 2 — **HIGH — STRUCTURAL**: pooled sample sizes overstate independent evidence (cross-coin correlation ignored)
Crypto majors are strongly cross-correlated (shared BTC beta). 900 pooled trades across 12 coins over the *same* window are far fewer than 900 independent observations; simultaneous same-direction trades on correlated coins are near-duplicates. All SEs, CIs, P(beats null), and Bonferroni sizes treat trades as independent → **standard errors are understated by an unquantified factor**, and "n=919" in a regime overstates evidence. The system's *other* conservatisms (sig_z=1.65, null gate, floor) buy back margin, but this is the single largest unmodeled statistical risk in the verdicts. Mitigations: cluster-robust SEs (cluster by time-bucket), or deflate n by an effective-sample-size estimate from cross-coin trade overlap. At minimum the caveat belongs in the report footer next to the survivorship one.

### FINDING 3 — **MEDIUM**: universe selection bias (beyond acknowledged survivorship)
The pooled universe = *today's* top-volume screen survivors. Coins liquid/large today disproportionately had strong recent histories → edges estimated on them inherit selection bias (in addition to Binance delisting survivorship, which the tool *does* disclose in its caveats). No point-in-time universe exists. For the stated research use, verdicts should be read as "edge on currently-liquid names," which is weaker than "edge." Mitigation is hard without PIT data; honest labeling is cheap.

### FINDING 4 — **MEDIUM**: i.i.d./normality assumptions at several statistical joints
- Bootstrap CI and Monte-Carlo permutation resample trades i.i.d. — destroys autocorrelation and clustering (MC drawdown tails likely **understated**).
- Null shadows (k=10 per real trade) are drawn on the *same tape* and overlap in time → the null's SE is optimistic.
- P(beats null) is a normal approximation on R-distributions with observed skew +1.3…+2.9; acceptable at pooled n but optimistic in thin regime cells near `min_regime_n=20`.
None of these is disqualifying individually — they are standard first-generation simplifications — but an industrial system would document them and stress at least the MC with block-bootstrap.

### FINDING 5 — **MEDIUM — ENGINEERING**: research-integrity infrastructure is missing
- **The project is not a git repository.** For a research tool whose selling point is honesty, absence of version control is the most serious professionalism gap found: results are not reproducible against a code state, and there is no change audit trail. (Also: single-copy risk.)
- No CI (the 268-test suite runs only when someone remembers), no lint/type-checking config, dependencies bounded only from below (`>=`) — a future pandas/ccxt major can silently change behavior.
- Cache artifacts carry no schema-version stamp (backward-compat is handled by defensive `get`s — works, but drift is silent).

### FINDING 6 — **MEDIUM**: no market-data sanitation layer
`fetch_ohlcv` trusts the exchange: no checks for gapped candles, zero-volume bars, or exchange-outage artifacts before indicators/backtests consume them. Binance klines are relatively clean, but a single bad tape silently contaminates a pooled profile. Industrial ingest validates (monotonic ts ✓ — the new pagination does check that — plus gap/outlier/zero-volume screens ✗).

### FINDING 7 — **LOW-MEDIUM**: statistical depth-of-history vs claim strength
Default history is ~166 days (1000×4h; now extensible to 2000+ via the new pagination — good). One regime cycle. Regime-conditional claims from `min_regime_n=20` trades in that window are labeled with confidence tiers (good) but remain fragile; the empty current board is partly a *sample* phenomenon, not only a market one. The time-decay option further reduces effective n when enabled (correctly reflected in wider CIs via Kish — verified).

### FINDING 8 — **LOW** (known/accepted, listed for completeness)
- Exit engine is fixed TP1/TP2 + breakeven (multi-TP/trailing deferred by explicit owner decision) — edge estimates are conditional on this exit.
- Cost model: flat slippage bps + flat funding assumption; no depth/impact model (fine at intended sizes; wrong for size).
- Single venue, polling-based watch, desktop-notification ops — appropriate for one operator, not for a desk.
- Minor: a matched context cell reports the pool's n (not the cell's) into confidence-n; `features` importance scalar mixes breadth+depth heuristically (display-only).

---

## 4. Maturity assessment against the stated aim

**Aim:** *statistically proven opportunities → guided (never automated) trade decisions.*

| Capability the aim requires | Present? | Quality |
|---|---|---|
| Honest edge measurement (vs a real baseline, not zero) | ✅ | Excellent — null baseline + sig_z is the right bar and survived adversarial probing |
| Refusal to signal without evidence | ✅ | Excellent — empty board on risk-off tape; verified anti-fabrication |
| Multiple-testing discipline | ✅ | Strong (Bonferroni in conditioning; board-level caveat + strict z) |
| Regime/context awareness | ✅ | Strong, strictly gated; dormant until earned (correct) |
| Probability-language outputs | ✅ | P(beats null), P(>0), CIs, distributions, drawdown percentiles |
| Sizing guided by evidence | ✅ | Conservative, composable, opt-in; correlation at basket level |
| Execution safety | ✅ | Code-enforced rails; among the best parts of the system |
| Statistically *correct* robustness metrics | ⚠️ | Finding 1 must be fixed on the pooled path |
| Independent-evidence accounting | ⚠️ | Finding 2 unmodeled |
| Reproducible research infrastructure | ❌ | Finding 5 — no VCS/CI |

**Conclusion:** as a *decision-support research engine for a single disciplined operator*, it is at or above professional standard and unusually honest — I have audited commercial products that claim more and verify less. As an *industrial production system*, it is not there yet: the gap is not intelligence but **infrastructure (git/CI/pinning), data foundation (PIT universe, sanitation, venue redundancy), and two statistical corrections (pooled ordering; correlation-aware SEs)**.

## 5. Prioritized remediation (= the agreed roadmap, 2026-07-02)

**PHASE 1 — fix the measurement layer (the audit findings):**
1. **Fix Finding 1** — ✅ **DONE 2026-07-02.** Pooled trades now time-sorted at the source (`build_pooled_profiles`), plus defensive stable sorts in `evaluate` and `evaluate_conditioning` (chronology is a contract for any caller; no-op when timestamps are absent/ordered). Proven order-invariant by regression tests (the audit's exact flip experiment now asserts equality; 271 tests green). Cache rebuilt on corrected chronology (combined with the operator's 3000-candle + time-decay `.env` posture): chronological folds surfaced real decay coin-blocking had hidden (breakout_retest fold 0.33→0.00) and a genuine improver (range_fade → +EV, recent +0.93R); down-regime board unchanged (nothing proven; trend_pullback/down near-miss +0.02R vs 0.05R floor).
2. **`git init` + pin deps + CI** — ✅ **DONE 2026-07-02.** Repo live at `github.com/CaptiveON/seniortrade` (main, owner's LICENSE preserved). Pre-commit secret scan verified `.env`/keys excluded (defense-grade `.gitignore` kept). `requirements.txt` now upper-bounded per major; `requirements.lock` pins the exact proven versions (ccxt 4.5.59 / pandas 3.0.3 / numpy 2.2.6 / rich 13.9.4 / dotenv 1.2.2 / pytest 9.1.0). GitHub Actions CI (`.github/workflows/ci.yml`) runs the full offline suite from the lock on every push/PR (py3.12, no secrets needed).
3. Cross-coin correlation (Finding 2) — ✅ **DONE 2026-07-02** (effective-n deflation, MEASURED not assumed). `expectancy.design_effect`: one-way ANOVA intra-cluster ρ over same-UTC-day buckets → Kish DEFF; every SE/CI √DEFF-widened (per-regime cells get their own estimate; nulls inherit the setup's — they share the tapes); all sample gates + confidence sample-quality use n_eff = n/DEFF. ON by default (honesty, not appetite). Live measurement: ρ 0.06–0.66 per setup, DEFF 1.0–1.4 (SE ×1.0–1.2); e.g. trend_pullback 1636 pooled → ~1149 effective, range_fade ρ=0.66 (same-day chop trades echo). Footer caveats on board + backtest; per-setup correlation line in `backtest`. Planted-edge recovery still 4/4 under widened SEs (sensitivity retained). Residual (documented): conditioning lift tests (Welch) remain i.i.d.-based — Bonferroni over all splits is the standing guard there.
4. Candle sanitation (Finding 6) — ✅ **DONE 2026-07-02.** `sanitize_ohlcv` at the fetch choke point (every consumer inherits): DROPS only provably-broken rows (NaN/≤0 prices, high<low, wick-inconsistent body, bad volume); FLAGS — never "fixes" — gaps vs the tf grid, zero-volume bars, and bad-print-shaped spikes (range >12× median fully reverted next bar) into `df.attrs['quality']`; fully-broken tape → DataError instead of silent contamination. Surfaced as a yellow `data quality:` header line in `backtest`. Tests +5 → 281. Live: BTC/ETH/SOL/PEPE clean over 2000×4h; AGLD carries 1 flagged suspect print (kept as data — a real crash must stay data).
5. Deeper history + decay posture (Finding 7) — ✅ **DONE 2026-07-02** (owner chose *documented posture*, not changed defaults): `.env.example` ships a "RECOMMENDED RESEARCH POSTURE" block (`CANDLE_LIMIT=2000` + `TIME_DECAY_ENABLED=true` + rebuild command); MANUAL states it; shipped defaults untouched (1000/off).
6. Block-bootstrap MC + assumption/universe labeling (Findings 4+3) — ✅ **DONE 2026-07-02.** `monte_carlo` now uses a **circular block bootstrap** by default (auto ≈√n; `BACKTEST_MC_BLOCK=1` restores legacy i.i.d.) so streaks/clustered volatility survive into drawdown tails — on a losing-regime series iid said p95 DD 5R/ruin 0% vs block 41R/ruin 55%; on a real ETH tape the block read was slightly *shallower* (follows measured dependence, either direction — accuracy, not blanket conservatism). Remaining i.i.d. simplifications (bootstrap CI, null-shadow overlap, normal approximation) + universe-selection/survivorship biases are now **stated** in MANUAL and in the backtest/board footers.

**PHASE 1 COMPLETE (all 6 items, 2026-07-02) — 283 tests green, every item committed to `main` with live proof.**

**PHASE 2 — board scope upgrade (owner's beta observation, validated 2026-07-02):**
7. **Tide/beta lane + near-miss transparency.** The engine already *measures* trend-beta (positive matched-geometry nulls in the down regime: random shorts +0.01…+0.14R) but only *offers* timing-alpha. Add a clearly-labeled MARKET-POSTURE (beta) lane when BTC trends — best structure-timed with-tide candidates, labeled "rides the tide; ~0 timing alpha," sized by the same risk pipeline — plus a near-miss panel (best unproven with exact gate reasons, e.g. trend_pullback/down +0.04R vs 0.05R floor). Never mixed with PROVEN alpha rows; the alpha bar does not move.

**PHASE 3 — parked (owner-sequenced):**
8. Risk-profile system (L0 conservative → L3 high): profiles move capital/exposure + selectivity only; honesty gates + safety rails frozen; separate explicit "strictness" axis (design discussed 2026-07-01).
9. Exit strategy: N structure-placed TPs + trailing runner, per-setup, optimize-proven.

*Report generated from direct code inspection, 268-test suite execution, live Binance runs, and adversarial statistical experiments. No finding above is speculative; each carries reproduction evidence in the session log.*
