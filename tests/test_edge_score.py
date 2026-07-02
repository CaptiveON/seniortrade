"""Tests for the Edge Score, cache, and present-conditions logic."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.config import BacktestConfig, EdgeScoreConfig, Market
from src.edge_score import (
    EDGE_NEGATIVE,
    EDGE_PROVEN,
    EDGE_UNPROVEN,
    _cache_path,
    _conf_tier,
    _letter,
    _tide_aligned,
    edge_gate,
    load_cached,
    save_cache,
    score_opportunity,
)
from src.expectancy import evaluate
from src.market_context import NEUTRAL, RISK_OFF, RISK_ON

CFG = EdgeScoreConfig()


def _verdict(n, expectancy, ci_low):
    return {"regime": {"n": n, "expectancy": expectancy, "ci_low": ci_low}, "overall": None}


def test_edge_gate_classifies_pooled_verdict():
    # proven: enough sample, +EV, lower bound above the floor
    lvl, _ = edge_gate(_verdict(40, 0.35, 0.12), CFG)
    assert lvl == EDGE_PROVEN
    # negative: proven loser -> stage must block
    lvl, _ = edge_gate(_verdict(60, -0.20, -0.35), CFG)
    assert lvl == EDGE_NEGATIVE
    # unproven: lower bound at/below the floor (indistinguishable from zero)
    lvl, _ = edge_gate(_verdict(40, 0.10, 0.02), CFG)
    assert lvl == EDGE_UNPROVEN
    # unproven: too few pooled samples in this regime
    lvl, _ = edge_gate(_verdict(5, 0.50, 0.30), CFG)
    assert lvl == EDGE_UNPROVEN
    # unproven: no profile cached at all
    lvl, _ = edge_gate(None, CFG)
    assert lvl == EDGE_UNPROVEN


def _profile(by_regime):
    return {"setup": "trend_pullback", "verdict": "+EV", "fold_consistency": 1.0,
            "overall": {"n": 50, "expectancy": 0.3, "ci_low": 0.2},
            "by_regime": by_regime}


def test_score_opportunity_proven_regime():
    prof = _profile({"up": {"n": 30, "expectancy": 0.4, "ci_low": 0.25}})
    o = score_opportunity(setup="trend_pullback", side="long", provisional_grade="A",
                          freshness="fresh", tide_aligned=True, regime="up", profile=prof, cfg=CFG)
    assert o is not None
    assert o["edge_score_r"] == pytest.approx(0.25)     # 0.25 * (1*1*1)
    assert o["grade"] == "C"                            # 0.15 <= 0.25 < 0.30
    assert o["confidence_tier"] == "high"              # 30/50 = 0.6


def test_score_opportunity_unproven_in_regime():
    prof = _profile({"range": {"n": 5, "expectancy": 0.3, "ci_low": 0.2}})  # too few in regime
    assert score_opportunity(setup="trend_pullback", side="long", provisional_grade="A",
                             freshness="fresh", tide_aligned=True, regime="range",
                             profile=prof, cfg=CFG) is None
    # regime absent entirely
    assert score_opportunity(setup="trend_pullback", side="long", provisional_grade="A",
                             freshness="fresh", tide_aligned=True, regime="down",
                             profile=prof, cfg=CFG) is None


def test_score_opportunity_no_edge_when_lower_bound_at_floor():
    prof = _profile({"up": {"n": 40, "expectancy": 0.2, "ci_low": 0.04}})   # <= floor 0.05
    assert score_opportunity(setup="trend_pullback", side="long", provisional_grade="A",
                             freshness="fresh", tide_aligned=True, regime="up",
                             profile=prof, cfg=CFG) is None


def test_edge_gate_blocks_money_loser_that_beats_a_worse_null():
    # divergence_reversal-style: LOSES money, but random entries lose MORE. "Less bad
    # than random" is still a loser — the gate must BLOCK, never call it proven.
    v = {"regime": {"n": 65, "expectancy": -0.14, "ci_low": -0.33},
         "null": {"n": 65, "expectancy": -0.99, "ci_low": -1.40}, "overall": None}
    lvl, _ = edge_gate(v, CFG)
    assert lvl == EDGE_NEGATIVE


def test_score_opportunity_rejects_money_loser_beating_worse_null():
    prof = _profile({"range": {"n": 65, "expectancy": -4.0, "ci_low": -6.8}})
    prof["null_by_regime"] = {"range": {"n": 65, "expectancy": -8.4, "ci_low": -11.0}}
    assert score_opportunity(setup="divergence_reversal", side="long", provisional_grade="C",
                             freshness="fresh", tide_aligned=True, regime="range",
                             profile=prof, cfg=CFG) is None


def test_score_opportunity_rejects_positive_not_significant_over_null():
    # positive expectancy but an informative null is ~equal -> not distinguishable -> None
    prof = _profile({"up": {"n": 100, "expectancy": 0.12, "ci_low": 0.02}})
    prof["null_by_regime"] = {"up": {"n": 100, "expectancy": 0.11, "ci_low": 0.01}}
    assert score_opportunity(setup="x", side="long", provisional_grade="A",
                             freshness="fresh", tide_aligned=True, regime="up",
                             profile=prof, cfg=CFG) is None


def test_score_opportunity_credits_real_edge_capped_at_expectancy():
    # genuine edge that significantly beats the null; the shown edge never exceeds expectancy
    prof = _profile({"up": {"n": 60, "expectancy": 0.40, "ci_low": 0.25}})
    prof["null_by_regime"] = {"up": {"n": 60, "expectancy": -0.10, "ci_low": -0.20}}
    o = score_opportunity(setup="x", side="long", provisional_grade="A", freshness="fresh",
                          tide_aligned=True, regime="up", profile=prof, cfg=CFG)
    assert o is not None
    assert 0 < o["trustworthy_edge_r"] <= 0.40 + 1e-9   # never invents profit above expectancy


def test_sig_z_rejects_marginal_edge_that_passes_at_z1():
    # An edge significant at the 1-SE bound but NOT at the stricter sig_z (small-sample guard).
    from dataclasses import replace
    v = {"regime": {"n": 60, "expectancy": 0.15, "ci_low": 0.08},
         "null": {"n": 60, "expectancy": 0.0, "ci_low": -0.10}, "overall": None}
    assert edge_gate(v, CFG)[0] == EDGE_UNPROVEN                 # default sig_z=1.65 → not significant
    assert edge_gate(v, replace(CFG, sig_z=1.0))[0] == EDGE_PROVEN  # the old 1-SE bound would pass it


def test_present_multiplier_penalises_counter_tide_and_extended():
    prof = _profile({"up": {"n": 40, "expectancy": 0.6, "ci_low": 0.5}})
    strong = score_opportunity(setup="x", side="long", provisional_grade="A", freshness="fresh",
                               tide_aligned=True, regime="up", profile=prof, cfg=CFG)
    weak = score_opportunity(setup="x", side="long", provisional_grade="C", freshness="extended",
                             tide_aligned=False, regime="up", profile=prof, cfg=CFG)
    assert weak["edge_score_r"] < strong["edge_score_r"]


def test_shrunk_reg_falls_back_to_pool_without_symbol_data():
    from src.edge_score import _shrunk_reg
    prof = {"by_regime": {"up": {"n": 200, "expectancy": 0.10, "ci_low": 0.06}}}
    reg = _shrunk_reg(prof, "ENA/USDT:USDT", "up")    # no 'symbols' block (old cache) → pool reg
    assert reg["expectancy"] == 0.10 and reg["n"] == 200


def test_shrunk_reg_pulls_toward_own_data():
    from src.edge_score import _shrunk_reg
    prof = {"by_regime": {"up": {"n": 200, "expectancy": 0.10, "ci_low": 0.06}},
            "tau2_by_regime": {"up": 0.05},
            "symbols": {"ENA/USDT:USDT": {"up": {"n": 120, "mean": 0.40, "sd": 0.3}}}}
    reg = _shrunk_reg(prof, "ENA/USDT:USDT", "up")
    assert reg["n"] == 200                            # anchored to pool evidence for the n-gate
    assert 0.10 < reg["expectancy"] <= 0.40           # pulled toward own
    assert reg["weight"] > 0.5 and reg["own_n"] == 120


def test_score_opportunity_symbol_shrinkage_backward_compatible():
    # no 'symbols' block in the profile → scoring with a symbol is identical to without one
    prof = _profile({"up": {"n": 60, "expectancy": 0.4, "ci_low": 0.25}})
    base = dict(setup="x", side="long", provisional_grade="A", freshness="fresh",
                tide_aligned=True, regime="up", profile=prof, cfg=CFG)
    o_pool = score_opportunity(**base)
    o_sym = score_opportunity(**base, symbol="ENA/USDT:USDT")
    assert o_pool and o_sym and o_sym["edge_score_r"] == o_pool["edge_score_r"]


def test_evaluate_conditioning_detects_a_real_split():
    import numpy as np
    from src.edge_score import evaluate_conditioning
    rng = np.random.default_rng(0)
    trades = [SimpleNamespace(r=float(rng.normal(0.5, 0.5)), regime="up", context={"vol": "squeeze"})
              for _ in range(200)]
    trades += [SimpleNamespace(r=float(rng.normal(-0.1, 0.5)), regime="up", context={"vol": "normal"})
               for _ in range(200)]
    refs = evaluate_conditioning({"s": trades}, CFG, min_child_n=20)
    assert any(r["feature"] == "vol" and r["value"] == "squeeze" and r["proven"] for r in refs)


def test_evaluate_conditioning_rejects_pure_noise():
    import numpy as np
    from src.edge_score import evaluate_conditioning
    rng = np.random.default_rng(1)
    trades = [SimpleNamespace(r=float(rng.normal(0.0, 0.5)), regime="up",
                              context={"vol": rng.choice(["squeeze", "normal", "expanding"]),
                                       "mom": rng.choice(["bull", "bear", "neutral"])})
              for _ in range(450)]
    refs = evaluate_conditioning({"s": trades}, CFG, min_child_n=20)
    assert not any(r["proven"] for r in refs)        # no real split → nothing earns specificity


def test_resolve_context_maps_htf_align():
    from src.edge_score import resolve_context
    r = resolve_context({"vol": "squeeze", "mom": "bull", "loc": "mid", "div": "none", "htf": "up"}, "long")
    assert r["htf_align"] == "aligned" and r["vol"] == "squeeze"
    assert resolve_context({"htf": "up"}, "short")["htf_align"] == "opposed"
    assert resolve_context(None, "long") is None


def test_context_cell_reg_matches_proven_refinement():
    from src.edge_score import _context_cell_reg
    prof = {"by_regime": {"down": {"n": 300, "expectancy": 0.10, "ci_low": 0.06}},
            "context_refinements": {"down": [
                {"feature": "vol", "value": "expanding", "n": 80, "exp": 0.40, "ci_low": 0.28,
                 "lift": 0.25, "tau2": 0.10}]}}
    reg = _context_cell_reg(prof, "down", {"vol": "expanding"})           # coin IS in vol=expanding
    assert reg is not None and reg["context_label"] == "vol=expanding"
    assert 0.10 < reg["expectancy"] <= 0.40                               # pulled up from the parent
    assert _context_cell_reg(prof, "down", {"vol": "squeeze"}) is None    # not in that context → parent
    assert _context_cell_reg(prof, "down", None) is None


def test_score_opportunity_activates_proven_context_cell():
    prof = _profile({"down": {"n": 300, "expectancy": 0.10, "ci_low": 0.06}})
    prof["context_refinements"] = {"down": [
        {"feature": "vol", "value": "expanding", "n": 80, "exp": 0.40, "ci_low": 0.28,
         "lift": 0.25, "tau2": 0.10}]}
    common = dict(setup="x", side="short", provisional_grade="A", freshness="fresh",
                  tide_aligned=True, regime="down", profile=prof, cfg=CFG)
    base = score_opportunity(**common)
    ctx = score_opportunity(**common, context={"vol": "expanding"})
    assert base and ctx and ctx["context_label"] == "vol=expanding"
    assert ctx["edge_score_r"] > base["edge_score_r"]    # the proven context cell raised the edge


def test_mappings():
    assert _letter(0.55, CFG) == "A" and _letter(0.31, CFG) == "B" and _letter(0.0, CFG) == "F"
    assert _conf_tier(0.7, CFG) == "high" and _conf_tier(0.4, CFG) == "mod" and _conf_tier(0.1, CFG) == "low"
    assert _tide_aligned("long", RISK_ON) and not _tide_aligned("long", RISK_OFF)
    assert _tide_aligned("short", RISK_OFF) and _tide_aligned("long", NEUTRAL)


def test_resolve_context_carries_archetype():
    from src.edge_score import resolve_context
    r = resolve_context({"vol": "expanding", "arch": "euphoria", "htf": "up"}, "long")
    assert r["arch"] == "euphoria" and r["htf_align"] == "aligned"


def test_arch_is_a_conditioning_feature():
    # an 'arch' split that genuinely makes money must be detectable by the SAME gate as any feature.
    from src.edge_score import CONTEXT_FEATURES, evaluate_conditioning
    import numpy as np
    assert "arch" in CONTEXT_FEATURES
    rng = np.random.default_rng(3)
    trades = [SimpleNamespace(r=float(rng.normal(0.6, 0.5)), regime="up", context={"arch": "euphoria"})
              for _ in range(120)]
    trades += [SimpleNamespace(r=float(rng.normal(-0.05, 0.5)), regime="up", context={"arch": "trending_up"})
               for _ in range(120)]
    refs = evaluate_conditioning({"s": trades}, CFG, min_child_n=20)
    proven = [r for r in refs if r["feature"] == "arch" and r["value"] == "euphoria" and r["proven"]]
    assert proven, "a real archetype edge must clear the money+Bonferroni+OOS gate"


def test_archetype_table_shrinks_toward_parent():
    from src.edge_score import _archetype_table
    # loud +0.5R cell with few trades is pulled toward the 0.0R parent; a flat cell stays put.
    parent = {"expectancy": 0.0, "ci_low": -0.05}
    trades = ([SimpleNamespace(r=0.5, regime="up", context={"arch": "euphoria"}) for _ in range(6)]
              + [SimpleNamespace(r=0.0, regime="up", context={"arch": "trending_up"}) for _ in range(200)])
    table = _archetype_table(trades, parent, min_n=5)
    assert "euphoria" in table and table["euphoria"]["raw_exp"] == pytest.approx(0.5)
    assert table["euphoria"]["exp"] < 0.5            # shrunk toward the parent
    assert table["euphoria"]["n"] == 6


def test_score_opportunity_activates_proven_archetype_cell():
    prof = _profile({"down": {"n": 300, "expectancy": 0.10, "ci_low": 0.06}})
    prof["context_refinements"] = {"down": [
        {"feature": "arch", "value": "panic", "n": 80, "exp": 0.45, "ci_low": 0.30,
         "lift": 0.30, "tau2": 0.10}]}
    common = dict(setup="x", side="short", provisional_grade="A", freshness="fresh",
                  tide_aligned=True, regime="down", profile=prof, cfg=CFG)
    base = score_opportunity(**common)
    panic = score_opportunity(**common, context={"arch": "panic"})
    assert base and panic and panic["context_label"] == "panic"   # named label, not 'arch=panic'
    assert panic["edge_score_r"] > base["edge_score_r"]


def _mk_ctx(rng, vol, htf, mean, k, regime="up"):
    return [SimpleNamespace(r=float(rng.normal(mean, 0.4)), regime=regime,
                            context={"vol": vol, "htf_align": htf}) for _ in range(k)]


def test_evaluate_conditioning_detects_a_real_interaction():
    # P4: edge exists ONLY in vol=expanding AND htf=aligned — a genuine synergy.
    import numpy as np
    from src.edge_score import evaluate_conditioning
    rng = np.random.default_rng(7)
    trades = (_mk_ctx(rng, "expanding", "aligned", 0.6, 90) + _mk_ctx(rng, "expanding", "opposed", -0.1, 90)
              + _mk_ctx(rng, "normal", "aligned", -0.05, 90) + _mk_ctx(rng, "normal", "opposed", -0.05, 90))
    refs = evaluate_conditioning({"s": trades}, CFG, min_child_n=20)
    pairs = [r for r in refs if r["interaction"] and r["proven"]]
    assert any({tuple(c) for c in r["conditions"]} == {("vol", "expanding"), ("htf_align", "aligned")}
               for r in pairs), "the real interaction must be found"
    p = next(r for r in pairs)
    assert p["exp"] > p["parent_exp"]          # beats its best single parent (true synergy)
    assert "&" in p["label"]


def test_pair_rejected_when_it_only_inherits_one_features_edge():
    # vol=expanding is +0.5R REGARDLESS of htf → the pair adds nothing → must NOT promote as an interaction.
    import numpy as np
    from src.edge_score import evaluate_conditioning
    rng = np.random.default_rng(11)
    trades = (_mk_ctx(rng, "expanding", "aligned", 0.5, 90) + _mk_ctx(rng, "expanding", "opposed", 0.5, 90)
              + _mk_ctx(rng, "normal", "aligned", -0.1, 90) + _mk_ctx(rng, "normal", "opposed", -0.1, 90))
    refs = evaluate_conditioning({"s": trades}, CFG, min_child_n=20)
    assert any(not r["interaction"] and r["proven"] and r["value"] == "expanding" for r in refs)  # single proves
    proven_pairs = [r for r in refs if r["interaction"] and r["proven"]]
    assert not proven_pairs, "a pair that only inherits one feature's edge is not a real interaction"


def test_hard_mode_tests_more_candidates_than_greedy():
    import numpy as np
    from src.edge_score import evaluate_conditioning
    rng = np.random.default_rng(3)
    # pure noise across two features → greedy screens out most pairs; hard tests them all
    trades = [SimpleNamespace(r=float(rng.normal(0, 1)), regime="up",
                              context={"vol": rng.choice(["squeeze", "expanding"]),
                                       "htf_align": rng.choice(["aligned", "opposed"])}) for _ in range(400)]
    greedy = evaluate_conditioning({"s": trades}, CFG, min_child_n=20, hard=False)
    hard = evaluate_conditioning({"s": trades}, CFG, min_child_n=20, hard=True)
    assert hard[0]["splits_tested"] >= greedy[0]["splits_tested"]


def test_context_cell_reg_matches_a_proven_pair():
    from src.edge_score import _context_cell_reg
    prof = {"by_regime": {"up": {"n": 300, "expectancy": 0.05, "ci_low": 0.01}},
            "context_refinements": {"up": [
                {"feature": "vol", "value": "expanding",
                 "conditions": [["vol", "expanding"], ["htf_align", "aligned"]],
                 "label": "vol=expanding & htf_align=aligned", "interaction": True,
                 "n": 80, "exp": 0.40, "ci_low": 0.28, "lift": 0.30, "tau2": 0.10}]}}
    # BOTH conditions present → the pair cell activates
    reg = _context_cell_reg(prof, "up", {"vol": "expanding", "htf_align": "aligned"})
    assert reg is not None and reg["context_label"] == "vol=expanding & htf_align=aligned"
    # only ONE condition present → no match → fall back
    assert _context_cell_reg(prof, "up", {"vol": "expanding", "htf_align": "opposed"}) is None


def test_statistical_confidence_is_evidence_driven():
    from src.edge_score import statistical_confidence
    strong = statistical_confidence({"expectancy": 0.20, "ci_low": 0.16, "n": 300},
                                    {"expectancy": 0.02, "ci_low": -0.01, "n": 300}, 1.0, CFG)
    marginal = statistical_confidence({"expectancy": 0.10, "ci_low": 0.03, "n": 300},
                                      {"expectancy": 0.06, "ci_low": 0.02, "n": 300}, 1.0, CFG)
    # a wide-margin edge is MORE certain to exist than one that barely clears the null
    assert strong["confidence"] > marginal["confidence"]
    # 'beats random' is a HIGHER bar than 'beats zero' when the null is positive
    assert marginal["p_beats_null"] < marginal["p_positive"]


def test_statistical_confidence_scales_with_fold_and_falls_back_without_null():
    from src.edge_score import statistical_confidence
    full = statistical_confidence({"expectancy": 0.20, "ci_low": 0.16, "n": 300},
                                  {"expectancy": 0.02, "ci_low": -0.01, "n": 300}, 1.0, CFG)
    half = statistical_confidence({"expectancy": 0.20, "ci_low": 0.16, "n": 300},
                                  {"expectancy": 0.02, "ci_low": -0.01, "n": 300}, 0.5, CFG)
    assert half["confidence"] == pytest.approx(0.5 * full["p_beats_null"])     # fold scales linearly
    # no informative null sample → confidence falls back to P(expectancy > 0)
    none = statistical_confidence({"expectancy": 0.10, "ci_low": 0.05, "n": 300}, None, 1.0, CFG)
    assert none["p_beats_null"] == pytest.approx(none["p_positive"])


def test_score_opportunity_exposes_both_probabilities():
    prof = _profile({"down": {"n": 300, "expectancy": 0.20, "ci_low": 0.14}})
    prof["null_by_regime"] = {"down": {"n": 300, "expectancy": 0.02, "ci_low": -0.02}}
    o = score_opportunity(setup="x", side="short", provisional_grade="A", freshness="fresh",
                          tide_aligned=True, regime="down", profile=prof, cfg=CFG)
    assert o and 0.0 <= o["confidence"] <= 1.0
    assert "p_positive" in o and "p_beats_null" in o
    assert o["confidence"] == pytest.approx(o["p_beats_null"] * prof["fold_consistency"])


def test_feature_importance_ranks_predictive_over_noise():
    import numpy as np
    from src.edge_score import feature_importance
    rng = np.random.default_rng(5)
    tr = ([SimpleNamespace(r=float(rng.normal(0.6, 0.4)), regime="up",
                           context={"vol": "squeeze", "div": rng.choice(["none", "bull"])}) for _ in range(120)]
          + [SimpleNamespace(r=float(rng.normal(-0.1, 0.4)), regime="up",
                             context={"vol": "normal", "div": rng.choice(["none", "bull"])}) for _ in range(120)])
    fi = feature_importance({"s": tr}, CFG, min_child_n=20)
    assert fi["vol"]["importance"] > fi["div"]["importance"]     # predictive feature outranks noise
    assert fi["vol"]["n_significant"] >= 1 and fi["div"]["n_significant"] == 0


def test_singles_only_and_active_features_restrict_pairs():
    import numpy as np
    from src.edge_score import evaluate_conditioning
    rng = np.random.default_rng(4)
    tr = [SimpleNamespace(r=float(rng.normal(0, 1)), regime="up",
                          context={"vol": rng.choice(["a", "b"]), "mom": rng.choice(["x", "y"]),
                                   "div": rng.choice(["p", "q"])}) for _ in range(300)]
    assert all(not r["interaction"] for r in evaluate_conditioning({"s": tr}, CFG, min_child_n=20, singles_only=True))
    allp = {r["label"] for r in evaluate_conditioning({"s": tr}, CFG, min_child_n=20, hard=True) if r["interaction"]}
    restricted = {r["label"] for r in evaluate_conditioning({"s": tr}, CFG, min_child_n=20, hard=True,
                                                            active_features={"vol", "mom"}) if r["interaction"]}
    assert restricted <= allp and not any("div" in lbl for lbl in restricted)   # div excluded from pairs


def test_record_feature_importance_persists_and_trends(tmp_path, monkeypatch):
    import src.edge_score as esmod
    monkeypatch.setattr(esmod, "_CACHE_DIR", tmp_path / "edge_cache")
    (tmp_path / "edge_cache").mkdir()
    imp1 = {"vol": {"importance": 1.0}, "div": {"importance": 3.0}}
    t1 = esmod.record_feature_importance(Market.USDM, "4h", imp1)
    assert t1["vol"]["trend"] == "new" and t1["div"]["trend"] == "new"     # first snapshot
    imp2 = {"vol": {"importance": 2.0}, "div": {"importance": 1.0}}          # vol up, div down
    t2 = esmod.record_feature_importance(Market.USDM, "4h", imp2)
    assert t2["vol"]["trend"] == "rising" and t2["div"]["trend"] == "decaying"
    assert esmod._fi_history_path().exists()


def test_robustness_distinguishes_consistent_from_fluke():
    from src.edge_score import robustness
    # consistently profitable: +EV, all folds positive, bootstrap CI>0, recent holds up
    c = robustness(0.20, 1.0, 0.05, 0.40, 0.22)
    assert c["score"] == pytest.approx(1.0) and c["label"] == "consistently profitable"
    # fluke: +EV overall but folds inconsistent, bootstrap straddles 0, recent collapsed (<0)
    f = robustness(0.20, 0.33, -0.10, 0.50, -0.05)
    assert f["score"] < c["score"] and f["walk_forward"] == 0.0   # recent collapse kills walk-forward
    assert f["label"] == "profitable but not robust"


def test_robustness_bootstrap_and_loser_axes():
    from src.edge_score import robustness
    # bootstrap CI entirely below zero → not distribution-free positive → robustness 0
    b = robustness(0.20, 1.0, -0.30, -0.05, 0.22)
    assert b["bootstrap"] == 0.0 and b["score"] == 0.0
    # an outright money-loser is 'not an edge' (robustness of a non-edge is moot)
    assert robustness(-0.10, 1.0, -0.2, 0.1, -0.1)["label"] == "not an edge"


def test_lookup_pooled_verdict_includes_robustness(tmp_path, monkeypatch):
    # report-only field must ride along on the verdict for analyze/board to show it
    import src.edge_score as esmod
    from src.expectancy import evaluate
    trades = [SimpleNamespace(r=0.5, regime="up", gross_r=0.5, funding_r=0.0) for _ in range(40)]
    prof = evaluate(trades, setup="trend_pullback", n_combos_tested=1, cfg=BacktestConfig())
    sym = "ROBCOIN/USDT:USDT"
    monkeypatch.setattr(esmod, "_CACHE_DIR", tmp_path)
    import json, time
    (tmp_path / f"{Market.USDM.value}_4h_UNIVERSE.json").write_text(json.dumps(
        {"timestamp": time.time(), "tf": "4h", "coins": [sym],
         "profiles": {"trend_pullback": esmod._compact(prof)}}))
    v = esmod.lookup_pooled_verdict(Market.USDM, "4h", "trend_pullback", "up")
    assert v is not None and "robustness" in v and 0.0 <= v["robustness"]["score"] <= 1.0


def test_shrunk_reg_carries_sigma_r_for_vol_target():
    from src.edge_score import _shrunk_reg
    # P12: the per-trade σ_R must flow through to sizing (pool fallback path)
    prof = {"by_regime": {"up": {"n": 200, "expectancy": 0.10, "ci_low": 0.05, "std_r": 1.8}}}
    reg = _shrunk_reg(prof, None, "up")
    assert reg["std_r"] == 1.8


def test_explain_signal_splits_for_and_against():
    from src.edge_score import EDGE_PROVEN, explain_signal
    # lens directions are LOWERCASE in real AnalysisResult (bull/bear/neutral) — match that contract
    lenses = [SimpleNamespace(name="trend", direction="bull", score=0.8, note="HTF up, HH/HL"),
              SimpleNamespace(name="momentum", direction="bear", score=0.5, note="RSI rolling over"),
              SimpleNamespace(name="noise", direction="bull", score=0.02, note="too weak to count")]
    out = explain_signal(direction="long", lenses=lenses, edge_level=EDGE_PROVEN,
                         edge_msg="proven edge +0.12R over 300", regime="up", freshness="fresh",
                         posture=RISK_ON, context_label="panic", confidence_tier="high")
    # decisive contributor leads the positives
    assert out["positive"][0].startswith("Proven edge in the up regime")
    assert any("trend:" in s for s in out["positive"])          # aligned lens supports
    assert any("momentum:" in s for s in out["negative"])       # opposed lens is against
    assert not any("noise" in s for s in out["positive"] + out["negative"])   # |score|<0.10 dropped
    assert any("panic" in s for s in out["positive"])           # earned context cell
    assert any("tide" in s.lower() for s in out["positive"])    # RISK_ON + long aligned


def test_explain_signal_flags_negative_edge_and_counter_tide():
    from src.edge_score import EDGE_NEGATIVE, explain_signal
    out = explain_signal(direction="long", lenses=[], edge_level=EDGE_NEGATIVE,
                         edge_msg="PROVEN −EV", regime="down", freshness="extended",
                         posture=RISK_OFF, context_label=None, confidence_tier="low")
    assert out["negative"][0].startswith("Backtests −EV")
    assert any("chasing" in s for s in out["negative"])         # extended entry
    assert any("Counter-tide" in s for s in out["negative"])    # long into RISK_OFF


def test_cache_round_trip():
    trades = [SimpleNamespace(r=0.5, regime="up") for _ in range(20)]
    prof = evaluate(trades, setup="trend_pullback", n_combos_tested=4, cfg=BacktestConfig())
    sym = "TESTCOIN/USDT:USDT"
    save_cache(Market.USDM, sym, "4h", {"trend_pullback": prof}, 1000)
    try:
        loaded = load_cached(Market.USDM, sym, "4h", ttl_hours=24.0)
        assert loaded is not None
        assert loaded["trend_pullback"]["overall"]["n"] == 20
        assert "up" in loaded["trend_pullback"]["by_regime"]
    finally:
        _cache_path(Market.USDM, sym, "4h").unlink(missing_ok=True)


def test_evaluate_conditioning_folds_are_chronological_not_input_order():
    # AUDIT FINDING 1 (conditioning path): a cell's fold-consistency must be computed on TIME
    # order — identical trades in reversed input order must yield the identical refinement.
    import src.edge_score as es
    early = [SimpleNamespace(r=+0.5, regime="up", entry_ts=1_000_000 + i * 3600,
                             context={"vol": "squeeze"}) for i in range(30)]
    late = [SimpleNamespace(r=-0.5, regime="up", entry_ts=2_000_000 + i * 3600,
                            context={"vol": "squeeze"}) for i in range(30)]
    rest = [SimpleNamespace(r=0.0, regime="up", entry_ts=1_500_000 + i * 3600,
                            context={"vol": "normal"}) for i in range(40)]
    a = es.evaluate_conditioning({"s": early + late + rest}, CFG, min_child_n=20)
    b = es.evaluate_conditioning({"s": list(reversed(early + late + rest))}, CFG, min_child_n=20)
    ra = next(r for r in a if r["feature"] == "vol" and r["value"] == "squeeze")
    rb = next(r for r in b if r["feature"] == "vol" and r["value"] == "squeeze")
    assert ra["fold_consistency"] == pytest.approx(rb["fold_consistency"])      # order-invariant
    assert ra["fold_consistency"] < 1.0                                          # early + / late − → not all folds +


def test_edge_gate_uses_effective_n_not_raw_pooled_n():
    # audit finding 2: 40 pooled trades that are only ~10 independent must be UNPROVEN
    v = {"regime": {"n": 40, "n_eff": 10.0, "expectancy": 0.35, "ci_low": 0.12}, "overall": None}
    lvl, msg = edge_gate(v, CFG)
    assert lvl == EDGE_UNPROVEN and "EFFECTIVE" in msg
    # same numbers with full independence stay proven (backward compat: n_eff absent → raw n)
    v2 = {"regime": {"n": 40, "expectancy": 0.35, "ci_low": 0.12}, "overall": None}
    assert edge_gate(v2, CFG)[0] == EDGE_PROVEN
