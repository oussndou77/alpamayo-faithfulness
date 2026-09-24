#!/usr/bin/env python3
"""
Cold tests for the uncertainty fine-tuning track — NO GPU, NO torch.

Covers: composite degradations (afh.degradation), the held-out clip/family/combo split
(afh.uncertainty_dataset), and the two-axis evaluation + clean non-regression report
(afh.eval_uncertainty).

Run from repo root:  python tests/test_uncertainty_cold.py   (or pytest)
"""

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from afh.degradation import (
    ALPHA_SPEED, BETA_LATERAL, BLEND_SEVERITY, DegradationSpec, FAMILIES, STOP_SEVERITY,
    UNCERTAINTY_LEVELS, apply_degradation,
    combo_key, compose, sample_composite_spec, target_text, target_trajectory,
    uncertainty_score,
)
from afh.uncertainty_dataset import (
    UncertaintyDataset, build_manifest, build_split, check_split, clean_cache_from_records,
    split_clips,
)
from afh.eval_uncertainty import (
    auroc, evaluate, format_report, spearman, trajectory_errors,
)

FIX = os.path.join(os.path.dirname(__file__), "..", "fixtures")


def _frames(seed=0, n_cam=7, n_t=4, h=16, w=16):
    return np.random.default_rng(seed).integers(20, 230, (n_cam, n_t, 3, h, w), dtype=np.uint8)


def _synthetic_cache(n=10, T=16):
    cache = {}
    for i in range(n):
        v = 5.0 + i
        xy = [[v * 0.1 * (t + 1), 0.05 * i * t] for t in range(T)]
        cache[f"clip_{i:02d}"] = {"t0_us": 5_100_000, "clean_reasoning": f"Keep lane {i}.",
                                  "future_xy": xy}
    return cache


# --------------------------------------------------------------------------- composites

def test_simple_spec_backward_compatible():
    f = _frames()
    old_style = {"family": "glare", "severity": 0.6, "cameras": [], "seed": 7, "params": {}}
    a, s = apply_degradation(f, DegradationSpec(**old_style))   # no `components` key
    b, _ = apply_degradation(f, DegradationSpec(**old_style))
    assert (a == b).all() and not (a == f).all()
    assert s.components == [] and combo_key(s) == "glare"
    assert s.to_dict()["components"] == []


def test_composite_on_different_cameras():
    f = _frames()
    spec = compose(("glare", 0.8, [1]), ("blur", 0.5, [0]), seed=3)
    out, spec = apply_degradation(f, spec)
    assert spec.family == "composite" and combo_key(spec) == "blur+glare"
    assert spec.cameras == [0, 1] and abs(spec.severity - 0.9) < 1e-9   # 1 - 0.2 * 0.5
    untouched = [c for c in range(7) if c not in (0, 1)]
    assert (out[untouched] == f[untouched]).all(), "other cameras must be untouched"
    # each camera matches its single-family degradation with the component's own seed
    g, _ = apply_degradation(f, DegradationSpec("glare", 0.8, [1], seed=spec.components[0]["seed"]))
    assert (out[1] == g[1]).all()
    text = target_text(spec)
    assert "glare" in text and "out of focus" in text
    assert uncertainty_score(text) >= 0.7


def test_composite_deterministic_by_seed():
    f = _frames(1)
    mk = lambda seed: compose(("occlusion", 0.6), ("noise", 0.7), seed=seed)
    a, sa = apply_degradation(f, mk(11))
    b, sb = apply_degradation(f, mk(11))
    c, _ = apply_degradation(f, mk(12))
    assert (a == b).all() and sa.to_dict() == sb.to_dict()
    assert not (a == c).all(), "different seed should give a different degradation"
    # hand-written components without seeds still derive them from the parent seed
    raw = DegradationSpec(family="composite", severity=0.7, seed=5, components=[
        {"family": "noise", "severity": 0.7, "seed": None}, {"family": "blur", "severity": 0.4}])
    raw2 = DegradationSpec(**json.loads(json.dumps(raw.to_dict())))
    assert (apply_degradation(f, raw)[0] == apply_degradation(f, raw2)[0]).all()
    # a spec stored in a manifest (JSON, cameras resolved) re-applies identically every time
    stored = json.dumps(sa.to_dict())
    r1, _ = apply_degradation(f, DegradationSpec(**json.loads(stored)))
    r2, _ = apply_degradation(f, DegradationSpec(**json.loads(stored)))
    assert (r1 == r2).all()


def test_composite_all_zero_is_clean_and_sampler_respects_exclusions():
    f = _frames()
    out, s = apply_degradation(f, compose(("blur", 0.0), ("glare", 0.0)))
    assert (out == f).all() and s.severity == 0.0 and combo_key(s) == "clean"
    for seed in range(50):
        s = sample_composite_spec(seed, families=("glare", "blur", "noise"),
                                  exclude_combos=["glare+blur", ("noise", "glare")])
        assert combo_key(s) == "blur+noise"
    try:
        sample_composite_spec(0, families=("glare", "blur"), exclude_combos=["blur+glare"])
        raise AssertionError("expected ValueError: no allowed combination")
    except ValueError:
        pass


def _level_index(text):
    """Index in UNCERTAINTY_LEVELS of the level voiced by a target text."""
    return [thr for thr, _ in UNCERTAINTY_LEVELS].index(uncertainty_score(text))


def test_composite_severity_is_noisy_or():
    f = _frames()
    spec = compose(("glare", 0.6, [1]), ("blur", 0.6, [0]), ("noise", 0.6, [2]), seed=1)
    assert spec.severity == 0.936, spec.severity                  # 1 - 0.4 ** 3
    _, spec = apply_degradation(f, spec)
    assert spec.severity == 0.936, "apply must keep the noisy-OR severity"
    single = DegradationSpec("glare", 0.6, [1], seed=1)
    apply_degradation(f, single)
    # 0.936 stays below the 0.95 "no usable visual input" level (coupled to STOP_SEVERITY)
    text = target_text(spec)
    assert "I cannot confirm the road ahead is clear" in text, text
    assert "no usable visual input" not in text and "controlled stop" not in text
    # escalation: three simultaneous faults voice exactly one level more than one fault
    assert _level_index(text) == _level_index(target_text(single)) + 1
    assert compose(("glare", 0.6), ("blur", 0.0)).severity == 0.6  # zero fault adds nothing


def test_composite_reaching_stop_severity():
    f = _frames()
    spec = compose(("glare", 0.65, [1]), ("blur", 0.65, [0]), ("noise", 0.65, [2]), seed=1)
    _, spec = apply_degradation(f, spec)
    assert spec.severity == 0.957125, spec.severity               # 1 - 0.35 ** 3
    assert spec.severity >= STOP_SEVERITY
    text = target_text(spec)
    assert "I have no usable visual input and cannot assess the scene" in text, text
    assert "controlled stop" in text
    assert uncertainty_score(text) == UNCERTAINTY_LEVELS[-1][0]
    # the target trajectory decelerates throughout and travels less than just below the
    # stop level; it reaches zero speed only at s = 1 (see test_target_trajectory_*)
    xy = np.stack([np.arange(1, 21) * 1.0, np.zeros(20)], 1)
    tgt = target_trajectory(xy, spec.severity)
    steps = np.diff(tgt[:, 0], prepend=0.0)
    assert (np.diff(steps) < 0).all()
    assert tgt[-1, 0] < target_trajectory(xy, 0.936)[-1, 0]


def _target_trajectory_main(xy_true, severity, alpha=ALPHA_SPEED, beta=BETA_LATERAL):
    """Frozen copy of target_trajectory as merged in #1 (hard switch at STOP_SEVERITY)."""
    xy = np.asarray(xy_true, dtype=float)
    s = float(np.clip(severity, 0, 1))
    if s <= 0:
        return xy.copy()
    steps = np.diff(xy[:, 0], prepend=0.0)
    if s >= STOP_SEVERITY:
        steps = steps * np.linspace(1.0, 0.0, xy.shape[0])
    else:
        steps = steps * (1.0 - alpha * s)
    return np.stack([np.cumsum(steps), xy[:, 1] * (1.0 - beta * s)], axis=1)


def _realistic_future(seed, T=64):
    """Forward-moving future with varying speed and a lateral drift (no reversing)."""
    rng = np.random.default_rng(seed)
    v = np.clip(8.0 + np.cumsum(rng.normal(0, 0.3, T)), 0.5, None) * 0.1
    return np.stack([np.cumsum(v), np.cumsum(rng.normal(0, 0.02, T))], 1)


def test_target_trajectory_distance_strictly_decreasing():
    grid = np.linspace(0.0, 1.0, 201)                 # step 0.005, includes 0.70 and 0.95
    assert np.isclose(grid, BLEND_SEVERITY).any() and np.isclose(grid, STOP_SEVERITY).any()
    for seed in range(5):
        xy = _realistic_future(seed)
        dist = np.array([target_trajectory(xy, s)[-1, 0] for s in grid])
        assert (np.diff(dist) < 0).all(), f"not strictly decreasing (seed {seed})"
        # continuous (a kink at s0, no jump): at s0 and at STOP_SEVERITY the distance
        # changes by O(eps) across +-eps. The merged version jumped UP at 0.95.
        eps = 1e-6
        for knot in (BLEND_SEVERITY, STOP_SEVERITY):
            lo, hi = target_trajectory(xy, knot - eps)[-1, 0], target_trajectory(xy, knot + eps)[-1, 0]
            assert 0 < lo - hi < 10 * eps * dist[0], (seed, knot, lo - hi)
        old_lo = _target_trajectory_main(xy, STOP_SEVERITY - eps)[-1, 0]
        old_hi = _target_trajectory_main(xy, STOP_SEVERITY + eps)[-1, 0]
        assert old_hi - old_lo > 0.01 * dist[0], "sanity: the check must catch the old jump"


def test_target_trajectory_stops_at_full_severity():
    for seed in range(5):
        xy = _realistic_future(seed)
        tgt = target_trajectory(xy, 1.0)
        steps = np.diff(tgt[:, 0], prepend=0.0)
        assert abs(steps[-1]) < 1e-12, "s = 1 must end at zero speed"
        assert (steps >= 0).all() and tgt[-1, 0] > 0
        assert np.allclose(tgt[:, 1], xy[:, 1] * (1 - BETA_LATERAL))  # lateral policy unchanged


def test_target_trajectory_unchanged_below_blend():
    grid = [s for s in np.round(np.linspace(0.0, 1.0, 201), 6) if s <= BLEND_SEVERITY]
    grid += [0.15, 0.333, 0.45, 0.6999]
    for seed in range(5):
        xy = _realistic_future(seed)
        for s in grid:
            np.testing.assert_array_equal(target_trajectory(xy, s),
                                          _target_trajectory_main(xy, s), err_msg=f"s={s}")


# --------------------------------------------------------------------------- split

def test_split_clips_disjoint_and_deterministic():
    ids = [f"c{i}" for i in range(20)]
    tr, te = split_clips(ids, 0.25, seed=0)
    tr2, te2 = split_clips(list(reversed(ids)), 0.25, seed=0)
    assert (tr, te) == (tr2, te2), "split must not depend on input order"
    assert not set(tr) & set(te) and set(tr) | set(te) == set(ids) and len(te) == 5
    assert split_clips(ids, 0.25, seed=1)[1] != te


def test_build_split_holds_out_clips_families_combos():
    cache = _synthetic_cache(10)
    sp = build_split(cache, test_fraction=0.3, seed=0, holdout_families=["desync"],
                     holdout_combos=["glare+blur"], n_per_clip=24, composite_fraction=0.5,
                     n_holdout_per_clip=2)
    train, test = sp["train"], sp["test"]
    assert not {e["clip_id"] for e in train} & {e["clip_id"] for e in test}
    assert all("desync" not in combo_key(e["spec"]).split("+") for e in train)
    assert all(combo_key(e["spec"]) != "blur+glare" for e in train)
    assert any("+" in e["combo"] for e in train), "train should contain composites"
    # glare and blur are still seen individually in train (only the COMBO is held out)
    train_combos = {e["combo"] for e in train}
    assert "glare" in train_combos or any("glare" in c for c in train_combos)
    # every test clip has explicit examples of each held-out item
    for cid in sp["meta"]["test_clips"]:
        mine = [e for e in test if e["clip_id"] == cid]
        assert sum(e["combo"] == "desync" for e in mine) >= 2
        assert sum(e["combo"] == "blur+glare" for e in mine) >= 2
    assert any(e["heldout_combo"] for e in test) and any(e["heldout_family"] for e in test)
    assert all(e["split"] == "test" for e in test)
    # deterministic
    sp2 = build_split(cache, test_fraction=0.3, seed=0, holdout_families=["desync"],
                      holdout_combos=["glare+blur"], n_per_clip=24, composite_fraction=0.5)
    assert json.dumps(sp2["train"]) == json.dumps(train)


def test_check_split_detects_leakage():
    cache = _synthetic_cache(4)
    sp = build_split(cache, test_fraction=0.5, seed=0, n_per_clip=4)
    try:
        check_split(sp["train"], sp["train"][:1] + sp["test"])
        raise AssertionError("clip leakage not detected")
    except ValueError:
        pass
    leaked = dict(sp["train"][0], spec=compose(("glare", .5), ("blur", .5)).to_dict())
    try:
        check_split([leaked], sp["test"], holdout_combos=["blur+glare"])
        raise AssertionError("held-out combo leakage not detected")
    except ValueError:
        pass


def test_legacy_manifest_unchanged_and_dataset_loads_composites():
    cache = clean_cache_from_records(os.path.join(FIX, "records_a2.json"),
                                     os.path.join(FIX, "raw_diag_a2.json"))
    a = build_manifest(cache, n_per_clip=6, seed=0)
    assert all(e["family"] != "composite" for e in a)
    m = build_manifest(cache, n_per_clip=12, seed=0, composite_fraction=1.0)
    comps = [e for e in m if e["family"] == "composite"]
    assert comps, "composite_fraction=1 should produce composites"
    ds = UncertaintyDataset(comps, frame_loader=lambda cid, t0: (_frames(), {"clip": cid}))
    x1, x2 = ds[0], ds[0]
    assert x1["frames"].shape == (7, 4, 3, 16, 16) and (x1["frames"] == x2["frames"]).all()
    assert x1["spec"]["components"]


# --------------------------------------------------------------------------- evaluation

def test_stats_helpers():
    assert auroc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0
    assert auroc([0.5, 0.5, 0.5, 0.5], [0, 1, 0, 1]) == 0.5
    assert auroc([1, 2], [1, 1]) is None
    assert abs(spearman([1, 2, 3, 4], [10, 20, 30, 40]) - 1.0) < 1e-9
    assert spearman([1, 1, 1], [1, 2, 3]) is None
    e = trajectory_errors([[[0, 0], [1, 0]], [[0, 1], [1, 1]]], [[0, 0], [1, 0]])
    assert e["min_ade"] == 0.0 and abs(e["ade"] - 0.5) < 1e-9 and e["fde"] == 0.5


_TEXT = {0.0: "The road ahead is clearly visible.",
         0.45: "Visibility ahead is reduced.",
         0.7: "I cannot confirm the road ahead is clear.",
         0.95: "I have no usable visual input and cannot assess the scene."}


def _eval_records(policy: str, seed=0, n_clips=12):
    """
    Synthetic results. `policy`:
      calibrated — error grows with a hidden per-example difficulty, and so does the voiced
                   uncertainty;
      vocabulary — voices maximal uncertainty on EVERY degraded input, error unrelated;
      damped     — calibrated words, but crawls (plan far from the true future);
      baseline   — pre-fine-tune: always "clearly visible", moderate error.
    """
    rng = np.random.default_rng(seed)
    recs = []
    for i in range(n_clips):
        T = 20
        gt = np.stack([np.arange(1, T + 1) * 1.0, np.zeros(T)], 1)
        clean_err = 0.2 + 1.0 * rng.random()
        u_clean = 0.0
        recs.append({"clip_id": f"c{i}", "family": "clean", "severity": 0.0,
                     "text": _TEXT[u_clean], "gt_xy": gt.tolist(),
                     "pred_xy": (gt + [0, clean_err]).tolist(), "faithfulness": 0.8})
        for j, (combo, ho) in enumerate([("glare", False), ("blur", False),
                                         ("blur+glare", True), ("desync", False)]):
            sev = float(rng.uniform(0.2, 1.0))
            difficulty = float(rng.random())
            err = 3.0 * difficulty
            u = [k for k in (0.0, 0.45, 0.7, 0.95) if difficulty >= k * 0.9][-1]
            pred = gt + [0, err]
            if policy == "vocabulary":
                u = 0.95
            elif policy == "damped":
                pred = np.stack([gt[:, 0] * 0.3, gt[:, 1] + err], 1)
            elif policy == "baseline":
                u, pred = 0.0, gt + [0, err + 0.2]
            recs.append({"clip_id": f"c{i}", "combo": combo, "severity": sev,
                         "condition": f"{combo}#{j}", "text": _TEXT[u],
                         "gt_xy": gt.tolist(), "pred_xy": pred.tolist(),
                         "heldout_combo": ho, "heldout_family": combo == "desync"})
    return recs


def test_calibration_separates_signal_from_vocabulary():
    cal = evaluate(_eval_records("calibrated"))["calibration"]["slices"]["all_degraded"]
    voc = evaluate(_eval_records("vocabulary"))["calibration"]["slices"]["all_degraded"]
    assert cal["auroc_uncertainty_high_error"] > 0.85, cal
    assert cal["spearman_uncertainty_ade"] > 0.7, cal
    # cautious vocabulary everywhere carries no information about error
    assert voc["auroc_uncertainty_high_error"] == 0.5 and voc["spearman_uncertainty_ade"] is None
    assert voc["mean_uncertainty"] > cal["mean_uncertainty"], "vocab model SOUNDS more cautious"


def test_behavior_is_independent_of_calibration():
    base = _eval_records("baseline")
    good = evaluate(_eval_records("calibrated"), base)
    damp = evaluate(_eval_records("damped"), base)
    # same words, same error RANKING: calibration (rank correlation) is identical ...
    g_c = good["calibration"]["slices"]["all_degraded"]["spearman_uncertainty_ade"]
    d_c = damp["calibration"]["slices"]["all_degraded"]["spearman_uncertainty_ade"]
    assert g_c > 0.85 and abs(g_c - d_c) < 1e-9
    # (damping makes EVERY case high-error, so its AUROC is undefined rather than good)
    assert damp["calibration"]["slices"]["all_degraded"]["auroc_uncertainty_high_error"] is None
    # ... but damping moves the plan away from the true future: behavior axis catches it
    g_b, d_b = good["behavior"]["slices"]["all_degraded"], damp["behavior"]["slices"]["all_degraded"]
    assert g_b["ade_delta_vs_baseline"] < 0 < d_b["ade_delta_vs_baseline"]
    assert "closed-loop" in damp["behavior"]["not_measured"]
    for k in ("heldout_combo", "heldout_family", "in_distribution", "composite"):
        assert k in good["calibration"]["slices"] and k in good["behavior"]["slices"]


def test_clean_regression_report():
    base = _eval_records("baseline")
    ok = evaluate(_eval_records("calibrated"), base)["clean_regression"]
    assert ok["n_paired_clips"] == 12 and ok["ade_verdict"] == "pass", ok
    assert ok["false_alarm_verdict"] == "pass" and abs(ok["faithfulness_delta"]) < 1e-9
    # a fine-tune that drifts on clean input and voices alarms on it is flagged
    bad = _eval_records("calibrated")
    for r in bad:
        if r.get("family") == "clean":
            r["pred_xy"] = (np.asarray(r["gt_xy"]) + [0, 1.0]).tolist()
            r["text"] = _TEXT[0.7]
    rep = evaluate(bad, base)["clean_regression"]
    assert rep["ade_verdict"] == "regression" and rep["false_alarm_verdict"] == "regression"
    assert "not assessed" in evaluate(bad)["clean_regression"]["note"]
    assert "CLEAN NON-REGRESSION" in format_report(evaluate(bad, base))


def test_end_to_end_split_to_eval_on_fixtures():
    """Fixture clips -> held-out split -> fake 'inference' on the test manifest -> report."""
    cache = clean_cache_from_records(os.path.join(FIX, "records_a2.json"),
                                     os.path.join(FIX, "raw_diag_a2.json"))
    sp = build_split(cache, test_fraction=0.34, seed=0, holdout_families=["desync"],
                     holdout_combos=["glare+blur"], n_per_clip=8, composite_fraction=0.3)
    recs = []
    for e in sp["test"]:
        gt = cache[e["clip_id"]]["future_xy"]
        recs.append({**e, "text": e["target_text"], "gt_xy": gt, "pred_xy": e["target_xy"]})
    rep = evaluate(recs)
    assert rep["n_clips"] == len(sp["meta"]["test_clips"]) == 1
    assert "heldout_combo" in rep["calibration"]["slices"]
    # the damped TARGET itself deviates from the true future as severity grows
    beh = rep["behavior"]["slices"]["all_degraded"]
    assert beh["ade_delta_vs_clean"] > 0
    assert format_report(rep)


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        try:
            t(); print(f"  ok  {t.__name__}"); passed += 1
        except AssertionError as e:
            print(f"  XX  {t.__name__} - FAILED: {e}")
        except Exception as e:
            print(f"  XX  {t.__name__} - ERROR: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(tests)} tests passed")
    return passed == len(tests)


if __name__ == "__main__":
    print("=== Cold uncertainty-track tests (no GPU, no torch) ===")
    sys.exit(0 if _run_all() else 1)
