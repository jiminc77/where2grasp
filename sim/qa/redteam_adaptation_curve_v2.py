"""Independent red-team for Item-1b C2-v2 (the GRADUATED, frozen-bank-executed adaptation curve).

FRESH QA-local recompute (no production scorers/stored scalars trusted for the verdict). Adds the
decisive v2 check the v1 as-run could not pass: the FROZEN new selection {2300-2302} + evaluation
{3300-3304} banks were EXECUTED EXACTLY this phase, and BOTH the teacher labels and the oracle were
built from THOSE draws (internally consistent). Also: C1-v2 single-file + git-ancestry of C2-v2;
every committed scalar + A-16 band recomputed from the persisted raw surfaces; exact-key sweep +
history inventory; ratio invariance + band IoU; NO performance gate (curve shape DESCRIPTIVE only);
pre_registered_outcome == ESTABLISHED.

SURVIVES == every check passes.  Run:  python -m sim.qa.redteam_adaptation_curve_v2
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MAN = ROOT / "manifests"

C1V2_FILE = "sim/manifests/adaptation_curve_manifest_v2.json"
C1V2_COMMIT = "138a1ac"

checks: list[tuple[str, bool, str]] = []


def ck(name, ok, detail=""):
    checks.append((name, bool(ok), detail))


def qa_map_rmse(pred, meas):
    p = np.asarray(pred, float); m = np.asarray(meas, float)
    return float(np.sqrt(np.mean((p - m) ** 2)))


def qa_tau_cross(curve, ells, tau=0.5):
    y = np.asarray(curve, float); g = np.asarray(ells, float)
    for i in range(len(y) - 1):
        if (y[i] - tau) * (y[i + 1] - tau) <= 0 and y[i] != y[i + 1]:
            t = (tau - y[i]) / (y[i + 1] - y[i]); return (g[i] + t * (g[i + 1] - g[i]) - g[0]) / (g[1] - g[0])
    return None


def qa_boundary_err(pred, meas, ells, tau=0.5):
    cp, cm = qa_tau_cross(pred, ells, tau), qa_tau_cross(meas, ells, tau)
    return None if (cp is None or cm is None) else float(abs(cp - cm))


def qa_selection_regret(pred, oracle):
    o = np.asarray(oracle, float); p = np.asarray(pred, float)
    return float(o[int(np.argmax(o))] - o[int(np.argmax(p))])


def qa_band_iou(pred, meas, tau=0.5):
    P = set(np.where(np.asarray(pred) >= tau)[0].tolist()); M = set(np.where(np.asarray(meas) >= tau)[0].tolist())
    return 1.0 if (not P and not M) else (float(len(P & M) / len(P | M)) if (P | M) else 1.0)


def qa_aggregate(vals):
    v = np.asarray([x for x in vals if x is not None], float)
    return float("nan") if v.size == 0 else float(v.mean())


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    import sim.adaptation_curve as ac

    man = json.loads((MAN / "adaptation_curve_manifest_v2.json").read_text())
    res = json.loads((MAN / "adaptation_curve_v2_results.json").read_text())
    curve = res["curve"]
    live_c1 = sha256(MAN / "adaptation_curve_manifest_v2.json")

    # 1) C1-v2 single-file freeze + git-ancestry of the C2-v2 data commit --- #
    try:
        stat = subprocess.run(["git", "log", "--oneline", "-1", "--stat", C1V2_COMMIT], cwd=ROOT.parent, capture_output=True, text=True, timeout=30).stdout
        touched = [ln for ln in stat.splitlines() if "|" in ln]
        single = len(touched) == 1 and "adaptation_curve_manifest_v2.json" in touched[0]
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT.parent, capture_output=True, text=True, timeout=30).stdout.strip()
        anc = subprocess.run(["git", "merge-base", "--is-ancestor", C1V2_COMMIT, head], cwd=ROOT.parent, capture_output=True, timeout=30).returncode == 0
        ck("C1-v2 single-file freeze", single, touched[0].strip() if touched else "?")
        ck("C1-v2 is git-ancestor of C2-v2 (HEAD)", anc, f"{C1V2_COMMIT}..{head[:8]}")
    except Exception as e:  # noqa: BLE001
        ck("C1-v2 git ancestry", False, f"git error: {e}")

    # 2) input-artifact digests still match the v2 freeze -------------------- #
    dig = man["input_artifact_sha256"]; bad = [f for f, h in dig.items() if not (MAN / f).exists() or sha256(MAN / f) != h]
    ck("v2 input-artifact digests match freeze", not bad, f"bad={bad}" if bad else f"{len(dig)} files OK")
    ck("results content-bound to C1-v2 sha", res.get("c1_v2_manifest_sha256") == live_c1, f"c1_sha={live_c1[:12]}")

    # 3) THE DECISIVE v2 CHECK: the FROZEN sel/eval banks were EXECUTED EXACTLY #
    sw = np.load(MAN / "adaptation_curve_v2_sweep_results.npz", allow_pickle=True)
    assert str(sw["manifest_digest"].item()) == live_c1, "v2 sweep not bound to v2 manifest"
    bk = sw["bank"].astype(str); seed = sw["seed"].astype(int)
    sel_seeds = sorted(set(seed[bk == "selection"].tolist())); ev_seeds = sorted(set(seed[bk == "evaluation"].tolist()))
    ck("FROZEN selection bank {2300-2302} executed", sel_seeds == sorted(ac.SEED_BANKS["selection"]), f"sel_seeds={sel_seeds}")
    ck("FROZEN evaluation bank {3300-3304} executed", ev_seeds == sorted(ac.SEED_BANKS["evaluation"]), f"eval_seeds={ev_seeds}")
    # oracle (v2 landscape) is the evaluation-bank mean success per (setting, cell); recompute + byte-match
    land = {x["id"]: x for x in json.loads((MAN / "adaptation_curve_v2_landscape.json").read_text())["settings"]}
    st = sw["setting"].astype(str); gr = sw["grasp"].astype(int); suc = sw["success"].astype(float)
    ng = len(json.loads((MAN / man["sweep_grid_source"]).read_text())["grasp"]["ell"])
    orc_bad = []
    for sid in ac.TEST_GROUPS:
        for gi in range(ng):
            qa = float(np.mean(suc[(st == sid) & (gr == gi) & (bk == "evaluation")]))
            if abs(qa - land[sid]["success_rate"][gi]) > 1e-9:
                orc_bad.append((sid, gi))
    ck("v2 ORACLE recomputed from evaluation bank (byte-match)", not orc_bad, f"bad={orc_bad[:3]}" if orc_bad else "all TEST oracle cells reproduced from the eval-bank draws")

    # 4) EXACT-KEY sweep inventory (no thinning) ----------------------------- #
    from sim.sweep import settings as _settings
    m = json.loads((MAN / man["sweep_grid_source"]).read_text()); ssids = [s["id"] for s in _settings(m)]
    tmpl = sw["template"].astype(int)
    selc = Counter(zip(st[bk == "selection"].tolist(), gr[bk == "selection"].tolist(), tmpl[bk == "selection"].tolist(), seed[bk == "selection"].tolist()))
    want_sel = Counter((s, g, t, sd) for s in ssids for g in range(ng) for t in range(len(m["templates"])) for sd in ac.SEED_BANKS["selection"])
    ck("selection sweep EXACT Cartesian (15x17x4x3)", selc == want_sel, f"rows={sum(selc.values())} want={sum(want_sel.values())}")
    # evaluation: exactly len(eval_seeds) rows per (setting, cell), all at one winner template
    eval_ok = all(int(np.sum((st == s) & (gr == g) & (bk == "evaluation"))) == len(ac.SEED_BANKS["evaluation"])
                  and len(set(tmpl[(st == s) & (gr == g) & (bk == "evaluation")].tolist())) == 1 for s in ssids for g in range(ng))
    ck("evaluation sweep exact (5 eval-seeds per cell at single winner template)", eval_ok, "per (setting,cell): 5 eval rows, one template")

    # 5) histories REUSED unchanged (identical bank + exact 2040 Cartesian) -- #
    hz = np.load(MAN / "adaptation_curve_lift_histories.npz", allow_pickle=True)
    Sh = hz["setting"].astype(str); Gh = hz["grasp"].astype(int); Th = hz["template"].astype(int); Dh = hz["seed"].astype(int)
    universe = list(ac.TRAIN_GROUPS + ac.VAL_GROUPS + ac.TEST_GROUPS)
    goth = Counter(zip(Sh.tolist(), Gh.tolist(), Th.tolist(), Dh.tolist()))
    wanth = Counter((s, g, t, sd) for s in universe for g in range(ac.LIFT_N_CELLS) for t in range(ac.N_TEMPLATES) for sd in ac.SEED_BANKS["history"])
    ck("history npz reused, EXACT 2040 Cartesian", goth == wanth and sha256(MAN / "adaptation_curve_lift_histories.npz") == dig["adaptation_curve_lift_histories.npz"],
       f"rows={sum(goth.values())} + digest-pinned")

    # 6) schedule + k=0==blind + estimator -------------------------------- #
    qa_sl = [((7 * j) % ac.LIFT_N_CELLS, round(0.12 + ac.LIFT_ELL_STEP * ((7 * j) % ac.LIFT_N_CELLS), 2), j % 4) for j in range(ac.N_SCHEDULE_STEPS)]
    ck("S_lift regen byte-match", qa_sl == ac.schedule_lift(), f"len={len(qa_sl)}")
    ck("k=0 estimator None (-> blind)", ac.prefix_summary(np.zeros((3, ac.FEATURE_DIM)), 0) is None, "prefix_summary(x,0) None")

    # 7) INDEPENDENT RECOMPUTE of every committed scalar + A-16 band from raw surfaces #
    surf = np.load(MAN / "adaptation_curve_v2_surfaces.npz", allow_pickle=True)
    ell = np.array(m["grasp"]["ell"], float)
    meas = {sid: np.asarray(surf[f"measured__{sid}"], float) for sid in ac.TEST_GROUPS}
    tseeds = surf["train_seeds"].tolist()
    field_fns = {"map_rmse": lambda c, sid: qa_map_rmse(c, meas[sid]),
                 "selection_regret": lambda c, sid: qa_selection_regret(c, meas[sid]),
                 "boundary_err": lambda c, sid: qa_boundary_err(c, meas[sid], ell),
                 "band_iou": lambda c, sid: qa_band_iou(c, meas[sid])}

    def qa_band(b, ki, fn):
        per = []
        for si in range(len(tseeds)):
            per.append(qa_aggregate([fn(surf[f"{b}__k{ac.K_AXIS[ki]}__{sid}"][si], sid) for sid in ac.TEST_GROUPS]))
        arr = np.asarray(per, float)
        if np.all(np.isnan(arr)):
            return dict(mean=None, seed_std=None, n_seeds=len(tseeds))
        return dict(mean=float(np.nanmean(arr)), seed_std=float(np.nanstd(arr)), n_seeds=len(tseeds))

    def near(a, b):
        if (a is None) != (b is None):
            return False
        return a is None or abs(a - b) <= 1e-9
    mism = []
    for b in ("teacher", "blind", "student", "sysid"):
        for field, fn in field_fns.items():
            for ki in range(len(ac.K_AXIS)):
                qa = qa_band(b, ki, fn); rep = curve[b][field][ki]
                if not near(qa["mean"], rep["mean"]) or not near(qa["seed_std"], rep.get("seed_std")) or qa["n_seeds"] != rep.get("n_seeds"):
                    mism.append(f"{b}.{field}.k{ac.K_AXIS[ki]}")
    ck("every committed scalar+A16 band recomputed from raw surfaces (1e-9)", not mism, f"mismatch {mism[:3]}" if mism else "4 baselines x 4 metrics x 7 k: mean+seed_std+n_seeds reproduced")
    k0_surf = all(np.array_equal(surf[f"student__k0__{sid}"], surf[f"blind__k0__{sid}"]) for sid in ac.TEST_GROUPS)
    ck("k=0 student surfaces == blind (membership)", k0_surf, "k0 predicted curves identical to blind")

    # 8) ratio invariance + band IoU emitted + recomputed -------------------- #
    ri = res.get("ratio_invariance", {}); ref_map = {"R0": "B1_w1", "R1": "B3_w2", "R2": "B2_w1"}
    ri_ok = all(rid in ri and ri[rid]["reference"] == ref and
                ri[rid]["offset_cells"] == abs(int(np.argmax(land[rid]["success_rate"])) - int(np.argmax(land[ref]["success_rate"]))) for rid, ref in ref_map.items())
    ck("ratio invariance emitted + recomputed (on v2 oracle)", ri_ok, f"offsets={{r: ri[r]['offset_cells'] for r in ri}}" if ri else "absent")
    ck("band IoU emitted for all baselines", all("band_iou" in curve[b] and len(curve[b]["band_iou"]) == len(ac.K_AXIS) for b in curve), "band_iou at every k")

    # 9) outcome + honesty --------------------------------------------------- #
    ck("pre-registered outcome ESTABLISHED (frozen banks executed exactly)", res.get("pre_registered_outcome") == "ESTABLISHED" and res.get("protocol_fidelity", {}).get("executed_exactly") is True, "v2 executed the frozen sel/eval banks exactly")
    ck("lift selection-regret degeneracy reported", res.get("lift_selection_regret_degenerate") is True, "committed degeneracy reported honestly")

    stu = [c["mean"] for c in curve["student"]["map_rmse"]]
    tea = curve["teacher"]["map_rmse"][0]["mean"]; bli = curve["blind"]["map_rmse"][0]["mean"]
    print(json.dumps(dict(DESCRIPTIVE_non_gating=dict(student_map_rmse=[round(x, 4) for x in stu], teacher_ceiling=round(tea, 4), blind=round(bli, 4),
                     note="reported for orientation ONLY; performance + monotonicity NEVER gate"))))

    npass = sum(1 for _, ok, _ in checks if ok)
    verdict = "SURVIVES" if npass == len(checks) else "FAILED"
    print(json.dumps(dict(verdict=verdict, passed=npass, total=len(checks),
                          checks=[dict(name=nm, ok=ok, detail=d) for nm, ok, d in checks]), indent=2))
    return 0 if verdict == "SURVIVES" else 1


if __name__ == "__main__":
    raise SystemExit(main())
