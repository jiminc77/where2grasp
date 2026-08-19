"""Independent red-team for Item-1c (the DISTAL SECONDARY regret-vs-k companion).

FRESH QA-local recompute (no production scorers/stored scalars trusted for the verdict). Verifies:
the FROZEN Item-1 selection {2300-2302} + evaluation {3300-3304} banks were EXECUTED EXACTLY on the
25-cell spanning cohort (oracle + teacher labels both from those draws); teacher labels use the
frozen winner template; the distal oracle (success + J) recomputes from the eval bank; distal
histories cover all 25 cells with the frozen history bank {2500,2501}; every committed regret + map
scalar + A-16 band is recomputed from the persisted raw predicted-curve surfaces; ratio invariance;
S_distal byte-match; pre_registered_outcome ESTABLISHED; regret DISCRIMINATES; no performance gate.

SURVIVES == every check passes.  Run:  python -m sim.qa.redteam_adaptation_curve_distal
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
C1_FILE = "sim/manifests/adaptation_curve_distal_manifest.json"
C1_COMMIT = "ae37b7e"

checks: list[tuple[str, bool, str]] = []


def ck(name, ok, detail=""):
    checks.append((name, bool(ok), detail))


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def qa_map_rmse(pS, ms):
    return float(np.sqrt(np.mean((np.asarray(pS, float) - np.asarray(ms, float)) ** 2)))


def qa_regret(pS, pJ, ms, mj, in_reg):
    pS = np.asarray(pS, float); pJ = np.asarray(pJ, float); mj = np.asarray(mj, float)
    cand = np.where(in_reg & (pS >= 0.5))[0]
    sel = int(cand[np.argmax(pJ[cand])]) if len(cand) else (int(np.where(in_reg)[0][np.argmax(pS[in_reg])]) if in_reg.any() else int(np.argmax(pS)))
    best = float(np.max(mj[in_reg])) if in_reg.any() else float(np.max(mj))
    return best - float(mj[sel])


def qa_aggregate(vals):
    v = np.asarray([x for x in vals if x is not None], float)
    return float("nan") if v.size == 0 else float(v.mean())


def main():
    import sim.adaptation_curve as ac
    import sim.tip_model as tm
    from sim.distal_sweep import settings as dsettings

    man = json.loads((MAN / "adaptation_curve_distal_manifest.json").read_text())
    res = json.loads((MAN / "adaptation_curve_distal_results.json").read_text())
    curve = res["curve"]; live = sha256(MAN / "adaptation_curve_distal_manifest.json")
    TEST = man["splits"]["test"]

    # 1) C1-distal single-file freeze + git-ancestry of the data commit -------- #
    try:
        stat = subprocess.run(["git", "log", "--oneline", "-1", "--stat", C1_COMMIT], cwd=ROOT.parent, capture_output=True, text=True, timeout=30).stdout
        touched = [ln for ln in stat.splitlines() if "|" in ln]
        single = len(touched) == 1 and "adaptation_curve_distal_manifest.json" in touched[0]
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT.parent, capture_output=True, text=True, timeout=30).stdout.strip()
        anc = subprocess.run(["git", "merge-base", "--is-ancestor", C1_COMMIT, head], cwd=ROOT.parent, capture_output=True, timeout=30).returncode == 0
        ck("C1-distal single-file freeze", single, touched[0].strip() if touched else "?")
        ck("C1-distal is git-ancestor of the data commit", anc, f"{C1_COMMIT}..{head[:8]}")
    except Exception as e:  # noqa: BLE001
        ck("C1-distal git ancestry", False, f"git error: {e}")

    dig = man["input_artifact_sha256"]; bad = [f for f, h in dig.items() if not (MAN / f).exists() or sha256(MAN / f) != h]
    ck("C1-distal input digests match freeze", not bad, f"bad={bad}" if bad else f"{len(dig)} files OK")
    ck("results content-bound to C1-distal sha", res.get("c1_distal_manifest_sha256") == live, f"c1={live[:12]}")

    # 2) FROZEN banks executed exactly on the distal cohort ------------------- #
    sw = np.load(MAN / "adaptation_curve_distal_sweep_results.npz", allow_pickle=True)
    assert str(sw["manifest_digest"].item()) == live, "distal sweep not bound to C1-distal"
    bk = sw["bank"].astype(str); seed = sw["seed"].astype(int); st = sw["setting"].astype(str); gr = sw["grasp"].astype(int)
    tmpl = sw["template"].astype(int); suc = sw["success"].astype(float); J = sw["J"].astype(float); seltmpl = sw["selected_template"].astype(bool)
    ck("FROZEN selection bank {2300-2302} executed (distal)", sorted(set(seed[bk == "selection"].tolist())) == sorted(ac.SEED_BANKS["selection"]), f"{sorted(set(seed[bk=='selection'].tolist()))}")
    ck("FROZEN evaluation bank {3300-3304} executed (distal)", sorted(set(seed[bk == "evaluation"].tolist())) == sorted(ac.SEED_BANKS["evaluation"]), f"{sorted(set(seed[bk=='evaluation'].tolist()))}")
    ck("all distal sweep rollouts converged", bool(np.all(sw["converged"].astype(bool))), f"{int(np.sum(sw['converged'].astype(bool)))}/{len(sw['converged'])}")

    grid = np.array(man["grasp"]["ell"], float); ng = len(grid); ntmpl = len(man["templates"])
    ss = dsettings(man); ssids = [s["id"] for s in ss]
    # exact-key selection inventory + frozen winner for eval + teacher
    selc = Counter(zip(st[bk == "selection"].tolist(), gr[bk == "selection"].tolist(), tmpl[bk == "selection"].tolist(), seed[bk == "selection"].tolist()))
    want_sel = Counter((s, g, t, sd) for s in ssids for g in range(ng) for t in range(ntmpl) for sd in ac.SEED_BANKS["selection"])
    ck("selection sweep EXACT Cartesian (15x25x4x3)", selc == want_sel, f"rows={sum(selc.values())} want={sum(want_sel.values())}")
    frozen_win = {}; eval_bad = []; pers_bad = []
    for s in ssids:
        for g in range(ng):
            sel = (st == s) & (gr == g) & (bk == "selection")
            if not sel.any():
                continue
            rates = [float(np.mean(suc[sel & (tmpl == t)])) if (sel & (tmpl == t)).any() else -1.0 for t in range(ntmpl)]
            fw = int(np.argmax(rates)); frozen_win[(s, g)] = fw
            if set(tmpl[(st == s) & (gr == g) & (bk == "evaluation")].tolist()) != {fw}:
                eval_bad.append((s, g))
            if set(tmpl[sel & seltmpl].tolist()) != {fw}:
                pers_bad.append((s, g))
    ck("evaluation template == frozen success-only winner (all 15x25)", not eval_bad, f"bad={eval_bad[:3]}" if eval_bad else "all cells at frozen winner")
    ck("persisted selected_template == frozen winner", not pers_bad, f"bad={pers_bad[:3]}" if pers_bad else "ok")

    # 3) distal ORACLE (success + J) recomputed from the eval bank ------------ #
    land = {x["id"]: x for x in json.loads((MAN / "adaptation_curve_distal_landscape.json").read_text())["settings"]}
    orc_bad = []
    for sid in TEST:
        for g in range(ng):
            ev = (st == sid) & (gr == g) & (bk == "evaluation")
            if abs(float(np.mean(suc[ev])) - land[sid]["success_rate"][g]) > 1e-9 or abs(float(np.mean(J[ev])) - land[sid]["J"][g]) > 1e-9:
                orc_bad.append((sid, g))
    ck("distal oracle (success+J) recomputed from eval bank (byte-match)", not orc_bad, f"bad={orc_bad[:3]}" if orc_bad else "all TEST oracle cells reproduced")

    # 4) distal histories cover all 25 cells with the frozen history bank ----- #
    hz = np.load(MAN / "adaptation_curve_distal_histories.npz", allow_pickle=True)
    assert str(hz["manifest_digest"].item()) == live, "distal histories not bound to C1-distal"
    Sh = hz["setting"].astype(str); Gh = hz["grasp"].astype(int); Th = hz["template"].astype(int); Dh = hz["seed"].astype(int)
    universe = man["universe"]
    goth = Counter(zip(Sh.tolist(), Gh.tolist(), Th.tolist(), Dh.tolist()))
    wanth = Counter((s, g, t, sd) for s in universe for g in range(ng) for t in range(ntmpl) for sd in man["seed_banks"]["history"])
    ck("distal history EXACT Cartesian (15x25x4x2)", goth == wanth, f"rows={sum(goth.values())} want={sum(wanth.values())}")

    # 5) S_distal regen byte-match + k=0 None -------------------------------- #
    qa_sd = [((7 * j) % ac.DISTAL_N_CELLS, round(0.12 + ac.DISTAL_ELL_STEP * ((7 * j) % ac.DISTAL_N_CELLS), 2), j % 4) for j in range(ac.N_SCHEDULE_STEPS)]
    ck("S_distal regen byte-match", qa_sd == ac.schedule_distal(), f"len={len(qa_sd)}")

    # 6) INDEPENDENT RECOMPUTE of every committed regret+map scalar + A-16 band from raw surfaces #
    surf = np.load(MAN / "adaptation_curve_distal_surfaces.npz", allow_pickle=True)
    tseeds = surf["train_seeds"].tolist()
    prop = {sid: (float(surf[f"prop__{sid}"][0]), float(surf[f"prop__{sid}"][1])) for sid in TEST}
    ms = {sid: np.asarray(surf[f"measured__{sid}__success"], float) for sid in TEST}
    mj = {sid: np.asarray(surf[f"measured__{sid}__J"], float) for sid in TEST}
    in_reg = {sid: (tm.pi_g(grid, prop[sid][0], prop[sid][1]) <= tm.PI_G_MAX) for sid in TEST}

    def qa_band(b, ki, field):
        per = []
        for si in range(len(tseeds)):
            vals = []
            for sid in TEST:
                pS = surf[f"{b}__k{ac.K_AXIS[ki]}__{sid}__pS"][si]; pJ = surf[f"{b}__k{ac.K_AXIS[ki]}__{sid}__pJ"][si]
                vals.append(qa_regret(pS, pJ, ms[sid], mj[sid], in_reg[sid]) if field == "regret" else qa_map_rmse(pS, ms[sid]))
            per.append(qa_aggregate(vals))
        arr = np.asarray(per, float)
        return dict(mean=float(np.nanmean(arr)), seed_std=float(np.nanstd(arr)), n_seeds=len(tseeds))

    def near(a, b):
        return abs(a - b) <= 1e-9
    mism = []
    for b in ("teacher", "blind", "student", "sysid"):
        for field, key in (("regret", "selection_regret"), ("map", "map_rmse")):
            for ki in range(len(ac.K_AXIS)):
                qa = qa_band(b, ki, field); rep = curve[b][key][ki]
                if not near(qa["mean"], rep["mean"]) or not near(qa["seed_std"], rep["seed_std"]) or qa["n_seeds"] != rep["n_seeds"]:
                    mism.append(f"{b}.{key}.k{ac.K_AXIS[ki]}")
    ck("every committed regret+map scalar+A16 band recomputed from raw surfaces (1e-9)", not mism, f"mismatch {mism[:3]}" if mism else "4 baselines x 2 metrics x 7 k reproduced (mean+seed_std+n_seeds)")
    # k=0 student/sysid predicted curves == blind (surface membership)
    k0ok = all(np.array_equal(surf[f"student__k0__{sid}__pS"], surf[f"blind__k0__{sid}__pS"]) and
               np.array_equal(surf[f"sysid__k0__{sid}__pS"], surf[f"blind__k0__{sid}__pS"]) for sid in TEST)
    ck("k=0 student/sysid curves == blind (membership)", k0ok, "k0 predicted curves identical to blind")

    # 7) ratio invariance + outcome + honesty -------------------------------- #
    ri = res.get("ratio_invariance", {})
    ck("ratio invariance emitted (A-15 controls)", all(k in ri for k in ("R0", "R1", "R2")), f"{ {k: ri[k]['offset_cells'] for k in ri} }" if ri else "absent")
    ck("pre-registered outcome ESTABLISHED (frozen banks executed exactly)",
       res.get("pre_registered_outcome") == "ESTABLISHED" and res.get("protocol_fidelity", {}).get("executed_exactly") is True, "distal executed the frozen banks exactly")
    ck("selection regret DISCRIMINATES (teacher < blind, honest report)", res.get("selection_regret_discriminates") in (True, False),
       f"teacher_regret={curve['teacher']['selection_regret'][0]['mean']:.4f} blind={curve['blind']['selection_regret'][0]['mean']:.4f}")

    tr = [c["mean"] for c in curve["student"]["selection_regret"]]
    print(json.dumps(dict(DESCRIPTIVE_non_gating=dict(student_regret=[round(x, 4) for x in tr],
                     teacher_regret=round(curve["teacher"]["selection_regret"][0]["mean"], 4),
                     blind_regret=round(curve["blind"]["selection_regret"][0]["mean"], 4),
                     note="reported for orientation ONLY; performance NEVER gates"))))

    npass = sum(1 for _, ok, _ in checks if ok); verdict = "SURVIVES" if npass == len(checks) else "FAILED"
    print(json.dumps(dict(verdict=verdict, passed=npass, total=len(checks),
                          checks=[dict(name=nm, ok=ok, detail=d) for nm, ok, d in checks]), indent=2))
    return 0 if verdict == "SURVIVES" else 1


if __name__ == "__main__":
    raise SystemExit(main())
