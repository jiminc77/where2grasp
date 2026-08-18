"""Independent red-team for the Item-2 r_N-corrected no-refit graduation gate.

Fresh QA-LOCAL recompute (no production scorers, no stored scalars) of the r_N-corrected direct-sag
error on the retrospective calibration window (committed calibration.json) AND the held-out
prospective cohort (from the committed raw npz), the verdict, the finding-5 attribution, the
in-regime guard, N-from-ell, C1 single-file + git-ancestry of C2, input-artifact digests, no cohort
thinning, and the unchanged mesh carve-out. Emits a per-check report + a survivor count.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MAN = ROOT / "manifests"
REPORT = ROOT / "qa" / "redteam_rn_gate_report.json"
G = 9.81
INTERVAL = 0.01
TOL = 0.05
COHORT = ("B1", "B2", "B3", "B4", "R0", "R1", "R2")


def _rN(N):                       # QA-local lumped-mass factor
    return (N + 1.0) * (3.0 * N + 1.0) / (3.0 * N ** 2)


def _corrected(ell, mass, bff):   # QA-local r_N-corrected quartic sag
    N = int(round(ell / INTERVAL))
    w = mass * G / INTERVAL
    return _rN(N) * w * ell ** 4 / (8.0 * bff)


def _pi_g(mass, ell, bff):
    return (mass * G / INTERVAL) * ell ** 3 / bff


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(*a):
    return subprocess.run(["git", *a], cwd=ROOT.parent, capture_output=True, text=True)


def _check(res, name, ok, detail=""):
    res.append(dict(check=name, status="survives" if ok else "FAIL", detail=detail))


def run():
    res = []
    man = json.loads((MAN / "rn_gate_manifest.json").read_text())
    verdict = json.loads((MAN / "rn_gate_verdict.json").read_text())
    data = np.load(MAN / "rn_gate_results.npz", allow_pickle=True)
    bff = {k: float(v) for k, v in man["cohort"]["B_eff_force"].items()}

    # 1. input-artifact digests re-asserted against the frozen C1 manifest
    frozen = man["input_artifact_sha256"]
    mism = [a for a in frozen if not (MAN / a).exists() or frozen[a] != sha256(MAN / a)]
    _check(res, "input_artifact_digests", not mism, "ok" if not mism else "; ".join(mism))

    # 2. C1 single-file freeze + ancestry of C2
    c1 = git("log", "--diff-filter=A", "--format=%H", "--", "sim/manifests/rn_gate_manifest.json").stdout.split()
    c2 = git("log", "--diff-filter=A", "--format=%H", "--", "sim/manifests/rn_gate_verdict.json").stdout.split()
    if c1 and c2:
        c1h, c2h = c1[-1], c2[-1]
        stat = git("show", "--stat", "--format=", c1h).stdout
        single = stat.count("|") == 1 and "rn_gate_manifest.json" in stat
        anc = git("merge-base", "--is-ancestor", c1h, c2h).returncode == 0
        _check(res, "c1_single_file", single, f"C1={c1h[:8]} {stat.strip().splitlines()}")
        _check(res, "c1_ancestor_of_c2", anc, f"merge-base --is-ancestor {c1h[:8]} {c2h[:8]} = {anc}")
    else:
        _check(res, "c1_single_file", False, "manifest add-commit not found")
        _check(res, "c1_ancestor_of_c2", False, "add-commits not found")

    # 3. prospective corrected-error recompute from the committed raw npz (QA-local)
    droop = np.asarray(data["prospective_droop"]); ells = np.asarray(data["prospective_ells"])
    labels = [str(x) for x in np.asarray(data["cohort"])]
    perr = []
    for row in verdict["prospective"]:
        li = int(np.argmin(np.abs(ells - row["ell"]))); mi = labels.index(row["label"])
        dobs = float(droop[li, mi])
        mine = abs(_corrected(row["ell"], row["mass"], bff[row["label"]]) - dobs) / abs(dobs)
        if abs(mine - row["rel_err"]) > 1e-6:
            perr.append(f"{row['label']}@{row['ell']}: {mine:.5f} vs {row['rel_err']:.5f}")
    _check(res, "prospective_recompute", not perr, "ok (QA-local corrected error from raw droop)" if not perr else "; ".join(perr[:4]))

    # 4. retrospective corrected-error recompute from committed calibration.json (QA-local)
    cal = json.loads((MAN / "calibration.json").read_text())
    RAW = (1_000_000.0, 2_370_946.5892385854, 5_621_387.729022082, 13_328_010.062912514, 31_600_000.0)
    rerr = []
    for row in verdict["retrospective"]:
        raw_e = RAW[int(row["label"][1])]
        mat = next(m for m in cal["materials"] if abs(m["raw_E"] - raw_e) < 1e-6)
        pl = mat["per_length"]["0.0004"]; idx = int(np.argmin(np.abs(np.array(pl["ell"]) - row["ell"])))
        dobs = float(pl["delta"][idx])
        mine = abs(_corrected(row["ell"], 0.0004, bff[row["label"]]) - dobs) / abs(dobs)
        if abs(mine - row["rel_err"]) > 1e-6:
            rerr.append(f"{row['label']}@{row['ell']}: {mine:.5f} vs {row['rel_err']:.5f}")
    _check(res, "retrospective_recompute", not rerr, "ok (QA-local from calibration.json)" if not rerr else "; ".join(rerr[:4]))

    # 5. verdict recompute (outcome-binding) using QA-locally RECOMPUTED convergence from the raw
    # consecutive-window (window < drift_threshold).all(axis=0) -- NOT the stored finite flags.
    win = np.asarray(data["prospective_window"]); thr = float(data["drift_threshold"])
    pell = np.asarray(data["prospective_ells"]); pcoh = [str(x) for x in np.asarray(data["cohort"])]
    prosp_conv = {}
    for c in verdict["prospective"]:
        li = int(np.argmin(np.abs(pell - c["ell"]))); mi = pcoh.index(c["label"])
        prosp_conv[(c["label"], c["ell"])] = bool((win[:, li, mi] < thr).all())
    prosp_breakdown = [k for k, v in prosp_conv.items() if not v]        # any non-converged prospective cell
    cells = verdict["retrospective"] + verdict["prospective"]
    in_regime = [c for c in cells if c["in_regime"]]
    nonfinite = [c for c in in_regime if not c["finite"]]
    if prosp_breakdown or nonfinite or not in_regime:
        mine_v = "INCONCLUSIVE"
    elif all(c["rel_err"] <= TOL for c in in_regime):
        mine_v = "PASS"
    else:
        mine_v = "MISS"
    _check(res, "verdict_recompute", mine_v == verdict["verdict"] and not prosp_breakdown,
           f"recomputed {mine_v} vs stored {verdict['verdict']}; worst in-regime "
           f"{max(c['rel_err'] for c in in_regime):.4f}; prospective reconverged (window<thr)={not prosp_breakdown}")

    # 6. N == round(ell/interval), nominal==realized, guard reproduced (QA-local)
    n_ok = all(c["N"] == int(round(c["ell"] / INTERVAL)) for c in cells)
    guard_ok = all(c["in_regime"] == (_pi_g(c["mass"], c["ell"], bff[c["label"]]) <= 0.5) for c in cells)
    prospN = sorted({c["N"] for c in verdict["prospective"]})
    disjoint = set(prospN).isdisjoint({18, 20, 22, 24} | {19, 21, 23})
    _check(res, "N_from_ell_and_guard", n_ok and guard_ok and disjoint and prospN == [15, 16, 17],
           f"N-from-ell={n_ok} guard={guard_ok} prospN={prospN} disjoint={disjoint}")

    # 7. finding-5 attribution reproduction (QA-local)
    gb = {**{f"B{i}": v for i, v in zip(range(1, 5), (0.01610520612280113, 0.03825920568613013,
              0.09110646283229176, 0.21527426878237318))},
          **{r: v for r, v in zip(("R0", "R1", "R2"), (0.032264843367045035, 0.045408831332695465, 0.05753320861264561))}}
    gaps = [(bff[l] / gb[l] - 1.0) * 100.0 for l in COHORT]
    mean_rN = (float(np.mean([_rN(N) for N in (18, 20, 22, 24)])) - 1.0) * 100.0
    residual = float(np.mean(gaps)) - mean_rN
    a = verdict["attribution"]
    attr_ok = (abs(mean_rN - a["calibration_window_mean_rN_pct"]) < 1e-6
               and abs(residual - a["residual_above_rN_pct"]) < 1e-6)
    _check(res, "finding5_attribution", attr_ok, f"mean_rN={mean_rN:.4f}% residual={residual:.4f}% (OPEN)")

    # 8. no cohort thinning: EXACT frozen inventory scored (21/21 prospective + 16/16 retrospective
    # in-regime), with every prospective cell reconverged from the raw window (not stored flags).
    exp_prosp_keys = {(lab, int(round(ell / INTERVAL))) for lab in COHORT for ell in man["prospective_cohort"]["lengths"]}
    got_prosp_keys = {(c["label"], c["N"]) for c in verdict["prospective"]}
    retro_in = [c for c in verdict["retrospective"] if c["in_regime"]]
    exp_retro_keys = {(lab, ell) for lab in ("B1", "B2", "B3", "B4") for ell in (0.18, 0.20, 0.22, 0.24)}
    got_retro_keys = {(c["label"], round(c["ell"], 2)) for c in retro_in}
    all_reconverged = all(prosp_conv.values()) and len(prosp_conv) == 21
    ok = (got_prosp_keys == exp_prosp_keys and got_retro_keys == exp_retro_keys
          and len(verdict["prospective"]) == 21 and len(retro_in) == 16 and all_reconverged)
    _check(res, "no_thinning", ok,
           f"prospective keys {len(got_prosp_keys)}/21 exact={got_prosp_keys == exp_prosp_keys}; "
           f"retrospective in-regime {len(retro_in)}/16 exact={got_retro_keys == exp_retro_keys}; "
           f"all 21 reconverged from raw window={all_reconverged}")

    # 9. mesh carve-out unchanged: the aggregate closure is still bounded by mesh B4@0.02 INCONCLUSIVE
    icv = json.loads((MAN / "independent_closure_verdict.json").read_text())
    mesh_unchanged = (icv["mesh_gate"]["full_cohort_B_resolved"] is False
                      and "B4" in icv["mesh_gate"]["unresolved"].get("0.02", []))
    grad = verdict["graduates_direct_sag_prong"]
    _check(res, "mesh_carveout_unchanged", mesh_unchanged and grad is False,
           f"mesh B4@0.02 full_cohort_B_resolved={icv['mesh_gate']['full_cohort_B_resolved']}; "
           f"graduates_direct_sag_prong={grad} (MISS -> no graduation)")

    # 10. claim boundary verbatim + r_N clause
    icm = json.loads((MAN / "independent_closure_manifest.json").read_text())
    claim = verdict["claim_language"]
    claim_ok = claim.startswith(icm["claim_language_mandatory_status"]) and claim.endswith("lumped-mass factor")
    _check(res, "claim_boundary_verbatim", claim_ok, "verbatim + r_N clause" if claim_ok else "claim mismatch")

    survivors = sum(1 for r in res if r["status"] == "survives")
    fails = sum(1 for r in res if r["status"] == "FAIL")
    report = dict(checks=res, survivors=survivors, fails=fails, survived_all=bool(fails == 0),
                  verdict_under_review=verdict["verdict"])
    REPORT.write_text(json.dumps(report, indent=2, default=float))
    print(f"RED-TEAM: SURVIVES {survivors}/{survivors + fails} (fails={fails}) -> {REPORT}")
    for r in res:
        print(f"  [{r['status']:8}] {r['check']}: {r['detail']}")
    return report


if __name__ == "__main__":
    run()
