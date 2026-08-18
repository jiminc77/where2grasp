"""Independent red-team recompute for the independent mechanics closure (baseline subtraction).

Fresh-code recompute (NOT a re-read of stored scalars) of B_eff_force, Observable A/B + the
mesh-gate outcome, the three no-refit errors, and finding-7, from the committed C2 raw arrays
(independent_closure_results.npz) + manifests; plus a dataflow/provenance assertion (the force
fitter only ever sees force-mode residual arrays), input-array digests, C1 single-file /
ancestry proof, and a DETERMINISTIC REPLAY of a representative material to detect fabricated
arrays. Emits a per-check report + a survivor count.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from sim import ic_common as ic

ROOT = Path(__file__).resolve().parents[1]
MAN = ROOT / "manifests"
REPORT = ROOT / "qa" / "redteam_independent_closure_report.json"
INPUTS = ("calibration.json", "hard_gate_verdict.json", "hard_sweep_landscape.json",
          "distal_manifest.json", "distal_sweep_landscape.json", "distal_gate_verdict.json",
          "distal_critic_results.json", "addendum_results.json", "spanning_results.json")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT.parent, capture_output=True, text=True)


def _check(res, name, ok, detail=""):
    res.append(dict(check=name, status="survives" if ok else "FAIL", detail=detail))


def run():
    res = []
    verdict = json.loads((MAN / "independent_closure_verdict.json").read_text())
    data = np.load(MAN / "independent_closure_results.npz", allow_pickle=True)
    manifest = json.loads((MAN / "independent_closure_manifest.json").read_text())
    labels = list(verdict["cohort"]); raw_es = list(np.asarray(data["raw_es"]))

    # 1. input-artifact digests re-asserted against the frozen manifest list
    frozen = manifest["input_artifact_sha256"]
    mism = [f"{a}:{'missing' if not (MAN/a).exists() else 'drift'}"
            for a in INPUTS if not (MAN/a).exists() or frozen.get(a) != sha256(MAN/a)]
    _check(res, "input_artifact_digests", not mism, "ok" if not mism else "; ".join(mism))

    # 2. C1 single-file freeze + ancestry of C2 (git-provable)
    c1 = git("log", "--diff-filter=A", "--format=%H", "--", "sim/manifests/independent_closure_manifest.json").stdout.split()
    c2 = git("log", "--diff-filter=A", "--format=%H", "--", "sim/manifests/independent_closure_verdict.json").stdout.split()
    if c1 and c2:
        c1h, c2h = c1[-1], c2[-1]
        stat = git("show", "--stat", "--format=", c1h).stdout
        single = stat.count("|") == 1 and "independent_closure_manifest.json" in stat
        anc = git("merge-base", "--is-ancestor", c1h, c2h).returncode == 0
        _check(res, "c1_single_file", single, f"C1={c1h[:8]} touches: {stat.strip().splitlines()}")
        _check(res, "c1_ancestor_of_c2", anc, f"merge-base --is-ancestor {c1h[:8]} {c2h[:8]} = {anc}")
    else:
        _check(res, "c1_single_file", False, "manifest add-commit not found")
        _check(res, "c1_ancestor_of_c2", False, "add-commits not found")

    # 3. B_eff_force recompute from committed force-residual arrays (independent cubic fit), per interval
    drift = []
    bmap_by_iv = {0.02: {}, 0.01: {}}
    for iv in (0.02, 0.01):
        dp = np.asarray(data[f"delta_point_{iv}"]); ell = np.asarray(data[f"ell_{iv}"]); F = np.asarray(data[f"forces_{iv}"])
        for mi, lab in enumerate(labels):
            k = float(np.sum(ell**3 * dp[:, mi]) / np.sum(ell**3 * ell**3))
            mine = F[mi] / (3.0 * k)
            stored = [c for c in verdict["calibration"][str(iv)] if c["label"] == lab][0]["B_eff_force"]
            if abs(mine - stored) / stored > 1e-6:
                drift.append(f"{iv}/{lab}: {mine:.6g} vs {stored:.6g}")
            bmap_by_iv[iv][float(raw_es[mi])] = mine
    bmap_ref = bmap_by_iv[0.01]
    _check(res, "beff_force_recompute", not drift, "ok (cubic F/(3k) matches)" if not drift else "; ".join(drift))

    # QA-LOCAL formulas (reimplemented here; the production ic.* scorers are what is UNDER TEST)
    def w_of(iv):
        return ic.OBS_B_REF_MASS * ic.G / iv                     # interval-specific linear weight

    def _rawE(be):
        for re_, gb in list(zip(ic.RAW_E_GRID, ic.GRAV_B_EFF)) + list(zip(ic.RATIO_RAW_E, ic.RATIO_GRAV_B_EFF)):
            if abs(gb - be) <= 1e-6 * max(1.0, abs(gb)):
                return re_
        raise ValueError(f"no exact B_eff match for {be}")

    def _shape(f):                                               # analytical end-load shape (QA-local)
        f = np.asarray(f, dtype=float)
        return (3.0 * f ** 2 - f ** 3) / 2.0

    def _cross(ells, droops, h=ic.DROOP_CLEAR_H):                # linear-interp droop=h crossing (QA-local)
        ells = np.asarray(ells, float); droops = np.asarray(droops, float)
        for k in range(len(ells) - 1):
            if droops[k] <= h < droops[k + 1]:
                t = (h - droops[k]) / (droops[k + 1] - droops[k])
                return float(ells[k] + t * (ells[k + 1] - ells[k])), (k, k + 1)
        return None, None

    def _resolved_boundary(iv, mi):
        """Driver-equivalent resolved boundary: the crossing whose TWO bracketing rods converged; else None."""
        droop = np.asarray(data[f"grav_droop_{iv}"])[:, mi]; ells = np.asarray(data["grav_ells"])
        convd = np.asarray(data[f"sweep_conv_{iv}"])[:, mi]
        b, br = _cross(ells, droop)
        if b is None or br is None:
            return None
        return b if (bool(convd[br[0]]) and bool(convd[br[1]])) else None   # unbracketed -> INCONCLUSIVE

    # 4. Observable A recompute (QA-local end-load shape) vs stored max-abs
    aerr = []
    for iv in (0.02, 0.01):
        shp = np.asarray(data[f"shape_ub_{iv}"]); fr = np.asarray(data["obs_A_fracs"])
        mine = float(max(np.max(np.abs(shp[i] - _shape(fr))) for i in range(len(labels))))
        stored = verdict["observable_A"][str(iv)]["max_abs"]
        if abs(mine - stored) > 1e-9:
            aerr.append(f"{iv}: {mine:.5f} vs {stored:.5f}")
    _check(res, "observable_A_recompute", not aerr, "ok (QA-local shape)" if not aerr else "; ".join(aerr))

    # 5. Observable B recompute (QA-local crossing + bracket-convergence) + resolution completeness
    berr = []
    unres_re = {0.02: [], 0.01: []}
    for iv in (0.02, 0.01):
        for mi, lab in enumerate(labels):
            b = _resolved_boundary(iv, mi)
            stored = verdict["observable_B"][str(iv)][lab]["ell_boundary"]
            if (b is None) != (stored is None) or (b is not None and abs(b - stored) > 1e-6):
                berr.append(f"{iv}/{lab} boundary {b} vs {stored}")
            if b is None:
                unres_re[iv].append(lab)
            else:
                kn = b / (bmap_by_iv[iv][float(raw_es[mi])] / w_of(iv)) ** 0.25
                stored_kn = verdict["observable_B"][str(iv)][lab]["K_N"]
                if abs(kn - stored_kn) > 1e-6:
                    berr.append(f"{iv}/{lab} K_N {kn:.5f} vs {stored_kn}")
    _check(res, "observable_B_recompute", not berr,
           "ok (QA-local crossing + K_N, bracket-convergence checked)" if not berr else "; ".join(berr[:4]))
    stored_unres = verdict["mesh_gate"]["unresolved"]
    comp_ok = all(sorted(unres_re[iv]) == sorted(stored_unres.get(str(iv), [])) for iv in (0.02, 0.01))
    _check(res, "mesh_completeness", comp_ok,
           f"unresolved recomputed 0.02={unres_re[0.02]} 0.01={unres_re[0.01]} vs stored {stored_unres}; "
           f"full_cohort_B_resolved stored={verdict['mesh_gate'].get('full_cohort_B_resolved')} "
           f"(unresolved cells REPORTED, not dropped; bracket-convergence enforced)")

    # 6. no-refit INDEPENDENT recompute (fresh inline arithmetic from committed manifests; NO production scorers)
    cal = json.loads((MAN / "calibration.json").read_text())
    a_rel = []
    for mat in cal["materials"]:
        bff = bmap_ref.get(mat["raw_E"])
        if bff is None:
            continue                                  # B0: out of the superposition cohort, not calibrated
        for mkey, pl in mat["per_length"].items():
            m = float(mkey)
            if abs(m - ic.BASELINE_ARM_MASS) <= 1e-12:
                continue                              # baseline mass == m0 -> descriptive, not primary
            w = m * ic.G / cal["interval"]
            for L, dobs, pig in zip(pl["ell"], pl["delta"], pl["Pi_g"]):
                if pig > ic.PI_G_MAX:
                    continue                          # out of the small-deflection regime
                a_rel.append(abs(w * L ** 4 / (8.0 * bff) - dobs) / abs(dobs))
    a_max = float(max(a_rel))
    hv = json.loads((MAN / "hard_gate_verdict.json").read_text())
    Ks = [bnd["boundary"] / (bmap_ref[_rawE(bnd["B_eff"])] / bnd["w"]) ** 0.25 for bnd in hv["boundaries"]]
    b_mean = float(np.mean(Ks))
    land = json.loads((MAN / "distal_sweep_landscape.json").read_text())
    worst = 0
    for s in land["settings"]:
        if not s.get("measured_feasible") or s.get("measured_argmax_ell") is None:
            continue
        bff = bmap_ref.get(_rawE(s["B_eff"]))
        if bff is None:
            continue
        sstar = (8.0 * bff * ic.DISTAL_DELTA / s["w"]) ** 0.25
        snap = round(0.12 + round((sstar - 0.12) / 0.02) * 0.02, 2)
        worst = max(worst, abs(int(round((snap - s["measured_argmax_ell"]) / 0.02))))
    nr_ok = (abs(a_max - verdict["no_refit"]["a_retrospective"]["max_rel_err"]) < 1e-6
             and abs(b_mean - verdict["no_refit"]["b_prefactor"]["mean_K"]) < 1e-6
             and worst == verdict["no_refit"]["c_distal"]["worst_offset_cells"])
    _check(res, "no_refit_recompute", nr_ok,
           f"independent arithmetic: a max_rel={a_max:.4f} (>5%={a_max > ic.NOREFIT_SAG_TOL}) "
           f"b meanK={b_mean:.4f} (in-bound={abs(b_mean - ic.PREDICTED_PREFACTOR) <= ic.OBS_B_TOL}) "
           f"c worst={worst} cell(s)")

    # 6b. no-refit PROSPECTIVE family-a recompute (fresh score-only sims committed in prospective_droop)
    pd = np.asarray(data["prospective_droop"])                  # (n_prospective_lengths, n_material)
    wp = ic.PROSPECTIVE_MASS * ic.G / ic.REFERENCE_INTERVAL
    p_rel = [abs(wp * L ** 4 / (8.0 * bmap_ref[float(raw_es[mi])]) - pd[li, mi]) / abs(pd[li, mi])
             for li, L in enumerate(ic.PROSPECTIVE_LENGTHS) for mi in range(len(labels))]
    p_max = float(max(p_rel))
    _check(res, "no_refit_prospective_recompute",
           abs(p_max - verdict["no_refit"]["a_prospective"]["max_rel_err"]) < 1e-6,
           f"prospective (mass {ic.PROSPECTIVE_MASS}) max_rel={p_max:.4f} (>5%={p_max > ic.NOREFIT_SAG_TOL}) "
           f"vs stored {verdict['no_refit']['a_prospective']['max_rel_err']:.4f}")

    # 7. mesh-gate recompute from RAW arrays (QA-local shape + crossing; cells resolved at BOTH levels)
    fr = np.asarray(data["obs_A_fracs"])
    A_ok = all(float(max(np.max(np.abs(np.asarray(data[f"shape_ub_{iv}"])[i] - _shape(fr)))
                         for i in range(len(labels)))) <= ic.OBS_A_ABSOLUTE_CEILING for iv in (0.02, 0.01))
    resolved_both = [labels[mi] for mi in range(len(labels))
                     if all(_resolved_boundary(iv, mi) is not None for iv in (0.02, 0.01))]
    K_ok = True
    for iv in (0.02, 0.01):
        devs = []
        for mi, lab in enumerate(labels):
            if lab not in resolved_both:
                continue
            bnd = _resolved_boundary(iv, mi)
            devs.append(abs(bnd / (bmap_by_iv[iv][float(raw_es[mi])] / w_of(iv)) ** 0.25 - ic.PREDICTED_PREFACTOR))
        if devs and max(devs) > ic.OBS_B_TOL:
            K_ok = False
    mesh_subset = bool(A_ok and K_ok)
    full_resolved = len(resolved_both) == len(labels)
    stored_subset = verdict["mesh_gate"]["subset_passed"]; stored_full = verdict["mesh_gate"]["full_cohort_B_resolved"]
    _check(res, "mesh_gate_recompute", mesh_subset == stored_subset and full_resolved == stored_full,
           f"QA-local recompute: subset_pass={mesh_subset} full_resolved={full_resolved} vs stored subset={stored_subset} full={stored_full}")

    # 7b. settle-integrity verification: calibration + prospective converged; EVERY resolved boundary's
    # two bracketing rods converged (bracket convergence is what makes a boundary admissible)
    si = verdict["settle_integrity"]
    calib_conv = (all(si["detail"]["calibration"][str(iv)]["converged"] and si["detail"]["calibration_F2"][str(iv)]["converged"]
                      for iv in ("0.02", "0.01")) and si["detail"]["prospective"]["converged"])
    bracket_ok = True
    for iv in (0.02, 0.01):
        convd = np.asarray(data[f"sweep_conv_{iv}"])
        for mi, lab in enumerate(labels):
            if verdict["observable_B"][str(iv)][lab]["ell_boundary"] is None:
                continue                                        # unresolved cells excused
            _b, br = _cross(np.asarray(data["grav_ells"]), np.asarray(data[f"grav_droop_{iv}"])[:, mi])
            if br is None or not (bool(convd[br[0], mi]) and bool(convd[br[1], mi])):
                bracket_ok = False
    _check(res, "settle_integrity_verify", bool(si["all_ok"] and calib_conv and bracket_ok),
           f"settle_all_ok={si['all_ok']} calib+prosp_converged={calib_conv} "
           f"every_resolved_boundary_bracket_converged={bracket_ok}")

    # 8a. finding-7 recompute from COMMITTED F2 arrays (fresh cubic fit; no re-sim)
    f7err = []
    dpF2 = np.asarray(data["delta_point_F2_0.01"]); FF2 = np.asarray(data["forces_F2_0.01"]); ellF2 = np.asarray(data["ell_0.01"])
    for mi, lab in enumerate(labels):
        kF2 = float(np.sum(ellF2**3 * dpF2[:, mi]) / np.sum(ellF2**3 * ellF2**3)); b2 = FF2[mi] / (3.0 * kF2)
        f7 = abs(b2 - bmap_ref[float(raw_es[mi])]) / bmap_ref[float(raw_es[mi])] * 100.0
        if abs(f7 - verdict["finding7"]["per_material"][lab]) > 1e-4:
            f7err.append(f"{lab}: {f7:.3f} vs {verdict['finding7']['per_material'][lab]:.3f}")
    _check(res, "finding7_recompute", not f7err,
           "ok (F2 committed arrays; worst %.3f%%)" % max(verdict["finding7"]["per_material"].values())
           if not f7err else "; ".join(f7err[:3]))

    # 8b. deterministic replay (B4 @ 0.01, F1) -> anti-fabrication
    try:
        from sim.calibrate_beff_force import force_calibrate_baseline
        r1 = force_calibrate_baseline([ic.RAW_E_GRID[4]], [ic.GRAV_B_EFF[4]], 0.01, target_ratio=ic.FORCE_TARGET_RATIO)
        b1r = r1["per_material"][0]["B_eff_force"]; stored_b4 = bmap_ref[float(ic.RAW_E_GRID[4])]
        _check(res, "deterministic_replay_B4", abs(b1r - stored_b4) / stored_b4 < 5e-3,
               f"replay B_eff_force={b1r:.5g} vs committed {stored_b4:.5g} (arrays not fabricated)")
    except Exception as exc:  # noqa: BLE001
        _check(res, "deterministic_replay_B4", False, f"EXCEPTION {type(exc).__name__}: {exc}")

    # 9. dataflow provenance: B_eff_force fit sees ONLY the force-mode residual (delta_point), never a gravity-sag fit
    src = (ROOT / "calibrate_beff_force.py").read_text()
    fit_on_residual = "through_origin_cubic(ell, d_point" in src or "cubic_fit_stats(ell, d_point" in src
    no_sag_fit = "w*ell**4/(8" not in src.replace(" ", "") and "ell**4/(8.0*" not in src.replace(" ", "")
    _check(res, "dataflow_provenance", fit_on_residual and no_sag_fit,
           "B_eff_force fit is cubic on the force residual delta_point; no gravity-sag law fit in the calibration path")

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
