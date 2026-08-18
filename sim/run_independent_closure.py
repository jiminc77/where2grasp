"""C2 data + figures + verdict for the independent mechanics closure (baseline subtraction).

Runs the FROZEN pipeline (sim/manifests/independent_closure_manifest.json) and emits:
 - sim/manifests/independent_closure_results.npz  (raw arrays)
 - sim/manifests/independent_closure_verdict.json (all scored numbers + aggregate verdict)
 - sim/figures/ic_calibration_curves.png, ic_mesh_sequence.png, ic_no_refit_overlay.png

Every threshold is read from the frozen manifest / ic_common (pinned to prior committed
numbers). No threshold is computed from this phase's data. A miss is a REPORTED NULL.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sim import ic_common as ic
from sim import ic_norefit
from sim.calibrate_beff_force import force_calibrate_baseline
from sim.ic_gravity_boundary import gravity_droop_sweep, extract_boundary_and_K

ROOT = Path(__file__).resolve().parent
MAN = ROOT / "manifests"
FIG = ROOT / "figures"
REALIZED = (0.02, 0.01)                 # frozen realized mesh sequence (manifest)


def _cohort():
    c = ic.included_cohort()
    return [r for r, _, _ in c], [b for _, b, _ in c], [lab for _, _, lab in c]


def run_calibration(raw_es, b_effs, interval, target_ratio):
    res = force_calibrate_baseline(raw_es, b_effs, interval, target_ratio=target_ratio)
    bmap = {float(raw_es[i]): res["per_material"][i]["B_eff_force"] for i in range(len(raw_es))}
    return res, bmap


def observable_A(res, labels):
    """Max-abs normalized-shape error vs the analytical end-load shape, at the upper-bracket length."""
    ub = int(np.argmax(res["ell"]))
    shp = np.asarray(res["shape"][str(ub)])          # (n_materials, n_fracs)
    per = {lab: ic.observable_A_error(shp[i]) for i, lab in enumerate(labels)}
    return per, float(max(per.values()))


def observable_B(raw_es, labels, interval, bmap):
    """Measured distributed-gravity droop-clear boundary + K_N per material; returns per-material
    K_N and the droop arrays (reused for the r_N diagnostic)."""
    sweep = gravity_droop_sweep(raw_es, interval, mass=ic.OBS_B_REF_MASS)
    droop = np.asarray(sweep["droop"])               # (n_ell, n_materials)
    ells = np.asarray(sweep["ells"])
    perK = {}
    for i, (raw_e, lab) in enumerate(zip(raw_es, labels)):
        ext = extract_boundary_and_K(ells, droop[:, i], bmap[float(raw_e)], w=sweep["w"])
        perK[lab] = ext
    return perK, sweep


def rN_diagnostic(sweep, raw_es, labels, bmap, interval):
    """DESCRIPTIVE: measured gravity sag / continuum (~r_N). Reports per-length residual and the
    exponent before/after dividing by r_N, for the stiffest cohort material (representative)."""
    ells = np.asarray(sweep["ells"]); droop = np.asarray(sweep["droop"]); w = sweep["w"]
    out = {}
    for i, (raw_e, lab) in enumerate(zip(raw_es, labels)):
        d = droop[:, i]
        Ns = np.array([ic.segment_count(e, interval) for e in ells])
        rN = np.array([ic.r_N(N) for N in Ns])
        cont = w * ells ** 4 / (8.0 * bmap[float(raw_e)])
        ratio = d / cont
        exp_raw = float(np.polyfit(np.log(ells), np.log(np.maximum(d, 1e-12)), 1)[0])
        exp_corr = float(np.polyfit(np.log(ells), np.log(np.maximum(d / rN, 1e-12)), 1)[0])
        out[lab] = dict(sag_ratio=ratio.tolist(), rN=rN.tolist(),
                        exponent_raw=exp_raw, exponent_over_rN=exp_corr)
    return out


def run_prospective_sag(raw_es, labels, interval=ic.REFERENCE_INTERVAL):
    """Prospective score-only cohort: gravity sag at mass 0.0003, lengths {0.19,0.21,0.23}."""
    sweep = gravity_droop_sweep(raw_es, interval, mass=ic.PROSPECTIVE_MASS, ells=ic.PROSPECTIVE_LENGTHS)
    droop = np.asarray(sweep["droop"])
    rows = []
    for li, ell in enumerate(ic.PROSPECTIVE_LENGTHS):
        for mi, raw_e in enumerate(raw_es):
            rows.append(dict(raw_E=float(raw_e), mass=ic.PROSPECTIVE_MASS, ell=float(ell),
                             delta_obs=float(droop[li, mi])))
    return rows, sweep


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    raw_es, b_effs, labels = _cohort()
    print(f"cohort: {labels}", flush=True)

    # --- calibration + observables per realized mesh ---
    calib, obsA, obsB, sweeps, rN = {}, {}, {}, {}, {}
    bmap_ref = None
    calib_F2 = {}
    for interval in REALIZED:
        print(f"[calibration interval {interval}]", flush=True)
        res, bmap = run_calibration(raw_es, b_effs, interval, ic.FORCE_TARGET_RATIO)
        resF2, bmapF2 = run_calibration(raw_es, b_effs, interval, ic.FORCE_TARGET_RATIO * ic.F2_RATIO)
        calib[interval] = res; calib_F2[interval] = resF2
        obsA[interval] = observable_A(res, labels)
        perK, sweep = observable_B(raw_es, labels, interval, bmap)
        obsB[interval] = perK; sweeps[interval] = sweep
        rN[interval] = rN_diagnostic(sweep, raw_es, labels, bmap, interval)
        if interval == ic.REFERENCE_INTERVAL:
            bmap_ref = bmap
            bmap_ref_F2 = bmapF2

    # --- no-refit (B_eff_force at the reference interval 0.01) ---
    a = ic_norefit.score_sag_retrospective(bmap_ref)
    prospective_rows, prosp_sweep = run_prospective_sag(raw_es, labels)
    a_prosp = ic_norefit.score_sag_prospective(bmap_ref, prospective_rows)
    b = ic_norefit.score_prefactor(bmap_ref)
    c = ic_norefit.score_distal(bmap_ref)

    # --- finding-7 invariance (F1 vs F2, both in-guard) at the reference interval ---
    f7 = {}
    for lab, raw_e in zip(labels, raw_es):
        b1 = bmap_ref[float(raw_e)]; b2 = bmap_ref_F2[float(raw_e)]
        f7[lab] = float(abs(b2 - b1) / b1 * 100.0)
    f7_worst = float(max(f7.values()))

    # --- mesh gate (two-level fixed-discretization validation) ---
    A_abs = {iv: obsA[iv][1] for iv in REALIZED}
    K_dev = {iv: max(abs(obsB[iv][lab]["K_N"] - ic.PREDICTED_PREFACTOR)
                     for lab in labels if obsB[iv][lab]["K_N"] is not None) for iv in REALIZED}
    A_ceiling_ok = all(A_abs[iv] <= ic.OBS_A_ABSOLUTE_CEILING for iv in REALIZED)
    K_bound_ok = all(K_dev[iv] <= ic.OBS_B_TOL for iv in REALIZED)
    A_succ = abs(A_abs[0.02] - A_abs[0.01])
    meanK = {iv: float(np.mean([obsB[iv][l]["K_N"] for l in labels if obsB[iv][l]["K_N"] is not None]))
             for iv in REALIZED}
    K_succ = abs(meanK[0.02] - meanK[0.01])
    mesh_pass = bool(A_ceiling_ok and K_bound_ok)
    mesh_label = ("two-level fixed-discretization validation" if mesh_pass else "MESH-FAIL")

    # --- calibration-quality prerequisites ---
    cal_ok = all(m["max_residual"] <= ic.ACCEPT_RESIDUAL and m["CV"] <= ic.ACCEPT_CV
                 and ic.ACCEPT_EXPONENT_LO <= m["exponent"] <= ic.ACCEPT_EXPONENT_HI
                 for iv in REALIZED for m in calib[iv]["per_material"])
    guard_ok = all(m["guard_ok"] and m["superposition_ok"]
                   for iv in REALIZED for m in calib[iv]["per_material"])
    f7_ok = bool(f7_worst <= ic.FINDING7_ACCEPT)
    prereq_ok = bool(cal_ok and guard_ok and f7_ok)

    norefit_pass = bool(a["passed"] and a_prosp["passed"] and b["passed"] and c["passed"])
    if prereq_ok and mesh_pass and norefit_pass:
        verdict = "CLOSURE-PASS"
    elif not prereq_ok:
        verdict = "INCONCLUSIVE (calibration/guard/finding-7 prerequisite failed)"
    elif not mesh_pass:
        verdict = "REPORTED NULL (mesh prong MESH-FAIL)"
    else:
        verdict = "REPORTED NULL (>=1 no-refit family missed its frozen bound)"

    results = dict(
        cohort=labels, realized_intervals=list(REALIZED),
        calibration={str(iv): [dict(label=labels[i], **{k: calib[iv]["per_material"][i][k]
                     for k in ("raw_E", "B_eff_force", "CV", "max_residual", "exponent",
                               "point_ratio_ub", "total_ratio_ub", "guard_ok", "superposition_ok")})
                     for i in range(len(labels))] for iv in REALIZED},
        observable_A={str(iv): dict(per_material=obsA[iv][0], max_abs=obsA[iv][1]) for iv in REALIZED},
        observable_B={str(iv): {lab: obsB[iv][lab] for lab in labels} for iv in REALIZED},
        mesh_gate=dict(label=mesh_label, passed=mesh_pass, A_abs=A_abs, K_dev=K_dev,
                       A_successive=A_succ, K_successive=float(K_succ),
                       A_ceiling=ic.OBS_A_ABSOLUTE_CEILING, K_bound=ic.OBS_B_TOL,
                       note="two levels cannot satisfy the decreasing-trend 'convergence' clause; the "
                            "achievable pass is fixed-discretization validation via the reference-level absolute bounds."),
        no_refit=dict(a_retrospective=a, a_prospective=a_prosp, b_prefactor=b, c_distal=c, passed=norefit_pass),
        finding7=dict(per_material=f7, worst_pct=f7_worst, tol=ic.FINDING7_ACCEPT, passed=f7_ok),
        finding5_rN=rN,
        prerequisites=dict(calibration_ok=cal_ok, guard_ok=guard_ok, finding7_ok=f7_ok, all=prereq_ok),
        verdict=verdict,
        claim_language=json.loads((MAN / "independent_closure_manifest.json").read_text())["claim_language_mandatory_status"],
    )
    (MAN / "independent_closure_verdict.json").write_text(json.dumps(results, indent=2, default=float))

    # raw arrays for the red-team recompute
    np.savez(MAN / "independent_closure_results.npz",
             cohort=np.array(labels), raw_es=np.array(raw_es),
             **{f"delta_point_{iv}": np.asarray(calib[iv]["delta_point"]) for iv in REALIZED},
             **{f"delta_sw_{iv}": np.asarray(calib[iv]["delta_sw"]) for iv in REALIZED},
             **{f"delta_total_{iv}": np.asarray(calib[iv]["delta_total"]) for iv in REALIZED},
             **{f"ell_{iv}": np.asarray(calib[iv]["ell"]) for iv in REALIZED},
             **{f"forces_{iv}": np.asarray(calib[iv]["forces"]) for iv in REALIZED},
             **{f"grav_droop_{iv}": np.asarray(sweeps[iv]["droop"]) for iv in REALIZED},
             **{f"shape_ub_{iv}": np.asarray(calib[iv]["shape"][str(int(np.argmax(calib[iv]["ell"])))]) for iv in REALIZED},
             obs_A_fracs=np.asarray(ic.OBS_A_FRACS),
             grav_ells=np.asarray(sweeps[0.01]["ells"]),
             prospective_droop=np.asarray(prosp_sweep["droop"]))

    _figures(calib, obsA, obsB, a, b, c, a_prosp, labels, raw_es, b_effs)
    print("VERDICT:", verdict, flush=True)
    print("mesh:", mesh_label, "| A_abs", A_abs, "| K_dev", K_dev, flush=True)
    print("no-refit: a=%s a_prosp=%s b=%s(meanK=%.4f) c=%s(worst=%d) | finding7 worst=%.2f%%"
          % (a["passed"], a_prosp["passed"], b["passed"], b["mean_K"], c["passed"],
             c["worst_offset_cells"], f7_worst), flush=True)
    return results


def _figures(calib, obsA, obsB, a, b, c, a_prosp, labels, raw_es, b_effs):
    # (1) per-mode calibration curves
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    iv = ic.REFERENCE_INTERVAL
    ell = np.asarray(calib[iv]["ell"]); x3 = ell ** 3
    for i, lab in enumerate(labels):
        dp = np.asarray(calib[iv]["delta_point"])[:, i]
        ax[0].plot(x3, dp, "o", ms=4, label=lab)
        k = ic.through_origin_cubic(ell, dp); ax[0].plot(x3, k * x3, "-", lw=1)
    ax[0].set_xlabel(r"$\ell^3$ (m$^3$)"); ax[0].set_ylabel(r"$\delta_{point}$ (m)")
    ax[0].set_title("Force residual: through-origin $\\delta=F\\ell^3/(3B)$ (interval 0.01)"); ax[0].legend(fontsize=7)
    bff = np.array([m["B_eff_force"] for m in calib[iv]["per_material"]])
    ax[1].loglog(raw_es, bff, "o-", label="B_eff_force (baseline-subtracted)")
    ax[1].loglog(raw_es, b_effs, "s--", label="gravity B_eff (committed)")
    ax[1].set_xlabel("raw E"); ax[1].set_ylabel(r"$B_{eff}$ (N·m$^2$)")
    ax[1].set_title("Calibration map (cohort)"); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "ic_calibration_curves.png", dpi=130); plt.close(fig)

    # (2) mesh sequence
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    ivs = list(REALIZED)
    Avals = [obsA[iv][1] for iv in ivs]
    ax[0].plot(ivs, Avals, "o-"); ax[0].axhline(ic.OBS_A_ABSOLUTE_CEILING, color="r", ls="--", label="5% ceiling")
    ax[0].set_xlabel("interval"); ax[0].set_ylabel("Observable A max-abs shape error")
    ax[0].set_title("Observable A vs mesh (two-level)"); ax[0].legend(fontsize=8); ax[0].invert_xaxis()
    for lab in labels:
        Ks = [obsB[iv][lab]["K_N"] for iv in ivs]
        ax[1].plot(ivs, Ks, "o-", ms=4, label=lab)
    ax[1].axhline(ic.PREDICTED_PREFACTOR, color="k", ls="-", lw=1, label="0.5180")
    ax[1].axhspan(ic.PREDICTED_PREFACTOR - ic.OBS_B_TOL, ic.PREDICTED_PREFACTOR + ic.OBS_B_TOL, color="0.85")
    ax[1].set_xlabel("interval"); ax[1].set_ylabel(r"measured $K_N$")
    ax[1].set_title("Observable B $K_N$ vs mesh"); ax[1].legend(fontsize=6); ax[1].invert_xaxis()
    fig.tight_layout(); fig.savefig(FIG / "ic_mesh_sequence.png", dpi=130); plt.close(fig)

    # (3) no-refit overlay
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    rel = [r["rel_err"] * 100 for r in a["primary"]] + [r["rel_err"] * 100 for r in a_prosp["rows"]]
    ax[0].hist(rel, bins=12); ax[0].axvline(ic.NOREFIT_SAG_TOL * 100, color="r", ls="--", label="5%")
    ax[0].set_xlabel("sag rel err (%)"); ax[0].set_title("(a) no-refit sag"); ax[0].legend(fontsize=8)
    Ks = [cc["K_force"] for cc in b["per_cell"]]
    ax[1].plot(range(len(Ks)), Ks, "o"); ax[1].axhline(ic.PREDICTED_PREFACTOR, color="k")
    ax[1].axhspan(ic.PREDICTED_PREFACTOR - ic.OBS_B_TOL, ic.PREDICTED_PREFACTOR + ic.OBS_B_TOL, color="0.85")
    ax[1].set_title("(b) prefactor K_force per cell"); ax[1].set_xlabel("cell")
    offs = [cc["offset_cells"] for cc in c["per_cell"]]
    ax[2].bar(range(len(offs)), offs); ax[2].axhline(1, color="r", ls="--"); ax[2].axhline(-1, color="r", ls="--")
    ax[2].set_title("(c) distal grid-cell offset"); ax[2].set_xlabel("cell")
    fig.tight_layout(); fig.savefig(FIG / "ic_no_refit_overlay.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
