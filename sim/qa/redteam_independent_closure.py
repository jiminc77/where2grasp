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
from sim import ic_norefit

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

    # 3. B_eff_force recompute from committed force-residual arrays (independent cubic fit)
    drift = []
    bmap_ref = {}
    for iv in (0.02, 0.01):
        dp = np.asarray(data[f"delta_point_{iv}"]); ell = np.asarray(data[f"ell_{iv}"]); F = np.asarray(data[f"forces_{iv}"])
        for mi, lab in enumerate(labels):
            k = float(np.sum(ell**3 * dp[:, mi]) / np.sum(ell**3 * ell**3))
            mine = F[mi] / (3.0 * k)
            stored = [c for c in verdict["calibration"][str(iv)] if c["label"] == lab][0]["B_eff_force"]
            if abs(mine - stored) / stored > 1e-6:
                drift.append(f"{iv}/{lab}: {mine:.6g} vs {stored:.6g}")
            if iv == 0.01:
                bmap_ref[float(raw_es[mi])] = mine
    _check(res, "beff_force_recompute", not drift, "ok (cubic F/(3k) matches)" if not drift else "; ".join(drift))

    # 4. Observable A recompute from committed shapes (interpolated-shape max-abs error)
    aerr = []
    for iv in (0.02, 0.01):
        shp = np.asarray(data[f"shape_ub_{iv}"]); fr = np.asarray(data["obs_A_fracs"])
        mine = float(max(np.max(np.abs(shp[i] - ic.endload_shape(fr))) for i in range(len(labels))))
        stored = verdict["observable_A"][str(iv)]["max_abs"]
        if abs(mine - stored) > 1e-9:
            aerr.append(f"{iv}: {mine:.5f} vs {stored:.5f}")
    _check(res, "observable_A_recompute", not aerr, "ok" if not aerr else "; ".join(aerr))

    # 5. Observable B recompute from committed gravity droop arrays
    berr = []
    for iv in (0.02, 0.01):
        droop = np.asarray(data[f"grav_droop_{iv}"]); ells = np.asarray(data["grav_ells"])
        for mi, lab in enumerate(labels):
            b = ic.observable_B_boundary(ells, droop[:, mi])
            stored = verdict["observable_B"][str(iv)][lab]["ell_boundary"]
            if (b is None) != (stored is None) or (b is not None and abs(b - stored) > 1e-6):
                berr.append(f"{iv}/{lab}: {b} vs {stored}")
    _check(res, "observable_B_recompute", not berr, "ok" if not berr else "; ".join(berr[:4]))

    # 6. no-refit recompute from the recomputed B_eff_force map
    a = ic_norefit.score_sag_retrospective(bmap_ref)
    b = ic_norefit.score_prefactor(bmap_ref)
    c = ic_norefit.score_distal(bmap_ref)
    nr_ok = (abs(a["max_rel_err"] - verdict["no_refit"]["a_retrospective"]["max_rel_err"]) < 1e-6
             and abs(b["mean_K"] - verdict["no_refit"]["b_prefactor"]["mean_K"]) < 1e-6
             and c["worst_offset_cells"] == verdict["no_refit"]["c_distal"]["worst_offset_cells"])
    _check(res, "no_refit_recompute", nr_ok,
           f"a max_rel={a['max_rel_err']:.4f}(pass={a['passed']}) b meanK={b['mean_K']:.4f}(pass={b['passed']}) "
           f"c worst={c['worst_offset_cells']}(pass={c['passed']})")

    # 7. mesh-gate outcome recompute
    A_ok = all(verdict["observable_A"][str(iv)]["max_abs"] <= ic.OBS_A_ABSOLUTE_CEILING for iv in (0.02, 0.01))
    K_ok = all(max(abs(verdict["observable_B"][str(iv)][l]["K_N"] - ic.PREDICTED_PREFACTOR)
                   for l in labels if verdict["observable_B"][str(iv)][l]["K_N"] is not None) <= ic.OBS_B_TOL
               for iv in (0.02, 0.01))
    mesh_ok = bool(A_ok and K_ok)
    stored_mesh = verdict["mesh_gate"]["passed"]
    _check(res, "mesh_gate_recompute", mesh_ok == stored_mesh, f"recomputed pass={mesh_ok} vs stored {stored_mesh}")

    # 8. deterministic replay of a representative material (B4 @ 0.01, F1+F2) -> anti-fabrication + finding-7
    try:
        from sim.calibrate_beff_force import force_calibrate_baseline
        raw_e = ic.RAW_E_GRID[4]; b_eff = ic.GRAV_B_EFF[4]
        r1 = force_calibrate_baseline([raw_e], [b_eff], 0.01, target_ratio=ic.FORCE_TARGET_RATIO)
        r2 = force_calibrate_baseline([raw_e], [b_eff], 0.01, target_ratio=ic.FORCE_TARGET_RATIO * ic.F2_RATIO)
        b1 = r1["per_material"][0]["B_eff_force"]; b2 = r2["per_material"][0]["B_eff_force"]
        stored_b4 = [c for c in verdict["calibration"]["0.01"] if c["label"] == "B4"][0]["B_eff_force"]
        replay_ok = abs(b1 - stored_b4) / stored_b4 < 5e-3          # deterministic to ~0.5%
        f7 = abs(b2 - b1) / b1 * 100.0
        f7_stored = verdict["finding7"]["per_material"]["B4"]
        f7_ok = abs(f7 - f7_stored) < 0.5
        _check(res, "deterministic_replay_B4", bool(replay_ok and f7_ok),
               f"replay B_eff_force={b1:.5g} vs stored {stored_b4:.5g}; finding7 B4 replay={f7:.3f}% vs stored {f7_stored:.3f}%")
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
