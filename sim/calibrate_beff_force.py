"""Force-mode B_eff calibration by BASELINE SUBTRACTION (owner ruling; supersedes the
epsilon-arm option (iv), which the C0 stability probe proved numerically non-viable at the
reference interval -- escalation commits a6cfb21 + bf688ec).

Realization: the arm keeps a NORMAL stable uniform mass m0. Two settles isolate the
concentrated tip-load response by linear superposition:
  A) arm uniform m0, no tip load                 -> delta_sw   (self-weight baseline, MEASURED data)
  B) arm uniform m0, tip vertex = m0 + m_load     -> delta_total
  delta_point = delta_total - delta_sw = F*ell^3/(3B),  F = m_load*g.
B_eff_force is fit ONLY to the CUBIC point-load law on the residual delta_point. The
subtracted baseline is a measured settle (the gravity channel reused as DATA, not the sag
LAW); a common multiplicative g/mass scale error cancels (delta_sw and delta_point both
scale, so B_eff_force = F/(3k) = B/lambda, matching the gravity calibration).

The frozen <1% contamination bound is NOT relaxed; it is SUPERSEDED by this design (there
is no epsilon-arm distributed-mass contamination to bound). Validity is instead governed by
the pre-declared linear-superposition guard (total delta/ell <= Pi_g_max/8 = 0.0625), which
excludes the softest B0 as regime-of-validity (ic_common.superposition_included).
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import genesis as gs

from sim.scene import build_scene, add_straight_rod, vertices
from sim import ic_common as ic

ROOT = Path(__file__).resolve().parent
PROBE_LOG = ROOT / "logs" / "ic_stability_probe.log"


def build_baseline_tensors(nv, m_loads, m0=ic.BASELINE_ARM_MASS):
    """Return (baseline, loaded) (n_envs, nv) mass tensors. baseline = uniform m0; loaded =
    uniform m0 with the tip vertex carrying m0 + m_load (the added concentrated point load)."""
    m_loads = np.asarray(m_loads, dtype=float).reshape(-1)
    n_envs = m_loads.size
    baseline = np.full((n_envs, nv), m0)
    loaded = np.full((n_envs, nv), m0)
    loaded[:, ic.tip_index(nv)] = m0 + m_loads
    return baseline, loaded


def _settle(scene, rods, chunk=200, max_steps=16000, drift_tol=5e-3, interval=0.01):
    """Chunked static settle keyed on per-rod tip drift. Returns (converged, steps)."""
    tips = [ic.tip_index(r.n_vertices) for r in rods]
    prev = [vertices(r)[:, ti, 2].copy() for r, ti in zip(rods, tips)]
    steps = 0
    while steps < max_steps:
        for _ in range(chunk):
            scene.step()
        steps += chunk
        cur = [vertices(r)[:, ti, 2].copy() for r, ti in zip(rods, tips)]
        drift = max(float(np.max(np.abs(c - p))) for c, p in zip(cur, prev))
        prev = cur
        if drift < drift_tol * interval:
            return True, steps
    return False, steps


def force_calibrate_baseline(raw_es, b_eff_sizing, interval, m0=ic.BASELINE_ARM_MASS,
                             lengths=ic.CALIB_LENGTHS, target_ratio=ic.FORCE_TARGET_RATIO,
                             max_steps=16000):
    """Baseline-subtraction force calibration. envs = materials, rods = lengths, ONE scene:
    settle the self-weight baseline, then ADD the tip load and settle again; subtract.

    m_load is sized ONCE per material at the upper-bracket ell from b_eff_sizing (prior
    committed gravity B_eff) to delta_point/ell ~= target_ratio, so F is a known constant.
    Returns per-(length,material) delta_sw/delta_total/delta_point, the residual droop
    profiles (Observable A), and per-material cubic-fit stats + guards.
    """
    raw_es = np.asarray(raw_es, dtype=float)
    b_eff_sizing = np.asarray(b_eff_sizing, dtype=float)
    m_loads = np.array([ic.m_tip_for(b, target_ratio=target_ratio) for b in b_eff_sizing])
    forces = m_loads * ic.G
    n_envs = raw_es.size

    scene = build_scene(dt=ic.DT, substeps=ic.SUBSTEPS, damping=ic.DAMPING,
                        angular_damping=ic.ANGULAR_DAMPING)
    nvs = [ic.n_vertices_for(L, interval) for L in lengths]
    rods = [add_straight_rod(scene, nv, interval=interval, E=1e7, segment_mass=m0,
                             segment_radius=ic.SEGMENT_RADIUS, G=ic.SHEAR_G, pos=(0, 0.35 * j, 0.7))
            for j, nv in enumerate(nvs)]
    scene.build(n_envs=n_envs, env_spacing=(5, 5))
    baselines, loadeds = [], []
    for rod, nv in zip(rods, nvs):
        rod.set_fixed_states(fixed_ids=[0, 1])
        rod.set_bending_stiffness(torch.tensor(raw_es, dtype=gs.tc_float, device="cuda"))
        b2d, l2d = build_baseline_tensors(nv, m_loads, m0)
        baselines.append(b2d); loadeds.append(l2d)
        rod.set_segment_mass(torch.tensor(b2d, dtype=gs.tc_float, device="cuda"))
        got = rod.get_all_segment_mass_tc().detach().cpu().numpy()
        assert np.allclose(got, b2d, rtol=1e-5, atol=1e-12), "baseline segment-mass read-back mismatch"

    z0 = [vertices(r).copy() for r in rods]
    conv_sw, steps_sw = _settle(scene, rods, max_steps=max_steps, interval=interval)
    z_sw = [vertices(r).copy() for r in rods]
    for rod, nv, l2d in zip(rods, nvs, loadeds):     # add the concentrated tip load
        rod.set_segment_mass(torch.tensor(l2d, dtype=gs.tc_float, device="cuda"))
    conv_tot, steps_tot = _settle(scene, rods, max_steps=max_steps, interval=interval)
    z_tot = [vertices(r).copy() for r in rods]

    ell = np.array([(nv - 2) * interval for nv in nvs])
    d_sw = np.empty((len(lengths), n_envs))
    d_tot = np.empty((len(lengths), n_envs))
    shape = {}
    for li, nv in enumerate(nvs):
        ti = ic.tip_index(nv)
        d_sw[li] = z0[li][:, ti, 2] - z_sw[li][:, ti, 2]
        d_tot[li] = z0[li][:, ti, 2] - z_tot[li][:, ti, 2]
        point_profile = (z_sw[li][:, :, 2] - z_tot[li][:, :, 2])          # residual tip-load droop, per vertex
        fr = np.array(ic.OBS_A_FRACS)
        idx = np.clip(np.round(1 + fr * (nv - 2)).astype(int), 1, nv - 1)
        tip_point = point_profile[:, ti]
        shape[li] = point_profile[:, idx] / tip_point[:, None]
    d_point = d_tot - d_sw

    per_material = []
    ub = int(np.argmax(ell))
    for mi in range(n_envs):
        stats = ic.cubic_fit_stats(ell, d_point[:, mi], forces[mi])
        stats.update(raw_E=float(raw_es[mi]), m_load=float(m_loads[mi]), F=float(forces[mi]),
                     delta_sw=d_sw[:, mi].tolist(), delta_total=d_tot[:, mi].tolist(),
                     delta_point=d_point[:, mi].tolist(),
                     point_ratio_ub=ic.deflection_ratio(d_point[ub, mi], ell[ub]),
                     total_ratio_ub=ic.deflection_ratio(d_tot[ub, mi], ell[ub]),
                     guard_ok=ic.guard_ok(d_point[ub, mi], ell[ub]),
                     superposition_ok=bool(ic.deflection_ratio(d_tot[ub, mi], ell[ub]) <= ic.SUPERPOSITION_GUARD))
        per_material.append(stats)

    return dict(interval=interval, m0=m0, ell=ell.tolist(),
                delta_sw=d_sw.tolist(), delta_total=d_tot.tolist(), delta_point=d_point.tolist(),
                shape={str(k): v.tolist() for k, v in shape.items()},
                converged=bool(conv_sw and conv_tot), steps=int(steps_sw + steps_tot),
                finite=bool(np.isfinite(d_point).all()), per_material=per_material,
                m_loads=m_loads.tolist(), forces=forces.tolist())


# ---------------------------------------------------------------------------
# C0 pre-freeze per-mesh viability probe (NON-scoring) under baseline subtraction
# ---------------------------------------------------------------------------
def stability_probe(intervals=(0.02, 0.01, 0.005)):
    """Per-mesh feasibility of baseline subtraction over the in-regime cohort (B1..B4 + R0/R1/R2;
    B0 excluded by the superposition guard). For each mesh checks both settles converge finite
    and the residual gives a clean cubic (residual/exponent/guard). NON-scoring; writes a log
    and reports the realized mesh sequence for the C1 freeze."""
    cohort = ic.included_cohort()
    raw_es = [r for r, _, _ in cohort]
    b_effs = [b for _, b, _ in cohort]
    labels = [lab for _, _, lab in cohort]

    lines = []

    def log(msg=""):
        lines.append(msg)
        print(msg, flush=True)

    log(f"# IC C0 per-mesh feasibility probe (baseline subtraction)  ({datetime.now(timezone.utc).isoformat()})")
    log("# NON-scoring. Method: arm uniform m0=%.4g; residual = (arm+tip)-(arm) fit to cubic F*ell^3/(3B)." % ic.BASELINE_ARM_MASS)
    log(f"# in-regime cohort (superposition guard, total delta/ell<=0.0625): {labels}  (B0 excluded as regime-of-validity)")
    log("")

    realized = []
    for interval in intervals:
        log(f"[interval {interval}]")
        try:
            res = force_calibrate_baseline(raw_es, b_effs, interval, max_steps=16000)
            clean = res["converged"] and res["finite"]
            all_guard = all(m["guard_ok"] for m in res["per_material"])
            all_super = all(m["superposition_ok"] for m in res["per_material"])
            all_cubic = all(m["max_residual"] <= ic.ACCEPT_RESIDUAL and m["CV"] <= ic.ACCEPT_CV
                            and ic.ACCEPT_EXPONENT_LO <= m["exponent"] <= ic.ACCEPT_EXPONENT_HI
                            for m in res["per_material"])
            for lab, m in zip(labels, res["per_material"]):
                log(f"    {lab} B_eff_force={m['B_eff_force']:.5g}  CV={m['CV']*100:.2f}%  "
                    f"resid={m['max_residual']*100:.2f}%  exp={m['exponent']:.3f}  "
                    f"point/ell(ub)={m['point_ratio_ub']:.4f} guard_ok={m['guard_ok']}  "
                    f"total/ell(ub)={m['total_ratio_ub']:.4f} super_ok={m['superposition_ok']}")
            viable = bool(clean and all_guard and all_super and all_cubic)
            log(f"    -> converged={res['converged']} finite={res['finite']} guard={all_guard} "
                f"superposition={all_super} clean_cubic={all_cubic}  => {'VIABLE' if viable else 'NOT VIABLE'}")
            if viable:
                realized.append(interval)
        except Exception as exc:  # noqa: BLE001
            log(f"    EXCEPTION {type(exc).__name__}: {exc}  => NOT VIABLE (diverged?)")
        log("")

    log("=== realized mesh sequence ===")
    log(f"viable intervals: {realized}")
    if 0.005 in realized:
        naming = "three-level sequence {0.02,0.01,0.005} viable; 'convergence' claimable if the trend holds."
    elif set(realized) >= {0.02, 0.01}:
        naming = ("interval 0.005 NOT viable under the frozen integrator -> freeze the REDUCED sequence "
                  "{0.02,0.01} + a finer-mesh-infeasibility note; naming = 'two-level fixed-discretization "
                  "validation' (convergence only if the two-level trend holds).")
    else:
        naming = "fewer than two viable meshes -> ESCALATION (baseline subtraction unexpectedly non-viable)."
    log("VERDICT: " + naming)

    PROBE_LOG.parent.mkdir(parents=True, exist_ok=True)
    PROBE_LOG.write_text("\n".join(lines) + "\n")
    print(f"\n[probe log -> {PROBE_LOG}]", flush=True)
    return dict(realized=realized, naming=naming)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true", help="run the C0 per-mesh feasibility probe")
    args = ap.parse_args()
    if args.probe:
        stability_probe()
    else:
        print("C0 module: import force_calibrate_baseline/build_baseline_tensors, or run with --probe.")
