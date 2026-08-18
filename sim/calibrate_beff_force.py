"""Force-mode B_eff calibration (independent of self-weight) + the C0 stability probe.

A concentrated TIP point-load is realized by a directly-built 2-D (n_envs, n_vertices)
segment-mass tensor: every free-arm vertex carries eps = f_eps * m_tip, the single tip
vertex carries m_tip, gravity is ON so F = m_tip * g is a clean concentrated END load.
This is NOT apply_properties (whose 1-D path broadcasts a scalar over vertices); the tensor
is built here and pushed through rod.set_segment_mass, with get_all_segment_mass_tc read-back.

Euler-Bernoulli:  delta_tip = F * ell**3 / (3 * B)  ->  B_eff_force = F / (3k), delta = k*ell**3.

C0 usage: this module ships the calibration + probe CODE and the one-shot logged NON-scoring
stability/feasibility probe. It writes NO manifest/figure/data JSON (only the probe log).
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
from sim.material import apply_properties
from sim import ic_common as ic

ROOT = Path(__file__).resolve().parent
PROBE_LOG = ROOT / "logs" / "ic_stability_probe.log"


def build_mass_tensor(nv, m_tips, f_eps):
    """(n_envs, nv) mass tensor: free-arm + clamped verts = f_eps*m_tip, tip vertex = m_tip.

    m_tips : (n_envs,) per-env tip mass. Clamped verts 0,1 are dynamically inert
    (set_fixed_states) so their eps value is irrelevant to deflection; they are set to
    eps for a uniform, read-back-checkable tensor.
    """
    m_tips = np.asarray(m_tips, dtype=float).reshape(-1)
    n_envs = m_tips.size
    mass = np.empty((n_envs, nv), dtype=float)
    mass[:] = (f_eps * m_tips)[:, None]      # eps everywhere
    mass[:, ic.tip_index(nv)] = m_tips        # tip carries the full concentrated load
    return mass


def _settle_drift(scene, rods, chunk=200, max_steps=16000, drift_tol=5e-3, interval=0.01):
    """Chunked static settle keyed on tip-drift (mirrors calibrate_beff.chunked_settle but
    parameterised by interval). Returns (converged, steps, per_rod_tip_z)."""
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
            return True, steps, cur
    return False, steps, prev


def force_calibrate(raw_es, b_eff_sizing, interval, f_eps,
                    lengths=ic.CALIB_LENGTHS, target_ratio=ic.FORCE_TARGET_RATIO,
                    max_steps=16000):
    """Batched force calibration: envs = materials (raw_es), rods = lengths.

    Returns dict with per-(length,material) delta_tip, the settled droop profiles (for
    Observable A), per-material cubic-fit stats + observed guard, and convergence flags.
    m_tip is sized ONCE per material at the upper-bracket ell from b_eff_sizing (prior
    committed gravity B_eff) — F is therefore a known constant per material.
    """
    raw_es = np.asarray(raw_es, dtype=float)
    b_eff_sizing = np.asarray(b_eff_sizing, dtype=float)
    m_tips = np.array([ic.m_tip_for(b, target_ratio=target_ratio) for b in b_eff_sizing])
    forces = m_tips * ic.G
    n_envs = raw_es.size

    scene = build_scene(dt=ic.DT, substeps=ic.SUBSTEPS, damping=ic.DAMPING,
                        angular_damping=ic.ANGULAR_DAMPING)
    nvs = [ic.n_vertices_for(L, interval) for L in lengths]
    rods = [add_straight_rod(scene, nv, interval=interval, E=1e7, segment_mass=1e-4,
                             segment_radius=ic.SEGMENT_RADIUS, G=ic.SHEAR_G, pos=(0, 0.35 * j, 0.7))
            for j, nv in enumerate(nvs)]
    scene.build(n_envs=n_envs, env_spacing=(5, 5))
    for rod, nv in zip(rods, nvs):
        rod.set_fixed_states(fixed_ids=[0, 1])
        rod.set_bending_stiffness(torch.tensor(raw_es, dtype=gs.tc_float, device="cuda"))
        mass2d = build_mass_tensor(nv, m_tips, f_eps)
        rod.set_segment_mass(torch.tensor(mass2d, dtype=gs.tc_float, device="cuda"))
        got = rod.get_all_segment_mass_tc().detach().cpu().numpy()
        assert np.allclose(got, mass2d, rtol=1e-5, atol=1e-12), "segment-mass read-back mismatch (eps floored?)"

    z0 = [vertices(r).copy() for r in rods]
    conv, steps, _ = _settle_drift(scene, rods, max_steps=max_steps, interval=interval)
    zf = [vertices(r).copy() for r in rods]

    ell = np.array([(nv - 2) * interval for nv in nvs])
    delta = np.empty((len(lengths), n_envs))         # [length, material]
    shape = {}                                        # (li) -> normalized droop at OBS_A_FRACS, per env
    for li, nv in enumerate(nvs):
        droop = z0[li][:, :, 2] - zf[li][:, :, 2]     # (n_envs, nv) downward displacement
        delta[li] = droop[:, ic.tip_index(nv)]
        fr = np.array(ic.OBS_A_FRACS)
        idx = np.clip(np.round(1 + fr * (nv - 2)).astype(int), 1, nv - 1)   # arclength fraction -> vertex
        shape[li] = droop[:, idx] / droop[:, ic.tip_index(nv)][:, None]

    per_material = []
    for mi in range(n_envs):
        stats = ic.cubic_fit_stats(ell, delta[:, mi], forces[mi])
        ub = int(np.argmax(ell))                       # upper-bracket length
        stats.update(raw_E=float(raw_es[mi]), m_tip=float(m_tips[mi]), F=float(forces[mi]),
                     observed_deflection_ratio=ic.deflection_ratio(delta[ub, mi], ell[ub]),
                     guard_ok=ic.guard_ok(delta[ub, mi], ell[ub]))
        per_material.append(stats)

    return dict(interval=interval, f_eps=f_eps, ell=ell.tolist(), delta=delta.tolist(),
                shape={str(k): v.tolist() for k, v in shape.items()},
                converged=bool(conv), steps=int(steps), finite=bool(np.isfinite(delta).all()),
                per_material=per_material, m_tips=m_tips.tolist(), forces=forces.tolist())


# ---------------------------------------------------------------------------
# C0 one-shot, logged, NON-scoring stability / feasibility probe
# ---------------------------------------------------------------------------
def _probe_interval(interval, log, ell=ic.UPPER_BRACKET_ELL, max_steps=8000, cap_fraction=0.9):
    """Batched stability probe at one interval across ALL 5 materials at the largest
    admissible f_eps (cap_fraction * contamination cap for THIS mesh -> most stable that
    still keeps contamination <1%). Returns per-material STABLE/UNSTABLE at ell.

    The contamination cap is mesh-dependent (fewer arm masses at coarse meshes), so f_eps
    is chosen per interval, not globally. Larger f_eps == larger arm mass == better
    conditioned mass matrix (the instability is the tip/arm mass ratio, rod_solver.py:625).
    """
    nv = ic.n_vertices_for(ell, interval)
    cap = ic.max_f_eps_for_contamination(nv, interval, cap=0.01)
    f_eps = cap_fraction * cap
    cont = ic.contamination_ratio(nv, interval, f_eps)
    log(f"[interval {interval}]  nv(ell={ell})={nv}  free_verts={len(ic.free_vertex_indices(nv))}  "
        f"contam_cap={cap:.3g}  f_eps={f_eps:.3g} (contamination {cont*100:.3f}%)")
    stable = {}
    try:
        res = force_calibrate(list(ic.RAW_E_GRID), list(ic.GRAV_B_EFF), interval, f_eps,
                              lengths=(ell,), max_steps=max_steps)
        delta = np.asarray(res["delta"])[0]                # (n_materials,)
        for mi, (raw_e, m) in enumerate(zip(ic.RAW_E_GRID, res["per_material"])):
            d = float(delta[mi])
            ratio = d / ell if np.isfinite(d) else float("nan")
            ok = bool(np.isfinite(d) and d > 0 and ratio <= ic.GUARD_DEFLECTION_RATIO)
            stable[mi] = ok
            log(f"    B_eff~{ic.GRAV_B_EFF[mi]:.4g} (raw_E={raw_e:.3g}): delta={d:.4g}  "
                f"delta/ell={ratio:.4f}  -> {'STABLE' if ok else 'UNSTABLE'}")
        log(f"    scene converged={res['converged']} steps={res['steps']}")
    except Exception as exc:  # noqa: BLE001
        log(f"    EXCEPTION {type(exc).__name__}: {exc}")
        stable = {mi: False for mi in range(len(ic.RAW_E_GRID))}
    return dict(interval=interval, nv=nv, cap=cap, f_eps=f_eps, contamination=cont, stable=stable)


def stability_probe(intervals=(0.02, 0.01, 0.005, 0.0075)):
    """Rigorous mesh x material stability/feasibility probe (NON-scoring).

    For each interval (the 3 mesh levels + 0.0075 substitution candidate), tests the
    largest admissible f_eps against ALL 5 materials at the upper-bracket length. Reports
    per (interval, material) stability, then classifies:
      - reference interval 0.01 stable for every material          -> option (iv) viable
      - only interval 0.005 fails (some materials)                 -> substitution branch
      - reference interval fails for >=1 material                  -> ESCALATION / STOP
    Writes the probe log; freezes nothing.
    """
    lines = []

    def log(msg=""):
        lines.append(msg)
        print(msg, flush=True)

    log(f"# IC C0 stability/feasibility probe  ({datetime.now(timezone.utc).isoformat()})")
    log("# NON-scoring: no headline observable is measured or frozen here.")
    log("# Option (iv): concentrated tip point-load, arm mass = f_eps*m_tip, gravity ON.")
    log("# Instability source = tip/arm mass ratio (rod_solver.py:625 gradient/mass); f_eps")
    log("# is chosen per-mesh at the largest value keeping exact-discrete contamination <1%.")
    log("")

    results = {}
    for interval in intervals:
        results[interval] = _probe_interval(interval, log)
        log("")

    def all_stable(interval):
        s = results[interval]["stable"]
        return all(s.values()) and len(s) == len(ic.RAW_E_GRID)

    def stable_materials(interval):
        return sorted(mi for mi, ok in results[interval]["stable"].items() if ok)

    ref_ok = all_stable(0.01)
    coarse_ok = all_stable(0.02)
    fine_ok = all_stable(0.005)
    sub_ok = all_stable(0.0075)

    log("=== classification ===")
    log(f"interval 0.02  all-materials-stable: {coarse_ok}  stable={stable_materials(0.02)}")
    log(f"interval 0.01  all-materials-stable: {ref_ok}  stable={stable_materials(0.01)}")
    log(f"interval 0.005 all-materials-stable: {fine_ok}  stable={stable_materials(0.005)}")
    log(f"interval 0.0075 (subst) all-stable:  {sub_ok}  stable={stable_materials(0.0075)}")
    log("")

    if not ref_ok:
        verdict = ("ESCALATION: option (iv) is NOT viable at the REFERENCE interval 0.01 for every "
                   "required material (the tip/arm mass-ratio instability vs the <1% contamination "
                   "cap leaves an EMPTY f_eps window). No executable fallback (option iii dropped) "
                   "-> STOP for owner ruling.")
    elif fine_ok:
        verdict = "OPTION (iv) VIABLE at all three mesh levels {0.02,0.01,0.005}; no substitution needed."
    elif sub_ok:
        verdict = ("OPTION (iv) VIABLE with SUBSTITUTION: interval 0.005 is infeasible for some "
                   "materials; the frozen finest level becomes 0.0075 (all-materials-stable). "
                   "Mesh naming stays 'fixed-discretization validation' unless the 3-level trend holds.")
    else:
        # 0.005 and 0.0075 both fail for some materials, but 0.01 & 0.02 hold -> drop-finest branch
        verdict = ("OPTION (iv) VIABLE at {0.02,0.01} only; the finest level is INFEASIBLE for some "
                   "materials even with the 0.0075 substitution -> frozen branch = DROP the finest "
                   "level (2-level fixed-discretization validation over the common cohort). If >1 "
                   "material lacks a 3rd level this is Escalation (c) -> owner ruling.")
    log("VERDICT: " + verdict)

    PROBE_LOG.parent.mkdir(parents=True, exist_ok=True)
    PROBE_LOG.write_text("\n".join(lines) + "\n")
    print(f"\n[probe log written -> {PROBE_LOG}]", flush=True)
    return dict(results={k: v["stable"] for k, v in results.items()},
                ref_ok=ref_ok, coarse_ok=coarse_ok, fine_ok=fine_ok, sub_ok=sub_ok,
                verdict=verdict)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true", help="run the C0 stability/feasibility probe")
    args = ap.parse_args()
    if args.probe:
        stability_probe()
    else:
        print("C0 module: import for force_calibrate/build_mass_tensor, or run with --probe.")
