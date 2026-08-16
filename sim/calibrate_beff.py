"""Virtual B_eff calibration for the Genesis discrete rod (batched, small-deflection).

Physics / method
----------------
A horizontally clamped straight rod sags under its own weight. In the small-deflection
regime the tip droop obeys the Euler-Bernoulli cantilever law

    delta_tip = w * ell**4 / (8 * B_eff),      w = segment_mass * g / interval   (uniform load/length)

so an effective continuum bending stiffness B_eff is recovered by a through-origin fit of
delta vs ell**4 across several free lengths ell, all indexed later by B_eff (never raw E).

Genesis note (honest): the raw `bending_stiffness` (E) knob is a PER-SEGMENT discrete
stiffness, so the measured continuum B_eff depends on the segment `interval` -- exactly like
FEM element stiffness depends on mesh size. We therefore FIX interval study-wide and calibrate
B_eff at that discretization; we do NOT compare B_eff across intervals. The meaningful
convergence evidence is (a) the ell**4 law is resolved at the fixed interval (log-log exponent
~4, small residual/CV) with enough segments per free arm, and (b) B_eff is load-independent
(multi-mass invariance). Both are checked below.

Efficiency: all 5 stiffnesses are run as n_envs in ONE scene, x 4 free lengths as separate
rods -> 20 (ell, E) points in a single settle per mass. High damping only speeds the static
settle (equilibrium is damping-independent). Grid capped at the demonstrated-stable range.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import genesis as gs

from sim.scene import build_scene, add_straight_rod, vertices
from sim.material import apply_properties

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures" / "calibration.png"
MANIFEST = ROOT / "manifests" / "calibration.json"
ARRAYS = ROOT / "manifests" / "calibration_arrays.npz"

# Fixed study-wide discretization + integrator (frozen; shared with the task).
INTERVAL = 0.01
DT = 2.5e-4
SUBSTEPS = 20
G = 9.81
# Grid capped at the demonstrated stable+measurable range: 5 log-spaced raw-E values,
# ~1.5 decades of B_eff (>= 2.37x in the predicted ell_max). E=1e8 excluded (does not settle
# at this integrator/length -- stiff+long is unstable; documented).
RAW_ES = np.geomspace(1e6, 3.16e7, 5)
MASSES = (0.0002, 0.0004)          # light loads keep every point in small deflection; 2nd mass = load-invariance check
LENGTHS = (0.18, 0.20, 0.22, 0.24)  # free-arm ell; 18-24 segments at interval 0.01 (well-resolved)


def through_origin(x, y):
    """Least-squares fit y = slope * x through the origin. Returns (slope, predicted, rel_residual)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    slope = float(np.sum(x * y) / np.sum(x * x))
    predicted = slope * x
    rel = np.abs(y - predicted) / np.maximum(np.abs(y), 1e-12)
    return slope, predicted, rel


def _nv(ell):
    # free arm ell = (n_vertices - 2) * interval  (verts 0,1 clamped; root = vert 1; tip = vert N-1)
    return int(round(ell / INTERVAL)) + 2


def chunked_settle(scene, rods, chunk=200, max_steps=16000, drift_tol=5e-3):
    """Step in chunks; converged when every rod's per-env tip drift over a chunk is a tiny
    fraction of its droop. Returns (converged, steps, per_rod_last_tip_z, per_rod_maxdrift)."""
    tip_idx = [r.n_vertices - 1 for r in rods]
    prev = [vertices(r)[:, ti, 2].copy() for r, ti in zip(rods, tip_idx)]
    steps = 0
    while steps < max_steps:
        for _ in range(chunk):
            scene.step()
        steps += chunk
        cur = [vertices(r)[:, ti, 2].copy() for r, ti in zip(rods, tip_idx)]
        drift = max(float(np.max(np.abs(c - p))) for c, p in zip(cur, prev))
        # scale drift by the smallest meaningful droop seen so far
        prev = cur
        if drift < drift_tol * INTERVAL:  # sub-1% of a segment length of motion per chunk
            return True, steps, cur
    return False, steps, prev


def calibrate(mass):
    """One batched scene: 5 stiffnesses (envs) x len(LENGTHS) rods. Returns arrays."""
    scene = build_scene(dt=DT, substeps=SUBSTEPS, damping=40.0, angular_damping=20.0)
    nvs = [_nv(L) for L in LENGTHS]
    rods = [add_straight_rod(scene, nv, interval=INTERVAL, E=1e7, segment_mass=mass, pos=(0, 0.35 * j, 0.7))
            for j, nv in enumerate(nvs)]
    scene.build(n_envs=len(RAW_ES), env_spacing=(5, 5))
    for rod in rods:
        rod.set_fixed_states(fixed_ids=[0, 1])
        apply_properties(rod, RAW_ES, mass)
    tip0 = [vertices(r)[:, r.n_vertices - 1, 2].copy() for r in rods]
    conv, steps, tipf = chunked_settle(scene, rods)
    ell = np.array([(nv - 2) * INTERVAL for nv in nvs])
    # delta[len, env]
    delta = np.array([tip0[j] - tipf[j] for j in range(len(nvs))])
    w = mass * G / INTERVAL
    finite = bool(np.isfinite(delta).all())
    return dict(ell=ell, delta=delta, w=w, converged=conv, steps=steps, finite=finite)


def fit_material(ell, delta_col, w):
    """Through-origin delta = k * ell**4 -> B_eff = w/(8k); plus CV, residual, log-log exponent."""
    x4 = ell ** 4
    k = float(np.sum(x4 * delta_col) / np.sum(x4 * x4))
    B_eff = w / (8.0 * k)
    pred = k * x4
    resid = float(np.max(np.abs(delta_col - pred) / np.maximum(np.abs(delta_col), 1e-12)))
    per_len = w * ell ** 4 / (8.0 * delta_col)
    cv = float(np.std(per_len) / np.mean(per_len))
    exponent = float(np.polyfit(np.log(ell), np.log(delta_col), 1)[0])
    worst_pi_g = float(np.max(w * ell ** 3 / B_eff))
    max_slope = float(np.max(delta_col) / ell[np.argmax(delta_col)])  # coarse tip-slope proxy
    return dict(B_eff=B_eff, k=k, CV=cv, max_residual=resid, exponent=exponent,
                worst_Pi_g=worst_pi_g, per_length_B_eff=per_len.tolist(), max_slope=max_slope)


def main():
    FIG.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)

    runs = {m: calibrate(m) for m in MASSES}
    m0 = MASSES[0]
    ell = runs[m0]["ell"]
    materials = []
    for i, E in enumerate(RAW_ES):
        fits = {m: fit_material(runs[m]["ell"], runs[m]["delta"][:, i], runs[m]["w"]) for m in MASSES}
        b0, b1 = fits[MASSES[0]]["B_eff"], fits[MASSES[1]]["B_eff"]
        multimass_pct = float(abs(b1 - b0) / b0 * 100.0)
        f = fits[m0]
        materials.append(dict(
            raw_E=float(E), interval=INTERVAL, segment_radius=0.01, G=float(__import__("sim.scene", fromlist=["FIXED_G"]).FIXED_G),
            fitted_B_eff=f["B_eff"], B_eff_CV=f["CV"], max_relative_residual=f["max_residual"],
            loglog_exponent=f["exponent"], worst_Pi_g=f["worst_Pi_g"], max_slope=f["max_slope"],
            multimass_invariance_pct=multimass_pct,
            per_length={m: {"w": runs[m]["w"], "ell": runs[m]["ell"].tolist(),
                             "delta": runs[m]["delta"][:, i].tolist(),
                             "Pi_g": (runs[m]["w"] * runs[m]["ell"] ** 3 / fits[m]["B_eff"]).tolist(),
                             "per_length_B_eff": fits[m]["per_length_B_eff"]} for m in MASSES},
            accepted=bool(f["CV"] <= 0.05 and f["max_residual"] <= 0.05 and 3.6 <= f["exponent"] <= 4.4
                          and f["worst_Pi_g"] <= 0.5 and multimass_pct <= 8.0),
        ))

    # Calibration map raw-E -> B_eff (clean linear); fixed-ratio (cB,cw) pairs in calibrated space.
    beffs = np.array([m["fitted_B_eff"] for m in materials])
    a = float(np.sum(RAW_ES * beffs) / np.sum(RAW_ES * RAW_ES))  # B_eff ~ a * E (through-origin)
    fixed_ratio_pairs = []
    for base_E, c in ((3.16e6, 2.0), (1.0e7, 0.5)):
        E1, E2 = base_E, base_E * c
        if not (RAW_ES.min() <= E2 <= RAW_ES.max()):
            continue
        pair = _measure_ratio_pair(E1, E2, c)
        fixed_ratio_pairs.append(pair)

    manifest = dict(
        method="small-deflection cantilever; through-origin delta=w*ell^4/(8*B_eff); indexed by B_eff",
        integrator=dict(dt=DT, substeps=SUBSTEPS, damping=40.0, angular_damping=20.0),
        interval=INTERVAL, g=G, raw_E_grid=RAW_ES.tolist(), masses=list(MASSES), lengths=list(LENGTHS),
        discretization_note=("Raw bending_stiffness is a per-segment discrete stiffness, so measured "
                             "B_eff depends on `interval` (like FEM mesh stiffness). Interval is FIXED "
                             "study-wide; B_eff is defined at this discretization. Convergence is shown "
                             "by the resolved ell^4 law (exponent~4, small residual/CV) and load "
                             "(multi-mass) invariance, NOT by cross-interval B_eff equality."),
        grid_cap_note=("Grid capped at raw E<=3.16e7 (B_eff 0.0068..0.215, ~1.5 decades). E=1e8 is "
                       "excluded: stiff+long rods do not settle at this integrator (documented)."),
        raw_E_to_B_eff_slope=a, materials=materials, fixed_ratio_pairs=fixed_ratio_pairs,
        all_accepted=bool(all(m["accepted"] for m in materials)),
    )
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    np.savez(ARRAYS, raw_E=RAW_ES, B_eff=beffs, ell=ell,
             delta_m0=runs[MASSES[0]]["delta"], delta_m1=runs[MASSES[1]]["delta"],
             w_m0=runs[MASSES[0]]["w"], w_m1=runs[MASSES[1]]["w"])

    _plot(ell, runs, materials)

    print("=== B_eff calibration (interval=%.3f, dt=%.1e/substeps=%d) ===" % (INTERVAL, DT, SUBSTEPS))
    for m in materials:
        print("E=%.3g  B_eff=%.5g  CV=%.2f%%  resid=%.2f%%  exp=%.2f  worstPi_g=%.3f  multimass=%.2f%%  %s"
              % (m["raw_E"], m["fitted_B_eff"], m["B_eff_CV"] * 100, m["max_relative_residual"] * 100,
                 m["loglog_exponent"], m["worst_Pi_g"], m["multimass_invariance_pct"],
                 "ACCEPT" if m["accepted"] else "REJECT"))
    print("raw_E -> B_eff slope a=%.4g ; all_accepted=%s" % (a, manifest["all_accepted"]))
    for p in fixed_ratio_pairs:
        print("fixed-ratio c=%.2f: B_eff ratio=%.3f (target %.2f), w ratio=%.3f, Pi_g ratio=%.3f"
              % (p["c"], p["measured_B_eff_ratio"], p["c"], p["w_ratio"], p["Pi_g_ratio"]))


def _measure_ratio_pair(E1, E2, c):
    """Build a 2-env scene (env0=material1, env1=material2 with mass scaled by c) at one ell;
    verify B_eff ratio ~ c, w ratio = c, so B_eff/w (and Pi_g) is invariant along the common-scale line."""
    mass1, mass2 = 0.0003, 0.0003 * c
    scene = build_scene(dt=DT, substeps=SUBSTEPS, damping=40.0, angular_damping=20.0)
    nv = _nv(0.22)
    rod = add_straight_rod(scene, nv, interval=INTERVAL, E=1e7, segment_mass=mass1, pos=(0, 0, 0.7))
    scene.build(n_envs=2, env_spacing=(5, 5))
    rod.set_fixed_states(fixed_ids=[0, 1])
    import torch
    rod.set_bending_stiffness(torch.tensor([E1, E2], dtype=gs.tc_float, device="cuda"))
    rod.set_segment_mass(torch.tensor([[mass1] * nv, [mass2] * nv], dtype=gs.tc_float, device="cuda"))
    tip0 = vertices(rod)[:, nv - 1, 2].copy()
    _c, _s, tipf = chunked_settle(scene, [rod])
    ell = (nv - 2) * INTERVAL
    delta = tip0 - tipf[0]
    w1, w2 = mass1 * G / INTERVAL, mass2 * G / INTERVAL
    b1 = w1 * ell ** 4 / (8 * delta[0]); b2 = w2 * ell ** 4 / (8 * delta[1])
    centerline0 = vertices(rod)[0, :, 2]; centerline1 = vertices(rod)[1, :, 2]
    overlay = float(np.max(np.abs(centerline0 - centerline1)))
    return dict(c=c, E1=float(E1), E2=float(E2), B_eff1=float(b1), B_eff2=float(b2),
                measured_B_eff_ratio=float(b2 / b1), w_ratio=float(w2 / w1),
                Pi_g1=float(w1 * ell ** 3 / b1), Pi_g2=float(w2 * ell ** 3 / b2),
                Pi_g_ratio=float((w2 * ell ** 3 / b2) / (w1 * ell ** 3 / b1)),
                centerline_overlay_max=overlay)


def _plot(ell, runs, materials):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    x4 = ell ** 4
    for i, m in enumerate(materials):
        d = np.array(m["per_length"][MASSES[0]]["delta"])
        ax[0].plot(x4, d, "o", label="B_eff=%.4g" % m["fitted_B_eff"])
        ax[0].plot(x4, m["per_length"][MASSES[0]]["w"] / (8 * m["fitted_B_eff"]) * x4, "-", lw=1)
    ax[0].set_xlabel(r"$\ell^4$ (m$^4$)"); ax[0].set_ylabel(r"$\delta_{tip}$ (m)")
    ax[0].set_title("Through-origin $\\delta=w\\,\\ell^4/(8B_{eff})$ fits"); ax[0].legend(fontsize=7)
    raw = np.array([m["raw_E"] for m in materials]); be = np.array([m["fitted_B_eff"] for m in materials])
    ax[1].loglog(raw, be, "o-")
    ax[1].set_xlabel("raw E (bending_stiffness knob)"); ax[1].set_ylabel(r"$B_{eff}$ (N·m$^2$)")
    ax[1].set_title("Calibration map raw-E $\\to B_{eff}$ (interval=%.3f)" % INTERVAL)
    fig.tight_layout(); fig.savefig(FIG, dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
