"""Observable B: a MEASURED distributed-gravity droop-clear boundary extractor.

This is a SEPARATE extractor and deliberately does NOT import or reuse the frozen
`analyze_gate` surface. On a common cohort it runs a real distributed self-weight
settle sweep over the frozen ell grid [0.12, 0.60] step 0.03 at a given mesh interval,
measures the settled tip droop, and extracts the boundary as the largest free length
whose droop clears the fixed step h=0.009 (linear-interpolated crossing). The measured
mesh prefactor is K_N = ell_boundary / (B_eff_force/w)**0.25.
"""
from __future__ import annotations

import numpy as np
import torch
import genesis as gs

from sim.scene import build_scene, add_straight_rod, vertices
from sim import ic_common as ic


def gravity_droop_sweep(raw_es, interval, mass=ic.OBS_B_REF_MASS, ells=ic.GRAV_SWEEP_ELL,
                        max_steps=16000):
    """Distributed self-weight settle sweep. envs = materials (raw_es), rods = ells.

    Returns dict: ells, droop[ell_idx, material] (settled downward tip displacement), w, flags.
    Uniform segment_mass (true distributed self-weight); bending stiffness per env.
    """
    raw_es = np.asarray(raw_es, dtype=float)
    n_envs = raw_es.size
    scene = build_scene(dt=ic.DT, substeps=ic.SUBSTEPS, damping=ic.DAMPING,
                        angular_damping=ic.ANGULAR_DAMPING)
    nvs = [ic.n_vertices_for(L, interval) for L in ells]
    rods = [add_straight_rod(scene, nv, interval=interval, E=1e7, segment_mass=mass,
                             segment_radius=ic.SEGMENT_RADIUS, G=ic.SHEAR_G, pos=(0, 0.3 * j, 0.8))
            for j, nv in enumerate(nvs)]
    scene.build(n_envs=n_envs, env_spacing=(6, 6))
    for rod, nv in zip(rods, nvs):
        rod.set_fixed_states(fixed_ids=[0, 1])
        rod.set_bending_stiffness(torch.tensor(raw_es, dtype=gs.tc_float, device="cuda"))
        rod.set_segment_mass(torch.tensor(np.full((n_envs, nv), mass), dtype=gs.tc_float, device="cuda"))

    z0 = [vertices(r)[:, ic.tip_index(r.n_vertices), 2].copy() for r in rods]
    conv, steps, per_ell_env = _settle(scene, rods, max_steps=max_steps, interval=interval)
    zf = [vertices(r)[:, ic.tip_index(r.n_vertices), 2].copy() for r in rods]

    droop = np.array([z0[i] - zf[i] for i in range(len(ells))])   # [ell, material]
    w = mass * ic.G / interval
    return dict(ells=list(ells), droop=droop.tolist(), w=float(w), interval=interval,
                converged=bool(conv), steps=int(steps), finite=bool(np.isfinite(droop).all()),
                per_ell_env_converged=per_ell_env.tolist())      # (n_ell, n_material) settle proof


def _settle(scene, rods, chunk=200, max_steps=16000, drift_tol=5e-3, interval=0.01):
    """Chunked settle; tracks PER-(rod, env) convergence so each boundary's bracketing samples can
    be proven settled (a global flag cannot distinguish an unsettled bracket rod from an unsettled
    long post-boundary rod). Returns (all_converged, steps, per_ell_env_converged[(n_rods,n_envs)])."""
    tips = [ic.tip_index(r.n_vertices) for r in rods]
    prev = [vertices(r)[:, ti, 2].copy() for r, ti in zip(rods, tips)]
    n_envs = prev[0].shape[0]
    conv = np.zeros((len(rods), n_envs), dtype=bool)
    steps = 0
    while steps < max_steps and not conv.all():
        for _ in range(chunk):
            scene.step()
        steps += chunk
        cur = [vertices(r)[:, ti, 2].copy() for r, ti in zip(rods, tips)]
        for i, (c, p) in enumerate(zip(cur, prev)):
            conv[i] = conv[i] | (np.abs(c - p) < drift_tol * interval)
        prev = cur
    return bool(conv.all()), steps, conv


def boundary_bracket(ells, droops_one_material, h=ic.DROOP_CLEAR_H):
    """Grid indices (k, k+1) bracketing the droop=h crossing (droops ascending in ell), or None."""
    droops = np.asarray(droops_one_material, dtype=float)
    for k in range(len(droops) - 1):
        if droops[k] <= h < droops[k + 1]:
            return (k, k + 1)
    return None


def extract_boundary_and_K(ells, droops_one_material, b_eff_force, w=ic.OBS_B_REF_W,
                           h=ic.DROOP_CLEAR_H):
    """ell_boundary (largest length whose droop clears h) + measured K_N. None if no crossing."""
    ell_boundary = ic.observable_B_boundary(ells, droops_one_material, h=h)
    if ell_boundary is None:
        return dict(ell_boundary=None, K_N=None)
    return dict(ell_boundary=ell_boundary, K_N=ic.prefactor_K(ell_boundary, b_eff_force, w=w))
