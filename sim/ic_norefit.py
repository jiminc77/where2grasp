"""No-refit scoring (§5): B_eff_force is the ONLY calibration input; every target is a
prior committed quantity computed under the gravity pipeline. A miss is a REPORTED NULL.

Families:
 (a) held-out self-weight tip sag  -> calibration.json per-length delta (retrospective,
     in-regime primary) + a prospective new-mass/new-length cohort (scored in C2).
 (b) hardening-A boundary prefactor -> hard_gate_verdict.json boundaries (measured ell_max).
 (c) distal optimum locations       -> distal_sweep_landscape.json measured_argmax_ell.

This module SCORES; it never fits. `beff_force_by_rawE` maps each calibrated raw_E to its
measured B_eff_force (produced in C2 by sim.calibrate_beff_force.force_calibrate).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sim import ic_common as ic

MAN = Path(__file__).resolve().parent / "manifests"


def _load(name):
    return json.loads((MAN / name).read_text())


def _rawE_for_beff(b_eff, tol=1e-6):
    """Map a committed gravity B_eff to its calibrated force raw_E (5 grid + 3 ratio cells) by
    EXACT match. No silent nearest-neighbour fallback: an unknown committed B_eff is a hard
    error, never a guessed cohort mapping."""
    for raw_e, gb in list(zip(ic.RAW_E_GRID, ic.GRAV_B_EFF)) + list(zip(ic.RATIO_RAW_E, ic.RATIO_GRAV_B_EFF)):
        if abs(gb - b_eff) <= tol * max(1.0, abs(gb)):
            return raw_e
    raise ValueError(f"no exact committed B_eff match for {b_eff!r}; refusing to guess a cohort mapping")


def score_sag_retrospective(beff_force_by_rawE, m0=ic.BASELINE_ARM_MASS):
    """(a) retrospective no-refit sag vs calibration.json delta table.

    PRIMARY rows: material in the superposition cohort (B1..B4 + R) AND Pi_g <= 0.5 AND
    mass != m0. The mass == m0 rows are DESCRIPTIVE, not primary: their self-weight settle is
    the subtracted baseline (the gravity channel reused as data), so predicting them would be
    near-circular. Out-of-superposition-regime materials (B0) and Pi_g>0.5 rows are descriptive.
    """
    cal = _load("calibration.json")
    primary, descriptive = [], []
    for mat in cal["materials"]:
        raw_e = mat["raw_E"]
        in_cohort = ic.superposition_included(mat["fitted_B_eff"], m0)
        bff = beff_force_by_rawE.get(raw_e)
        for mass_key, pl in mat["per_length"].items():
            mass = float(mass_key)
            for ell, d_obs, pig in zip(pl["ell"], pl["delta"], pl["Pi_g"]):
                is_baseline_mass = abs(mass - m0) <= 1e-12
                if bff is None or not in_cohort:
                    descriptive.append(dict(raw_E=raw_e, mass=mass, ell=ell, Pi_g=pig,
                                            delta_obs=d_obs, delta_pred=None, rel_err=None,
                                            reason="out-of-superposition-regime (no B_eff_force)"))
                    continue
                d_pred = ic.predict_sag(ell, mass, bff, interval=cal["interval"])
                rel = abs(d_pred - d_obs) / abs(d_obs)
                row = dict(raw_E=raw_e, mass=mass, ell=ell, Pi_g=pig,
                           delta_obs=d_obs, delta_pred=d_pred, rel_err=rel)
                if pig <= ic.PI_G_MAX and not is_baseline_mass:
                    primary.append(row)
                else:
                    row["reason"] = ("baseline-mass (subtracted; near-circular)" if is_baseline_mass
                                     else "out-of-regime Pi_g>0.5")
                    descriptive.append(row)
    max_rel = max((r["rel_err"] for r in primary), default=float("nan"))
    return dict(family="a_retrospective", primary=primary, descriptive=descriptive,
                max_rel_err=max_rel, tol=ic.NOREFIT_SAG_TOL, m0=m0,
                passed=bool(max_rel <= ic.NOREFIT_SAG_TOL))


def score_sag_prospective(beff_force_by_rawE, prospective_rows):
    """(a) prospective score-only cohort: rows = [{raw_E, mass, ell, delta_obs}] from FRESH
    C2 gravity sims accessed only after C1. Predicts from B_eff_force, scores rel error."""
    out = []
    for r in prospective_rows:
        bff = beff_force_by_rawE[r["raw_E"]]
        d_pred = ic.predict_sag(r["ell"], r["mass"], bff)
        out.append(dict(**r, delta_pred=d_pred, rel_err=abs(d_pred - r["delta_obs"]) / abs(r["delta_obs"])))
    max_rel = max((r["rel_err"] for r in out), default=float("nan"))
    return dict(family="a_prospective", rows=out, max_rel_err=max_rel, tol=ic.NOREFIT_SAG_TOL,
                passed=bool(max_rel <= ic.NOREFIT_SAG_TOL))


def score_prefactor(beff_force_by_rawE):
    """(b) K_force = ell_max_measured/(B_eff_force/w)**0.25 for the 15 committed boundaries;
    mean within 0.5180 +/- 0.026."""
    hv = _load("hard_gate_verdict.json")
    per_cell = []
    for b in hv["boundaries"]:
        raw_e = _rawE_for_beff(b["B_eff"])
        bff = beff_force_by_rawE[raw_e]
        K = ic.predict_prefactor_K(b["boundary"], bff, b["w"])
        per_cell.append(dict(id=b["id"], ell_max=b["boundary"], w=b["w"], raw_E=raw_e, K_force=K))
    mean_K = float(np.mean([c["K_force"] for c in per_cell]))
    lo, hi = ic.PREDICTED_PREFACTOR - ic.OBS_B_TOL, ic.PREDICTED_PREFACTOR + ic.OBS_B_TOL
    return dict(family="b_prefactor", per_cell=per_cell, mean_K=mean_K,
                predicted=ic.PREDICTED_PREFACTOR, tol=ic.OBS_B_TOL,
                passed=bool(lo <= mean_K <= hi))


def score_distal(beff_force_by_rawE):
    """(c) s* = (8*B_eff_force*|z*|/w)**0.25 vs distal_sweep_landscape measured_argmax_ell;
    eligibility = measured_feasible cells with a measured argmax; |snap(s*)-target| <= 1 cell."""
    land = _load("distal_sweep_landscape.json")
    per_cell = []
    worst = 0
    for s in land["settings"]:
        if not s.get("measured_feasible") or s.get("measured_argmax_ell") is None:
            continue
        raw_e = _rawE_for_beff(s["B_eff"])
        bff = beff_force_by_rawE[raw_e]
        sstar = ic.predict_distal_sstar(bff, s["w"])
        off = ic.grid_cell_offset(sstar, s["measured_argmax_ell"])
        worst = max(worst, abs(off))
        per_cell.append(dict(id=s["id"], target=s["measured_argmax_ell"], s_star=sstar,
                             snapped=ic.snap_to_grid(sstar), offset_cells=off))
    return dict(family="c_distal", per_cell=per_cell, worst_offset_cells=worst,
                tol_cells=ic.NOREFIT_DISTAL_CELL_TOL,
                passed=bool(worst <= ic.NOREFIT_DISTAL_CELL_TOL))
