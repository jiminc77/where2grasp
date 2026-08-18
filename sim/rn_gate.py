"""Item 2 (simulation wrap-up): the r_N-corrected no-refit graduation gate.

Pre-registered follow-up to the independent mechanics closure. The direct-sag no-refit prong
MISSED the frozen 5% bound because B_eff_force sits ~+7.9% above gravity B_eff. Finding 5 attributes
most of that gap to the analytical lumped-mass quadrature factor r_N=(N+1)(3N+1)/(3N^2). This gate
predicts the gravity self-weight sag from B_eff_force CORRECTED by r_N, with NO refit, against the
UNCHANGED 5% bound (ic.NOREFIT_SAG_TOL), on the committed calibration window (retrospective) AND a
genuinely held-out prospective cohort (mass 0.00025 / ell{0.15,0.16,0.17}, N{15,16,17}).

Scope (graduation): a PASS graduates ONLY the family-a direct-sag no-refit prong (the component that
missed). The full conjunctive closure remains bounded by the separately-reported, UNCHANGED mesh
B4@0.02 Observable-B INCONCLUSIVE. The mandatory claim boundary carries over verbatim, now
r_N-corrected. The ~1.38% residual above the calibration-window mean r_N stays an OPEN item either
way. Every frozen in-regime cell is OUTCOME-BINDING: a post-freeze divergence/non-convergence/
out-of-guard makes that cell + the prong INCONCLUSIVE + STOP; a genuine >5% is a MISS; never thinned
to a reduced-subset PASS.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sim import ic_common as ic

ROOT = Path(__file__).resolve().parent
MAN = ROOT / "manifests"

# --- held-out prospective cohort (frozen; nominal==realized; disjoint from calib N{18,20,22,24}
#     and prior-prospective N{19,21,23}) ---
PROSPECTIVE_MASS = 0.00025
PROSPECTIVE_LENGTHS = (0.15, 0.16, 0.17)          # N = round(ell/0.01) = {15, 16, 17}

# --- in-regime cohort: B1..B4 + R0/R1/R2 (B0 excluded), label -> (gravity B_eff, B_eff_force) ---
COHORT_LABELS = ("B1", "B2", "B3", "B4", "R0", "R1", "R2")


def committed_beff_force():
    """Per-material B_eff_force from the committed independent-closure verdict (the source of truth)."""
    v = json.loads((MAN / "independent_closure_verdict.json").read_text())
    per = v["finding5_attribution"]["per_material"]
    return {lab: float(per[lab]["B_eff_force"]) for lab in COHORT_LABELS}


def committed_gravity_beff():
    """Per-material committed gravity B_eff (B1..B4 from GRAV_B_EFF[1:], R0/R1/R2 from RATIO_GRAV_B_EFF)."""
    g = {f"B{i}": ic.GRAV_B_EFF[i] for i in range(1, 5)}
    g.update({rid: b for rid, b in zip(ic.RATIO_IDS, ic.RATIO_GRAV_B_EFF)})
    return {lab: float(g[lab]) for lab in COHORT_LABELS}


def corrected_predict(ell, mass, b_eff_force, interval=ic.REFERENCE_INTERVAL):
    """r_N-corrected no-refit self-weight sag: delta_pred = r_N(N) * w * ell^4 / (8 * B_eff_force),
    i.e. the quartic sag law with B_eff_force DIVIDED by the analytical lumped-mass factor r_N(N),
    N = round(ell/interval). No refit: r_N and B_eff_force are both fixed from committed numbers."""
    rN = ic.r_N(ic.segment_count(ell, interval))
    return rN * ic.predict_sag(ell, mass, b_eff_force, interval=interval)


def cell_in_regime(mass, ell, b_eff_force, interval=ic.REFERENCE_INTERVAL):
    """In-regime iff the gravity small-deflection number Pi_g = w*ell^3/B_eff_force <= 0.5
    (equivalently delta/ell = Pi_g/8 <= 0.0625). Screened ANALYTICALLY from committed B_eff."""
    w = ic.w_of_mass(mass, interval)
    return ic.pi_g(w, ell, b_eff_force) <= ic.PI_G_MAX


def score_cell(ell, mass, b_eff_force, delta_obs, interval=ic.REFERENCE_INTERVAL):
    """One outcome-binding cell: corrected prediction vs measured gravity sag. finite/convergence is
    the caller's responsibility (delta_obs must be a converged finite settle)."""
    d_pred = corrected_predict(ell, mass, b_eff_force, interval)
    finite = bool(np.isfinite(delta_obs) and np.isfinite(d_pred) and abs(delta_obs) > 0)
    rel = float(abs(d_pred - delta_obs) / abs(delta_obs)) if finite else float("inf")
    return dict(ell=float(ell), mass=float(mass), N=ic.segment_count(ell, interval),
                b_eff_force=float(b_eff_force), delta_pred=float(d_pred), delta_obs=float(delta_obs),
                pi_g=float(ic.pi_g(ic.w_of_mass(mass, interval), ell, b_eff_force)),
                in_regime=cell_in_regime(mass, ell, b_eff_force, interval),
                finite=finite, rel_err=rel, within_bound=bool(finite and rel <= ic.NOREFIT_SAG_TOL))


def prospective_keys():
    """The exact FROZEN prospective cell inventory: (label, N) for every cohort material x length."""
    return [(lab, ic.segment_count(ell, ic.REFERENCE_INTERVAL))
            for lab in COHORT_LABELS for ell in PROSPECTIVE_LENGTHS]


def prong_verdict(cells, expected_prospective_keys=None, tol=ic.NOREFIT_SAG_TOL):
    """Outcome-binding prong verdict over the frozen eligible cells (retrospective + prospective).

    The FROZEN-ELIGIBLE prospective cohort must be present EXACTLY, and every prospective cell must be
    in-regime AND finite: a post-freeze out-of-guard / non-converged / non-finite cell is a genuine
    breakdown -> INCONCLUSIVE + STOP, NEVER filtered/thinned to a reduced-subset or empty PASS.

    PASS  : inventory intact AND every in-regime cell (prospective + retrospective) is within tol.
    INCONCLUSIVE : inventory mismatch, any frozen prospective cell out-of-guard/non-finite, any
                   in-regime cell non-finite, or an empty in-regime set.
    MISS  : all eligible cells finite/in-regime but at least one exceeds tol (a genuine >5% outcome).
    """
    prosp = [c for c in cells if c.get("cohort") == "prospective"]
    integrity_fail = []
    if expected_prospective_keys is not None:
        exp = {tuple(k) for k in expected_prospective_keys}
        got = {(c["label"], c["N"]) for c in prosp}
        # exact key set AND exact row count (reject duplicate rows, not just a matching set)
        if got != exp or len(prosp) != len(expected_prospective_keys):
            integrity_fail.append(f"prospective_inventory_mismatch rows={len(prosp)} keys={len(got)} exp={len(exp)}")
    prosp_breakdown = [(c["label"], c["N"]) for c in prosp if not c["in_regime"] or not c["finite"]]
    in_regime = [c for c in cells if c["in_regime"]]
    nonfinite = [c for c in in_regime if not c["finite"]]
    if integrity_fail or prosp_breakdown or nonfinite or not in_regime:
        verdict = "INCONCLUSIVE"
    elif all(c["within_bound"] for c in in_regime):
        verdict = "PASS"
    else:
        verdict = "MISS"
    worst = max((c["rel_err"] for c in in_regime if c["finite"]), default=float("nan"))
    return dict(verdict=verdict, tol=tol, n_in_regime=len(in_regime), n_prospective=len(prosp),
                integrity_ok=bool(not integrity_fail and not prosp_breakdown),
                integrity_detail=(integrity_fail + [f"breakdown:{b}" for b in prosp_breakdown]) or "ok",
                worst_rel_err=float(worst), cells=cells,
                graduates_direct_sag_prong=bool(verdict == "PASS"),
                aggregate_note=("PASS graduates ONLY the family-a direct-sag no-refit prong; the full "
                                "conjunctive closure remains bounded by the separately-reported, "
                                "UNCHANGED mesh B4@0.02 Observable-B INCONCLUSIVE."))


# --------------------------------------------------------------------------- #
# attribution reproduction (C0 test target): the finding-5 aggregate diagnostic
# --------------------------------------------------------------------------- #
def reproduce_attribution():
    """Reproduce the committed finding-5 aggregate attribution from committed numbers:
    calibration-window mean r_N and the residual of the mean force-vs-gravity gap above it."""
    bff = committed_beff_force(); gb = committed_gravity_beff()
    gaps = [(bff[lab] / gb[lab] - 1.0) * 100.0 for lab in COHORT_LABELS]
    mean_gap = float(np.mean(gaps))
    mean_rN_pct = (float(np.mean([ic.r_N(ic.segment_count(L, ic.REFERENCE_INTERVAL))
                                  for L in ic.CALIB_LENGTHS])) - 1.0) * 100.0
    residual = mean_gap - mean_rN_pct
    return dict(mean_gap_pct=mean_gap, calibration_window_mean_rN_pct=mean_rN_pct,
                residual_above_rN_pct=residual, per_material_gap_pct=dict(zip(COHORT_LABELS, gaps)))


def prospective_disjoint():
    """Assert the prospective cohort's realized N is disjoint from calibration + prior prospective,
    and nominal==realized (ell == N*interval exactly)."""
    calib_N = {ic.segment_count(L, ic.REFERENCE_INTERVAL) for L in ic.CALIB_LENGTHS}          # {18,20,22,24}
    prior_N = {ic.segment_count(L, ic.REFERENCE_INTERVAL) for L in ic.PROSPECTIVE_LENGTHS}    # {19,21,23}
    new_N = [ic.segment_count(L, ic.REFERENCE_INTERVAL) for L in PROSPECTIVE_LENGTHS]         # {15,16,17}
    nominal_eq_realized = all(abs(L - N * ic.REFERENCE_INTERVAL) < 1e-12
                              for L, N in zip(PROSPECTIVE_LENGTHS, new_N))
    disjoint = set(new_N).isdisjoint(calib_N | prior_N)
    return dict(new_N=new_N, calib_N=sorted(calib_N), prior_N=sorted(prior_N),
                nominal_eq_realized=nominal_eq_realized, disjoint=disjoint)


CLAIM_BOUNDARY_SUFFIX = " ; now r_N-corrected via the analytical lumped-mass factor"


def claim_language():
    """The mandatory claim boundary carried over VERBATIM from the independent-closure manifest,
    plus the r_N-corrected clause."""
    m = json.loads((MAN / "independent_closure_manifest.json").read_text())
    return m["claim_language_mandatory_status"] + CLAIM_BOUNDARY_SUFFIX
