"""C0 tests for the Item-2 r_N-corrected no-refit graduation gate (sim/rn_gate.py).

Pure-math / synthetic / prior-committed-fixture only. NO simulator execution on the cohort
(the held-out prospective settle happens in C2, after the C1 freeze).
"""
import json
from pathlib import Path

import numpy as np
import pytest

from sim import ic_common as ic
from sim import rn_gate

MAN = Path(__file__).resolve().parents[1] / "manifests"


def test_attribution_reproduces_committed_finding5():
    """(a) The gate reproduces the committed finding-5 aggregate attribution from committed numbers:
    calibration-window mean r_N and the residual of the mean force-vs-gravity gap above it."""
    got = rn_gate.reproduce_attribution()
    committed = json.loads((MAN / "independent_closure_verdict.json").read_text())["finding5_attribution"]
    assert got["calibration_window_mean_rN_pct"] == pytest.approx(
        committed["calibration_window_mean_rN_pct"], abs=1e-6)
    assert got["residual_above_rN_pct"] == pytest.approx(committed["residual_above_rN_pct"], abs=1e-6)
    # sanity vs the plan's frozen headline numbers
    assert got["calibration_window_mean_rN_pct"] == pytest.approx(6.5008, abs=1e-3)
    assert got["residual_above_rN_pct"] == pytest.approx(1.3834, abs=1e-3)


def test_prospective_cohort_disjoint_and_nominal_equals_realized():
    """(b) The held-out prospective cohort has realized N={15,16,17}, nominal==realized, and is
    disjoint from the calibration N{18,20,22,24} and the prior-prospective N{19,21,23}."""
    d = rn_gate.prospective_disjoint()
    assert d["new_N"] == [15, 16, 17]
    assert d["nominal_eq_realized"] is True
    assert d["disjoint"] is True
    assert d["calib_N"] == [18, 20, 22, 24] and d["prior_N"] == [19, 21, 23]


def test_guard_reproduces_cohort_and_prospective_in_regime():
    """(c) The analytic superposition guard reproduces the committed cohort (B0 excluded), and every
    prospective cell (mass 0.00025 / ell{0.15,0.16,0.17}) is in-regime (Pi_g<=0.5)."""
    assert not ic.superposition_included(ic.GRAV_B_EFF[0])                 # B0 softest OUT
    assert all(ic.superposition_included(b) for b in ic.GRAV_B_EFF[1:])    # B1..B4 IN
    assert all(ic.superposition_included(b) for b in ic.RATIO_GRAV_B_EFF)  # R0/R1/R2 IN
    bff = rn_gate.committed_beff_force()
    for lab in rn_gate.COHORT_LABELS:
        for ell in rn_gate.PROSPECTIVE_LENGTHS:
            assert rn_gate.cell_in_regime(rn_gate.PROSPECTIVE_MASS, ell, bff[lab]), (lab, ell)
    # softest in-cohort B1 has the largest Pi_g; still well under 0.5 (~0.07)
    pg = ic.pi_g(ic.w_of_mass(rn_gate.PROSPECTIVE_MASS), 0.17, bff["B1"])
    assert pg < 0.1


def test_corrected_predictor_is_rN_times_uncorrected():
    """The corrected predictor is exactly r_N(N) * the uncorrected B_eff_force sag law."""
    bff = rn_gate.committed_beff_force()["B3"]
    for ell in (0.15, 0.18, 0.24):
        N = ic.segment_count(ell, ic.REFERENCE_INTERVAL)
        uncorr = ic.predict_sag(ell, 0.00025, bff)
        assert rn_gate.corrected_predict(ell, 0.00025, bff) == pytest.approx(ic.r_N(N) * uncorr, rel=1e-12)
    # r_N > 1 -> the correction INCREASES the (previously under-) prediction toward the measured sag
    assert ic.r_N(24) > 1.0


def test_retrospective_committed_window_passes_under_5pct():
    """(committed fixture) On the calibration window (calibration.json gravity sag, mass 0.0004), the
    r_N-corrected no-refit error for B1..B4 is <= 5% — the correction rescues the uncorrected 6-11% miss.
    No new simulation: calibration.json is a prior-committed artifact."""
    cal = json.loads((MAN / "calibration.json").read_text())
    bff = rn_gate.committed_beff_force()
    raw_for = {f"B{i}": ic.RAW_E_GRID[i] for i in range(1, 5)}
    worst = 0.0
    for lab in ("B1", "B2", "B3", "B4"):
        mat = next(m for m in cal["materials"] if abs(m["raw_E"] - raw_for[lab]) < 1e-6)
        pl = mat["per_length"]["0.0004"]
        for ell, dobs, pig in zip(pl["ell"], pl["delta"], pl["Pi_g"]):
            if pig > ic.PI_G_MAX:
                continue
            c = rn_gate.score_cell(ell, 0.0004, bff[lab], dobs)
            assert c["within_bound"], (lab, ell, c["rel_err"])
            worst = max(worst, c["rel_err"])
    assert worst <= ic.NOREFIT_SAG_TOL and worst < 0.05


def test_outcome_binding_no_reduced_subset_pass():
    """(d) Outcome-binding: a single non-finite/non-converged in-regime cell forces INCONCLUSIVE
    (never thinned to a reduced-subset PASS); a genuine >5% is a MISS; all within-bound is PASS."""
    bff = rn_gate.committed_beff_force()["B2"]
    good = rn_gate.score_cell(0.16, 0.00025, bff, rn_gate.corrected_predict(0.16, 0.00025, bff))  # exact -> 0 err
    assert good["within_bound"] and good["rel_err"] == pytest.approx(0.0, abs=1e-12)
    diverged = rn_gate.score_cell(0.15, 0.00025, bff, float("nan"))                                # non-finite
    over = rn_gate.score_cell(0.17, 0.00025, bff, rn_gate.corrected_predict(0.17, 0.00025, bff) * 1.2)  # +20%
    assert rn_gate.prong_verdict([good])["verdict"] == "PASS"
    assert rn_gate.prong_verdict([good, over])["verdict"] == "MISS"
    # a diverged in-regime cell forces INCONCLUSIVE even though `good` passes -> no reduced-subset PASS
    assert rn_gate.prong_verdict([good, diverged])["verdict"] == "INCONCLUSIVE"
    assert rn_gate.prong_verdict([good, over, diverged])["verdict"] == "INCONCLUSIVE"


def test_claim_boundary_verbatim_plus_rN_clause():
    """The mandatory claim boundary carries over VERBATIM + the r_N clause."""
    m = json.loads((MAN / "independent_closure_manifest.json").read_text())
    verbatim = m["claim_language_mandatory_status"]
    got = rn_gate.claim_language()
    assert got.startswith(verbatim)
    assert got.endswith("now r_N-corrected via the analytical lumped-mass factor")
    assert "same-simulator, different-load-law" in got and "baseline-subtracted" in got
