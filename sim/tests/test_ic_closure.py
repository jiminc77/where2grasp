"""C0 pre-data tests for the independent mechanics closure. NO data/manifest/figures.

Pure tests (no Genesis) validate every estimator/scorer/guard/index against hand-checked
answers and the committed manifests. Two Genesis integration tests validate the 2-D
mass-tensor read-back (epsilon not floored) and the finding-5 broadcast premise.
"""
from __future__ import annotations

import numpy as np
import pytest

from sim import ic_common as ic
from sim import ic_norefit


# --------------------------------------------------------------------------- pure
def test_nvertices_and_free_index_convention():
    assert ic.n_vertices_for(0.18, 0.02) == 11
    assert ic.n_vertices_for(0.24, 0.01) == 26
    assert ic.n_vertices_for(0.24, 0.005) == 50
    nv = ic.n_vertices_for(0.24, 0.005)
    N = ic.segment_count(0.24, 0.005)          # 48
    assert nv == N + 2
    free = ic.free_vertex_indices(nv)
    assert free[0] == 2 and free[-1] == nv - 1  # 2 .. N+1
    assert ic.tip_index(nv) == nv - 1
    assert len(free) == N                       # N free (non-clamped) vertices


def test_cubic_fit_recovers_known_B_exact():
    B_true, F = 0.0382, 0.0597
    ell = np.array(ic.CALIB_LENGTHS)
    delta = F * ell ** 3 / (3.0 * B_true)       # exact cubic
    k = ic.through_origin_cubic(ell, delta)
    assert abs(ic.b_eff_force_from_fit(F, k) - B_true) / B_true < 1e-9
    stats = ic.cubic_fit_stats(ell, delta, F)
    assert stats["max_residual"] < 1e-9 and abs(stats["exponent"] - 3.0) < 1e-6
    assert stats["CV"] < 1e-9


def test_contamination_exact_below_one_percent():
    nv = ic.n_vertices_for(0.24, 0.005)         # finest mesh, worst case (nv=50)
    cap = ic.max_f_eps_for_contamination(nv, 0.005, cap=0.01)
    # at the cap the exact discrete contamination is <= 1% (strictly, since discrete < uniform bound)
    assert ic.contamination_ratio(nv, 0.005, cap) <= 0.01 + 1e-12
    # a comfortably-inside f_eps stays well under 1%
    assert ic.contamination_ratio(nv, 0.005, cap * 0.5) < 0.006
    # contamination scales linearly in f_eps
    assert abs(ic.contamination_ratio(nv, 0.005, 2e-4) - 2 * ic.contamination_ratio(nv, 0.005, 1e-4)) < 1e-12


def test_guard_upper_bracket_logic():
    # delta/ell = 0.0625 is exactly the cap; just above fails, just below passes
    ell = 0.24
    assert ic.guard_ok(0.0625 * ell, ell)
    assert not ic.guard_ok(0.0625 * ell * 1.0001, ell)
    assert ic.GUARD_DEFLECTION_RATIO == pytest.approx(ic.PI_G_MAX / 8.0)
    # m_tip sizing: F = 1.5625*B at ell=0.24, target 0.03
    B = 0.091106
    assert ic.m_tip_for(B) * ic.G == pytest.approx(1.5625 * B, rel=1e-9)


def test_rN_formula_and_limit():
    assert ic.r_N(1) == pytest.approx(2.0 * 4.0 / 3.0)      # (2)(4)/3
    assert ic.r_N(2) == pytest.approx(3.0 * 7.0 / 12.0)     # (3)(7)/(3*4)
    assert ic.r_N(1000) > 1.0 and ic.r_N(1000) < 1.01       # -> 1 from above
    assert ic.r_N(24) > ic.r_N(48)                          # decreases with refinement


def test_endload_shape_and_observable_A():
    assert ic.endload_shape(0.0) == 0.0
    assert ic.endload_shape(1.0) == pytest.approx(1.0)
    perfect = ic.endload_shape(np.array(ic.OBS_A_FRACS))
    assert ic.observable_A_error(perfect) < 1e-12
    perturbed = perfect + np.array([0.0, 0.03, -0.01, 0.0, 0.0])
    assert ic.observable_A_error(perturbed) == pytest.approx(0.03)


def test_observable_B_crossing_and_none():
    ells = np.array(ic.GRAV_SWEEP_ELL)
    # monotone droop crossing h between two grid points -> linear interp
    droops = 0.009 * (ells / 0.30) ** 4          # crosses h=0.009 at ell=0.30 exactly
    b = ic.observable_B_boundary(ells, droops, h=0.009)
    assert b == pytest.approx(0.30, abs=1e-9)
    # no interior crossing (all clear) -> None
    assert ic.observable_B_boundary(ells, np.full_like(ells, 0.001)) is None
    assert ic.prefactor_K(0.5180 * (0.091106 / 0.1962) ** 0.25, 0.091106, 0.1962) == pytest.approx(0.5180, rel=1e-9)


def test_no_refit_scorers_hand_checked():
    # sag: delta = w*ell^4/(8B), w = m*g/interval
    d = ic.predict_sag(0.24, 0.0002, 0.01610520612280113)
    w = 0.0002 * ic.G / 0.01
    assert d == pytest.approx(w * 0.24 ** 4 / (8 * 0.01610520612280113), rel=1e-12)
    # prefactor: exact inverse of K definition
    assert ic.predict_prefactor_K(0.345, 0.038259, 0.1962) == pytest.approx(0.345 / (0.038259 / 0.1962) ** 0.25)
    # distal s*
    assert ic.predict_distal_sstar(0.21527426878237318, 0.1962) == pytest.approx(
        (8 * 0.21527426878237318 * 0.012 / 0.1962) ** 0.25)
    assert ic.snap_to_grid(0.5697) == 0.56 and ic.snap_to_grid(0.371) == 0.38
    assert ic.grid_cell_offset(0.5697, 0.58) == -1


def test_convergence_trend_rule():
    assert ic.convergence_trend_ok([0.015, 0.008], tol=0.02)          # decreasing + under tol
    assert not ic.convergence_trend_ok([0.008, 0.015], tol=0.02)      # increasing -> not converged
    assert not ic.convergence_trend_ok([0.03, 0.01], tol=0.02)        # first over tol


def test_superposition_guard_and_cohort():
    """The linear-superposition inclusion guard (Condition 1) excludes the softest B0 as
    regime-of-validity and admits B1..B4 + R0/R1/R2 (hardening-A's in-regime grid)."""
    assert not ic.superposition_included(ic.GRAV_B_EFF[0])          # B0 softest OUT
    assert all(ic.superposition_included(b) for b in ic.GRAV_B_EFF[1:])
    assert all(ic.superposition_included(b) for b in ic.RATIO_GRAV_B_EFF)
    labels = [lab for _, _, lab in ic.included_cohort()]
    assert "B0" not in labels and set(labels) == {"B1", "B2", "B3", "B4", "R0", "R1", "R2"}
    # B0 total delta/ell exceeds the 0.0625 cap; B1 is inside
    assert ic.superposition_total_ratio(ic.GRAV_B_EFF[0]) > ic.SUPERPOSITION_GUARD
    assert ic.superposition_total_ratio(ic.GRAV_B_EFF[1]) <= ic.SUPERPOSITION_GUARD
    # baseline self-weight ratio for B0 matches the committed sag (delta/ell ~ 0.0505 at ell=0.24)
    assert ic.sw_deflection_ratio(ic.GRAV_B_EFF[0]) == pytest.approx(0.0505, abs=0.001)


def test_baseline_subtraction_cubic_recovery():
    """Superposition: delta_total = delta_sw(quartic) + delta_point(cubic); subtracting the
    baseline and fitting the residual to the cubic law recovers B exactly."""
    B_true, F, m0 = 0.038259, 0.0597, ic.BASELINE_ARM_MASS
    ell = np.array(ic.CALIB_LENGTHS)
    w0 = m0 * ic.G / ic.REFERENCE_INTERVAL
    d_sw = w0 * ell ** 4 / (8.0 * B_true)              # quartic self-weight baseline
    d_point = F * ell ** 3 / (3.0 * B_true)            # cubic tip-load response
    d_total = d_sw + d_point
    k = ic.through_origin_cubic(ell, d_total - d_sw)   # subtract, fit cubic
    assert abs(ic.b_eff_force_from_fit(F, k) - B_true) / B_true < 1e-9


def test_no_refit_perfect_map_meets_frozen_bounds():
    """A perfect force==gravity B_eff map must satisfy every frozen no-refit bound; this
    validates the scorer arithmetic + the frozen tolerances against the real manifests."""
    bmap = dict(zip(ic.RAW_E_GRID, ic.GRAV_B_EFF))
    bmap.update(dict(zip(ic.RATIO_RAW_E, ic.RATIO_GRAV_B_EFF)))
    a = ic_norefit.score_sag_retrospective(bmap)
    b = ic_norefit.score_prefactor(bmap)
    c = ic_norefit.score_distal(bmap)
    assert a["passed"], (a["max_rel_err"], a["tol"])
    assert b["passed"], (b["mean_K"], b["predicted"], b["tol"])
    assert c["passed"], (c["worst_offset_cells"], [r for r in c["per_cell"] if abs(r["offset_cells"]) > 1])
    # family-(a) primary excludes out-of-regime, the B0 material, AND the baseline mass m0
    assert all(r["Pi_g"] <= ic.PI_G_MAX for r in a["primary"])
    assert all(abs(r["mass"] - ic.BASELINE_ARM_MASS) > 1e-12 for r in a["primary"])
    assert all(r["raw_E"] != ic.RAW_E_GRID[0] for r in a["primary"])       # B0 never primary
    assert any(r.get("reason", "").startswith("baseline-mass") for r in a["descriptive"])
    assert b["mean_K"] == pytest.approx(ic.OBS_PREFACTOR_MEAN, abs=1e-6)


# --------------------------------------------------------------------- integration (Genesis)
@pytest.mark.integration
def test_baseline_mass_tensor_readback():
    """The baseline-subtraction tensors read back exactly: arm = m0 everywhere, tip = m0 + m_load."""
    from sim.scene import build_scene, add_straight_rod
    from sim.calibrate_beff_force import build_baseline_tensors
    import torch, genesis as gs
    nv = 12
    m_loads = np.array([1.07e-3, 3.43e-2])
    baseline, loaded = build_baseline_tensors(nv, m_loads, m0=ic.BASELINE_ARM_MASS)
    scene = build_scene(dt=ic.DT, substeps=ic.SUBSTEPS, damping=ic.DAMPING, angular_damping=ic.ANGULAR_DAMPING)
    rod = add_straight_rod(scene, nv, interval=0.01, E=1e7, segment_mass=ic.BASELINE_ARM_MASS,
                           segment_radius=ic.SEGMENT_RADIUS, G=ic.SHEAR_G)
    scene.build(n_envs=2)
    rod.set_fixed_states(fixed_ids=[0, 1])
    rod.set_segment_mass(torch.tensor(loaded, dtype=gs.tc_float, device="cuda"))
    got = rod.get_all_segment_mass_tc().detach().cpu().numpy()
    assert np.allclose(got, loaded, rtol=1e-5, atol=1e-12)
    assert got[0, 2] == pytest.approx(ic.BASELINE_ARM_MASS)                      # arm = m0 (stable)
    assert got[1, nv - 1] == pytest.approx(ic.BASELINE_ARM_MASS + 3.43e-2, rel=1e-5)  # tip = m0 + m_load
    assert np.allclose(baseline, ic.BASELINE_ARM_MASS)


@pytest.mark.integration
def test_finding5_broadcast_premise():
    """apply_properties: scalar mass -> full (E.size, n_vertices); ndim==1 -> repeat to n_vertices
    (full segment_mass on every free vertex, right-endpoint lumping)."""
    from sim.scene import build_scene, add_straight_rod
    from sim.material import apply_properties
    nv = 10
    scene = build_scene(dt=ic.DT, substeps=ic.SUBSTEPS, damping=ic.DAMPING, angular_damping=ic.ANGULAR_DAMPING)
    rod = add_straight_rod(scene, nv, interval=0.01, E=1e7, segment_mass=1e-3,
                           segment_radius=ic.SEGMENT_RADIUS, G=ic.SHEAR_G)
    scene.build(n_envs=2)
    rod.set_fixed_states(fixed_ids=[0, 1])
    apply_properties(rod, [1e6, 2e6], 0.002)        # scalar mass -> broadcast to every vertex
    got = rod.get_all_segment_mass_tc().detach().cpu().numpy()
    assert np.allclose(got, 0.002)
    apply_properties(rod, [1e6, 2e6], [0.001, 0.003])  # ndim==1 -> per-env, repeated across vertices
    got = rod.get_all_segment_mass_tc().detach().cpu().numpy()
    assert np.allclose(got[0], 0.001) and np.allclose(got[1], 0.003)
