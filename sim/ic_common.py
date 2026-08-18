"""Pure numerical core for the independent mechanics closure (adjudication §C-1).

NO Genesis import — every function here is deterministic and unit-testable on CPU.
Frozen constants are pinned to PRIOR committed artifacts (calibration.json,
hard_gate_verdict.json, distal_manifest.json / distal_sweep_landscape.json). No
threshold is derived from data this phase generates. See the approved plan
(stage-03-revision.md) §2-§7.

Physics summary
---------------
Force mode (independent of self-weight): a concentrated TIP point-load on a
horizontally clamped rod. Free-arm vertices carry mass epsilon; the single tip
vertex carries m_tip; gravity is ON so F = m_tip * g is a concentrated END load.
Euler-Bernoulli:  delta_tip = F * ell**3 / (3 * B)  (CUBIC), structurally
different from distributed self-weight sag  delta = w * ell**4 / (8 * B) (QUARTIC).
Independence is scoped to LAW / LOAD-DISTRIBUTION / no-refit within ONE solver
(a common multiplicative g/mass error cancels — see the plan §1).
"""
from __future__ import annotations

import numpy as np

# --- frozen study-wide integrator / discretization (calibration.json) ---
DT = 2.5e-4
SUBSTEPS = 20
DAMPING = 40.0
ANGULAR_DAMPING = 20.0
G = 9.81                       # calibration.json:g
SEGMENT_RADIUS = 0.01          # calibration.json:materials[*].segment_radius
SHEAR_G = 1e4                  # sim/scene.FIXED_G

REFERENCE_INTERVAL = 0.01      # calibration.json:interval
MESH_INTERVALS = (0.02, 0.01, 0.005)

# --- committed gravity calibration grid (calibration.json) ---
RAW_E_GRID = (1_000_000.0, 2_370_946.5892385854, 5_621_387.729022082,
              13_328_010.062912514, 31_600_000.0)
GRAV_B_EFF = (0.006713764347921196, 0.01610520612280113, 0.03825920568613013,
              0.09110646283229176, 0.21527426878237318)
# ratio-cell FORCE points (distal_manifest.json:ratio_pairs): raw_E to SET; gravity B_eff sizes m_tip
RATIO_IDS = ("R0", "R1", "R2")
RATIO_RAW_E = (4_741_893.178477171, 6_664_005.031456257, 8_432_081.593533123)
RATIO_GRAV_B_EFF = (0.032264843367045035, 0.045408831332695465, 0.05753320861264561)
RATIO_W = (0.8454001723685116, 0.45533986457741354, 0.6340501292763837)

CALIB_LENGTHS = (0.18, 0.20, 0.22, 0.24)
CALIB_MASSES = (0.0002, 0.0004)

# --- force-mode small-deflection guard (hard_gate_verdict.json:Pi_g_max=0.5) ---
PI_G_MAX = 0.5
GUARD_DEFLECTION_RATIO = PI_G_MAX / 8.0     # 0.0625 : matches the gravity delta/ell regime
FORCE_TARGET_RATIO = 0.03                   # operating point (half the 0.0625 cap)
UPPER_BRACKET_ELL = 0.24                    # finding-3 conservative upper-bracket endpoint
F2_RATIO = 0.5                              # second force level for finding-7 load-invariance

# --- BASELINE-SUBTRACTION realization (owner ruling; supersedes the epsilon-arm option iv) ---
# The frozen <1% contamination bound is NOT relaxed; it is SUPERSEDED by this subtraction
# design (the self-weight is MEASURED and subtracted, so there is no epsilon-arm distributed-
# mass contamination to bound). Authorization: escalation commits a6cfb21 + bf688ec + the
# owner ruling. See sim/calibrate_beff_force.force_calibrate_baseline.
BASELINE_ARM_MASS = 0.0002                  # m0: normal stable uniform arm mass (== committed gravity mass)
SUPERPOSITION_GUARD = GUARD_DEFLECTION_RATIO  # 0.0625: TOTAL delta/ell cap for linear superposition

# --- calibration acceptance (analog of sim/calibrate_beff.py:main accept code) ---
ACCEPT_CV = 0.05
ACCEPT_RESIDUAL = 0.05
ACCEPT_EXPONENT_LO = 2.6       # cubic analog of the quartic band [3.6, 4.4] (theoretical exponent 3, +/-0.4)
ACCEPT_EXPONENT_HI = 3.4
ACCEPT_MULTIMASS = 8.0         # outer code bound
FINDING7_ACCEPT = 4.5          # calibration.json worst multimass_invariance_pct = 4.5039

# --- mesh observables (§4) ---
OBS_A_SUCCESSIVE_TOL = 0.02    # calibration.json worst B_eff_CV 1.84% -> 2%
OBS_A_ABSOLUTE_CEILING = 0.05  # sim/calibrate_beff.py:main max_residual <= 0.05
OBS_B_TOL = 0.026326769214488028   # hard_gate_verdict.json:grid_resolution_bound_pm
OBS_A_FRACS = (0.2, 0.4, 0.6, 0.8, 1.0)
GRAV_SWEEP_ELL = tuple(round(0.12 + 0.03 * i, 2)
                       for i in range(int(round((0.60 - 0.12) / 0.03)) + 1))  # [0.12..0.60] step 0.03
DROOP_CLEAR_H = 0.009          # (8h)^(1/4) = predicted_prefactor 0.5180 -> 8h = 0.072 -> h = 0.009
OBS_B_REF_MASS = 0.0002
OBS_B_REF_W = OBS_B_REF_MASS * G / REFERENCE_INTERVAL   # 0.1962

# --- no-refit targets (hard_gate_verdict.json + distal) ---
PREDICTED_PREFACTOR = 0.5180040128222703       # (8h)^(1/4)
OBS_PREFACTOR_MEAN = 0.5188501907210823
DISTAL_DELTA = 0.012           # |z*| (distal_manifest.json:objective.Delta)
DISTAL_D = 0.24
DISTAL_ELL_STEP = 0.02
DISTAL_ELL_LO = 0.12
DISTAL_ELL_HI = 0.60
NOREFIT_SAG_TOL = 0.05         # sim/calibrate_beff.py:main max_residual <= 0.05
NOREFIT_PREFACTOR_TOL = OBS_B_TOL
NOREFIT_DISTAL_CELL_TOL = 1    # distal_manifest.json:gate_truth_table.ratio_tolerance_cells
PROSPECTIVE_MASS = 0.0003
PROSPECTIVE_LENGTHS = (0.19, 0.21, 0.23)


# ---------------------------------------------------------------------------
# discretization / indexing
# ---------------------------------------------------------------------------
def n_vertices_for(ell, interval):
    """Total vertices for a free arm of length ``ell`` at ``interval``.

    Verts 0,1 are clamped; the free arm ell = (n_vertices - 2) * interval, so
    n_vertices = round(ell / interval) + 2. Matches sim/calibrate_beff._nv.
    """
    return int(round(ell / interval)) + 2


def segment_count(ell, interval):
    """N = free-arm SEGMENT count = round(ell / interval) = n_vertices - 2."""
    return int(round(ell / interval))


def free_vertex_indices(nv):
    """Non-clamped vertex indices 2..nv-1 (finding-5 corrected convention 2..N+1,
    N = nv-2): interior-free 2..nv-2 plus the tip nv-1."""
    return list(range(2, nv))


def tip_index(nv):
    return nv - 1


# ---------------------------------------------------------------------------
# force-mode calibration (cubic)
# ---------------------------------------------------------------------------
def m_tip_for(b_eff, target_ratio=FORCE_TARGET_RATIO, ell=UPPER_BRACKET_ELL, g=G):
    """Tip mass sizing F at the upper-bracket ell to delta/ell = target_ratio.

    delta/ell = F*ell**2/(3B) = target_ratio  =>  F = 3*B*target_ratio/ell**2 ; m_tip = F/g.
    At the primary target_ratio=0.03, F = 1.5625*B (ell=0.24).
    """
    F = 3.0 * b_eff * target_ratio / (ell ** 2)
    return F / g


def through_origin_cubic(ell, delta):
    """Through-origin fit delta = k * ell**3; returns slope k."""
    ell = np.asarray(ell, dtype=float)
    delta = np.asarray(delta, dtype=float)
    x = ell ** 3
    return float(np.sum(x * delta) / np.sum(x * x))


def b_eff_force_from_fit(force, k):
    """delta = F*ell**3/(3B) and delta = k*ell**3  =>  B_eff_force = F/(3k)."""
    return force / (3.0 * k)


def cubic_fit_stats(ell, delta, force):
    """Per-material force-calibration stats: B_eff_force, k, CV, max residual,
    log-log exponent (theoretical 3), and worst deflection ratio at the upper bracket."""
    ell = np.asarray(ell, dtype=float)
    delta = np.asarray(delta, dtype=float)
    k = through_origin_cubic(ell, delta)
    b_eff = b_eff_force_from_fit(force, k)
    pred = k * ell ** 3
    residual = float(np.max(np.abs(delta - pred) / np.maximum(np.abs(delta), 1e-12)))
    per_len_b = force * ell ** 3 / (3.0 * delta)
    cv = float(np.std(per_len_b) / np.mean(per_len_b))
    exponent = float(np.polyfit(np.log(ell), np.log(delta), 1)[0])
    worst_ratio = float(np.max(delta / ell))
    return dict(B_eff_force=b_eff, k=k, CV=cv, max_residual=residual, exponent=exponent,
                worst_deflection_ratio=worst_ratio, per_length_B_eff=per_len_b.tolist())


def deflection_ratio(delta_tip, ell):
    """delta_tip/ell : the force-mode guard quantity (analog of Pi_g/8)."""
    return float(delta_tip / ell)


def guard_ok(delta_tip, ell, cap=GUARD_DEFLECTION_RATIO):
    """Observed small-deflection guard: delta_tip/ell <= 0.0625."""
    return deflection_ratio(delta_tip, ell) <= cap


def sw_deflection_ratio(b_eff, m0=BASELINE_ARM_MASS, interval=REFERENCE_INTERVAL, ell=UPPER_BRACKET_ELL, g=G):
    """Self-weight baseline deflection ratio delta_sw/ell = Pi_g(m0)/8 = (m0*g/interval)*ell**3/(8B),
    computed from a PRIOR committed B_eff (no new data). Upper-bracket ell by finding-3 convention."""
    w0 = m0 * g / interval
    return float(w0 * ell ** 3 / (8.0 * b_eff))


def superposition_total_ratio(b_eff, m0=BASELINE_ARM_MASS, target_ratio=FORCE_TARGET_RATIO,
                              interval=REFERENCE_INTERVAL, ell=UPPER_BRACKET_ELL):
    """TOTAL tip-deflection ratio delta_total/ell = delta_sw/ell + delta_point/ell (design target).
    Pre-computable from committed numbers; the linear-superposition inclusion guard."""
    return sw_deflection_ratio(b_eff, m0, interval, ell) + target_ratio


def superposition_included(b_eff, m0=BASELINE_ARM_MASS, target_ratio=FORCE_TARGET_RATIO,
                           interval=REFERENCE_INTERVAL, ell=UPPER_BRACKET_ELL, cap=SUPERPOSITION_GUARD):
    """Linear-superposition INCLUSION guard: total delta/ell <= 0.0625 (Pi_g_max/8). A material is
    in-regime iff its self-weight baseline + tip operating point stays in the committed small-
    deflection regime. Excludes the softest B0 as regime-of-validity (like hardening-A)."""
    return superposition_total_ratio(b_eff, m0, target_ratio, interval, ell) <= cap


def included_cohort(m0=BASELINE_ARM_MASS):
    """The raw_E in the superposition cohort: 5 grid (B0..B4) + 3 ratio, keeping only in-regime.
    Returns list of (raw_E, gravity_B_eff, label). B0 (softest) falls out by the guard."""
    out = []
    for raw_e, gb, lab in (list(zip(RAW_E_GRID, GRAV_B_EFF, [f"B{i}" for i in range(5)]))
                           + list(zip(RATIO_RAW_E, RATIO_GRAV_B_EFF, RATIO_IDS))):
        if superposition_included(gb, m0):
            out.append((raw_e, gb, lab))
    return out


# --- SUPERSEDED epsilon-arm contamination machinery (retained to document the superseded
#     option-(iv) design and for the C0 tests; NOT used by the baseline-subtraction method) ---
def contamination_ratio(nv, interval, f_eps):
    """SUPERSEDED (epsilon-arm option iv). EXACT discrete self-weight contamination of the tip-load response.

    Free interior verts 2..nv-2 each carry eps = f_eps*m_tip at distance
    x_i = (i-1)*interval from the clamped root (vertex 1). Cantilever point-load
    tip influence delta(P@x) = P*x**2*(3L-x)/(6B); the tip mass m_tip at x=L gives
    m_tip*L**3/(3B). g, B and m_tip all cancel, leaving

        R = [ sum_i f_eps * x_i**2 * (3L - x_i) / 6 ] / ( L**3 / 3 ).
    """
    L = (nv - 2) * interval
    interior = np.arange(2, nv - 1)               # 2 .. nv-2 (exclude clamped 0,1 and tip nv-1)
    if interior.size == 0:
        return 0.0
    x = (interior - 1) * interval                 # distance from clamped root vertex 1
    num = np.sum(f_eps * x ** 2 * (3.0 * L - x) / 6.0)
    den = L ** 3 / 3.0
    return float(num / den)


def max_f_eps_for_contamination(nv, interval, cap=0.01):
    """Largest per-vertex f_eps keeping the EXACT discrete contamination <= cap."""
    unit = contamination_ratio(nv, interval, f_eps=1.0)   # linear in f_eps
    return cap / unit if unit > 0 else float("inf")


# ---------------------------------------------------------------------------
# finding-5 discretization-bias reference (DESCRIPTIVE, no gate)
# ---------------------------------------------------------------------------
def r_N(N):
    """Lumped-mass discretization reference r_N = (N+1)(3N+1)/(3N**2),
    N = free-arm segment count = round(ell/interval). Sag ratio
    delta_grav/delta_continuum ~ r_N (stiffness ratio ~ 1/r_N; reciprocal, not equal)."""
    N = float(N)
    return (N + 1.0) * (3.0 * N + 1.0) / (3.0 * N ** 2)


# ---------------------------------------------------------------------------
# mesh observables (§4)
# ---------------------------------------------------------------------------
def endload_shape(frac):
    """Analytical cantilever end-load shape y(x)/delta_tip = (3*frac**2 - frac**3)/2, frac = x/L."""
    frac = np.asarray(frac, dtype=float)
    return (3.0 * frac ** 2 - frac ** 3) / 2.0


def observable_A_error(u_over_utip, fracs=OBS_A_FRACS):
    """Observable A: max absolute deviation of the measured normalized force shape
    from the analytical end-load shape (NOT self-normalized by a fitted B)."""
    u = np.asarray(u_over_utip, dtype=float)
    return float(np.max(np.abs(u - endload_shape(fracs))))


def observable_B_boundary(ells, droops, h=DROOP_CLEAR_H):
    """Observable B: largest free length whose settled tip droop clears h, via linear
    interpolation of the droop = h crossing on the frozen sweep grid. Droop increases
    with ell; returns None (INCONCLUSIVE) if no interior crossing exists."""
    ells = np.asarray(ells, dtype=float)
    droops = np.asarray(droops, dtype=float)
    order = np.argsort(ells)
    ells, droops = ells[order], droops[order]
    for k in range(len(ells) - 1):
        if droops[k] <= h < droops[k + 1]:
            t = (h - droops[k]) / (droops[k + 1] - droops[k])
            return float(ells[k] + t * (ells[k + 1] - ells[k]))
    return None


def prefactor_K(ell_boundary, b_eff_force, w=OBS_B_REF_W):
    """Measured mesh prefactor K_N = ell_boundary / (B_eff_force / w)**0.25."""
    return float(ell_boundary / (b_eff_force / w) ** 0.25)


def convergence_trend_ok(successive_diffs, tol):
    """Successive-difference convergence: last diff < first diff (decreasing trend)
    AND every diff <= tol. `successive_diffs` ordered coarse->fine."""
    d = [abs(float(x)) for x in successive_diffs]
    if len(d) < 2:
        return False
    return d[-1] < d[0] and all(x <= tol for x in d)


# ---------------------------------------------------------------------------
# no-refit predictors (§5) — B_eff_force is the ONLY calibration input
# ---------------------------------------------------------------------------
def pi_g(w, ell, b_eff):
    """Gravity small-deflection number Pi_g = w*ell**3/B."""
    return float(w * ell ** 3 / b_eff)


def w_of_mass(mass, interval=REFERENCE_INTERVAL, g=G):
    return mass * g / interval


def predict_sag(ell, mass, b_eff_force, interval=REFERENCE_INTERVAL, g=G):
    """(a) held-out self-weight tip sag from B_eff_force ONLY: delta = w*ell**4/(8*B)."""
    w = w_of_mass(mass, interval, g)
    return float(w * ell ** 4 / (8.0 * b_eff_force))


def predict_prefactor_K(ell_max, b_eff_force, w):
    """(b) hardening-A boundary prefactor K_force = ell_max_measured / (B_eff_force/w)**0.25."""
    return float(ell_max / (b_eff_force / w) ** 0.25)


def predict_distal_sstar(b_eff_force, w, delta=DISTAL_DELTA):
    """(c) distal optimum s* = (8*B_eff_force*|z*|/w)**0.25."""
    return float((8.0 * b_eff_force * delta / w) ** 0.25)


def snap_to_grid(value, step=DISTAL_ELL_STEP, lo=DISTAL_ELL_LO, hi=DISTAL_ELL_HI):
    """Snap a continuous ell to the nearest distal grid cell."""
    n = int(round((hi - lo) / step))
    idx = int(round((value - lo) / step))
    idx = max(0, min(idx, n))
    return round(lo + idx * step, 2)


def grid_cell_offset(pred_ell, target_ell, step=DISTAL_ELL_STEP):
    """Integer grid-cell offset between a snapped prediction and a target ell."""
    return int(round((snap_to_grid(pred_ell) - target_ell) / step))
