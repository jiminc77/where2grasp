"""Pure analytic small-deflection tip model for the distal-tip-placement task.

Single source of truth for every PRE-REGISTERED prediction (no Genesis import):
- vertical tip droop      delta_tip(l) = w*l^4/(8*B)                         [calibrate_beff law]
- horizontal tip reach    x(l) = l - w^2*l^7/(112*B^2)                       [leading foreshortening]
- small-deflection group  Pi_g(l) = w*l^3/B  (validity guard <= 0.5)
- continuous objective    J(l) = rho - |delta_tip(l) - Delta|,  Delta = |z*|
- UNIQUE interior optimum s* = l_delta = (8*B*|z*|/w)^(1/4)                  [root of delta=Delta;
  unique because delta_tip(l) is strictly increasing in l => |delta-Delta| is strictly
  decreasing then increasing => a single interior maximiser of J (no endpoint/plateau optimum)]

Feasibility (IN-REGIME by construction; finding-1/2/3 corrected framework):
  l_regime = (0.5*B/w)^(1/3)              (Pi_g = 0.5 exactly: the in-regime arm ceiling)
  l_sag    = (8*B*(Delta+rho)/w)^(1/4)    (deepest acceptable droop = sag/"droops-past" bound)
  l_dlo    = (8*B*(Delta-rho)/w)^(1/4)    (shallowest acceptable droop)
  l_top    = min(l_sag, l_regime)         (reach-maximising in-regime depth-acceptable grasp)
  l_reach  = smallest exact root of x(l) = d - rho_x   (reach lower bound; foreshortening-aware)
  FEASIBLE iff  x(l_top) >= d - rho_x  AND  l_top >= l_dlo.
  Feasible band [l_L, l_U] = [max(l_reach, l_dlo), l_top].
  reach shapes the feasible SET (lower bound l_L); sag/regime the upper bound l_U + the optimum s*.

Frozen decision rules used identically by tip_model (predicted) and distal_gate (measured):
  TIE      : argmax over in-regime reach-feasible grasps of the score; ties (|dJ|<=TIE_EPS) -> lowest index.
  CLIPPED  : an argmax at a grid endpoint (0 or N-1) is flagged; excluded from off-grid-direction adjacency.
  CENSORED : no feasible in-regime grasp -> no argmax (excluded from adjacency, feeds feasibility only);
             a flat feasible-region score curve (max-min <= TIE_EPS) -> censored.
  CONSERVATIVE REGIME GUARD (finding 3): an estimand value v is in-regime only if Pi_g(upper bracket
             endpoint of v's grid bracket) <= 0.5, else INCONCLUSIVE-regime.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

# ---- frozen design constants (distal_grid_design re-derives + asserts these before manifest freeze) ----
DELTA = 0.012          # target depth |z*| below clamp height (m)
RHO = 0.006            # depth (sag) tolerance (m)
D = 0.24               # target horizontal distance beyond the edge (m)
RHO_X = 0.015          # horizontal reach tolerance (m); reach floor R = D - RHO_X
J_INF = -1.0           # finite pinned infeasible score (J units); < any in-regime reach-satisfying J
TAU = 0.5              # success-rate crossing threshold
PI_G_MAX = 0.5         # small-deflection validity guard
G_GRAV = 9.81
ELL_MIN, ELL_MAX, ELL_STEP = 0.12, 0.60, 0.02   # finder grid so every eligible contrast shifts >= 2 cells
MIN_SHIFT_CELLS = 2    # finding-1: predicted optimum shift >= 2 grid cells per eligible contrast
TIE_EPS = 1e-9


def ell_grid(lo=ELL_MIN, hi=ELL_MAX, step=ELL_STEP):
    """Uniform free-arm grid (inclusive)."""
    n = int(round((hi - lo) / step)) + 1
    return np.array([round(lo + step * k, 10) for k in range(n)])


def delta_tip(ell, B, w):
    ell = np.asarray(ell, float)
    return w * ell ** 4 / (8.0 * B)


def x_reach(ell, B, w):
    ell = np.asarray(ell, float)
    return ell - w ** 2 * ell ** 7 / (112.0 * B ** 2)


def pi_g(ell, B, w):
    ell = np.asarray(ell, float)
    return w * ell ** 3 / B


def objective_J(ell, B, w, Delta=DELTA, rho=RHO, d=D, rho_x=RHO_X, j_inf=J_INF):
    """J(l) = rho - |delta - Delta| among IN-REGIME reach-satisfying grasps, else j_inf (scalar or array)."""
    ell = np.asarray(ell, float)
    dt = delta_tip(ell, B, w)
    reach_ok = x_reach(ell, B, w) >= (d - rho_x)
    regime_ok = pi_g(ell, B, w) <= PI_G_MAX
    depth_ok = np.abs(dt - Delta) <= rho
    feasible = reach_ok & regime_ok & depth_ok
    j = rho - np.abs(dt - Delta)
    return np.where(feasible, j, j_inf)


def success(ell, B, w, Delta=DELTA, rho=RHO, d=D, rho_x=RHO_X):
    """Binary success = reach AND depth AND in-regime (the 0->1->0 landscape)."""
    ell = np.asarray(ell, float)
    return ((x_reach(ell, B, w) >= (d - rho_x)) &
            (np.abs(delta_tip(ell, B, w) - Delta) <= rho) &
            (pi_g(ell, B, w) <= PI_G_MAX))


def ell_optimum(B, w, Delta=DELTA):
    """UNIQUE continuous optimum s* = root of delta_tip(l)=Delta = (8*B*Delta/w)^(1/4)."""
    return (8.0 * B * Delta / w) ** 0.25


def ell_sag(B, w, Delta=DELTA, rho=RHO):
    return (8.0 * B * (Delta + rho) / w) ** 0.25


def ell_dlo(B, w, Delta=DELTA, rho=RHO):
    if Delta - rho <= 0:
        return 0.0
    return (8.0 * B * (Delta - rho) / w) ** 0.25


def ell_regime(B, w, pi_max=PI_G_MAX):
    return (pi_max * B / w) ** (1.0 / 3.0)


def ell_top(B, w, Delta=DELTA, rho=RHO):
    return min(ell_sag(B, w, Delta, rho), ell_regime(B, w))


def ell_reach(B, w, d=D, rho_x=RHO_X):
    """Smallest exact root of x(l) = d - rho_x by bisection on the increasing branch of x(l).

    x(l)=l - w^2 l^7/(112 B^2) increases from 0, peaks at l_peak=(16 B^2/w^2)^(1/6), then decreases.
    The reach lower bound is the FIRST (smallest) crossing, which lies on the increasing branch."""
    R = d - rho_x
    l_peak = (16.0 * B ** 2 / w ** 2) ** (1.0 / 6.0)
    x_peak = float(x_reach(l_peak, B, w))
    if x_peak < R:                      # cannot reach R at all (even at maximum reach)
        return None
    lo, hi = 0.0, l_peak
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if float(x_reach(mid, B, w)) < R:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def nearest_grid_index(value, grid):
    grid = np.asarray(grid, float)
    return int(np.argmin(np.abs(grid - value)))


def bracket_upper(value, grid):
    """Upper endpoint of the grid bracket containing `value` (for the conservative regime guard).
    value below grid[0] -> grid[1]; above grid[-1] -> grid[-1]; else grid[j+1] with grid[j]<=value<grid[j+1]."""
    grid = np.asarray(grid, float)
    if value <= grid[0]:
        return float(grid[min(1, len(grid) - 1)])
    if value >= grid[-1]:
        return float(grid[-1])
    j = int(np.searchsorted(grid, value, side='right'))   # grid[j-1] <= value < grid[j]
    return float(grid[min(j, len(grid) - 1)])


def regime_in(value, B, w, grid, pi_max=PI_G_MAX):
    """Conservative guard: in-regime iff Pi_g(upper bracket endpoint of value's bracket) <= pi_max."""
    if value is None:
        return False
    return float(pi_g(bracket_upper(value, grid), B, w)) <= pi_max


def cell_analysis(B, w, Delta=DELTA, rho=RHO, d=D, rho_x=RHO_X, grid=None):
    """Full per-cell analytic record used by distal_grid_design asserts + distal_gate predictions."""
    if grid is None:
        grid = ell_grid()
    grid = np.asarray(grid, float)
    R = d - rho_x
    lstar = ell_optimum(B, w, Delta)
    ls = ell_sag(B, w, Delta, rho)
    ldlo = ell_dlo(B, w, Delta, rho)
    lreg = ell_regime(B, w)
    ltop = min(ls, lreg)
    lreach = ell_reach(B, w, d, rho_x)
    x_top = float(x_reach(ltop, B, w))
    feasible = bool((x_top >= R) and (ltop >= ldlo))
    lL = max(lreach if lreach is not None else 0.0, ldlo)
    lU = ltop
    # constrained predicted argmax on the grid (in-regime reach-feasible grasp maximising J)
    Jg = objective_J(grid, B, w, Delta, rho, d, rho_x)
    feas_mask = Jg > J_INF
    if feas_mask.any():
        best = float(np.max(Jg[feas_mask]))
        cand = np.where(feas_mask & (Jg >= best - TIE_EPS))[0]
        argmax_idx = int(cand.min())            # tie -> lowest index
        clipped = bool(argmax_idx == 0 or argmax_idx == len(grid) - 1)
        flat = bool(float(Jg[feas_mask].max() - Jg[feas_mask].min()) <= TIE_EPS)
    else:
        argmax_idx = None
        clipped = False
        flat = False
    censored = bool(argmax_idx is None or flat)
    return {
        'B_eff': float(B), 'w': float(w), 'w_over_B': float(w / B),
        'ell_star': float(lstar), 'ell_sag': float(ls), 'ell_dlo': float(ldlo),
        'ell_regime': float(lreg), 'ell_top': float(ltop),
        'ell_reach': (float(lreach) if lreach is not None else None),
        'ell_L': float(lL), 'ell_U': float(lU),
        'x_at_top': x_top, 'reach_floor_R': float(R),
        'feasible': feasible,
        'argmax_idx': argmax_idx, 'argmax_ell': (float(grid[argmax_idx]) if argmax_idx is not None else None),
        'clipped': clipped, 'censored': censored,
        'pi_g_star': float(pi_g(lstar, B, w)),
        'regime_star': regime_in(lstar, B, w, grid),
        'regime_L': regime_in(lL if feasible else None, B, w, grid),
        'regime_U': regime_in(lU if feasible else None, B, w, grid),
    }


def predicted_shift_cells(cellsById, contrasts, step=ELL_STEP):
    """Finding-1(b): per eligible adjacent contrast, |l*_a - l*_b| in grid cells (from prior calibration).
    contrasts: list of (id_a, id_b). Returns list of dicts + min shift; asserts belong to distal_grid_design."""
    out = []
    for a, b in contrasts:
        ca, cb = cellsById[a], cellsById[b]
        shift_m = abs(ca['ell_star'] - cb['ell_star'])
        out.append({'a': a, 'b': b, 'shift_m': float(shift_m),
                    'shift_cells': float(shift_m / step),
                    'meets_min': bool(shift_m / step >= MIN_SHIFT_CELLS)})
    return out


def a1_a2_enumeration(B, w, Delta=DELTA, rho=RHO, d=D, rho_x=RHO_X, z_star=None, grid=None):
    """Finding-1c/HB-C5: enumerate A1 (argmax J) vs A2 (argmin Euclidean ||tip-P||) over the grid.
    Demonstrates A2 grid-resolution degeneracy (near-d cell) vs A1's material-dependent interior optimum."""
    if grid is None:
        grid = ell_grid()
    if z_star is None:
        z_star = -Delta
    grid = np.asarray(grid, float)
    Jg = objective_J(grid, B, w, Delta, rho, d, rho_x)
    feas = Jg > J_INF
    a1 = int(np.where(feas & (Jg >= float(np.max(Jg[feas])) - TIE_EPS))[0].min()) if feas.any() else None
    # A2: minimise Euclidean distance of settled tip (x, -delta) to target P=(d, z_star)
    xr = x_reach(grid, B, w)
    zt = -delta_tip(grid, B, w)
    e2 = (xr - d) ** 2 + (zt - z_star) ** 2
    a2 = int(np.argmin(e2))
    return {'a1_idx': a1, 'a1_ell': (float(grid[a1]) if a1 is not None else None),
            'a2_idx': int(a2), 'a2_ell': float(grid[a2])}


def measured_argmax(mean_J_by_grasp, feasible_mask, grid):
    """distal_gate measured argmax with the SAME tie/clipped/censored rules (measured mean J)."""
    mean_J = np.asarray(mean_J_by_grasp, float)
    fm = np.asarray(feasible_mask, bool)
    if not fm.any():
        return {'argmax_idx': None, 'clipped': False, 'censored': True, 'reason': 'no_feasible_grasp'}
    vals = mean_J[fm]
    if float(vals.max() - vals.min()) <= TIE_EPS:
        return {'argmax_idx': None, 'clipped': False, 'censored': True, 'reason': 'flat_curve'}
    best = float(vals.max())
    cand = np.where(fm & (mean_J >= best - TIE_EPS))[0]
    idx = int(cand.min())
    return {'argmax_idx': idx, 'clipped': bool(idx == 0 or idx == len(grid) - 1),
            'censored': False, 'reason': None}


def band_crossings(success_rate_by_grasp, grid, tau=TAU):
    """Finding-2: two-sided band edges. ell_L = FIRST upward tau crossing; ell_U = LAST downward.
    Returns (ell_L, ell_U, censor) with censor in {None,'upper','both'}; interpolated in ell units."""
    y = np.asarray(success_rate_by_grasp, float)
    g = np.asarray(grid, float)
    up = None
    for i in range(len(y) - 1):
        if y[i] < tau <= y[i + 1] and y[i + 1] != y[i]:
            up = float(g[i] + (tau - y[i]) / (y[i + 1] - y[i]) * (g[i + 1] - g[i]))
            break
    if up is None and y[0] >= tau:            # already above at the first grasp
        up = float(g[0])
    down = None
    for i in range(len(y) - 1):
        if y[i] >= tau > y[i + 1] and y[i + 1] != y[i]:
            down = float(g[i] + (tau - y[i]) / (y[i + 1] - y[i]) * (g[i + 1] - g[i]))
    if up is None:
        return None, None, 'both'
    if down is None:
        return up, None, 'upper'            # up but no down -> upper edge censored
    return up, down, None


def interval_iou(a_L, a_U, b_L, b_U):
    """IoU of predicted [a_L,a_U] and measured [b_L,b_U]; None if either interval undefined."""
    if a_L is None or a_U is None or b_L is None or b_U is None:
        return None
    lo_i, hi_i = max(a_L, b_L), min(a_U, b_U)
    inter = max(0.0, hi_i - lo_i)
    union = (a_U - a_L) + (b_U - b_L) - inter
    return float(inter / union) if union > 0 else 0.0


def hausdorff(a_L, a_U, b_L, b_U):
    if None in (a_L, a_U, b_L, b_U):
        return None
    return float(max(abs(a_L - b_L), abs(a_U - b_U)))
