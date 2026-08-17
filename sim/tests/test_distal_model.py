"""C0 unit tests for the pure analytic distal tip model (no Genesis).

Validates the frozen physics + every owner-mandated rule against the REAL committed
hardening-A in-regime grid (read from hard_sweep_manifest.json = prior committed data):
findings 1 (unique-root objective + >=2-cell shift + tie/clipped/censored), 2 (two-sided
band + IoU), 3 (conservative upper-bracket regime guard), and the model-valid B1_w2 fail.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import sim.tip_model as tm

ROOT = Path(__file__).resolve().parents[1]


def _grid_cells():
    m = json.loads((ROOT / 'manifests/hard_sweep_manifest.json').read_text())
    return {c['id']: (float(c['B_eff']), float(c['w'])) for c in m['grid']}


def test_closed_forms():
    B, w = 0.09110646283229176, 0.4227000861842558
    ell = 0.30
    assert abs(tm.delta_tip(ell, B, w) - w * ell ** 4 / (8 * B)) < 1e-15
    assert abs(tm.x_reach(ell, B, w) - (ell - w ** 2 * ell ** 7 / (112 * B ** 2))) < 1e-15
    assert abs(tm.pi_g(ell, B, w) - w * ell ** 3 / B) < 1e-15
    # optimum is the exact root of delta=Delta
    lstar = tm.ell_optimum(B, w, tm.DELTA)
    assert abs(tm.delta_tip(lstar, B, w) - tm.DELTA) < 1e-12


def test_unique_interior_optimum_monotone():
    # J = rho-|delta-Delta| has a single interior max at l_delta; delta strictly increasing => uniqueness
    B, w = 0.03825920568613013, 0.4227000861842558
    grid = np.linspace(0.05, 0.9, 4000)
    dt = tm.delta_tip(grid, B, w)
    assert np.all(np.diff(dt) > 0)                      # strict monotonicity of droop
    J = tm.RHO - np.abs(dt - tm.DELTA)
    k = int(np.argmax(J))
    assert 0 < k < len(grid) - 1                        # interior
    lstar = tm.ell_optimum(B, w, tm.DELTA)
    assert abs(grid[k] - lstar) < (grid[1] - grid[0]) * 2


def test_constrained_optimum_table():
    cells = _grid_cells()
    grid = tm.ell_grid()
    expect = {  # constrained ell_star (m), 3dp, from (8*B*0.012/w)^(1/4)
        'B1_w0': 0.298, 'B1_w1': 0.246, 'B2_w0': 0.370, 'B2_w1': 0.305, 'B2_w2': 0.252,
        'B3_w0': 0.460, 'B3_w1': 0.379, 'B3_w2': 0.313, 'B4_w0': 0.570, 'B4_w1': 0.470, 'B4_w2': 0.388}
    for cid, ls in expect.items():
        B, w = cells[cid]
        assert abs(tm.ell_optimum(B, w, tm.DELTA) - ls) < 1.5e-3, (cid, tm.ell_optimum(B, w, tm.DELTA))
        a = tm.cell_analysis(B, w, grid=grid)
        assert a['feasible'], cid
        assert 0 < a['argmax_idx'] < len(grid) - 1      # interior, not clipped
        assert not a['clipped'] and not a['censored']


def test_B1_w2_is_model_valid_fail():
    cells = _grid_cells()
    grid = tm.ell_grid()
    B, w = cells['B1_w2']
    a = tm.cell_analysis(B, w, grid=grid)
    assert not a['feasible']                            # infeasible
    # fail is certified at the in-regime ceiling: l_top == l_regime (Pi_g=0.5), x(l_top) < R with margin
    assert abs(a['ell_top'] - a['ell_regime']) < 1e-9
    assert abs(tm.pi_g(a['ell_top'], B, w) - 0.5) < 1e-6
    assert a['x_at_top'] < a['reach_floor_R']
    assert a['reach_floor_R'] - a['x_at_top'] > tm.ELL_STEP / 2   # margin > half grid step
    assert a['argmax_idx'] is None and a['censored']


def test_feasibility_partition():
    cells = _grid_cells()
    grid = tm.ell_grid()
    feasible = [cid for cid, (B, w) in cells.items() if cid.startswith(('B1', 'B2', 'B3', 'B4'))
                and tm.cell_analysis(B, w, grid=grid)['feasible']]
    infeasible = [cid for cid, (B, w) in cells.items() if cid.startswith(('B1', 'B2', 'B3', 'B4'))
                  and not tm.cell_analysis(B, w, grid=grid)['feasible']]
    assert set(infeasible) == {'B1_w2'}
    assert len(feasible) == 11


def test_worst_pi_g_star_in_regime():
    cells = _grid_cells()
    grid = tm.ell_grid()
    worst = max(tm.cell_analysis(B, w, grid=grid)['pi_g_star']
                for cid, (B, w) in cells.items() if cid != 'B1_w2' and cid.startswith('B'))
    assert worst <= 0.5, worst


def test_predicted_shift_at_least_two_cells():
    cells = _grid_cells()
    grid = tm.ell_grid()
    ana = {cid: tm.cell_analysis(B, w, grid=grid) for cid, (B, w) in cells.items() if cid.startswith('B')}
    feasible = {cid for cid, a in ana.items() if a['feasible']}
    bis = ['B1', 'B2', 'B3', 'B4']; wis = ['w0', 'w1', 'w2']
    contrasts = []
    for wj in wis:                                      # fixed w, adjacent B
        col = [f'{b}_{wj}' for b in bis if f'{b}_{wj}' in feasible]
        contrasts += [(col[k], col[k + 1]) for k in range(len(col) - 1)]
    for bi in bis:                                      # fixed B, adjacent w
        row = [f'{bi}_{wj}' for wj in wis if f'{bi}_{wj}' in feasible]
        contrasts += [(row[k], row[k + 1]) for k in range(len(row) - 1)]
    shifts = tm.predicted_shift_cells(ana, contrasts)
    assert all(s['meets_min'] for s in shifts), [s for s in shifts if not s['meets_min']]
    assert min(s['shift_cells'] for s in shifts) >= tm.MIN_SHIFT_CELLS


def test_a2_grid_degeneracy_vs_a1():
    cells = _grid_cells()
    grid = tm.ell_grid()
    bis = ['B1', 'B2', 'B3', 'B4']
    # A2 argmin selection barely moves across adjacent B contrasts (< 1 cell) => degenerate null;
    # A1 argmax moves >= 2 cells. Use a fixed-w column with all feasible cells.
    a1 = {b: tm.a1_a2_enumeration(*cells[f'{b}_w0'], grid=grid) for b in bis}
    for k in range(len(bis) - 1):
        a, b = bis[k], bis[k + 1]
        a1_shift = abs(a1[a]['a1_idx'] - a1[b]['a1_idx'])
        a2_shift = abs(a1[a]['a2_idx'] - a1[b]['a2_idx'])
        assert a1_shift >= 2, (a, b, a1_shift)
        assert a2_shift <= 1, (a, b, a2_shift)          # 2-D Euclidean collapses to the near-d cell


def test_conservative_regime_guard_flags_ell_U():
    cells = _grid_cells()
    grid = tm.ell_grid()
    # every feasible cell's optimum + ell_L pass the upper-bracket guard
    for cid, (B, w) in cells.items():
        if not cid.startswith('B'):
            continue
        a = tm.cell_analysis(B, w, grid=grid)
        if a['feasible']:
            assert a['regime_star'], cid
            assert a['regime_L'], cid
    # the high-w/B feasible cells' sag boundary ell_U is INCONCLUSIVE-regime (honest exclusion)
    for cid in ('B1_w1', 'B2_w2'):
        B, w = cells[cid]
        a = tm.cell_analysis(B, w, grid=grid)
        assert a['feasible'] and not a['regime_U'], cid


def test_band_crossings_and_iou():
    grid = tm.ell_grid()
    # synthetic 0->1->0 success curve: succeed on a middle window
    y = np.zeros(len(grid)); y[6:12] = 1.0
    lL, lU, cens = tm.band_crossings(y, grid)
    assert cens is None and lL is not None and lU is not None and lL < lU
    # all-below -> both censored; up-no-down -> upper censored
    assert tm.band_crossings(np.zeros(len(grid)), grid)[2] == 'both'
    y2 = np.zeros(len(grid)); y2[6:] = 1.0
    assert tm.band_crossings(y2, grid)[2] == 'upper'
    # IoU/Hausdorff sanity
    assert tm.interval_iou(0.2, 0.3, 0.2, 0.3) == 1.0
    assert abs(tm.hausdorff(0.20, 0.30, 0.22, 0.31) - 0.02) < 1e-12


def test_j_inf_below_feasible_J():
    cells = _grid_cells()
    grid = tm.ell_grid()
    min_feasible_J = np.inf
    for cid, (B, w) in cells.items():
        if not cid.startswith('B'):
            continue
        Jg = tm.objective_J(grid, B, w)
        feas = Jg > tm.J_INF
        if feas.any():
            min_feasible_J = min(min_feasible_J, float(Jg[feas].min()))
    assert tm.J_INF < min_feasible_J
