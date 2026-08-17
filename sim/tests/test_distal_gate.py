"""C0 golden tests for the frozen distal_gate.decide() decision surface.

Byte-stable golden hash of decide(); GO / NO-GO(wrong-sign) / INCONCLUSIVE(censored) / mixed
fixtures on self-consistent synthetic landscapes; and a hard assertion that distal_gate.py
never imports sim.analyze_gate (the byte-frozen lift surface stays untouched).
"""
from __future__ import annotations
import ast, hashlib
from pathlib import Path
import numpy as np
import sim.tip_model as tm
import sim.distal_gate as dg

ROOT = Path(__file__).resolve().parents[1]

# hardening-A in-regime grid values (prior committed)
CELLS = {'B1_w0': (0.01610520612280113, 0.1962), 'B2_w0': (0.03825920568613013, 0.1962),
         'B3_w0': (0.09110646283229176, 0.1962), 'B4_w0': (0.21527426878237318, 0.1962)}


def _golden_hash():
    src = Path(dg.__file__).read_text(); tree = ast.parse(src); lines = src.splitlines(keepends=True)
    for fn in tree.body:
        if isinstance(fn, ast.FunctionDef) and fn.name == 'decide':
            seg = ''.join(lines[fn.lineno - 1:fn.end_lineno]); return hashlib.sha256(seg.encode()).hexdigest()
    raise AssertionError('decide() not found')


def test_decide_source_byte_stable():
    # pin the frozen decision surface; any semantic edit must consciously update this hash
    assert _golden_hash() == GOLDEN_DECIDE_SHA, _golden_hash()


def test_no_analyze_gate_import():
    src = Path(dg.__file__).read_text(); tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all('analyze_gate' not in n.name for n in node.names), 'must not import analyze_gate'
        if isinstance(node, ast.ImportFrom):
            assert node.module is None or 'analyze_gate' not in node.module, 'must not import from analyze_gate'


def _frozen(ids, grid):
    """Minimal synthetic frozen manifest for a fixed-w B-up column."""
    cells = []
    for cid in ids:
        B, w = CELLS[cid]; a = tm.cell_analysis(B, w, grid=grid); a['id'] = cid
        cells.append(dict(id=cid, B_eff=B, w=w, feasible=a['feasible'], argmax_idx=a['argmax_idx'],
                          ell_L=a['ell_L'], ell_U=a['ell_U'], regime_star=a['regime_star'],
                          regime_L=a['regime_L'], regime_U=a['regime_U']))
    contrasts = [dict(a=ids[k], b=ids[k + 1], kind='B_up') for k in range(len(ids) - 1)]
    return dict(objective=dict(tau=tm.TAU), grasp=dict(step=tm.ELL_STEP), grid=cells, ratio_pairs=[],
                eligible_adjacency_contrasts=contrasts, gate_truth_table=dict(ratio_tolerance_cells=1))


def _ideal_curves(ids, grid):
    """Self-consistent measured landscape = analytic success/J for each cell (=> measured==predicted)."""
    ps = {}
    for cid in ids:
        B, w = CELLS[cid]
        ps[cid] = dict(mean_success=[float(x) for x in tm.success(grid, B, w)],
                       mean_J=[float(x) for x in tm.objective_J(grid, B, w)], B_eff=B, w=w)
    return ps


def test_go_on_self_consistent_landscape():
    grid = tm.ell_grid(); ids = ['B1_w0', 'B2_w0', 'B3_w0', 'B4_w0']
    v = dg.decide(_ideal_curves(ids, grid), _frozen(ids, grid), grid)
    assert v['verdict'] == 'GO', (v['fails'], v['inconclusive'])
    assert all(d['status'] == 'PASS' for d in v['argmax_direction'])


def test_no_go_on_wrong_sign():
    grid = tm.ell_grid(); ids = ['B1_w0', 'B2_w0', 'B3_w0', 'B4_w0']
    ps = _ideal_curves(ids, grid)
    # swap the two extreme cells' curves -> argmax moves the wrong way on the adjacent contrasts
    ps['B1_w0'], ps['B4_w0'] = ps['B4_w0'], ps['B1_w0']
    v = dg.decide(ps, _frozen(ids, grid), grid)
    assert v['verdict'] == 'NO-GO', v['inconclusive']
    assert any('wrong-signed' in f or 'feasibility contradiction' in f or '< 2 cells' in f for f in v['fails'])


def test_inconclusive_on_censored():
    grid = tm.ell_grid(); ids = ['B1_w0', 'B2_w0', 'B3_w0', 'B4_w0']
    ps = _ideal_curves(ids, grid)
    ps['B2_w0'] = dict(mean_success=[0.0] * len(grid), mean_J=[tm.J_INF] * len(grid),
                       B_eff=CELLS['B2_w0'][0], w=CELLS['B2_w0'][1])   # all-fail -> censored + feasibility contradiction
    v = dg.decide(ps, _frozen(ids, grid), grid)
    # B2_w0 is predicted feasible but measured infeasible -> that is a FAIL (contradiction); ensure it's caught
    assert v['verdict'] == 'NO-GO'
    assert any('feasibility contradiction B2_w0' in f for f in v['fails'])


def test_inconclusive_without_contradiction():
    grid = tm.ell_grid(); ids = ['B1_w0', 'B2_w0']
    fr = _frozen(ids, grid)
    fr['grid'][1]['feasible'] = False                 # pretend B2_w0 predicted infeasible
    ps = _ideal_curves(ids, grid)
    ps['B2_w0'] = dict(mean_success=[0.0] * len(grid), mean_J=[tm.J_INF] * len(grid),
                       B_eff=CELLS['B2_w0'][0], w=CELLS['B2_w0'][1])
    v = dg.decide(ps, fr, grid)
    # no contradiction (both predict+measure infeasible); the B1->B2 contrast lacks a valid argmax -> INCONCLUSIVE
    assert v['verdict'] == 'inconclusive', (v['fails'], v['inconclusive'])


# filled in after first computation
GOLDEN_DECIDE_SHA = "12c10c7820f01439204f4b913b2c394b1931b072b4ecafdfb647e88fe19c0e34"
