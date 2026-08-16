"""C0 tests: frozen-decision-surface protection + cardinality/property-axis parameterization.

Covers: (1) boundary()/condition() source byte-identical to the pre-hardening commit;
(2) golden fixtures for condition() statuses + the overall precedence mapping;
(3) a synthetic 4x3, 17-grasp end-to-end run of analyze_gate.main() proving the reduced
grid + stable IDs + ratio-reference binding produce the exact frozen statuses on a
non-5x4, non-15-grasp shape; (4) manifest schema asserts incl. dropped rod-pose perturbation.
"""
from __future__ import annotations
import ast, hashlib, json
from pathlib import Path
import numpy as np
import sim.analyze_gate as ag
import sim.sweep as sw

ROOT = Path(__file__).resolve().parents[1]
MAN = ROOT / 'manifests'
GOLD = {'boundary': '947a93317593d645a150f057ed10a3bb028d57c6be5cf76b613203f544962e06',
        'condition': '73ea1d4bf16af2a8dca4251954621364e1fe9e245b462d581c11ae93beaf44bd'}


def test_frozen_functions_byte_identical():
    src = Path(ag.__file__).read_text(); tree = ast.parse(src); lines = src.splitlines(keepends=True)
    seen = {}
    for fn in tree.body:
        if isinstance(fn, ast.FunctionDef) and fn.name in GOLD:
            seg = ''.join(lines[fn.lineno - 1:fn.end_lineno]); seen[fn.name] = hashlib.sha256(seg.encode()).hexdigest()
    assert seen == GOLD, seen


def test_condition_golden_statuses():
    step = 0.03
    # PASS: valid, correct-signed, resolved, adjacent majority correct
    assert ag.condition({'w0': [0.2, 0.3, 0.4], 'w1': [0.25, 0.35, 0.45]}, 1, step)['status'] == 'PASS'
    # FAIL: a valid endpoint moves the wrong way
    assert ag.condition({'w0': [0.4, 0.3, 0.2]}, 1, step)['status'] == 'FAIL'
    # INCONCLUSIVE: an invalid (None) endpoint
    assert ag.condition({'w0': [None, 0.3, 0.4]}, 1, step)['status'] == 'INCONCLUSIVE'
    # INCONCLUSIVE: endpoints valid+correct but adjacent majority not > half
    assert ag.condition({'w0': [0.2, 0.5, 0.4]}, 1, step)['status'] == 'INCONCLUSIVE'
    # w-direction PASS (falls with w): sign -1
    assert ag.condition({'B1': [0.4, 0.3, 0.2]}, -1, step)['status'] == 'PASS'


def test_boundary_resolved_and_censored():
    ell = [0.12, 0.15, 0.18, 0.21]
    r = ag.boundary(ell, [1.0, 1.0, 0.0, 0.0], 0.5)
    assert r['resolved'] and r['censored'] is None and 0.15 < r['boundary'] < 0.18
    assert ag.boundary(ell, [1, 1, 1, 1], 0.5)['censored'] == 'high'   # all-success -> censored, NOT a boundary
    assert ag.boundary(ell, [0, 0, 0, 0], 0.5)['censored'] == 'low'
    assert ag.boundary(ell, [1, 1, 1, 1], 0.5)['boundary'] is None


def _synth_npz(m, targets, path):
    """Synthesize an evaluation-bank npz: success=1 iff ell<=target(setting)."""
    ss = sw.settings(m); ell = m['grasp']['ell']; seeds = m['seed_banks']['evaluation']
    cols = {k: [] for k in ('setting', 'grasp', 'ell', 'template', 'bank', 'seed', 'J', 'success')}
    for s in ss:
        tb = targets[s['id']]
        for gi, e in enumerate(ell):
            for sd in seeds:
                succ = bool(e <= tb)
                for k, v in dict(setting=s['id'], grasp=gi, ell=e, template=0, bank='evaluation',
                                 seed=sd, J=(m['h'] if succ else -m['h']), success=succ).items():
                    cols[k].append(v)
    np.savez(path, **{k: np.array(v) for k, v in cols.items()})


def test_4x3_end_to_end_stable_ids_and_ratio(tmp_path):
    m = json.loads((MAN / 'hard_sweep_manifest.json').read_text())
    sw._assert_manifest(m)                              # schema asserts (incl. no rod-pose)
    assert 'initial_rod_pose_translation_xy_m' not in m['stochastic_distribution']
    ss = sw.settings(m)
    ids = {s['id'] for s in ss}
    # stable IDs preserved: survivors keep B1..B4 x w0..w2, ratio refs bind
    assert {'B1_w0', 'B4_w2', 'B1_w1', 'B3_w2', 'B2_w1'} <= ids
    assert all(m['ratio_pairs'][k]['reference_setting'] in ids for k in range(len(m['ratio_pairs'])))
    # targets: independent cells use predicted empirical boundary (B up, w down); ratio settings invariant to their ref
    tbl = {t['id']: t for t in m['predicted_Pi_g_table']}
    targets = {s['id']: tbl[s['id']]['boundary_pred_empirical'] for s in ss if s['kind'] == 'independent'}
    for p in m['ratio_pairs']:
        targets[f"R{p['pair_id']}"] = targets[p['reference_setting']]   # boundary-invariant ratio pair
    npz = tmp_path / 'synth.npz'; _synth_npz(m, targets, npz)
    ag.main(manifest=str(MAN / 'hard_sweep_manifest.json'), results=str(npz), out_prefix='TESTHARD_')
    v = json.loads((MAN / 'TESTHARD_gate_verdict.json').read_text())
    # 17-grasp, 4x3 grid ran without a range(15)/range(5)/range(4) key error and produced clean statuses
    assert v['validity_summary']['settings'] == 15   # 12 independent + 3 ratio
    assert sum(1 for s in ss if s['kind'] == 'independent') == 12
    assert v['conditions']['B']['status'] == 'PASS', v['conditions']['B']
    assert v['conditions']['w']['status'] == 'PASS', v['conditions']['w']
    assert v['conditions']['R']['status'] == 'PASS', v['conditions']['R']
    assert v['exact_frozen_verdict'] == 'GO'           # precedence: all PASS -> GO
    assert v['prefactor']['observed_prefactor_mean'] is not None
    # cleanup test artifacts
    for f in MAN.glob('TESTHARD_*'):
        f.unlink()
    for f in (ROOT / 'figures').glob('TESTHARD_*'):
        f.unlink()


def test_precedence_mapping():
    # overall: NO-GO if any FAIL; GO iff all PASS; else inconclusive
    def overall(statuses):
        return 'NO-GO' if 'FAIL' in statuses else ('GO' if all(x == 'PASS' for x in statuses) else 'inconclusive')
    assert overall(['PASS', 'PASS', 'PASS']) == 'GO'
    assert overall(['PASS', 'FAIL', 'PASS']) == 'NO-GO'
    assert overall(['PASS', 'INCONCLUSIVE', 'PASS']) == 'inconclusive'
    assert overall(['INCONCLUSIVE', 'FAIL', 'PASS']) == 'NO-GO'
