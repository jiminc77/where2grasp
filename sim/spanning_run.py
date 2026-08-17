"""Spanning-cohort selection-regret re-test (owner-approved phase after the addendum).

The distal C4 selection-regret PRIMARY was NULL because the pre-registered TEST cohort's optima
CLUSTERED (bands overlapped at l~=0.26 -> a blind constant grasp hedged). This phase re-runs the
SAME distal physics on NEW seed banks (disjoint from ALL prior) with a NEW pre-registered split
whose TEST cohort's unique-(B,w)-group optima SPAN the l_delta range with WELL-SEPARATED bands
(predicted optima pairwise >=2 grid cells apart, asserted at freeze from prior committed calibration
data only), so selection-regret can discriminate. Selection-regret PRIMARY + map recovery co-primary;
ratio pairs = invariance controls; multiple training seeds; unified grid-argmax upper-bracket
regime-guard convention (owner ruling; future gates). The spanning_manifest is distal_sweep- and
distal_history-compatible (grid = frozen distal grid; NEW seeds) so those runners are reused as-is.
Single-file freeze BEFORE any data.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import sim.tip_model as tm

ROOT = Path(__file__).resolve().parent
MAN = ROOT / 'manifests'

# Spanning split (measured/predicted bands well-separated): B1_w1 ~0.246 / B2_w0 ~0.370 / B4_w0 ~0.570.
TEST = ['B1_w1', 'B2_w0', 'B4_w0']
VAL = ['B2_w1', 'B3_w2']
TRAIN = ['B1_w0', 'B1_w2', 'B2_w2', 'B3_w0', 'B3_w1', 'B4_w1', 'B4_w2']
NEW_SEL = [2200, 2201, 2202]; NEW_EVAL = [3200, 3201, 3202, 3203, 3204]; NEW_HIST = [2200, 2201]
ALL_PRIOR = set(range(2000, 2003)) | set(range(3000, 3005)) | set(range(1000, 1012)) | set(range(2100, 2103)) | set(range(3100, 3105))


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def freeze():
    dm = json.loads((MAN / 'distal_manifest.json').read_text())
    grid = np.array(dm['grasp']['ell']); step = dm['grasp']['step']
    prop = {c['id']: (c['B_eff'], c['w']) for c in dm['grid']}
    # ---- pre-freeze asserts (analytic, prior committed data only) ----
    viol = []
    assert set(TRAIN) | set(VAL) | set(TEST) == {c['id'] for c in dm['grid']}, 'split must cover the 12 independent cells'
    assert not (set(TRAIN) & set(VAL)) and not (set(TRAIN) & set(TEST)) and not (set(VAL) & set(TEST))
    newseeds = set(NEW_SEL) | set(NEW_EVAL) | set(NEW_HIST)
    if newseeds & ALL_PRIOR:
        viol.append(f'new seeds overlap prior banks: {sorted(newseeds & ALL_PRIOR)}')
    # TEST unique-group predicted optima pairwise >=2 grid cells apart (well-separated bands)
    test_argmax = {}
    for sid in TEST:
        B, w = prop[sid]; a = tm.cell_analysis(B, w, grid=grid)
        if a['argmax_idx'] is None or not a['feasible']:
            viol.append(f'{sid}: not a feasible interior optimum')
        test_argmax[sid] = a['argmax_idx']
    idxs = [(sid, test_argmax[sid]) for sid in TEST]
    for i in range(len(idxs)):
        for j in range(i + 1, len(idxs)):
            if idxs[i][1] is None or idxs[j][1] is None:
                continue
            gap = abs(idxs[i][1] - idxs[j][1])
            if gap < 2:
                viol.append(f'{idxs[i][0]} vs {idxs[j][0]}: predicted optima {gap} cells apart (<2)')
    if viol:
        (MAN / 'spanning_infeasible.json').write_text(json.dumps({'infeasible': True, 'violations': viol}, indent=2))
        raise SystemExit('SPANNING COHORT INFEASIBLE (escalate; never weaken):\n' + '\n'.join(viol))
    # ---- assemble a distal_sweep- + distal_history-compatible manifest with NEW seeds ----
    manifest = dict(
        schema_version=1, frozen=True, task='distal_tip_placement', purpose='spanning-cohort selection-regret re-test',
        distal_manifest='spanning_manifest.json',              # self: grid lives here (distal grid copy)
        objective=dm['objective'], integrator=dm['integrator'], interval=dm['interval'], gravity=dm['gravity'],
        drive_steps=dm['drive_steps'], pi_g_max=dm['pi_g_max'],
        regime_guard=dict(**dm['regime_guard'], unified_convention='grid-argmax upper-bracket endpoint for predicted AND measured (owner ruling; future gates)'),
        grasp=dm['grasp'], templates=dm['templates'], stochastic_distribution=dm['stochastic_distribution'],
        grid=dm['grid'], ratio_pairs=dm['ratio_pairs'],
        seed_banks=dict(selection=NEW_SEL, evaluation=NEW_EVAL, history=NEW_HIST,
                        note='NEW draws disjoint from ALL prior banks (2000-2002/3000-3004/1000-series/2100-2102/3100-3104)'),
        selection_rule=dm['selection_rule'], evaluation_rule=dm['evaluation_rule'],
        universe=[c['id'] for c in dm['grid']] + [r['id'] for r in dm['ratio_pairs']],
        splits=dict(train=TRAIN, val=VAL, test=TEST),
        history_policy=dict(grasps=[0, 1, 2, 3], templates=[0, 1, 2, 3], seeds=NEW_HIST, action_metadata=True,
                            note='grasp/free-length action metadata in the student input (A-17)'),
        test_cohort=dict(rationale='unique-(B,w)-group TEST optima SPAN l_delta with WELL-SEPARATED bands; pairwise predicted optima >=2 grid cells apart (asserted at freeze from prior committed calibration data)',
                         test_predicted_argmax={sid: int(test_argmax[sid]) for sid in TEST},
                         test_predicted_ell={sid: float(grid[test_argmax[sid]]) for sid in TEST},
                         pairwise_cell_gaps={f'{TEST[i]}-{TEST[j]}': int(abs(test_argmax[TEST[i]] - test_argmax[TEST[j]])) for i in range(len(TEST)) for j in range(i + 1, len(TEST))}),
        primary=dict(name='selection_regret_discriminates',
                     rule='teacher AND task-only student regret < blind regret, aggregated over UNIQUE (B,w) TEST groups; feasible-masked J-head argmax; regret vs oracle max_g E[J]',
                     J_inf=tm.J_INF),
        co_primary=dict(name='map_recovery', rule='held-out TEST map RMSE on the success curve; teacher & student < blind'),
        ratio_controls='ratio pairs R0/R1/R2 reported as invariance controls (argmax(R)~=argmax(ref)); they do not add unique groups',
        training_seeds=[3403, 3413, 3423],
        input_artifact_sha256={f: sha256(MAN / f) for f in ('distal_manifest.json', 'calibration.json')},
        tip_model_constants=dm['tip_model_constants'],
    )
    out = MAN / 'spanning_manifest.json'; out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    print(json.dumps(dict(output=str(out), sha256=sha256(out), test=TEST,
                          test_predicted_ell={sid: float(grid[test_argmax[sid]]) for sid in TEST},
                          pairwise_cell_gaps=manifest['test_cohort']['pairwise_cell_gaps'],
                          new_seeds=dict(selection=NEW_SEL, evaluation=NEW_EVAL, history=NEW_HIST)), indent=2))
    return sha256(out)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--stage', choices=['freeze'], default='freeze'); a = p.parse_args(); freeze()
