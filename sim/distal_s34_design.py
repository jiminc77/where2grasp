"""Build + freeze sim/manifests/distal_s34_manifest.json (distal Part-3 pre-registration).

Records the C2 input shas + the source-encoder bundle hash, the leak-free split (distal TEST is
disjoint from the SOURCE encoder's TRAIN+VAL by construction: distal TRAIN/VAL == source
TRAIN/VAL, so distal TEST is the disjoint complement + the 3 ratio pairs as invariance controls),
the pre-registered PRIMARY (selection regret) + map/band co-primary, and the transfer
non-inferiority rule. Written as its OWN single-file commit (C3) BEFORE any Part-3 data.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MAN = ROOT / 'manifests'

# distal universe = 12 property cells + 3 ratio pairs (B1_w2 is an infeasible landscape but its
# history still encodes B/w). Split reuses the hardening-A structure so distal TEST is disjoint
# from the source encoder's TRAIN+VAL (which are the distal TRAIN+VAL).
UNIVERSE = ['B1_w0', 'B1_w1', 'B1_w2', 'B2_w0', 'B2_w1', 'B2_w2',
            'B3_w0', 'B3_w1', 'B3_w2', 'B4_w0', 'B4_w1', 'B4_w2', 'R0', 'R1', 'R2']
TRAIN = ['B1_w0', 'B1_w2', 'B2_w0', 'B3_w0', 'B3_w1', 'B4_w0', 'B4_w2']
VAL = ['B2_w2', 'B4_w1']
TEST = ['B1_w1', 'R0', 'B3_w2', 'R1', 'B2_w1', 'R2']
FINAL_TEST_PAIRS = [['B1_w1', 'R0'], ['B3_w2', 'R1'], ['B2_w1', 'R2']]
SOURCE_TRAIN = ['B1_w0', 'B1_w2', 'B2_w0', 'B3_w0', 'B3_w1', 'B4_w0', 'B4_w2']
SOURCE_VAL = ['B2_w2', 'B4_w1']


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def folds(train, K=3):
    s = sorted(train); f = {str(k): [] for k in range(K)}
    for i, sid in enumerate(s):
        f[str(i % K)].append(sid)
    return f


def build(source_bundle='source_encoder_bundle.pt'):
    assert set(TRAIN) | set(VAL) | set(TEST) == set(UNIVERSE)
    assert not (set(TRAIN) & set(VAL)) and not (set(TRAIN) & set(TEST)) and not (set(VAL) & set(TEST))
    # transfer split hygiene: distal TEST disjoint from source encoder TRAIN+VAL AND distal head TRAIN/VAL
    assert not (set(TEST) & (set(SOURCE_TRAIN) | set(SOURCE_VAL))), 'distal TEST leaks into source encoder fit'
    inputs = {}
    for f in ('distal_manifest.json', 'distal_sweep_results.npz', 'distal_sweep_landscape.json',
              'distal_gate_verdict.json', 'calibration.json'):
        p = MAN / f
        inputs[f] = sha256(p) if p.exists() else None
    bundle_p = MAN / source_bundle
    manifest = dict(
        schema_version=1, frozen=True, task='distal_tip_placement',
        distal_manifest='distal_manifest.json',
        universe=UNIVERSE, splits=dict(train=TRAIN, val=VAL, test=TEST, final_test_pairs=FINAL_TEST_PAIRS),
        source_encoder=dict(bundle=source_bundle, bundle_sha256=(sha256(bundle_p) if bundle_p.exists() else None),
                            source_train=SOURCE_TRAIN, source_val=SOURCE_VAL,
                            note='FROZEN lift-and-clear task-only encoder; distal TEST disjoint from source TRAIN+VAL'),
        history_policy=dict(grasps=[0, 1, 2, 3], templates=[0, 1, 2, 3], seeds=[2000, 2001],
                            N_hist_per_setting=32, matched_CRN=True,
                            action_metadata=True,
                            action_note='grasp index + free-arm length (the action) included in the student input (A-17)'),
        features=dict(shape_frames=7, frame_steps=[60, 120, 180, 240, 300, 360, 'settled'], M=8,
                      task_shape_dim=112, pooled_shape_dim=32, proprio_dim=8, action_dim=2,
                      student_input='proprio(8) + pool_temporal(112)->32 + action(2) = 42-D',
                      transfer_encoder_input='proprio(8) + pool_temporal(112)->32 = 40-D (source contract)'),
        rows=['teacher', 'blind', 'task_student', 'sysid', 'transfer_student'],
        critic_head=dict(kind='J_regression', note='J-regression head defines selection (BCE cannot); success head optional',
                         hidden=[32, 32], seed=3403, epochs=300, loss='mse_on_J', loss_weights=dict(J=1.0)),
        primary=dict(name='selection_regret',
                     formula='regret(s) = max_g E[J|s,g] - E[J|s,g_selected] on winner-only evaluation',
                     J_inf=-1.0, tie='lowest grid index', aggregation='mean over UNIQUE (B,w) TEST groups',
                     ratio_pairs='invariance controls (ratio-cluster uncertainty), not extra unique groups (A-15)',
                     sign='student & teacher regret < blind', bootstrap=dict(n=2000, seed=7, kind='paired block over eval-seed blocks',
                     label='LABELED conditional on the trained models; multiple training seeds report seed variability (A-16)'),
                     precedence='PASS if student & teacher regret CI< blind; INCONCLUSIVE if CI overlaps; report null'),
        co_primary=dict(name='map_and_band_recovery', map='held-out TEST map RMSE',
                        band='ell_L(first-up)/ell_U(last-down) crossings; censored/regime edges excluded; IoU + per-edge + Hausdorff',
                        combined_rule='student-blind CI<0 on map RMSE AND each present in-regime band edge AND student/teacher<=1.5x'),
        transfer=dict(rows=['transfer_student(frozen encoder)', 'scratch_student', 'blind'],
                      metrics=['map_rmse', 'band_edge_error', 'selection_regret'],
                      non_inferiority_margin=0.05, ci=dict(n=2000, seed=7, kind='paired block-bootstrap'),
                      rule=('YES iff frozen NON-INFERIOR to scratch (CI upper of (frozen-scratch) map RMSE < 0.05) '
                            'AND BETTER than blind ((blind-frozen) CI excludes 0 positive); NO iff frozen not better '
                            'than blind; else NOT-ESTABLISHED'),
                      equal_budget=dict(scratch='encoder+head joint', identical='histories/preproc/arch/head-init/seeds/updates/early-stop/eval',
                                        report='equal-update budget AND trainable-param-count difference; multiple seeds'),
                      frozen_state='requires_grad=False; head optimizer excludes encoder tensors; encoder+scaler hash identical pre/post'),
        training_seeds=[3403, 3413, 3423],
        multi_seed_note='report seed variability across training_seeds; eval-draw bootstrap LABELED conditional-on-models (A-16)',
        grouped_cv=dict(K=3, folds=folds(TRAIN)),
        honest_nulls=dict(Q1='argmax shift <2 cells / wrong sign -> NO-GO',
                          Q2='student & teacher regret CI overlaps blind -> conditioning does not discriminate',
                          Q3='frozen not better than blind -> NO (or NOT-ESTABLISHED if overlapping)'),
        input_artifact_sha256=inputs,
    )
    out = MAN / 'distal_s34_manifest.json'
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    print(json.dumps(dict(output=str(out), sha256=sha256(out), train=len(TRAIN), val=len(VAL),
                          test=len(TEST), universe=len(UNIVERSE),
                          test_disjoint_from_source=not (set(TEST) & (set(SOURCE_TRAIN) | set(SOURCE_VAL)))), indent=2))
    return sha256(out)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--source-bundle', default='source_encoder_bundle.pt')
    a = p.parse_args(); build(a.source_bundle)
