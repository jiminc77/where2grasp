"""Build the frozen hard_s34_manifest.json (Part-2 pre-registration).

Consumes the frozen probe length (from the exploratory qualification screen) and the
C2 input-artifact sha256s (sweep manifest/results/landscape/gate_verdict). Freezes the
unified leak-free split, exact feature dims, inherited tolerances + truth table, and the
map-recovery-as-PRIMARY success rule. Written as its OWN commit (C4) BEFORE any histories.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MAN = ROOT / 'manifests'

UNIVERSE = ['B1_w0', 'B1_w1', 'B1_w2', 'B2_w0', 'B2_w1', 'B2_w2',
            'B3_w0', 'B3_w1', 'B3_w2', 'B4_w0', 'B4_w1', 'B4_w2', 'R0', 'R1', 'R2']
TEST = ['B1_w1', 'R0', 'B3_w2', 'R1', 'B2_w1', 'R2']
VAL = ['B2_w2', 'B4_w1']
TRAIN = ['B1_w0', 'B1_w2', 'B2_w0', 'B3_w0', 'B3_w1', 'B4_w0', 'B4_w2']
FINAL_TEST_PAIRS = [['B1_w1', 'R0'], ['B3_w2', 'R1'], ['B2_w1', 'R2']]


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def folds(train, K=3):
    s = sorted(train); f = {str(k): [] for k in range(K)}
    for i, sid in enumerate(s):
        f[str(i % K)].append(sid)
    return f


def build(ell_probe):
    assert set(TRAIN) | set(VAL) | set(TEST) == set(UNIVERSE)
    assert not (set(TRAIN) & set(VAL)) and not (set(TRAIN) & set(TEST)) and not (set(VAL) & set(TEST))
    qual = json.loads((ROOT / 'hardening/exploratory/probe_qualification.json').read_text())
    assert qual['chosen_ell_probe'] == ell_probe, (qual['chosen_ell_probe'], ell_probe)
    manifest = dict(
        schema_version=1, frozen=True,
        sweep_manifest='hard_sweep_manifest.json',
        universe=UNIVERSE,
        splits=dict(train=TRAIN, val=VAL, test=TEST, final_test_pairs=FINAL_TEST_PAIRS),
        history_policy=dict(grasps=[0, 1, 2, 3], templates=[0, 1, 2, 3], seeds=[2000, 2001],
                            N_hist_per_setting=32, matched_CRN=True,
                            note='each setting: 4 grasps x 4 templates x 2 seeds = 32; ref and R share (grasp,template,seed) CRN'),
        features=dict(shape_frames=7, frame_steps=[60, 120, 180, 240, 300, 360, 'settled'], M=8, per_frame_dim=16,
                      task_shape_dim=112, probe_shape_dim=16, probe_enriched_shape_dim=128, proprio_dim=8, wrench_dim=1,
                      feature_vectors=dict(proprio=8, task_shape=120, task_shape_wrench=121,
                                           probe_shape=136, probe_shape_wrench=137),
                      ridge_reduction='temporal mean+last-frame pooling of the (y,z) frames (task 112->32; probe-enriched 128->48)',
                      allow_list=['shape_yz_temporal_112', 'probe_shape_yz_16', 'proprio_terminal_pose_and_drive_8', 'normalized_supported_Fz'],
                      deny_list=['setting_id', 'pair_id', 'B_eff', 'w', 'ratio', 'raw_vertex_count',
                                 'absolute_rod_length', 'filename', 'order_index', 'target-derived fields']),
        probe=dict(ell_probe=ell_probe, regime='small-deflection Pi_g<=0.3',
                   contributes='settled-only 16-D (y,z) frame prepended to the 7 task frames',
                   qualification='hardening/exploratory/probe_qualification.json', qualification_verdict=qual['verdict']),
        targets=['log10_B_eff', 'log10_w', 'log10_B_eff_over_w'], metric='log10-RMSE',
        margins=dict(tol_ratio=0.10, tol_shape=0.05, K=3, tol_indiv=0.15),
        truth_table=('failed positive-control OR guard => INCONCLUSIVE; premises pass but null OR repair fails => FAIL; all pass => PASS'),
        grouped_cv=dict(K=3, rule='lexicographic sort of TRAIN setting IDs; fold=index mod 3', folds=folds(TRAIN)),
        ridge_alpha_grid=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
        models=dict(ridge=dict(seed=3401), mlp=dict(hidden=[64, 64], epochs=200, seed=3402),
                    critic=dict(hidden=[32, 32], epochs=300, seed=3403), phi_theta=dict(hidden=[32], z=4),
                    student_encoder=dict(hidden=[64, 32], z=4), distill_epochs=300,
                    repeated_seeds=[3401, 3411, 3421]),
        map_recovery=dict(primary=True, metrics=['map_rmse', 'correlation', 'tau_boundary_index_error'],
                          boundary_rule='first tau=0.5 crossing linearly interpolated in grasp-index units; censored (excluded) if no crossing',
                          bootstrap=dict(n=2000, seed=7, paired_over='matched evaluation-seed blocks'),
                          success_rule=('probe-enriched student - blind CI95 < 0 (better) on map RMSE AND boundary error, '
                                        'AND probe-enriched student / teacher map-RMSE ratio <= 1.5')),
        input_artifact_sha256={
            'hard_sweep_manifest.json': sha256(MAN / 'hard_sweep_manifest.json'),
            'hard_sweep_results.npz': sha256(MAN / 'hard_sweep_results.npz'),
            'hard_sweep_landscape.json': sha256(MAN / 'hard_sweep_landscape.json'),
            'hard_gate_verdict.json': sha256(MAN / 'hard_gate_verdict.json'),
            'calibration.json': sha256(MAN / 'calibration.json'),
        },
    )
    out = MAN / 'hard_s34_manifest.json'; out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    print(json.dumps(dict(output=str(out), sha256=sha256(out), ell_probe=ell_probe,
                          train=len(TRAIN), val=len(VAL), test=len(TEST), universe=len(UNIVERSE)), indent=2))
    return sha256(out)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--ell-probe', type=float, required=True); a = p.parse_args(); build(a.ell_probe)
