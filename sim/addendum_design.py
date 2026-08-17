"""Build + freeze addendum_manifest.json — the owner-approved lift-and-clear upgrade re-run.

Graduates hardening-A's Q2 from EXPLORATORY to PRE-REGISTERED: names the lift-and-clear TASK-ONLY
temporal-history student as the PRIMARY success criterion (approach the privileged teacher on
held-out map/boundary recovery), on a CLEAN re-run with NEW seed banks (no reuse of the 2000-2002 /
3000-3004 / 1000-series draws that produced the exploratory result). Two pre-registered secondaries:
(a) same-split feature contrast full temporal (y,z) 112-D vs settled-terminal-only summary at matched
encoder capacity; (b) frame-truncation curve k=1..7, per-setting sufficient_k (tol_k=0.02) + a
DESCRIPTIVE settling-timescale panel (~ell^2*sqrt(lambda/B), lambda=w/g). A-15/16/17/22 folded.
Single-file C6 freeze BEFORE any addendum data.
"""
from __future__ import annotations
import hashlib, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MAN = ROOT / 'manifests'


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def build():
    hard_sweep = json.loads((MAN / 'hard_sweep_manifest.json').read_text())
    hard_s34 = json.loads((MAN / 'hard_s34_manifest.json').read_text())
    # NEW seed banks, disjoint from 2000-2002 / 3000-3004 / 1000-series (asserted)
    new_sel = [2100, 2101, 2102]; new_eval = [3100, 3101, 3102, 3103, 3104]; new_hist = [2100, 2101]
    old = set(range(2000, 2003)) | set(range(3000, 3005)) | set(range(1000, 1012))
    assert not (set(new_sel) | set(new_eval) | set(new_hist)) & old, 'new seeds must be disjoint from exploratory draws'
    manifest = dict(
        schema_version=1, frozen=True, task='lift_and_clear', purpose='upgrade re-run: task-only Q2 graduation',
        sweep_grid_source='hard_sweep_manifest.json', splits=hard_s34['splits'],
        seed_banks=dict(selection=new_sel, evaluation=new_eval, history=new_hist,
                        note='NEW draws; disjoint from exploratory 2000-2002/3000-3004/1000-series (red-team asserts)'),
        history_policy=dict(grasps=[0, 1, 2, 3], templates=[0, 1, 2, 3], seeds=new_hist,
                            action_metadata=True, action_note='grasp index + free-arm length in the student input (A-17)'),
        features=dict(shape_frames=7, frame_steps=[60, 120, 180, 240, 300, 360, 'settled'], M=8,
                      task_shape_dim=112, pooled_shape_dim=32, proprio_dim=8, action_dim=2,
                      full_temporal='proprio(8) + pool_temporal(112)->32 + action(2) = 42-D',
                      terminal_only='proprio(8) + settled-frame(16) + action(2) = 26-D (settled-terminal summary)'),
        critic_head=dict(kind='J_regression_plus_success', hidden=[32, 32], seed=3403, epochs=300),
        primary=dict(name='task_only_temporal_student_approaches_teacher',
                     rule=('task-only temporal-history student map RMSE within 1.5x teacher AND student-blind CI/contrast '
                           '< 0 on map RMSE AND the tau=0.5 boundary-index error, on held-out TEST'),
                     metric='held-out TEST map RMSE + tau=0.5 boundary-index error',
                     note='PRE-REGISTERED as PRIMARY (fixes the exploratory "no frozen task-only rule" caveat)'),
        secondary_a=dict(name='feature_schema_contrast',
                         rule='full temporal (y,z) 112-D vs settled-terminal-only summary at MATCHED encoder capacity',
                         note='pins whether the temporal schema (not just capacity) fixed the student'),
        secondary_b=dict(name='frame_truncation_curve', k_range=[1, 2, 3, 4, 5, 6, 7],
                         sufficient_k=dict(tol_k=0.02, rule='smallest k with per-setting map RMSE_k <= RMSE_7 + tol_k; else 7; unresolved RMSE_7 -> no sufficient k'),
                         stratified='PER-SETTING (A owner: watching horizon is property-dependent), not only the average',
                         timescale_panel=dict(descriptive=True, formula='sufficient_k vs predicted settling timescale ~ ell^2*sqrt(lambda/B_eff), lambda=segment_mass/interval=w/g',
                                              note='DESCRIPTIVE observation only (theory prescribing how long to watch), NOT a pre-registered claim'),
                         naming='named "frame-truncation curve" (frame-prefix), NEVER "adaptation curve" (reserved for interaction-prefix, deferred to finding 32)'),
        adopt_now=dict(A15='ratio pairs = invariance controls; generalization + uncertainty at the unique-(B,w)-group level (not per-rollout, not extra unique settings)',
                       A16='multiple training seeds [3403,3413,3423] + seed variability reported; eval-draw bootstrap LABELED conditional on the trained models; carry hardening-A degenerate-bootstrap caveat',
                       A17='explicit grasp/free-length action metadata in the history features',
                       A22='outputs named history-variant comparisons; frame-truncation curve is a k-FRAME prefix, never "adaptation curve"'),
        training_seeds=[3403, 3413, 3423],
        follow_up_out_of_scope='adaptive stopping rule (watch until the latent stabilizes) is a deployment-protocol follow-up, NOT run here',
        input_artifact_sha256={f: sha256(MAN / f) for f in ('hard_sweep_manifest.json', 'hard_s34_manifest.json', 'calibration.json')},
    )
    out = MAN / 'addendum_manifest.json'
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    print(json.dumps(dict(output=str(out), sha256=sha256(out), new_selection=new_sel, new_eval=new_eval,
                          new_history=new_hist, primary=manifest['primary']['name']), indent=2))
    return sha256(out)


if __name__ == '__main__':
    build()
