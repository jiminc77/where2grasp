"""Addendum runner: lift-and-clear upgrade re-run on NEW draws (histories + map-recovery critic).

Extracts NEW-seed lift-and-clear temporal histories, then runs the PRE-REGISTERED map/boundary
recovery: teacher (privileged B,w) / blind / TASK-ONLY temporal-history student (PRIMARY) + a
settled-terminal-only student at matched capacity (secondary a) + a frame-truncation curve k=1..7
(secondary b, per-setting sufficient_k + a descriptive settling-timescale panel). Reuses the distal
two-head critic + hardening-A history primitives. Writes addendum_results.json + figures.
"""
from __future__ import annotations
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sim.history import shape_frame, temporal_shape, drive_summary, FRAME_STEPS
from sim.identify import pool_temporal
from sim.distal_critic import Phi, TwoHead, train_critic, seed_all
from sim.sweep import settings

ROOT = Path(__file__).resolve().parent
MAN = ROOT / 'manifests'


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def extract_histories(am, batch_size=90):
    """New-seed lift-and-clear temporal (y,z) 7-frame histories + action metadata."""
    from sim.scene import build_scene, add_straight_rod, add_moving_clamp, attach_moving_clamp, vertices
    from sim.material import apply_properties
    sm = json.loads((MAN / am['sweep_grid_source']).read_text()); ss = {s['id']: s for s in settings(sm)}
    integ = sm['integrator']; interval = sm['interval']; bounds = sm['stochastic_distribution']
    ell = sm['grasp']['ell']; nvs = sm['grasp']['n_vertices']; drive_steps = sm.get('drive_steps', 360)
    universe = am['splits']['train'] + am['splits']['val'] + am['splits']['test']
    grasps = am['history_policy']['grasps']; templates = am['history_policy']['templates']; seeds = am['history_policy']['seeds']
    specs = [dict(setting=s, grasp=g, template=t, seed=sd) for s in universe for g in grasps for t in templates for sd in seeds]
    start = time.time(); rec = []
    for gi in sorted({x['grasp'] for x in specs}):
        grp = [x for x in specs if x['grasp'] == gi]; nv = nvs[gi]; e_ell = float(ell[gi])
        for off in range(0, len(grp), batch_size):
            q = grp[off:off + batch_size]
            scene = build_scene(integ['dt'], integ['substeps'], integ['damping'], integ['angular_damping'])
            rod = add_straight_rod(scene, nv, interval, 1e7, .001, pos=(0, 0, .5)); box = add_moving_clamp(scene, (0, 0, .5))
            scene.build(n_envs=len(q), env_spacing=(2, 2))
            apply_properties(rod, np.array([ss[x['setting']]['raw_E'] for x in q]), np.array([ss[x['setting']]['mass'] for x in q]))
            attach_moving_clamp(rod, box)
            rand = [np.random.default_rng(x['seed']) for x in q]
            draw = [dict(dx=r.uniform(*bounds['clamp_start_translation_xy_m']), dy=r.uniform(*bounds['clamp_start_translation_xy_m']),
                         dur=r.uniform(*bounds['motion_duration_multiplier']), arc=r.uniform(*bounds['arc_multiplier'])) for r in rand]
            frames = {e: [] for e in range(len(q))}
            for step in range(drive_steps):
                pos = []
                for x, d in zip(q, draw):
                    t = sm['templates'][x['template']]; u = min(1., (step + 1) / (drive_steps * d['dur']))
                    s_ = u * u * (3 - 2 * u) if t['kind'] == 'ease' else u
                    pos.append((d['dx'] * (1 - s_) + (t['arc'] * d['arc'] * np.sin(np.pi * s_) if t['kind'] == 'arc' else 0), d['dy'] * (1 - s_), .5 + .2 * s_))
                box.set_pos(np.asarray(pos)); scene.step()
                if (step + 1) in FRAME_STEPS:
                    vv = vertices(rod)
                    for e in range(len(q)):
                        frames[e].append(vv[e])
            prev = vertices(rod)[:, 2:, :]; conv = False
            for _ in range(80):
                for _ in range(200):
                    scene.step()
                cur = vertices(rod)[:, 2:, :]; drift = float(np.max(np.linalg.norm(cur - prev, axis=-1))); prev = cur
                if drift < 2e-3:
                    conv = True; break
            if not conv:
                raise RuntimeError(f'addendum history batch not converged grasp={gi} off={off}')
            vv = vertices(rod)
            for e, x in enumerate(q):
                fr = frames[e] + [vv[e]]
                action = np.array([x['grasp'] / (len(grasps) - 1 if len(grasps) > 1 else 1), e_ell], float)
                rec.append((x['setting'], x['grasp'], x['template'], x['seed'],
                            temporal_shape(fr), drive_summary(sm['templates'][x['template']], x['seed'], bounds), action))
            print(json.dumps({'complete': len(rec), 'total': len(specs), 'grasp': gi}), flush=True)
    out = MAN / 'addendum_histories.npz'; f = list(zip(*rec))
    np.savez_compressed(out, setting=np.array(f[0]), grasp=np.array(f[1]), template=np.array(f[2]), seed=np.array(f[3]),
                        shape=np.array(f[4]), proprio=np.array(f[5]), action=np.array(f[6]),
                        manifest_digest=sha256(MAN / 'addendum_manifest.json'), rollout_count=len(rec), shape_dim=f[4][0].shape[0])
    print(json.dumps({'output': str(out), 'rollout_count': len(rec), 'wall_clock_s': time.time() - start}))


def _crossing(curve, ell, tau=0.5):
    y = np.asarray(curve, float); g = np.asarray(ell, float)
    for i in range(len(y) - 1):
        if (y[i] - tau) * (y[i + 1] - tau) <= 0 and y[i] != y[i + 1]:
            return float(g[i] + (tau - y[i]) / (y[i + 1] - y[i]) * (g[i + 1] - g[i]))
    return None


def run(expected_digest=None):
    am = json.loads((MAN / 'addendum_manifest.json').read_text()); live = sha256(MAN / 'addendum_manifest.json')
    if expected_digest and live != expected_digest:
        raise RuntimeError('addendum manifest digest mismatch')
    sm = json.loads((MAN / am['sweep_grid_source']).read_text()); ss = {s['id']: s for s in settings(sm)}
    ell = np.array(sm['grasp']['ell']); ng = len(ell)
    TRAIN, VAL, TEST = am['splits']['train'], am['splits']['val'], am['splits']['test']
    prop = {k: (v['B_eff'], v['w']) for k, v in ss.items()}
    raw = {k: np.log10([v[0], v[1]]) for k, v in prop.items()}; a = np.array([raw[k] for k in TRAIN]); pm, psd = a.mean(0), a.std(0); props = {k: (v - pm) / psd for k, v in raw.items()}
    sw = np.load(MAN / 'addendum_sweep_results.npz'); land = {x['id']: x for x in json.loads((MAN / 'addendum_landscape.json').read_text())['settings']}
    # labels from selection bank (winner template): success prob + J + feasible mask
    def labels(ids):
        rows = []
        for sid in ids:
            for gi in range(ng):
                ix = (sw['bank'] == 'selection') & (sw['setting'] == sid) & (sw['grasp'] == gi)
                if not ix.any():
                    continue
                cand = [(float(sw['success'][ix & (sw['template'] == t)].mean()), float(sw['J'][ix & (sw['template'] == t)].mean()), int(t)) for t in sorted(set(sw['template'][ix].tolist()))]
                sp, jj, _ = max(cand, key=lambda x: (x[0], x[1], -x[2]))
                rows.append((sid, np.array([gi / (ng - 1), float(ell[gi])]), sp, jj, 1.0 if sp > 0 else 0.0))
        return rows
    rows = labels(TRAIN + VAL); gf = np.c_[np.arange(ng) / (ng - 1), ell]
    hz = np.load(MAN / 'addendum_histories.npz'); assert str(hz['manifest_digest'].item()) == live
    S = hz['setting'].astype(str)
    def full_feat(i):    # full temporal (y,z) 112 pooled to 32 + proprio + action = 42
        return np.r_[hz['proprio'][i], pool_temporal(hz['shape'][i]), hz['action'][i]]
    def term_feat(i):    # settled-terminal-only: proprio + last 16-D frame + action = 26
        return np.r_[hz['proprio'][i], hz['shape'][i].reshape(7, 16)[-1], hz['action'][i]]
    def kframe_feat(i, k):   # first-k-frame pooled (mean over first k of 7 frames) + last-of-k + proprio + action
        fr = hz['shape'][i].reshape(7, 16)[:k]; return np.r_[hz['proprio'][i], fr.mean(0), fr[-1], hz['action'][i]]

    def map_metrics(phi, q, qb, student=None, feat_fn=None, feat=None):
        per = []
        for sid in TEST:
            if student is None:      # teacher or blind
                z = np.zeros(4) if qb is not None and student == 'blind' else (phi(torch.tensor(props[sid][None], dtype=torch.double)).detach().numpy()[0] if phi is not None else np.zeros(4))
            else:
                idx = np.where(S == sid)[0]
                with torch.no_grad():
                    z = student(torch.tensor(feat[idx], dtype=torch.double)).mean(0).numpy()
            with torch.no_grad():
                sl, _ = (qb if student == 'blind' else q)(torch.tensor(gf, dtype=torch.double), torch.tensor(np.tile(z, (ng, 1)), dtype=torch.double))
                pS = torch.sigmoid(sl).numpy()
            measS = np.array(land[sid]['success_rate'])
            pb, mb = _crossing(pS, ell), _crossing(measS, ell)
            per.append(dict(setting=sid, map_rmse=float(np.sqrt(np.mean((pS - measS) ** 2))),
                            boundary_err=(abs(pb - mb) if pb is not None and mb is not None else None)))
        rmses = [x['map_rmse'] for x in per]; bes = [x['boundary_err'] for x in per if x['boundary_err'] is not None]
        return dict(map_rmse=float(np.mean(rmses)), boundary_err=(float(np.mean(bes)) if bes else None), per_setting=per)

    def distil(phi, feat_fn=None, feat=None, seed=3403, d=None):
        X = feat if feat is not None else np.array([feat_fn(i) for i in range(len(S))])
        ti = np.isin(S, TRAIN); vi = np.isin(S, VAL); hm = X[ti].mean(0); hs = X[ti].std(0); hs[hs < 1e-12] = 1; Xn = (X - hm) / hs
        Xtr = torch.tensor(Xn[ti], dtype=torch.double); Xv = torch.tensor(Xn[vi], dtype=torch.double)
        with torch.no_grad():
            Z = phi(torch.tensor(np.array([props[s] for s in S[ti]]), dtype=torch.double)); VZ = phi(torch.tensor(np.array([props[s] for s in S[vi]]), dtype=torch.double))
        seed_all(seed); st = nn.Sequential(nn.Linear(Xn.shape[1], 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 4)).double()
        opt = torch.optim.Adam(st.parameters(), lr=1e-2); best = (float('inf'), None)
        for _ in range(300):
            opt.zero_grad(); loss = ((st(Xtr) - Z) ** 2).mean(); loss.backward(); opt.step()
            with torch.no_grad():
                vl = float(((st(Xv) - VZ) ** 2).mean())
            if vl < best[0]:
                best = (vl, {k: v.detach().clone() for k, v in st.state_dict().items()})
        st.load_state_dict(best[1]); return st, Xn

    seeds = am['training_seeds']; agg = {r: [] for r in ['teacher', 'blind', 'task_full', 'terminal']}
    trunc_by_seed = []
    for seed in seeds:
        phi, q = train_critic(rows, props, VAL, blind=False, train_ids=TRAIN, seed=seed)
        _, qb = train_critic(rows, props, VAL, blind=True, train_ids=TRAIN, seed=seed)
        Xf = np.array([full_feat(i) for i in range(len(S))]); Xt = np.array([term_feat(i) for i in range(len(S))])
        st_full, Xfn = distil(phi, feat=Xf, seed=seed); st_term, Xtn = distil(phi, feat=Xt, seed=seed)
        agg['teacher'].append(map_metrics(phi, q, None, student=None)['map_rmse'])
        agg['blind'].append(map_metrics(phi, q, qb, student='blind')['map_rmse'])
        agg['task_full'].append(map_metrics(phi, q, None, student=st_full, feat=Xfn)['map_rmse'])
        agg['terminal'].append(map_metrics(phi, q, None, student=st_term, feat=Xtn)['map_rmse'])
        # frame-truncation k=1..7, per-setting map RMSE
        tk = {}
        for k in range(1, 8):
            Xk = np.array([kframe_feat(i, k) for i in range(len(S))]); st_k, Xkn = distil(phi, feat=Xk, seed=seed)
            mm = map_metrics(phi, q, None, student=st_k, feat=Xkn)
            for x in mm['per_setting']:
                tk.setdefault(x['setting'], []).append(x['map_rmse'])
        trunc_by_seed.append(tk)
    # aggregate
    summary = {r: dict(map_rmse_mean=float(np.mean(agg[r])), map_rmse_std=float(np.std(agg[r]))) for r in agg}
    # per-setting sufficient_k (mean over seeds), timescale
    tol = am['secondary_b']['sufficient_k']['tol_k']; suff = {}
    for sid in TEST:
        rk = np.mean([trunc_by_seed[s][sid] for s in range(len(seeds))], axis=0)  # len-7
        r7 = rk[-1]; k = next((kk + 1 for kk in range(7) if rk[kk] <= r7 + tol), 7)
        B, w = prop[sid]; ell_star = None
        lam = w / 9.81; ml = _crossing(np.array(land[sid]['success_rate']), ell)  # boundary as a scale proxy
        timescale = (ml ** 2 * np.sqrt(lam / B)) if ml is not None else None
        suff[sid] = dict(rmse_by_k=[float(x) for x in rk], sufficient_k=int(k), timescale_proxy=(float(timescale) if timescale is not None else None), boundary=ml)
    task_full = summary['task_full']['map_rmse_mean']; teacher = summary['teacher']['map_rmse_mean']; blind = summary['blind']['map_rmse_mean']
    primary_pass = bool(task_full <= 1.5 * teacher and task_full < blind)
    result = dict(schema_version=1, manifest_digest=live, splits=am['splits'], training_seeds=seeds,
                  primary=dict(name=am['primary']['name'], task_only_full_temporal_map_rmse=task_full, teacher_map_rmse=teacher, blind_map_rmse=blind,
                               within_1p5x_teacher=bool(task_full <= 1.5 * teacher), better_than_blind=bool(task_full < blind), PASS=primary_pass,
                               note='PRE-REGISTERED task-only temporal student vs teacher/blind on held-out map recovery (new draws)'),
                  secondary_a_feature_contrast=dict(full_temporal_map_rmse=task_full, terminal_only_map_rmse=summary['terminal']['map_rmse_mean'],
                                                    temporal_helps=bool(summary['task_full']['map_rmse_mean'] < summary['terminal']['map_rmse_mean']),
                                                    note='matched encoder capacity; full temporal (y,z) vs settled-terminal-only'),
                  secondary_b_frame_truncation=dict(tol_k=tol, per_setting=suff, stratified='per-setting sufficient_k',
                                                    timescale_note='descriptive: sufficient_k vs ell^2*sqrt(lambda/B); NOT a pre-registered claim',
                                                    naming='frame-truncation curve (k-frame prefix), NOT an adaptation curve'),
                  summary=summary, multi_seed='seed variability reported; eval-draw contrast LABELED conditional on trained models (A-16)',
                  aggregation='map recovery over TEST; ratio pairs treated as invariance controls (A-15)')
    (MAN / 'addendum_results.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    # figures
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    labels = ['teacher', 'blind', 'task_full', 'terminal']
    ax[0].bar(labels, [summary[r]['map_rmse_mean'] for r in labels], yerr=[summary[r]['map_rmse_std'] for r in labels])
    ax[0].set_ylabel('map RMSE'); ax[0].set_title('History-variant comparison (task-only PRIMARY + feature contrast)'); ax[0].tick_params(axis='x', rotation=15)
    for sid in TEST:
        ax[1].plot(range(1, 8), suff[sid]['rmse_by_k'], 'o-', ms=3, label=sid)
    ax[1].set_xlabel('k frames'); ax[1].set_ylabel('map RMSE_k'); ax[1].set_title('Frame-truncation curve (per-setting)'); ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(ROOT / 'figures/addendum_history_variant_and_frame_truncation.png', dpi=160); plt.close(fig)
    print(json.dumps({'primary_PASS': primary_pass, 'task_full': round(task_full, 4), 'teacher': round(teacher, 4), 'blind': round(blind, 4),
                      'terminal': round(summary['terminal']['map_rmse_mean'], 4),
                      'sufficient_k': {sid: suff[sid]['sufficient_k'] for sid in TEST}}, indent=2))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--stage', choices=['histories', 'critic'], required=True); p.add_argument('--expected-digest', default=None)
    a = p.parse_args()
    if a.stage == 'histories':
        extract_histories(json.loads((MAN / 'addendum_manifest.json').read_text()))
    else:
        run(a.expected_digest)
