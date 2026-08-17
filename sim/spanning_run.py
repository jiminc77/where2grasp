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


def histories(expected_digest=None):
    from sim.distal_history import extract_real
    extract_real(str(MAN / 'spanning_manifest.json'), expected_digest, out=str(MAN / 'spanning_histories.npz'))


def _render_new_draw_mp4(cfg):
    """A distal rollout from the NEW spanning draws (owner viz mandate) — WIN cell B2_w0 at its optimum."""
    import imageio.v2 as iio, genesis as gs
    from sim.scene import build_scene, add_straight_rod, add_moving_clamp, attach_moving_clamp
    from sim.material import apply_properties
    from sim.sweep import draws
    cell = next(c for c in cfg['grid'] if c['id'] == 'B2_w0'); grid = np.array(cfg['grasp']['ell'])
    ell = float(grid[cell['argmax_idx']]); nv = int(round(ell / cfg['interval'])) + 2; seed = NEW_HIST[0]
    q = draws(seed, cfg['stochastic_distribution'])
    scene = build_scene(cfg['integrator']['dt'], cfg['integrator']['substeps'], cfg['integrator']['damping'], cfg['integrator']['angular_damping'])
    surface = gs.surfaces.Default(diffuse_texture=gs.textures.ImageTexture(image_path='dlo-lab/textures/rope01.png'), vis_mode='recon')
    rod = add_straight_rod(scene, nv, cfg['interval'], 1e7, .001, pos=(0, 0, .5), surface=surface); box = add_moving_clamp(scene, (0, 0, .5))
    cam = scene.add_camera(res=(480, 480), pos=(.35, -.9, .55), lookat=(.25, 0, .45), fov=42, GUI=False)
    scene.build(n_envs=1); apply_properties(rod, np.array([cell['raw_E']]), np.array([cell['mass']])); attach_moving_clamp(rod, box)
    movie = []
    for i in range(480):
        u = min(1., (i + 1) / (360 * q['dur'])); s = u * u * (3 - 2 * u)
        box.set_pos(np.array([[q['dx'] * (1 - s), q['dy'] * (1 - s), .5 + .2 * s]])); scene.step()
        if i % 12 == 0:
            movie.append(np.asarray(cam.render()[0])[..., :3])
    path = ROOT / 'figures' / 'spanning_new_draw_rollout.mp4'; iio.mimwrite(path, movie, fps=15, codec='libx264')
    reader = iio.get_reader(path); dec = reader.count_frames(); mid = np.asarray(reader.get_data(dec // 2)); reader.close()
    assert dec == len(movie) and float(mid.std()) > 3
    print(json.dumps({'mp4': str(path), 'frames': dec, 'std': float(mid.std()), 'cell': 'B2_w0', 'ell': ell, 'new_seed': int(seed)}), flush=True)


def run(expected_digest=None):
    import torch
    from torch import nn
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sim.identify import pool_temporal
    from sim.distal_critic import Phi, TwoHead, train_critic, seed_all
    cfg = json.loads((MAN / 'spanning_manifest.json').read_text()); live = sha256(MAN / 'spanning_manifest.json')
    if expected_digest and live != expected_digest:
        raise RuntimeError('spanning manifest digest mismatch')
    grid = np.array(cfg['grasp']['ell']); ng = len(grid)
    prop = {c['id']: (c['B_eff'], c['w']) for c in cfg['grid']}; prop.update({r['id']: (r['B_eff'], r['w']) for r in cfg['ratio_pairs']})
    raw = {k: np.log10([v[0], v[1]]) for k, v in prop.items()}; a = np.array([raw[k] for k in TRAIN]); pm, psd = a.mean(0), a.std(0); props = {k: (v - pm) / psd for k, v in raw.items()}
    sw = np.load(MAN / 'spanning_sweep_results.npz'); assert str(sw['manifest_digest'].item()) == live
    def meas(sid):
        ms = np.array([float(np.mean(sw['success'][(sw['bank'] == 'evaluation') & (sw['setting'] == sid) & (sw['grasp'] == gi)])) for gi in range(ng)])
        qm = (sw['bank'] == 'evaluation') & (sw['setting'] == sid)
        mj = np.array([float(np.mean(sw['J'][qm & (sw['grasp'] == gi)])) if (qm & (sw['grasp'] == gi)).any() else tm.J_INF for gi in range(ng)])
        return ms, mj
    rows = []
    for sid in TRAIN + VAL:
        for gi in range(ng):
            ix = (sw['bank'] == 'selection') & (sw['setting'] == sid) & (sw['grasp'] == gi)
            if not ix.any():
                continue
            cand = [(float(sw['success'][ix & (sw['template'] == t)].mean()), float(sw['J'][ix & (sw['template'] == t)].mean()), int(t)) for t in sorted(set(sw['template'][ix].tolist()))]
            sp, jj, _ = max(cand, key=lambda x: (x[0], x[1], -x[2]))
            rows.append((sid, np.array([gi / (ng - 1), float(grid[gi])]), sp, jj, 1.0 if sp > 0 else 0.0))
    gf = np.c_[np.arange(ng) / (ng - 1), grid]
    hz = np.load(MAN / 'spanning_histories.npz'); assert str(hz['manifest_digest'].item()) == live
    S = hz['setting'].astype(str)
    Xt = np.array([np.r_[hz['proprio'][i], pool_temporal(hz['shape'][i]), hz['action'][i]] for i in range(len(S))])
    agg = {r: dict(regret=[], map=[]) for r in ['teacher', 'blind', 'task_student']}
    ratio_inv = {r['id']: [] for r in cfg['ratio_pairs']}
    for seed in cfg['training_seeds']:
        phi, q = train_critic(rows, props, VAL, blind=False, train_ids=TRAIN, seed=seed)
        _, qb = train_critic(rows, props, VAL, blind=True, train_ids=TRAIN, seed=seed)
        ti = np.isin(S, TRAIN); vi = np.isin(S, VAL); hm = Xt[ti].mean(0); hs = Xt[ti].std(0); hs[hs < 1e-12] = 1; Xn = (Xt - hm) / hs
        with torch.no_grad():
            Z = phi(torch.tensor(np.array([props[s] for s in S[ti]]), dtype=torch.double)); VZ = phi(torch.tensor(np.array([props[s] for s in S[vi]]), dtype=torch.double))
        seed_all(seed); st = nn.Sequential(nn.Linear(Xn.shape[1], 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 4)).double()
        opt = torch.optim.Adam(st.parameters(), lr=1e-2); best = (float('inf'), None)
        for _ in range(300):
            opt.zero_grad(); loss = ((st(torch.tensor(Xn[ti], dtype=torch.double)) - Z) ** 2).mean(); loss.backward(); opt.step()
            with torch.no_grad():
                vl = float(((st(torch.tensor(Xn[vi], dtype=torch.double)) - VZ) ** 2).mean())
            if vl < best[0]:
                best = (vl, {k: v.detach().clone() for k, v in st.state_dict().items()})
        st.load_state_dict(best[1])
        def zcurve(kind, sid):
            if kind == 'teacher':
                z = phi(torch.tensor(props[sid][None], dtype=torch.double)).detach().numpy()[0]; qq = q
            elif kind == 'blind':
                z = np.zeros(4); qq = qb
            else:
                idx = np.where(S == sid)[0]
                with torch.no_grad():
                    z = st(torch.tensor(Xn[idx], dtype=torch.double)).mean(0).numpy()
                qq = q
            with torch.no_grad():
                sl, jv = qq(torch.tensor(gf, dtype=torch.double), torch.tensor(np.tile(z, (ng, 1)), dtype=torch.double))
                return torch.sigmoid(sl).numpy(), jv.numpy()
        for kind in ['teacher', 'blind', 'task_student']:
            uniq = {}; rmses = []
            for sid in TEST:
                ms, mj = meas(sid); B, w = prop[sid]; in_reg = tm.pi_g(grid, B, w) <= tm.PI_G_MAX
                pS, pJ = zcurve(kind, sid)
                cand = np.where(in_reg & (pS >= 0.5))[0]
                sel = int(cand[np.argmax(pJ[cand])]) if len(cand) else (int(np.where(in_reg)[0][np.argmax(pS[in_reg])]) if in_reg.any() else int(np.argmax(pS)))
                best_meas = float(np.max(mj[in_reg])) if in_reg.any() else float(np.max(mj))
                uniq.setdefault(tuple(np.round(np.log10(prop[sid]), 6)), []).append(best_meas - mj[sel])
                rmses.append(float(np.sqrt(np.mean((pS - ms) ** 2))))
            agg[kind]['regret'].append(float(np.mean([np.mean(v) for v in uniq.values()]))); agg[kind]['map'].append(float(np.mean(rmses)))
        for r in cfg['ratio_pairs']:
            def margmax(sid):
                _, mj = meas(sid); B, w = prop[sid]; inr = tm.pi_g(grid, B, w) <= tm.PI_G_MAX
                return int(np.where(inr)[0][np.argmax(mj[inr])]) if inr.any() else None
            ai, ar = margmax(r['id']), margmax(r['reference'])
            if ai is not None and ar is not None:
                ratio_inv[r['id']].append(abs(ai - ar))
    summary = {k: dict(regret_mean=float(np.mean(agg[k]['regret'])), regret_std=float(np.std(agg[k]['regret'])),
                       map_rmse_mean=float(np.mean(agg[k]['map'])), map_rmse_std=float(np.std(agg[k]['map']))) for k in agg}
    discriminates = bool(summary['teacher']['regret_mean'] < summary['blind']['regret_mean'] and summary['task_student']['regret_mean'] < summary['blind']['regret_mean'])
    map_helps = bool(summary['teacher']['map_rmse_mean'] < summary['blind']['map_rmse_mean'] and summary['task_student']['map_rmse_mean'] < summary['blind']['map_rmse_mean'])
    result = dict(schema_version=1, manifest_digest=live, splits=dict(train=TRAIN, val=VAL, test=TEST), training_seeds=cfg['training_seeds'],
                  primary_selection_regret={k: dict(mean=summary[k]['regret_mean'], std=summary[k]['regret_std']) for k in agg},
                  co_primary_map={k: dict(map_rmse=summary[k]['map_rmse_mean'], std=summary[k]['map_rmse_std']) for k in agg},
                  discriminates=discriminates, map_helps=map_helps,
                  ratio_invariance={rid: dict(mean_offset_cells=(float(np.mean(v)) if v else None), invariant=bool(v and max(v) <= 1)) for rid, v in ratio_inv.items()},
                  test_cohort=cfg['test_cohort'],
                  note='NEW-seed spanning re-test; well-separated TEST bands; selection-regret PRIMARY + map co-primary; ratio pairs = invariance controls; multi-seed variability (A-16)')
    (MAN / 'spanning_results.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    fig, ax = plt.subplots(1, 2, figsize=(11, 4)); L = ['teacher', 'blind', 'task_student']
    ax[0].bar(L, [summary[k]['regret_mean'] for k in L], yerr=[summary[k]['regret_std'] for k in L]); ax[0].set_ylabel('selection regret'); ax[0].set_title('Spanning-cohort selection regret (PRIMARY)'); ax[0].tick_params(axis='x', rotation=15)
    ax[1].bar(L, [summary[k]['map_rmse_mean'] for k in L], yerr=[summary[k]['map_rmse_std'] for k in L]); ax[1].set_ylabel('map RMSE'); ax[1].set_title('Map recovery (co-primary)'); ax[1].tick_params(axis='x', rotation=15)
    fig.tight_layout(); fig.savefig(ROOT / 'figures/spanning_regret_and_map.png', dpi=160); plt.close(fig)
    try:
        _render_new_draw_mp4(cfg)
    except Exception as e:
        print('mp4 skipped:', e, flush=True)
    print(json.dumps({'discriminates': discriminates, 'map_helps': map_helps,
                      'regret': {k: round(summary[k]['regret_mean'], 4) for k in agg},
                      'map_rmse': {k: round(summary[k]['map_rmse_mean'], 4) for k in agg}}, indent=2))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--stage', choices=['freeze', 'histories', 'run'], default='freeze')
    p.add_argument('--expected-digest', default=None); a = p.parse_args()
    if a.stage == 'freeze':
        freeze()
    elif a.stage == 'histories':
        histories(a.expected_digest)
    else:
        run(a.expected_digest)
