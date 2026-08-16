"""Frozen batched Genesis grasp-landscape sweep (manifest-driven grid + cardinality).

Hardening note: the grid (property values AND stable setting IDs) and every loop
cardinality are read from the manifest, never hardcoded, so a reduced in-regime
4x3 grid with stable IDs B1..B4 x w0..w2 runs without renumbering the survivors.
The stochastic estimand applies ONLY the clamp-start translation + duration/arc
draws: no verified batched per-env rod-pose setter exists in Genesis c5026a9
(set_position is a no-op w.r.t. get_vertices_pos), so the registered
initial-rod-pose perturbation is dropped and the manifest declares only what runs.
"""
from __future__ import annotations
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
from sim.scene import build_scene, add_straight_rod, add_moving_clamp, attach_moving_clamp, vertices
from sim.material import apply_properties
ROOT = Path(__file__).resolve().parent
DEFAULT_MAN = ROOT / 'manifests/sweep_manifest.json'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def settings(m):
    """Return the list of setting dicts. Prefers an explicit stable `grid`
    (hardening manifests: each cell carries its own id/bi/wi); falls back to the
    legacy positional B{i}_w{j} enumeration for the original 5x4 manifest."""
    out = []
    if 'grid' in m:
        for c in m['grid']:
            out.append(dict(id=c['id'], kind='independent', bi=c['bi'], wi=c['wi'],
                            raw_E=c['raw_E'], B_eff=c['B_eff'], mass=c['mass'], w=c['w']))
    else:
        for i, (E, B) in enumerate(zip(m['raw_E'], m['B_eff'])):
            for j, (mass, w) in enumerate(zip(m['segment_masses'], m['w'])):
                out.append(dict(id=f'B{i}_w{j}', kind='independent', bi=i, wi=j,
                                raw_E=E, B_eff=B, mass=mass, w=w))
    for p in m['ratio_pairs']:
        q = p['scaled']; E = q['raw_E']; mass = q['segment_mass']
        # B_eff for the scaled point via the calibrated raw_E->B_eff map (grid values).
        if 'grid' in m:
            xs = sorted({(c['raw_E'], c['B_eff']) for c in m['grid']})
            B = float(np.interp(E, [x[0] for x in xs], [x[1] for x in xs]))
        else:
            B = float(np.interp(E, m['raw_E'], m['B_eff']))
        out.append(dict(id=f'R{p["pair_id"]}', kind='ratio', pair_id=p['pair_id'],
                        raw_E=E, B_eff=B, mass=mass, w=mass * m['gravity'] / m['interval']))
    return out


def draws(seed, bounds):
    """Per-draw stochasticity: clamp-start xy translation + duration/arc multipliers.
    (No initial-rod-pose translation: no verified batched per-env rod setter exists.)"""
    r = np.random.default_rng(seed)
    cx = bounds['clamp_start_translation_xy_m']
    dm = bounds['motion_duration_multiplier']
    am = bounds['arc_multiplier']
    return dict(dx=r.uniform(cx[0], cx[1]), dy=r.uniform(cx[0], cx[1]),
                dur=r.uniform(dm[0], dm[1]), arc=r.uniform(am[0], am[1]))


def run_batch(m, envspec):
    ell = m['grasp']['ell']; nvs = m['grasp']['n_vertices']; ng = len(ell)
    n = len(envspec)
    scene = build_scene(m['integrator']['dt'], m['integrator']['substeps'],
                        m['integrator']['damping'], m['integrator']['angular_damping'])
    rods = []; boxes = []
    for k, nv in enumerate(nvs):
        y = .12 * k
        rods.append(add_straight_rod(scene, nv, m['interval'], 1e7, .001, pos=(0, y, .5)))
        boxes.append(add_moving_clamp(scene, (0, y, .5)))
    scene.build(n_envs=n, env_spacing=(2, 2))
    Es = np.array([x['setting']['raw_E'] for x in envspec])
    masses = np.array([x['setting']['mass'] for x in envspec])
    for rod, box in zip(rods, boxes):
        apply_properties(rod, Es, masses); attach_moving_clamp(rod, box)
    bounds = m['stochastic_distribution']
    rand = [draws(x['seed'], bounds) for x in envspec]
    drive_steps = m.get('drive_steps', 360)
    for step in range(drive_steps):
        pos = []
        for x, q in zip(envspec, rand):
            tmpl = m['templates'][x['template']]
            u = min(1., (step + 1) / (drive_steps * q['dur']))
            s = u * u * (3 - 2 * u) if tmpl['kind'] == 'ease' else u
            xx = q['dx'] * (1 - s) + (tmpl['arc'] * q['arc'] * np.sin(np.pi * s) if tmpl['kind'] == 'arc' else 0)
            yy = q['dy'] * (1 - s)
            pos.append((xx, yy, .5 + .2 * s))
        P = np.asarray(pos)
        for k, box in enumerate(boxes):
            Q = P.copy(); Q[:, 1] += .12 * k; box.set_pos(Q)
        scene.step()
    prev = [vertices(r)[:, -1, 2] for r in rods]; stable = False
    for chunk in range(100):
        for _ in range(200):
            scene.step()
        cur = [vertices(r)[:, -1, 2] for r in rods]
        drift = max(float(np.max(np.abs(a - b))) for a, b in zip(cur, prev)); prev = cur
        if drift < 1e-3:
            stable = True; break
    rows = []
    for gi, (rod, e_ell) in enumerate(zip(rods, ell)):
        v = vertices(rod); clamp = v[:, :2, 2].mean(1); delta = clamp - v[:, -1, 2]
        for e, x in enumerate(envspec):
            rows.append(dict(setting=x['setting']['id'], grasp=gi, ell=e_ell, template=x['template'],
                             bank=x['bank'], seed=x['seed'], J=float(m['h'] - delta[e]),
                             success=bool(delta[e] <= m['h']), converged=stable,
                             last_chunk_tip_drift=float(drift),
                             draw_dx=rand[e]['dx'], draw_dy=rand[e]['dy'],
                             draw_dur=rand[e]['dur'], draw_arc=rand[e]['arc']))
    return rows


def _assert_manifest(m):
    ell = np.asarray(m['grasp']['ell'], float); nv = np.asarray(m['grasp']['n_vertices'])
    assert len(ell) == len(nv), (len(ell), len(nv))
    assert np.all(np.diff(ell) > 0), 'ell must be strictly increasing'
    step = np.diff(ell)
    assert np.allclose(step, step[0], atol=1e-9), 'ell must be uniform (one grid step rule)'
    b = m['stochastic_distribution']
    for k in ('clamp_start_translation_xy_m', 'motion_duration_multiplier', 'arc_multiplier'):
        assert k in b, f'missing stochastic bound {k}'
    assert 'initial_rod_pose_translation_xy_m' not in b, 'rod-pose perturbation must be absent (unimplementable)'


def main(manifest=DEFAULT_MAN, out=None, expected_digest=None, batch_size=64):
    manifest = Path(manifest)
    if expected_digest and sha256(manifest) != expected_digest:
        raise RuntimeError(f'manifest digest mismatch: {sha256(manifest)} != {expected_digest}')
    m = json.loads(manifest.read_text()); _assert_manifest(m)
    ss = settings(m); ng = len(m['grasp']['ell'])
    sel_seeds = m['seed_banks']['selection']; eval_seeds = m['seed_banks']['evaluation']
    start = time.time(); rows = []
    selection = [dict(setting=s, template=t, seed=seed, bank='selection')
                 for s in ss for t in range(len(m['templates'])) for seed in sel_seeds]
    for i in range(0, len(selection), batch_size):
        rows += run_batch(m, selection[i:i + batch_size]); print('selection', min(i + batch_size, len(selection)), len(selection), flush=True)
    winners = {}
    for s in ss:
        for g in range(ng):
            rates = [np.mean([r['success'] for r in rows if r['setting'] == s['id'] and r['grasp'] == g and r['template'] == t])
                     for t in range(len(m['templates']))]
            winners[s['id'], g] = int(np.argmax(rates))
    evaluation = [dict(setting=s, template=t, seed=seed, bank='evaluation')
                  for s in ss for t in range(len(m['templates'])) for seed in eval_seeds
                  if any(winners[s['id'], g] == t for g in range(ng))]
    evalrows = []
    for i in range(0, len(evaluation), batch_size):
        evalrows += run_batch(m, evaluation[i:i + batch_size]); print('evaluation', min(i + batch_size, len(evaluation)), len(evaluation), flush=True)
    evalrows = [r for r in evalrows if winners[r['setting'], r['grasp']] == r['template']]; rows += evalrows
    for r in rows:
        r['selected_template'] = bool(winners[r['setting'], r['grasp']] == r['template'])
    out = Path(out) if out else ROOT / 'manifests/hard_sweep_results.npz'
    keys = ['setting', 'grasp', 'ell', 'template', 'bank', 'seed', 'J', 'success', 'selected_template',
            'converged', 'last_chunk_tip_drift', 'draw_dx', 'draw_dy', 'draw_dur', 'draw_arc']
    np.savez_compressed(out, **{k: np.array([r[k] for r in rows]) for k in keys},
                        wall_clock_s=time.time() - start, manifest_digest=sha256(manifest))
    print(json.dumps({'physical_rollouts': len(rows), 'wall_clock_s': time.time() - start,
                      'nonconverged_fraction': float(np.mean([not r['converged'] for r in rows])),
                      'manifest_digest': sha256(manifest), 'output': str(out)}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', default=str(DEFAULT_MAN))
    p.add_argument('--out', default=None)
    p.add_argument('--expected-digest', default=None)
    p.add_argument('--batch-size', type=int, default=64)
    a = p.parse_args()
    main(a.manifest, a.out, a.expected_digest, a.batch_size)
