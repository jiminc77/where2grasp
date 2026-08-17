"""Addendum lift-and-clear landscape sweep on NEW seeds (grasp-outer; the clean re-run map target).

Reuses sim/scene.py + the frozen lift-and-clear scoring (delta = clamp_z - tip_z; success = delta<=h;
J = h - delta) on the hard_sweep_manifest grid but with the addendum_manifest's NEW seed banks.
Grasp-outer batching (one rod size per scene) + chunked full-3D convergence, for speed. sweep.py is
left UNTOUCHED (byte-frozen). Writes addendum_sweep_results.npz + addendum_landscape.json.
"""
from __future__ import annotations
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
from sim.scene import build_scene, add_straight_rod, add_moving_clamp, attach_moving_clamp, vertices
from sim.material import apply_properties
from sim.sweep import draws, settings

ROOT = Path(__file__).resolve().parent
MAN = ROOT / 'manifests'


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def run_grasp_batch(m, gi, nv, e_ell, envspec, h):
    n = len(envspec)
    scene = build_scene(m['integrator']['dt'], m['integrator']['substeps'], m['integrator']['damping'], m['integrator']['angular_damping'])
    rod = add_straight_rod(scene, nv, m['interval'], 1e7, .001, pos=(0, 0, .5)); box = add_moving_clamp(scene, (0, 0, .5))
    scene.build(n_envs=n, env_spacing=(2, 2))
    apply_properties(rod, np.array([x['setting']['raw_E'] for x in envspec]), np.array([x['setting']['mass'] for x in envspec]))
    attach_moving_clamp(rod, box)
    bounds = m['stochastic_distribution']; rand = [draws(x['seed'], bounds) for x in envspec]
    drive_steps = m.get('drive_steps', 360)
    for step in range(drive_steps):
        pos = []
        for x, q in zip(envspec, rand):
            tmpl = m['templates'][x['template']]; u = min(1., (step + 1) / (drive_steps * q['dur']))
            s = u * u * (3 - 2 * u) if tmpl['kind'] == 'ease' else u
            xx = q['dx'] * (1 - s) + (tmpl['arc'] * q['arc'] * np.sin(np.pi * s) if tmpl['kind'] == 'arc' else 0)
            pos.append((xx, q['dy'] * (1 - s), .5 + .2 * s))
        box.set_pos(np.asarray(pos)); scene.step()
    prev = vertices(rod)[:, 2:, :]; converged = False; steps = drive_steps; drift = float('inf')
    for chunk in range(80):
        for _ in range(200):
            scene.step()
        steps += 200
        cur = vertices(rod)[:, 2:, :]; drift = float(np.max(np.linalg.norm(cur - prev, axis=-1))); prev = cur
        if drift < 2e-3:
            converged = True; break
    if not converged:
        raise RuntimeError(f'addendum lift sweep not converged grasp={gi}: drift={drift}')
    v = vertices(rod); clamp = v[:, :2, 2].mean(1); delta = clamp - v[:, -1, 2]
    rows = []
    for e, x in enumerate(envspec):
        rows.append(dict(setting=x['setting']['id'], grasp=gi, ell=e_ell, template=x['template'], bank=x['bank'],
                         seed=x['seed'], J=float(h - delta[e]), success=bool(delta[e] <= h), delta_tip=float(delta[e]),
                         converged=True, settle_steps=int(steps)))
    return rows


def main(out=None, expected_digest=None, batch_size=90):
    am = json.loads((MAN / 'addendum_manifest.json').read_text())
    if expected_digest and sha256(MAN / 'addendum_manifest.json') != expected_digest:
        raise RuntimeError('addendum manifest digest mismatch')
    m = json.loads((MAN / am['sweep_grid_source']).read_text())
    ss = settings(m); ell = m['grasp']['ell']; nvs = m['grasp']['n_vertices']; ng = len(ell); h = m['h']; ntmpl = len(m['templates'])
    sel_seeds = am['seed_banks']['selection']; eval_seeds = am['seed_banks']['evaluation']
    start = time.time(); rows = []
    for gi in range(ng):
        specs = [dict(setting=s, template=t, seed=sd, bank='selection') for s in ss for t in range(ntmpl) for sd in sel_seeds]
        for i in range(0, len(specs), batch_size):
            rows += run_grasp_batch(m, gi, nvs[gi], float(ell[gi]), specs[i:i + batch_size], h)
        print('addendum selection grasp', gi + 1, ng, 'rows', len(rows), flush=True)
    winners = {}
    for s in ss:
        for gi in range(ng):
            rates = [np.mean([r['success'] for r in rows if r['setting'] == s['id'] and r['grasp'] == gi and r['template'] == t]) for t in range(ntmpl)]
            winners[s['id'], gi] = int(np.argmax(rates))
    for gi in range(ng):
        specs = [dict(setting=s, template=winners[s['id'], gi], seed=sd, bank='evaluation') for s in ss for sd in eval_seeds]
        for i in range(0, len(specs), batch_size):
            rows += run_grasp_batch(m, gi, nvs[gi], float(ell[gi]), specs[i:i + batch_size], h)
        print('addendum evaluation grasp', gi + 1, ng, 'rows', len(rows), flush=True)
    for r in rows:
        r['selected_template'] = bool(winners[r['setting'], r['grasp']] == r['template'])
    out = Path(out) if out else (MAN / 'addendum_sweep_results.npz')
    keys = ['setting', 'grasp', 'ell', 'template', 'bank', 'seed', 'J', 'success', 'delta_tip', 'selected_template', 'converged', 'settle_steps']
    np.savez_compressed(out, **{k: np.array([r[k] for r in rows]) for k in keys}, wall_clock_s=time.time() - start,
                        manifest_digest=sha256(MAN / 'addendum_manifest.json'))
    # measured evaluation-bank landscape per setting
    land = []
    for s in ss:
        rate = [float(np.mean([r['success'] for r in rows if r['setting'] == s['id'] and r['grasp'] == gi and r['bank'] == 'evaluation'])) for gi in range(ng)]
        land.append(dict(id=s['id'], B_eff=s['B_eff'], w=s['w'], success_rate=rate, ell_grid=[float(x) for x in ell], tau=float(m['tau'])))
    (MAN / 'addendum_landscape.json').write_text(json.dumps({'settings': land}, indent=2) + '\n')
    print(json.dumps({'physical_rollouts': len(rows), 'wall_clock_s': time.time() - start,
                      'nonconverged_fraction': float(np.mean([not r['converged'] for r in rows])), 'output': str(out)}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--out', default=None); p.add_argument('--expected-digest', default=None); p.add_argument('--batch-size', type=int, default=90)
    a = p.parse_args(); main(a.out, a.expected_digest, a.batch_size)
