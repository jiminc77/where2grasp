"""Frozen batched Genesis distal-tip-placement landscape sweep (manifest-driven).

Reuses sim/scene.py verbatim + sim.sweep.draws (identical per-draw stochastic policy) and the
proven selection/evaluation discipline (disjoint banks, winner-only evaluation, 0% non-converged
via full free-vertex 3-D settle). sim/sweep.py is intentionally left UNTOUCHED so the frozen
hardening-A lift runner + its committed results stay byte-identical (safety over a shared-core
refactor of the lift path); de-duplication is via scene reuse + sweep.draws. Distal writes only
distal_* outputs and never touches hard_* artifacts. Scoring is rooted at vertex 1 (distal_tip).
"""
from __future__ import annotations
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
from sim.scene import build_scene, add_straight_rod, add_moving_clamp, attach_moving_clamp, settle, vertices
from sim.material import apply_properties
from sim.sweep import draws
from sim.tasks.distal_tip import score_tip

ROOT = Path(__file__).resolve().parent
MAN = ROOT / 'manifests'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def settings(m):
    """Distal grid cells + ratio pairs, each {id, B_eff, mass, w}."""
    out = [dict(id=c['id'], kind='independent', B_eff=c['B_eff'], mass=c['mass'], w=c['w']) for c in m['grid']]
    out += [dict(id=r['id'], kind='ratio', reference=r['reference'], B_eff=r['B_eff'], mass=r['mass'], w=r['w'])
            for r in m['ratio_pairs']]
    return out


def run_batch(m, envspec):
    ell = m['grasp']['ell']; nvs = m['grasp']['n_vertices']; ng = len(ell)
    n = len(envspec)
    scene = build_scene(m['integrator']['dt'], m['integrator']['substeps'],
                        m['integrator']['damping'], m['integrator']['angular_damping'])
    rods = []; boxes = []
    for k, nv in enumerate(nvs):
        y = .18 * k
        rods.append(add_straight_rod(scene, nv, m['interval'], 1e7, .001, pos=(0, y, .5)))
        boxes.append(add_moving_clamp(scene, (0, y, .5)))
    scene.build(n_envs=n, env_spacing=(2, 2))
    Es = np.array([x['setting']['B_eff_rawE'] for x in envspec])
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
            Q = P.copy(); Q[:, 1] += .18 * k; box.set_pos(Q)
        scene.step()
    # full free-vertex 3-D convergence (hard-fail if any rod not settled)
    ok, steps, speed = settle(scene, rods, vel_tol=2e-3, window=50, max_steps=8000)
    if not ok:
        raise RuntimeError(f'distal sweep batch not converged (full-3D): steps={steps} speed={speed}')
    rows = []
    for gi, (rod, e_ell) in enumerate(zip(rods, ell)):
        state = vertices(rod)
        for e, x in enumerate(envspec):
            sc = score_tip(state[e:e + 1], x['setting']['B_eff'], x['setting']['w'], e_ell)[0]
            rows.append(dict(setting=x['setting']['id'], grasp=gi, ell=e_ell, template=x['template'],
                             bank=x['bank'], seed=x['seed'], reach=sc['reach'], droop=sc['droop'],
                             pi_g=sc['pi_g'], success=bool(sc['success']), J=float(sc['J']),
                             converged=True, settle_steps=int(steps),
                             draw_dx=rand[e]['dx'], draw_dy=rand[e]['dy'], draw_dur=rand[e]['dur'], draw_arc=rand[e]['arc']))
    return rows


def main(manifest=None, out=None, expected_digest=None, batch_size=50):
    manifest = Path(manifest) if manifest else (MAN / 'distal_manifest.json')
    if expected_digest and sha256(manifest) != expected_digest:
        raise RuntimeError(f'manifest digest mismatch: {sha256(manifest)} != {expected_digest}')
    m = json.loads(manifest.read_text())
    # attach the raw-E knob per setting (bending_stiffness setter takes raw E; B_eff is the calibrated value)
    rawE = {c['id']: c['raw_E'] for c in m['grid']}; rawE.update({r['id']: r['raw_E'] for r in m['ratio_pairs']})
    ss = settings(m)
    for s in ss:
        s['B_eff_rawE'] = rawE[s['id']]
    ng = len(m['grasp']['ell'])
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
    out = Path(out) if out else (MAN / 'distal_sweep_results.npz')
    keys = ['setting', 'grasp', 'ell', 'template', 'bank', 'seed', 'reach', 'droop', 'pi_g', 'success', 'J',
            'selected_template', 'converged', 'settle_steps', 'draw_dx', 'draw_dy', 'draw_dur', 'draw_arc']
    np.savez_compressed(out, **{k: np.array([r[k] for r in rows]) for k in keys},
                        wall_clock_s=time.time() - start, manifest_digest=sha256(manifest))
    print(json.dumps({'physical_rollouts': len(rows), 'wall_clock_s': time.time() - start,
                      'nonconverged_fraction': float(np.mean([not r['converged'] for r in rows])),
                      'manifest_digest': sha256(manifest), 'output': str(out)}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', default=None); p.add_argument('--out', default=None)
    p.add_argument('--expected-digest', default=None); p.add_argument('--batch-size', type=int, default=50)
    a = p.parse_args(); main(a.manifest, a.out, a.expected_digest, a.batch_size)
