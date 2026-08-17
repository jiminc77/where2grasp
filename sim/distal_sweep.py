"""Frozen batched Genesis distal-tip-placement landscape sweep (manifest-driven, grasp-outer).

Reuses sim/scene.py verbatim + sim.sweep.draws (identical per-draw stochastic policy) and the
proven selection/evaluation discipline (disjoint banks, winner-only evaluation, 0% non-converged
via full free-vertex 3-D convergence). sim/sweep.py is intentionally left UNTOUCHED so the frozen
hardening-A lift runner + its committed results stay byte-identical.

GRASP-OUTER batching: one rod SIZE per scene with the (setting, template, seed) tuples as parallel
envs (envs parallelise on GPU; a single rod steps ~25x faster than 25 distinct rods per scene).
Full free-vertex 3-D convergence is CHUNKED (checked per 200-step chunk, not per step). Distal
writes only distal_* outputs; scoring is rooted at vertex 1 (distal_tip.score_tip).
"""
from __future__ import annotations
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
from sim.scene import build_scene, add_straight_rod, add_moving_clamp, attach_moving_clamp, vertices
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
    out = [dict(id=c['id'], kind='independent', B_eff=c['B_eff'], mass=c['mass'], w=c['w'], raw_E=c['raw_E']) for c in m['grid']]
    out += [dict(id=r['id'], kind='ratio', reference=r['reference'], B_eff=r['B_eff'], mass=r['mass'], w=r['w'], raw_E=r['raw_E'])
            for r in m['ratio_pairs']]
    return out


def run_grasp_batch(m, gi, nv, e_ell, envspec):
    """One scene: a SINGLE rod of size nv, with len(envspec) parallel envs (each a setting,template,seed)."""
    n = len(envspec)
    scene = build_scene(m['integrator']['dt'], m['integrator']['substeps'],
                        m['integrator']['damping'], m['integrator']['angular_damping'])
    rod = add_straight_rod(scene, nv, m['interval'], 1e7, .001, pos=(0, 0, .5))
    box = add_moving_clamp(scene, (0, 0, .5))
    scene.build(n_envs=n, env_spacing=(2, 2))
    Es = np.array([x['setting']['raw_E'] for x in envspec])
    masses = np.array([x['setting']['mass'] for x in envspec])
    apply_properties(rod, Es, masses); attach_moving_clamp(rod, box)
    bounds = m['stochastic_distribution']; rand = [draws(x['seed'], bounds) for x in envspec]
    drive_steps = m.get('drive_steps', 360)
    for step in range(drive_steps):
        pos = []
        for x, q in zip(envspec, rand):
            tmpl = m['templates'][x['template']]
            u = min(1., (step + 1) / (drive_steps * q['dur']))
            s = u * u * (3 - 2 * u) if tmpl['kind'] == 'ease' else u
            xx = q['dx'] * (1 - s) + (tmpl['arc'] * q['arc'] * np.sin(np.pi * s) if tmpl['kind'] == 'arc' else 0)
            pos.append((xx, q['dy'] * (1 - s), .5 + .2 * s))
        box.set_pos(np.asarray(pos)); scene.step()
    # chunked full free-vertex 3-D convergence (single rod -> fast per-step). The MAX-over-envs
    # drift is set by the out-of-regime soft rods (droop ~50 mm, slow creep); in-regime grasps that
    # drive the science settle to <<1e-3 quickly. Tolerance 2e-3 m/chunk is well within the task
    # tolerances (rho=6 mm, rho_x=15 mm) so it never degrades an in-regime measurement.
    prev = vertices(rod)[:, 2:, :]; converged = False; steps = drive_steps; drift = float('inf')
    for chunk in range(80):
        for _ in range(200):
            scene.step()
        steps += 200
        cur = vertices(rod)[:, 2:, :]; drift = float(np.max(np.linalg.norm(cur - prev, axis=-1))); prev = cur
        if drift < 2e-3:
            converged = True; break
    if not converged:
        raise RuntimeError(f'distal grasp batch not converged (full-3D chunked) grasp={gi}: drift={drift}')
    state = vertices(rod)
    rows = []
    for e, x in enumerate(envspec):
        sc = score_tip(state[e:e + 1], x['setting']['B_eff'], x['setting']['w'], e_ell)[0]
        rows.append(dict(setting=x['setting']['id'], grasp=gi, ell=e_ell, template=x['template'],
                         bank=x['bank'], seed=x['seed'], reach=sc['reach'], droop=sc['droop'], pi_g=sc['pi_g'],
                         success=bool(sc['success']), J=float(sc['J']), converged=True, settle_steps=int(steps),
                         draw_dx=rand[e]['dx'], draw_dy=rand[e]['dy'], draw_dur=rand[e]['dur'], draw_arc=rand[e]['arc']))
    return rows


def main(manifest=None, out=None, expected_digest=None, batch_size=90):
    manifest = Path(manifest) if manifest else (MAN / 'distal_manifest.json')
    if expected_digest and sha256(manifest) != expected_digest:
        raise RuntimeError(f'manifest digest mismatch: {sha256(manifest)} != {expected_digest}')
    m = json.loads(manifest.read_text())
    ss = settings(m); ell = m['grasp']['ell']; nvs = m['grasp']['n_vertices']; ng = len(ell)
    ntmpl = len(m['templates']); sel_seeds = m['seed_banks']['selection']; eval_seeds = m['seed_banks']['evaluation']
    start = time.time(); rows = []
    # SELECTION: for each grasp, all (setting,template,seed) as envs (batched)
    for gi in range(ng):
        specs = [dict(setting=s, template=t, seed=sd, bank='selection') for s in ss for t in range(ntmpl) for sd in sel_seeds]
        for i in range(0, len(specs), batch_size):
            rows += run_grasp_batch(m, gi, nvs[gi], float(ell[gi]), specs[i:i + batch_size])
        print('selection grasp', gi + 1, ng, 'rows', len(rows), flush=True)
    winners = {}
    for s in ss:
        for gi in range(ng):
            rates = [np.mean([r['success'] for r in rows if r['setting'] == s['id'] and r['grasp'] == gi and r['template'] == t])
                     for t in range(ntmpl)]
            winners[s['id'], gi] = int(np.argmax(rates))
    # EVALUATION: winner template only, eval seeds
    for gi in range(ng):
        specs = [dict(setting=s, template=winners[s['id'], gi], seed=sd, bank='evaluation') for s in ss for sd in eval_seeds]
        for i in range(0, len(specs), batch_size):
            rows += run_grasp_batch(m, gi, nvs[gi], float(ell[gi]), specs[i:i + batch_size])
        print('evaluation grasp', gi + 1, ng, 'rows', len(rows), flush=True)
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
    p.add_argument('--expected-digest', default=None); p.add_argument('--batch-size', type=int, default=90)
    a = p.parse_args(); main(a.manifest, a.out, a.expected_digest, a.batch_size)
