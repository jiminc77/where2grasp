"""Distal Part-3 interaction-history assembly (full temporal (y,z) schema + action metadata).

Replays the frozen (setting,grasp,template,seed) tuples of the distal task in Genesis, recording
the 7-frame temporal (y,z) task shape (M=8 arclength-even clamp-relative samples at drive steps
{60,120,180,240,300,360} + settled = 112-D), proprio (8-D), and EXPLICIT ACTION metadata
(grasp index + free-arm length ell = the action, per adjudication A-17). Reuses history.py's pure
primitives (shape_frame, temporal_shape, supported_wrench, FRAME_STEPS). Writes distal_histories.npz.
"""
from __future__ import annotations
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
from sim.history import shape_frame, temporal_shape, supported_wrench, drive_summary, FRAME_STEPS

ROOT = Path(__file__).resolve().parent
MAN = ROOT / 'manifests'


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def extract_real(s34_manifest=None, expected_digest=None, batch_size=50, out=None):
    s34 = Path(s34_manifest) if s34_manifest else (MAN / 'distal_s34_manifest.json')
    if expected_digest and sha256(s34) != expected_digest:
        raise RuntimeError('distal s34 manifest digest mismatch')
    cfg = json.loads(s34.read_text())
    dm = json.loads((MAN / cfg['distal_manifest']).read_text())
    from sim.scene import build_scene, add_straight_rod, add_moving_clamp, attach_moving_clamp, vertices
    from sim.material import apply_properties
    ss = {c['id']: c for c in dm['grid']}; ss.update({r['id']: r for r in dm['ratio_pairs']})
    rawE = {c['id']: c['raw_E'] for c in dm['grid']}; rawE.update({r['id']: r['raw_E'] for r in dm['ratio_pairs']})
    integ = dm['integrator']; interval = dm['interval']; g = dm['gravity']; bounds = dm['stochastic_distribution']
    ell = dm['grasp']['ell']; nvs = dm['grasp']['n_vertices']; drive_steps = dm.get('drive_steps', 360)
    universe = cfg['universe']; grasps = cfg['history_policy']['grasps']
    templates = cfg['history_policy']['templates']; seeds = cfg['history_policy']['seeds']
    specs = [dict(setting=sid, grasp=gi, template=t, seed=sd)
             for sid in universe for gi in grasps for t in templates for sd in seeds]
    start = time.time(); records = []
    for gi in sorted({x['grasp'] for x in specs}):
        group = [x for x in specs if x['grasp'] == gi]; nv = nvs[gi]; e_ell = float(ell[gi])
        for off in range(0, len(group), batch_size):
            q = group[off:off + batch_size]
            scene = build_scene(integ['dt'], integ['substeps'], integ['damping'], integ['angular_damping'])
            rod = add_straight_rod(scene, nv, interval, 1e7, .001, pos=(0, 0, .5))
            box = add_moving_clamp(scene, (0, 0, .5)); scene.build(n_envs=len(q), env_spacing=(2, 2))
            apply_properties(rod, np.array([rawE[x['setting']] for x in q]),
                             np.array([ss[x['setting']]['mass'] for x in q]))
            attach_moving_clamp(rod, box)
            rand = [np.random.default_rng(x['seed']) for x in q]
            draw = [dict(dx=r.uniform(*bounds['clamp_start_translation_xy_m']),
                         dy=r.uniform(*bounds['clamp_start_translation_xy_m']),
                         dur=r.uniform(*bounds['motion_duration_multiplier']),
                         arc=r.uniform(*bounds['arc_multiplier'])) for r in rand]
            frames = {e: [] for e in range(len(q))}
            for step in range(drive_steps):
                pos = []
                for x, d in zip(q, draw):
                    t = dm['templates'][x['template']]; u = min(1., (step + 1) / (drive_steps * d['dur']))
                    s_ = u * u * (3 - 2 * u) if t['kind'] == 'ease' else u
                    pos.append((d['dx'] * (1 - s_) + (t['arc'] * d['arc'] * np.sin(np.pi * s_) if t['kind'] == 'arc' else 0),
                                d['dy'] * (1 - s_), .5 + .2 * s_))
                box.set_pos(np.asarray(pos)); scene.step()
                if (step + 1) in FRAME_STEPS:
                    vv = vertices(rod)
                    for e in range(len(q)):
                        frames[e].append(vv[e])
            from sim.scene import settle
            ok, steps, speed = settle(scene, rod, vel_tol=2e-3, window=50, max_steps=8000)
            if not ok:
                raise RuntimeError(f'distal history batch not converged grasp={gi} off={off}: {speed}')
            vv = vertices(rod)
            for e, x in enumerate(q):
                fr = frames[e] + [vv[e]]
                s = ss[x['setting']]
                action = np.array([x['grasp'] / (len(grasps) - 1 if len(grasps) > 1 else 1), e_ell], float)  # A-17
                records.append((x['setting'], x['grasp'], x['template'], x['seed'],
                                temporal_shape(fr), drive_summary(dm['templates'][x['template']], x['seed'], bounds),
                                supported_wrench(s['mass'], nv, interval, g), action, int(steps)))
            print(json.dumps({'complete': len(records), 'total': len(specs), 'grasp': gi, 'off': off}), flush=True)
    out = Path(out) if out else (MAN / 'distal_histories.npz'); f = list(zip(*records))
    np.savez_compressed(out, setting=np.array(f[0]), grasp=np.array(f[1]), template=np.array(f[2]),
                        seed=np.array(f[3]), shape=np.array(f[4]), proprio=np.array(f[5]),
                        wrench=np.array(f[6]), action=np.array(f[7]), settle_steps=np.array(f[8]),
                        manifest_digest=sha256(s34), wall_clock_s=time.time() - start,
                        rollout_count=len(records), shape_dim=f[4][0].shape[0], action_dim=f[7][0].shape[0],
                        source='actual Genesis distal-tip temporal 7-frame (y,z) + action metadata')
    print(json.dumps({'output': str(out), 'rollout_count': len(records), 'shape_dim': int(f[4][0].shape[0]),
                      'action_dim': int(f[7][0].shape[0]), 'wall_clock_s': time.time() - start}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--manifest', default=None)
    p.add_argument('--expected-digest', default=None); p.add_argument('--batch-size', type=int, default=50)
    a = p.parse_args(); extract_real(a.manifest, a.expected_digest, a.batch_size)
