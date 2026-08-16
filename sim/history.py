"""Part-2 (hardening) history assembly: FULL temporal (y,z) schema + probe primitive.

Fixes the fast-loop terminal-z-summary limitation. The task shape is now M=8
arclength-even clamp-relative (y,z) samples recorded at 7 frames
{drive 60,120,180,240,300,360, settled} = 112-D. A small-deflection PROBE
contributes one settled 16-D frame (probe-enriched shape = probe frame prepended
to the 7 task frames = 128-D). The extractor is label-blind: it emits only
measured shape/proprio/wrench; setting identifiers are provenance only.
"""
from __future__ import annotations
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
FRAME_STEPS = [60, 120, 180, 240, 300, 360]      # drive-step checkpoints; + settled = 7 frames
M = 8


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def supported_wrench(segment_mass, n_vertices, interval=0.01, g=9.81):
    """Ideal supported-weight wrench: supported weight / free length (normalized Fz)."""
    n_free = n_vertices - 2
    ell = n_free * interval
    return segment_mass * n_free * g / ell


def shape_frame(vertices, m=M):
    """M arclength-even clamp-relative (y,z) samples of the settled free arm -> 2*m-D."""
    v = np.asarray(vertices, float)
    if v.ndim != 2 or v.shape[1] != 3 or len(v) < 3 or not np.isfinite(v).all():
        raise ValueError('invalid vertices')
    q = v[2:] - v[:2].mean(0)
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))]
    if arc[-1] <= 0:
        raise ValueError('zero free-arm arclength')
    at = np.linspace(0, arc[-1], m)
    return np.c_[np.interp(at, arc, q[:, 1]), np.interp(at, arc, q[:, 2])].ravel()


def temporal_shape(frames):
    """Concatenate per-frame (y,z) samples over the 7 frames -> 2*m*7-D (=112 for m=8)."""
    return np.concatenate([shape_frame(f) for f in frames])


def drive_summary(template, seed, bounds):
    """Terminal pose + frozen drive parameters; no property metadata (proprio, 8-D)."""
    r = np.random.default_rng(seed)
    cx = bounds['clamp_start_translation_xy_m']; dm = bounds['motion_duration_multiplier']; am = bounds['arc_multiplier']
    dx = r.uniform(cx[0], cx[1]); dy = r.uniform(cx[0], cx[1]); dur = r.uniform(dm[0], dm[1]); arc = r.uniform(am[0], am[1])
    return np.array([0., 0., .7, template['arc'], dx, dy, dur, arc], float)


def probe_shape(scene_build, add_rod, apply_props, vertices_fn, B_eff_rawE, segment_mass,
                ell_probe, integrator, interval=0.01):
    """Prescribed small-deflection clamped-cantilever settle on the setting's OWN material.
    Returns the settled 16-D (y,z) frame. Reuses the Step-1 calibration machinery: fixed
    verts [0,1], gravity settle, chunked convergence. Label-blind (uses material params only
    to build the scene; the returned feature is measured shape)."""
    nv = int(round(ell_probe / interval)) + 2
    scene = scene_build(integrator['dt'], integrator['substeps'], integrator['damping'], integrator['angular_damping'])
    rod = add_rod(scene, nv, interval, 1e7, 0.001, pos=(0, 0, 0.7))
    scene.build(n_envs=1)
    rod.set_fixed_states(fixed_ids=[0, 1])
    apply_props(rod, np.array([B_eff_rawE]), np.array([segment_mass]))
    prev = vertices_fn(rod)[:, -1, 2].copy(); 
    for _ in range(100):
        for _ in range(200):
            scene.step()
        cur = vertices_fn(rod)[:, -1, 2].copy()
        if float(np.max(np.abs(cur - prev))) < 1e-4:
            break
        prev = cur
    return shape_frame(vertices_fn(rod)[0])


def extract_real(s34_manifest, expected_digest, batch_size=64):
    """Replay every frozen (setting,grasp,template,seed) tuple in Genesis, recording the 7-frame
    temporal (y,z) task shape + proprio + wrench; also the per-setting small-deflection probe frame."""
    s34_manifest = Path(s34_manifest)
    if sha256(s34_manifest) != expected_digest:
        raise RuntimeError('s34 manifest digest mismatch')
    cfg = json.loads(s34_manifest.read_text())
    from sim.scene import build_scene, add_straight_rod, add_moving_clamp, attach_moving_clamp, vertices
    from sim.material import apply_properties
    from sim.sweep import settings
    sm = json.loads((ROOT / 'manifests' / cfg['sweep_manifest']).read_text())
    ss = {s['id']: s for s in settings(sm)}
    integ = sm['integrator']; interval = sm['interval']; g = sm['gravity']; bounds = sm['stochastic_distribution']
    universe = cfg['universe']; grasps = cfg['history_policy']['grasps']; templates = cfg['history_policy']['templates']
    seeds = cfg['history_policy']['seeds']; ell_probe = cfg['probe']['ell_probe']
    ell = sm['grasp']['ell']; nvs = sm['grasp']['n_vertices']; drive_steps = sm.get('drive_steps', 360)
    # --- per-setting probe frames (small-deflection, label-blind) ---
    probes = {}
    for sid in universe:
        s = ss[sid]
        probes[sid] = probe_shape(build_scene, add_straight_rod, apply_properties, vertices,
                                  s['raw_E'], s['mass'], ell_probe, integ, interval)
    # --- task histories: temporal 7-frame (y,z) ---
    specs = []
    for sid in universe:
        for grasp in grasps:
            for template in templates:
                for seed in seeds:
                    specs.append(dict(setting=sid, grasp=grasp, template=template, seed=seed))
    start = time.time(); records = []
    for grasp in sorted({x['grasp'] for x in specs}):
        group = [x for x in specs if x['grasp'] == grasp]; nv = nvs[grasp]
        for off in range(0, len(group), batch_size):
            q = group[off:off + batch_size]
            scene = build_scene(integ['dt'], integ['substeps'], integ['damping'], integ['angular_damping'])
            rod = add_straight_rod(scene, nv, interval, 1e7, .001, pos=(0, 0, .5))
            box = add_moving_clamp(scene, (0, 0, .5)); scene.build(n_envs=len(q), env_spacing=(2, 2))
            apply_properties(rod, np.array([ss[x['setting']]['raw_E'] for x in q]),
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
                    t = sm['templates'][x['template']]; u = min(1., (step + 1) / (drive_steps * d['dur']))
                    s_ = u * u * (3 - 2 * u) if t['kind'] == 'ease' else u
                    pos.append((d['dx'] * (1 - s_) + (t['arc'] * d['arc'] * np.sin(np.pi * s_) if t['kind'] == 'arc' else 0),
                                d['dy'] * (1 - s_), .5 + .2 * s_))
                box.set_pos(np.asarray(pos)); scene.step()
                if (step + 1) in FRAME_STEPS:
                    vv = vertices(rod)
                    for e in range(len(q)):
                        frames[e].append(vv[e])
            # settle -> final frame
            prev = vertices(rod)[:, -1, 2]; stable = False
            for chunk in range(100):
                for _ in range(200):
                    scene.step()
                cur = vertices(rod)[:, -1, 2]; drift = float(np.max(np.abs(cur - prev))); prev = cur
                if drift < 1e-3:
                    stable = True; break
            if not stable:
                raise RuntimeError(f'history batch failed convergence grasp={grasp} off={off}')
            vv = vertices(rod)
            for e, x in enumerate(q):
                fr = frames[e] + [vv[e]]                       # 7 frames
                s = ss[x['setting']]
                records.append((x['setting'], x['grasp'], x['template'], x['seed'],
                                temporal_shape(fr), drive_summary(sm['templates'][x['template']], x['seed'], bounds),
                                supported_wrench(s['mass'], nv, interval, g), probes[x['setting']],
                                chunk + 1, drift))
            print(json.dumps({'complete': len(records), 'total': len(specs), 'grasp': grasp, 'off': off}), flush=True)
    out = ROOT / 'manifests/hard_histories_v2.npz'; f = list(zip(*records))
    np.savez_compressed(out, setting=np.array(f[0]), grasp=np.array(f[1]), template=np.array(f[2]),
                        seed=np.array(f[3]), shape=np.array(f[4]), proprio=np.array(f[5]),
                        wrench=np.array(f[6]), probe_shape=np.array(f[7]),
                        settle_chunks=np.array(f[8]), last_tip_drift=np.array(f[9]),
                        manifest_digest=expected_digest, wall_clock_s=time.time() - start,
                        rollout_count=len(records), shape_dim=f[4][0].shape[0], probe_dim=f[7][0].shape[0],
                        source='actual Genesis get_vertices_pos temporal 7-frame + small-deflection probe')
    print(json.dumps({'output': str(out), 'rollout_count': len(records), 'shape_dim': int(f[4][0].shape[0]),
                      'probe_dim': int(f[7][0].shape[0]), 'wall_clock_s': time.time() - start}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', default=str(ROOT / 'manifests/hard_s34_manifest.json'))
    p.add_argument('--expected-digest', required=True); p.add_argument('--batch-size', type=int, default=64)
    a = p.parse_args(); extract_real(a.manifest, a.expected_digest, a.batch_size)
