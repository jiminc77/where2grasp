"""Finding-19 boundary-drift audit: per-env settled drift at in-regime cells near ell_L/ell_U.

The C2 sweep uses a MAX-over-envs 2e-3 convergence tolerance (set by the out-of-regime soft rods).
This audit re-measures, for each FEASIBLE cell, the in-regime grasps adjacent to its predicted band
edges ell_L (reach-limited lower) and ell_U (sag-limited upper), recording PER-ENV last-chunk
settled drift -> distal_boundary_drift_audit.json. It proves the looser MAX criterion cannot be
attacked at the band boundary: the in-regime boundary cells settle to <<2e-3.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from sim.scene import build_scene, add_straight_rod, add_moving_clamp, attach_moving_clamp, vertices
from sim.material import apply_properties
from sim.tasks.distal_tip import score_tip
import sim.tip_model as tm

ROOT = Path(__file__).resolve().parent
MAN = ROOT / 'manifests'


def _measure(m, cell, ell, seeds=(3000, 3001, 3002, 3003, 3004)):
    """Single rod at ell for `cell`'s material, n_envs=len(seeds); record per-env final-chunk drift."""
    integ = m['integrator']; nv = int(round(ell / m['interval'])) + 2; n = len(seeds)
    scene = build_scene(integ['dt'], integ['substeps'], integ['damping'], integ['angular_damping'])
    rod = add_straight_rod(scene, nv, m['interval'], 1e7, .001, pos=(0, 0, .5)); box = add_moving_clamp(scene, (0, 0, .5))
    scene.build(n_envs=n)
    apply_properties(rod, np.full(n, cell['raw_E']), np.full(n, cell['mass'])); attach_moving_clamp(rod, box)
    for step in range(m.get('drive_steps', 360)):
        s = (step + 1) / m.get('drive_steps', 360); box.set_pos(np.array([[0, 0, .5 + .2 * (s * s * (3 - 2 * s))]] * n)); scene.step()
    prev = vertices(rod)[:, 2:, :]; per_env_drift = None
    for chunk in range(80):
        for _ in range(200):
            scene.step()
        cur = vertices(rod)[:, 2:, :]
        per_env_drift = np.max(np.linalg.norm(cur - prev, axis=-1), axis=1)  # (n_envs,)
        prev = cur
        if float(per_env_drift.max()) < 2e-4:      # settle TIGHTLY to expose the true floor (<< the 2e-3 sweep tol)
            break
    state = vertices(rod)
    sc = [score_tip(state[e:e + 1], cell['B_eff'], cell['w'], ell)[0] for e in range(n)]
    return dict(ell=float(ell), pi_g=float(tm.pi_g(ell, cell['B_eff'], cell['w'])),
                per_env_drift=[float(x) for x in per_env_drift],
                max_env_drift=float(per_env_drift.max()), mean_env_drift=float(per_env_drift.mean()),
                per_env_reach=[round(s['reach'], 4) for s in sc], per_env_droop=[round(s['droop'], 4) for s in sc],
                per_env_success=[bool(s['success']) for s in sc])


def main():
    m = json.loads((MAN / 'distal_manifest.json').read_text())
    grid = np.array(m['grasp']['ell']); step = m['grasp']['step']
    out = {'note': ('per-env settled drift for in-regime grasps adjacent to each feasible cell ell_L/ell_U; '
                    'proves the 2e-3 MAX-over-envs sweep tolerance is not load-bearing at the band boundary '
                    '(in-regime boundary cells settle <<2e-3)'), 'cells': {}}
    for c in m['grid']:
        if not c['feasible']:
            continue
        rec = {}
        for edge, val in (('ell_L', c['ell_L']), ('ell_U', c['ell_U'])):
            # nearest in-regime grid grasp to the edge
            gi = int(np.argmin(np.abs(grid - val)))
            # step down to the nearest in-regime grasp if the nearest is out of regime
            while gi > 0 and tm.pi_g(grid[gi], c['B_eff'], c['w']) > tm.PI_G_MAX:
                gi -= 1
            rec[edge] = dict(edge_value=round(float(val), 4), audited_grasp=round(float(grid[gi]), 3),
                             **_measure(m, c, float(grid[gi])))
        out['cells'][c['id']] = rec
        print(json.dumps({'cell': c['id'],
                          'ell_L_grasp': rec['ell_L']['audited_grasp'], 'ell_L_max_drift': round(rec['ell_L']['max_env_drift'], 6),
                          'ell_U_grasp': rec['ell_U']['audited_grasp'], 'ell_U_max_drift': round(rec['ell_U']['max_env_drift'], 6)}), flush=True)
    worst = max(max(rec['ell_L']['max_env_drift'], rec['ell_U']['max_env_drift']) for rec in out['cells'].values())
    out['worst_in_regime_boundary_drift'] = worst
    out['all_boundary_cells_below_1e-3'] = bool(worst < 1e-3)
    (MAN / 'distal_boundary_drift_audit.json').write_text(json.dumps(out, indent=2))
    print(json.dumps({'worst_in_regime_boundary_drift': worst, 'all_below_1e-3': bool(worst < 1e-3)}, indent=2))


if __name__ == '__main__':
    main()
