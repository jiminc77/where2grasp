"""Build the frozen hard_sweep_manifest.json for the in-regime gate re-sweep.

Design is ANALYTIC and uses ONLY prior committed data (calibration.json +
sweep_landscape.json) + the fourth-root law. NO new-grid simulator peeking, NO
adaptive-h. Fixed h=0.009, fixed interval=0.01. The grid is the LOCKED Option-A
4x3 rectangular sub-grid with STABLE IDs (survivors keep original numbering
B1..B4 x w0..w2). Every cell's predicted Pi_g is asserted <=0.5 (with margin) and
its predicted boundary asserted >= the calibration-validated arm range (>=0.18 m)
BEFORE the manifest is written; infeasibility raises (escalate to owner, never
narrow the grid or lower h).
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
MAN = ROOT / 'manifests'
H = 0.009
INTERVAL = 0.01
G = 9.81
PI_G_MAX = 0.5
PI_G_MARGIN = 0.02          # require predicted Pi_g <= 0.5 - margin
MIN_BOUNDARY = 0.18         # calibration-validated arm range floor (18 segments)
ELL_GRID = [round(0.12 + 0.03 * k, 2) for k in range(17)]   # 0.12..0.60 step 0.03


def fit_prefactor():
    """Empirical prefactor p from PRIOR committed Step-2 data: ell_b = p*(8h*B_eff/w)^(1/4).
    Uses only resolved, in-regime prior boundaries (never the new grid)."""
    land = json.loads((MAN / 'sweep_landscape.json').read_text())['settings']
    ps = []
    for s in land:
        if s.get('resolved') and s.get('in_regime') and s.get('boundary') and s.get('w'):
            base = (8 * H * float(s['B_eff']) / float(s['w'])) ** 0.25
            ps.append(float(s['boundary']) / base)
    return float(np.mean(ps)), len(ps)


def predict(B_eff, w, p):
    boundary = p * (8 * H * B_eff / w) ** 0.25
    pi_g = w * boundary ** 3 / B_eff
    return boundary, pi_g


def build():
    cal = json.loads((MAN / 'calibration.json').read_text())
    old = json.loads((MAN / 'sweep_manifest.json').read_text())
    p_emp, n_fit = fit_prefactor()
    p_theory = 1.0
    # Stable-ID grid: keep original bi=1..4 (drop softest B0), wi=0..2 (drop heaviest w3).
    raw_E = old['raw_E']; B_eff = old['B_eff']; masses = old['segment_masses']; w = old['w']
    bis = [1, 2, 3, 4]; wis = [0, 1, 2]
    grid = []; table = []
    for bi in bis:
        for wi in wis:
            Be = float(B_eff[bi]); ww = float(w[wi])
            b_th, pg_th = predict(Be, ww, p_theory)
            b_em, pg_em = predict(Be, ww, p_emp)
            cell = dict(id=f'B{bi}_w{wi}', bi=bi, wi=wi, raw_E=float(raw_E[bi]),
                        B_eff=Be, mass=float(masses[wi]), w=ww)
            grid.append(cell)
            table.append(dict(id=cell['id'], w_over_B=round(ww / Be, 2),
                              boundary_pred_theory=round(b_th, 4), Pi_g_pred_theory=round(pg_th, 4),
                              boundary_pred_empirical=round(b_em, 4), Pi_g_pred_empirical=round(pg_em, 4),
                              in_regime_pred=bool(max(pg_th, pg_em) <= PI_G_MAX),
                              resolved_pred=bool(min(b_th, b_em) >= MIN_BOUNDARY and max(b_th, b_em) <= max(ELL_GRID))))
    # --- feasibility asserts (pre-data, analytic). Infeasibility => raise (escalate). ---
    viol = []
    for t in table:
        worst_pg = max(t['Pi_g_pred_theory'], t['Pi_g_pred_empirical'])
        min_b = min(t['boundary_pred_theory'], t['boundary_pred_empirical'])
        max_b = max(t['boundary_pred_theory'], t['boundary_pred_empirical'])
        if worst_pg > PI_G_MAX - PI_G_MARGIN:
            viol.append(f"{t['id']}: predicted Pi_g {worst_pg} > {PI_G_MAX-PI_G_MARGIN}")
        if min_b < MIN_BOUNDARY:
            viol.append(f"{t['id']}: predicted boundary {min_b} < {MIN_BOUNDARY} (under-resolved)")
        if max_b > max(ELL_GRID):
            viol.append(f"{t['id']}: predicted boundary {max_b} > ell grid top {max(ELL_GRID)}")
    if viol:
        (MAN / 'hard_grid_infeasible.json').write_text(json.dumps(
            {'infeasible': True, 'violations': viol, 'table': table}, indent=2))
        raise SystemExit('GRID INFEASIBLE (escalate to owner; do NOT narrow grid or lower h):\n' + '\n'.join(viol))

    nvs = [int(round(e / INTERVAL)) + 2 for e in ELL_GRID]
    manifest = dict(
        schema_version=1, frozen=True,
        design_note=('Fixed h=0.009, interval=0.01. Locked Option-A 4x3 in-regime grid, stable IDs '
                     'B1..B4 x w0..w2 (softest B0 row + heaviest w3 col excluded as small-deflection '
                     'regime-of-validity). Grid designed analytically from PRIOR committed data '
                     '(calibration.json + sweep_landscape.json fourth-root fit); no new-grid peeking, '
                     'no adaptive-h. Every predicted Pi_g asserted <=0.5 with margin before freeze.'),
        h=H, tau=0.5, interval=INTERVAL, gravity=G,
        integrator=old['integrator'], drive_steps=360,
        prefactor_fit=dict(p_empirical=p_emp, n_prior_cells=n_fit, p_theory=p_theory,
                           predicted_prefactor_8h_quarter=(8 * H) ** 0.25),
        grid=grid,
        predicted_Pi_g_table=table,
        grasp=dict(orientation='larger index means longer free arm', ell=ELL_GRID,
                   n_vertices=nvs, minimum_segments=12),
        templates=old['templates'],
        stochastic_distribution=dict(
            clamp_start_translation_xy_m=[-0.0035, 0.0035],
            motion_duration_multiplier=[0.92, 1.08], arc_multiplier=[0.9, 1.1],
            distribution='independent bounded uniform; terminal clamp pose exact',
            dropped_perturbation_note=('initial_rod_pose_translation_xy_m removed: no batched per-env '
                                       'rod-pose setter in Genesis c5026a9 verifies via read-back '
                                       '(set_position is a no-op w.r.t. get_vertices_pos), so the '
                                       'estimand applies only the clamp-start + duration/arc draws.')),
        seed_banks=dict(selection=[2000, 2001, 2002], evaluation=[3000, 3001, 3002, 3003, 3004],
                        pilot_reserved=[1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011]),
        selection_rule='CRN; maximize 3-draw mean success; ties lowest template index',
        evaluation_rule='selected template only; disjoint 5-draw mean success and mean J; no reselection',
        boundary_rule='all tau crossings by linear interpolation; boundary=max crossing; uncertainty=half grid step; all-success/all-failure unresolved',
        decision_rule=old['decision_rule'],
        ratio_pairs=old['ratio_pairs'],
    )
    out = MAN / 'hard_sweep_manifest.json'
    out.write_text(json.dumps(manifest, indent=2))
    import hashlib
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    print(json.dumps(dict(output=str(out), sha256=digest, p_empirical=round(p_emp, 4), n_prior_cells=n_fit,
                          worst_pred_Pi_g=max(max(t['Pi_g_pred_theory'], t['Pi_g_pred_empirical']) for t in table),
                          boundary_range=[min(t['boundary_pred_theory'] for t in table),
                                          max(t['boundary_pred_empirical'] for t in table)],
                          cells=len(grid)), indent=2))
    return digest


if __name__ == '__main__':
    build()
