"""Build + freeze sim/manifests/distal_manifest.json for the distal-tip-placement task.

ANALYTIC and pre-data: reads ONLY prior committed data (hard_sweep_manifest.json grid =
the hardening-A in-regime property set; calibration.json for provenance) + sim.tip_model.
Runs every owner-mandated pre-freeze assert (findings 1/2/3) and RAISES SystemExit on any
violation (escalate to owner; NEVER narrow the grid / lower the depth / relax the regime).
Writes distal_manifest.json as a SINGLE FILE (committed alone in C1 before any data).
"""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np
import sim.tip_model as tm

ROOT = Path(__file__).resolve().parent
MAN = ROOT / 'manifests'


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _load_grid():
    """Prior committed grid + ratio pairs from hard_sweep_manifest.json (hardening-A in-regime set)."""
    m = json.loads((MAN / 'hard_sweep_manifest.json').read_text())
    cells = {c['id']: dict(id=c['id'], bi=c['bi'], wi=c['wi'], raw_E=float(c['raw_E']),
                           B_eff=float(c['B_eff']), mass=float(c['mass']), w=float(c['w'])) for c in m['grid']}
    ratios = []
    for p in m['ratio_pairs']:
        q = p['scaled']; E = q['raw_E']; mass = q['segment_mass']
        xs = sorted({(c['raw_E'], c['B_eff']) for c in m['grid']})
        B = float(np.interp(E, [x[0] for x in xs], [x[1] for x in xs]))
        ratios.append(dict(id=f'R{p["pair_id"]}', pair_id=p['pair_id'], reference=p['reference_setting'],
                           raw_E=float(E), B_eff=B, mass=float(mass), w=float(mass) * m['gravity'] / m['interval'], c=p['c']))
    return m, cells, ratios


def build():
    m0, cells, ratios = _load_grid()
    grid = tm.ell_grid()
    step = tm.ELL_STEP
    Delta, rho, d, rho_x = tm.DELTA, tm.RHO, tm.D, tm.RHO_X
    R = d - rho_x

    # ---- per-cell analytic records (independent grid cells) ----
    ana = {cid: tm.cell_analysis(c['B_eff'], c['w'], Delta, rho, d, rho_x, grid) for cid, c in cells.items()}
    for cid, a in ana.items():
        a['id'] = cid
    feasible = {cid for cid, a in ana.items() if a['feasible']}
    infeasible = {cid for cid, a in ana.items() if not a['feasible']}

    viol = []
    # (1) feasibility partition must be exactly {B1_w2 fail} + 11 feasible (else escalate; never narrow)
    if infeasible != {'B1_w2'}:
        viol.append(f'feasibility partition changed: infeasible={sorted(infeasible)} (expected {{B1_w2}})')
    # (2) FAIL cell model-valid: l_top==l_regime (Pi_g=0.5) AND x(l_top)+half_step <= R
    fa = ana.get('B1_w2')
    if fa is not None:
        if abs(fa['ell_top'] - fa['ell_regime']) > 1e-9:
            viol.append('B1_w2 fail not certified at the in-regime ceiling (l_top != l_regime)')
        if not (fa['x_at_top'] + step / 2 <= R):
            viol.append(f"B1_w2 reach margin < half step: x(l_top)={fa['x_at_top']:.4f}, R={R}")
        if tm.pi_g(fa['ell_top'], fa['B_eff'], fa['w']) > tm.PI_G_MAX + 1e-9:
            viol.append('B1_w2 fail-defining grasp out of regime')
    # (3) every feasible cell: interior non-clipped in-regime optimum, l_reach<l*<l_top, conservative guard on l*,l_L
    for cid in sorted(feasible):
        a = ana[cid]
        if a['argmax_idx'] is None or a['clipped'] or a['censored']:
            viol.append(f'{cid}: optimum clipped/censored/missing')
        if not (a['ell_reach'] is not None and a['ell_reach'] < a['ell_star'] < a['ell_top']):
            viol.append(f"{cid}: not l_reach<l*<l_top ({a['ell_reach']},{a['ell_star']},{a['ell_top']})")
        if not a['regime_star']:
            viol.append(f'{cid}: optimum fails conservative upper-bracket regime guard')
        if not a['regime_L']:
            viol.append(f'{cid}: l_L fails conservative upper-bracket regime guard')
        if tm.pi_g(a['ell_star'], a['B_eff'], a['w']) > tm.PI_G_MAX:
            viol.append(f'{cid}: Pi_g* > 0.5')
    # (4) eligible adjacent feasible contrasts shift the nearest-grid argmax >= 2 cells
    bis = ['B1', 'B2', 'B3', 'B4']; wis = ['w0', 'w1', 'w2']
    contrasts = []
    for wj in wis:
        col = [f'{b}_{wj}' for b in bis if f'{b}_{wj}' in feasible]
        contrasts += [(col[k], col[k + 1], 'B_up') for k in range(len(col) - 1)]
    for bi in bis:
        row = [f'{bi}_{wj}' for wj in wis if f'{bi}_{wj}' in feasible]
        contrasts += [(row[k], row[k + 1], 'w_down') for k in range(len(row) - 1)]
    shift_table = []
    for a_id, b_id, kind in contrasts:
        sm = abs(ana[a_id]['ell_star'] - ana[b_id]['ell_star'])
        idx_a = ana[a_id]['argmax_idx']; idx_b = ana[b_id]['argmax_idx']
        cell_shift = abs(idx_a - idx_b)
        rec = dict(a=a_id, b=b_id, kind=kind, shift_m=round(sm, 5), shift_cells_continuous=round(sm / step, 3),
                   argmax_idx_a=idx_a, argmax_idx_b=idx_b, argmax_shift_cells=cell_shift,
                   meets_min=bool(sm / step >= tm.MIN_SHIFT_CELLS and cell_shift >= 1))
        shift_table.append(rec)
        if sm / step < tm.MIN_SHIFT_CELLS:
            viol.append(f'{a_id}->{b_id}: predicted shift {sm/step:.2f} < {tm.MIN_SHIFT_CELLS} cells')
    # (5) A2 grid-degeneracy demonstration (A2 argmin shift < 1 cell where A1 >= 2), reported not asserted-blocking
    a2demo = []
    for a_id, b_id, kind in contrasts:
        ea = tm.a1_a2_enumeration(ana[a_id]['B_eff'], ana[a_id]['w'], Delta, rho, d, rho_x, grid=grid)
        eb = tm.a1_a2_enumeration(ana[b_id]['B_eff'], ana[b_id]['w'], Delta, rho, d, rho_x, grid=grid)
        a2demo.append(dict(a=a_id, b=b_id, a1_shift=abs(ea['a1_idx'] - eb['a1_idx']) if ea['a1_idx'] is not None and eb['a1_idx'] is not None else None,
                           a2_shift=abs(ea['a2_idx'] - eb['a2_idx'])))
    # ratio-pair references must be feasible (invariance controls)
    for r in ratios:
        if r['reference'] not in feasible:
            viol.append(f"ratio {r['id']} reference {r['reference']} not feasible")

    if viol:
        (MAN / 'distal_grid_infeasible.json').write_text(json.dumps(
            {'infeasible': True, 'violations': viol,
             'analysis': {cid: ana[cid] for cid in ana}}, indent=2))
        raise SystemExit('DISTAL GRID INFEASIBLE (escalate to owner; do NOT narrow grid / relax regime):\n'
                         + '\n'.join(viol))

    # ---- assemble the frozen manifest ----
    def cellrec(cid):
        c = cells[cid]; a = ana[cid]
        return dict(id=cid, bi=c['bi'], wi=c['wi'], raw_E=c['raw_E'], B_eff=c['B_eff'], mass=c['mass'], w=c['w'],
                    ell_reach=a['ell_reach'], ell_dlo=a['ell_dlo'], ell_star=a['ell_star'], ell_sag=a['ell_sag'],
                    ell_regime=a['ell_regime'], ell_top=a['ell_top'], ell_L=a['ell_L'], ell_U=a['ell_U'],
                    feasible=a['feasible'], argmax_idx=a['argmax_idx'],
                    argmax_ell=a['argmax_ell'], clipped=a['clipped'], censored=a['censored'],
                    pi_g_star=a['pi_g_star'], regime_star=a['regime_star'], regime_L=a['regime_L'], regime_U=a['regime_U'])

    nvs = [int(round(e / m0['interval'])) + 2 for e in grid]
    manifest = dict(
        schema_version=1, frozen=True, task='distal_tip_placement',
        design_note=('Distal tip placement (optimum-bearing). Continuous objective J=rho-|delta_tip-Delta|, '
                     'UNIQUE interior optimum s*=(8*B_eff*|z*|/w)^(1/4) (root of delta=Delta; unique by strict '
                     'monotonicity of delta_tip). Reach = feasible-set lower bound; sag/regime = upper bound + '
                     'optimum. Constants chosen ONCE; grid = hardening-A in-regime property set; all asserts pass.'),
        objective=dict(formula='J(ell)=rho-|delta_tip(ell)-Delta|', optimum='s*=(8*B_eff*|z*|/w)^(1/4)',
                       uniqueness='delta_tip(ell)=w*ell^4/(8B) strictly increasing => single interior maximiser',
                       Delta=Delta, z_star=-Delta, rho=rho, d=d, rho_x=rho_x, reach_floor_R=R, J_inf=tm.J_INF, tau=tm.TAU),
        integrator=m0['integrator'], interval=m0['interval'], gravity=m0['gravity'], drive_steps=m0.get('drive_steps', 360),
        pi_g_max=tm.PI_G_MAX,
        regime_guard=dict(rule='per-estimand: in-regime iff Pi_g(upper bracket endpoint of estimand grid bracket)<=0.5',
                          note='conservative (finding 3); ell_U for high-w/B feasible cells flagged INCONCLUSIVE-regime'),
        feasibility_rule=('IN-REGIME feasible iff x(l_top)>=d-rho_x AND l_top>=l_dlo, l_top=min(l_sag,l_regime), '
                          'l_regime=(0.5B/w)^(1/3); exact x(l) root (NOT the descriptive w/B approximation)'),
        descriptive_w_over_B_threshold=round(8 * (Delta + rho) / (d - rho_x) ** 4, 3),
        grasp=dict(orientation='larger index means longer free arm', ell=[float(x) for x in grid],
                   n_vertices=nvs, step=step, half_step=step / 2, min_shift_cells=tm.MIN_SHIFT_CELLS),
        tie_rule='argmax over in-regime reach-feasible grasps of eval-only mean J; |dJ|<=1e-9 -> lowest grid index',
        clipped_rule='argmax at grid endpoint (0 or N-1) flagged + excluded from off-grid-direction adjacency',
        censored_rule='no feasible in-regime grasp -> no argmax (feasibility only); flat feasible curve (<=1e-9) -> censored',
        gate_truth_table=dict(
            argmax='argmax over in-regime feasible grasps of evaluation-only mean J (tie/clipped/censored rules)',
            eligible_adjacency='predeclared feasible reach-slack non-clipped in-regime cells only',
            direction='argmax rises with B_eff (fixed w) and falls with w (fixed B); required shift >= 2 cells',
            ratio_tolerance_cells=1,
            band='ell_L=first-up tau crossing, ell_U=last-down; IoU (primary) + |dl_L|,|dl_U| + Hausdorff',
            precedence=('FAIL if any eligible valid contrast wrong-signed OR shift<2 cells OR ratio argmax-variance>1 cell '
                        'OR predicted/measured feasibility contradiction; INCONCLUSIVE if no FAIL but a required endpoint '
                        'censored/missing/clipped/out-of-regime OR eligible contrast lacks a valid argmax OR flat curve; '
                        'GO iff all required PASS; NO-GO if any FAIL; else inconclusive')),
        templates=m0['templates'], stochastic_distribution=m0['stochastic_distribution'],
        seed_banks=dict(selection=[2000, 2001, 2002], evaluation=[3000, 3001, 3002, 3003, 3004]),
        selection_rule=m0['selection_rule'], evaluation_rule=m0['evaluation_rule'],
        grid=[cellrec(cid) for cid in sorted(cells)],
        ratio_pairs=ratios,
        predicted_optimum_shift_table=shift_table,
        a2_degeneracy_demo=a2demo,
        eligible_adjacency_contrasts=[dict(a=a, b=b, kind=k) for a, b, k in contrasts],
        feasible_cells=sorted(feasible), infeasible_cells=sorted(infeasible),
        win_demo_cell='B2_w1', fail_demo_cell='B1_w2',
        input_artifact_sha256={'hard_sweep_manifest.json': sha256(MAN / 'hard_sweep_manifest.json'),
                               'calibration.json': sha256(MAN / 'calibration.json')},
        tip_model_constants=dict(DELTA=tm.DELTA, RHO=tm.RHO, D=tm.D, RHO_X=tm.RHO_X, J_INF=tm.J_INF,
                                 TAU=tm.TAU, PI_G_MAX=tm.PI_G_MAX, ELL_MIN=tm.ELL_MIN, ELL_MAX=tm.ELL_MAX,
                                 ELL_STEP=tm.ELL_STEP, MIN_SHIFT_CELLS=tm.MIN_SHIFT_CELLS),
    )
    out = MAN / 'distal_manifest.json'
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    digest = sha256(out)
    print(json.dumps(dict(output=str(out), sha256=digest, feasible=len(feasible), infeasible=sorted(infeasible),
                          worst_pi_g_star=max(ana[c]['pi_g_star'] for c in feasible),
                          min_shift_cells=min(s['shift_cells_continuous'] for s in shift_table),
                          grasps=len(grid), win=manifest['win_demo_cell'], fail=manifest['fail_demo_cell']), indent=2))
    return digest


if __name__ == '__main__':
    build()
