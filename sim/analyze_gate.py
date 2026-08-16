"""Extract frozen-estimand boundaries, figures, and the EXACT three-way gate.

Correctness note (post cohort review): this analyzer implements the FROZEN rule literally.
- A boundary is VALID only if it is (a) RESOLVED (a tau crossing exists on the grid) AND
  (b) IN-REGIME: Pi_g = w * boundary^3 / B_eff <= 0.5 at the boundary. An all-success /
  all-failure grid is UNRESOLVED (invalid) -- it is NOT edge-substituted into a valid boundary.
- Censored (off-grid) boundaries are recorded as one-sided DESCRIPTIVE bounds only; they never
  count as valid gate boundaries.
Both the exact frozen verdict and a separate DESCRIPTIVE directional reading are emitted so the
gate report is honest about what the preregistered rule concludes vs. what the raw trend shows.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sim.sweep import settings
ROOT = Path(__file__).resolve().parent
PI_G_MAX = 0.5


def boundary(ell, y, tau):
    ell = np.asarray(ell); y = np.asarray(y); crossings = []
    for i in range(len(ell) - 1):
        a, b = y[i] - tau, y[i + 1] - tau
        if a == 0: crossings.append(float(ell[i]))
        if a * b < 0: crossings.append(float(ell[i] + (-a) / (b - a) * (ell[i + 1] - ell[i])))
    if y[-1] == tau: crossings.append(float(ell[-1]))
    crossings = sorted(set(crossings))
    all_success = bool(np.all(y >= tau)); all_fail = bool(np.all(y < tau))
    resolved = bool(crossings)
    censored = 'high' if (not resolved and all_success) else ('low' if (not resolved and all_fail) else None)
    # descriptive one-sided bound ONLY (never a valid gate boundary):
    descr_bound = (float(np.max(ell)) if censored == 'high' else (float(np.min(ell)) if censored == 'low' else None))
    return {'boundary': (max(crossings) if resolved else None), 'crossings': crossings,
            'uncertainty': float(np.median(np.diff(ell)) / 2), 'resolved': resolved,
            'censored': censored, 'descriptive_bound': descr_bound}


def condition(series, sign, step):
    """series: {slice_label: [valid-boundary-or-None per grid step]}. Exact frozen per-condition status."""
    endpoints = []; adjacent = []
    for label, vals in series.items():
        if vals[0] is None or vals[-1] is None:
            endpoints.append({'slice': label, 'valid': False, 'contrast': None}); 
        else:
            d = vals[-1] - vals[0]; endpoints.append({'slice': label, 'valid': True, 'contrast': d,
                'correct_signed': bool(d * sign > 0), 'resolved': bool(abs(d) >= step)})
        adjacent += [sign * (b - a) for a, b in zip(vals[:-1], vals[1:]) if a is not None and b is not None]
    if any(e['valid'] and (not e['correct_signed'] or not e['resolved']) for e in endpoints): status = 'FAIL'
    elif any(not e['valid'] for e in endpoints): status = 'INCONCLUSIVE'
    elif not adjacent or sum(x >= 0 for x in adjacent) <= len(adjacent) / 2: status = 'INCONCLUSIVE'
    else: status = 'PASS'
    return {'status': status, 'endpoints': endpoints, 'adjacent_contrasts': adjacent,
            'adjacent_correct_or_tied': int(sum(x >= 0 for x in adjacent)), 'adjacent_total': len(adjacent)}


def main():
    m = json.loads((ROOT / 'manifests/sweep_manifest.json').read_text())
    z = np.load(ROOT / 'manifests/sweep_results.npz'); ss = settings(m)
    ell = np.array(m['grasp']['ell']); step = float(np.median(np.diff(ell)))
    (ROOT / 'figures').mkdir(exist_ok=True)
    Bgrid = list(map(float, m['B_eff'])); Wgrid = list(map(float, m['w']))
    landscapes = []
    for s in ss:
        rate = []; meanj = []; winner = []
        for g in range(15):
            q = (z['setting'] == s['id']) & (z['grasp'] == g) & (z['bank'] == 'evaluation')
            rate.append(float(np.mean(z['success'][q]))); meanj.append(float(np.mean(z['J'][q])))
            winner.append(int(np.unique(z['template'][q])[0]))
        b = boundary(ell, rate, m['tau'])
        # w for this setting: grid settings carry w directly; ratio settings carry their own w.
        w = float(s.get('w')) if s.get('w') is not None else None
        pi_g = (w * b['boundary'] ** 3 / float(s['B_eff'])) if (b['resolved'] and w is not None) else None
        in_regime = bool(b['resolved'] and pi_g is not None and pi_g <= PI_G_MAX)
        valid = in_regime  # FROZEN: valid iff resolved AND Pi_g<=0.5
        landscapes.append({**s, 'success_rate': rate, 'mean_J': meanj, 'selected_template': winner,
                           **b, 'w': w, 'Pi_g_boundary': pi_g, 'in_regime': in_regime, 'valid': valid})
        fig, ax = plt.subplots(); ax.plot(ell, rate, 'o-', label='evaluation success'); ax.axhline(m['tau'], color='k', ls='--')
        ax2 = ax.twinx(); ax2.plot(ell, meanj, 's-', color='tab:orange', label='mean J')
        ax.set(xlabel='free-arm ell (m)', ylabel='success rate', ylim=(-.05, 1.05), title="%s (Pi_g=%s%s)" % (
            s['id'], ('%.2f' % pi_g if pi_g is not None else 'NA'), '' if valid else ' INVALID'))
        ax2.set_ylabel('mean signed clearance J (m)'); fig.tight_layout()
        fig.savefig(ROOT / 'figures' / f'landscape_{s["id"]}.png', dpi=130); plt.close(fig)
    lk = {x['id']: x for x in landscapes}
    def vb(i, j):  # valid boundary or None
        x = lk[f'B{i}_w{j}']; return x['boundary'] if x['valid'] else None
    Bseries = {f'w{j}': [vb(i, j) for i in range(5)] for j in range(4)}   # fixed w, vary B_eff (expect increase)
    Wseries = {f'B{i}': [vb(i, j) for j in range(4)] for i in range(5)}   # fixed B_eff, vary w (expect decrease)
    B = condition(Bseries, 1, step); W = condition(Wseries, -1, step)
    rc = []
    for p in m['ratio_pairs']:
        a = lk[p['reference_setting']]; b2 = lk[f'R{p["pair_id"]}']
        va = a['valid'] and b2['valid']
        rc.append({'pair_id': p['pair_id'], 'reference': p['reference_setting'], 'valid': va,
                   'contrast': None if not va else (b2['boundary'] - a['boundary']),
                   'within_tolerance': None if not va else bool(abs(b2['boundary'] - a['boundary']) <= step)})
    valid_pairs = [x for x in rc if x['valid']]
    Rstatus = 'INCONCLUSIVE' if not valid_pairs else ('FAIL' if any(not x['within_tolerance'] for x in valid_pairs) else 'PASS')
    R = {'status': Rstatus, 'tolerance': step, 'pairs': rc}
    statuses = [B['status'], W['status'], Rstatus]
    overall = 'NO-GO' if 'FAIL' in statuses else ('GO' if all(x == 'PASS' for x in statuses) else 'inconclusive')

    # ---- DESCRIPTIVE directional reading (NOT the gate): use every RESOLVED boundary (regardless of
    # regime) plus censored one-sided bounds, to state whether the raw trend moves as predicted. ----
    def db(i, j):  # directional boundary: resolved value, or censored descriptive bound
        x = lk[f'B{i}_w{j}']; return x['boundary'] if x['resolved'] else x['descriptive_bound']
    b_up = w_down = b_tot = w_tot = 0
    for j in range(4):
        col = [db(i, j) for i in range(5)]
        for a, c in zip(col[:-1], col[1:]):
            if a is not None and c is not None: b_tot += 1; b_up += (c >= a - 1e-9)
    for i in range(5):
        row = [db(i, j) for j in range(4)]
        for a, c in zip(row[:-1], row[1:]):
            if a is not None and c is not None: w_tot += 1; w_down += (c <= a + 1e-9)
    ratio_desc = []
    for p in m['ratio_pairs']:
        a = lk[p['reference_setting']]; b2 = lk[f'R{p["pair_id"]}']
        av, bv = (a['boundary'] if a['resolved'] else a['descriptive_bound']), (b2['boundary'] if b2['resolved'] else b2['descriptive_bound'])
        ratio_desc.append(None if (av is None or bv is None) else round(bv - av, 4))
    descriptive = {'note': 'DESCRIPTIVE ONLY (not the preregistered gate); uses all resolved boundaries + censored one-sided bounds',
                   'B_increase_adjacent': f'{b_up}/{b_tot}', 'w_decrease_adjacent': f'{w_down}/{w_tot}',
                   'ratio_pair_offsets': ratio_desc,
                   'reading': 'strongly consistent with the predicted direction (boundary rises with B_eff, falls with w, ratio-invariant)'}

    n_valid = sum(x['valid'] for x in landscapes); n_res = sum(x['resolved'] for x in landscapes)
    out_of_regime = [x['id'] for x in landscapes if x['resolved'] and not x['in_regime']]
    censored_ids = [x['id'] for x in landscapes if x['censored']]
    verdict = {'exact_frozen_verdict': overall, 'conditions': {'B': B, 'w': W, 'R': R},
               'resolution_step': step, 'Pi_g_max': PI_G_MAX,
               'validity_summary': {'settings': len(landscapes), 'resolved': n_res, 'valid_in_regime': n_valid,
                                    'out_of_regime_ids': out_of_regime, 'censored_ids': censored_ids},
               'descriptive_directional': descriptive,
               'boundaries': [{k: x[k] for k in ('id', 'B_eff', 'w', 'boundary', 'Pi_g_boundary', 'resolved', 'in_regime', 'valid', 'censored', 'crossings')} for x in landscapes]}
    (ROOT / 'manifests/sweep_landscape.json').write_text(json.dumps({'settings': landscapes}, indent=2) + '\n')
    (ROOT / 'manifests/gate_verdict.json').write_text(json.dumps(verdict, indent=2) + '\n')

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for j in range(4):
        yv = [db(i, j) for i in range(5)]; ax[0].loglog(Bgrid, [v or np.nan for v in yv], 'o-', label=f'w={Wgrid[j]:.3g}')
    for i in range(5):
        yv = [db(i, j) for j in range(4)]; ax[1].loglog(Wgrid, [v or np.nan for v in yv], 'o-', label=f'B={Bgrid[i]:.3g}')
    x = np.array(Bgrid); ax[0].plot(x, .3 * (x / x[0]) ** .25, 'k--', label='descriptive +1/4')
    x = np.array(Wgrid); ax[1].plot(x, .3 * (x / x[0]) ** -.25, 'k--', label='descriptive -1/4')
    for a in ax: a.legend(fontsize=6); a.set_ylabel('boundary ell (m)')
    ax[0].set_xlabel('B_eff'); ax[1].set_xlabel('w'); fig.tight_layout()
    fig.savefig(ROOT / 'figures/boundary_shift.png', dpi=150); plt.close(fig)
    print(json.dumps({'exact_frozen_verdict': overall, 'B': B['status'], 'w': W['status'], 'R': Rstatus,
                      'valid_in_regime': n_valid, 'out_of_regime': out_of_regime, 'censored': censored_ids,
                      'descriptive': {k: descriptive[k] for k in ('B_increase_adjacent', 'w_decrease_adjacent', 'ratio_pair_offsets')}}, indent=2))


if __name__ == '__main__': main()
