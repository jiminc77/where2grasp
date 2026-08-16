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
import argparse, hashlib, json
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


def _prefactor_bootstrap(z, landscapes, h, n_boot=2000, seed=7):
    """Seeded paired block-bootstrap of the prefactor ell_max/(B_eff/w)^(1/4) over the
    matched evaluation-seed blocks. Returns predicted (8h)^(1/4), point estimate, CI, and
    a grid-resolution bound. Honest-negative capable: censored/invalid cells are excluded."""
    from sim.analyze_gate import boundary as _b  # frozen boundary fn
    predicted = float((8.0 * h) ** 0.25)
    valid = [x for x in landscapes if x['valid']]
    def pf(x, rate):
        bb = _b(x['ell_grid'], rate, x['tau'])['boundary']
        return None if bb is None else bb / (float(x['B_eff']) / float(x['w'])) ** 0.25
    point = [pf(x, x['success_rate']) for x in valid]
    point = [p for p in point if p is not None]
    eval_seeds = sorted(set(int(s) for s in z['seed'][z['bank'] == 'evaluation'].tolist()))
    ns = len(eval_seeds)
    # per (setting,grasp) success by eval seed
    succ = {}
    for x in valid:
        for gi, e_ell in enumerate(x['ell_grid']):
            vals = []
            for sd in eval_seeds:
                q = (z['setting'] == x['id']) & (z['grasp'] == gi) & (z['bank'] == 'evaluation') & (z['seed'] == sd)
                vals.append(float(np.mean(z['success'][q])) if q.any() else np.nan)
            succ[(x['id'], gi)] = np.array(vals)
    rng = np.random.default_rng(seed); boots = []
    for _ in range(n_boot):
        pick = rng.integers(0, ns, ns)
        vals = []
        for x in valid:
            rate = [float(np.nanmean(succ[(x['id'], gi)][pick])) for gi in range(len(x['ell_grid']))]
            p = pf(x, rate)
            if p is not None:
                vals.append(p)
        if vals:
            boots.append(float(np.mean(vals)))
    lo, hi = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))) if boots else (None, None)
    step = float(np.median(np.diff(valid[0]['ell_grid']))) if valid else None
    # grid-resolution bound: +/- half-step in boundary propagated to the mean prefactor
    denom = [(float(x['B_eff']) / float(x['w'])) ** 0.25 for x in valid]
    grid_bound = float(np.mean([(step / 2) / d for d in denom])) if valid else None
    return {'predicted_prefactor_8h_quarter': predicted,
            'observed_prefactor_mean': float(np.mean(point)) if point else None,
            'observed_prefactor_per_cell': {x['id']: pf(x, x['success_rate']) for x in valid},
            'bootstrap_ci95': [lo, hi], 'bootstrap_n': len(boots),
            'grid_resolution_bound_pm': grid_bound,
            'note': 'seeded paired block-bootstrap over matched evaluation-seed blocks; valid in-regime cells only'}


def main(manifest=None, results=None, out_prefix='hard_', figdir=None):
    manifest = Path(manifest) if manifest else (ROOT / 'manifests/sweep_manifest.json')
    results = Path(results) if results else (ROOT / 'manifests/sweep_results.npz')
    m = json.loads(manifest.read_text())
    z = np.load(results); ss = settings(m)
    ell = np.array(m['grasp']['ell']); step = float(np.median(np.diff(ell)))
    ng = len(ell)
    (ROOT / 'figures').mkdir(exist_ok=True)
    ind = [s for s in ss if s['kind'] == 'independent']
    bis = sorted({s['bi'] for s in ind}); wis = sorted({s['wi'] for s in ind})
    B_by_bi = {s['bi']: float(s['B_eff']) for s in ind}; w_by_wi = {s['wi']: float(s['w']) for s in ind}
    Bgrid = [B_by_bi[i] for i in bis]; Wgrid = [w_by_wi[j] for j in wis]
    landscapes = []
    for s in ss:
        rate = []; meanj = []; winner = []
        for g in range(ng):
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
                           'ell_grid': [float(e) for e in ell], 'tau': float(m['tau']),
                           **b, 'w': w, 'Pi_g_boundary': pi_g, 'in_regime': in_regime, 'valid': valid})
        fig, ax = plt.subplots(); ax.plot(ell, rate, 'o-', label='evaluation success'); ax.axhline(m['tau'], color='k', ls='--')
        ax2 = ax.twinx(); ax2.plot(ell, meanj, 's-', color='tab:orange', label='mean J')
        ax.set(xlabel='free-arm ell (m)', ylabel='success rate', ylim=(-.05, 1.05), title="%s (Pi_g=%s%s)" % (
            s['id'], ('%.2f' % pi_g if pi_g is not None else 'NA'), '' if valid else ' INVALID'))
        ax2.set_ylabel('mean signed clearance J (m)'); fig.tight_layout()
        fig.savefig(ROOT / 'figures' / f'{out_prefix}landscape_{s["id"]}.png', dpi=130); plt.close(fig)
    lk = {x['id']: x for x in landscapes}
    def vb(i, j):  # valid boundary or None
        x = lk[f'B{i}_w{j}']; return x['boundary'] if x['valid'] else None
    Bseries = {f'w{j}': [vb(i, j) for i in bis] for j in wis}   # fixed w, vary B_eff (expect increase)
    Wseries = {f'B{i}': [vb(i, j) for j in wis] for i in bis}   # fixed B_eff, vary w (expect decrease)
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
    for j in wis:
        col = [db(i, j) for i in bis]
        for a, c in zip(col[:-1], col[1:]):
            if a is not None and c is not None: b_tot += 1; b_up += (c >= a - 1e-9)
    for i in bis:
        row = [db(i, j) for j in wis]
        for a, c in zip(row[:-1], row[1:]):
            if a is not None and c is not None: w_tot += 1; w_down += (c <= a + 1e-9)
    ratio_desc = []
    for p in m['ratio_pairs']:
        a = lk[p['reference_setting']]; b2 = lk[f'R{p["pair_id"]}']
        av, bv = (a['boundary'] if a['resolved'] else a['descriptive_bound']), (b2['boundary'] if b2['resolved'] else b2['descriptive_bound'])
        ratio_desc.append(None if (av is None or bv is None) else round(bv - av, 4))
    descriptive = {'note': 'DESCRIPTIVE ONLY (not the preregistered gate); uses all resolved boundaries + censored one-sided bounds',
                   'B_increase_adjacent': f'{b_up}/{b_tot}', 'w_decrease_adjacent': f'{w_down}/{w_tot}',
                   'ratio_pair_offsets': ratio_desc}

    prefactor = _prefactor_bootstrap(z, landscapes, float(m['h']))
    n_valid = sum(x['valid'] for x in landscapes); n_res = sum(x['resolved'] for x in landscapes)
    out_of_regime = [x['id'] for x in landscapes if x['resolved'] and not x['in_regime']]
    censored_ids = [x['id'] for x in landscapes if x['censored']]
    verdict = {'exact_frozen_verdict': overall, 'conditions': {'B': B, 'w': W, 'R': R},
               'resolution_step': step, 'Pi_g_max': PI_G_MAX, 'manifest_digest': hashlib.sha256(manifest.read_bytes()).hexdigest(),
               'validity_summary': {'settings': len(landscapes), 'resolved': n_res, 'valid_in_regime': n_valid,
                                    'out_of_regime_ids': out_of_regime, 'censored_ids': censored_ids},
               'prefactor': prefactor, 'descriptive_directional': descriptive,
               'boundaries': [{k: x[k] for k in ('id', 'B_eff', 'w', 'boundary', 'Pi_g_boundary', 'resolved', 'in_regime', 'valid', 'censored', 'crossings')} for x in landscapes]}
    (ROOT / f'manifests/{out_prefix}sweep_landscape.json').write_text(json.dumps({'settings': landscapes}, indent=2) + '\n')
    (ROOT / f'manifests/{out_prefix}gate_verdict.json').write_text(json.dumps(verdict, indent=2) + '\n')

    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for j in wis:
        yv = [db(i, j) for i in bis]; ax[0].loglog(Bgrid, [v or np.nan for v in yv], 'o-', label=f'w={w_by_wi[j]:.3g}')
    for i in bis:
        yv = [db(i, j) for j in wis]; ax[1].loglog(Wgrid, [v or np.nan for v in yv], 'o-', label=f'B={B_by_bi[i]:.3g}')
    x = np.array(Bgrid); ax[0].plot(x, .3 * (x / x[0]) ** .25, 'k--', label='reference +1/4')
    x = np.array(Wgrid); ax[1].plot(x, .3 * (x / x[0]) ** -.25, 'k--', label='reference -1/4')
    for a in ax[:2]: a.legend(fontsize=6); a.set_ylabel('boundary ell (m)')
    ax[0].set_xlabel('B_eff'); ax[1].set_xlabel('w')
    # prefactor panel
    pc = prefactor['observed_prefactor_per_cell']; ids = list(pc)
    ax[2].axhline(prefactor['predicted_prefactor_8h_quarter'], color='k', ls='--', label='(8h)^1/4 predicted')
    ci = prefactor['bootstrap_ci95']
    if ci[0] is not None:
        ax[2].axhspan(ci[0], ci[1], color='tab:blue', alpha=0.2, label='mean 95% CI')
    ax[2].plot(range(len(ids)), [pc[i] for i in ids], 'o'); ax[2].set_xticks(range(len(ids))); ax[2].set_xticklabels(ids, rotation=90, fontsize=6)
    ax[2].set_ylabel('ell_max/(B_eff/w)^1/4'); ax[2].set_title('prefactor'); ax[2].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(ROOT / f'figures/{out_prefix}boundary_shift.png', dpi=150); plt.close(fig)
    print(json.dumps({'exact_frozen_verdict': overall, 'B': B['status'], 'w': W['status'], 'R': Rstatus,
                      'valid_in_regime': n_valid, 'out_of_regime': out_of_regime, 'censored': censored_ids,
                      'prefactor': {k: prefactor[k] for k in ('predicted_prefactor_8h_quarter', 'observed_prefactor_mean', 'bootstrap_ci95')},
                      'descriptive': {k: descriptive[k] for k in ('B_increase_adjacent', 'w_decrease_adjacent', 'ratio_pair_offsets')}}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', default=None); p.add_argument('--results', default=None)
    p.add_argument('--out-prefix', default='hard_')
    a = p.parse_args(); main(a.manifest, a.results, a.out_prefix)