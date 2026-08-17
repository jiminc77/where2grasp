"""C2 figures + landscape artifact for the distal-tip gate.

Reads distal_manifest.json + distal_sweep_results.npz + distal_gate_verdict.json and writes
distal_sweep_landscape.json (per-setting measured mean-success / mean-J curves + predicted vs
measured optimum + band edges) plus two figures: predicted-vs-measured optimum curve (argmax
tracking B^1/4 / w^-1/4) and the material feasibility map.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sim.tip_model as tm

ROOT = Path(__file__).resolve().parent
MAN = ROOT / 'manifests'; FIG = ROOT / 'figures'


def main():
    FIG.mkdir(exist_ok=True)
    m = json.loads((MAN / 'distal_manifest.json').read_text())
    v = json.loads((MAN / 'distal_gate_verdict.json').read_text())
    sw = np.load(MAN / 'distal_sweep_results.npz')
    grid = np.array(m['grasp']['ell']); pred = {c['id']: c for c in m['grid']}
    prop = {c['id']: (c['B_eff'], c['w']) for c in m['grid']}; prop.update({r['id']: (r['B_eff'], r['w']) for r in m['ratio_pairs']})
    settings_out = []
    for sid, me in v['measured'].items():
        p = pred.get(sid, {})
        settings_out.append(dict(id=sid, B_eff=prop[sid][0], w=prop[sid][1],
                                 mean_success=me['mean_success'], mean_J=me['mean_J'],
                                 measured_argmax_ell=me['argmax_ell'], measured_ell_L=me['ell_L'], measured_ell_U=me['ell_U'],
                                 predicted_argmax_ell=(grid[p['argmax_idx']] if p.get('argmax_idx') is not None else None),
                                 predicted_ell_L=p.get('ell_L'), predicted_ell_U=p.get('ell_U'),
                                 predicted_feasible=p.get('feasible'), measured_feasible=me['cell_feasible']))
    (MAN / 'distal_sweep_landscape.json').write_text(json.dumps({'settings': settings_out}, indent=2) + '\n')

    # figure 1: predicted vs measured optimum (feasible independent cells), argmax vs B_eff/w
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    feas = [c for c in m['grid'] if c['feasible']]
    xs = np.array([c['B_eff'] / c['w'] for c in feas])
    pred_l = np.array([grid[c['argmax_idx']] for c in feas])
    meas_l = np.array([v['measured'][c['id']]['argmax_ell'] if v['measured'][c['id']]['argmax_ell'] is not None else np.nan for c in feas])
    order = np.argsort(xs)
    ax[0].plot(xs[order], pred_l[order], 'k--o', label='predicted s* = (8 B_eff Delta/w)^1/4', ms=4)
    ax[0].plot(xs[order], meas_l[order], 'rs', label='measured argmax', ms=6, alpha=.7)
    ax[0].set_xscale('log'); ax[0].set_xlabel('B_eff / w'); ax[0].set_ylabel('optimum free-arm ell (m)')
    ax[0].set_title('Predicted vs measured optimum (feasible cells)'); ax[0].legend(fontsize=8)
    # figure 2: feasibility map (B rows x w cols)
    bis = sorted({c['bi'] for c in m['grid']}); wis = sorted({c['wi'] for c in m['grid']})
    grid_ids = {(c['bi'], c['wi']): c for c in m['grid']}
    Z = np.zeros((len(bis), len(wis)))
    for i, bi in enumerate(bis):
        for j, wj in enumerate(wis):
            c = grid_ids[(bi, wj)]; mfeas = v['measured'][c['id']]['cell_feasible']
            Z[i, j] = 1.0 if mfeas else 0.0
    ax[1].imshow(Z, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto', origin='lower')
    ax[1].set_xticks(range(len(wis))); ax[1].set_xticklabels([f'w{j}' for j in wis])
    ax[1].set_yticks(range(len(bis))); ax[1].set_yticklabels([f'B{i}' for i in bis])
    for i, bi in enumerate(bis):
        for j, wj in enumerate(wis):
            c = grid_ids[(bi, wj)]
            ax[1].text(j, i, 'FEAS' if Z[i, j] else 'FAIL', ha='center', va='center', fontsize=8)
    ax[1].set_title('Measured material feasibility (green=feasible)')
    fig.tight_layout(); fig.savefig(FIG / 'distal_optimum_and_feasibility.png', dpi=150); plt.close(fig)
    # per-cell success-curve panel for a few cells (0->1->0 band)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for sid in ('B1_w0', 'B2_w1', 'B3_w2', 'B4_w0', 'B1_w2'):
        ax.plot(grid, v['measured'][sid]['mean_success'], 'o-', ms=3, label=sid)
    ax.axhline(0.5, color='k', ls=':'); ax.set_xlabel('free-arm ell (m)'); ax.set_ylabel('evaluation success rate')
    ax.set_title('Distal 0->1->0 success bands (per material)'); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / 'distal_success_bands.png', dpi=150); plt.close(fig)
    print(json.dumps({'landscape': str(MAN / 'distal_sweep_landscape.json'),
                      'figures': ['distal_optimum_and_feasibility.png', 'distal_success_bands.png'],
                      'feasible_cells': len(feas)}, indent=2))


if __name__ == '__main__':
    main()
