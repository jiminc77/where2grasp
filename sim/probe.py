"""Exploratory small-deflection PROBE qualification screen (feasibility-only).

Runs BEFORE the frozen C4 probe manifest, committed under hardening/exploratory/.
Deterministic gravity settles (no stochastic draws -> the disjoint 1000-series pilot
bank is nominal; there is no seed contamination path). For each candidate probe
length it fits the Euler-Bernoulli law delta = w*ell^4/(8*B_eff) over a bracket and
applies the Step-1 calibration acceptance gate (log-log exponent ~= 4, CV/residual
< 5%, load invariance, Pi_g <= 0.3 for the softest kept material). It FREEZES the
shortest candidate that passes for every kept material. Qualification failure =>
committed INCONCLUSIVE/STOP; ell_probe is NEVER re-picked after governed data.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'hardening' / 'exploratory'
INTERVAL = 0.01
G = 9.81
CANDIDATES = [0.12, 0.15, 0.18]
PI_G_MAX = 0.3
EXP_LO, EXP_HI = 3.6, 4.4
CV_MAX = 0.05
RESID_MAX = 0.05
LOADINV_MAX = 0.08
PILOT_BANK = [1000, 1001, 1002, 1003, 1004]      # nominal; settle is deterministic


def _nv(ell):
    return int(round(ell / INTERVAL)) + 2


def _fit(ell, delta, w):
    x4 = np.asarray(ell) ** 4
    k = float(np.sum(x4 * delta) / np.sum(x4 * x4)); B = w / (8 * k)
    pred = k * x4
    resid = float(np.max(np.abs(delta - pred) / np.maximum(np.abs(delta), 1e-12)))
    per = w * np.asarray(ell) ** 4 / (8 * delta); cv = float(np.std(per) / np.mean(per))
    exponent = float(np.polyfit(np.log(ell), np.log(delta), 1)[0])
    return dict(B_eff=B, CV=cv, residual=resid, exponent=exponent)


def _bracket_run(rawEs, mass, lengths, integ):
    """One scene: len(lengths) rods x len(rawEs) materials (as envs) at a single mass.
    Returns delta[len, material]."""
    from sim.scene import build_scene, add_straight_rod, vertices
    from sim.material import apply_properties
    scene = build_scene(integ['dt'], integ['substeps'], integ['damping'], integ['angular_damping'])
    nvs = [_nv(L) for L in lengths]
    rods = [add_straight_rod(scene, nv, INTERVAL, 1e7, mass, pos=(0, 0.35 * j, 0.7)) for j, nv in enumerate(nvs)]
    scene.build(n_envs=len(rawEs), env_spacing=(5, 5))
    for rod in rods:
        rod.set_fixed_states(fixed_ids=[0, 1]); apply_properties(rod, np.asarray(rawEs), mass)
    tip0 = [vertices(r)[:, r.n_vertices - 1, 2].copy() for r in rods]
    prev = [vertices(r)[:, r.n_vertices - 1, 2].copy() for r in rods]
    for _ in range(160):
        for _ in range(200):
            scene.step()
        cur = [vertices(r)[:, r.n_vertices - 1, 2].copy() for r in rods]
        if max(float(np.max(np.abs(c - p))) for c, p in zip(cur, prev)) < 5e-5:
            break
        prev = cur
    tipf = [vertices(r)[:, r.n_vertices - 1, 2].copy() for r in rods]
    return np.array([tip0[j] - tipf[j] for j in range(len(nvs))])   # [len, material]


def screen():
    OUT.mkdir(parents=True, exist_ok=True)
    sm = json.loads((ROOT / 'manifests/hard_sweep_manifest.json').read_text())
    grid = sm['grid']; integ = sm['integrator']
    bis = sorted({c['bi'] for c in grid}); wis = sorted({c['wi'] for c in grid})
    rawE_by_bi = {c['bi']: c['raw_E'] for c in grid}; B_by_bi = {c['bi']: c['B_eff'] for c in grid}
    mass_by_wi = {c['wi']: c['mass'] for c in grid}; w_by_wi = {c['wi']: c['w'] for c in grid}
    rawEs = [rawE_by_bi[i] for i in bis]                     # B1..B4
    heavy = mass_by_wi[max(wis)]; light = mass_by_wi[min(wis)]; w_heavy = w_by_wi[max(wis)]
    results = []
    chosen = None
    for cand in CANDIDATES:
        lengths = [round(cand - 0.03, 2), cand, round(cand + 0.03, 2)]
        if lengths[0] < 0.06:
            continue
        d_heavy = _bracket_run(rawEs, heavy, lengths, integ)     # [len, material]
        d_light = _bracket_run(rawEs, light, lengths, integ)
        w_h = heavy * G / INTERVAL; w_l = light * G / INTERVAL
        per_mat = []
        for mi, bi in enumerate(bis):
            fh = _fit(lengths, d_heavy[:, mi], w_h); fl = _fit(lengths, d_light[:, mi], w_l)
            loadinv = abs(fh['B_eff'] - fl['B_eff']) / fl['B_eff']
            pi_g = w_heavy * cand ** 3 / fh['B_eff']             # worst load at the candidate length
            law_ok = (EXP_LO <= fh['exponent'] <= EXP_HI and fh['CV'] < CV_MAX and fh['residual'] < RESID_MAX)
            per_mat.append(dict(bi=bi, B_eff_fit=fh['B_eff'], exponent=fh['exponent'], CV=fh['CV'],
                                residual=fh['residual'], load_invariance=loadinv, Pi_g_at_cand=pi_g,
                                law_ok=bool(law_ok), pi_g_ok=bool(pi_g <= PI_G_MAX),
                                loadinv_ok=bool(loadinv <= LOADINV_MAX)))
        passed = all(pm['law_ok'] and pm['pi_g_ok'] and pm['loadinv_ok'] for pm in per_mat)
        worst_pi = max(pm['Pi_g_at_cand'] for pm in per_mat)
        results.append(dict(ell_probe=cand, bracket=lengths, passed=bool(passed),
                            worst_Pi_g=worst_pi, per_material=per_mat))
        if passed and chosen is None:
            chosen = cand                                       # shortest passing candidate
    report = dict(schema='probe_qualification.v1', kind='exploratory-feasibility-only',
                  pilot_bank=PILOT_BANK, note='deterministic gravity settle; no stochastic draws',
                  acceptance=dict(exponent=[EXP_LO, EXP_HI], CV_max=CV_MAX, residual_max=RESID_MAX,
                                  Pi_g_max=PI_G_MAX, loadinv_max=LOADINV_MAX),
                  selection_rule='shortest candidate passing the acceptance gate for every kept material',
                  candidates=results, chosen_ell_probe=chosen,
                  verdict=('QUALIFIED' if chosen is not None else 'INCONCLUSIVE-STOP'))
    (OUT / 'probe_qualification.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'chosen_ell_probe': chosen, 'verdict': report['verdict'],
                      'candidates': [{'ell': r['ell_probe'], 'passed': r['passed'], 'worst_Pi_g': round(r['worst_Pi_g'], 3)} for r in results]}, indent=2))
    return chosen


if __name__ == '__main__':
    screen()
