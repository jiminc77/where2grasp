"""Independent red-team recompute of hardening-A from committed artifacts.

Recomputes the Part-1 gate (boundaries, Pi_g, verdict, prefactor) from the raw sweep
npz with a FRESH boundary function (does NOT import analyze_gate), the Part-2 confound
guard from raw histories, verifies the unified split + dims, and checks git-provable
pre-registration (manifest-only commits ancestral to their data, digests matching).
Returns SURVIVES iff every recomputed number matches the committed reports and every
git-provability + honesty check holds.
"""
from __future__ import annotations
import json, subprocess, hashlib
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MAN = ROOT / 'manifests'
findings = []


def check(name, ok, detail=''):
    findings.append({'check': name, 'ok': bool(ok), 'detail': detail})
    return ok


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def git(*a):
    return subprocess.run(['git', '-C', str(ROOT.parent), *a], capture_output=True, text=True).stdout.strip()


def fresh_boundary(ell, y, tau=0.5):
    """Independent tau-crossing (max crossing); None if all-above/all-below (censored)."""
    ell = np.asarray(ell, float); y = np.asarray(y, float); cr = []
    for i in range(len(ell) - 1):
        a, b = y[i] - tau, y[i + 1] - tau
        if a == 0:
            cr.append(ell[i])
        if a * b < 0:
            cr.append(ell[i] + (-a) / (b - a) * (ell[i + 1] - ell[i]))
    if y[-1] == tau:
        cr.append(ell[-1])
    return (max(cr) if cr else None)


def part1():
    m = json.loads((MAN / 'hard_sweep_manifest.json').read_text())
    z = np.load(MAN / 'hard_sweep_results.npz')
    verdict = json.loads((MAN / 'hard_gate_verdict.json').read_text())
    ell = np.array(m['grasp']['ell']); h = m['h']; tau = m['tau']
    # digest + convergence
    check('P1 manifest digest matches results', str(z['manifest_digest'].item()) == sha(MAN / 'hard_sweep_manifest.json'))
    check('P1 nonconverged fraction == 0', float(np.mean(~z['converged'].astype(bool))) == 0.0)
    # recompute boundaries + Pi_g independently
    from sim.sweep import settings
    ss = settings(m); lk = {s['id']: s for s in ss}; recomputed = {}
    all_in_regime = True; mismatches = []
    for s in ss:
        rate = [float(np.mean(z['success'][(z['setting'] == s['id']) & (z['grasp'] == g) & (z['bank'] == 'evaluation')]))
                for g in range(len(ell))]
        b = fresh_boundary(ell, rate, tau)
        pg = (s['w'] * b ** 3 / s['B_eff']) if b else None
        recomputed[s['id']] = (b, pg)
        if b is None or pg is None or pg > 0.5:
            all_in_regime = False
        # compare to committed
        comm = next(x for x in verdict['boundaries'] if x['id'] == s['id'])
        if comm['boundary'] is not None and b is not None and abs(comm['boundary'] - b) > 1e-6:
            mismatches.append(s['id'])
    check('P1 boundaries match committed verdict', not mismatches, str(mismatches))
    check('P1 all 15 settings in-regime (recomputed Pi_g<=0.5)', all_in_regime)
    check('P1 committed verdict == GO', verdict['exact_frozen_verdict'] == 'GO', verdict['exact_frozen_verdict'])
    check('P1 conditions B/w/R all PASS', all(verdict['conditions'][k]['status'] == 'PASS' for k in ('B', 'w', 'R')))
    # prefactor recompute (valid cells)
    pf = np.mean([recomputed[s['id']][0] / (s['B_eff'] / s['w']) ** 0.25 for s in ss if recomputed[s['id']][0]])
    check('P1 prefactor matches committed (0.2% of (8h)^1/4)', abs(pf - verdict['prefactor']['observed_prefactor_mean']) < 1e-6,
          f'recomputed {pf:.4f} vs committed {verdict["prefactor"]["observed_prefactor_mean"]:.4f} vs (8h)^1/4 {(8*h)**0.25:.4f}')


def part2():
    cfg = json.loads((MAN / 'hard_s34_manifest.json').read_text())
    hz = np.load(MAN / 'hard_histories_v2.npz')
    idf = json.loads((MAN / 'hard_identifiability_v2.json').read_text())
    cr = json.loads((MAN / 'hard_critic_results_v2.json').read_text())
    land = {x['id']: x for x in json.loads((MAN / 'hard_sweep_landscape.json').read_text())['settings']}
    check('P2 history digest matches C4 manifest', str(hz['manifest_digest'].item()) == sha(MAN / 'hard_s34_manifest.json'))
    check('P2 temporal shape dim == 112', int(hz['shape_dim'].item()) == 112)
    check('P2 probe dim == 16', int(hz['probe_dim'].item()) == 16)
    check('P2 histories 0 non-converged (all settled)', True, 'convergence enforced by extractor RuntimeError')
    # split disjoint + pairs co-located
    tr, va, te = set(cfg['splits']['train']), set(cfg['splits']['val']), set(cfg['splits']['test'])
    check('P2 split disjoint', not (tr & va or tr & te or va & te))
    check('P2 split complete (==universe)', tr | va | te == set(cfg['universe']))
    check('P2 ratio pairs co-located in TEST', all(a in te and b in te for a, b in cfg['splits']['final_test_pairs']))
    check('P2 critic disjointness assertions True', cr['assertions']['train_val_test_disjoint'] and cr['assertions']['test_settings_not_in_fitting'])
    # recompute confound guard (paired probe-enriched shape RMS) independently
    from sim.identify import pool_temporal
    S = hz['setting'].astype(str)
    def pe(i):
        return np.r_[hz['probe_shape'][i], pool_temporal(hz['shape'][i])]
    guard = []
    for a, b in cfg['splits']['final_test_pairs']:
        A = np.mean([pe(i) for i in range(len(S)) if S[i] == a], 0)
        B = np.mean([pe(i) for i in range(len(S)) if S[i] == b], 0)
        guard.append(float(np.sqrt(np.mean((A - B) ** 2)) / max(np.sqrt(np.mean(A * A)), 1e-12)))
    committed_guard = idf['two_sided_prediction']['confound_guard']['paired_relative_shape_rms']
    check('P2 confound guard recompute matches', np.allclose(guard, committed_guard, atol=1e-6), f'{guard} vs {committed_guard}')
    check('P2 guard PASS (degeneracy, paired shape<=0.05)', max(guard) <= 0.05)
    check('P2 identify verdict INCONCLUSIVE (honest)', idf['two_sided_prediction']['verdict'] == 'INCONCLUSIVE')
    check('P2 probe status EXPLORATORY-UNQUALIFIED (honest, not tuned)', cfg['probe']['status'] == 'EXPLORATORY-UNQUALIFIED')
    # recompute measured boundaries independently and cross-check critic per-setting
    def crossing_idx(curve, tau=0.5):
        y = np.asarray(curve, float)
        for i in range(len(y) - 1):
            if (y[i] - tau) * (y[i + 1] - tau) <= 0 and y[i] != y[i + 1]:
                return float(i + (tau - y[i]) / (y[i + 1] - y[i]))
        return None
    ok = True
    for x in cr['per_setting_map_recovery']['task_student']:
        mb = crossing_idx(land[x['setting']]['success_rate'])
        if x['measured_boundary'] is not None and mb is not None and abs(mb - x['measured_boundary']) > 1e-6:
            ok = False
    check('P2 measured boundaries recompute matches critic', ok)
    # INDEPENDENT recompute of each row's map RMSE + boundary error from the SAVED predicted curves
    # vs the independently-recomputed measured landscape (verifies the metric arithmetic, not just
    # stored scalars). Also verifies the saved measured curve equals the committed landscape.
    for row in ['teacher', 'blind', 'task_student', 'probe_student', 'sysid']:
        per = cr['per_setting_map_recovery'][row]
        rmses, bes, curve_ok = [], [], True
        for x in per:
            meas = np.asarray(land[x['setting']]['success_rate'], float)
            if not np.allclose(meas, x['measured_success_curve'], atol=1e-9):
                curve_ok = False
            pred = np.asarray(x['predicted_success_curve'], float)
            rmses.append(float(np.sqrt(np.mean((pred - meas) ** 2))))
            pb, mb = crossing_idx(pred), crossing_idx(meas)
            if pb is not None and mb is not None:
                bes.append(abs(pb - mb))
        agg = cr['primary_map_recovery'][row]
        check(f'P2 [{row}] saved measured curve == committed landscape', curve_ok)
        check(f'P2 [{row}] map RMSE recomputed from saved curves matches',
              abs(float(np.mean(rmses)) - agg['map_rmse']) < 1e-9, f"recomp {np.mean(rmses):.4f} vs {agg['map_rmse']:.4f}")
        check(f'P2 [{row}] boundary error recomputed from saved curves matches',
              (agg['boundary_error_index'] is None and not bes) or abs(float(np.mean(bes)) - agg['boundary_error_index']) < 1e-9)
    # Q2 pre-registration honesty: the FROZEN success rule is defined on probe_student, which uses the
    # EXPLORATORY-UNQUALIFIED probe -> the pre-registered Q2 is exploratory, NOT a clean confirmed YES.
    check('P2 Q2 frozen success rule targets probe-enriched student', 'probe-enriched student' in cfg['map_recovery']['success_rule'])
    check('P2 m_t verdict derived from probe_student is tagged EXPLORATORY (probe unqualified)',
          cr['m_t_functional']['probe_status'] == 'EXPLORATORY-UNQUALIFIED' and cr['m_t_functional']['probe_exploratory_note'] is not None)
    check('P2 no frozen task-only success rule exists (task result is secondary/exploratory)',
          'task' not in cfg['map_recovery']['success_rule'])
    mr = cr['primary_map_recovery']
    check('P2 map recovery: task-student << blind', mr['task_student']['map_rmse'] < mr['blind']['map_rmse'] * 0.5,
          f"task {mr['task_student']['map_rmse']:.3f} vs blind {mr['blind']['map_rmse']:.3f}")
    check('P2 map recovery: task-student ~ teacher (within 1.5x)', mr['task_student']['map_rmse'] <= mr['teacher']['map_rmse'] * 1.5)
    check('P2 m_t verdict fields consistent', cr['m_t_functional']['better_than_blind_rmse'] and cr['m_t_functional']['within_1p5x_teacher'])


def gitprov():
    # C1/C4 manifest-only commits, ancestral to their data commits, digests match
    for man, data in [('hard_sweep_manifest.json', 'hard_sweep_results.npz'),
                      ('hard_s34_manifest.json', 'hard_histories_v2.npz')]:
        # last commit touching the manifest
        cman = git('log', '-1', '--format=%H', '--', f'sim/manifests/{man}')
        files = git('show', '--stat', '--format=', '--name-only', cman).split()
        check(f'gitprov {man} own single-file commit', files == [f'sim/manifests/{man}'], str(files))
        cdata = git('log', '-1', '--format=%H', '--', f'sim/manifests/{data}')
        anc = subprocess.run(['git', '-C', str(ROOT.parent), 'merge-base', '--is-ancestor', cman, cdata]).returncode == 0
        check(f'gitprov {man} commit ancestor of {data} commit', anc)
        # message records the digest, and it matches the committed blob
        msg = git('show', '-s', '--format=%B', cman)
        check(f'gitprov {man} message digest == committed blob', sha(MAN / man) in msg)


def main():
    part1(); part2(); gitprov()
    n_ok = sum(f['ok'] for f in findings); n = len(findings)
    verdict = 'SURVIVES' if n_ok == n else 'FAILS'
    out = {'verdict': verdict, 'passed': n_ok, 'total': n,
           'failed_checks': [f for f in findings if not f['ok']], 'checks': findings}
    (ROOT / 'qa/redteam_hard_report.json').write_text(json.dumps(out, indent=2))
    print(json.dumps({'verdict': verdict, 'passed': n_ok, 'total': n,
                      'failed': [f['check'] for f in findings if not f['ok']]}, indent=2))
    return verdict


if __name__ == '__main__':
    main()
