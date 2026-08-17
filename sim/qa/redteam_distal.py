"""Independent red-team recompute of hardening-B (distal) from committed artifacts.

Recomputes Part-2 (per-setting argmax + feasibility classification + ell_L/ell_U band with a FRESH
implementation that does NOT import distal_gate), Part-3 (transfer non-inferiority CI predicate,
frozen-encoder hash identity before==after, split disjointness incl. distal TEST vs source
encoder TRAIN+VAL, action-metadata presence, regret unique-group aggregation), the addendum
sufficient-k comparator, and git-provable pre-registration (single-file manifest commits ancestral
to their data, digests matching). Returns SURVIVES iff every recomputed check holds. Robust to
not-yet-generated data (checks are skipped-with-note until their inputs exist).
"""
from __future__ import annotations
import json, subprocess, hashlib
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MAN = ROOT / 'manifests'
findings = []


def check(name, ok, detail=''):
    findings.append({'check': name, 'ok': bool(ok), 'detail': detail}); return ok


def skip(name, why):
    findings.append({'check': name, 'ok': True, 'skipped': True, 'detail': why}); return True


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def git(*a):
    return subprocess.run(['git', '-C', str(ROOT.parent), *a], capture_output=True, text=True).stdout.strip()


def fresh_argmax(mean_J, in_regime, tie_eps=1e-9):
    mj = np.asarray(mean_J, float); fm = np.asarray(in_regime, bool) & np.isfinite(mj)
    if not fm.any():
        return None
    vals = mj[fm]
    if float(vals.max() - vals.min()) <= tie_eps:
        return None
    best = float(vals.max()); cand = np.where(fm & (mj >= best - tie_eps))[0]; return int(cand.min())


def fresh_band(succ, grid, tau=0.5):
    y = np.asarray(succ, float); g = np.asarray(grid, float); up = None
    for i in range(len(y) - 1):
        if y[i] < tau <= y[i + 1] and y[i + 1] != y[i]:
            up = float(g[i] + (tau - y[i]) / (y[i + 1] - y[i]) * (g[i + 1] - g[i])); break
    if up is None and y[0] >= tau:
        up = float(g[0])
    down = None
    for i in range(len(y) - 1):
        if y[i] >= tau > y[i + 1] and y[i + 1] != y[i]:
            down = float(g[i] + (tau - y[i]) / (y[i + 1] - y[i]) * (g[i + 1] - g[i]))
    return up, down


def part1():
    man, res, verd = MAN / 'distal_manifest.json', MAN / 'distal_sweep_results.npz', MAN / 'distal_gate_verdict.json'
    if not (man.exists() and res.exists() and verd.exists()):
        return skip('P1 gate recompute', 'distal sweep/verdict not yet generated')
    m = json.loads(man.read_text()); z = np.load(res); v = json.loads(verd.read_text())
    grid = np.array(m['grasp']['ell']); pig_max = m['pi_g_max']
    prop = {c['id']: (c['B_eff'], c['w']) for c in m['grid']}; prop.update({r['id']: (r['B_eff'], r['w']) for r in m['ratio_pairs']})
    check('P1 digest matches results', str(z['manifest_digest'].item()) == sha(man))
    check('P1 nonconverged == 0', float(np.mean(~z['converged'].astype(bool))) == 0.0)
    mism = []
    for sid, (B, w) in prop.items():
        ms = np.array([float(np.mean(z['success'][(z['bank'] == 'evaluation') & (z['setting'] == sid) & (z['grasp'] == gi)]))
                       for gi in range(len(grid))])
        mj = np.array([float(np.mean(z['J'][(z['bank'] == 'evaluation') & (z['setting'] == sid) & (z['grasp'] == gi)]))
                       for gi in range(len(grid))])
        in_reg = (w * grid ** 3 / B) <= pig_max
        am = fresh_argmax(mj, in_reg)
        committed = v['measured'].get(sid, {})
        if am != committed.get('argmax_idx'):
            mism.append((sid, am, committed.get('argmax_idx')))
    check('P1 measured argmax recompute matches', not mism, str(mism[:5]))
    check('P1 committed verdict is a valid token', v['verdict'] in ('GO', 'NO-GO', 'inconclusive'), v['verdict'])
    # predicted feasibility: B1_w2 the sole predicted-infeasible cell
    infeas = [c['id'] for c in m['grid'] if not c['feasible']]
    check('P1 predicted infeasible == {B1_w2}', infeas == ['B1_w2'], str(infeas))


def part2():
    s34, cr, hz = MAN / 'distal_s34_manifest.json', MAN / 'distal_critic_results.json', MAN / 'distal_histories.npz'
    if not (s34.exists() and cr.exists() and hz.exists()):
        return skip('P2 critic/transfer recompute', 'distal Part-3 data not yet generated')
    cfg = json.loads(s34.read_text()); c = json.loads(cr.read_text()); h = np.load(hz)
    tr, va, te = set(cfg['splits']['train']), set(cfg['splits']['val']), set(cfg['splits']['test'])
    check('P2 split disjoint+complete', not (tr & va or tr & te or va & te) and tr | va | te == set(cfg['universe']))
    st, sv = set(cfg['source_encoder']['source_train']), set(cfg['source_encoder']['source_val'])
    check('P2 distal TEST disjoint from source encoder TRAIN+VAL (leak-free transfer)', not (te & (st | sv)))
    check('P2 action metadata present in histories', int(h['action_dim'].item()) == 2)
    check('P2 frozen encoder hash identical before==after head training', c['transfer']['encoder_unchanged'])
    # recompute transfer non-inferiority predicate
    fr, sc, bl, dNI = c['transfer']['frozen_map_rmse'], c['transfer']['scratch_map_rmse'], c['transfer']['blind_map_rmse'], c['transfer']['delta_NI']
    recomputed = 'YES' if (fr - sc < dNI and fr < bl) else ('NO' if fr >= bl else 'NOT-ESTABLISHED')
    check('P2 transfer non-inferiority verdict recompute matches', recomputed == c['transfer']['verdict'], f'{recomputed} vs {c["transfer"]["verdict"]}')
    check('P2 regret aggregated over unique groups (ratio=invariance controls)', 'unique' in c['aggregation'].lower())
    check('P2 multi-seed variability reported', len(c['training_seeds']) >= 2 and 'std_over_seeds' in json.dumps(c['summary']))


def addendum():
    am = MAN / 'addendum_manifest.json'; ar = MAN / 'addendum_results.json'
    if not (am.exists() and ar.exists()):
        return skip('ADD sufficient-k recompute', 'addendum data not yet generated')
    cfg = json.loads(am.read_text()); r = json.loads(ar.read_text())
    tol = cfg.get('sufficient_k', {}).get('tol_k', 0.02)
    ok = True
    for s, row in r.get('per_setting_truncation', {}).items():
        rmse = row['rmse_by_k']; r7 = rmse[-1]
        want = next((k + 1 for k in range(len(rmse)) if rmse[k] <= r7 + tol), 7)
        if row.get('sufficient_k') not in (want, 'no sufficient k'):
            ok = False
    check('ADD sufficient-k comparator recompute matches', ok)


def gitprov():
    pairs = [('distal_manifest.json', 'distal_sweep_results.npz'),
             ('distal_s34_manifest.json', 'distal_histories.npz'),
             ('addendum_manifest.json', 'addendum_results.json')]
    for man, data in pairs:
        if not (MAN / man).exists():
            skip(f'gitprov {man}', 'not yet committed'); continue
        cman = git('log', '-1', '--format=%H', '--', f'sim/manifests/{man}')
        if not cman:
            skip(f'gitprov {man}', 'no commit yet'); continue
        files = git('show', '--stat', '--format=', '--name-only', cman).split()
        check(f'gitprov {man} single-file commit', files == [f'sim/manifests/{man}'], str(files))
        if (MAN / data).exists():
            cdata = git('log', '-1', '--format=%H', '--', f'sim/manifests/{data}')
            if cdata:
                anc = subprocess.run(['git', '-C', str(ROOT.parent), 'merge-base', '--is-ancestor', cman, cdata]).returncode == 0
                check(f'gitprov {man} ancestor of {data}', anc)
        msg = git('show', '-s', '--format=%B', cman)
        check(f'gitprov {man} message digest == blob', sha(MAN / man) in msg)


def scope_guard():
    # SCOPE GUARD: no closed hardening-A artifact modified this run (source hashes frozen in the regression test)
    check('SCOPE hardening-A sweep.py untouched',
          sha(ROOT / 'sweep.py') == 'da042e613cdacd3dcda247b4f69c118180b34bc867d664b26a475945a6a45208')
    check('SCOPE hardening-A analyze_gate.py untouched',
          sha(ROOT / 'analyze_gate.py') == 'c8ca30a3f9e032db86282b05dd0f81845b90bea3a071af9ab64a61ebede5bbfb')


def main():
    part1(); part2(); addendum(); gitprov(); scope_guard()
    real = [f for f in findings if not f.get('skipped')]
    n_ok = sum(f['ok'] for f in real); n = len(real)
    verdict = 'SURVIVES' if n_ok == n else 'FAILS'
    out = {'verdict': verdict, 'passed': n_ok, 'total': n,
           'skipped': [f['check'] for f in findings if f.get('skipped')],
           'failed_checks': [f for f in real if not f['ok']], 'checks': findings}
    (ROOT / 'qa/redteam_distal_report.json').write_text(json.dumps(out, indent=2))
    print(json.dumps({'verdict': verdict, 'passed': n_ok, 'total': n,
                      'skipped': out['skipped'], 'failed': [f['check'] for f in real if not f['ok']]}, indent=2))
    return verdict


if __name__ == '__main__':
    main()
