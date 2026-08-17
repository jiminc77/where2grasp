"""Frozen distal-tip-placement GATE decision function (its OWN pure surface).

Does NOT import sim.analyze_gate (the byte-frozen lift-and-clear boundary()/condition() surface
is untouched; the distal 0->1->0 landscape needs different semantics). The pure core `decide()`
takes per-setting measured curves + the frozen manifest predictions and returns the GO/NO-GO/
INCONCLUSIVE verdict; it is golden-hashable and Genesis/npz-free. `main()` loads the sweep npz +
manifest and calls decide, then writes distal_gate_verdict.json + a LABELED eval-draw bootstrap.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import sim.tip_model as tm

ROOT = Path(__file__).resolve().parent
MAN = ROOT / 'manifests'


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def decide(per_setting, frozen, grid):
    """Pure frozen decision. per_setting[id] = {mean_success:[...], mean_J:[...], B_eff, w}.
    frozen = manifest dict (predicted argmax/feasible/ell_star/ell_L/ell_U + eligible contrasts + ratio pairs).
    Returns the full verdict dict."""
    grid = np.asarray(grid, float)
    tau = frozen['objective']['tau']
    step = frozen['grasp']['step']
    pred = {c['id']: c for c in frozen['grid']}
    ratio_ref = {r['id']: r['reference'] for r in frozen['ratio_pairs']}

    measured = {}
    for sid, rec in per_setting.items():
        B = float(rec['B_eff']); w = float(rec['w'])
        ms = np.asarray(rec['mean_success'], float); mj = np.asarray(rec['mean_J'], float)
        pig = tm.pi_g(grid, B, w)
        in_regime = pig <= tm.PI_G_MAX
        feasible_grasp = in_regime & (ms >= tau)
        cell_feasible = bool(feasible_grasp.any())
        am = tm.measured_argmax(mj, in_regime & np.isfinite(mj), grid)
        lL, lU, cens = tm.band_crossings(ms, grid, tau)
        # conservative per-estimand regime flags on the MEASURED estimands
        reg_star = tm.regime_in(float(grid[am['argmax_idx']]) if am['argmax_idx'] is not None else None, B, w, grid)
        reg_L = tm.regime_in(lL, B, w, grid)
        reg_U = tm.regime_in(lU, B, w, grid)
        measured[sid] = dict(cell_feasible=cell_feasible, argmax_idx=am['argmax_idx'],
                             argmax_ell=(float(grid[am['argmax_idx']]) if am['argmax_idx'] is not None else None),
                             clipped=am['clipped'], censored=am['censored'],
                             ell_L=lL, ell_U=lU, band_censor=cens,
                             regime_star=reg_star, regime_L=reg_L, regime_U=reg_U,
                             mean_success=[float(x) for x in ms], mean_J=[float(x) for x in mj])

    fails, inconclusive = [], []
    # feasibility classification: predicted vs measured contradiction => FAIL
    feas_report = []
    for sid, p in pred.items():
        if sid not in measured:
            continue
        pm = bool(p['feasible']); mm = measured[sid]['cell_feasible']
        feas_report.append(dict(id=sid, predicted=pm, measured=mm, match=bool(pm == mm)))
        if pm != mm:
            fails.append(f'feasibility contradiction {sid}: predicted={pm} measured={mm}')

    # argmax-direction on eligible feasible reach-slack non-clipped in-regime contrasts
    contrasts = frozen['eligible_adjacency_contrasts']
    dir_report = []
    for con in contrasts:
        a, b, kind = con['a'], con['b'], con['kind']
        ma, mb = measured.get(a), measured.get(b)
        if ma is None or mb is None or ma['argmax_idx'] is None or mb['argmax_idx'] is None:
            inconclusive.append(f'{a}->{b}: missing/censored measured argmax'); dir_report.append(dict(**con, status='INCONCLUSIVE')); continue
        if ma['clipped'] or mb['clipped']:
            inconclusive.append(f'{a}->{b}: clipped measured argmax'); dir_report.append(dict(**con, status='INCONCLUSIVE')); continue
        if not (ma['regime_star'] and mb['regime_star']):
            inconclusive.append(f'{a}->{b}: out-of-regime measured optimum'); dir_report.append(dict(**con, status='INCONCLUSIVE')); continue
        shift_cells = mb['argmax_idx'] - ma['argmax_idx']   # b relative to a
        # kind B_up (a=lower B): expect argmax to RISE (shift>0); kind w_down (a=lower w): expect FALL (shift<0)
        signed = shift_cells if kind == 'B_up' else -shift_cells
        ok = bool(signed >= tm.MIN_SHIFT_CELLS)
        rec = dict(**con, measured_shift_cells=int(shift_cells), signed_shift=int(signed), meets_min=ok,
                   status='PASS' if ok else 'FAIL')
        dir_report.append(rec)
        if not ok:
            if signed < 0:
                fails.append(f'{a}->{b}: wrong-signed argmax shift {signed}')
            else:
                fails.append(f'{a}->{b}: argmax shift {signed} < {tm.MIN_SHIFT_CELLS} cells')

    # fixed-ratio invariance (invariance controls): |argmax(R) - argmax(ref)| <= ratio tolerance
    ratio_tol = frozen['gate_truth_table']['ratio_tolerance_cells']
    ratio_report = []
    for rid, ref in ratio_ref.items():
        mr, mref = measured.get(rid), measured.get(ref)
        if mr is None or mref is None or mr['argmax_idx'] is None or mref['argmax_idx'] is None:
            inconclusive.append(f'ratio {rid}: missing measured argmax'); ratio_report.append(dict(id=rid, ref=ref, status='INCONCLUSIVE')); continue
        off = abs(mr['argmax_idx'] - mref['argmax_idx'])
        ok = bool(off <= ratio_tol)
        ratio_report.append(dict(id=rid, ref=ref, offset_cells=int(off), within_tol=ok, status='PASS' if ok else 'FAIL'))
        if not ok:
            fails.append(f'ratio {rid}: argmax offset {off} cells > {ratio_tol}')

    # predicted-vs-measured optimum + two-sided band recovery (co-primaries; regime-flagged exclusions)
    recovery = []
    for sid, p in pred.items():
        if sid not in measured or not p['feasible']:
            continue
        m = measured[sid]
        argmin_err_cells = (abs(m['argmax_idx'] - p['argmax_idx']) if m['argmax_idx'] is not None and p['argmax_idx'] is not None else None)
        # band recovery only where BOTH edges present AND in-regime
        pL, pU = p['ell_L'], p['ell_U']
        use_U = bool(p['regime_U'] and m['regime_U'] and m['ell_U'] is not None)
        iou = tm.interval_iou(pL, pU if use_U else None, m['ell_L'], m['ell_U'] if use_U else None) if use_U else None
        recovery.append(dict(id=sid, pred_argmax=p['argmax_idx'], meas_argmax=m['argmax_idx'],
                             argmin_error_cells=argmin_err_cells,
                             pred_ell_L=pL, meas_ell_L=m['ell_L'], dL=(abs(pL - m['ell_L']) if m['ell_L'] is not None else None),
                             pred_ell_U=pU, meas_ell_U=m['ell_U'], ell_U_in_regime=use_U,
                             iou=iou, hausdorff=(tm.hausdorff(pL, pU, m['ell_L'], m['ell_U']) if use_U else None)))

    status = 'NO-GO' if fails else ('GO' if not inconclusive else 'inconclusive')
    return dict(verdict=status, fails=fails, inconclusive=inconclusive,
                feasibility=feas_report, argmax_direction=dir_report, ratio_invariance=ratio_report,
                recovery=recovery, measured=measured, tau=tau, step=step,
                min_shift_cells=tm.MIN_SHIFT_CELLS)


def _bootstrap_argmax(sw, per_setting, grid, tau, n=2000, seed=7):
    """LABELED eval-draw block-bootstrap of measured argmax/ell_L/ell_U per feasible setting."""
    eval_seeds = sorted(set(int(s) for s in sw['seed'][sw['bank'] == 'evaluation'].tolist()))
    ns = len(eval_seeds); rng = np.random.default_rng(seed); grid = np.asarray(grid, float)
    out = {}
    for sid, rec in per_setting.items():
        B = float(rec['B_eff']); w = float(rec['w']); in_regime = tm.pi_g(grid, B, w) <= tm.PI_G_MAX
        # per (grasp, eval seed) success + J
        succ = np.full((len(grid), ns), np.nan); jj = np.full((len(grid), ns), np.nan)
        for gi in range(len(grid)):
            for si, sd in enumerate(eval_seeds):
                q = (sw['bank'] == 'evaluation') & (sw['setting'] == sid) & (sw['grasp'] == gi) & (sw['seed'] == sd)
                if q.any():
                    succ[gi, si] = float(np.mean(sw['success'][q])); jj[gi, si] = float(np.mean(sw['J'][q]))
        am_b, lL_b, lU_b = [], [], []
        for _ in range(n):
            pick = rng.integers(0, ns, ns)
            ms = np.nanmean(succ[:, pick], axis=1); mj = np.nanmean(jj[:, pick], axis=1)
            a = tm.measured_argmax(mj, in_regime & np.isfinite(mj), grid)
            if a['argmax_idx'] is not None:
                am_b.append(float(grid[a['argmax_idx']]))
            lL, lU, _ = tm.band_crossings(ms, grid, tau)
            if lL is not None:
                lL_b.append(lL)
            if lU is not None:
                lU_b.append(lU)
        def ci(v):
            return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v else [None, None]
        out[sid] = dict(argmax_ci95=ci(am_b), ell_L_ci95=ci(lL_b), ell_U_ci95=ci(lU_b),
                        note='LABELED eval-draw block-bootstrap; conditional on the fixed sweep (not training/object uncertainty)')
    return out


def main(manifest=None, results=None, out=None):
    manifest = Path(manifest) if manifest else (MAN / 'distal_manifest.json')
    results = Path(results) if results else (MAN / 'distal_sweep_results.npz')
    frozen = json.loads(manifest.read_text()); grid = np.asarray(frozen['grasp']['ell'], float)
    sw = np.load(results)
    ids = [c['id'] for c in frozen['grid']] + [r['id'] for r in frozen['ratio_pairs']]
    prop = {c['id']: (c['B_eff'], c['w']) for c in frozen['grid']}
    prop.update({r['id']: (r['B_eff'], r['w']) for r in frozen['ratio_pairs']})
    per_setting = {}
    for sid in ids:
        ms, mj = [], []
        for gi in range(len(grid)):
            q = (sw['bank'] == 'evaluation') & (sw['setting'] == sid) & (sw['grasp'] == gi)
            ms.append(float(np.mean(sw['success'][q])) if q.any() else 0.0)
            mj.append(float(np.mean(sw['J'][q])) if q.any() else tm.J_INF)
        per_setting[sid] = dict(mean_success=ms, mean_J=mj, B_eff=prop[sid][0], w=prop[sid][1])
    verdict = decide(per_setting, frozen, grid)
    verdict['manifest_digest'] = sha256(manifest)
    verdict['nonconverged_fraction'] = float(np.mean(~sw['converged'].astype(bool))) if 'converged' in sw else None
    verdict['bootstrap'] = _bootstrap_argmax(sw, per_setting, grid, frozen['objective']['tau'])
    out = Path(out) if out else (MAN / 'distal_gate_verdict.json')
    out.write_text(json.dumps(verdict, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'verdict': verdict['verdict'], 'fails': verdict['fails'][:5],
                      'inconclusive': verdict['inconclusive'][:5],
                      'nonconverged_fraction': verdict['nonconverged_fraction']}, indent=2))
    return verdict


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--manifest', default=None)
    p.add_argument('--results', default=None); p.add_argument('--out', default=None)
    a = p.parse_args(); main(a.manifest, a.results, a.out)
