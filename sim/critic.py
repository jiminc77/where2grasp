"""Hardening Part-2 critic v2: five equal-budget rows + censor-aware map-recovery PRIMARY.

Rows (shared phi_theta:(B_eff,w)->z(4)->Q critic): privileged teacher / property-blind /
task-history student / probe-enriched student / leak-free explicit-sysID (probe-enriched).
Labels come from the selection bank (seeds 2000-2001); the measured evaluation-bank landscape
is opened oracle-only after all fitting. Splits are read from the manifest (unified leak-free).
PRIMARY metric = held-out map recovery (map RMSE, correlation, tau=0.5 boundary-index error)
with CENSOR-AWARE boundary scoring and a seeded PAIRED block-bootstrap over matched
evaluation-seed blocks reporting the student-blind and student/teacher contrasts directly.
"""
from __future__ import annotations
import argparse, hashlib, json, logging, random
from pathlib import Path
import numpy as np
import torch
from torch import nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sim.sweep import settings
from sim.identify import pool_temporal, cv_ridge

ROOT = Path(__file__).resolve().parent


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class Phi(nn.Module):
    def __init__(self):
        super().__init__(); self.net = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 4))
    def forward(self, x):
        return self.net(x)


class Critic(nn.Module):
    def __init__(self):
        super().__init__(); self.net = nn.Sequential(nn.Linear(6, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, g, z):
        return self.net(torch.cat((g, z), 1)).squeeze(1)


class Student(nn.Module):
    def __init__(self, d):
        super().__init__(); self.net = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 4))
    def forward(self, x):
        return self.net(x)


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def train_critic(rows, props, VAL, blind=False, train_ids=None, epochs=300, seed=3403):
    seed_all(seed); phi = Phi().double(); q = Critic().double()
    pars = list(q.parameters()) if blind else list(q.parameters()) + list(phi.parameters())
    opt = torch.optim.Adam(pars, lr=1e-2)
    def batch(ids):
        r = [x for x in rows if x[0] in ids]
        g = torch.tensor([x[1] for x in r], dtype=torch.double); y = torch.tensor([x[2] for x in r], dtype=torch.double)
        p = torch.tensor([props[x[0]] for x in r], dtype=torch.double); return g, p, y
    tg, tp, ty = batch(train_ids); vg, vp, vy = batch(VAL); best = (float('inf'), None, 0)
    for ep in range(1, epochs + 1):
        opt.zero_grad(); z = torch.zeros((len(tg), 4), dtype=torch.double) if blind else phi(tp)
        loss = nn.functional.binary_cross_entropy_with_logits(q(tg, z), ty); loss.backward(); opt.step()
        with torch.no_grad():
            vz = torch.zeros((len(vg), 4), dtype=torch.double) if blind else phi(vp)
            vl = float(nn.functional.binary_cross_entropy_with_logits(q(vg, vz), vy))
        if vl < best[0]:
            best = (vl, ({k: v.detach().clone() for k, v in phi.state_dict().items()},
                         {k: v.detach().clone() for k, v in q.state_dict().items()}), ep)
    phi.load_state_dict(best[1][0]); q.load_state_dict(best[1][1])
    return phi, q, {'best_epoch': best[2], 'best_val_bce': best[0]}


def crossing(curve, tau=0.5):
    """Censor-aware tau crossing: linearly interpolated grasp-index, or None if no crossing
    (all-above / all-below) -- NOT collapsed to index 0 or N-1."""
    y = np.asarray(curve, float)
    for i in range(len(y) - 1):
        if (y[i] - tau) * (y[i + 1] - tau) <= 0 and y[i] != y[i + 1]:
            return float(i + (tau - y[i]) / (y[i + 1] - y[i]))
    return None


def run(manifest, expected_digest=None):
    live = digest(manifest)
    if expected_digest and live != expected_digest:
        raise RuntimeError(f'manifest digest mismatch: {live} != {expected_digest}')
    cfg = json.loads(Path(manifest).read_text())
    (ROOT / 'logs').mkdir(exist_ok=True); (ROOT / 'figures').mkdir(exist_ok=True)
    logging.basicConfig(filename=ROOT / 'logs/hard_critic_train.log', filemode='w', level=logging.INFO, format='%(asctime)s %(message)s')
    sm = json.loads((ROOT / 'manifests' / cfg['sweep_manifest']).read_text()); ss = {x['id']: x for x in settings(sm)}
    ell = sm['grasp']['ell']; ng = len(ell)
    TRAIN = cfg['splits']['train']; VAL = cfg['splits']['val']; TEST = cfg['splits']['test']
    seed_all(cfg['models']['critic']['seed'])
    # train-only property scaler
    raw = {k: np.log10([v['B_eff'], v['w']]) for k, v in ss.items()}
    a = np.array([raw[k] for k in TRAIN]); pm, psd = a.mean(0), a.std(0); props = {k: (v - pm) / psd for k, v in raw.items()}
    # critic labels from selection bank (seeds 2000-2001), best template per (setting,grasp)
    sw = np.load(ROOT / 'manifests/hard_sweep_results.npz')
    keep = (sw['bank'] == 'selection') & np.isin(sw['seed'], [2000, 2001])
    rows = []
    for sid in TRAIN + VAL:
        for gi in range(ng):
            ix = keep & (sw['setting'] == sid) & (sw['grasp'] == gi)
            if not ix.any():
                continue
            cand = []
            for t in sorted(set(sw['template'][ix].tolist())):
                tx = ix & (sw['template'] == t); cand.append((float(sw['success'][tx].mean()), int(t), tx))
            _, _, tx = max(cand, key=lambda x: (x[0], -x[1]))
            rows.append((sid, np.array([gi / (ng - 1), float(sw['ell'][tx][0])]), float(sw['success'][tx].mean())))
    phi, q, ts = train_critic(rows, props, VAL, blind=False, train_ids=TRAIN, seed=cfg['models']['critic']['seed'])
    _, qb, bs = train_critic(rows, props, VAL, blind=True, train_ids=TRAIN, seed=cfg['models']['critic']['seed'])
    logging.info('teacher %s blind %s', ts, bs)
    # histories -> wrench-free student features
    hz = np.load(ROOT / 'manifests/hard_histories_v2.npz'); assert str(hz['manifest_digest'].item()) == live
    S = hz['setting'].astype(str)
    def task_feat(i):
        return np.r_[hz['proprio'][i], pool_temporal(hz['shape'][i])]
    def probe_feat(i):
        return np.r_[hz['proprio'][i], hz['probe_shape'][i], pool_temporal(hz['shape'][i])]
    def train_student(feat_fn, tag):
        X = np.array([feat_fn(i) for i in range(len(S))])
        ti = np.isin(S, TRAIN); vi = np.isin(S, VAL)
        hm = X[ti].mean(0); hs = X[ti].std(0); hs[hs < 1e-12] = 1
        Xtr = torch.tensor((X[ti] - hm) / hs, dtype=torch.double); Xv = torch.tensor((X[vi] - hm) / hs, dtype=torch.double)
        with torch.no_grad():
            Z = phi(torch.tensor([props[s] for s in S[ti]], dtype=torch.double)); VZ = phi(torch.tensor([props[s] for s in S[vi]], dtype=torch.double))
        seed_all(cfg['models']['critic']['seed']); st = Student(X.shape[1]).double(); opt = torch.optim.Adam(st.parameters(), lr=1e-2); best = (float('inf'), None)
        for _ in range(cfg['models']['distill_epochs']):
            opt.zero_grad(); loss = ((st(Xtr) - Z) ** 2).mean(); loss.backward(); opt.step()
            with torch.no_grad():
                vl = float(((st(Xv) - VZ) ** 2).mean())
            if vl < best[0]:
                best = (vl, {k: v.detach().clone() for k, v in st.state_dict().items()})
        st.load_state_dict(best[1]); logging.info('student[%s] val=%s', tag, best[0])
        return st, hm, hs
    st_task, tm, tstd = train_student(task_feat, 'task')
    st_probe, pmn, pstd = train_student(probe_feat, 'probe')
    # leak-free explicit sysID: ridge on TRAIN histories (probe-enriched), predict [logB,logw]
    ri = np.isin(S, TRAIN); rX = np.array([probe_feat(i) for i in range(len(S)) if ri[i]]); rS = S[ri]
    rY = np.array([np.log10([ss[s]['B_eff'], ss[s]['w'], ss[s]['B_eff'] / ss[s]['w']]) for s in rS])
    ridge, alpha, cvs, _ = cv_ridge(rX, rY, rS, cfg); logging.info('sysid ridge alpha=%s cv=%s', alpha, min(cvs))
    # oracle landscape (opened only now)
    land = {x['id']: x for x in json.loads((ROOT / 'manifests/hard_sweep_landscape.json').read_text())['settings']}
    gf = np.c_[np.arange(ng) / (ng - 1), np.asarray(ell)]
    def z_for(row, sid):
        idx = np.where(S == sid)[0]
        if row == 'teacher':
            with torch.no_grad():
                return phi(torch.tensor(props[sid][None], dtype=torch.double)).numpy()[0]
        if row == 'blind':
            return np.zeros(4)
        if row in ('task_student', 'probe_student'):
            st, hm, hs, ff = (st_task, tm, tstd, task_feat) if row == 'task_student' else (st_probe, pmn, pstd, probe_feat)
            X = np.array([ff(i) for i in idx])
            with torch.no_grad():
                return st(torch.tensor((X - hm) / hs, dtype=torch.double)).mean(0).numpy()
        # sysid
        X = np.array([probe_feat(i) for i in idx]); pred = ridge.predict(X).mean(0)
        p = (pred[:2] - pm) / psd
        with torch.no_grad():
            return phi(torch.tensor(p[None], dtype=torch.double)).numpy()[0]
    def predict_curve(row, sid):
        z = z_for(row, sid); qq = qb if row == 'blind' else q
        with torch.no_grad():
            return torch.sigmoid(qq(torch.tensor(gf, dtype=torch.double), torch.tensor(np.tile(z, (ng, 1)), dtype=torch.double))).numpy()
    ROWS = ['teacher', 'blind', 'task_student', 'probe_student', 'sysid']
    # per-TEST-setting map recovery vs measured landscape
    def map_metrics(row):
        per = []
        for sid in TEST:
            pred = predict_curve(row, sid); meas = np.asarray(land[sid]['success_rate'], float)
            corr = float(np.corrcoef(pred, meas)[0, 1]) if np.std(pred) > 1e-9 and np.std(meas) > 1e-9 else None
            pb, mb = crossing(pred), crossing(meas)
            be = None if (pb is None or mb is None) else abs(pb - mb)   # censored -> excluded
            per.append({'setting': sid, 'map_rmse': float(np.sqrt(np.mean((pred - meas) ** 2))),
                        'correlation': corr, 'predicted_boundary': pb, 'measured_boundary': mb,
                        'boundary_error_index': be, 'boundary_censored': bool(be is None)})
        rmses = [x['map_rmse'] for x in per]; corrs = [x['correlation'] for x in per if x['correlation'] is not None]
        bes = [x['boundary_error_index'] for x in per if x['boundary_error_index'] is not None]
        return {'map_rmse': float(np.mean(rmses)), 'correlation': float(np.mean(corrs)) if corrs else None,
                'boundary_error_index': float(np.mean(bes)) if bes else None,
                'boundary_censored_count': sum(x['boundary_censored'] for x in per), 'per_setting': per}
    metrics = {r: map_metrics(r) for r in ROWS}
    # seeded PAIRED block-bootstrap over matched evaluation-seed blocks -> student-blind & student/teacher contrasts
    eval_seeds = sorted(set(int(s) for s in sw['seed'][sw['bank'] == 'evaluation'].tolist()))
    succ = {}
    for sid in TEST:
        for gi in range(ng):
            vals = []
            for sd in eval_seeds:
                m = (sw['bank'] == 'evaluation') & (sw['setting'] == sid) & (sw['grasp'] == gi) & (sw['seed'] == sd)
                vals.append(float(sw['success'][m].mean()) if m.any() else np.nan)
            succ[(sid, gi)] = np.array(vals)
    preds = {r: {sid: predict_curve(r, sid) for sid in TEST} for r in ROWS}
    rng = np.random.default_rng(cfg['map_recovery']['bootstrap']['seed']); nb = cfg['map_recovery']['bootstrap']['n']
    boots = {r: {'rmse': [], 'be': []} for r in ROWS}
    for _ in range(nb):
        pick = rng.integers(0, len(eval_seeds), len(eval_seeds))
        for r in ROWS:
            rr, bb = [], []
            for sid in TEST:
                meas = np.array([float(np.nanmean(succ[(sid, gi)][pick])) for gi in range(ng)])
                pred = preds[r][sid]; rr.append(np.sqrt(np.mean((pred - meas) ** 2)))
                pb, mb = crossing(pred), crossing(meas)
                if pb is not None and mb is not None:
                    bb.append(abs(pb - mb))
            boots[r]['rmse'].append(float(np.mean(rr))); boots[r]['be'].append(float(np.mean(bb)) if bb else np.nan)
    def ci(arr):
        arr = np.array([x for x in arr if np.isfinite(x)]); return [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))] if len(arr) else [None, None]
    def contrast_ci(a, b):  # a - b, paired over bootstrap draws
        d = np.array(boots[a]['rmse']) - np.array(boots[b]['rmse'])
        d2 = np.array(boots[a]['be']) - np.array(boots[b]['be']); d2 = d2[np.isfinite(d2)]
        return {'rmse_delta_ci95': [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                'boundary_delta_ci95': ([float(np.percentile(d2, 2.5)), float(np.percentile(d2, 97.5))] if len(d2) else [None, None])}
    student_minus_blind = contrast_ci('probe_student', 'blind')
    student_over_teacher = float(metrics['probe_student']['map_rmse'] / max(metrics['teacher']['map_rmse'], 1e-9))
    # pre-registered success: probe-student BETTER than blind (delta CI < 0 on RMSE AND boundary) AND within 1.5x teacher
    better_rmse = student_minus_blind['rmse_delta_ci95'][1] is not None and student_minus_blind['rmse_delta_ci95'][1] < 0
    bd = student_minus_blind['boundary_delta_ci95']
    better_bound = bd[1] is not None and bd[1] < 0
    within_teacher = student_over_teacher <= 1.5
    m_t_functional = bool(better_rmse and better_bound and within_teacher)
    # adaptation: history composition task-only vs probe-prepended (map RMSE on TEST)
    adaptation = {'task_only_student_map_rmse': metrics['task_student']['map_rmse'],
                  'probe_prepended_student_map_rmse': metrics['probe_student']['map_rmse'],
                  'blind_map_rmse': metrics['blind']['map_rmse'], 'teacher_map_rmse': metrics['teacher']['map_rmse']}
    result = {'schema_version': 1, 'manifest_digest': live,
              'splits': cfg['splits'], 'assertions': {
                  'critic_input_dim': 6, 'latent_dim': 4, 'grasp_count': ng,
                  'train_val_test_disjoint': not (set(TRAIN) & set(VAL) or set(TRAIN) & set(TEST) or set(VAL) & set(TEST)),
                  'test_settings_not_in_fitting': not (set(TEST) & (set(TRAIN) | set(VAL))),
                  'sysid_wrench_free': True, 'sysid_and_probe_student_same_features': True},
              'primary_map_recovery': {r: {k: metrics[r][k] for k in ('map_rmse', 'correlation', 'boundary_error_index', 'boundary_censored_count')} for r in ROWS},
              'map_recovery_bootstrap_ci95': {r: {'map_rmse': ci(boots[r]['rmse']), 'boundary_error_index': ci(boots[r]['be'])} for r in ROWS},
              'contrasts': {'probe_student_minus_blind': student_minus_blind, 'probe_student_over_teacher_rmse': student_over_teacher},
              'm_t_functional': {'verdict': m_t_functional, 'better_than_blind_rmse': bool(better_rmse),
                                 'better_than_blind_boundary': bool(better_bound), 'within_1p5x_teacher': bool(within_teacher),
                                 'rule': cfg['map_recovery']['success_rule'],
                                 'probe_status': cfg['probe']['status'],
                                 'probe_exploratory_note': (None if cfg['probe']['status'] == 'QUALIFIED' else
                                     'The probe_student and leak-free-sysID rows use an UNQUALIFIED probe (qualification '
                                     'INCONCLUSIVE at interval 0.01); the m_t_functional verdict via the probe path is '
                                     'EXPLORATORY, not a clean preregistered result. teacher/blind/task_student are unaffected.')},
              'adaptation_history_composition': adaptation,
              'sysid_ridge': {'alpha': alpha, 'mean_grouped_cv_rmse': float(min(cvs))},
              'per_setting_map_recovery': {r: metrics[r]['per_setting'] for r in ROWS},
              'notes': {'primary': 'held-out TEST map recovery; argmax is material-independent so selection regret is a known null',
                        'oracle': 'evaluation-bank landscape opened only after fitting',
                        'censor_aware': 'no tau crossing => boundary censored (excluded), never index 0/N-1'}}
    (ROOT / 'manifests/hard_critic_results_v2.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    # figures
    fig, ax = plt.subplots(1, 2, figsize=(10, 4)); labels = ROWS
    ax[0].bar(labels, [metrics[r]['map_rmse'] for r in labels]); ax[0].set_ylabel('map RMSE (held-out TEST)'); ax[0].tick_params(axis='x', rotation=20)
    ax[1].bar(labels, [metrics[r]['boundary_error_index'] or 0 for r in labels]); ax[1].set_ylabel('tau=0.5 boundary error (index)'); ax[1].tick_params(axis='x', rotation=20)
    fig.suptitle('Critic v2 map recovery (PRIMARY, censor-aware)'); fig.tight_layout(); fig.savefig(ROOT / 'figures/hard_map_recovery_by_row.png', dpi=170); plt.close(fig)
    fig, ax = plt.subplots(figsize=(5.5, 4)); comp = ['blind', 'task_student', 'probe_student', 'teacher']
    ax.plot(range(len(comp)), [metrics[r]['map_rmse'] for r in comp], 'o-'); ax.set_xticks(range(len(comp))); ax.set_xticklabels(comp, rotation=15)
    ax.set_ylabel('map RMSE'); ax.set_title('Adaptation by history composition'); fig.tight_layout()
    fig.savefig(ROOT / 'figures/hard_adaptation_curve.png', dpi=170); plt.close(fig)
    print(json.dumps({'manifest_digest': live, 'm_t_functional': m_t_functional,
                      'map_rmse': {r: round(metrics[r]['map_rmse'], 4) for r in ROWS},
                      'probe_student_minus_blind': student_minus_blind, 'probe_student_over_teacher': round(student_over_teacher, 3)}, indent=2))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--manifest', default=str(ROOT / 'manifests/hard_s34_manifest.json'))
    p.add_argument('--expected-digest', default=None); a = p.parse_args(); run(a.manifest, a.expected_digest)
