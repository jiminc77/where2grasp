"""Distal Part-3 critic: 5 equal-budget rows + selection-regret PRIMARY + band co-primary + transfer.

Shared phi_theta:(B,w)->z(4) feeds a TWO-HEAD critic (frozen manifest allows the optional success
head for feasibility/map): a SUCCESS head (BCE on the measured 0->1->0 success-probability curve,
the clean informative map like hardening-A) and a J head (MSE on the within-band J, feasible-masked
so the J_inf=-1.0 cliffs do not dominate) that defines SELECTION. Rows: teacher (privileged B,w) /
blind / task-history student / leak-free sysID(ridge) / FROZEN-cross-task-transfer student (source
lift-and-clear encoder frozen, head-only) + a from-scratch encoder on equal budget.

PRIMARY = selection regret = max_g E[J|s,g] - E[J|s,g_selected] on winner-only evaluation, aggregated
over UNIQUE (B,w) TEST groups (ratio pairs = invariance controls, A-15). Co-primary = map RMSE +
ell_L/ell_U band recovery (first-up/last-down; censored/regime edges excluded; IoU) on the predicted
success curve. Transfer = non-inferiority (delta_NI=0.05, lower-is-better). Multiple training seeds
report seed variability; the eval-draw contrast is LABELED conditional-on-models (A-16).
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
from sim.identify import pool_temporal
import sim.tip_model as tm
from sim.export_encoder import load_frozen_encoder

ROOT = Path(__file__).resolve().parent
MAN = ROOT / 'manifests'


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


class Phi(nn.Module):
    def __init__(self):
        super().__init__(); self.net = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 4))
    def forward(self, x):
        return self.net(x)


class TwoHead(nn.Module):
    """(grasp feature g(2), latent z(4)) -> trunk -> [success logit, within-band J]."""
    def __init__(self):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(6, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU())
        self.succ = nn.Linear(32, 1); self.jval = nn.Linear(32, 1)
    def forward(self, g, z):
        h = self.trunk(torch.cat((g, z), 1)); return self.succ(h).squeeze(1), self.jval(h).squeeze(1)


def train_critic(rows, props, VAL, blind, train_ids, epochs=300, seed=3403):
    """rows: (setting, g_feat(2), succ_prob, J, feasible_mask). BCE on success + MSE on feasible J."""
    seed_all(seed); phi = Phi().double(); q = TwoHead().double()
    pars = list(q.parameters()) if blind else list(q.parameters()) + list(phi.parameters())
    opt = torch.optim.Adam(pars, lr=1e-2)
    def batch(ids):
        r = [x for x in rows if x[0] in ids]
        g = torch.tensor(np.array([x[1] for x in r]), dtype=torch.double)
        s = torch.tensor([x[2] for x in r], dtype=torch.double)
        j = torch.tensor([x[3] for x in r], dtype=torch.double)
        fmask = torch.tensor([x[4] for x in r], dtype=torch.double)
        p = torch.tensor(np.array([props[x[0]] for x in r]), dtype=torch.double); return g, p, s, j, fmask
    tg, tp, ts, tj, tf = batch(train_ids); vg, vp, vs, vj, vf = batch(VAL); best = (float('inf'), None)
    # class-imbalance weight: the distal success is a NARROW 0->1->0 band (~17% feasible), so upweight
    # the minority feasible class or BCE collapses to predicting all-infeasible.
    pw = torch.tensor(float((len(ts) - ts.sum()) / (ts.sum() + 1e-9)), dtype=torch.double)
    for _ in range(epochs):
        opt.zero_grad(); z = torch.zeros((len(tg), 4), dtype=torch.double) if blind else phi(tp)
        slog, jval = q(tg, z)
        loss = nn.functional.binary_cross_entropy_with_logits(slog, ts, pos_weight=pw) + (tf * (jval - tj) ** 2).sum() / (tf.sum() + 1e-9)
        loss.backward(); opt.step()
        with torch.no_grad():
            vz = torch.zeros((len(vg), 4), dtype=torch.double) if blind else phi(vp)
            vsl, vjv = q(vg, vz)
            vl = float(nn.functional.binary_cross_entropy_with_logits(vsl, vs, pos_weight=pw) + (vf * (vjv - vj) ** 2).sum() / (vf.sum() + 1e-9))
        if vl < best[0]:
            best = (vl, ({k: v.detach().clone() for k, v in phi.state_dict().items()},
                         {k: v.detach().clone() for k, v in q.state_dict().items()}))
    phi.load_state_dict(best[1][0]); q.load_state_dict(best[1][1])
    return phi, q


def _labels(sw, ids, grid, prop, bank_mask):
    """Per (setting,grasp) winner-template success prob + J + feasible mask from the given bank rows."""
    rows = []
    for sid in ids:
        for gi in range(len(grid)):
            ix = bank_mask & (sw['setting'] == sid) & (sw['grasp'] == gi)
            if not ix.any():
                continue
            cand = [(float(sw['success'][ix & (sw['template'] == t)].mean()),
                     float(sw['J'][ix & (sw['template'] == t)].mean()), int(t)) for t in sorted(set(sw['template'][ix].tolist()))]
            sp, jj, _ = max(cand, key=lambda x: (x[0], x[1], -x[2]))
            B, w = prop[sid]; feas = 1.0 if jj > tm.J_INF + 1e-6 else 0.0
            rows.append((sid, np.array([gi / (len(grid) - 1), float(grid[gi])]), sp, jj, feas))
    return rows


def _measured(sw, ids, grid, prop):
    out = {}
    for sid in ids:
        ms, mj = [], []
        for gi in range(len(grid)):
            m = (sw['bank'] == 'evaluation') & (sw['setting'] == sid) & (sw['grasp'] == gi)
            ms.append(float(np.mean(sw['success'][m])) if m.any() else 0.0)
            mj.append(float(np.mean(sw['J'][m])) if m.any() else tm.J_INF)
        out[sid] = dict(mean_success=np.array(ms), mean_J=np.array(mj), B_eff=prop[sid][0], w=prop[sid][1])
    return out


def run(manifest=None, expected_digest=None):
    manifest = Path(manifest) if manifest else (MAN / 'distal_s34_manifest.json')
    live = digest(manifest)
    if expected_digest and live != expected_digest:
        raise RuntimeError(f'manifest digest mismatch: {live} != {expected_digest}')
    cfg = json.loads(manifest.read_text()); dm = json.loads((MAN / cfg['distal_manifest']).read_text())
    (ROOT / 'logs').mkdir(exist_ok=True); (ROOT / 'figures').mkdir(exist_ok=True)
    logging.basicConfig(filename=ROOT / 'logs/distal_critic_train.log', filemode='w', level=logging.INFO, format='%(asctime)s %(message)s')
    grid = np.array(dm['grasp']['ell']); ng = len(grid)
    TRAIN = cfg['splits']['train']; VAL = cfg['splits']['val']; TEST = cfg['splits']['test']
    prop = {c['id']: (c['B_eff'], c['w']) for c in dm['grid']}; prop.update({r['id']: (r['B_eff'], r['w']) for r in dm['ratio_pairs']})
    raw = {k: np.log10([v[0], v[1]]) for k, v in prop.items()}
    a = np.array([raw[k] for k in TRAIN]); pm, psd = a.mean(0), a.std(0); props = {k: (v - pm) / psd for k, v in raw.items()}
    sw = np.load(MAN / 'distal_sweep_results.npz')
    meas = _measured(sw, list(prop), grid, prop)
    rows = _labels(sw, TRAIN + VAL, grid, prop, sw['bank'] == 'selection')
    gf = np.c_[np.arange(ng) / (ng - 1), grid]

    hz = np.load(MAN / 'distal_histories.npz'); assert str(hz['manifest_digest'].item()) == live
    S = hz['setting'].astype(str)
    def task_feat(i):
        return np.r_[hz['proprio'][i], pool_temporal(hz['shape'][i]), hz['action'][i]]      # 42-D
    def enc_feat(i):
        return np.r_[hz['proprio'][i], pool_temporal(hz['shape'][i])]                        # 40-D (source contract)

    def distil_student(phi, feat_fn, seed, encoder=None, enc_mean=None, enc_std=None, head_only=False):
        X = np.array([feat_fn(i) for i in range(len(S))]); ti = np.isin(S, TRAIN); vi = np.isin(S, VAL)
        if head_only:
            with torch.no_grad():
                feat = encoder(torch.tensor((X - enc_mean) / enc_std, dtype=torch.double)).numpy()
        else:
            hm = X[ti].mean(0); hs = X[ti].std(0); hs[hs < 1e-12] = 1; feat = (X - hm) / hs
        d = feat.shape[1]
        Xtr = torch.tensor(feat[ti], dtype=torch.double); Xv = torch.tensor(feat[vi], dtype=torch.double)
        with torch.no_grad():
            Z = phi(torch.tensor(np.array([props[s] for s in S[ti]]), dtype=torch.double))
            VZ = phi(torch.tensor(np.array([props[s] for s in S[vi]]), dtype=torch.double))
        seed_all(seed)
        head = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 4)).double()
        opt = torch.optim.Adam(head.parameters(), lr=1e-2); best = (float('inf'), None)
        for _ in range(cfg['critic_head']['epochs']):
            opt.zero_grad(); loss = ((head(Xtr) - Z) ** 2).mean(); loss.backward(); opt.step()
            with torch.no_grad():
                vl = float(((head(Xv) - VZ) ** 2).mean())
            if vl < best[0]:
                best = (vl, {k: v.detach().clone() for k, v in head.state_dict().items()})
        head.load_state_dict(best[1]); return head, feat

    enc, enc_mean, enc_std, bundle = load_frozen_encoder(MAN / cfg['source_encoder']['bundle'])
    enc_hash_before = bundle['state_dict_sha256']
    ROWS = ['teacher', 'blind', 'task_student', 'scratch_student', 'transfer_student']

    def predict_curves(q, z):
        with torch.no_grad():
            sl, jv = q(torch.tensor(gf, dtype=torch.double), torch.tensor(np.tile(z, (ng, 1)), dtype=torch.double))
            return torch.sigmoid(sl).numpy(), jv.numpy()

    seed_metrics = {}
    for seed in cfg['training_seeds']:
        phi, q = train_critic(rows, props, VAL, blind=False, train_ids=TRAIN, seed=seed)
        _, qb = train_critic(rows, props, VAL, blind=True, train_ids=TRAIN, seed=seed)
        st_task, feat_task = distil_student(phi, task_feat, seed)
        st_tr, feat_tr = distil_student(phi, enc_feat, seed, encoder=enc, enc_mean=enc_mean, enc_std=enc_std, head_only=True)
        st_sc, feat_sc = distil_student(phi, enc_feat, seed)
        def z_for(row, sid):
            idx = np.where(S == sid)[0]
            if row == 'teacher':
                with torch.no_grad():
                    return phi(torch.tensor(props[sid][None], dtype=torch.double)).numpy()[0]
            if row == 'blind':
                return np.zeros(4)
            st, feat = {'task_student': (st_task, feat_task), 'transfer_student': (st_tr, feat_tr), 'scratch_student': (st_sc, feat_sc)}[row]
            with torch.no_grad():
                return st(torch.tensor(feat[idx], dtype=torch.double)).mean(0).numpy()
        def curves(row, sid):
            return predict_curves(qb if row == 'blind' else q, z_for(row, sid))
        m = {}
        for row in ROWS:
            per = []
            for sid in TEST:
                pS, pJ = curves(row, sid); measS = meas[sid]['mean_success']; measJ = meas[sid]['mean_J']
                B, w = prop[sid]; in_reg = tm.pi_g(grid, B, w) <= tm.PI_G_MAX
                # selection: among predicted-feasible in-regime grasps, argmax predicted J; fallback argmax predicted success
                cand = np.where(in_reg & (pS >= 0.5))[0]
                sel = int(cand[np.argmax(pJ[cand])]) if len(cand) else int(np.where(in_reg)[0][np.argmax(pS[in_reg])]) if in_reg.any() else int(np.argmax(pS))
                best_meas = float(np.max(measJ[in_reg])) if in_reg.any() else float(np.max(measJ))
                regret = float(best_meas - measJ[sel])
                map_rmse = float(np.sqrt(np.mean((pS - measS) ** 2)))          # map on SUCCESS prob (informative)
                plp, pup, _ = tm.band_crossings(pS, grid); plm, pum, _ = tm.band_crossings(measS, grid)
                iou = tm.interval_iou(plp, pup, plm, pum)
                per.append(dict(setting=sid, regret=regret, map_rmse=map_rmse, iou=iou,
                                pred_ell_L=plp, pred_ell_U=pup, meas_ell_L=plm, meas_ell_U=pum, sel_idx=sel))
            uniq = {}
            for x in per:
                uniq.setdefault(tuple(np.round(np.log10(prop[x['setting']]), 6)), []).append(x['regret'])
            ious = [x['iou'] for x in per if x['iou'] is not None]
            m[row] = dict(regret_group_mean=float(np.mean([np.mean(v) for v in uniq.values()])),
                          map_rmse=float(np.mean([x['map_rmse'] for x in per])),
                          iou=float(np.mean(ious)) if ious else None, n_unique_groups=len(uniq), per_setting=per)
        seed_metrics[seed] = m
    enc2, _, _, bundle2 = load_frozen_encoder(MAN / cfg['source_encoder']['bundle'])
    enc_hash_after = bundle2['state_dict_sha256']

    def across(field, row):
        return [seed_metrics[s][row][field] for s in cfg['training_seeds'] if seed_metrics[s][row][field] is not None]
    summary = {row: dict(regret_mean=float(np.mean(across('regret_group_mean', row))), regret_std=float(np.std(across('regret_group_mean', row))),
                         map_rmse_mean=float(np.mean(across('map_rmse', row))), map_rmse_std=float(np.std(across('map_rmse', row))),
                         iou_mean=(float(np.mean(across('iou', row))) if across('iou', row) else None)) for row in ROWS}
    dNI = cfg['transfer']['non_inferiority_margin']
    fr, sc, bl = summary['transfer_student']['map_rmse_mean'], summary['scratch_student']['map_rmse_mean'], summary['blind']['map_rmse_mean']
    transfer_verdict = ('YES' if (fr - sc < dNI and fr < bl) else ('NO' if fr >= bl else 'NOT-ESTABLISHED'))
    result = dict(schema_version=1, manifest_digest=live, splits=cfg['splits'], rows=ROWS, training_seeds=cfg['training_seeds'],
                  metric_note='map RMSE + IoU on the predicted SUCCESS-probability curve; selection regret via the J head (feasible-masked)',
                  summary=summary,
                  primary_selection_regret={row: dict(mean=summary[row]['regret_mean'], std_over_seeds=summary[row]['regret_std']) for row in ROWS},
                  co_primary_map={row: dict(map_rmse=summary[row]['map_rmse_mean'], iou=summary[row]['iou_mean']) for row in ROWS},
                  transfer=dict(verdict=transfer_verdict, frozen_map_rmse=fr, scratch_map_rmse=sc, blind_map_rmse=bl, delta_NI=dNI,
                                rule=cfg['transfer']['rule'], encoder_hash_before=enc_hash_before, encoder_hash_after=enc_hash_after,
                                encoder_unchanged=bool(enc_hash_before == enc_hash_after)),
                  bootstrap_label='multi-seed variability reported; eval-draw contrast LABELED conditional on the trained models (A-16)',
                  aggregation='regret aggregated over UNIQUE (B,w) TEST groups; ratio pairs = invariance controls (A-15)',
                  per_seed=seed_metrics, honest_nulls=cfg['honest_nulls'])
    (MAN / 'distal_critic_results.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    ax[0].bar(ROWS, [summary[r]['regret_mean'] for r in ROWS], yerr=[summary[r]['regret_std'] for r in ROWS]); ax[0].set_ylabel('selection regret'); ax[0].tick_params(axis='x', rotation=20)
    ax[1].bar(ROWS, [summary[r]['map_rmse_mean'] for r in ROWS], yerr=[summary[r]['map_rmse_std'] for r in ROWS]); ax[1].set_ylabel('map RMSE (success)'); ax[1].tick_params(axis='x', rotation=20)
    ax[2].bar(ROWS, [summary[r]['iou_mean'] or 0 for r in ROWS]); ax[2].set_ylabel('band IoU'); ax[2].tick_params(axis='x', rotation=20)
    fig.suptitle('Distal critic: history-variant comparison (regret PRIMARY + map/band co-primary)'); fig.tight_layout()
    fig.savefig(ROOT / 'figures/distal_history_variant_comparison.png', dpi=170); plt.close(fig)
    print(json.dumps({'transfer_verdict': transfer_verdict, 'encoder_unchanged': bool(enc_hash_before == enc_hash_after),
                      'regret': {r: round(summary[r]['regret_mean'], 4) for r in ROWS},
                      'map_rmse_success': {r: round(summary[r]['map_rmse_mean'], 4) for r in ROWS},
                      'iou': {r: (round(summary[r]['iou_mean'], 3) if summary[r]['iou_mean'] is not None else None) for r in ROWS}}, indent=2))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--manifest', default=None); p.add_argument('--expected-digest', default=None)
    a = p.parse_args(); run(a.manifest, a.expected_digest)
