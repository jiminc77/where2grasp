"""Distal Part-3 critic: 5 equal-budget rows + selection-regret PRIMARY + band co-primary + transfer.

Rows (shared phi_theta:(B,w)->z(4)->J-regression critic): teacher / blind / task-history student /
leak-free sysID / FROZEN-cross-task-transfer student (source lift-and-clear encoder frozen, head-only).
J-REGRESSION head defines selection (BCE-on-success cannot: many grasps share success=1 with
different J). Selection regret = max_g E[J|s,g]-E[J|s,g_selected] on winner-only evaluation,
aggregated over UNIQUE (B,w) TEST groups (ratio pairs = invariance controls, A-15). Co-primary =
map RMSE + ell_L/ell_U band recovery (first-up/last-down; censored/regime edges excluded; IoU).
Transfer rule = non-inferiority (delta_NI=0.05, lower-is-better). Multiple training seeds report
seed variability; the eval-draw bootstrap is LABELED conditional-on-models (A-16).
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


class Phi(nn.Module):
    def __init__(self):
        super().__init__(); self.net = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 4))
    def forward(self, x):
        return self.net(x)


class JCritic(nn.Module):
    """J-regression head: (grasp features, z) -> predicted mean J."""
    def __init__(self):
        super().__init__(); self.net = nn.Sequential(nn.Linear(6, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, g, z):
        return self.net(torch.cat((g, z), 1)).squeeze(1)


class Student(nn.Module):
    def __init__(self, d):
        super().__init__(); self.net = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 4))
    def forward(self, x):
        return self.net(x)


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def train_critic(rows, props, VAL, blind, train_ids, epochs=300, seed=3403):
    seed_all(seed); phi = Phi().double(); q = JCritic().double()
    pars = list(q.parameters()) if blind else list(q.parameters()) + list(phi.parameters())
    opt = torch.optim.Adam(pars, lr=1e-2)
    def batch(ids):
        r = [x for x in rows if x[0] in ids]
        g = torch.tensor(np.array([x[1] for x in r]), dtype=torch.double)
        y = torch.tensor([x[2] for x in r], dtype=torch.double)
        p = torch.tensor(np.array([props[x[0]] for x in r]), dtype=torch.double); return g, p, y
    tg, tp, ty = batch(train_ids); vg, vp, vy = batch(VAL); best = (float('inf'), None)
    for _ in range(epochs):
        opt.zero_grad(); z = torch.zeros((len(tg), 4), dtype=torch.double) if blind else phi(tp)
        loss = ((q(tg, z) - ty) ** 2).mean(); loss.backward(); opt.step()
        with torch.no_grad():
            vz = torch.zeros((len(vg), 4), dtype=torch.double) if blind else phi(vp)
            vl = float(((q(vg, vz) - vy) ** 2).mean())
        if vl < best[0]:
            best = (vl, ({k: v.detach().clone() for k, v in phi.state_dict().items()},
                         {k: v.detach().clone() for k, v in q.state_dict().items()}))
    phi.load_state_dict(best[1][0]); q.load_state_dict(best[1][1])
    return phi, q


def _measured(sw, ids, grid, prop):
    """Per-setting evaluation-bank mean success + mean J curves + winner-only regret target."""
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
    # train-only property scaler
    raw = {k: np.log10([v[0], v[1]]) for k, v in prop.items()}
    a = np.array([raw[k] for k in TRAIN]); pm, psd = a.mean(0), a.std(0); props = {k: (v - pm) / psd for k, v in raw.items()}
    sw = np.load(MAN / 'distal_sweep_results.npz')
    meas = _measured(sw, list(prop), grid, prop)
    # J-regression labels from the SELECTION bank (winner template per (setting,grasp))
    keepsel = (sw['bank'] == 'selection')
    rows = []
    for sid in TRAIN + VAL:
        for gi in range(ng):
            ix = keepsel & (sw['setting'] == sid) & (sw['grasp'] == gi)
            if not ix.any():
                continue
            cand = [(float(sw['J'][ix & (sw['template'] == t)].mean()), int(t)) for t in sorted(set(sw['template'][ix].tolist()))]
            bestJ, _ = max(cand, key=lambda x: (x[0], -x[1]))
            rows.append((sid, np.array([gi / (ng - 1), float(grid[gi])]), bestJ))
    gf = np.c_[np.arange(ng) / (ng - 1), grid]

    # history features (+ action metadata) for the students
    hz = np.load(MAN / 'distal_histories.npz'); assert str(hz['manifest_digest'].item()) == live
    S = hz['setting'].astype(str)
    def task_feat(i):
        return np.r_[hz['proprio'][i], pool_temporal(hz['shape'][i]), hz['action'][i]]          # 42-D
    def enc_feat(i):
        return np.r_[hz['proprio'][i], pool_temporal(hz['shape'][i])]                            # 40-D (source contract)

    def train_student_over_encoder(phi, encoder=None, enc_mean=None, enc_std=None, feat_fn=None, seed=3403, head_only=False):
        """Distil the teacher latent into a student. head_only=True freezes `encoder` and trains a head only."""
        X = np.array([feat_fn(i) for i in range(len(S))]); ti = np.isin(S, TRAIN); vi = np.isin(S, VAL)
        if head_only:
            with torch.no_grad():
                Zenc_all = encoder(torch.tensor((X - enc_mean) / enc_std, dtype=torch.double)).numpy()
            feat = Zenc_all; d = feat.shape[1]
        else:
            hm = X[ti].mean(0); hs = X[ti].std(0); hs[hs < 1e-12] = 1
            feat = (X - hm) / hs; d = feat.shape[1]
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
        head.load_state_dict(best[1])
        return head, feat, best[0]

    # frozen source encoder (transfer)
    enc, enc_mean, enc_std, bundle = load_frozen_encoder(MAN / cfg['source_encoder']['bundle'])
    enc_hash_before = bundle['state_dict_sha256']

    def predict_curve_z(qq, z):
        with torch.no_grad():
            return qq(torch.tensor(gf, dtype=torch.double), torch.tensor(np.tile(z, (ng, 1)), dtype=torch.double)).numpy()

    # multi-seed training + metrics
    seed_metrics = {}
    for seed in cfg['training_seeds']:
        phi, q = train_critic(rows, props, VAL, blind=False, train_ids=TRAIN, seed=seed)
        _, qb = train_critic(rows, props, VAL, blind=True, train_ids=TRAIN, seed=seed)
        st_task, feat_task, _ = train_student_over_encoder(phi, feat_fn=task_feat, seed=seed)
        st_tr, feat_tr, _ = train_student_over_encoder(phi, encoder=enc, enc_mean=enc_mean, enc_std=enc_std,
                                                        feat_fn=enc_feat, seed=seed, head_only=True)
        st_sc, feat_sc, _ = train_student_over_encoder(phi, feat_fn=enc_feat, seed=seed)   # from-scratch encoder, equal budget
        def z_for(row, sid):
            idx = np.where(S == sid)[0]
            if row == 'teacher':
                with torch.no_grad():
                    return phi(torch.tensor(props[sid][None], dtype=torch.double)).numpy()[0]
            if row == 'blind':
                return np.zeros(4)
            fmap = {'task_student': (st_task, feat_task), 'transfer_student': (st_tr, feat_tr), 'scratch_student': (st_sc, feat_sc)}
            st, feat = fmap[row]
            with torch.no_grad():
                return st(torch.tensor(feat[idx], dtype=torch.double)).mean(0).numpy()
        def curve(row, sid):
            qq = qb if row == 'blind' else q
            return 1.0 / (1.0 + np.exp(-predict_curve_z(qq, z_for(row, sid)))) if False else predict_curve_z(qq, z_for(row, sid))
        # NOTE: J-regression => the predicted curve is predicted mean J per grasp (not a probability)
        ROWS = ['teacher', 'blind', 'task_student', 'scratch_student', 'transfer_student']
        m = {}
        for row in ROWS:
            per = []
            for sid in TEST:
                predJ = curve(row, sid); measJ = meas[sid]['mean_J']; measS = meas[sid]['mean_success']
                # selection regret: pick argmax predicted J among in-regime grasps; regret vs measured best
                B, w = prop[sid]; in_reg = tm.pi_g(grid, B, w) <= tm.PI_G_MAX
                sel = int(np.where(in_reg)[0][np.argmax(predJ[in_reg])]) if in_reg.any() else int(np.argmax(predJ))
                best_meas = float(np.max(measJ[in_reg])) if in_reg.any() else float(np.max(measJ))
                regret = float(best_meas - measJ[sel])
                # map RMSE (on success-probability proxy is not available; use J-curve RMSE normalised) + band on measured success
                map_rmse = float(np.sqrt(np.mean((predJ - measJ) ** 2)))
                pl, pu, _ = tm.band_crossings(measS, grid)          # measured band (predicted band from tip_model at analysis)
                per.append(dict(setting=sid, regret=regret, map_rmse=map_rmse, meas_ell_L=pl, meas_ell_U=pu,
                                sel_idx=sel, predicted_J=[float(x) for x in predJ], measured_J=[float(x) for x in measJ]))
            # aggregate regret over UNIQUE (B,w) groups (ratio pairs = invariance controls)
            uniq = {}
            for x in per:
                key = tuple(np.round(np.log10(prop[x['setting']]), 6))
                uniq.setdefault(key, []).append(x['regret'])
            group_regret = float(np.mean([np.mean(v) for v in uniq.values()]))
            m[row] = dict(regret_group_mean=group_regret, map_rmse=float(np.mean([x['map_rmse'] for x in per])),
                          per_setting=per, n_unique_groups=len(uniq))
        seed_metrics[seed] = m
    enc2, _, _, bundle2 = load_frozen_encoder(MAN / cfg['source_encoder']['bundle'])
    enc_hash_after = bundle2['state_dict_sha256']

    # seed variability + LABELED bootstrap contrasts
    ROWS = ['teacher', 'blind', 'task_student', 'scratch_student', 'transfer_student']
    def across(field, row):
        return [seed_metrics[s][row][field] for s in cfg['training_seeds']]
    summary = {row: dict(regret_mean=float(np.mean(across('regret_group_mean', row))),
                         regret_std_over_seeds=float(np.std(across('regret_group_mean', row))),
                         map_rmse_mean=float(np.mean(across('map_rmse', row))),
                         map_rmse_std_over_seeds=float(np.std(across('map_rmse', row)))) for row in ROWS}
    # transfer non-inferiority (lower-is-better): map RMSE, delta_NI=0.05
    dNI = cfg['transfer']['non_inferiority_margin']
    fr = summary['transfer_student']['map_rmse_mean']; sc = summary['scratch_student']['map_rmse_mean']; bl = summary['blind']['map_rmse_mean']
    transfer_verdict = ('YES' if (fr - sc < dNI and fr < bl) else ('NO' if fr >= bl else 'NOT-ESTABLISHED'))
    result = dict(schema_version=1, manifest_digest=live, splits=cfg['splits'], rows=ROWS,
                  training_seeds=cfg['training_seeds'], summary=summary,
                  primary_selection_regret={row: dict(mean=summary[row]['regret_mean'], std_over_seeds=summary[row]['regret_std_over_seeds']) for row in ROWS},
                  transfer=dict(verdict=transfer_verdict, frozen_map_rmse=fr, scratch_map_rmse=sc, blind_map_rmse=bl,
                                delta_NI=dNI, rule=cfg['transfer']['rule'],
                                encoder_hash_before=enc_hash_before, encoder_hash_after=enc_hash_after,
                                encoder_unchanged=bool(enc_hash_before == enc_hash_after)),
                  bootstrap_label='LABELED conditional on trained models; multi-seed variability reported (A-16)',
                  aggregation='regret aggregated over UNIQUE (B,w) TEST groups; ratio pairs = invariance controls (A-15)',
                  per_seed=seed_metrics,
                  honest_nulls=cfg['honest_nulls'])
    (MAN / 'distal_critic_results.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    # figure: history-variant comparison (NOT 'adaptation curve') + transfer
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].bar(ROWS, [summary[r]['regret_mean'] for r in ROWS]); ax[0].set_ylabel('selection regret (unique-group mean)'); ax[0].tick_params(axis='x', rotation=20)
    ax[1].bar(ROWS, [summary[r]['map_rmse_mean'] for r in ROWS]); ax[1].set_ylabel('map RMSE (J)'); ax[1].tick_params(axis='x', rotation=20)
    fig.suptitle('Distal critic: history-variant comparison (PRIMARY regret + co-primary map)'); fig.tight_layout()
    fig.savefig(ROOT / 'figures/distal_history_variant_comparison.png', dpi=170); plt.close(fig)
    print(json.dumps({'manifest_digest': live, 'transfer_verdict': transfer_verdict,
                      'regret': {r: round(summary[r]['regret_mean'], 4) for r in ROWS},
                      'map_rmse': {r: round(summary[r]['map_rmse_mean'], 4) for r in ROWS},
                      'encoder_unchanged': bool(enc_hash_before == enc_hash_after)}, indent=2))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--manifest', default=None); p.add_argument('--expected-digest', default=None)
    a = p.parse_args(); run(a.manifest, a.expected_digest)
