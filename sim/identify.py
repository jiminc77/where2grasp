"""Hardening Part-2 channel-wise identifiability (unified leak-free split).

Ridge is PRIMARY (train-only scaler, setting-grouped CV, pre-registered temporal
mean+last pooling of the (y,z) frames); MLP is CONFIRMATORY with repeated-seed
stability. The frozen two-sided prediction + inherited truth table are unchanged
from Step-3: positive control (probe-enriched shape recovers B_eff/w), confound
guard (paired (cB,cw) shapes identical), null side (shape cannot separate B,w),
repair side (+wrench separates B,w). Splits come from the manifest; nothing here
consumes module constants.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sim.sweep import settings

ROOT = Path(__file__).resolve().parent


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def pool_temporal(shape_112, n_frames=7, per=16):
    """Pre-registered ridge reduction: mean over frames + last frame -> 2*per (=32)."""
    fr = np.asarray(shape_112).reshape(n_frames, per)
    return np.concatenate([fr.mean(0), fr[-1]])


class LinearModel:
    def __init__(self, mu, sd, W):
        self.mu, self.sd, self.W = mu, sd, W
    def predict(self, X):
        return np.c_[np.ones(len(X)), (np.asarray(X) - self.mu) / self.sd] @ self.W


def fit_ridge(X, Y, alpha):
    mu = X.mean(0); sd = X.std(0); sd[sd < 1e-12] = 1
    A = np.c_[np.ones(len(X)), (X - mu) / sd]; reg = np.eye(A.shape[1]) * alpha; reg[0, 0] = 0
    return LinearModel(mu, sd, np.linalg.solve(A.T @ A + reg, A.T @ Y))


def cv_ridge(X, Y, S, cfg):
    folds = cfg['grouped_cv']['folds']; alphas = cfg['ridge_alpha_grid']; scores = []; by = []
    for a in alphas:
        ft = []
        for ids in folds.values():
            va = np.isin(S, ids); tr = ~va
            if not va.any() or not tr.any():
                continue
            pred = fit_ridge(X[tr], Y[tr], a).predict(X[va]); ft.append([rmse(Y[va, j], pred[:, j]) for j in range(Y.shape[1])])
        by.append(np.mean(ft, axis=0).tolist()); scores.append(float(np.mean(ft)))
    k = int(np.argmin(scores)); return fit_ridge(X, Y, alphas[k]), alphas[k], scores, by[k]


def fit_mlp(X, Y, seed, epochs=200):
    mu = X.mean(0); sd = X.std(0); sd[sd < 1e-12] = 1
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 64), torch.nn.ReLU(),
                              torch.nn.Linear(64, 64), torch.nn.ReLU(), torch.nn.Linear(64, Y.shape[1])).double()
    torch.manual_seed(seed); opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    xx = torch.tensor((X - mu) / sd); yy = torch.tensor(Y)
    for _ in range(epochs):
        opt.zero_grad(); loss = ((net(xx) - yy) ** 2).mean(); loss.backward(); opt.step()
    class M:
        def predict(self, Z):
            return net(torch.tensor((np.asarray(Z) - mu) / sd)).detach().numpy()
    return M()


def load(manifest):
    cfg = json.loads(Path(manifest).read_text())
    sm = json.loads((ROOT / 'manifests' / cfg['sweep_manifest']).read_text())
    ss = {s['id']: s for s in settings(sm)}
    hz = np.load(ROOT / 'manifests/hard_histories_v2.npz')
    if str(hz['manifest_digest'].item()) != digest(manifest):
        raise RuntimeError('history/manifest digest mismatch')
    return cfg, sm, ss, hz


def channels(hz, i):
    """Return per-channel feature vectors for history i (pooled task shape)."""
    prop = hz['proprio'][i]; task = pool_temporal(hz['shape'][i]); probe = hz['probe_shape'][i]; wr = np.array([hz['wrench'][i]])
    return {
        'proprioception-only': prop,
        '+shape': np.r_[prop, task],
        '+wrench': np.r_[prop, task, wr],
        'probe+shape': np.r_[prop, probe, task],
        'probe+shape+wrench': np.r_[prop, probe, task, wr],
    }


def run(manifest, expected_digest=None):
    live = digest(manifest)
    if expected_digest and live != expected_digest:
        raise RuntimeError(f'manifest digest mismatch: {live} != {expected_digest}')
    cfg, sm, ss, hz = load(manifest)
    T = cfg['targets']; mar = cfg['margins']
    train = set(cfg['splits']['train']); test = set(cfg['splits']['test'])
    S = hz['setting'].astype(str); n = len(S)
    feats = [channels(hz, i) for i in range(n)]
    Y = np.array([np.log10([ss[s]['B_eff'], ss[s]['w'], ss[s]['B_eff'] / ss[s]['w']]) for s in S])
    tr = np.isin(S, list(train)); te = np.isin(S, list(test))
    result = {}
    for name in ['proprioception-only', '+shape', '+wrench', 'probe+shape', 'probe+shape+wrench']:
        X = np.array([f[name] for f in feats])
        ridge, a, cvs, cvt = cv_ridge(X[tr], Y[tr], S[tr], cfg); pr = ridge.predict(X[te])
        # confirmatory MLP with repeated-seed stability
        mlp_rmses = []
        for sd in cfg['models']['repeated_seeds']:
            pm = fit_mlp(X[tr], Y[tr], sd, cfg['models']['mlp']['epochs']).predict(X[te])
            mlp_rmses.append([rmse(Y[te, j], pm[:, j]) for j in range(3)])
        mlp_rmses = np.array(mlp_rmses)
        result[name] = {
            'ridge': {'alpha': a, 'train_cv_rmse_mean': min(cvs),
                      'test_rmse': dict(zip(T, [rmse(Y[te, j], pr[:, j]) for j in range(3)]))},
            'mlp_confirmatory': {'test_rmse_mean': dict(zip(T, mlp_rmses.mean(0).tolist())),
                                 'test_rmse_std_over_seeds': dict(zip(T, mlp_rmses.std(0).tolist())),
                                 'seeds': cfg['models']['repeated_seeds']},
        }
    # paired confound guard on probe-enriched shape (matched CRN)
    def probe_enriched_shape(i):
        return np.r_[hz['probe_shape'][i], pool_temporal(hz['shape'][i])]
    pair_shape = []
    for a_id, b_id in cfg['splits']['final_test_pairs']:
        A = np.mean([probe_enriched_shape(i) for i in range(n) if S[i] == a_id], 0)
        B = np.mean([probe_enriched_shape(i) for i in range(n) if S[i] == b_id], 0)
        pair_shape.append(float(np.sqrt(np.mean((A - B) ** 2)) / max(np.sqrt(np.mean(A * A)), 1e-12)))
    sh = result['probe+shape']['ridge']['test_rmse']; wr = result['probe+shape+wrench']['ridge']['test_rmse']
    positive = sh[T[2]] <= mar['tol_ratio']
    guard = max(pair_shape) <= mar['tol_shape']
    null = sh[T[0]] >= mar['K'] * sh[T[2]] and sh[T[1]] >= mar['K'] * sh[T[2]]
    repair = wr[T[0]] <= mar['tol_indiv'] and wr[T[1]] <= mar['tol_indiv']
    verdict = 'INCONCLUSIVE' if not (positive and guard) else ('PASS' if null and repair else 'FAIL')
    out = {
        'schema_version': 1, 'manifest_digest': live,
        'history_source': {'kind': str(hz['source'].item()), 'rollout_count': int(hz['rollout_count'].item()),
                           'shape_dim': int(hz['shape_dim'].item()), 'probe_dim': int(hz['probe_dim'].item())},
        'splits': cfg['splits'], 'ridge_reduction': 'temporal mean+last pooling (task 112->32; probe-enriched 128->48)',
        'models': result,
        'two_sided_prediction': {
            'positive_control': {'passed': bool(positive), 'probe_shape_ratio_rmse': sh[T[2]], 'tol_ratio': mar['tol_ratio']},
            'confound_guard': {'passed': bool(guard), 'paired_relative_shape_rms': pair_shape, 'tol_shape': mar['tol_shape']},
            'null_side': {'passed': bool(null), 'K': mar['K'],
                          'shape_individual_over_ratio': [sh[T[0]] / max(sh[T[2]], 1e-15), sh[T[1]] / max(sh[T[2]], 1e-15)]},
            'repair_side': {'passed': bool(repair), 'tol_indiv': mar['tol_indiv'],
                            'wrench_individual': [wr[T[0]], wr[T[1]]]},
            'verdict': verdict,
            'truth_table': cfg['truth_table'],
        },
        'proprioception_baseline': result['proprioception-only']['ridge']['test_rmse'],
    }
    (ROOT / 'manifests/hard_identifiability_v2.json').write_text(json.dumps(out, indent=2, sort_keys=True) + '\n')
    # figure: channel x target test RMSE (ridge primary)
    labels = T; x = np.arange(3); fig, ax = plt.subplots(figsize=(9, 4)); width = .15
    order = ['proprioception-only', '+shape', '+wrench', 'probe+shape', 'probe+shape+wrench']
    for k, name in enumerate(order):
        ax.bar(x + (k - 2) * width, [result[name]['ridge']['test_rmse'][t] for t in labels], width, label=name)
    ax.set_xticks(x, labels, rotation=12); ax.set_ylabel('test RMSE (dex, ridge)'); ax.legend(fontsize=7)
    ax.set_title('Identifiability v2: channel x history-type (unified leak-free split)')
    fig.tight_layout(); fig.savefig(ROOT / 'figures/hard_identifiability_table.png', dpi=170); plt.close(fig)
    print(json.dumps({'manifest_digest': live, 'verdict': verdict, 'positive_control': bool(positive),
                      'guard': bool(guard), 'null': bool(null), 'repair': bool(repair)}))
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--manifest', default=str(ROOT / 'manifests/hard_s34_manifest.json'))
    p.add_argument('--expected-digest', default=None); a = p.parse_args(); run(a.manifest, a.expected_digest)
