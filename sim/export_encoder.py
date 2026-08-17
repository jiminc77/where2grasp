"""Deterministically materialise the FROZEN hardening-A lift-and-clear task-only encoder bundle.

The transfer probe needs a loadable inference bundle, but hardening-A trained the task-only
Student in memory (critic.py) and never serialised it. This tool RECONSTRUCTS it deterministically
from committed hardening-A inputs ONLY (hard_histories_v2.npz + hard_s34_manifest.json +
hard_sweep_results.npz + critic.py recipe), consuming NO distal data, and writes:

  sim/manifests/source_encoder_bundle.pt   (torch bundle: state_dict + scaler + provenance)

Contract captured: task-only Student(40->64->32->4), TRAIN mean/std (40-D), the exact
proprio(8) + pool_temporal(112)->32 input, arch/dtype/seed/epoch-rule/chosen-epoch, source
TRAIN/VAL ids, source manifest digest, and input-artifact sha256s. Frozen-state tests assert
requires_grad=False and hash-identity before/after distal head training.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import torch
from sim.critic import Phi, Critic, Student, seed_all, train_critic
from sim.identify import pool_temporal
from sim.sweep import settings

ROOT = Path(__file__).resolve().parent
MAN = ROOT / 'manifests'


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _tensor_hash(sd):
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode()); h.update(np.ascontiguousarray(sd[k].detach().cpu().numpy()).tobytes())
    return h.hexdigest()


def reconstruct(out=None):
    cfg = json.loads((MAN / 'hard_s34_manifest.json').read_text())
    sm = json.loads((MAN / 'hard_sweep_manifest.json').read_text())
    ss = {x['id']: x for x in settings(sm)}
    TRAIN = cfg['splits']['train']; VAL = cfg['splits']['val']
    ng = len(sm['grasp']['ell'])
    # train-only property scaler (matches critic.run)
    raw = {k: np.log10([v['B_eff'], v['w']]) for k, v in ss.items()}
    a = np.array([raw[k] for k in TRAIN]); pm, psd = a.mean(0), a.std(0)
    props = {k: (v - pm) / psd for k, v in raw.items()}
    # teacher critic (phi,q) from the selection-bank labels (deterministic seed)
    sw = np.load(MAN / 'hard_sweep_results.npz')
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
    seed = cfg['models']['critic']['seed']
    phi, q, _ = train_critic(rows, props, VAL, blind=False, train_ids=TRAIN, seed=seed)
    # task-only history features + train scaler (matches critic.train_student(task_feat))
    hz = np.load(MAN / 'hard_histories_v2.npz')
    S = hz['setting'].astype(str)
    def task_feat(i):
        return np.r_[hz['proprio'][i], pool_temporal(hz['shape'][i])]
    X = np.array([task_feat(i) for i in range(len(S))])
    ti = np.isin(S, TRAIN); vi = np.isin(S, VAL)
    hm = X[ti].mean(0); hs = X[ti].std(0); hs[hs < 1e-12] = 1
    Xtr = torch.tensor((X[ti] - hm) / hs, dtype=torch.double); Xv = torch.tensor((X[vi] - hm) / hs, dtype=torch.double)
    with torch.no_grad():
        Z = phi(torch.tensor([props[s] for s in S[ti]], dtype=torch.double))
        VZ = phi(torch.tensor([props[s] for s in S[vi]], dtype=torch.double))
    seed_all(seed); st = Student(X.shape[1]).double(); opt = torch.optim.Adam(st.parameters(), lr=1e-2)
    best = (float('inf'), None, 0)
    for ep in range(1, cfg['models']['distill_epochs'] + 1):
        opt.zero_grad(); loss = ((st(Xtr) - Z) ** 2).mean(); loss.backward(); opt.step()
        with torch.no_grad():
            vl = float(((st(Xv) - VZ) ** 2).mean())
        if vl < best[0]:
            best = (vl, {k: v.detach().clone() for k, v in st.state_dict().items()}, ep)
    st.load_state_dict(best[1])
    bundle = dict(
        schema='source_encoder_bundle.v1', task='lift_and_clear', role='task_only_history_encoder',
        state_dict={k: v.detach().cpu() for k, v in st.state_dict().items()},
        state_dict_sha256=_tensor_hash(st.state_dict()),
        input_contract=dict(feature='proprio(8) + pool_temporal(shape_112)->32 = 40-D', input_dim=40,
                            pool='temporal mean(7 frames) + last frame of the 16-D (y,z) samples',
                            arch='Linear(40,64)-ReLU-Linear(64,32)-ReLU-Linear(32,4)', z=4, dtype='float64'),
        train_scaler=dict(mean=hm.tolist(), std=hs.tolist()),
        training=dict(seed=seed, distill_epochs=cfg['models']['distill_epochs'],
                      epoch_selection='min validation MSE', chosen_epoch=best[2], best_val_mse=best[0]),
        source_splits=dict(train=TRAIN, val=VAL),
        source_manifest_digest=sha256(MAN / 'hard_s34_manifest.json'),
        input_artifact_sha256={f: sha256(MAN / f) for f in
                               ('hard_s34_manifest.json', 'hard_histories_v2.npz', 'hard_sweep_results.npz',
                                'hard_sweep_manifest.json', 'hard_sweep_landscape.json')},
        provenance='reconstructed deterministically from committed hardening-A inputs; consumes NO distal data',
    )
    out = Path(out) if out else (MAN / 'source_encoder_bundle.pt')
    torch.save(bundle, out)
    print(json.dumps(dict(output=str(out), bundle_sha256=sha256(out),
                          state_dict_sha256=bundle['state_dict_sha256'], chosen_epoch=best[2],
                          best_val_mse=round(best[0], 6), input_dim=40), indent=2))
    return sha256(out)


def load_frozen_encoder(bundle_path=None):
    """Load the task-only encoder for inference; every parameter requires_grad=False."""
    bundle_path = Path(bundle_path) if bundle_path else (MAN / 'source_encoder_bundle.pt')
    b = torch.load(bundle_path, weights_only=False)
    enc = Student(b['input_contract']['input_dim']).double()
    enc.load_state_dict(b['state_dict'])
    for p in enc.parameters():
        p.requires_grad_(False)
    enc.eval()
    return enc, np.array(b['train_scaler']['mean']), np.array(b['train_scaler']['std']), b


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--out', default=None); a = p.parse_args(); reconstruct(a.out)
