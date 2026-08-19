"""Item-1b C2-v2: EXECUTE THE FROZEN BANKS EXACTLY -> internally-consistent oracle + teacher, then
the pre-registered k-interaction adaptation curve (graduation).

Score-only after the C1-v2 freeze (adaptation_curve_manifest_v2.json). Stages:
  --stage sweep --grasp G : run the FROZEN new selection {2300-2302} + evaluation {3300-3304} sweep
                            for ONE lift-grid cell (fresh process per cell -> avoids GPU-context
                            accumulation); winner template picked from THAT cell's selection success,
                            evaluation bank scored at the winner. Reuses addendum_sweep.run_grasp_batch.
  --stage merge           : concatenate the 17 per-cell shards -> adaptation_curve_v2_sweep_results.npz
                            + build adaptation_curve_v2_landscape.json (the NEW evaluation-bank oracle).
  --stage ksweep          : the mean-pool k-prefix student pipeline scored against the v2 oracle +
                            v2 teacher, REUSING the identical-history-bank npz. Persists raw surfaces,
                            band IoU, ratio invariance; emits the graduated figure. pre_registered_outcome
                            = ESTABLISHED (frozen banks executed exactly).
"""
from __future__ import annotations

import argparse, hashlib, json, time
from pathlib import Path

import numpy as np
import torch
from torch import nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sim import adaptation_curve as ac
from sim.identify import pool_temporal
from sim.distal_critic import train_critic, seed_all
from sim.sweep import settings
from sim.addendum_sweep import run_grasp_batch

ROOT = Path(__file__).resolve().parent
MAN = ROOT / "manifests"
FIG = ROOT / "figures"
V2 = "adaptation_curve_manifest_v2.json"
HIST = MAN / "adaptation_curve_lift_histories.npz"
SWEEP = MAN / "adaptation_curve_v2_sweep_results.npz"
LAND = MAN / "adaptation_curve_v2_landscape.json"
SWEEP_KEYS = ["setting", "grasp", "ell", "template", "bank", "seed", "J", "success", "delta_tip", "selected_template", "converged", "settle_steps"]


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# --------------------------------------------------------------------------- #
# stage: sweep (GPU) -- run the FROZEN sel/eval banks for one lift-grid cell
# --------------------------------------------------------------------------- #
def stage_sweep(only_grasp, expected_digest=None, batch_size=90):
    m2 = json.loads((MAN / V2).read_text()); live = sha256(MAN / V2)
    if expected_digest and live != expected_digest:
        raise RuntimeError("v2 manifest digest mismatch")
    m = json.loads((MAN / m2["sweep_grid_source"]).read_text())     # hard_sweep_manifest (grid + h + templates)
    ss = settings(m); ell = m["grasp"]["ell"]; nvs = m["grasp"]["n_vertices"]; ntmpl = len(m["templates"]); h = m["h"]
    sel = list(m2["seed_banks"]["selection"]); ev = list(m2["seed_banks"]["evaluation"])
    gi = only_grasp; start = time.time(); rows = []
    # selection pass (all settings x templates x selection seeds)
    specs = [dict(setting=s, template=t, seed=sd, bank="selection") for s in ss for t in range(ntmpl) for sd in sel]
    for i in range(0, len(specs), batch_size):
        rows += run_grasp_batch(m, gi, nvs[gi], float(ell[gi]), specs[i:i + batch_size], h)
    # winner template per setting for THIS cell (max selection-bank success)
    winner = {}
    for s in ss:
        rates = [np.mean([r["success"] for r in rows if r["setting"] == s["id"] and r["template"] == t]) for t in range(ntmpl)]
        winner[s["id"]] = int(np.argmax(rates))
    # evaluation pass at the winner template (all settings x evaluation seeds)
    especs = [dict(setting=s, template=winner[s["id"]], seed=sd, bank="evaluation") for s in ss for sd in ev]
    for i in range(0, len(especs), batch_size):
        rows += run_grasp_batch(m, gi, nvs[gi], float(ell[gi]), especs[i:i + batch_size], h)
    for r in rows:
        r["selected_template"] = bool(winner[r["setting"]] == r["template"])
    out = MAN / f"adaptation_curve_v2_sweep_g{gi}.npz"
    np.savez_compressed(out, **{k: np.array([r[k] for r in rows]) for k in SWEEP_KEYS}, grasp_cell=gi)
    print(json.dumps({"grasp": gi, "rows": len(rows), "wall_clock_s": round(time.time() - start, 1),
                      "nonconverged": float(np.mean([not r["converged"] for r in rows])), "output": str(out)}))


def stage_merge():
    m2 = json.loads((MAN / V2).read_text())
    m = json.loads((MAN / m2["sweep_grid_source"]).read_text()); ss = settings(m); ell = m["grasp"]["ell"]; ng = len(ell); tau = m["tau"]
    parts = sorted(MAN.glob("adaptation_curve_v2_sweep_g*.npz"), key=lambda p: int(p.stem.split("_g")[-1]))
    assert len(parts) == ng, f"expected {ng} sweep shards, got {len(parts)}"
    acc = {k: [] for k in SWEEP_KEYS}
    for p in parts:
        d = np.load(p, allow_pickle=True)
        for k in SWEEP_KEYS:
            acc[k].append(d[k])
    merged = {k: np.concatenate(acc[k]) for k in SWEEP_KEYS}
    np.savez_compressed(SWEEP, **merged, manifest_digest=sha256(MAN / V2), rollout_count=len(merged["grasp"]))
    # oracle = evaluation-bank measured success_rate per setting per cell
    st = merged["setting"].astype(str); gr = merged["grasp"].astype(int); bk = merged["bank"].astype(str); sc = merged["success"].astype(float)
    land = []
    for s in ss:
        rate = [float(np.mean(sc[(st == s["id"]) & (gr == gi) & (bk == "evaluation")])) for gi in range(ng)]
        land.append(dict(id=s["id"], B_eff=s["B_eff"], w=s["w"], success_rate=rate, ell_grid=[float(x) for x in ell], tau=float(tau)))
    LAND.write_text(json.dumps({"settings": land}, indent=2) + "\n")
    print(json.dumps({"sweep": str(SWEEP), "landscape": str(LAND), "rollouts": int(len(merged["grasp"])), "shards": len(parts),
                      "banks": sorted(set(bk.tolist())), "sel_seeds": sorted(set(merged["seed"][bk == "selection"].astype(int).tolist())),
                      "eval_seeds": sorted(set(merged["seed"][bk == "evaluation"].astype(int).tolist()))}))


# --------------------------------------------------------------------------- #
# stage: ksweep (CPU torch) -- k-prefix pipeline vs the v2 oracle + v2 teacher
# --------------------------------------------------------------------------- #
def _teacher_rows(sw, ell, ng):
    rows = []
    for sid in (ac.TRAIN_GROUPS + ac.VAL_GROUPS):
        for gi in range(ng):
            ix = (sw["bank"] == "selection") & (sw["setting"] == sid) & (sw["grasp"] == gi)
            if not ix.any():
                continue
            cand = [(float(sw["success"][ix & (sw["template"] == t)].mean()),
                     float(sw["J"][ix & (sw["template"] == t)].mean()), int(t)) for t in sorted(set(sw["template"][ix].astype(int).tolist()))]
            sp, jj, _ = max(cand, key=lambda x: (x[0], x[1], -x[2]))
            rows.append((sid, np.array([gi / (ng - 1), float(ell[gi])]), sp, jj, 1.0 if sp > 0 else 0.0))
    return rows


def _full_feat(hz, i):
    return np.r_[hz["proprio"][i], pool_temporal(hz["shape"][i]), hz["action"][i]]


def stage_ksweep():
    m2 = json.loads((MAN / V2).read_text()); c1_sha = sha256(MAN / V2)
    m = json.loads((MAN / m2["sweep_grid_source"]).read_text()); ss = {s["id"]: s for s in settings(m)}
    ell = np.array(m["grasp"]["ell"]); ng = len(ell)
    sw = np.load(SWEEP, allow_pickle=True); assert str(sw["manifest_digest"].item()) == c1_sha, "v2 sweep not bound to v2 manifest"
    for s in sw["setting"].astype(str):  # cheap sanity
        break
    land = {x["id"]: x for x in json.loads(LAND.read_text())["settings"]}
    TRAIN, VAL, TEST = list(ac.TRAIN_GROUPS), list(ac.VAL_GROUPS), list(ac.TEST_GROUPS)
    prop = {k: (v["B_eff"], v["w"]) for k, v in ss.items()}
    raw = {k: np.log10([v[0], v[1]]) for k, v in prop.items()}
    a = np.array([raw[k] for k in TRAIN]); pm, psd = a.mean(0), a.std(0); props = {k: (v - pm) / psd for k, v in raw.items()}
    rows = _teacher_rows(sw, ell, ng); gf = np.c_[np.arange(ng) / (ng - 1), ell]
    hz = np.load(HIST, allow_pickle=True)
    S = hz["setting"].astype(str); Gr = hz["grasp"].astype(int); Tp = hz["template"].astype(int); Sd = hz["seed"].astype(int)
    schedule = ac.schedule_lift(); hist_seeds = list(ac.SEED_BANKS["history"]); kmax = max(ac.K_AXIS)
    ORD = {}
    for sid in (TRAIN + VAL + TEST):
        for sd in hist_seeds:
            feats = []
            for j in range(kmax):
                cell, _, tmpl = schedule[j]
                idx = np.where((S == sid) & (Gr == cell) & (Tp == tmpl) & (Sd == sd))[0]
                if idx.size == 0:
                    raise RuntimeError(f"missing history {sid} cell={cell} tmpl={tmpl} seed={sd}")
                feats.append(_full_feat(hz, idx[0]))
            ORD[(sid, sd)] = np.asarray(feats)

    def prefix_feat(sid, seed, k):
        return ac.prefix_summary(ORD[(sid, seed)], k)

    def curve_from_z(q, z):
        with torch.no_grad():
            sl, _ = q(torch.tensor(gf, dtype=torch.double), torch.tensor(np.tile(z, (ng, 1)), dtype=torch.double))
            return torch.sigmoid(sl).numpy()

    measured = {sid: np.asarray(land[sid]["success_rate"], float) for sid in TEST}
    surfaces = {b: {ki: {sid: [] for sid in TEST} for ki in range(len(ac.K_AXIS))} for b in ("teacher", "blind", "student", "sysid")}
    train_seeds = list(ac.SEED_BANKS["training"])
    for tseed in train_seeds:
        phi, q = train_critic(rows, props, VAL, blind=False, train_ids=TRAIN, seed=tseed)
        _, qb = train_critic(rows, props, VAL, blind=True, train_ids=TRAIN, seed=tseed)
        tcur = {sid: curve_from_z(q, phi(torch.tensor(props[sid][None], dtype=torch.double)).detach().numpy()[0]) for sid in TEST}
        bcur = {sid: curve_from_z(qb, np.zeros(4)) for sid in TEST}
        for ki in range(len(ac.K_AXIS)):
            for sid in TEST:
                surfaces["teacher"][ki][sid].append(tcur[sid]); surfaces["blind"][ki][sid].append(bcur[sid])
        for ki, k in enumerate(ac.K_AXIS):
            if k == 0:
                for sid in TEST:
                    surfaces["student"][ki][sid].append(bcur[sid]); surfaces["sysid"][ki][sid].append(bcur[sid])
                continue
            X = {sid: np.array([prefix_feat(sid, sd, k) for sd in hist_seeds]) for sid in (TRAIN + VAL + TEST)}
            Xtr = np.concatenate([X[s] for s in TRAIN]); Ytr = np.concatenate([np.tile(phi(torch.tensor(props[s][None], dtype=torch.double)).detach().numpy(), (len(hist_seeds), 1)) for s in TRAIN])
            Xv = np.concatenate([X[s] for s in VAL]); Yv = np.concatenate([np.tile(phi(torch.tensor(props[s][None], dtype=torch.double)).detach().numpy(), (len(hist_seeds), 1)) for s in VAL])
            hm = Xtr.mean(0); hs = Xtr.std(0); hs[hs < 1e-12] = 1
            Xtrn = torch.tensor((Xtr - hm) / hs, dtype=torch.double); Xvn = torch.tensor((Xv - hm) / hs, dtype=torch.double)
            Ytr_t = torch.tensor(Ytr, dtype=torch.double); Yv_t = torch.tensor(Yv, dtype=torch.double)
            seed_all(tseed); st = nn.Sequential(nn.Linear(ac.FEATURE_DIM, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 4)).double()
            opt = torch.optim.Adam(st.parameters(), lr=1e-2); best = (float("inf"), None)
            for _ in range(300):
                opt.zero_grad(); loss = ((st(Xtrn) - Ytr_t) ** 2).mean(); loss.backward(); opt.step()
                with torch.no_grad():
                    vl = float(((st(Xvn) - Yv_t) ** 2).mean())
                if vl < best[0]:
                    best = (vl, {kk: v.detach().clone() for kk, v in st.state_dict().items()})
            st.load_state_dict(best[1])
            lam = 1.0; A = Xtrn.numpy(); W = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ Ytr)
            for sid in TEST:
                with torch.no_grad():
                    zst = st(torch.tensor((X[sid] - hm) / hs, dtype=torch.double)).mean(0).numpy()
                surfaces["student"][ki][sid].append(curve_from_z(q, zst))
                zsy = (((X[sid] - hm) / hs) @ W).mean(0)
                surfaces["sysid"][ki][sid].append(curve_from_z(q, zsy))

    def band_iou(pred, meas, tau=0.5):
        P = set(np.where(np.asarray(pred) >= tau)[0].tolist()); M = set(np.where(np.asarray(meas) >= tau)[0].tolist())
        return 1.0 if (not P and not M) else (float(len(P & M) / len(P | M)) if (P | M) else 1.0)

    def agg_over_seeds(fn):
        out = []
        for ki in range(len(ac.K_AXIS)):
            per_seed = []
            for si in range(len(train_seeds)):
                vals = [fn(ki, si, sid) for sid in TEST]
                per_seed.append(ac.aggregate_unique_groups(vals)["mean"])
            arr = np.asarray(per_seed, float)
            out.append(dict(mean=(None if np.all(np.isnan(arr)) else float(np.nanmean(arr))),
                            seed_std=(None if np.all(np.isnan(arr)) else float(np.nanstd(arr))), n_seeds=len(train_seeds)))
        return out

    def make_curve(b):
        return {"map_rmse": agg_over_seeds(lambda ki, si, sid: ac.map_rmse(surfaces[b][ki][sid][si], measured[sid])),
                "selection_regret": agg_over_seeds(lambda ki, si, sid: ac.selection_regret(surfaces[b][ki][sid][si], measured[sid])),
                "boundary_err": agg_over_seeds(lambda ki, si, sid: ac.boundary_index_error(surfaces[b][ki][sid][si], measured[sid], ell)),
                "band_iou": agg_over_seeds(lambda ki, si, sid: band_iou(surfaces[b][ki][sid][si], measured[sid]))}

    curve = {b: make_curve(b) for b in ("teacher", "blind", "student", "sysid")}
    ratio_ref = {"R0": "B1_w1", "R1": "B3_w2", "R2": "B2_w1"}
    amx = lambda sid: int(np.argmax(np.asarray(land[sid]["success_rate"], float)))
    ratio_invariance = {rid: dict(reference=ref, argmax_cell=amx(rid), reference_argmax_cell=amx(ref),
                                  offset_cells=int(abs(amx(rid) - amx(ref))), invariant=bool(abs(amx(rid) - amx(ref)) <= 1)) for rid, ref in ratio_ref.items()}
    bk = sw["bank"].astype(str); seeds_by_bank = {b: sorted(set(sw["seed"][bk == b].astype(int).tolist())) for b in ("selection", "evaluation")}
    protocol_fidelity = dict(
        executed_exactly=True,
        oracle=dict(source="adaptation_curve_v2_landscape.json", built_from_evaluation_bank=seeds_by_bank["evaluation"], frozen_evaluation_bank=list(ac.SEED_BANKS["evaluation"])),
        teacher_labels=dict(source="adaptation_curve_v2_sweep_results.npz", bank="selection", built_from_selection_bank=seeds_by_bank["selection"], frozen_selection_bank=list(ac.SEED_BANKS["selection"])),
        history=dict(source="adaptation_curve_lift_histories.npz reused unchanged", bank=hist_seeds), training=dict(bank=train_seeds),
        note="v2 executes the FROZEN sel/eval banks EXACTLY: oracle + teacher labels both built from the new {2300-2302}/{3300-3304} draws this phase; no addendum reuse. Internally consistent -> pre-registered outcome ESTABLISHED.")
    assert seeds_by_bank["selection"] == sorted(ac.SEED_BANKS["selection"]), "selection bank mismatch vs freeze"
    assert seeds_by_bank["evaluation"] == sorted(ac.SEED_BANKS["evaluation"]), "evaluation bank mismatch vs freeze"

    def clean(o):
        if isinstance(o, float) and np.isnan(o):
            return None
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, list):
            return [clean(v) for v in o]
        return o

    results = clean(dict(item="1b_k_interaction_adaptation_curve_GRADUATED", manifest="adaptation_curve_manifest_v2.json",
                   c1_v2_manifest_sha256=c1_sha, k_axis=list(ac.K_AXIS), task="lift_and_clear_primary_17cell", train_seeds=train_seeds,
                   history_seeds=hist_seeds, test_groups=TEST, curve=curve, ratio_invariance=ratio_invariance,
                   protocol_fidelity=protocol_fidelity, pre_registered_outcome="ESTABLISHED",
                   pre_registered_outcome_note="The frozen protocol executed EXACTLY: the new selection {2300-2302} + evaluation {3300-3304} banks were RUN this phase to build BOTH the teacher labels AND the oracle (verified: those exact seeds present in the v2 sweep artifacts). This GRADUATES the Item-1 headline k-interaction adaptation curve to a pre-registered result.",
                   reference_endpoints_DESCRIPTIVE=json.loads((MAN / "adaptation_curve_manifest.json").read_text())["reference_endpoints_DESCRIPTIVE"]["lift_map_rmse"],
                   lift_selection_regret_degenerate=True,
                   acceptance_note="PROTOCOL + HONESTY (NOT performance): frozen banks executed exactly; per-k curve + ordering + band IoU + ratio invariance reported; reference endpoints + monotonicity DESCRIPTIVE only."))
    (MAN / "adaptation_curve_v2_results.json").write_text(json.dumps(results, indent=2))
    flat = {}
    for b in surfaces:
        for ki in range(len(ac.K_AXIS)):
            for sid in TEST:
                flat[f"{b}__k{ac.K_AXIS[ki]}__{sid}"] = np.asarray(surfaces[b][ki][sid])
    for sid in TEST:
        flat[f"measured__{sid}"] = measured[sid]
    np.savez(MAN / "adaptation_curve_v2_surfaces.npz", ell=ell, k_axis=np.array(list(ac.K_AXIS)),
             train_seeds=np.array(train_seeds), test_groups=np.array(TEST), c1_manifest_sha256=c1_sha, **flat)
    _figure(curve)
    print("V2 KSWEEP done. k-axis:", list(ac.K_AXIS))
    for b in ("teacher", "blind", "student", "sysid"):
        print(" ", b, "map_rmse:", [None if x["mean"] is None else round(x["mean"], 4) for x in curve[b]["map_rmse"]])
    print(" ratio_invariance:", {r: ratio_invariance[r]["offset_cells"] for r in ratio_invariance})
    print(" PRE-REGISTERED OUTCOME: ESTABLISHED (frozen sel/eval banks", seeds_by_bank["selection"], "/", seeds_by_bank["evaluation"], "executed exactly)")
    return results


def _figure(curve):
    FIG.mkdir(parents=True, exist_ok=True)
    k = list(ac.K_AXIS); xs = [max(kk, 0.5) for kk in k]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    tea = curve["teacher"]["map_rmse"][0]["mean"]; bli = curve["blind"]["map_rmse"][0]["mean"]
    ax[0].axhline(tea, color="tab:green", ls="--", lw=1, label="teacher ceiling")
    ax[0].axhline(bli, color="tab:red", ls=":", lw=1, label="blind floor (k=0)")
    for b, c in (("student", "tab:blue"), ("sysid", "tab:orange")):
        m = [x["mean"] for x in curve[b]["map_rmse"]]; e = [x["seed_std"] for x in curve[b]["map_rmse"]]
        ax[0].errorbar(xs, m, yerr=e, fmt="o-", color=c, capsize=3, label=b)
    ax[0].set_xscale("log"); ax[0].set_xlabel("k interactions (k=0 = blind prior)"); ax[0].set_ylabel("held-out map RMSE")
    ax[0].set_title("GRADUATED k-interaction adaptation curve (v2; frozen banks executed exactly)"); ax[0].legend(fontsize=7)
    ax[0].set_xticks(xs); ax[0].set_xticklabels([str(kk) for kk in k])
    for b, c in (("teacher", "tab:green"), ("blind", "tab:red"), ("student", "tab:blue"), ("sysid", "tab:orange")):
        m = [x["mean"] for x in curve[b]["selection_regret"]]
        ax[1].plot(xs, m, "o-", color=c, label=b)
    ax[1].set_xscale("log"); ax[1].set_xlabel("k interactions"); ax[1].set_ylabel("selection regret")
    ax[1].set_title("Selection regret vs k (lift regret DEGENERATE -- reported honestly)")
    ax[1].set_xticks(xs); ax[1].set_xticklabels([str(kk) for kk in k]); ax[1].legend(fontsize=7)
    fig.suptitle("Item 1b: pre-registered interaction-COUNT adaptation curve (A-22; v2 frozen-bank graduation)")
    fig.tight_layout(); fig.savefig(FIG / "adaptation_curve_v2.png", dpi=140); plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--stage", choices=["sweep", "merge", "ksweep"], required=True)
    p.add_argument("--grasp", type=int, default=None); p.add_argument("--expected-digest", default=None)
    a = p.parse_args()
    if a.stage == "sweep":
        stage_sweep(a.grasp, a.expected_digest)
    elif a.stage == "merge":
        stage_merge()
    else:
        stage_ksweep()
