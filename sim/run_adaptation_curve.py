"""Item-1 C2: the k-INTERACTION adaptation-curve GPU sweep (lift-and-clear PRIMARY, 17-cell / 0.03).

Score-only after the C1 freeze (adaptation_curve_manifest.json). Two stages:
  --stage histories : extract NEW-seed (frozen {2500,2501}) lift temporal histories over ALL 17 grasp
                      cells x 4 templates x 15 settings (the acquisition schedule visits every cell).
  --stage sweep     : per k in {0,1,2,4,8,16,32}, build the MEAN-POOL k-prefix summary over the frozen
                      schedule's first k interactions, distil the teacher latent z, and score held-out
                      TEST map RMSE / tau-boundary error / selection regret, aggregated over unique-(B,w)
                      TEST groups with training-seed bands (A-16). Emits results + the mandated figure.

Reuses the committed cores (build_scene/add_rod/apply_properties, sim.history, sim.identify.pool_temporal,
sim.distal_critic.{train_critic,Phi}, sim.sweep.settings) -- the k-prefix estimand is this module's own.
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path

import numpy as np
import torch
from torch import nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sim import adaptation_curve as ac
from sim.history import temporal_shape, drive_summary, FRAME_STEPS
from sim.identify import pool_temporal
from sim.distal_critic import train_critic, seed_all
from sim.sweep import settings

ROOT = Path(__file__).resolve().parent
MAN = ROOT / "manifests"
FIG = ROOT / "figures"
HIST = MAN / "adaptation_curve_lift_histories.npz"


# --------------------------------------------------------------------------- #
# stage: histories (GPU) -- NEW-seed lift temporal histories over all 17 grasp cells
# --------------------------------------------------------------------------- #
def extract_lift_histories(batch_size=120, only_grasp=None):
    from sim.scene import build_scene, add_straight_rod, add_moving_clamp, attach_moving_clamp, vertices
    from sim.material import apply_properties
    am = json.loads((MAN / "adaptation_curve_manifest.json").read_text())
    sm = json.loads((MAN / "hard_sweep_manifest.json").read_text())
    ss = {s["id"]: s for s in settings(sm)}
    integ = sm["integrator"]; interval = sm["interval"]; bounds = sm["stochastic_distribution"]
    ell = sm["grasp"]["ell"]; nvs = sm["grasp"]["n_vertices"]; drive_steps = sm.get("drive_steps", 360)
    universe = list(ac.TRAIN_GROUPS + ac.VAL_GROUPS + ac.TEST_GROUPS)
    grasps = [only_grasp] if only_grasp is not None else list(range(ac.LIFT_N_CELLS))
    templates = list(range(ac.N_TEMPLATES)); seeds = list(ac.SEED_BANKS["history"])
    specs = [dict(setting=s, grasp=g, template=t, seed=sd)
             for s in universe for g in grasps for t in templates for sd in seeds]
    start = time.time(); rec = []
    for gi in grasps:
        grp = [x for x in specs if x["grasp"] == gi]; nv = nvs[gi]; e_ell = float(ell[gi])
        for off in range(0, len(grp), batch_size):
            q = grp[off:off + batch_size]
            scene = build_scene(integ["dt"], integ["substeps"], integ["damping"], integ["angular_damping"])
            rod = add_straight_rod(scene, nv, interval, 1e7, .001, pos=(0, 0, .5)); box = add_moving_clamp(scene, (0, 0, .5))
            scene.build(n_envs=len(q), env_spacing=(2, 2))
            apply_properties(rod, np.array([ss[x["setting"]]["raw_E"] for x in q]), np.array([ss[x["setting"]]["mass"] for x in q]))
            attach_moving_clamp(rod, box)
            rand = [np.random.default_rng(x["seed"]) for x in q]
            draw = [dict(dx=r.uniform(*bounds["clamp_start_translation_xy_m"]), dy=r.uniform(*bounds["clamp_start_translation_xy_m"]),
                         dur=r.uniform(*bounds["motion_duration_multiplier"]), arc=r.uniform(*bounds["arc_multiplier"])) for r in rand]
            frames = {e: [] for e in range(len(q))}
            for step in range(drive_steps):
                pos = []
                for x, d in zip(q, draw):
                    t = sm["templates"][x["template"]]; u = min(1., (step + 1) / (drive_steps * d["dur"]))
                    s_ = u * u * (3 - 2 * u) if t["kind"] == "ease" else u
                    pos.append((d["dx"] * (1 - s_) + (t["arc"] * d["arc"] * np.sin(np.pi * s_) if t["kind"] == "arc" else 0), d["dy"] * (1 - s_), .5 + .2 * s_))
                box.set_pos(np.asarray(pos)); scene.step()
                if (step + 1) in FRAME_STEPS:
                    vv = vertices(rod)
                    for e in range(len(q)):
                        frames[e].append(vv[e])
            prev = vertices(rod)[:, 2:, :]; conv = False
            for _ in range(80):
                for _ in range(200):
                    scene.step()
                cur = vertices(rod)[:, 2:, :]; drift = float(np.max(np.linalg.norm(cur - prev, axis=-1))); prev = cur
                if drift < 2e-3:
                    conv = True; break
            if not conv:
                raise RuntimeError(f"adaptation-curve history batch not converged grasp={gi} off={off}")
            vv = vertices(rod)
            for e, x in enumerate(q):
                fr = frames[e] + [vv[e]]
                action = np.array([x["grasp"] / (ac.LIFT_N_CELLS - 1), e_ell], float)
                rec.append((x["setting"], x["grasp"], x["template"], x["seed"],
                            temporal_shape(fr), drive_summary(sm["templates"][x["template"]], x["seed"], bounds), action))
            print(json.dumps({"complete": len(rec), "total": len(specs), "grasp": gi}), flush=True)
    f = list(zip(*rec))
    out = (MAN / f"adaptation_curve_lift_histories_g{only_grasp}.npz") if only_grasp is not None else HIST
    np.savez_compressed(out, setting=np.array(f[0]), grasp=np.array(f[1]), template=np.array(f[2]), seed=np.array(f[3]),
                        shape=np.array(f[4]), proprio=np.array(f[5]), action=np.array(f[6]),
                        manifest_digest_of="adaptation_curve_manifest.json", rollout_count=len(rec))
    print(json.dumps({"output": str(out), "rollout_count": len(rec), "wall_clock_s": round(time.time() - start, 1)}))


def merge_histories():
    """Concatenate the per-grasp history shards (fresh-process extraction) into the final npz."""
    parts = sorted(MAN.glob("adaptation_curve_lift_histories_g*.npz"), key=lambda p: int(p.stem.split("_g")[-1]))
    assert len(parts) == ac.LIFT_N_CELLS, f"expected {ac.LIFT_N_CELLS} shards, got {len(parts)}"
    keys = ("setting", "grasp", "template", "seed", "shape", "proprio", "action")
    acc = {k: [] for k in keys}
    for p in parts:
        d = np.load(p, allow_pickle=True)
        for k in keys:
            acc[k].append(d[k])
    merged = {k: np.concatenate(acc[k]) for k in keys}
    np.savez_compressed(HIST, manifest_digest_of="adaptation_curve_manifest.json",
                        rollout_count=len(merged["grasp"]), **merged)
    print(json.dumps({"output": str(HIST), "rollout_count": int(len(merged["grasp"])), "shards": len(parts)}))


# --------------------------------------------------------------------------- #
# stage: sweep (CPU torch) -- per-k mean-pool distillation + metrics
# --------------------------------------------------------------------------- #
def _teacher_rows(sm, ss, ell, ng):
    sw = np.load(MAN / "addendum_sweep_results.npz")
    rows = []
    for sid in (ac.TRAIN_GROUPS + ac.VAL_GROUPS):
        for gi in range(ng):
            ix = (sw["bank"] == "selection") & (sw["setting"] == sid) & (sw["grasp"] == gi)
            if not ix.any():
                continue
            cand = [(float(sw["success"][ix & (sw["template"] == t)].mean()),
                     float(sw["J"][ix & (sw["template"] == t)].mean()), int(t)) for t in sorted(set(sw["template"][ix].tolist()))]
            sp, jj, _ = max(cand, key=lambda x: (x[0], x[1], -x[2]))
            rows.append((sid, np.array([gi / (ng - 1), float(ell[gi])]), sp, jj, 1.0 if sp > 0 else 0.0))
    return rows


def _full_feat(hz, i):
    return np.r_[hz["proprio"][i], pool_temporal(hz["shape"][i]), hz["action"][i]]


def sweep():
    sm = json.loads((MAN / "hard_sweep_manifest.json").read_text()); ss = {s["id"]: s for s in settings(sm)}
    ell = np.array(sm["grasp"]["ell"]); ng = len(ell)
    land = {x["id"]: x for x in json.loads((MAN / "addendum_landscape.json").read_text())["settings"]}
    TRAIN, VAL, TEST = list(ac.TRAIN_GROUPS), list(ac.VAL_GROUPS), list(ac.TEST_GROUPS)
    prop = {k: (v["B_eff"], v["w"]) for k, v in ss.items()}
    raw = {k: np.log10([v[0], v[1]]) for k, v in prop.items()}
    a = np.array([raw[k] for k in TRAIN]); pm, psd = a.mean(0), a.std(0); props = {k: (v - pm) / psd for k, v in raw.items()}
    rows = _teacher_rows(sm, ss, ell, ng); gf = np.c_[np.arange(ng) / (ng - 1), ell]
    hz = np.load(HIST, allow_pickle=True)
    S = hz["setting"].astype(str); Gr = hz["grasp"].astype(int); Tp = hz["template"].astype(int); Sd = hz["seed"].astype(int)
    schedule = ac.schedule_lift()                              # frozen S_lift: [(cell, ell, template), ...]
    hist_seeds = list(ac.SEED_BANKS["history"])

    def prefix_feat(sid, seed, k):
        """MEAN-POOL of the first k scheduled interactions' 42-D features for (setting, history seed)."""
        if k <= 0:
            return None
        feats = []
        for j in range(k):
            cell, _, tmpl = schedule[j]
            idx = np.where((S == sid) & (Gr == cell) & (Tp == tmpl) & (Sd == seed))[0]
            if idx.size == 0:
                raise RuntimeError(f"missing history rollout {sid} cell={cell} tmpl={tmpl} seed={seed}")
            feats.append(_full_feat(hz, idx[0]))
        return np.mean(feats, axis=0)

    def curve_from_z(q, z):
        with torch.no_grad():
            sl, _ = q(torch.tensor(gf, dtype=torch.double), torch.tensor(np.tile(z, (ng, 1)), dtype=torch.double))
            return torch.sigmoid(sl).numpy()

    def metrics_for_curves(curve_of_setting):
        rmses, berrs, regrets = [], [], []
        for sid in TEST:
            pS = curve_of_setting(sid); measS = np.array(land[sid]["success_rate"])
            rmses.append(ac.map_rmse(pS, measS))
            be = ac.boundary_index_error(pS, measS, ell)
            if be is not None:
                berrs.append(be)
            regrets.append(ac.selection_regret(pS, measS))
        return dict(map_rmse=ac.aggregate_unique_groups(rmses), boundary_err=ac.aggregate_unique_groups(berrs),
                    selection_regret=ac.aggregate_unique_groups(regrets))

    train_seeds = list(ac.SEED_BANKS["training"])
    per_k = {k: {"teacher": [], "blind": [], "student": [], "sysid": []} for k in ac.K_AXIS}
    for tseed in train_seeds:
        phi, q = train_critic(rows, props, VAL, blind=False, train_ids=TRAIN, seed=tseed)
        _, qb = train_critic(rows, props, VAL, blind=True, train_ids=TRAIN, seed=tseed)
        teacher_curves = {sid: curve_from_z(q, phi(torch.tensor(props[sid][None], dtype=torch.double)).detach().numpy()[0]) for sid in TEST}
        blind_curves = {sid: curve_from_z(qb, np.zeros(4)) for sid in TEST}
        tm = metrics_for_curves(lambda sid: teacher_curves[sid])
        bm = metrics_for_curves(lambda sid: blind_curves[sid])
        for k in ac.K_AXIS:
            per_k[k]["teacher"].append(tm); per_k[k]["blind"].append(bm)
        # per-k student (mean-pool k-prefix distillation) + leak-free sysID-per-k
        for k in ac.K_AXIS:
            if k == 0:                                          # k=0 == blind
                per_k[k]["student"].append(bm); per_k[k]["sysid"].append(bm); continue
            # build (setting, seed) k-prefix summaries
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
            def student_curve(sid):
                with torch.no_grad():
                    z = st(torch.tensor((X[sid] - hm) / hs, dtype=torch.double)).mean(0).numpy()
                return curve_from_z(q, z)
            per_k[k]["student"].append(metrics_for_curves(student_curve))
            # leak-free sysID-per-k: ridge (closed form) from k-prefix features -> z, fit on TRAIN
            lam = 1.0; A = Xtrn.numpy(); W = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ Ytr)
            def sysid_curve(sid):
                z = ((X[sid] - hm) / hs) @ W
                return curve_from_z(q, z.mean(0))
            per_k[k]["sysid"].append(metrics_for_curves(sysid_curve))

    def band(metric_dicts, field):
        means = [m[field]["mean"] for m in metric_dicts]
        return dict(mean=float(np.mean(means)), seed_std=float(np.std(means)), n_seeds=len(means))

    curve = {b: {"map_rmse": [band(per_k[k][b], "map_rmse") for k in ac.K_AXIS],
                 "selection_regret": [band(per_k[k][b], "selection_regret") for k in ac.K_AXIS],
                 "boundary_err": [band(per_k[k][b], "boundary_err") for k in ac.K_AXIS]}
             for b in ("teacher", "blind", "student", "sysid")}
    results = dict(item="1_k_interaction_adaptation_curve", manifest_digest_of="adaptation_curve_manifest.json",
                   k_axis=list(ac.K_AXIS), task="lift_and_clear_primary_17cell", train_seeds=train_seeds,
                   history_seeds=hist_seeds, test_groups=TEST, curve=curve,
                   reference_endpoints_DESCRIPTIVE={"lift_map_rmse_task": 0.043304, "teacher": 0.122776, "blind": 0.327524},
                   lift_selection_regret_degenerate=True,
                   acceptance_note="PROTOCOL + HONESTY: frozen protocol executed; ordering (blind=k0, teacher flat ceiling, sysID shown) reported; reference endpoints + monotonicity DESCRIPTIVE only; a flat/non-monotone curve is a valid honest outcome.")
    (MAN / "adaptation_curve_results.json").write_text(json.dumps(results, indent=2, default=float))
    np.savez(MAN / "adaptation_curve_sweep_results.npz",
             **{f"{b}_map_rmse_mean": np.array([band(per_k[k][b], "map_rmse")["mean"] for k in ac.K_AXIS]) for b in curve})
    _figure(curve)
    print("SWEEP done. k-axis:", list(ac.K_AXIS))
    for b in ("teacher", "blind", "student", "sysid"):
        print(" ", b, "map_rmse:", [round(x["mean"], 4) for x in curve[b]["map_rmse"]])
    return results


def _figure(curve):
    FIG.mkdir(parents=True, exist_ok=True)
    k = list(ac.K_AXIS); xs = [max(kk, 0.5) for kk in k]      # log-friendly x (k=0 at 0.5)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    for b, c in (("teacher", "tab:green"), ("blind", "tab:red"), ("student", "tab:blue"), ("sysid", "tab:orange")):
        m = [x["mean"] for x in curve[b]["map_rmse"]]; e = [x["seed_std"] for x in curve[b]["map_rmse"]]
        ax[0].errorbar(xs, m, yerr=e, fmt="o-", color=c, capsize=3, label=b)
    ax[0].set_xscale("log"); ax[0].set_xlabel("k interactions (k=0 = blind prior)"); ax[0].set_ylabel("held-out map RMSE")
    ax[0].set_title("k-interaction adaptation curve (lift-and-clear, 17-cell)"); ax[0].legend(fontsize=7)
    ax[0].set_xticks(xs); ax[0].set_xticklabels([str(kk) for kk in k])
    for b, c in (("teacher", "tab:green"), ("blind", "tab:red"), ("student", "tab:blue"), ("sysid", "tab:orange")):
        m = [x["mean"] for x in curve[b]["selection_regret"]]
        ax[1].plot(xs, m, "o-", color=c, label=b)
    ax[1].set_xscale("log"); ax[1].set_xlabel("k interactions"); ax[1].set_ylabel("selection regret")
    ax[1].set_title("Selection regret vs k (lift regret is DEGENERATE -- reported honestly)")
    ax[1].set_xticks(xs); ax[1].set_xticklabels([str(kk) for kk in k]); ax[1].legend(fontsize=7)
    fig.suptitle("Item 1: interaction-COUNT adaptation curve (A-22; NOT frame-truncation)")
    fig.tight_layout(); fig.savefig(FIG / "adaptation_curve.png", dpi=140); plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--stage", choices=["histories", "merge", "sweep"], required=True)
    p.add_argument("--grasp", type=int, default=None)
    a = p.parse_args()
    if a.stage == "histories":
        extract_lift_histories(only_grasp=a.grasp)
    elif a.stage == "merge":
        merge_histories()
    else:
        sweep()
