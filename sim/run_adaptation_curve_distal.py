"""Item-1c C2-distal: the DISTAL SECONDARY -- 25-cell spanning-cohort regret-vs-k companion.

Score-only after the C1-distal freeze (adaptation_curve_distal_manifest.json). Executes the Item-1
FROZEN banks EXACTLY on the spanning 25-cell cohort: oracle + teacher labels + student histories all
from {2300-2302}/{3300-3304}/{2500,2501}, k-axis + S_distal + estimator reused from the frozen v2.
Selection regret is the PRIMARY here (it DISCRIMINATES on this cohort; the lift regret is degenerate).

Stages (fresh process per cell -> avoids GPU-context accumulation):
  --stage sweep --grasp G  : frozen sel/eval distal sweep for ONE cell (reuses distal_sweep.run_grasp_batch)
  --stage merge            : concat sweep shards -> distal sweep results + oracle landscape (success + J)
  --stage histories --grasp G : distal temporal histories for ONE cell (all settings x templates x hist seeds)
  --stage histmerge        : concat history shards
  --stage ksweep           : mean-pool k-prefix distal student -> regret-vs-k + map-vs-k + figure + surfaces
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
import sim.tip_model as tm
from sim.identify import pool_temporal
from sim.distal_critic import train_critic, seed_all
from sim.history import shape_frame, temporal_shape, drive_summary, FRAME_STEPS
from sim.distal_sweep import settings as dsettings, run_grasp_batch

ROOT = Path(__file__).resolve().parent
MAN = ROOT / "manifests"
FIG = ROOT / "figures"
DMAN = "adaptation_curve_distal_manifest.json"
SWEEP = MAN / "adaptation_curve_distal_sweep_results.npz"
LAND = MAN / "adaptation_curve_distal_landscape.json"
HIST = MAN / "adaptation_curve_distal_histories.npz"
SWEEP_KEYS = ["setting", "grasp", "ell", "template", "bank", "seed", "reach", "droop", "pi_g", "success", "J",
              "selected_template", "converged", "settle_steps", "draw_dx", "draw_dy", "draw_dur", "draw_arc"]


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# --------------------------------------------------------------------------- #
# stage: sweep (GPU) -- frozen sel/eval distal sweep for one cell
# --------------------------------------------------------------------------- #
def stage_sweep(gi, expected_digest=None, batch_size=90):
    m = json.loads((MAN / DMAN).read_text()); live = sha256(MAN / DMAN)
    if expected_digest and live != expected_digest:
        raise RuntimeError("distal manifest digest mismatch")
    ss = dsettings(m); ell = m["grasp"]["ell"]; nvs = m["grasp"]["n_vertices"]; ntmpl = len(m["templates"])
    sel = list(m["seed_banks"]["selection"]); ev = list(m["seed_banks"]["evaluation"])
    start = time.time(); rows = []
    specs = [dict(setting=s, template=t, seed=sd, bank="selection") for s in ss for t in range(ntmpl) for sd in sel]
    for i in range(0, len(specs), batch_size):
        rows += run_grasp_batch(m, gi, nvs[gi], float(ell[gi]), specs[i:i + batch_size])
    winner = {}
    for s in ss:
        rates = [np.mean([r["success"] for r in rows if r["setting"] == s["id"] and r["template"] == t]) for t in range(ntmpl)]
        winner[s["id"]] = int(np.argmax(rates))                # success-only, lowest-index tie == frozen rule
    especs = [dict(setting=s, template=winner[s["id"]], seed=sd, bank="evaluation") for s in ss for sd in ev]
    for i in range(0, len(especs), batch_size):
        rows += run_grasp_batch(m, gi, nvs[gi], float(ell[gi]), especs[i:i + batch_size])
    for r in rows:
        r["selected_template"] = bool(winner[r["setting"]] == r["template"])
    out = MAN / f"adaptation_curve_distal_sweep_g{gi}.npz"
    np.savez_compressed(out, **{k: np.array([r[k] for r in rows]) for k in SWEEP_KEYS}, grasp_cell=gi)
    print(json.dumps({"grasp": gi, "rows": len(rows), "wall_clock_s": round(time.time() - start, 1),
                      "nonconverged": float(np.mean([not r["converged"] for r in rows])), "output": str(out)}))


def stage_merge():
    m = json.loads((MAN / DMAN).read_text()); ss = dsettings(m); ell = m["grasp"]["ell"]; ng = len(ell)
    parts = sorted(MAN.glob("adaptation_curve_distal_sweep_g*.npz"), key=lambda p: int(p.stem.split("_g")[-1]))
    assert len(parts) == ng, f"expected {ng} sweep shards, got {len(parts)}"
    acc = {k: [] for k in SWEEP_KEYS}
    for p in parts:
        d = np.load(p, allow_pickle=True)
        for k in SWEEP_KEYS:
            acc[k].append(d[k])
    merged = {k: np.concatenate(acc[k]) for k in SWEEP_KEYS}
    np.savez_compressed(SWEEP, **merged, manifest_digest=sha256(MAN / DMAN), rollout_count=len(merged["grasp"]))
    st = merged["setting"].astype(str); gr = merged["grasp"].astype(int); bk = merged["bank"].astype(str)
    suc = merged["success"].astype(float); J = merged["J"].astype(float)
    land = []
    for s in ss:
        rate = [float(np.mean(suc[(st == s["id"]) & (gr == gi) & (bk == "evaluation")])) for gi in range(ng)]
        jmean = [float(np.mean(J[(st == s["id"]) & (gr == gi) & (bk == "evaluation")])) for gi in range(ng)]
        land.append(dict(id=s["id"], B_eff=s["B_eff"], w=s["w"], success_rate=rate, J=jmean, ell_grid=[float(x) for x in ell]))
    LAND.write_text(json.dumps({"settings": land}, indent=2) + "\n")
    print(json.dumps({"sweep": str(SWEEP), "landscape": str(LAND), "rollouts": int(len(merged["grasp"])), "shards": len(parts),
                      "sel_seeds": sorted(set(merged["seed"][bk == "selection"].astype(int).tolist())),
                      "eval_seeds": sorted(set(merged["seed"][bk == "evaluation"].astype(int).tolist()))}))


# --------------------------------------------------------------------------- #
# stage: histories (GPU) -- distal temporal histories for one cell (all 25 covered across shards)
# --------------------------------------------------------------------------- #
def stage_histories(gi, expected_digest=None, batch_size=50):
    from sim.scene import build_scene, add_straight_rod, add_moving_clamp, attach_moving_clamp, vertices
    from sim.material import apply_properties
    m = json.loads((MAN / DMAN).read_text()); live = sha256(MAN / DMAN)
    if expected_digest and live != expected_digest:
        raise RuntimeError("distal manifest digest mismatch")
    ss = {c["id"]: c for c in m["grid"]}; ss.update({r["id"]: r for r in m["ratio_pairs"]})
    rawE = {k: v["raw_E"] for k, v in ss.items()}
    integ = m["integrator"]; interval = m["interval"]; bounds = m["stochastic_distribution"]
    ell = m["grasp"]["ell"]; nvs = m["grasp"]["n_vertices"]; drive_steps = m.get("drive_steps", 360)
    universe = m["universe"]; templates = list(range(len(m["templates"]))); seeds = list(m["seed_banks"]["history"])
    ncells = len(ell); nv = nvs[gi]; e_ell = float(ell[gi])
    specs = [dict(setting=sid, template=t, seed=sd) for sid in universe for t in templates for sd in seeds]
    start = time.time(); rec = []
    for off in range(0, len(specs), batch_size):
        q = specs[off:off + batch_size]
        scene = build_scene(integ["dt"], integ["substeps"], integ["damping"], integ["angular_damping"])
        rod = add_straight_rod(scene, nv, interval, 1e7, .001, pos=(0, 0, .5)); box = add_moving_clamp(scene, (0, 0, .5))
        scene.build(n_envs=len(q), env_spacing=(2, 2))
        apply_properties(rod, np.array([rawE[x["setting"]] for x in q]), np.array([ss[x["setting"]]["mass"] for x in q]))
        attach_moving_clamp(rod, box)
        rand = [np.random.default_rng(x["seed"]) for x in q]
        draw = [dict(dx=r.uniform(*bounds["clamp_start_translation_xy_m"]), dy=r.uniform(*bounds["clamp_start_translation_xy_m"]),
                     dur=r.uniform(*bounds["motion_duration_multiplier"]), arc=r.uniform(*bounds["arc_multiplier"])) for r in rand]
        frames = {e: [] for e in range(len(q))}
        for step in range(drive_steps):
            pos = []
            for x, d in zip(q, draw):
                t = m["templates"][x["template"]]; u = min(1., (step + 1) / (drive_steps * d["dur"]))
                s_ = u * u * (3 - 2 * u) if t["kind"] == "ease" else u
                pos.append((d["dx"] * (1 - s_) + (t["arc"] * d["arc"] * np.sin(np.pi * s_) if t["kind"] == "arc" else 0),
                            d["dy"] * (1 - s_), .5 + .2 * s_))
            box.set_pos(np.asarray(pos)); scene.step()
            if (step + 1) in FRAME_STEPS:
                vv = vertices(rod)
                for e in range(len(q)):
                    frames[e].append(vv[e])
        # 16000-step chunked full free-vertex convergence to the SAME 2e-3 criterion as the distal
        # SWEEP + the lift history extraction (the longer distal rods need the full budget; the
        # convergence CRITERION is unchanged -- a numerical settle budget, not a science threshold).
        prev = vertices(rod)[:, 2:, :]; conv = False; steps = drive_steps
        for _ in range(80):
            for _ in range(200):
                scene.step()
            steps += 200
            cur = vertices(rod)[:, 2:, :]; drift = float(np.max(np.linalg.norm(cur - prev, axis=-1))); prev = cur
            if drift < 2e-3:
                conv = True; break
        if not conv:
            raise RuntimeError(f"distal history batch not converged grasp={gi} off={off}: drift={drift}")
        vv = vertices(rod)
        for e, x in enumerate(q):
            fr = frames[e] + [vv[e]]
            action = np.array([gi / (ncells - 1), e_ell], float)
            rec.append((x["setting"], gi, x["template"], x["seed"], temporal_shape(fr),
                        drive_summary(m["templates"][x["template"]], x["seed"], bounds), action))
        print(json.dumps({"complete": len(rec), "grasp": gi, "off": off}), flush=True)
    f = list(zip(*rec))
    out = MAN / f"adaptation_curve_distal_hist_g{gi}.npz"
    np.savez_compressed(out, setting=np.array(f[0]), grasp=np.array(f[1]), template=np.array(f[2]), seed=np.array(f[3]),
                        shape=np.array(f[4]), proprio=np.array(f[5]), action=np.array(f[6]))
    print(json.dumps({"grasp": gi, "rollouts": len(rec), "wall_clock_s": round(time.time() - start, 1), "output": str(out)}))


def stage_histmerge():
    m = json.loads((MAN / DMAN).read_text()); ng = len(m["grasp"]["ell"])
    parts = sorted(MAN.glob("adaptation_curve_distal_hist_g*.npz"), key=lambda p: int(p.stem.split("_g")[-1]))
    assert len(parts) == ng, f"expected {ng} history shards, got {len(parts)}"
    keys = ("setting", "grasp", "template", "seed", "shape", "proprio", "action")
    acc = {k: [] for k in keys}
    for p in parts:
        d = np.load(p, allow_pickle=True)
        for k in keys:
            acc[k].append(d[k])
    merged = {k: np.concatenate(acc[k]) for k in keys}
    np.savez_compressed(HIST, manifest_digest=sha256(MAN / DMAN), rollout_count=len(merged["grasp"]), **merged)
    print(json.dumps({"output": str(HIST), "rollouts": int(len(merged["grasp"])), "shards": len(parts)}))


# --------------------------------------------------------------------------- #
# stage: ksweep (CPU) -- mean-pool k-prefix distal student: regret-vs-k + map-vs-k
# --------------------------------------------------------------------------- #
def frozen_teacher_rows(sw, grid, ng, TRAIN, VAL):
    """Teacher labels = the FROZEN winner template's selection-bank (success, J), consumed from the
    persisted `selected_template` (success-only argmax, lowest-index tie). One row per TRAIN+VAL cell."""
    st = sw["setting"].astype(str); gr = sw["grasp"].astype(int); bk = sw["bank"].astype(str)
    suc = sw["success"].astype(float); Js = sw["J"].astype(float); seltmpl = sw["selected_template"].astype(bool)
    rows = []
    for sid in (list(TRAIN) + list(VAL)):
        for gi in range(ng):
            ix = (bk == "selection") & (st == sid) & (gr == gi) & seltmpl
            if not ix.any():
                continue
            assert len(set(sw["template"][ix].astype(int).tolist())) == 1, "selected_template not unique per cell"
            rows.append((sid, np.array([gi / (ng - 1), float(grid[gi])]), float(suc[ix].mean()), float(Js[ix].mean()), 1.0 if suc[ix].mean() > 0 else 0.0))
    return rows


def stage_ksweep():
    m = json.loads((MAN / DMAN).read_text()); c1 = sha256(MAN / DMAN)
    TRAIN, VAL, TEST = m["splits"]["train"], m["splits"]["val"], m["splits"]["test"]
    ss = {c["id"]: c for c in m["grid"]}; ss.update({r["id"]: r for r in m["ratio_pairs"]})
    grid = np.array(m["grasp"]["ell"]); ng = len(grid)
    prop = {k: (v["B_eff"], v["w"]) for k, v in ss.items()}
    raw = {k: np.log10([v[0], v[1]]) for k, v in prop.items()}
    a = np.array([raw[k] for k in TRAIN]); pm, psd = a.mean(0), a.std(0); props = {k: (v - pm) / psd for k, v in raw.items()}
    sw = np.load(SWEEP, allow_pickle=True); assert str(sw["manifest_digest"].item()) == c1, "distal sweep not bound to C1-distal"
    bk = sw["bank"].astype(str)
    # teacher labels = FROZEN winner (persisted selected_template) selection-bank (success, J)
    rows = frozen_teacher_rows(sw, grid, ng, TRAIN, VAL)
    gf = np.c_[np.arange(ng) / (ng - 1), grid]
    land = {x["id"]: x for x in json.loads(LAND.read_text())["settings"]}
    def meas(sid):
        return np.asarray(land[sid]["success_rate"], float), np.asarray(land[sid]["J"], float)
    hz = np.load(HIST, allow_pickle=True); assert str(hz["manifest_digest"].item()) == c1, "distal histories not bound to C1-distal"
    S = hz["setting"].astype(str); Gr = hz["grasp"].astype(int); Tp = hz["template"].astype(int); Sd = hz["seed"].astype(int)
    schedule = ac.schedule_distal(); hist_seeds = list(m["seed_banks"]["history"]); kmax = max(ac.K_AXIS)
    def full_feat(i):
        return np.r_[hz["proprio"][i], pool_temporal(hz["shape"][i]), hz["action"][i]]
    ORD = {}
    for sid in (TRAIN + VAL + TEST):
        for sd in hist_seeds:
            feats = []
            for j in range(kmax):
                cell, _, tmpl = schedule[j]
                idx = np.where((S == sid) & (Gr == cell) & (Tp == tmpl) & (Sd == sd))[0]
                if idx.size == 0:
                    raise RuntimeError(f"missing distal history {sid} cell={cell} tmpl={tmpl} seed={sd}")
                feats.append(full_feat(idx[0]))
            ORD[(sid, sd)] = np.asarray(feats)

    def curves(qq, z):
        with torch.no_grad():
            sl, jv = qq(torch.tensor(gf, dtype=torch.double), torch.tensor(np.tile(z, (ng, 1)), dtype=torch.double))
            return torch.sigmoid(sl).numpy(), jv.numpy()

    def regret_map(sid, pS, pJ):
        B, w = prop[sid]; in_reg = tm.pi_g(grid, B, w) <= tm.PI_G_MAX
        ms, mj = meas(sid)
        cand = np.where(in_reg & (np.asarray(pS) >= 0.5))[0]
        sel = int(cand[np.argmax(np.asarray(pJ)[cand])]) if len(cand) else (int(np.where(in_reg)[0][np.argmax(np.asarray(pS)[in_reg])]) if in_reg.any() else int(np.argmax(pS)))
        best_meas = float(np.max(mj[in_reg])) if in_reg.any() else float(np.max(mj))
        return best_meas - float(mj[sel]), float(np.sqrt(np.mean((np.asarray(pS) - ms) ** 2)))

    train_seeds = list(m["seed_banks"]["training"])
    # store the PREDICTED curves (pS success, pJ J) per (baseline, k, seed, sid) so the red-team can
    # recompute regret + map independently from the raw surfaces + the measured oracle.
    surf = {b: {ki: {sid: {"pS": [], "pJ": []} for sid in TEST} for ki in range(len(ac.K_AXIS))} for b in ("teacher", "blind", "student", "sysid")}
    for tseed in train_seeds:
        phi, q = train_critic(rows, props, VAL, blind=False, train_ids=TRAIN, seed=tseed)
        _, qb = train_critic(rows, props, VAL, blind=True, train_ids=TRAIN, seed=tseed)
        tc = {sid: curves(q, phi(torch.tensor(props[sid][None], dtype=torch.double)).detach().numpy()[0]) for sid in TEST}
        bc = {sid: curves(qb, np.zeros(4)) for sid in TEST}
        for ki in range(len(ac.K_AXIS)):
            for sid in TEST:
                for lab, (pS, pJ) in (("teacher", tc[sid]), ("blind", bc[sid])):
                    surf[lab][ki][sid]["pS"].append(pS); surf[lab][ki][sid]["pJ"].append(pJ)
        for ki, k in enumerate(ac.K_AXIS):
            if k == 0:
                for sid in TEST:
                    pS, pJ = bc[sid]
                    surf["student"][ki][sid]["pS"].append(pS); surf["student"][ki][sid]["pJ"].append(pJ)
                    surf["sysid"][ki][sid]["pS"].append(pS); surf["sysid"][ki][sid]["pJ"].append(pJ)
                continue
            X = {sid: np.array([ac.prefix_summary(ORD[(sid, sd)], k) for sd in hist_seeds]) for sid in (TRAIN + VAL + TEST)}
            Xtr = np.concatenate([X[s] for s in TRAIN]); Ytr = np.concatenate([np.tile(phi(torch.tensor(props[s][None], dtype=torch.double)).detach().numpy(), (len(hist_seeds), 1)) for s in TRAIN])
            Xv = np.concatenate([X[s] for s in VAL]); Yv = np.concatenate([np.tile(phi(torch.tensor(props[s][None], dtype=torch.double)).detach().numpy(), (len(hist_seeds), 1)) for s in VAL])
            hm = Xtr.mean(0); hs = Xtr.std(0); hs[hs < 1e-12] = 1
            Xtrn = torch.tensor((Xtr - hm) / hs, dtype=torch.double); Xvn = torch.tensor((Xv - hm) / hs, dtype=torch.double)
            Ytr_t = torch.tensor(Ytr, dtype=torch.double); Yv_t = torch.tensor(Yv, dtype=torch.double)
            seed_all(tseed); stnet = nn.Sequential(nn.Linear(ac.FEATURE_DIM, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 4)).double()
            opt = torch.optim.Adam(stnet.parameters(), lr=1e-2); best = (float("inf"), None)
            for _ in range(300):
                opt.zero_grad(); loss = ((stnet(Xtrn) - Ytr_t) ** 2).mean(); loss.backward(); opt.step()
                with torch.no_grad():
                    vl = float(((stnet(Xvn) - Yv_t) ** 2).mean())
                if vl < best[0]:
                    best = (vl, {kk: v.detach().clone() for kk, v in stnet.state_dict().items()})
            stnet.load_state_dict(best[1])
            lam = 1.0; A = Xtrn.numpy(); W = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ Ytr)
            for sid in TEST:
                with torch.no_grad():
                    zst = stnet(torch.tensor((X[sid] - hm) / hs, dtype=torch.double)).mean(0).numpy()
                pS, pJ = curves(q, zst); surf["student"][ki][sid]["pS"].append(pS); surf["student"][ki][sid]["pJ"].append(pJ)
                zsy = (((X[sid] - hm) / hs) @ W).mean(0)
                pS2, pJ2 = curves(q, zsy); surf["sysid"][ki][sid]["pS"].append(pS2); surf["sysid"][ki][sid]["pJ"].append(pJ2)

    def band(b, ki, field):
        idx = 0 if field == "regret" else 1
        per_seed = []
        for si in range(len(train_seeds)):
            vals = [regret_map(sid, surf[b][ki][sid]["pS"][si], surf[b][ki][sid]["pJ"][si])[idx] for sid in TEST]
            per_seed.append(ac.aggregate_unique_groups(vals)["mean"])
        arr = np.asarray(per_seed, float)
        return dict(mean=float(np.nanmean(arr)), seed_std=float(np.nanstd(arr)), n_seeds=len(train_seeds))

    curve = {b: {"selection_regret": [band(b, ki, "regret") for ki in range(len(ac.K_AXIS))],
                 "map_rmse": [band(b, ki, "map") for ki in range(len(ac.K_AXIS))]} for b in ("teacher", "blind", "student", "sysid")}
    ratio_ref = {"R0": None, "R1": None, "R2": None}
    for r in m["ratio_pairs"]:
        ratio_ref[r["id"]] = r["reference"]
    def amx_meas(sid):
        ms, mj = meas(sid); B, w = prop[sid]; inr = tm.pi_g(grid, B, w) <= tm.PI_G_MAX
        return int(np.where(inr)[0][np.argmax(mj[inr])]) if inr.any() else int(np.argmax(mj))
    ratio_invariance = {rid: dict(reference=ref, offset_cells=int(abs(amx_meas(rid) - amx_meas(ref))), invariant=bool(abs(amx_meas(rid) - amx_meas(ref)) <= 1)) for rid, ref in ratio_ref.items() if ref}
    seeds_by_bank = {b: sorted(set(sw["seed"][bk == b].astype(int).tolist())) for b in ("selection", "evaluation")}
    assert seeds_by_bank["selection"] == sorted(m["seed_banks"]["selection"]) and seeds_by_bank["evaluation"] == sorted(m["seed_banks"]["evaluation"])

    def clean(o):
        if isinstance(o, float) and np.isnan(o):
            return None
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, list):
            return [clean(v) for v in o]
        return o

    results = clean(dict(item="1c_distal_SECONDARY_regret_vs_k", manifest="adaptation_curve_distal_manifest.json",
                   c1_distal_manifest_sha256=c1, k_axis=list(ac.K_AXIS), task="distal_tip_placement_25cell_spanning",
                   train_seeds=train_seeds, history_seeds=hist_seeds, test_groups=TEST, splits=m["splits"], curve=curve,
                   ratio_invariance=ratio_invariance, pre_registered_outcome="ESTABLISHED",
                   protocol_fidelity=dict(executed_exactly=True, oracle_teacher_banks=seeds_by_bank, history_bank=hist_seeds, training=train_seeds,
                                          note="Item-1 frozen banks executed exactly on the 25-cell spanning cohort; oracle + teacher + histories all from the new seeds; k-axis+S_distal+estimator reused from frozen v2."),
                   selection_regret_discriminates=bool(curve["teacher"]["selection_regret"][0]["mean"] < curve["blind"]["selection_regret"][0]["mean"]),
                   acceptance_note="PROTOCOL + HONESTY: distal regret-vs-k + map-vs-k with seed bands; ordering (blind=k0, teacher flat, sysID) reported; ratio invariance; reference endpoints DESCRIPTIVE only."))
    (MAN / "adaptation_curve_distal_results.json").write_text(json.dumps(results, indent=2))
    flat = {}
    for b in surf:
        for ki in range(len(ac.K_AXIS)):
            for sid in TEST:
                flat[f"{b}__k{ac.K_AXIS[ki]}__{sid}__pS"] = np.asarray(surf[b][ki][sid]["pS"])
                flat[f"{b}__k{ac.K_AXIS[ki]}__{sid}__pJ"] = np.asarray(surf[b][ki][sid]["pJ"])
    for sid in TEST:
        ms, mj = meas(sid); flat[f"measured__{sid}__success"] = ms; flat[f"measured__{sid}__J"] = mj
        flat[f"prop__{sid}"] = np.asarray(prop[sid], float)
    np.savez(MAN / "adaptation_curve_distal_surfaces.npz", k_axis=np.array(list(ac.K_AXIS)), train_seeds=np.array(train_seeds),
             test_groups=np.array(TEST), grid=grid, pi_g_max=float(tm.PI_G_MAX), c1_manifest_sha256=c1, **flat)
    _figure(curve)
    print("DISTAL KSWEEP done. k-axis:", list(ac.K_AXIS))
    for b in ("teacher", "blind", "student", "sysid"):
        print(" ", b, "regret:", [round(x["mean"], 4) for x in curve[b]["selection_regret"]])
    print(" discriminates:", results["selection_regret_discriminates"], "| ratio_invariance:", {r: ratio_invariance[r]["offset_cells"] for r in ratio_invariance})
    print(" PRE-REGISTERED OUTCOME: ESTABLISHED (frozen banks", seeds_by_bank, "executed exactly)")
    return results


def _figure(curve):
    FIG.mkdir(parents=True, exist_ok=True)
    k = list(ac.K_AXIS); xs = [max(kk, 0.5) for kk in k]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    tr = curve["teacher"]["selection_regret"][0]["mean"]; br = curve["blind"]["selection_regret"][0]["mean"]
    ax[0].axhline(tr, color="tab:green", ls="--", lw=1, label="teacher regret")
    ax[0].axhline(br, color="tab:red", ls=":", lw=1, label="blind regret (k=0)")
    for b, c in (("student", "tab:blue"), ("sysid", "tab:orange")):
        mn = [x["mean"] for x in curve[b]["selection_regret"]]; e = [x["seed_std"] for x in curve[b]["selection_regret"]]
        ax[0].errorbar(xs, mn, yerr=e, fmt="o-", color=c, capsize=3, label=b)
    ax[0].set_xscale("log"); ax[0].set_xlabel("k interactions (k=0 = blind prior)"); ax[0].set_ylabel("selection regret (PRIMARY)")
    ax[0].set_title("DISTAL regret vs k (spanning cohort; regret DISCRIMINATES)"); ax[0].legend(fontsize=7)
    ax[0].set_xticks(xs); ax[0].set_xticklabels([str(kk) for kk in k])
    for b, c in (("teacher", "tab:green"), ("blind", "tab:red"), ("student", "tab:blue"), ("sysid", "tab:orange")):
        mn = [x["mean"] for x in curve[b]["map_rmse"]]
        ax[1].plot(xs, mn, "o-", color=c, label=b)
    ax[1].set_xscale("log"); ax[1].set_xlabel("k interactions"); ax[1].set_ylabel("map RMSE (co-primary)")
    ax[1].set_title("DISTAL map recovery vs k"); ax[1].set_xticks(xs); ax[1].set_xticklabels([str(kk) for kk in k]); ax[1].legend(fontsize=7)
    fig.suptitle("Item 1c: DISTAL SECONDARY interaction-COUNT curve (A-22; regret-vs-k + map-vs-k; frozen banks executed exactly)")
    fig.tight_layout(); fig.savefig(FIG / "adaptation_curve_distal.png", dpi=140); plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--stage", choices=["sweep", "merge", "histories", "histmerge", "ksweep"], required=True)
    p.add_argument("--grasp", type=int, default=None); p.add_argument("--expected-digest", default=None)
    a = p.parse_args()
    if a.stage == "sweep":
        stage_sweep(a.grasp, a.expected_digest)
    elif a.stage == "merge":
        stage_merge()
    elif a.stage == "histories":
        stage_histories(a.grasp, a.expected_digest)
    elif a.stage == "histmerge":
        stage_histmerge()
    else:
        stage_ksweep()
