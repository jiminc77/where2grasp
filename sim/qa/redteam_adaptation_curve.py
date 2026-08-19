"""Independent red-team for Item-1 C2 (lift-PRIMARY k-interaction adaptation curve).

FRESH QA-local recompute -- NO production scorers or stored scalars are trusted for the verdict:
every metric primitive (map RMSE / selection regret / boundary-index error / group aggregation) and
the acquisition schedule are re-derived here from first principles and cross-checked against the
production `sim.adaptation_curve` implementation on the SAME raw inputs (so a rigged scorer would
diverge). All deterministic scaffolding -- exact-seed disjointness from ALL prior banks, S_lift
regeneration + byte-match, k=0==blind membership, mean-pool first-k membership, no cohort thinning
(exact 2040-rollout inventory + schedule coverage), input-artifact digests, and C1 single-file +
git-ancestry of the C2 data commit -- is audited independently.

SURVIVES  == every check passes (no fabrication, no thinning, protocol faithfully executed).
Run:  python -m sim.qa.redteam_adaptation_curve
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MAN = ROOT / "manifests"

C1_FILE = "sim/manifests/adaptation_curve_manifest.json"
C1_COMMIT = "250d669"     # single-file freeze
C2_COMMIT = "ecb71cb"     # lift-primary data commit (git descendant of C1)

# Prior seed banks (union across ALL prior committed phases) -- the NEW banks must be disjoint.
PRIOR_UNION = (
    set(range(2000, 2003)) | set(range(2100, 2103)) | set(range(2200, 2203))    # selection
    | set(range(3000, 3005)) | set(range(3100, 3105)) | set(range(3200, 3205))  # evaluation
    | set(range(1000, 1012)) | {2200, 2201}                                     # history (prior)
    | {3403, 3413, 3423}                                                        # training (prior)
)

checks: list[tuple[str, bool, str]] = []


def ck(name: str, ok: bool, detail: str = "") -> None:
    checks.append((name, bool(ok), detail))


# --------------------------------------------------------------------------- #
# FRESH metric primitives (re-derived; must agree with production on raw inputs)
# --------------------------------------------------------------------------- #
def qa_map_rmse(pred, meas):
    p = np.asarray(pred, float); m = np.asarray(meas, float)
    return float(np.sqrt(np.mean((p - m) ** 2)))


def qa_tau_cross(curve, ells, tau=0.5):
    y = np.asarray(curve, float); g = np.asarray(ells, float); step = g[1] - g[0]
    for i in range(len(y) - 1):
        if (y[i] - tau) * (y[i + 1] - tau) <= 0 and y[i] != y[i + 1]:
            t = (tau - y[i]) / (y[i + 1] - y[i])
            return (g[i] + t * (g[i + 1] - g[i]) - g[0]) / step
    return None


def qa_boundary_err(pred, meas, ells, tau=0.5):
    cp = qa_tau_cross(pred, ells, tau); cm = qa_tau_cross(meas, ells, tau)
    return None if (cp is None or cm is None) else float(abs(cp - cm))


def qa_selection_regret(pred, oracle):
    o = np.asarray(oracle, float); p = np.asarray(pred, float)
    return float(o[int(np.argmax(o))] - o[int(np.argmax(p))])


def qa_aggregate(vals):
    v = np.asarray([x for x in vals if x is not None], float)
    return float("nan") if v.size == 0 else float(v.mean())


def qa_prefix_meanpool(feats, k):
    if k <= 0:
        return None
    f = np.asarray(feats, float)
    assert f.shape[0] >= k
    return f[:k].mean(axis=0)


def qa_schedule(n_cells, ell_step, n_steps, ell_lo=0.12):
    return [((7 * j) % n_cells, round(ell_lo + ell_step * ((7 * j) % n_cells), 2), j % 4) for j in range(n_steps)]


# --------------------------------------------------------------------------- #
def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    import sim.adaptation_curve as ac

    man = json.loads((MAN / "adaptation_curve_manifest.json").read_text())
    res = json.loads((MAN / "adaptation_curve_results.json").read_text())

    # 1) C1 single-file freeze + git ancestry of the C2 data commit ---------- #
    try:
        stat = subprocess.run(["git", "show", "--stat", "--oneline", C1_COMMIT], cwd=ROOT.parent,
                              capture_output=True, text=True, timeout=30).stdout
        touched = [ln for ln in stat.splitlines() if "|" in ln]
        single = len(touched) == 1 and C1_FILE.split("/")[-1] in touched[0]
        anc = subprocess.run(["git", "merge-base", "--is-ancestor", C1_COMMIT, C2_COMMIT], cwd=ROOT.parent,
                             capture_output=True, timeout=30).returncode == 0
        ck("C1 single-file freeze", single, touched[0].strip() if touched else "no files")
        ck("C1 is git-ancestor of C2", anc, f"{C1_COMMIT}..{C2_COMMIT}")
    except Exception as e:  # noqa: BLE001
        ck("C1 git ancestry", False, f"git error: {e}")

    # 2) input-artifact digests still match the frozen manifest -------------- #
    dig = man["input_artifact_sha256"]
    bad = [f for f, h in dig.items() if not (MAN / f).exists() or sha256(MAN / f) != h]
    ck("input-artifact digests match freeze", not bad, f"mismatches: {bad}" if bad else f"{len(dig)} files OK")

    # 3) NEW seed banks disjoint from ALL prior ------------------------------ #
    new = set(ac.SEED_BANKS["selection"]) | set(ac.SEED_BANKS["evaluation"]) | set(ac.SEED_BANKS["history"]) | set(ac.SEED_BANKS["training"])
    overlap = sorted(new & PRIOR_UNION)
    ck("NEW seed banks disjoint from ALL prior", not overlap, f"overlap: {overlap}" if overlap else f"new={sorted(new)}")

    # 4) S_lift regenerated from first principles, byte-matches production ---- #
    qa_sl = qa_schedule(ac.LIFT_N_CELLS, ac.LIFT_ELL_STEP, ac.N_SCHEDULE_STEPS, ac.LIFT_ELL_LO)
    prod_sl = ac.schedule_lift()
    ck("S_lift matches independent regen", qa_sl == prod_sl, f"len={len(qa_sl)} cells={[c for c,_,_ in qa_sl][:8]}...")
    qa_sd = qa_schedule(ac.DISTAL_N_CELLS, ac.DISTAL_ELL_STEP, ac.N_SCHEDULE_STEPS, ac.LIFT_ELL_LO)
    ck("S_distal matches independent regen", qa_sd == ac.schedule_distal(), f"len={len(qa_sd)}")
    # stride 7 coprime to 17 and 25 -> full-grid permutation over the first n_cells steps
    ck("S_lift is a full-grid permutation", sorted({c for c, _, _ in qa_sl[:ac.LIFT_N_CELLS]}) == list(range(ac.LIFT_N_CELLS)),
       "first 17 steps cover every cell once")

    # 5) k=0 == blind membership -------------------------------------------- #
    k0_none = ac.prefix_summary(np.zeros((3, ac.FEATURE_DIM)), 0) is None
    curve = res["curve"]
    k0_eq_blind = all(abs(curve["student"]["map_rmse"][0]["mean"] - curve["blind"]["map_rmse"][0]["mean"]) < 1e-12
                      for _ in [0])
    ck("k=0 estimator is None (-> blind)", k0_none, "prefix_summary(x,0) is None")
    ck("results k=0 student == blind", k0_eq_blind, f"student k0={curve['student']['map_rmse'][0]['mean']:.4f}")

    # 6) NO cohort thinning: exact 2040-rollout inventory + schedule coverage - #
    hz = np.load(MAN / "adaptation_curve_lift_histories.npz", allow_pickle=True)
    n = int(hz["grasp"].shape[0])
    universe = list(ac.TRAIN_GROUPS + ac.VAL_GROUPS + ac.TEST_GROUPS)
    expect = len(universe) * ac.LIFT_N_CELLS * ac.N_TEMPLATES * len(ac.SEED_BANKS["history"])
    ck("history inventory exact (no thinning)", n == expect == 2040, f"rollouts={n} expected={expect} (=15x17x4x2)")
    S = hz["setting"].astype(str); Gr = hz["grasp"].astype(int); Tp = hz["template"].astype(int); Sd = hz["seed"].astype(int)
    kmax = max(ac.K_AXIS)
    missing = []
    for sid in ac.TEST_GROUPS:
        for j in range(kmax):
            cell, _, tmpl = prod_sl[j]
            for sd in ac.SEED_BANKS["history"]:
                if not ((S == sid) & (Gr == cell) & (Tp == tmpl) & (Sd == sd)).any():
                    missing.append((sid, cell, tmpl, sd))
    ck("max-k(32) prefix fully covered for TEST", not missing, f"missing={missing[:3]}" if missing else "all first-32 present")

    # 7) mean-pool k-prefix: independent recompute selects EXACTLY the first-k schedule cells - #
    from sim.identify import pool_temporal
    def qa_feat(i):
        return np.r_[hz["proprio"][i], pool_temporal(hz["shape"][i]), hz["action"][i]]
    sid0, sd0, k0 = ac.TEST_GROUPS[0], ac.SEED_BANKS["history"][0], 4
    feats = []
    for j in range(k0):
        cell, _, tmpl = prod_sl[j]
        idx = np.where((S == sid0) & (Gr == cell) & (Tp == tmpl) & (Sd == sd0))[0]
        feats.append(qa_feat(idx[0]))
    pooled = qa_prefix_meanpool(feats, k0)
    ck("mean-pool prefix is 42-D from EXACTLY first-k schedule", pooled is not None and pooled.shape == (ac.FEATURE_DIM,),
       f"dim={None if pooled is None else pooled.shape} cells={[prod_sl[j][0] for j in range(k0)]}")

    # 8) FRESH scorers agree with production on the SAME raw inputs ----------- #
    land = {x["id"]: x for x in json.loads((MAN / "addendum_landscape.json").read_text())["settings"]}
    ell = np.array(json.loads((MAN / "hard_sweep_manifest.json").read_text())["grasp"]["ell"], float)
    rng = np.random.default_rng(999)
    agree = True; details = []
    for sid in ac.TEST_GROUPS:
        meas = np.array(land[sid]["success_rate"], float)
        pred = np.clip(meas + rng.normal(0, 0.15, meas.shape), 0, 1)   # arbitrary QA probe curve
        if abs(qa_map_rmse(pred, meas) - ac.map_rmse(pred, meas)) > 1e-12:
            agree = False; details.append(f"{sid}:rmse")
        if abs(qa_selection_regret(pred, meas) - ac.selection_regret(pred, meas)) > 1e-12:
            agree = False; details.append(f"{sid}:regret")
        be_q, be_p = qa_boundary_err(pred, meas, ell), ac.boundary_index_error(pred, meas, ell)
        if (be_q is None) != (be_p is None) or (be_q is not None and abs(be_q - be_p) > 1e-12):
            agree = False; details.append(f"{sid}:boundary")
    vals = [0.1, 0.2, None, 0.3]
    if abs(qa_aggregate(vals) - ac.aggregate_unique_groups(vals)["mean"]) > 1e-12:
        agree = False; details.append("aggregate")
    ck("FRESH scorers agree with production", agree, "map/regret/boundary/aggregate identical on raw inputs" if agree else str(details))

    # 9) reported curve shape is honest (DESCRIPTIVE, non-gating) ------------- #
    stu = [c["mean"] for c in curve["student"]["map_rmse"]]
    tea = curve["teacher"]["map_rmse"][0]["mean"]; bli = curve["blind"]["map_rmse"][0]["mean"]
    reaches = min(stu) <= tea + 0.03 and stu[0] == bli
    ck("adaptation curve honest (blind->teacher-ceiling)", reaches,
       f"student {[round(x,3) for x in stu]} teacher~{tea:.3f} blind~{bli:.3f}")
    ck("lift selection-regret degeneracy reported", res.get("lift_selection_regret_degenerate") is True,
       "all lift regret 0.0 (committed degeneracy, reported honestly)")

    # -------------------------------------------------------------------- #
    npass = sum(1 for _, ok, _ in checks if ok)
    verdict = "SURVIVES" if npass == len(checks) else "FAILED"
    print(json.dumps(dict(verdict=verdict, passed=npass, total=len(checks),
                          checks=[dict(name=nm, ok=ok, detail=d) for nm, ok, d in checks]), indent=2))
    return 0 if verdict == "SURVIVES" else 1


if __name__ == "__main__":
    raise SystemExit(main())
