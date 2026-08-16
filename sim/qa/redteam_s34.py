"""Adversarial, read-only recomputation of the frozen Step 3/4 claims."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MAN = ROOT / "manifests"
SOURCE_HASH = "sha256:8ee7435e815b414eeea56d74bea2d0e5d2b1e95fce5ff09613aa74061b13ec1c"
EXPECTED_DIGEST = "ce0c99494ca042e3d781786d0b84a8418f162c853d2a755567aecd267a1f7278"


def load(name: str):
    return json.loads((MAN / name).read_text())


def fit_ridge(x, y, alpha):
    mu, sd = x.mean(0), x.std(0)
    sd[sd < 1e-12] = 1
    a = np.c_[np.ones(len(x)), (x - mu) / sd]
    reg = np.eye(a.shape[1]) * alpha
    reg[0, 0] = 0
    return mu, sd, np.linalg.solve(a.T @ a + reg, a.T @ y)


def predict(model, x):
    mu, sd, weights = model
    return np.c_[np.ones(len(x)), (x - mu) / sd] @ weights


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def case(name, passed, numbers, attack):
    return {"case": name, "pass": bool(passed), "numbers": numbers, "attack": attack}


def run():
    cfg, ident, landscape, critic = (load(x) for x in (
        "s34_manifest.json", "identifiability.json", "sweep_landscape.json", "critic_results.json"
    ))
    hz = np.load(MAN / "histories_s3.npz")
    digest = hashlib.sha256((MAN / "s34_manifest.json").read_bytes()).hexdigest()
    settings = {s["id"]: s for s in landscape["settings"]}
    source = str(hz["source"].item())
    history_digest = str(hz["manifest_digest"].item())

    # Reconstruct exactly the primary +shape design and labels without importing production code.
    ell = load("sweep_manifest.json")["grasp"]["ell"]
    rows = []
    for i, sid0 in enumerate(hz["setting"]):
        sid = str(sid0)
        s = settings[sid]
        x = np.r_[hz["proprio"][i], hz["shape"][i], ell[int(hz["grasp"][i])], 0.0][:24]
        y = np.log10([s["B_eff"], s["w"], s["B_eff"] / s["w"]])
        rows.append((sid, x, y, np.r_[hz["shape"][i], ell[int(hz["grasp"][i])]]))
    train_ids = set(cfg["splits"]["step3_train"])
    train = [r for r in rows if r[0] in train_ids]
    test = [r for r in rows if r[0] not in train_ids]
    x, y, ids = np.array([r[1] for r in train]), np.array([r[2] for r in train]), np.array([r[0] for r in train])
    xt, yt = np.array([r[1] for r in test]), np.array([r[2] for r in test])
    scores = []
    for alpha in cfg["ridge_alpha_grid"]:
        fold_errors = []
        for fold_ids in cfg["grouped_cv"]["folds"].values():
            val = np.isin(ids, fold_ids)
            fold_errors.append(np.mean([rmse(y[val, j], predict(fit_ridge(x[~val], y[~val], alpha), x[val])[:, j]) for j in range(3)]))
        scores.append(float(np.mean(fold_errors)))
    alpha = cfg["ridge_alpha_grid"][int(np.argmin(scores))]
    pred = predict(fit_ridge(x, y, alpha), xt)
    ratio_rmse = rmse(yt[:, 2], pred[:, 2])

    pair_rms = {}
    for a, b in cfg["splits"]["final_test_pairs"]:
        aa = np.mean([r[3] for r in test if r[0] == a], 0)
        bb = np.mean([r[3] for r in test if r[0] == b], 0)
        pair_rms[f"{a}:{b}"] = float(np.sqrt(np.mean((aa - bb) ** 2)) / max(np.sqrt(np.mean(aa * aa)), 1e-12))

    argmaxes = {s["id"]: int(np.argmax(s["mean_J"])) for s in landscape["settings"]}
    secondary = critic["secondary_map_recovery"]
    recomputed = {}
    per_setting = {}
    for row, value in secondary.items():
        setting_errors = []
        per_setting[row] = {}
        for item in value["per_setting"]:
            e = rmse(item["measured_success_curve"], item["predicted_success_curve"])
            per_setting[row][item["setting"]] = e
            setting_errors.append(e)
        recomputed[row] = float(np.mean(setting_errors))

    splits = critic["splits"]
    seeds = cfg["seed_partition"]
    frozen_rule = "INCONCLUSIVE" if not (
        ratio_rmse <= cfg["margins"]["tol_ratio"] and max(pair_rms.values()) <= cfg["margins"]["tol_shape"]
    ) else "PASS_OR_FAIL"
    critic_text = (ROOT / "critic.py").read_text()
    status = (ROOT.parent / "STATUS.md").read_text()
    cases = [
        case("real_history_provenance", source == "actual Genesis rod.get_vertices_pos()" and history_digest == digest == EXPECTED_DIGEST,
             {"source": source, "npz_manifest_digest": history_digest, "actual_manifest_digest": digest, "rollouts": int(hz["rollout_count"])}, "Substitute analytical histories or a mismatched manifest."),
        case("paired_shape_confound_guard", max(pair_rms.values()) <= cfg["margins"]["tol_shape"],
             {"paired_relative_shape_rms": pair_rms, "tolerance": cfg["margins"]["tol_shape"]}, "Recompute pair means and normalized RMS directly from history arrays."),
        case("positive_control_and_two_sided_rule", abs(ratio_rmse - ident["positive_control"]["shape_ratio_rmse"]) < 1e-12 and ratio_rmse > cfg["margins"]["tol_ratio"] and frozen_rule == ident["verdict"] == "INCONCLUSIVE",
             {"selected_alpha": alpha, "ratio_test_rmse_dex": ratio_rmse, "tolerance_dex": cfg["margins"]["tol_ratio"], "derived_verdict": frozen_rule}, "Refit grouped-CV ridge independently and attempt to turn a failed positive control into PASS."),
        case("step4_argmax_null", len(argmaxes) == 23 and set(argmaxes.values()) == {0},
             {"setting_count": len(argmaxes), "argmax_histogram": {str(i): list(argmaxes.values()).count(i) for i in sorted(set(argmaxes.values()))}}, "Recompute every measured mean-J argmax; a nonzero optimum would falsify the null explanation."),
        case("map_recovery_recomputation", all(abs(recomputed[k] - secondary[k]["map_rmse"]) < 1e-12 for k in recomputed) and recomputed["teacher"] < recomputed["blind"] < recomputed["student"] and recomputed["blind"] < recomputed["explicit_sysid"],
             {"aggregate_map_rmse": recomputed, "held_out_B2_w1_rmse": {k: v["B2_w1"] for k, v in per_setting.items()}}, "Recompute from all stored held-out curves and check teacher advantage and student/sysID failures."),
        case("map_recovery_leak_free_structure", not (set(splits["test"]) & (set(splits["train"]) | set(splits["validation"]))) and "for sid in TRAIN+VAL" in critic_text and "props[s]" in critic_text and "if row=='blind': z=np.zeros(4)" in critic_text,
             {"test": splits["test"], "train_count": len(splits["train"]), "validation": splits["validation"], "teacher_inputs": ["B_eff", "w"], "blind_latent": "constant zero"}, "Search fitting loops for TEST membership and verify blind has one property-independent map."),
        case("seed_and_split_isolation", set(seeds["train_validation"]).isdisjoint(seeds["oracle_only"]) and splits["test"] == ["B2_w1", "B2_w2", "B3_w1"] and not ({"R2", "B2_w1"} & (set(splits["train"]) | set(splits["validation"]))),
             {"selection_seeds": seeds["train_validation"], "oracle_seeds": seeds["oracle_only"], "test": splits["test"], "excluded_from_fit": ["R2", "B2_w1"]}, "Look for seed overlap or held-out pair leakage."),
        case("status_no_overclaim", "student does NOT approach the teacher" in status and "student and the explicit-sysID pipeline are both worse than blind" in status and "exact Step-2 and Step-3 verdicts are both *inconclusive*" in status,
             {"status_claims_student_works": False, "status_claims_clean_step3_go": False}, "Search loop closure for a working-student or clean-GO claim."),
    ]
    report = {
        "artifact_kind": "algorithm/math test-report",
        "schema_version": 1,
        "sourceHash": SOURCE_HASH,
        "scope": "Committed Step 3/4 npz/json numerical artifacts; source text inspected only for fitting-flow and STATUS claim audits.",
        "verdict": "SURVIVES" if all(c["pass"] for c in cases) else "DOES_NOT_SURVIVE",
        "summary": "The negative/inconclusive verdicts survive adversarial recomputation; no dishonesty or overclaiming found." if all(c["pass"] for c in cases) else "At least one committed claim was falsified; inspect failed cases.",
        "adversarial_cases": cases,
    }
    out = ROOT / "qa" / "s34_redteam_report.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"report": str(out), "verdict": report["verdict"], "cases": len(cases)}))
    if not all(c["pass"] for c in cases):
        raise SystemExit(1)


if __name__ == "__main__":
    run()
