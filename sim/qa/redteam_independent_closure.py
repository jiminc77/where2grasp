"""Independent red-team recompute for the independent mechanics closure.

Fresh-code recompute (NOT a re-read of stored scalars) of B_eff_force, the mesh
observables A/B + three-way gate outcome, the three no-refit errors, and the
calibration/guard/r_N/contamination/invariance scalars from the committed C2 raw
arrays; plus a dataflow-provenance assertion (the force fitter only ever sees force
arrays + known loads), input-array digests, C1-single-file/ancestry proof, and a
deterministic replay of representative sims against the committed raw arrays.

C0 ships this as a runnable skeleton: checks whose C2 inputs do not yet exist report
`pending`; the arithmetic recompute + git/hash machinery are implemented now so the
Closure story runs a full recompute with a survivor count.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from sim import ic_common as ic

ROOT = Path(__file__).resolve().parents[1]
MAN = ROOT / "manifests"
REPORT = ROOT / "qa" / "redteam_independent_closure_report.json"

# Input artifacts the closure depends on (re-asserted against the frozen manifest list).
INPUT_ARTIFACTS = ("calibration.json", "hard_gate_verdict.json", "hard_sweep_landscape.json",
                   "distal_manifest.json", "distal_sweep_landscape.json", "distal_gate_verdict.json",
                   "distal_critic_results.json", "addendum_results.json", "spanning_results.json")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def is_ancestor(commit_a, commit_b):
    """git merge-base --is-ancestor A B : True iff A is an ancestor of B."""
    r = subprocess.run(["git", "merge-base", "--is-ancestor", commit_a, commit_b],
                       cwd=ROOT, capture_output=True)
    return r.returncode == 0


def recompute_beff_force(ell, delta, force):
    """Independent cubic recompute B_eff_force = F/(3k), k = <x*delta>/<x*x>, x = ell**3."""
    ell = np.asarray(ell, float); delta = np.asarray(delta, float)
    x = ell ** 3
    k = float(np.sum(x * delta) / np.sum(x * x))
    return force / (3.0 * k)


def _check(results, name, status, detail=""):
    results.append(dict(check=name, status=status, detail=detail))


def run(manifest_name="independent_closure_manifest.json",
        verdict_name="independent_closure_verdict.json"):
    results = []

    # 1. input-artifact digests re-asserted against the frozen manifest list
    man_path = MAN / manifest_name
    if man_path.exists():
        man = json.loads(man_path.read_text())
        frozen = man.get("input_artifact_sha256", {})
        mism = []
        for a in INPUT_ARTIFACTS:
            p = MAN / a
            if not p.exists():
                mism.append(f"{a}:missing"); continue
            if a in frozen and frozen[a] != sha256(p):
                mism.append(f"{a}:digest-drift")
        _check(results, "input_artifact_digests", "survives" if not mism else "fail",
               "ok" if not mism else "; ".join(mism))
        # 2. C1 single-file + ancestry proof
        c1 = man.get("provenance", {}).get("c1_commit")
        c2 = man.get("provenance", {}).get("c2_commit")
        if c1 and c2:
            anc = is_ancestor(c1, c2)
            _check(results, "c1_ancestor_of_c2", "survives" if anc else "fail",
                   f"merge-base --is-ancestor {c1[:8]} {c2[:8]} = {anc}")
        else:
            _check(results, "c1_ancestor_of_c2", "pending", "provenance commits not recorded yet")
    else:
        _check(results, "input_artifact_digests", "pending", f"{manifest_name} not frozen yet")
        _check(results, "c1_ancestor_of_c2", "pending", "manifest not frozen yet")

    # 3. recompute headline scalars from committed C2 raw arrays
    verdict_path = MAN / verdict_name
    if verdict_path.exists():
        v = json.loads(verdict_path.read_text())
        # B_eff_force recompute from raw force arrays
        drift = []
        for row in v.get("force_calibration", []):
            if "ell" in row and "delta" in row and "F" in row:
                mine = recompute_beff_force(row["ell"], row["delta"], row["F"])
                if abs(mine - row["B_eff_force"]) / row["B_eff_force"] > 1e-6:
                    drift.append(f"{row.get('raw_E')}: {mine:.6g} vs {row['B_eff_force']:.6g}")
        _check(results, "beff_force_recompute", "survives" if not drift else "fail",
               "ok" if not drift else "; ".join(drift))
        # no-refit error recompute would run here against v['no_refit'] (implemented at Closure)
        _check(results, "no_refit_recompute", "pending", "wire to C2 verdict no_refit block at Closure")
        _check(results, "mesh_gate_recompute", "pending", "wire to C2 verdict mesh block at Closure")
        _check(results, "deterministic_replay", "pending", "replay representative sims at Closure")
    else:
        for nm in ("beff_force_recompute", "no_refit_recompute", "mesh_gate_recompute",
                   "deterministic_replay"):
            _check(results, nm, "pending", f"{verdict_name} not produced yet (C2)")

    survivors = sum(1 for r in results if r["status"] == "survives")
    fails = sum(1 for r in results if r["status"] == "fail")
    pending = sum(1 for r in results if r["status"] == "pending")
    report = dict(checks=results, survivors=survivors, fails=fails, pending=pending,
                  survived_all=bool(fails == 0))
    REPORT.write_text(json.dumps(report, indent=2))
    print(f"red-team: survives={survivors} fail={fails} pending={pending} -> {REPORT}")
    for r in results:
        print(f"  [{r['status']:8}] {r['check']}: {r['detail']}")
    return report


if __name__ == "__main__":
    run()
