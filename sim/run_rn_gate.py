"""Item-2 C2: data + figure + verdict for the r_N-corrected no-refit graduation gate.

Score-only after the C1 freeze (rn_gate_manifest.json). Deterministically settles the held-out
prospective cohort (mass 0.00025 / ell{0.15,0.16,0.17}, 7 cohort materials), recomputes the
r_N-corrected direct-sag no-refit error on BOTH the retrospective calibration window
(calibration.json, mass 0.0004, committed) AND the prospective cohort, against the UNCHANGED 5%
bound. Emits rn_gate_verdict.json + rn_gate_overlay.png + raw arrays. Every frozen in-regime cell
is outcome-binding: a non-converged/out-of-guard cell -> INCONCLUSIVE + STOP (no thinning).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sim import ic_common as ic
from sim import rn_gate
from sim.ic_gravity_boundary import gravity_droop_sweep

ROOT = Path(__file__).resolve().parent
MAN = ROOT / "manifests"
FIG = ROOT / "figures"

COHORT_RAW_E = {**{f"B{i}": ic.RAW_E_GRID[i] for i in range(1, 5)},
                **{rid: r for rid, r in zip(ic.RATIO_IDS, ic.RATIO_RAW_E)}}


def score_retrospective(bff):
    """Committed calibration window (calibration.json, mass 0.0004); in-regime B1..B4 cells."""
    cal = json.loads((MAN / "calibration.json").read_text())
    raw_for = {f"B{i}": ic.RAW_E_GRID[i] for i in range(1, 5)}
    cells = []
    for lab in ("B1", "B2", "B3", "B4"):
        mat = next(m for m in cal["materials"] if abs(m["raw_E"] - raw_for[lab]) < 1e-6)
        pl = mat["per_length"]["0.0004"]
        for ell, dobs, pig in zip(pl["ell"], pl["delta"], pl["Pi_g"]):
            c = rn_gate.score_cell(ell, 0.0004, bff[lab], dobs)
            c.update(label=lab, cohort="retrospective")
            cells.append(c)
    return cells


def run_prospective(bff):
    """Deterministic gravity settle of the held-out prospective cohort (mass 0.00025 / ell{0.15,0.16,0.17})."""
    labels = list(rn_gate.COHORT_LABELS)
    raw_es = [COHORT_RAW_E[lab] for lab in labels]
    sweep = gravity_droop_sweep(raw_es, ic.REFERENCE_INTERVAL, mass=rn_gate.PROSPECTIVE_MASS,
                                ells=rn_gate.PROSPECTIVE_LENGTHS)
    droop = np.asarray(sweep["droop"])                       # (n_ell, n_material)
    win = np.asarray(sweep["per_ell_env_window"])            # (consec, n_ell, n_material)
    thr = float(sweep["drift_threshold"])
    cells = []
    for li, ell in enumerate(rn_gate.PROSPECTIVE_LENGTHS):
        for mi, lab in enumerate(labels):
            converged = bool((win[:, li, mi] < thr).all())   # consecutive-window settle proof
            dobs = float(droop[li, mi]) if converged else float("nan")
            c = rn_gate.score_cell(ell, rn_gate.PROSPECTIVE_MASS, bff[lab], dobs)
            c.update(label=lab, cohort="prospective", converged=converged)
            cells.append(c)
    return cells, sweep


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    bff = rn_gate.committed_beff_force()
    retro = score_retrospective(bff)
    prosp, sweep = run_prospective(bff)
    all_cells = retro + prosp
    verdict = rn_gate.prong_verdict(all_cells, expected_prospective_keys=rn_gate.prospective_keys())

    results = dict(
        item="2_rN_corrected_no_refit_gate", manifest_digest_of="rn_gate_manifest.json",
        bound=ic.NOREFIT_SAG_TOL, verdict=verdict["verdict"],
        graduates_direct_sag_prong=verdict["graduates_direct_sag_prong"],
        worst_rel_err=verdict["worst_rel_err"], n_in_regime=verdict["n_in_regime"],
        n_prospective=verdict["n_prospective"], integrity_ok=verdict["integrity_ok"],
        integrity_detail=verdict["integrity_detail"], aggregate_note=verdict["aggregate_note"],
        claim_language=rn_gate.claim_language(),
        attribution=rn_gate.reproduce_attribution(),
        open_residual_pct=rn_gate.reproduce_attribution()["residual_above_rN_pct"],
        retrospective=[{k: c[k] for k in ("label", "ell", "N", "mass", "pi_g", "in_regime", "finite",
                                          "delta_pred", "delta_obs", "rel_err", "within_bound")} for c in retro],
        prospective=[{k: c[k] for k in ("label", "ell", "N", "mass", "pi_g", "in_regime", "finite",
                                        "converged", "delta_pred", "delta_obs", "rel_err", "within_bound")} for c in prosp],
    )
    (MAN / "rn_gate_verdict.json").write_text(json.dumps(results, indent=2, default=float))
    np.savez(MAN / "rn_gate_results.npz",
             prospective_droop=np.asarray(sweep["droop"]),
             prospective_window=np.asarray(sweep["per_ell_env_window"]),
             prospective_ells=np.asarray(rn_gate.PROSPECTIVE_LENGTHS),
             cohort=np.array(list(rn_gate.COHORT_LABELS)),
             b_eff_force=np.array([bff[l] for l in rn_gate.COHORT_LABELS]),
             drift_threshold=float(sweep["drift_threshold"]))
    _figure(retro, prosp)
    print("VERDICT:", verdict["verdict"], "| graduates_direct_sag_prong:", verdict["graduates_direct_sag_prong"])
    print("worst rel_err = %.4f (bound %.2f) | in-regime cells = %d"
          % (verdict["worst_rel_err"], ic.NOREFIT_SAG_TOL, verdict["n_in_regime"]))
    print("prospective converged:", all(c["converged"] for c in prosp))
    return results


def _figure(retro, prosp):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    for name, cells, axi in (("retrospective (mass 0.0004)", retro, ax[0]),
                             ("prospective held-out (mass 0.00025)", prosp, ax[1])):
        labs = sorted({c["label"] for c in cells})
        for lab in labs:
            cc = [c for c in cells if c["label"] == lab and c["in_regime"]]
            if not cc:
                continue
            xs = [c["ell"] for c in cc]; ys = [c["rel_err"] * 100 for c in cc]
            axi.plot(xs, ys, "o-", ms=5, label=lab)
        axi.axhspan(0, ic.NOREFIT_SAG_TOL * 100, color="0.9", label="within 5% bound")
        axi.axhline(ic.NOREFIT_SAG_TOL * 100, color="r", ls="--")
        axi.axhline(rn_gate.reproduce_attribution()["residual_above_rN_pct"], color="k", ls=":",
                    label="~1.38%% residual (OPEN)")
        axi.set_xlabel(r"$\ell$ (m)"); axi.set_ylabel("r_N-corrected no-refit error (%)")
        axi.set_title(name); axi.legend(fontsize=6); axi.set_ylim(0, 6)
    fig.suptitle("Item 2: r_N-corrected direct-sag no-refit prediction vs the unchanged 5% bound")
    fig.tight_layout(); fig.savefig(FIG / "rn_gate_overlay.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
