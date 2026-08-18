"""Finding-5 r_N DIAGNOSTIC attribution of the force-vs-gravity B_eff gap (post-hoc, from
committed data only; does NOT change the REPORTED NULL verdict). Adds a finding5_attribution
block to the verdict + a diagnostic figure. Verifies the orchestrator's framing.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sim import ic_common as ic

MAN = Path("sim/manifests"); FIG = Path("sim/figures")
verdict = json.loads((MAN / "independent_closure_verdict.json").read_text())
cal = json.loads((MAN / "calibration.json").read_text())
labels = verdict["cohort"]
cal01 = {c["label"]: c for c in verdict["calibration"]["0.01"]}
gmap = {f"B{i}": b for i, b in enumerate(ic.GRAV_B_EFF)}
gmap.update({rid: b for rid, b in zip(ic.RATIO_IDS, ic.RATIO_GRAV_B_EFF)})
grav_delta = {m["raw_E"]: m["per_length"]["0.0002"] for m in cal["materials"]}  # committed gravity sag, mass 0.0002

LW = ic.CALIB_LENGTHS
w0 = ic.OBS_B_REF_W
rN_win = np.array([ic.r_N(ic.segment_count(L, 0.01)) for L in LW])
mean_rN = float(np.mean(rN_win))

# small-deflection diagnostic window on the committed gravity sweep grid [0.12..0.60] step 0.03
sweep_ells = np.array(ic.GRAV_SWEEP_ELL)
win = np.where((sweep_ells >= 0.15) & (sweep_ells <= 0.30))[0]   # in-regime sag-inflation window
f5 = verdict["finding5_rN"]["0.01"]
rows = {}
for lab in labels:
    bff = cal01[lab]["B_eff_force"]
    gap = bff / gmap[lab] - 1.0
    infl = np.array(f5[lab]["sag_ratio"])[win]      # delta_grav / continuum(B_eff_force) = r_N*(B_force/B_cont)
    rns = np.array(f5[lab]["rN"])[win]
    ratio = infl / rns                              # ~ constant (B_eff_force/B_cont)
    rows[lab] = dict(B_eff_force=bff, gravity_B_eff=gmap[lab], gap_pct=gap*100,
                     window_ells=sweep_ells[win].tolist(), sag_inflation=infl.tolist(), r_N=rns.tolist(),
                     inflation_over_rN=ratio.tolist(),
                     shape_match_std_pct=float(np.std(ratio) / np.mean(ratio) * 100),
                     residual_pct=float((np.mean(ratio) - 1.0) * 100))

gaps = {l: rows[l]["gap_pct"] for l in labels}
stiffer = [l for l in labels if l != "B1"]
match6 = float(np.max([rows[l]["shape_match_std_pct"] for l in stiffer]))
robustness = (1.0 + np.mean(list(gaps.values()))/100) ** 0.25 - 1.0   # 4th-root boundary shift

attribution = dict(
    note="DIAGNOSTIC ONLY -- explains the force-vs-gravity B_eff gap; does NOT change the REPORTED NULL verdict.",
    per_material_gap_pct=gaps,
    gap_range_pct=[float(min(gaps.values())), float(max(gaps.values()))],
    calibration_window_mean_rN_pct=(mean_rN - 1.0) * 100,
    residual_above_rN_pct=float(np.mean(list(gaps.values())) - (mean_rN - 1.0) * 100),
    six_stiffer_shape_match_maxdev_pct=match6,
    B1_note="B1 (softest in-cohort) deviates more -- larger self-weight deflection -> geometric nonlinearity beyond lumped-mass r_N.",
    fourth_root_boundary_shift_pct=float(robustness * 100),
    interpretation=("Gravity B_eff was fit from distributed self-weight on a lumped-mass rod, biased LOW by the "
                    "quadrature factor r_N; the tip point-load force calibration has no distributed-mass quadrature "
                    "bias, so B_eff_force ~= B_continuum, HIGHER than gravity B_eff by ~r_N. The calibration-window "
                    "mean r_N (~%.1f%%) accounts for most of the +%.1f%% mean gap; the ~%.1f%% residual is unexplained. "
                    "Because 4th-root boundaries shift only ~%.1f%% under a ~%.1f%% B_eff rescale (within the declared "
                    "grid bound), the prefactor(b)/distal(c) no-refit predictions PASS while the direct-sag(a) misses."
                    % ((mean_rN-1)*100, np.mean(list(gaps.values())), np.mean(list(gaps.values()))-(mean_rN-1)*100,
                       robustness*100, np.mean(list(gaps.values())))),
    per_material=rows,
)
verdict["finding5_attribution"] = attribution
(MAN / "independent_closure_verdict.json").write_text(json.dumps(verdict, indent=2, default=float))

# figure: per-length sag inflation vs analytical r_N (calibration window)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
wl = rows[labels[0]]["window_ells"]
for lab in labels:
    ax[0].plot(wl, rows[lab]["sag_inflation"], "o-", ms=4, label=lab)
ax[0].plot(wl, rows[labels[3]]["r_N"], "k--", lw=2, label="analytical $r_N$")
ax[0].set_xlabel(r"$\ell$ (m)"); ax[0].set_ylabel("sag inflation $\\delta_{grav}/\\delta_{cont}(B_{force})$")
ax[0].set_title("Finding-5: per-length sag inflation vs $r_N$"); ax[0].legend(fontsize=7)
g = [gaps[l] for l in labels]
ax[1].bar(range(len(labels)), g, color="steelblue"); ax[1].set_xticks(range(len(labels))); ax[1].set_xticklabels(labels)
ax[1].axhline((mean_rN-1)*100, color="k", ls="--", label="mean $r_N$ %.1f%%" % ((mean_rN-1)*100))
ax[1].set_ylabel("B_eff_force / gravity B_eff  gap (%)"); ax[1].set_title("Gap attributed to $r_N$ bias"); ax[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "ic_finding5_rN.png", dpi=130); plt.close(fig)

print("per-material gap %%: %s" % {l: round(gaps[l], 2) for l in labels})
print("gap range: [%.2f, %.2f]%%  mean %.2f%%" % (min(gaps.values()), max(gaps.values()), np.mean(list(gaps.values()))))
print("calibration-window mean r_N: +%.2f%%  residual: +%.2f%%" % ((mean_rN-1)*100, np.mean(list(gaps.values()))-(mean_rN-1)*100))
print("6 stiffer rows shape-match std: %.3f%%  |  B1 std: %.3f%%" % (match6, rows["B1"]["shape_match_std_pct"]))
print("per-material residual (inflation/r_N - 1): %s" % {l: round(rows[l]["residual_pct"], 2) for l in labels})
print("4th-root boundary shift under mean gap: %.2f%%" % (robustness*100))
print("wrote finding5_attribution + ic_finding5_rN.png")
