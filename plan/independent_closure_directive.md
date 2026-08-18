# Directive: independent mechanics closure (post-hardening-B, adjudication §C item 1)

Owner-approved 2026-08-18. Executes the top-priority follow-up from the external-inspection adjudication (`strategy/reviews/gptpro_inspection_adjudication_2026-08-17.md` §C-1, folding review findings 4, 5, 6, 7, 8 and preempting reviewer attack 1).

## Why

The strongest standing criticism of the Q1 result is circularity: B_eff was calibrated from self-weight sag with the same law, same clamp geometry, and same discretization later used to extract the boundary, so the 0.16% prefactor agreement is partly internal closure of the calibration pipeline. This phase makes the mechanics claim independent: calibrate with a different loading mode, show the calibrated observables converge across mesh resolution, and predict the gravity results with no refit.

## Scope (simulation only)

1. **Force-mode calibration.** Estimate `B_eff_force` from a loading mode independent of self-weight: a known tip force (linear Euler-Bernoulli: delta = F*ell^3/(3B)) or an imposed end moment/curvature, at the current fixed interval. Declare a small-deflection guard for the force mode (the analog of Pi_g) before any data. Choose force magnitudes so the deflection stays in the declared regime.
2. **Finding-5 premise check + reference curve.** Code-confirm against `sim/material.py` whether the audit's mass-lumping premise holds (full `segment_mass` on each free vertex, right-endpoint lumping). If it holds, compare the measured discretization bias against the analytical lumped-load reference r_N = (N+1)(3N+1)/(3N^2); report whether the observed effective sag exponent is consistent with quadrature bias plus continuum physics.
3. **Mesh sequence.** Repeat calibration at three intervals (target {0.02, 0.01, 0.005}; substitute with justification if a level is infeasible, e.g. stability or runtime). The raw knob-to-B_eff map MAY differ per interval; the claim under test is that calibrated OBSERVABLES converge: the normalized sag curve and the extracted boundary location. Only if the sequence shows convergence may the word "convergence" be used; otherwise the naming stays "fixed-discretization validation" (finding 6).
4. **No-refit prediction (the headline).** Using `B_eff_force` only — never refit on any gravity/sag data — predict: (a) held-out self-weight tip sag at new lengths and masses; (b) the hardening-A lift-and-clear boundary prefactor; (c) the distal-task optimum locations. Report prediction errors against the declared grid-resolution bounds. Success criteria are designed by the ralplan from PRIOR committed data only and frozen before data.
5. **Multi-mass invariance recompute (finding 7).** Re-evaluate load invariance using only lengths where BOTH masses are inside the declared regime guard (closing the light-mass-only acceptance loophole).

Out of scope, log as follow-ups, do NOT start: hardware/real-rod closure; a second solver; adjudication §C items 2 (k-interaction adaptation) and 3 (degeneracy-breaking force task).

## Process (unchanged)

Ralplan first; every threshold derived from prior committed artifacts and frozen in a SINGLE-FILE manifest commit before any data; new seed banks disjoint from all prior banks where stochastic draws are used; the finding-3 conservative regime-guard convention (grid-argmax upper-bracket endpoint) wherever a regime guard applies; independent red-team recompute; figures mandated (per-mode calibration curves, mesh-convergence plot, no-refit prediction overlay with error bars vs declared bounds); descriptive-commit convention; STATUS entry as its own dated section; STOP at closure for the owner ruling.
