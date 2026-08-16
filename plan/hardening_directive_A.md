# Hardening directive A: in-regime gate re-sweep + probe-enriched identifiability/student re-test

**Date:** 2026-08-17 · **Owner ruling:** hardening items 1+2 bundled and approved; item 3 (optimum-bearing task) follows as a separate directive after this closes. Ground rules of the fast-loop directives carry over (repo, commits+figures as ground truth, setsid, isolated env w2g, DLO-Lab clone untouched at c5026a9). Statistics are now PERMITTED (this is hardening): simple bootstrap CIs over evaluation draws are welcome where cheap.

## Process requirements

- **ralplan first, then ultragoal.** Inventory the deltas from the fast-loop code (grid redesign, probe primitive, full temporal schema, leak-free splits) and probe-until-clean anything new.
- **Pre-registration must be git-provable this time:** every frozen manifest is committed in its OWN commit BEFORE any data it governs is generated. This fixes the recorded fast-loop process caveat.

## Part 1 — audit-clean in-regime Step-2 re-sweep (formal gate)

- Redesign the property x free-length grid so EVERY boundary-bearing cell satisfies the frozen Pi_g <= 0.5 regime guard, and extend the s-grid top so the previously censored stiffest-lightest setting resolves in-grid.
- Same estimation hygiene as the fast loop (best template's MEAN over settles; disjoint selection/evaluation banks; winner-only evaluation). Keep the exact frozen three-way decision function unchanged; rerun; report the formal verdict.
- Report the descriptive prefactor ell_max / (B_eff/w)^(1/4) against the predicted (8h)^(1/4) across the clean grid, now with an uncertainty statement (grid-resolution bounds and/or bootstrap CI over evaluation draws).
- Deliverables: manifest pre-freeze commit, sweep data, gate_verdict v2, updated boundary_shift figure with reference slope and prefactor panel, STATUS entry.

## Part 2 — small-deflection probe + identifiability and student re-test

- **Probe primitive:** a prescribed small-deflection interaction (clamped cantilever at a short free length inside the calibrated linear regime, settled shape observation; reuse the Step-1 calibration machinery as the primitive).
- **Histories v2:** regenerate interaction histories with the FULL frozen temporal (y,z) schema (fixing the terminal-z-summary limitation), for task-only histories AND probe-enriched histories (probe prepended), each under the channel configurations (proprio / +shape / +wrench).
- **Pre-register (own commit, before data) the sharpened two-sided prediction:** (a) from probe-enriched shape histories, B_eff/w recovery passes the positive control (the Step-1 result says the information is there in this regime); (b) (cB,cw) pairs remain non-separable from shape alone (degeneracy) and separable once wrench is included. Also pre-register the map-recovery secondary as a PRIMARY metric for the critic comparison (fixing the post-hoc caveat).
- **Leak-free explicit-sysID refit:** rebuild the regressor with train/test splits fully disjoint from the critic TEST settings.
- **Critic comparison v2:** rerun the equal-budget rows (teacher / blind / task-history student / probe-enriched student / leak-free sysID) under the pre-registered metrics: map RMSE, correlation, tau-boundary index error on held-out settings; adaptation vs history composition (task-only vs probe-prepended).
- Deliverables: identifiability table v2 (channel x history-type), critic comparison v2 figures, verdict json, STATUS entry.

## Closure

Hardening-A report in STATUS.md answering exactly two questions from committed artifacts: (1) does the formal frozen gate now PASS? (2) does the probe-enriched student approach the teacher (is m_t functional)? Include an honest remaining-gaps list. Then STOP for the owner ruling before item 3 (distal tip placement, the optimum-bearing task).
