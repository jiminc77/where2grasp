# Hardening directive B: distal tip placement, the optimum-bearing task

**Date:** 2026-08-17 · **Owner ruling:** item 3 pre-approved after hardening-A closure. Ground rules and process requirements of hardening-A carry over unchanged (ralplan then ultragoal; every frozen manifest in its OWN commit before the data it governs; probe-until-clean for new paths; statistics permitted; setsid; STOP at closure).

## Why this task

Hardening-A established the boundary story but exposed that lift-and-clear's OPTIMUM is material-independent (argmax always the shortest arm), so selection-regret cannot discriminate conditioning there. This directive builds the task whose optimum is interior and material-dependent, which is what the grasp-SELECTION story of the paper needs.

## Task design (ralplan pins the details; the competing pressure is mandatory)

Distal tip placement: the rope's free tip must settle within a tolerance of a target point that is REACH-CONSTRAINED, so that too short a free arm cannot reach it (geometry lower bound) while too long a free arm droops past it (sag upper bound). Example scene: target at horizontal distance d beyond an edge the clamp cannot cross. The feasible interval is then [ell_reach(d), ell_sag(B_eff, w, tolerance)] and the tip-error objective has an interior, material-dependent optimum. The ralplan must:
- design the scene so the small-deflection regime covers the working range (declare the Pi_g validity range as in hardening-A),
- DERIVE the predicted feasible interval AND the predicted optimum curve s*(B_eff, w) under the declared objective BEFORE any sweep (pre-registered in the manifest commit),
- state the declared success criterion and objective explicitly (tip error tolerance; objective = settled tip error, or margin, chosen once).

## Part 1 — task build

Scene, motion templates (identical terminal clamp geometry discipline), property hooks reused from sim/, demo mp4s (one material where a mid-rope grasp wins vs one where it fails). Regime documentation. Unit tests in the style of Step 1.

## Part 2 — landscape sweep and the selection gate

Manifest pre-frozen (own commit): grid within the hardening-A in-regime property set, s-grid, seed banks, hygiene identical to hardening-A. Frozen decision function for THIS task's gate: does the measured ARGMAX (not just the boundary) move with material in the predicted direction, with the fixed-ratio pairs argmax-invariant? Report predicted-vs-measured optimum curve with the same uncertainty reporting as hardening-A.

## Part 3 — critic rows and the transfer probe

- Rerun the equal-budget rows on the new task (teacher / blind / task-history student with the full temporal schema / leak-free sysID): NOW selection-regret is a discriminating pre-registered primary (material-dependent argmax), with map/boundary recovery kept as co-primary.
- **Frozen cross-task transfer probe (the module claim):** take the hardening-A task-only history ENCODER trained on lift-and-clear, FROZEN, and use its latent for the tip-placement critic (only the critic head trains). Compare against a from-scratch encoder on equal budget. This is the first m_t-as-module evidence; pre-register the comparison.
- Adaptation curve on the new task (cold start from the blind prior, quality vs number of interactions).

## Closure

Hardening-B report in STATUS.md answering three questions from committed artifacts: (1) does the argmax move with material as predicted? (2) does selection-regret now discriminate conditioning (teacher and student vs blind)? (3) does the FROZEN lift-and-clear encoder transfer (student-frozen vs student-scratch vs blind)? Honest remaining-gaps list. Then STOP for the owner ruling.
