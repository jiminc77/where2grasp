# where2grasp

Where to grasp a deformable linear object: a mechanics-derived admissible envelope, an interaction-identified mechanics latent (m_t), and a property-adaptive per-point grasp critic. Paper 1 of a two-paper arc; the sequel (DLO-JEPA) consumes m_t as its history encoder.

**Canonical plan:** [strategy/proposal_v2.md](strategy/proposal_v2.md) (v3) · **Session log:** [STATUS.md](STATUS.md) · **Conventions:** [00_INDEX.md](00_INDEX.md)

## Current state (2026-08-16)

Proposal v3 finalized: three external feedback rounds absorbed, every claim and citation audited against primary sources. Nothing has been run yet. Next: fast validation loop, Step 0.

## The fast validation loop

One question gates everything: **does the boundary of the feasible grasp set move with material properties in the predicted direction?** Everything below exists to answer it as cheaply as possible.

| Step | What | Deliverable | Gate |
|---|---|---|---|
| 0 | Fresh DLO-Lab clone + engine setup on the RTX PRO 6000, smoke test | env-running render | build works |
| 1 | Minimal lift-and-clear task (horizontal-clamp scene per the regime block) + property variation hooks (B and w varied independently, fixed-ratio pairs included) + virtual B_eff calibration (simulated cantilever load-deflection test) | task demo video + calibration curve | task behaves |
| 2 | Mini grasp landscape: candidate points x property settings x motion templates in parallel envs; per-point value = best motion's MEAN over repeated settles (optimization and evaluation seeds separated); success criterion and expected exponent declared before data | per-property landscape figures + boundary-shift plot + rollout videos | **GO/NO-GO: boundary moves as predicted** |
| 3 | Identifiability: reuse Step 2 rollouts, add only targeted probes; two-sided prediction (shape channel alone fails to separate (cB, cw) pairs; adding the wrench channel separates them) | channel-wise identifiability table | properties readable |
| 4 | Miniature teacher-student: property-conditioned vs blind critic; three mechanism candidates side by side | first adaptation curves | conditioning helps |

After the loop: decide, then harden (task expansion, real-robot phase, statistics). No numeric targets before the loop reports.

## Execution

- Code is written fresh against a clean DLO-Lab clone; legacy DGCC code is intentionally not reused.
- Runner: gjc on AILAB-simx-remote, ralplan → ultragoal. Model roles: default/main `claude-opus-4.8:xhigh` · planner `claude-opus-4.8:medium` · architect `gpt-5.6-sol:xhigh` · executor `gpt-5.6-terra:medium` · critic `gpt-5.6-sol:high`.
- Every step ships visualizations (rollout videos + figures); simulator work is verified visually, not only by metrics.
