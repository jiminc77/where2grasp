# where2grasp

**Thesis.** Where to grasp a deformable linear object is a task-conditioned decision whose dependence on invisible mechanics (bending stiffness, linear weight, friction) is derivable from beam mechanics and identifiable by a robot through interaction history alone.

**What we build.** (1) A mechanics-derived feasible grasp boundary and its scaling. (2) m_t, a validated mechanics encoder over interaction histories, designed as the E_H component of the DLO-JEPA architecture (sequel system paper). (3) Its first consumer: a property-adaptive per-candidate-point grasp critic.

## Entry points

- [STATUS.md](STATUS.md): current state, updated every working session. Read this first.
- [strategy/proposal_v2.md](strategy/proposal_v2.md): the canonical research proposal (v2, post-feedback). Everything is derived from this document.

## Origin

Pivot from the DGCC project, finalized 2026-08-15/16. Full history (F11-R campaign, pivot ideation, prior-art verification logs, feedback adjudication) lives in the DGCC workspace and repo (jiminc77/DGCC) and is intentionally NOT carried over: that workspace contains large amounts of deprecated material. Treat this repo as self-contained; the proposal carries all load-bearing citations with verified facts inline.

## Conventions

- Working language: English everywhere (docs, commits, code); Korean only in chat with the owner.
- STATUS.md gets a dated entry per working session, newest first.
- Commits are descriptive strategy records, not just diffs.
- No numeric targets in plans until the fast validation loop reports; published numbers only with citations.
- Compute: fast loop runs on the RTX PRO 6000 (remote: AILAB-simx-remote, ~/where2grasp). Real phase: 2 Franka Pandas.
