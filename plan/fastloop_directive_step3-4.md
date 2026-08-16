# Fast-loop execution directive: Steps 3-4 (identifiability + miniature teacher-student)

**Date:** 2026-08-17 · **Owner ruling on the Step 2 gate: GO** (recorded here; gjc logs it in STATUS.md). Ground rules of `fastloop_directive_step0-2.md` carry over unchanged (repo layout, commits+figures+mp4s as ground truth, setsid for long runs, no touching other projects). Read strategy/proposal_v2.md (v3) Sections 7 and 9 before planning.

## Process requirement

**ralplan the new scope first, then ultragoal.** The new scope activates unverified paths: wrench-channel extraction, probe primitives, dataset assembly from Step 2 rollouts, regression/training loops, held-out-property evaluation. Inventory them and probe-until-clean before the main work. State all scouting-scale choices (dataset sizes, encoder sizes, training lengths) in the ralplan; keep total wall-clock for Steps 3-4 under a day.

## Step 3 — channel-wise identifiability (the m_t birth certificate)

Reuse the Step 2 sweep rollouts; **no new landscape sweep — the only new data are targeted probes.**

- **Datasets:** interaction histories across the property grid, assembled per observation-channel configuration: (i) action/proprioception only, (ii) + shape channel (vision-derived rope configuration, as in Step 2), (iii) + wrench channel. Implement the wrench channel as the vertical constraint force at the clamp (from physics state, or computed as supported weight); document in one paragraph how it maps to a real F/T sensor at the gripper.
- **Two-sided prediction test** (this is the headline of Step 3, pre-stated in proposal v3): from shape-channel observations alone, the fixed-ratio (cB, cw) pairs must NOT be separable — a simple regressor should recover B_eff/w (equivalently ell_gb) well but fail to recover B_eff and w individually on those pairs; adding the wrench channel should make B_eff and w individually recoverable. Both predictions are tested; report per-target error by channel configuration (simple regressors only: ridge and a small MLP; no architecture search).
- **Also report:** proprioception-only baseline (should be worst); whether B_eff/w recovery degrades gracefully with shorter histories (single interaction vs a few).
- **Friction probe (conditional):** attempt a minimal slip probe only if the engine at c5026a9 exposes a working friction parameter with a setter; otherwise document its absence in one paragraph and defer to hardening. Do not build heavy machinery for it.
- **Deliverables:** channel-wise identifiability table (figure + json), history-length figure, committed with a STATUS entry.

## Step 4 — miniature teacher-student (scouting scale)

Single task (lift-and-clear). The object of study is the per-candidate-point critic: score grasp points s for task success, across the property grid, evaluated on held-out properties.

- **Rows, side by side (all sharing the same critic architecture and training budget):**
  1. **Privileged teacher:** critic conditioned on true (B_eff, w) — the adaptation ceiling proxy.
  2. **Property-blind:** same critic, no conditioning.
  3. **History-latent student (the m_t seed):** encoder over the shape-channel interaction history, distilled RMA-style from the teacher's conditioning latent.
  4. **Explicit-sysID pipeline:** Step 3's regressor predicts properties from history, prediction fed to the critic.
  (A one-step deformation-prediction auxiliary variant is optional; include only if cheap.)
- **Evaluation:** held-out property settings (interpolation within the calibrated range). Because Step 2 measured the ground-truth landscape, report **selection regret against the measured landscape** (chosen point's measured value vs best measured point) — this is cheap and stronger than success-rate alone. Report a coarse **adaptation curve** for the student: selection quality vs number of interactions in the history (cold start = property-blind prior, per proposal v3's deployment protocol).
- **Directional outcome only:** does conditioning beat blind; does the student approach the teacher; how does explicit sysID compare. No pre-registered thresholds; this is scouting.
- **Deliverables:** comparison figure (regret by row), adaptation-curve figure, training logs, committed.

## Loop closure

After Step 4, write the **fast-loop verdict** in STATUS.md: one paragraph per loop question (boundary moves? properties readable, channel-resolved? conditioning helps? student approaches teacher?), each answered from committed artifacts, plus an honest list of what the loop did NOT establish (statistics, multi-task, real). Then **STOP** for the owner's hardening/expansion decision. Steps beyond the loop are not started.
