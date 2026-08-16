# Fast-loop execution directive: Steps 0-2 (lift-and-clear diagnostic)

**Date:** 2026-08-16 · **Owner-approved decisions in force** (Q1a/Q2a/Q3a/Q4a). This directive is the operative scope for the gjc session; read README.md and strategy/proposal_v2.md (v3) in this repo before planning. Where older text conflicts with this directive, this directive wins.

## Process requirement

**ralplan first, then ultragoal.** The ralplan must include:
- an unverified-path inventory: every code path this work activates that has never run on this machine (engine install, offscreen rendering, video export, parallel envs, property overrides), and
- a probe-until-clean smoke sequence covering: engine import → scene build → single rollout → offscreen render to an mp4 FILE → parallel batch → property override effectiveness (verify that changing B actually changes behavior).
- the Step 2 sweep sizes (your choice, scouting scale, total wall-clock under a day) and the declared success criterion + expected exponent, stated BEFORE the sweep runs.

## Ground rules

- All new code lives in THIS repo under `sim/`. **No DGCC legacy code reuse of any kind.**
- Clone DLO-Lab fresh to `~/DLO-Lab`, **pinned to commit c5026a9**, used as an external dependency. Do not develop inside the clone; any patch required for installation is recorded as a diff under `sim/third_party_patches/`.
- Do NOT touch `~/v2_research` or any existing project/env on this machine. Use a fresh isolated Python env.
- GPU: this host's RTX PRO 6000.
- Every step's deliverables (code, figures, mp4s, notes) are committed and pushed to this repo with descriptive messages; add a dated entry to STATUS.md per working session.
- Autonomy: run Steps 0-2 continuously, commit+report at each step boundary, then **STOP after the Step 2 gate report** for the owner's go/no-go ruling. Do not start Steps 3-4.
- Long runs are detached with `setsid` and log to files; commits and files are the ground truth, not chat claims.

## Step 0 — environment

Fresh DLO-Lab clone + engine installation per its requirements in an isolated env; smoke test one shipped example; produce ONE offscreen-rendered mp4 proving headless rendering works end to end. If installation is blocked (driver/CUDA conflict), STOP and report the exact blocker with diagnostics; do not improvise a different simulator without owner approval.

## Step 1 — diagnostic task build

- **Task: lift-and-clear**, designed to match the sag formula's regime (proposal Section 3, regimes of validity): the grasp clamp must fix the local tangent so the free arm starts HORIZONTAL (e.g., weld two or more adjacent vertices); free arm initially straight; success predicate = settled free-arm tip stays above a failure plane at depth h below the clamp. Declared criterion: absolute clearance; declared expected boundary scaling: ell_max proportional to (B_eff h / w)^(1/4).
- **Property hooks:** B and w controllable independently; the grid includes fixed-B/w pairs. DLO-Lab's material hooks exist but are commented out; re-enable properly in our own code.
- **Virtual B_eff calibration:** simulated clamped cantilever under self-weight at several free lengths in the small-deflection range; fit B_eff from delta = w ell^4 / (8 B_eff); produce a raw-parameter-to-B_eff calibration curve (figure + table). All later plots are indexed by B_eff, never by raw simulator elasticity.
- **Deliverables:** task demo mp4s (one stiff and one soft rope showing qualitatively different outcomes), calibration figure, and a short `sim/README.md` describing the scene and parameters.

## Step 2 — mini landscape (the gate)

- **Sweep:** candidate grasp points along the rope × property settings (B_eff and w varied independently, fixed-ratio pairs included) × a small family of lift motion templates, in parallel envs. Scouting scale only.
- **Estimation hygiene (mandatory):** per-point value = the best template's MEAN over repeated settles; the draws used to select the best template are separated from the draws used to report its value.
- **Outputs:** per-property landscape figures (success/value vs grasp position); a boundary-position vs (B_eff/w) plot with the predicted fourth-root reference slope drawn (informal comparison, no CI fitting at this stage); representative success and failure rollout mp4s at the property extremes; a gate report in STATUS.md stating whether the boundary moved in the predicted direction.
- Then **STOP** for the owner ruling.

If anything is ambiguous, state your assumption in the ralplan rather than blocking.
