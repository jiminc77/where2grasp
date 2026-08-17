# Adjudication of the 2026-08-17 external inspection (GPT-5.6 Pro)

Source: [gptpro_inspection_2026-08-17.md](gptpro_inspection_2026-08-17.md) (33 numbered findings: 2 BLOCKER / 23 MAJOR / 8 MINOR). Adjudicated by the orchestrator under the standing selective-adoption rule. Independent spot-checks performed before adoption: the finding-5 lumped-mass ratio r_N = (N+1)(3N+1)/(3N^2) was re-derived from the point-load superposition formula and matches; the finding-3 bracket-endpoint value Pi_g(0.21) = 0.524 recomputes; the finding-10 arithmetic (0.16% point difference vs ±5.1% grid bound) recomputes.

Headline: the audit **upholds the Q1 formal frozen GO** (no error found that overturns it) and re-characterizes the prefactor agreement as internal closure of the same-simulator B_eff calibration rather than independent mechanics confirmation. The narrowed four-conjunct novelty sentence **survives** the 2024–2026 re-search; the adjacent broad claim was reduced (applied, see below).

## A. Adopted now — folded into the hardening-B manifest freeze (delivered to the executor before freeze)

| # | Finding | Disposition |
|---|---|---|
| 1 | BLOCKER: interior optimum not guaranteed by mechanics | ADOPT. Freeze requirements: explicit continuous objective formula; existence/uniqueness argument on the admissible domain; predicted optimum shift ≥ 2 grid cells precomputed from prior committed calibration data; frozen tie/clipped/censored-optimum rules. Note: the executor's own ralplan independently surfaced the same defect (the mixed reach+sag scalar collapses to a material-independent argmin) and proposed the feasible-set design that resolves it. |
| 2 | BLOCKER: first-crossing vs max-crossing boundary conflict on a 0→1→0 landscape | ADOPT. ℓ_L (reach-limited) and ℓ_U (sag-limited) become separate co-primary estimands with declared crossing rules; interval metric (IoU or Hausdorff) + argmin error + selection regret; the scalar `boundary()` evaluator is not reused as-is. |
| 3 | Pi_g guard not robust to grid uncertainty | ADOPT for hardening-B onward: regime validity requires Pi_g ≤ 0.5 at the upper bracket endpoint (or the cell is INCONCLUSIVE-regime). Hardening-A's GO stands under its own frozen point-estimate rule (the audit agrees). |
| 15 | TEST = 3 unique ratio landscapes + 3 invariance controls | ADOPT for the upgrade re-run and hardening-B: ratio pairs reported as invariance controls, generalization over unique B/w groups, uncertainty at ratio-cluster level, more unique ratios where budget allows. |
| 16 | Bootstrap excludes training/cohort uncertainty | ADOPT (partial): multiple model-training seeds with seed variability reported; eval-draw bootstrap retained but labeled; full hierarchical bootstrap deferred to paper-time. |
| 17 | Student is offline batch, not sequential adaptation; missing explicit action metadata | ADOPT (partial): explicit grasp/length/action metadata added to history features (or the omission justified in the manifest). The k-interaction sequential adaptation curve is deferred to the post-hardening-B experiment (finding 32), which the logged upgrade path already points at. |
| 22 | `hard_adaptation_curve.png` misnamed | ADOPT: rename to history-variant comparison; "adaptation curve" reserved for interaction-prefix curves. |

## B. Adopted — wording/labeling corrections (applied in this commit or queued for the next doc pass)

- **10, 20**: prefactor language becomes "consistent within the declared grid resolution; point estimate differs by 0.16%"; dimensionless K = ℓ_max/(Bh/w)^{1/4} reporting; explicit fixed-h scope (h^{1/4} dependence not yet measured). Applied to README now; STATUS wording pass queued.
- **21**: task-only student label upgraded to "pre-specified secondary evidence; non-confirmatory because no frozen task-only PASS rule existed" (it was slightly under-claimed as plain exploratory).
- **24**: README headline and stale "nothing has been run yet" fixed in this commit; red-team 47/47 scope caveat added.
- **26**: proposal's "per-point-critic cell is empty" sentence reduced to the conjunction-based delta (applied in this commit).
- **6**: "fixed-discretization validation" replaces any "convergence" phrasing going forward.
- **19**: rollout counts not to be headlined as sample size (independent units = property × grasp cells); adopted as a reporting rule.
- **13**: degeneracy claim scoped to "quasi-static, gravity-only shape observations under fixed geometry/BCs"; the insertion caveat (displacement-controlled shape-only insertion may remain ratio-only; the degeneracy-breaking task must specify a fixed external load or wrench/reaction criterion) is adopted into the design constraints for the finding-33-class task.

## C. Adopted as future work (post-hardening-B experiment queue, in priority order)

1. **Finding 31 (+4, 5, 6, 7, 8)** — independent mechanics closure: known tip-force/moment B_eff calibration, mesh sequence (≥3 intervals) with the analytical r_N lumped-mass correction as reference, no-refit prediction of held-out gravity boundaries, small hardware closure. Premise of finding 5 (full segment_mass at each free vertex, right-endpoint lumping) to be code-confirmed in `material.py` before the correction is applied.
2. **Finding 32** — fresh pre-registered online cold-start adaptation trial: k-prefix (interactions, not frames) performance/regret curve, frozen acquisition policy, expanded unique-ratio cohort, multiple training seeds. This is the graduation experiment for the m_t thesis.
3. **Finding 33** — a real force/contact task that breaks the common-scale degeneracy by construction (fixed external P_task or measured reaction wrench), testing shape-only vs shape+wrench vs frozen-encoder variants.
4. Findings 9 (mean-J root as mechanics estimand, separate from stochastic success threshold), 12 (ties→INCONCLUSIVE in future gates), 14 (equivalence margin tied to a sensor/noise floor), 18 (teacher framed as adaptation ceiling; budget table), 28–30 (reviewer-attack preemption lists fold into 1–3 above).

## D. Deferred / rejected

- **11** (`max_slope` rename): cosmetic; logged, not worth a code churn now.
- **27** (related-work matrix rows: CG-CNN, In-Hand Following, CaRoBio, Saccani ICRA 2026, Active Perception for DLO Stiffness Estimation ICRA 2026): the matrix update is paper-time work, and per the standing rule these five search-surfaced works must be verified against primary sources before citation. The stiffness-estimation active-perception paper is flagged as the one that most directly touches the identification-novelty axis.
- **23**: no action needed — the audit endorses the existing STOP-branch deviation handling.

## Non-negotiables preserved

Q1 formal GO unchanged; Q2 pre-registered verdict unchanged (NOT ESTABLISHED); no hardening-A artifact is retro-edited beyond wording-level labels; every adopted item that touches confirmatory claims routes through a fresh frozen manifest, never a retroactive rule change.
