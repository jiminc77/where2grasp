# STATUS

## 2026-08-16 (fast-loop Step 0): environment + pinned-API smoke + headless render — GREEN

- **Runner:** gjc ralplan (4-iteration consensus, Architect CLEAR + Critic OKAY) → ultragoal, on AILAB-simx-remote. Approved plan: `.gjc/…/plans/ralplan/fastloop-s0-2/pending-approval.md`.
- **Env:** fresh conda env `w2g` (python 3.12.13); torch 2.10.0+cu128 sees the RTX PRO 6000 Blackwell (sm_120, cc 12.0), CUDA matmul OK. DLO-Lab (Genesis fork, genesis-world 1.0.0) cloned fresh to `~/DLO-Lab` at `c5026a9` (verified = upstream HEAD/main) and installed editable `.[dlo-lab,dev]`. No unrecorded source diff in the clone (`clone_status.txt` 0 bytes). Env exports: `sim/smoke/env_{freeze,conda}.txt`.
- **Probe-until-clean smoke** (`sim/smoke/`, all 9 PASS — `probe_results.json` + per-probe logs):
  - p0 import genesis 1.0.0 + clone-SHA provenance; p1 `gs.backend==gs.cuda`, CUDA 12.8, device RTX PRO 6000 + cuda matmul.
  - p2 cold vs warm Quadrants JIT with run-scoped `XDG_CACHE_HOME` isolation: cold 14.1 s (sm_120 kernel-compile evidence observed) vs warm 5.4 s — cold compile VERIFIED.
  - p3 static cantilever settle by velocity-tol + hard timeout (settled in 917 steps).
  - p4 moving two-vertex `attach_to_rigid_link` lift: attached verts track the rigid link (0.01), clamp tangent horizontal, tip hangs below clamp. (attach requires n_envs≥1 batched.)
  - p5 headless NVIDIA-EGL render → mp4 (30 frames, non-black std 26.6, fg 0.97).
  - p6 heterogeneous batch n_envs=4: droop monotone-decreasing with stiffness [0.372, 0.363, 0.348, 0.301]; batch-vs-serial delta 2.3e-4.
  - p7 directional property: higher bending stiffness → less droop; higher segment_mass → more droop; setter getter read-back.
  - asset_check: DLO textured assets (`genesis/assets/dlo-lab`) are AUTH-GATED (HTTP 401, UMass SharePoint) and absent, so textured examples (quick_example.py / grasp_rod.py) can't run; the asset-free README quick-start scene is used for the shipped-example render (`sim/smoke/asset_note.md`).
- **Step-0 deliverable mp4:** `sim/smoke/artifacts/step0_shipped_example.mp4` (asset-free DLO-Lab README quick-start scene, headless EGL, 70 frames, non-black) + `sim/smoke/artifacts/p5_render.mp4`.
- **Pinned-API facts locked for Steps 1–2:** `material.E` IS the bending-stiffness knob; live `set_bending_stiffness((n_envs,))` / `set_segment_mass((n_envs,n_vertices))`; moving clamp via `attach_to_rigid_link(box.base_link, verts_ids)` + `box.set_pos` (batched n_envs≥1); `ParameterizedRod(type="rod", interval, rest_state="straight")`; Rasterizer over NVIDIA EGL.
- **No blocker** (torch.cuda OK, sm_120 compiles, EGL renders). Next: Step 1 — lift-and-clear task + B/w hooks + virtual B_eff calibration.

## 2026-08-16 (latest): proposal v3 SHIPPED — third feedback round absorbed

- Round-3 external feedback (16 points) adjudicated: feedback's own errors caught (CMA-ES weak task is Separation not Wrapping; oscillation frequency does not separate B from w; duplicate citation of arXiv:2605.06323 under two names). Structural decisions: m_t scoped to a task-family-sufficient latent for the quasi-static self-weight family with the (cB, cw) shape-channel degeneracy derived (y'''' = -w/B) and wrench identified as the cheapest qualifying probe; z_obj/z_int factorization (friction is interface, out of the central claim); cold-start protocol with the adaptation curve as headline metric and a prescribed-probe one-shot as required baseline.
- New verified prior art absorbed: DeliGrasp (CoRL 2024), GraspCoT (ICCV 2025) — property-aware grasping conceded as populated; AssistDLO (arXiv:2605.06323, preprint) — nearest neighbor, measures rope EI/λ via heavy-elastica test and uses the same dimensionless group (K ≡ Pi_g), but no along-object landscape and no predicted-vs-measured boundary. Narrowed novelty sentence survives all four.
- v3 audit: 2 blockers (capstan regression re-fixed with correct tension roles; degeneracy claim scoped to shape observations across all five occurrences), 4 fixes (tau/k de-overload, S_mech rename, vocabulary discipline, Step 3 wording), 4 nits — all applied and anchor-verified. ~7,070 words; length accepted over cutting audited substance.

**Next:** fast validation loop Step 1: minimal diagnostic-task build (lift-and-clear, horizontal-clamp scene per the regime block) + property variation hooks + virtual B_eff calibration, on the RTX PRO 6000.

## 2026-08-16 (later): proposal v2 SHIPPED after three audit rounds

- strategy/proposal_v2.md is the canonical plan. Audit trail: round 1 full audit (2 blockers: unwritten friction inequality, thesis-sentence self-contradiction; 5 fixes; 10 nits) applied; round 2 targeted re-check caught an inverted capstan inequality (blocker) plus 2 wording nits; all corrected and re-verified. Final state: friction now enters as written Coulomb (ell_hang <= mu ell_sup) and capstan (T_load <= T_hold exp(mu beta)) inequalities; thesis split cleanly (admissible-set movement from mechanics, optimum movement given an objective, both identifiable from interaction); zero dashes, zero numeric targets in plan sections, all citations on the verified allow-list.
- Document length ~5,200 words after mandated additions; ~250 words of redundancy trimmed.

## 2026-08-16: workspace created, proposal v2 in progress

- Pivot from DGCC finalized. New thesis: task-conditioned grasp-point selection whose material-dependence is derivable (feasible free-length boundary) and identifiable by interaction (m_t module). Two-paper arc: this paper proves the invisible material axis and delivers m_t; DLO-JEPA integrates the obvious task axis as the sequel system.
- Proposal v1 was written, independently audited (14 arXiv IDs + 2 DOIs resolved against primary sources; 3 blockers, 11 fixes applied), then revised against 19-point external feedback whose factual claims were themselves verified (5 of 6 fully true, 1 partially).
- Key verified repositioning: Wiggle and Go! (arXiv:2604.22102) promoted to primary methodological threat (interaction sysID, cross-task module reuse, zero-shot real; but self-declared simulator-behavioral parameters, fixed tool attachment). Li and Choi delta corrected (they also interact once; our delta is multi-step history + task-sufficient latent + per-point map + derived boundary). Diagnostic tasks (lift-and-clear, distal tip placement) carry the law measurement and the go/no-go; gap placement demoted to downstream demonstration.
- Proposal v2 being drafted, to land at strategy/proposal_v2.md, followed by a verification pass.

**Next:** fast validation loop Step 1: minimal diagnostic-task build with property variation hooks, on the RTX PRO 6000.

**Go/no-go gate (the only one):** does the feasible grasp band move with material properties in the predicted direction, for the diagnostic task and motion family? If not, stop and rethink before any scaling.
