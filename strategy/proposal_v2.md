# where2grasp: Material-Adaptive Contact Selection for Deformable Linear Objects

**Date:** 2026-08-16
**Status:** v3, post-feedback revision round 2
**Codename:** where2grasp
**Note:** paper title is open; two candidates are recorded in Section 11.

---

## 1. Summary

Where a robot should grasp a rope is a task-conditioned decision. Write it as

    s* = s*(theta, x0, G, U)

with theta the mechanics parameters, x0 the object state, G the task geometry, and U the family of allowed motions. Nobody disputes that s* depends on all four. The claim of this paper separates two things mechanics treats differently: **at fixed task, how the admissible set of grasps moves with theta is derivable from mechanics; how the optimum s* within that set moves is derivable once an objective is fixed; and both are identifiable from interaction.**

The two axes of dependence have opposite epistemic status. The **task axis** is obvious: everyone grasps a rope differently for tying than for threading, so it needs *integration* into a working system, which is the sequel's job. The **material axis** is invisible: two ropes can be identical in an image and opposite in behavior, and the field designs the dependence away rather than measuring it, so it needs *proof*. That is this paper, and Section 10 develops the split.

The architecture uses two encoders, not one. **z_obj** carries object mechanics (bending stiffness, weight, damping, intrinsic curvature). **z_int** carries interface behavior (friction, contact response), because friction is a property of the object against a particular surface and no object latent can carry it across supports. The **per-point critic carries task-dependence**, instantiated per task, consuming both. Throughout, m_t means z_obj.

**Contribution 1, a mechanics-derived envelope and its scaling.** Not an "optimal grasp law". Beam and cantilever mechanics supply necessary conditions on which grasp positions can succeed and predict how those conditions move with stiffness, weight, length, and interface friction. Choosing a point inside the admissible set (throughout this document a synonym for the mechanics envelope defined in Section 3) requires an objective, which is a design choice.

**Contribution 2, m_t, a mechanics encoder with a stated scope.** It maps interaction history to a latent sufficient for a declared *task family*, not for all tasks, and the mechanics itself says which family and what extra excitation extends it. Probed against measured ground truth, reused frozen across tasks, specified so it can be lifted into a larger system.

**Contribution 3, the first consumer: property-adaptive per-point grasp selection.** A critic scoring every candidate point, conditioned on z_obj and z_int. The headline analyses are whether its selections show behavior consistent with the predicted mechanics scaling under held-out interventions, and how fast it adapts from a cold start.

One question gates everything: **on the cleanest diagnostic task, does the boundary of the empirical feasible set move with mechanics parameters in the predicted direction?** Section 9 settles it.

---

## 2. Problem and thesis

Manipulation policies degrade on deformables because the variables governing the contact decision are only partially recoverable from appearance.

The honest form of that matters, because the vision literature has moved. Purpose-built vision-based property estimation works: PhysGS (CVPR 2026, CVF proceedings pp. 18980-18990) demonstrates it, and SiPhy (arXiv:2607.22355, preprint, authors state ECCV 2026 acceptance) estimates mass, density, and Young's modulus from a single image plus depth, though on neither DLOs nor grasping. Vision-based estimation is advancing and our framing accommodates that rather than betting against it.

Our claim: **mechanics is only partially and ambiguously observable from appearance; interaction provides complementary evidence and resolves equivalence classes that visual priors alone may not distinguish.** Those classes are concrete. Section 7 shows that scaling stiffness and weight by a common factor leaves quasi-static self-weight *shape observations* unchanged while changing what insertion cares about, so any estimator, visual or interactive, must be judged against the specific ambiguity the task needs resolved and the sensor channel it reads. Generalist VLMs remain weak at property estimation, as PhysGS's own GPT-4V and GPT-5 baselines show (their stiffness comparison appears in the supplement), and Physion++ (NeurIPS 2023), ContPhy, and PhysBench (ICLR 2025 Oral) document the general-model side of that gap.

Humans behave as though the decision lives where we claim: grasp location shifts with deformability (Mazzeo et al., Scientific Reports 2024). A measured effect is a phenomenon to explain, not a hunch to defend.

Robotics routes around it. In DLO-Lab (ICML 2026, arXiv:2606.04206) the released environment implementations instantiate fixed control_idx values. Luo et al. (T-RO 2024, arXiv:2307.08927) train for robustness to grasp-point variability and explicitly use pull-taut and slack-reduction primitives, rather than optimizing grasp location as a decision variable. The decision is either fixed by hand or absorbed as variability.

The question, in one sentence: **at fixed task and fixed motion family, is the feasible grasp set a predictable function of mechanics parameters, and can a robot identify the relevant parameter combinations by interaction alone?**

---

## 3. Mechanics: the derivation program and the admissible envelope

### Notation and two arms

Let L_tot be total length, s in [0, L_tot] the grasp coordinate, and ell(s) the free length from grasp to task-relevant tip. Let B = EI be bending stiffness and w = lambda g weight per unit length. B is a structural property (material times cross-section), lambda a linear mass density, L_tot geometry, and friction an interface property. The grasp point is not itself a material property.

An interior grasp creates two free arms, ell_L(s) = s and ell_R(s) = L_tot - s. The derivation below applies to designated-tip tasks, where one arm is task-relevant and ell(s) means that arm. The general critic receives both arms and both shapes, and gap placement never uses the single-arm law directly.

### Dimensionless groups

The primary group for the gravity-bending constraint is

    Pi_g = w ell^3 / B = (ell / ell_gb)^3,   with ell_gb = (B / w)^(1/3)

the gravito-bending length (Miller et al., PRL 2014). Pi_g is primary for that constraint specifically, not a universal axis onto which everything collapses. Each task carries a **Pi vector**: Pi_g for gravity-bending, Pi_P = P_task ell^2 / B for buckling, eta_h = h / ell for clearance geometry, and mu, beta, and ell_hang / ell_sup for interface constraints. Each task declares its Pi vector before any collapse plot is drawn, because collapse is criterion-dependent. With delta_tip the tip droop below, an absolute clearance criterion gives

    delta_tip / h = Pi_g / (8 eta_h)

which is two-dimensional and does not collapse onto Pi_g alone, while a relative droop criterion gives

    delta_tip / ell = Pi_g / 8

which does. A one-dimensional collapse plot for an absolute-criterion task is a plotting error, not a finding.

### The constraints

**Sag.** For a cantilevered free length under self weight, delta_tip = w ell^4 / (8B). The exponent of the resulting bound depends on the success criterion, so each task states its criterion before quoting a scaling.

- Absolute clearance, delta_tip <= h, gives ell_sag proportional to (B h / w)^(1/4), a fourth-root scaling.
- Relative droop, delta_tip / ell <= epsilon, gives ell_sag proportional to (epsilon B / w)^(1/3), a cube-root scaling and a direct statement about ell_gb.

Measuring which exponent a task exhibits is a sharper test than asserting one exponent everywhere.

**Buckling.** For tasks driving the object against a constraint with load P_task, the free length must stay below

    ell_buckle = (pi / K) sqrt(B / P_task)

with K the effective-length factor for the tip boundary condition.

**Geometry.** Task geometry imposes a lower bound ell_min from required overhang, insertion depth, and collision avoidance.

**Mechanics envelope.**

    S_mech = { s : ell(s) in [ell_min_geometry, min{ell_sag, ell_buckle}] }

S_mech is the **mechanics envelope**, and "admissible set" is used in this document as a synonym for it.

Workspace kinematics sit on top of this rather than inside it: reachability and joint limits can truncate the envelope from either side depending on layout, reported per task.

**A correction worth stating plainly.** Cantilever stiffness k = 3B / ell^3 and the buckling load P_cr both *decrease* with ell, so they bound the free length from the *same* side. It is wrong to present them as squeezing the envelope from opposite ends. The opposing bound comes from task geometry.

**Friction.** Friction enters as an inequality once support contact geometry is fixed, in one of two forms. On a flat support the Coulomb condition bounds the tangential load T by the normal load N, with mu the interface friction coefficient:

    T <= mu N

The overhang-holding form follows. With ell_sup the length resting on the support and ell_hang the length hanging beyond, N = w ell_sup and T = w ell_hang, so the contact holds while

    w ell_hang <= mu w ell_sup,   that is,   ell_hang <= mu ell_sup

Under the self-weight-only idealization the weight scale cancels, leaving a bound in mu and a length ratio. Where the object wraps a curved support, the capstan relation applies. Let beta be the total wrap angle, T_hold the smaller tension applied on the holding side, and T_load the larger tension being resisted. The wrap holds without slipping while

    T_load <= T_hold exp(mu beta)

The wrap angle is the design lever, and its exponential effect is why draping over a peg tolerates load ratios a flat contact cannot.

### Regimes of validity

Each equation is a model with a domain, and the tasks are designed to sit inside those domains rather than the equations stretched to cover them.

*Sag* assumes an initially straight, horizontally clamped free segment under transverse gravity, small deflection, linear response, constant B. A vertically hanging segment is axially loaded and the formula does not apply. The lift-and-clear scene is therefore designed to present a horizontal clamp analog, and the linear-regime range is declared in advance through a Pi_g threshold rather than chosen after inspecting data.

*Euler buckling* is an idealized upper-bound reference calibrated for the task's effective boundary conditions. Real insertion carries initial curvature, eccentric contact, and gripper compliance, all lowering the achievable load, so it is a reference rather than a prediction of the failure point.

*Coulomb* cancellation holds under the self-weight-only idealization: horizontal support, quasi-static motion, no payload, edge friction neglected, full supported segment loading the contact. Departures reintroduce the weight scale.

*Capstan* is an incipient-slip model for a perfectly flexible line. Stiff rods and chains deviate, so it serves as a regime-dependent reference inequality.

### The feasible set is an empirical object

Define the planner landscape relative to the motion family U, and the empirical feasible set at threshold tau:

    Q*_U(s; theta, G, x0) = max over u in U of J(s, u; theta, G, x0)
    F_U(tau) = { s : Q*_U(s) >= tau }

Mechanics supplies **necessary conditions only**: F_U(tau) is contained in S_mech, provided tau is the same success criterion that defines ell_sag through h or epsilon. Containment is a statement about one criterion applied consistently on both sides, not a general relation between an arbitrary threshold and the envelope. Mechanics supplies no sufficiency, and F_U(tau) need not be an interval. Collision can remove interior regions, kinematics can disconnect it, contact modes can differ across candidates, motion templates can be locally invalid, and stochastic settling can make success non-monotone in s. Whether the empirical set is close to an interval is a finding of Step 2, not an assumption, and the two objects stay verbally distinct: **mechanics envelope** for S_mech, **empirical feasible set** for F_U(tau).

### Optimality and the compliance anchor

A unique optimum does not fall out of the admissible set. Choosing a point inside it requires an explicit objective: clearance margin, robustness margin to parameter error, expected success under the motion family, or effort. We report which one the critic is trained against.

"Control authority" is used across this literature as though defined, and DLO-Lab uses the phrase without defining or measuring it. We offer no closed form for it. Where an anchor is needed, k = 3B / ell^3 serves as a local stiffness proxy, with compliance ell^3 / (3B) giving deflection sensitivity to a force input. That is a proxy with a known domain, not a definition.

### Against the fitted alternatives

Berenson (IROS 2013) captured decay of influence with "diminishing rigidity", a hand-tuned scalar with no derivation. Li and Choi's Visual Baseline (ICRA 2024, arXiv:2410.23428) pushes a scalar flexibility estimate through a square-root function to a grasp index. Neither is the same object as ours: they fit an unexplained monotone map from a flexibility proxy to a position, while mechanics hands us a length scale derived from B and w plus a criterion. Both nonetheless prescribe how the grasp location should move as flexibility changes, which is why a fitted curve and a derived boundary can be compared.

---

## 4. Prior art and open ground

### Property-aware grasping is populated

Conditioning grasps on inferred physical properties is active, and the honest opening concedes that. DeliGrasp (arXiv:2403.07832, CoRL 2024) has an LLM infer mass, friction coefficient, and spring constant to parameterize adaptive grasp **force** on delicate and deformable items such as food, produce, and toys. GraspCoT (arXiv:2503.16013, ICCV 2025) performs chain-of-thought reasoning over physical properties to produce 6-DoF grasp poses for rigid tabletop objects with categorical property labels. Neither is a competitor: the decision variable is how hard to squeeze and which pose to approach from, and neither touches DLOs, continuous property interventions, or along-object landscapes.

### AssistDLO, the nearest neighbor

AssistDLO (arXiv:2605.06323, preprint, submitted May 2026) is the closest existing work overall. It builds assistive teleoperation for DLOs, characterizes four ropes by measured linear density and flexural rigidity using a large-deflection heavy-elastica cantilever test, organizes its analysis around the dimensionless gravity-loading group K = lambda g L^3 / EI, which is the same group as our Pi_g = w ell^3 / B under w = lambda g, explicitly notes that a DLO offers a continuous manifold of candidate grasp points, and reports success and completion time per rope type. Its status is unrefereed, stated neutrally.

It obliges a correction to our own claim. **We are not the first to use the gravito-elastic group in a DLO context.** What AssistDLO never does is resolve outcomes along the object, so there is no grasp-quality landscape, and it never compares a boundary against a mechanics prediction. Our position, to our knowledge and in the reviewed DLO literature: first to use the group to *predict grasp placement*, and first to *measure the along-object landscape it predicts*. Its rope-characterization methodology is close to the protocol we need, so it is prior art we also adopt from.

### Li and Choi

Li and Choi (ICRA 2024, arXiv:2410.23428) place the grasp particle index i_p in the action space conditioned on estimated flexibility. Their estimate follows a single prescribed robot interaction, grasping with a predefined pose and applying a GNN over the particle graph of the resulting hanging configuration. They report a with-flexibility versus without-flexibility gap (78% vs 56%), itself evidence the property signal carries the decision.

The correct delta: a single prescribed interaction producing an explicit scalar, versus multi-step task-driven history producing a task-family-sufficient latent; a scalar index emitted by the policy, versus a per-candidate-point map; and a fitted monotone mapping, versus a derived envelope with a stated criterion. HACMan (CoRL 2023, arXiv:2305.03942) shows the map is where value lives: removing their actor map costs little (83.5 vs 85.4), while removing contact-location selection costs a great deal. Li and Choi never report the learned flexibility-to-grasp map; we make that map the object of study. Their group's Hierarchical DLO Routing (arXiv:2510.19268) reuses a predicted grasping index downstream. Because their pipeline is a prescribed-probe one-shot estimator, it also defines a baseline we are obliged to beat (Section 7).

### Wiggle and Go!, closest on program structure

Wiggle and Go! (arXiv:2604.22102) is the closest existing work on **program structure**: a task-agnostic identification module built once and reused frozen across downstream tasks. It executes a predefined wiggle trajectory for system identification, and the same representation is reused across three downstream tasks (striking, lobbing, draping) without retraining, with zero-shot real-world execution.

Our deltas are specific. Their identified parameters are, in their own words, "behavioral descriptors of this ball-joint representation rather than direct measurements of a physical rope", and they state these parameters do not correspond one to one with measurable qualities of the ropes. We target measured physical quantities and derived dimensionless groups, which is what makes an external ground truth and a scaling prediction possible. Their tasks are dynamic open-loop swings; ours are quasi-static contact-rich decisions where the settled configuration is the outcome. And their rope is fixed to the tool, so along-object grasp position is not a decision variable. Wiggle and Go! anchors our interaction-sysID baseline row and, like Li and Choi, the prescribed-probe comparison.

### DLO-Lab, in two layers

**What it establishes.** DLO-Lab states that grasp selection is a critical determinant of success and compares VLM grasp proposal modes (Candidate, Coefficient, Marker). The roughly 19x return gap on Unknotting in its appendix grasp-mode table is a comparison *between VLM proposal modes*, so it is evidence that grasp proposal strongly affects return, not evidence of material-dependence.

**What its code does.** At commit c5026a9 (verified 2026-08-16), the eight released environment implementations instantiate fixed control_idx values, while the paper separately evaluates VLM grasp-proposal modes. The repository ships no VLM code.

DLO-Lab demonstrates that grasp proposal materially affects task return, but does not study how the grasp-quality landscape changes under controlled material-property interventions.

The anticipated objection is that DLO-Lab already asks a VLM for a grasp coefficient in [0,1], so grasp selection is not new. Correct, and our claim is narrower: to our knowledge, in the reviewed DLO literature, we contribute the first setting where that coefficient's mechanics envelope is physically derivable and demonstrably moves with material.

### AdaptiGraph, RAPiD, and the rest

**AdaptiGraph** (arXiv:2407.07889) is the closest existing work on **adaptation mechanism**: interaction-history property inference feeding a planner. Material-conditioned GNN dynamics, test-time property inference by optimization, MPPI planning, multiple rope varieties. Same adaptation signal, no critic. Related work and full-system benchmark both.

**RAPiD** (arXiv:2603.18246) applies RMA to deformables and evaluates both a 1D insertion task and a 2D covering task, so it is not 1D-only. It does not isolate along-object grasp-index selection, and for the covering task it reports no explicit randomization over pose, stiffness, or friction, inducing those variations through object-instance randomization instead. It reports no oracle bound.

**Strong non-learned selectors are a live risk.** The ICRA 2024 Cloth Competition report (IJRR, arXiv:2508.16749, doi 10.1177/02783649251414885) records non-learning methods placing 1st and 3rd of 11 teams. Beating a weak midpoint heuristic proves nothing.

**Foresightful Dense Affordance** (ICCV 2023, arXiv:2303.11057) already produces per-point value maps for ropes and fabrics, property-blind; we do not claim per-point maps as architecture. **GenDOM** (ICRA 2024, arXiv:2309.09051) anchors explicit parameter estimation plus conditioning. **Kamaras and Ramamoorthy** (arXiv:2502.18615) use BayesSim posteriors to shape a domain randomization distribution for PPO, never conditioning the policy on the parameters. **Navarro-Alarcon and Liu** (T-RO 2018) are the classical ancestor of the interaction buffer. **WireCraft** (arXiv:2606.18097) reports that vision and VLA pipelines bottleneck at contact-rich DLO tasks where privileged-state RL succeeds, and makes imitation rows table stakes.

### Open ground

Across the benchmarks, surveys, and primary sources we checked, and granting everything the neighbors establish (Li and Choi on property-conditioned grasp index, Wiggle and Go! on reusable identification modules, AdaptiGraph on interaction-history adaptation, DeliGrasp and GraspCoT on property-aware grasping, AssistDLO on gravito-elastic characterization of DLOs), this remains open:

**To our knowledge, no prior work in the reviewed DLO literature measures an along-object grasp-quality landscape under controlled mechanics-property interventions and compares its boundary against a task-specific mechanics prediction.**

Alongside it, the material-conditioning by per-point-critic cell is empty, and a mechanics-derived envelope with a stated criterion has not replaced a fitted monotone scalar.

---

## 5. The paper we intend to write

**§1 Introduction, diagnosis first.** Mechanics parameters are only partially recoverable from appearance. Humans shift grasp with deformability (Mazzeo et al., 2024) while benchmarks fix the grasp point or absorb its variation (DLO-Lab; Luo et al., T-RO 2024). State the task-axis versus material-axis split, then the question from Section 2.

**§2 The mechanics of where to grasp.** Section 3 in full: notation and two arms, the per-task Pi vector, the constraints with their regimes of validity, the friction inequalities, the envelope as a necessary condition, and optimality requiring an objective.

**§3 Grasp landscapes, Figure 1.** Using the Q*_U and F_U(tau) definitions, sweep candidate grasp points against controlled parameter settings. Three consequences are stated rather than buried: an insufficient template family misplaces the peaks, enlarging U can change the landscape, and the peaks depend on task geometry as well as material. Existing selectors' picks are plotted on the same landscape.

**§4 Method.** z_obj and z_int, the per-point critic, the task-family scope argument, the cold-start protocol. Lineage stated honestly: per-point critic from HACMan, teacher-student recipe from RMA (Kumar et al., RSS 2021), interaction buffer from Navarro-Alarcon and Liu, task-agnostic identification module from Wiggle and Go!, rope characterization close to AssistDLO's.

**§5 Experiments.** Two benchmarks, two ceilings, the adaptation curve, the scaling-consistency figure, the real-robot protocol.

**§6 Limitations and what we do not claim.** Not discovering that grasp depends on material (Mazzeo; Li and Choi). Not introducing per-point value maps (Foresightful Dense Affordance). Not the first task-agnostic identification module (Wiggle and Go!), the first property-aware grasping (DeliGrasp; GraspCoT), or the first gravito-elastic characterization of DLOs (AssistDLO). No claim to a good DLO dynamics model: that ground is AdaptiGraph's and the sequel's.

---

## 6. Task design

### Primary analytic diagnostics

**Lift-and-clear** and **distal tip placement** are primary, because Section 3 is directly measurable in them: a single grasp, a single primitive, a settled outcome, and a criterion mapping onto delta_tip without intermediate composition. Both scenes present the horizontal clamp analog the sag formula assumes. The go/no-go of Section 9 runs here.

### Gap placement, a downstream demonstration

Two rigid supports with a gap, a failure plane below the support tops, the object starting on the table, success requiring both ends supported with mid-span sag above the plane. This is a downstream demonstration rather than a primary diagnostic, for a mechanical reason.

Gap placement composes transport, contact, friction, and settling, and critically **the grasp point does not persist in the settled equilibrium**. A flexible spanning rope settles into a tension and catenary dominated shape, a stiff one into a beam dominated shape, and neither final configuration remembers where the gripper was. The grasp point acts *indirectly*, through the landing configuration it produces, the overhang it leaves, and the friction-locked metastable states it makes reachable. That is a real effect and a poor instrument for a scaling law. It is also a two-arm problem throughout, so the single-arm law never applies directly. If it is ever used for a law claim, the regime (beam versus catenary) must be classified first.

Its lineage and grounding stay. The nearest parent is the Draping task of Wiggle and Go!, rope thrown over a wall with heavy material variation but rigidly attached to the tool. A secondary lineage treats a spanned cable as a structure: the ETH aerial rope bridge (Augugliaro et al., Flying Machine Arena), and the catenary robot (RA-L, arXiv:2102.12519), which makes span and sag explicit control degrees of freedom with both cable ends held by actively controlled aerial robots, so again there is no along-object grasp choice. The Berkeley cable routing work is the clean inversion of our framing. Industrially, transmission line design evaluates ground clearance at final sag under worst-case conditions (USDA RUS Bulletin 1724E-200), and cable tray standards treat support span as the design lever against sagging (NEMA VE 2). The task remains unclaimed: DLO-Lab, DaXBench, SoftGym, DEDO, MoDeSuite, and WireCraft were checked, and the IJRR 2026 survey (doi 10.1177/02783649261432253) contains zero occurrences of "sag", "drape", "clearance", or "catenary", with no gap-spanning family in its taxonomy.

### Staging protocol

Complexity is admitted in stages, each supporting only the claims it can carry.

1. **Single grasp, fixed primitive, single release.** Law validation happens only here.
2. **Multi-step without regrasp.** Adaptation and history use, with a persistent contact.
3. **Regrasp-enabled full task.** System-level results only.

### Extension tasks

**Hook and peg draping** is the **z_int extension demonstration**: a wrap geometry where the capstan relation is binding, so the placement sets beta and therefore how large a resisted tension a given holding tension can support, and the same object is re-evaluated against a different support surface. It is the task showing why the interface latent cannot fold into the object latent.

**Tip insertion** is the real-world flagship and the direct contrast with Li and Choi. It also sits outside the primary encoder's scope until a degeneracy-breaking probe is added, for the reason in Section 7.

---

## 7. Method: encoders, scope, and the per-point critic

The interfaces are fixed now; the internals are decided by the loop.

### History and interface

Write the interaction history and policy as

    H_t = { (o_i, a_i, o_{i+1}, g_i, chi_i) }, i = 0 .. t-1
    s_t = pi(H_t, o_t, G)

with o the observation, a the action, g_i the grasp index, chi_i contact annotations. Two disciplines apply. Grasp index and contact annotations are kept as **metadata alongside** the observation sequence rather than concatenated into it, being categorical and sparse where observations are dense. And every component is labeled by availability: sensor-observable on hardware (gripper pose, wrench, vision-derived shape, contact events) versus simulation-privileged (exact particle states, true contact modes). Privileged components serve teacher supervision and analysis only, never the deployed student path.

The input channel of z_obj is declared rather than left implicit, because Section 7's scope argument turns on it. **By default z_obj consumes the shape channel (vision-derived configuration) plus action metadata. The wrench channel is an explicitly flagged optional input, and switching it on is exactly what admits insertion into the task family.** The theory therefore predicts which sensor channel breaks which degeneracy, and the evaluation grid tests that prediction directly.

### Scope: what m_t can be sufficient for

Self-weight statics is degenerate, and the degeneracy is stated rather than left for a reviewer. Take two objects related by a common scaling, (B2, w2) = (c B1, c w1). Static equilibrium satisfies y'''' = -w / B, which is invariant under that scaling, and their gravito-bending lengths are identical since ell_gb = (B / w)^(1/3) depends only on the ratio. So **every quasi-static self-weight interaction produces identical shape observations: the degeneracy lives in configuration space, not in force space.** Forces are not invariant, and neither are buckling limits, since ell_buckle = (pi / K) sqrt(B / P_task) differs by sqrt(c). A history of shape observations therefore cannot distinguish two objects that insertion must distinguish. This is precisely why a wrench reading is the cheapest qualifying probe: it reads the force channel the shape channel cannot.

The consequence is a scope statement. Define m_t as a **task-family-sufficient latent** for a declared family T: a latent z(theta) such that

    Q*_k(s; theta, x, G) = Qtilde_k(s; z(theta), x, G)   for all k in the family T

For the quasi-static self-weight family (lift-and-clear, relative-sag placement, quasi-static draping), the ratio-sensing latent is sufficient. Insertion joins the family only when the history contains a **degeneracy-breaking probe**, and mechanics derives which probes qualify: a known lateral force-deflection test, delta = F ell^3 / (3B), returning B alone; a known added mass, changing w by a known amount; or a wrench-based weight measurement, returning w directly and cheapest of the three since the gripper already holds the object. One probe does **not** qualify: free oscillation frequency, since omega scales as sqrt(B / lambda) / ell^2 and senses the same ratio statics already senses. Adding a wiggle does not break this degeneracy.

Framed positively, this is a result. **The theory predicts not only where to grasp but what excitation each task family requires**, and it rules out an intuitively appealing probe on dimensional grounds before any experiment runs.

### Factorization: object versus interface

Friction is mu(object, support, surface state), not a property of the object, so a single object latent cannot carry it across supports and pretending otherwise guarantees a transfer failure the first time the support changes. Hence:

- **z_obj = f_obj(object interaction history)** carries bending stiffness, weight, damping, intrinsic curvature.
- **z_int = f_int(contact interaction with the current interface)** carries friction and contact behavior.
- The critic is **Q_k(s; x0, G, z_obj, z_int)**.

This removes friction from the central mechanics claim, which now concerns object mechanics under a declared interface, and gives hook-and-peg draping a defined job. Throughout, m_t means z_obj.

### Cold start and the deployment protocol

At first encounter there is no history, and the protocol says what happens rather than assuming a warm latent. The first grasp is chosen by the **property-blind prior critic**. Adaptation begins with that first interaction's evidence and updates every step. The **adaptation curve**, task performance against number of interactions, converging toward the ground-truth-property ceiling, is promoted to a headline metric: it is the honest measure of what history-based adaptation buys and how fast.

That promotion imposes a burden we accept. Our stated delta over Li and Choi and Wiggle and Go! is that multi-step task-driven history beats a single prescribed probe, so a **prescribed-probe one-shot pipeline** (execute a predefined diagnostic interaction, estimate, then act) is a required baseline. If task-driven history does not beat it, the delta is not real, and we would rather learn that in the loop than in review. Latent persistence across episodes with the same object is a system option, useful in deployment, deliberately not part of the claim.

### Why a latent rather than a point estimate

RMA (Kumar et al., RSS 2021) and HORA (Qi et al.) both report explicit identification underperforming latent teacher-student adaptation. Our reason is bounded: **the latent avoids imposing a preselected physical parameterization, but it does not explicitly represent posterior uncertainty**. A deterministic encoder emits a vector, not a distribution. Stochastic or distributional encoding becomes worthwhile only if ambiguous histories are shown to matter in practice, a loop observation rather than a design commitment. Kamaras and Ramamoorthy (arXiv:2502.18615) motivate the concern, keeping BayesSim posteriors to shape randomization rather than conditioning on an estimate. We carry explicit sysID as a baseline rather than assuming the published result reproduces here.

Training is two-phase teacher-student: the teacher receives true mechanics parameters, the student reproduces the conditioning latent from interaction history alone.

### Module-level validation

The latent is **probed against measured ground truth**, in simulation and on real ropes. It is **reused frozen across tasks**, trained on one task's interactions and applied without retraining on another. Frozen transfer **provides evidence that the representation captures reusable mechanics information**; it does not by itself prove a material-versus-task factorization, since the encoder sees states and actions and can absorb task statistics along with mechanics. The stronger factorization evidence is reserved and listed in Section 11. And the **interface is liftable**, specified tightly enough to drop into a larger system unchanged.

Real properties are measured with a Peirce-inspired cantilever test, with AssistDLO's heavy-elastica characterization as a close methodological reference. ASTM D1388 (current edition D1388-23) is a fabric stiffness standard for woven and knit fabrics, and ropes and cords fall outside its scope by omission, so the protocol is described as Peirce-inspired, its conversion validated against an independent bending-stiffness measurement, reporting an effective stiffness B_eff that acknowledges tension dependence, internal strand sliding, and hysteresis.

### The per-point critic

The critic scores every candidate grasp point, conditioned on z_obj and z_int. FiLM is the default conditioning, motivated by DyWA and treated as a **motivated default from a rigid-object domain whose advantage is re-tested here, not assumed to transfer**. The output is a map, not an index, which is what makes point-by-point comparison against the mechanics envelope possible.

---

## 8. Evaluation design

Two benchmarks, deliberately separated, because mixing them is how selector claims get contaminated by planner budget.

### Selector-only benchmark

Identical candidate set, identical fixed motion family, identical interaction history for every row; only the grasp-index choice differs. Rows: random; tuned best-of geometric heuristics as one row; a VLM proposer; property-blind per-point critic (ours minus adaptation, essentially HACMan); scalar-flexibility monotone-map selector in the style of Li and Choi's Visual Baseline, labeled as their baseline and not their method; the prescribed-probe one-shot pipeline of Section 7; and ours.

The VLM proposer needs a caveat, since this document has established that DLO-Lab ships no VLM code. With no reference implementation to adopt, the row is reimplemented from the proposal modes described in the paper, Candidate and Coefficient. Fidelity is our responsibility and is reported as such, including prompt, model, and the mapping from model output to grasp index.

The **adaptation curve** is reported here, not only the endpoint, so a method reaching a good grasp after many interactions is not confused with one reaching it immediately.

### Full-system benchmark

Methods with different action spaces and budgets, with those differences reported rather than hidden. CMA-ES over grasp and motion, which is **our extension implementation**: DLO-Lab's own CMA-ES optimizes motion with the grasp fixed, and it is strongest on 7 of 8 tasks there, worst on Separation. AdaptiGraph-style material-conditioned dynamics with planning. An imitation row, since WireCraft made ACT, Diffusion Policy, and pi-0.5 standard. Explicit sysID pipelines, including a Wiggle-and-Go-style interaction sysID baseline. Every row reports **rollout counts, compute, and privileged access**.

### Two ceilings, clearly distinguished

- **Adaptation ceiling:** a policy given ground-truth parameters. It bounds what adaptation could recover but is not a true oracle, since learning can fail for reasons unrelated to property knowledge. The adaptation curve is read against this line.
- **Task oracle:** the best grasp under the same motion family, argmax over s of Q*_U(s). The upper bound on the decision itself.

RAPiD reports no such bound, so carrying both is also a differentiator.

### Property grid design

Vary B and w **independently**, and deliberately include pairs related by a common scale factor, (cB, cw). Those pairs are the experimental handle on the degeneracy of Section 7, and the prediction is conditioned on the encoder's input channel. If the encoder consumes shape-channel evidence only, (cB, cw) pairs should collapse. If the wrench channel is included, they should separate, and insertion should become solvable in step with that separation. The same grid therefore tests the sufficiency claim, its stated boundary, and the channel account of why the boundary sits where it does.

### Real-robot protocol sketch

A rope set with measured properties, using Wiggle and Go!'s published set as the template since it spans light twine through steel chain and therefore covers the stiffness-to-weight axis rather than clustering. Properties measured with the Peirce-inspired protocol so simulated and real objects share one axis, with the two-cohort design of Section 11 governing which claims each rope supports. Evaluation is held-out-parameter zero-shot on a Franka. The reportable outcome of the scaling analysis is behavior consistent with the predicted mechanics scaling under held-out interventions.

---

## 9. The fastest validation loop

The immediate plan, free of targets by design.

**Step 1: minimal diagnostic build, with stiffness calibration.** A lift-and-clear or distal-tip-placement scene on the existing simulator stack, using the parameter variation hooks already in place. Gap placement is not built here.

Step 1 also includes **virtual stiffness calibration**, which is not optional. Raw simulator elasticity parameters are not assumed equal to the continuum effective bending stiffness, so Step 1 runs a simulated cantilever load-deflection test producing a calibration curve from raw parameters to B_eff. Every scaling plot thereafter is indexed by B_eff measured with the same protocol later used on hardware, which is what makes the sim and real axes the same axis.

**Step 2: mini landscape.** A small candidate set crossed with a small parameter grid and a small motion-template family, in parallel environments. One question: **does the boundary of the empirical feasible set move with parameters in the predicted direction?**

Three hygiene rules apply from the first run, cheap now and unrecoverable later. The per-point value is the best over motion candidates of the **mean** outcome over repeated settles, never the maximum over noisy rollouts, which would buy the winner's curse into the landscape. Motion-optimization draws are kept separate from evaluation draws. And each task's success criterion, and therefore the exponent it predicts, is declared before the data are inspected.

The scope of a negative is stated honestly. If the boundary does not move for this task and action family, the task is unsuitable as the primary diagnostic. Failing on the cleanest instrument we can build is strong evidence against the thesis and would send us back to rethink rather than patch, but it does not falsify the thesis for all DLO tasks.

**Step 3, if the boundary moves: identifiability.** Reuse the Step 2 rollouts with no new landscape sweep; the only new data are the targeted probes, namely the interface probe for friction and the degeneracy-breaking probes of Section 7. Test whether the task-family-sufficient combinations are readable from short interaction histories better than from proprioception alone, then test both halves of the channel prediction: (cB, cw) pairs should be *not* separable from shape observations alone, and separable once the wrench channel is included. Predicting a failure and a repair, and observing both, is stronger evidence than either alone.

**Step 4, if readable: miniature teacher-student.** On the single diagnostic task, conditioned against blind, with three mechanism candidates side by side: history latent, explicit sysID, one-step deformation auxiliary. The prescribed-probe one-shot pipeline runs here too, since our delta claim depends on it. Directional outcomes only; this chooses the architecture.

**After the loop:** decide, then expand. Gap placement is built in post-loop expansion, alongside task expansion, the real-robot protocol, statistics, and ablations.

**Hardware.** Pilot runs on an RTX PRO 6000, with a 4090 also available. Two Franka Pandas and mountable cameras are available for the real phase. A larger multi-GPU machine exists but is not set up, and nothing here depends on it.

---

## 10. Relationship to DLO-JEPA

DLO-JEPA (owner-internal architecture document, 2026-08-11) is a VLA with explicit DLO state, a latent mechanics encoder E_H over interaction history, a frozen visual prior, and an action-conditioned predictor. It is the sequel system paper.

The two-axis argument makes the split between papers principled rather than convenient. The **task axis needs integration, not proof**: that grasp choice depends on the task is uncontroversial, and the work is building a system handling many tasks coherently, which is a system paper's job. The **material axis needs proof, not integration**: it is invisible, currently designed away, and provable only under controlled intervention against an external mechanics ground truth, which a system paper cannot deliver because every component's contribution is entangled with every other.

So this proposal builds and validates E_H under conditions that isolate it. z_obj *is* E_H. The interface contract is fixed here and inherited there: a history H_t of (o, a, o', g, chi) tuples in, a conditioning latent out, with the declared task family and its degeneracy-breaking probe requirements travelling with it, and frozen reuse without retraining as a stated deliverable of paper one. Handing the sequel a component with a known scope is worth more than handing it one whose scope is assumed.

---

## 11. Deferred hardening and open items

Held until after the loop: gap-placement construction, task expansion, the full real-robot protocol, statistical design, and appendix ablations (one-step deformation auxiliary, FiLM against concatenation, history encoder form).

**Reserved factorization analyses,** the claims frozen transfer alone cannot support: a canonical probing protocol applied identically across tasks, cross-task latent alignment on the same object, a task-ID adversarial probe testing whether task identity is decodable from the latent, and latent swap interventions between objects.

**Deferred rigor for the scaling claim.** Fitted exponents with confidence intervals against the criterion-specific predictions of Section 3. Nested motion families U1 contained in U2 contained in U3, testing whether the measured boundary is stable as the planner gets stronger. Formal boundary estimators for F_U(tau) rather than visual reading. Dimensionless collapse plotted against the task's declared Pi vector, never against Pi_g alone unless the criterion licenses it.

**Precision metrics.** Chosen-point regret against the task oracle. Set overlap (IoU) between the mechanics envelope and the empirical feasible set. Full-map rank correlation between the critic's map and Q*_U.

**Interventional splits.** Independent variation of B and w. Common-scale (cB, cw) pairs. Fixed ell_gb with varied total length. Friction-only variation across supports. Unseen-combination splits where a parameter pair is withheld entirely.

**Real phase, two cohorts.** A **model-validity cohort** (homogeneous wire, tubing, uniform cord) with repeatable effective stiffness validates the scaling claims. An **out-of-model stress cohort** (chain, braided rope, hysteretic cord, pre-curved cable) is evaluated only on selector robustness and regret, never on exponent agreement. The separation is the point: a chain departing from the law is a statement about the law's domain, not a learning failure, and mixing the cohorts makes the two indistinguishable. A specialized visual-property-estimator baseline row is also reserved for this phase.

**Style discipline.** Three phrasings are settled once and carried through the paper and its figures. The claim about the critic is always behavior consistent with the predicted mechanics scaling under held-out interventions, never that the policy rediscovers the physics: the teacher injects the properties, so "rediscovery" invites attack. Li and Choi's flexibility estimate follows a prescribed robot interaction, so no contrast of the form "they only look, we touch" appears. And DLO-Lab's grasp-mode gap is always described as evidence about grasp proposal, never as evidence of material-dependence.

**Title candidates,** choice deferred:

1. "Where to Grasp Depends on Material: Mechanics-Grounded Contact Selection for Deformable Linear Objects"
2. "Mechanics-Grounded Material-Adaptive Grasp Selection for Deformable Linear Objects"

**Open items.** Huang and Au (RA-L 2023) is unread pending institutional access, treated as a to-be-verified citation with nothing load-bearing attached. Hierarchical DLO Routing (arXiv:2510.19268) is cited for its reuse of a predicted grasping index only: any award or finalist status is unconfirmed and claimed nowhere. AssistDLO is an unrefereed preprint and its status is reported wherever it is cited. The regime boundary between beam-dominated and catenary-dominated spanning needs an operational classifier before gap placement can support any law claim. The choice of optimality objective inside the admissible set is open, and the loop should inform it. And the sufficiency of the motion-template family remains the main threat to Step 2's measurement, so escalation to CMA-ES over motions is held in reserve if the templates prove too coarse to locate the boundary.

The gate, sharpened: the boundary of the empirical feasible set moves with mechanics parameters in the predicted direction, or the thesis does not survive its cleanest test.
