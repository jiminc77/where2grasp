# where2grasp: Material-Adaptive Contact Selection for Deformable Linear Objects

**Date:** 2026-08-16
**Status:** v2, post-feedback revision
**Codename:** where2grasp
**Note:** paper title is open; two candidates are recorded in Section 11.

---

## 1. Summary

Where a robot should grasp a rope is a task-conditioned decision. Write it as

    s* = s*(theta, x0, G, U)

with theta the mechanics parameters, x0 the current state of the object, G the task geometry, and U the family of motions the robot is allowed to execute. Nobody disputes that s* depends on all four. The claim of this paper is narrower and testable, and it separates two things that mechanics treats differently: **at fixed task, how the admissible set of grasps moves with theta is derivable from mechanics; how the optimum s* within that set moves is derivable once an objective is fixed; and both are identifiable from interaction.**

The precision matters because the two axes of dependence have opposite epistemic status. The **task axis** is obvious: everyone grasps a rope differently for tying than for threading, so what it needs is *integration* into a working system, which is the sequel's job. The **material axis** is invisible: two ropes can be identical in an image and opposite in behavior, and the field currently designs the dependence away rather than measuring it, so what it needs is *proof*. That is this paper. Section 10 develops the split.

That split dictates the architecture. The **per-point critic carries task-dependence**, instantiated per task. **m_t carries material-dependence**, shared and frozen. Demonstrating frozen transfer of m_t across tasks is what proves the factorization is real rather than notational.

Three contributions follow.

**First, a mechanics-derived feasible boundary and its scaling.** Not an "optimal grasp law". Standard beam and cantilever mechanics bound the set of grasp positions that can succeed, and predict how those bounds move with bending stiffness, linear weight, length, and friction. Picking one point inside the feasible set requires an objective, which is a design choice rather than a consequence of mechanics.

**Second, m_t, a mechanics encoder validated as a module.** It maps interaction history to a latent that is a task-sufficient statistic of the mechanics, probed against measured ground truth, transferred frozen across tasks, and specified with an interface that can be lifted into a larger system.

**Third, the first consumer: property-adaptive per-point grasp selection.** A critic that scores every candidate point along the object, conditioned on m_t. The headline analysis is whether its selections show behavior consistent with the predicted mechanics scaling under held-out interventions.

One question gates everything, and we answer it before building anything substantial: **on the cleanest diagnostic task, does the feasible band edge move with mechanics parameters in the predicted direction?** Section 9 describes the loop that settles it.

---

## 2. Problem and thesis

Manipulation policies degrade on deformables, and the diagnosis is specific. The variables that govern the contact decision do not render reliably in appearance.

The honest form of that statement matters, because the vision literature has moved. Purpose-built vision-based property estimation works: PhysGS (CVPR 2026, CVF proceedings pp. 18980-18990) demonstrates it. So the claim is not that appearance-only inference is impossible, and not that vision-only policies are structurally unable to decide. The claim is this. Static appearance alone is not generally sufficient to uniquely identify DLO mechanics; visually similar objects can differ substantially in dynamics; generalist VLMs remain weak at property estimation, as PhysGS's own GPT-4V and GPT-5 baselines show (their stiffness comparison appears in the supplement); and while visual priors genuinely help, interaction supplies additional identifying evidence that a static view does not provide. Physion++ (NeurIPS 2023), ContPhy, and PhysBench (ICLR 2025 Oral) document the general-model side of that gap.

Humans behave as though the decision lives where we claim it does: grasp location shifts with an object's deformability (Mazzeo et al., Scientific Reports 2024). That is a measured effect, which makes it a phenomenon to explain rather than a hunch to defend.

Robotics has been routing around it. DLO-Lab's released code (ICML 2026, arXiv:2606.04206) hardcodes grasp vertices in all eight shipped environments, verified at repository HEAD. The multi-stage cable routing work of Luo et al. (T-RO 2024, arXiv:2307.08927) explicitly treats grasp variation as a disturbance and slack as a nuisance. The decision is either fixed by hand or defined as noise.

So the question, in one sentence: **at fixed task and fixed motion family, is the feasible grasp set a predictable function of mechanics parameters, and can a robot identify the relevant parameter combinations by interaction alone?**

---

## 3. Mechanics: the derivation program and the feasible boundary

### Notation

Let L_tot be the total object length, s in [0, L_tot] the grasp coordinate along the object, and ell(s) the free length from the grasp to the task-relevant tip. Let B = EI be the bending stiffness and w = lambda g the weight per unit length. Note what these are: B is a structural property (material times cross-section), lambda a linear mass density, L_tot geometry, and friction an interface property of the object against its environment. The grasp point is not itself a material property, and we do not describe it as one.

The governing dimensionless group is

    Gamma_g = w ell^3 / B = (ell / ell_gb)^3,   with ell_gb = (B / w)^(1/3)

the gravito-bending length (Miller et al., PRL 2014). Gamma_g compares gravitational droop against bending resistance over the free length, and it is the axis along which every constraint below organizes.

### The constraints

**Sag.** A task that requires clearance imposes a bound on tip droop. For a cantilevered free length under self weight, delta_tip = w ell^4 / (8B). The exponent of the resulting bound depends on the success criterion, and each task must state which criterion it uses before quoting a scaling.

- Absolute clearance, delta_tip <= h, gives ell_sag proportional to (B h / w)^(1/4), a fourth-root scaling.
- Dimensionless droop, delta_tip / ell <= epsilon, gives ell_sag proportional to (epsilon B / w)^(1/3), a cube-root scaling and a direct statement about ell_gb.

This is a feature of the program rather than a defect. Different tasks impose different criteria and therefore different exponents, and measuring which exponent a task actually exhibits is a sharper test than asserting one exponent everywhere.

**Buckling.** For tasks that drive the object against a constraint with load P_task, the free length must stay below

    ell_buckle = (pi / K) sqrt(B / P_task)

with K the effective-length factor for the boundary condition at the tip.

**Geometry.** Task geometry imposes a lower bound ell_min from required overhang, insertion depth, required extension to the goal, and collision avoidance.

**Feasible interval.**

    ell in [ell_min_geometry, min{ell_sag, ell_buckle}]

Workspace kinematics sit on top of this rather than inside it: reachability and joint limits can truncate the interval from either side depending on where the supports and the object sit relative to the arm, and that truncation is reported per task rather than folded into the mechanics.

**A correction worth stating plainly.** Cantilever stiffness k = 3B / ell^3 and the buckling load P_cr both *decrease* with ell. They therefore bound the free length from the *same* side, and it is wrong to present them as squeezing the band from opposite ends. The opposing bound comes from task geometry. Getting this right is what turns a rhetorical band into a derivation.

**Friction.** Friction enters as an inequality once the support contact geometry is fixed, and it takes one of two forms.

On a flat support, the Coulomb condition bounds the tangential load T that the contact can resist by the normal load N it carries, with mu the friction coefficient at that interface:

    T <= mu N

The overhang-holding form follows directly. Let ell_sup be the length of object resting on the support and ell_hang the length hanging beyond it. Then N = w ell_sup, the tangential load pulling the object off the support is T = w ell_hang, and the contact holds while

    w ell_hang <= mu w ell_sup,   that is,   ell_hang <= mu ell_sup

so the admissible overhang is set by mu alone, independent of w. This is the inequality that decides whether a placement holds or slides, and it is why an object can be secured by leaving more length on the support.

Where the object wraps a curved support, the capstan relation replaces it. Let beta be the total wrap angle subtended by the object over the support surface, T_hold the smaller tension applied on the holding side, and T_load the larger tension being resisted. The contact holds while

    T_load <= T_hold exp(mu beta)

Here the wrap angle is the design lever, and its exponential effect is why draping over a peg tolerates load ratios that a flat contact cannot.

### Optimality and control authority

A unique optimum does not fall out of the feasible interval. Choosing one point inside it requires an explicit objective: clearance margin, robustness margin to parameter error, expected success under the motion family, or effort. We state the objective as a design choice and report which one the critic is trained against.

"Control authority" is used across this literature as though it were defined. DLO-Lab uses the phrase without ever defining or measuring it. We do not claim the stiffness decay *is* control authority. We define the measured quantity as **action sensitivity**

    A(s) = | d y_task / d u |

the derivative of the task-relevant output with respect to the commanded action at grasp s, with k(ell) = 3B / ell^3 as its mechanics anchor and proxy. A(s) is measurable in simulation and on hardware; k(ell) explains its shape.

### Against the fitted alternatives

Berenson (IROS 2013) captured decay of influence with "diminishing rigidity", a hand-tuned scalar with no mechanical derivation. Li and Choi's Visual Baseline (ICRA 2024, arXiv:2410.23428) pushes a scalar flexibility estimate through a square-root function to a grasp index. Neither is the same object as ours: they fit an unexplained monotone map from a flexibility proxy to a position, while mechanics hands us a length scale derived from B and w plus a criterion. Both nonetheless prescribe the same thing, how the grasp location should move as flexibility changes, which is why a fitted curve and a derived boundary can be laid over each other and compared.

---

## 4. Prior art and open ground

### Li and Choi

Li and Choi (ICRA 2024, arXiv:2410.23428) place the grasp particle index i_p in the action space and condition it on estimated flexibility. Their flexibility estimate is not a passive look at the object: it follows a single prescribed robot interaction, grasping with a predefined pose and then applying a GNN over the particle graph of the resulting hanging configuration. They report a with-flexibility versus without-flexibility gap (78% vs 56%), which is itself evidence the property signal carries the decision.

The correct delta is: a single prescribed interaction producing an explicit scalar, versus multi-step task-driven interaction history producing a task-sufficient latent; a scalar index emitted by the policy, versus a per-candidate-point map over the whole object; and a fitted monotone mapping, versus a derived feasible boundary. HACMan (CoRL 2023, arXiv:2305.03942) supplies the evidence that the map matters: removing their actor map costs little (83.5 vs 85.4), while removing contact-location selection costs a great deal. Li and Choi never report the learned flexibility-to-grasp map itself. We make that map the object of study. Their group's Hierarchical DLO Routing (arXiv:2510.19268) reuses a predicted grasping index downstream.

### Wiggle and Go!, closest on program structure

Wiggle and Go! (arXiv:2604.22102) is the closest existing work on **program structure**: a task-agnostic identification module, built once and reused frozen across downstream tasks. It executes a predefined wiggle trajectory for system identification, and that module is genuinely task-agnostic: the same identified representation is reused across three downstream tasks (striking, lobbing, draping) without retraining, with zero-shot real-world execution. That is the structure we are proposing, already demonstrated once.

Our deltas are specific. Their identified parameters are, in their own words, "behavioral descriptors of this ball-joint representation rather than direct measurements of a physical rope", and they state that these parameters do not correspond one to one with measurable qualities of the ropes. We target measured physical quantities and derived dimensionless groups, which is what makes an external ground truth and a scaling prediction possible at all. Their tasks are dynamic open-loop swings; ours are quasi-static contact-rich decisions where the settled configuration is the outcome. And their rope is fixed to the tool, so along-object grasp position is not a decision variable, whereas for us it is the entire decision. Wiggle and Go! also anchors our interaction-sysID baseline row.

### DLO-Lab, in two layers

**What it establishes.** DLO-Lab states that grasp selection is a critical determinant of success, and compares VLM grasp proposal modes (Candidate, Coefficient, Marker). The roughly 19x return gap on Unknotting in its appendix grasp-mode table is a comparison *between VLM proposal modes*. It is therefore evidence that grasp proposal strongly affects return, and not evidence of material-dependence.

**What its code does.** The released repository hardcodes grasp vertices in all eight shipped environments and ships no VLM code at all, verified at HEAD.

DLO-Lab demonstrates that grasp proposal materially affects task return, but does not study how the grasp-quality landscape changes under controlled material-property interventions.

The anticipated objection is that DLO-Lab already asks a VLM for a grasp coefficient in [0,1] along the object, so grasp selection is not new. Correct, and our claim is narrower: we contribute the first setting where that coefficient's admissible set is physically derivable and demonstrably moves with material.

### AdaptiGraph, RAPiD, and the rest

**AdaptiGraph** (arXiv:2407.07889) is the closest existing work on **adaptation mechanism**: interaction-history property inference feeding a planner. Concretely, material-conditioned GNN dynamics, test-time property inference by optimization, MPPI planning, multiple rope varieties. Same adaptation signal, no critic. It belongs in related work and in the full-system benchmark.

**RAPiD** (arXiv:2603.18246) applies RMA to deformables and evaluates both a 1D insertion task and a 2D covering task, so it is not a 1D-only method. It does not isolate along-object grasp-index selection, and for the covering task it reports no explicit randomization over pose, stiffness, or friction, inducing those variations through object-instance randomization instead. It also reports no oracle bound.

**Strong non-learned selectors are a live risk.** The ICRA 2024 Cloth Competition report (IJRR, arXiv:2508.16749, doi 10.1177/02783649251414885) records non-learning methods placing 1st and 3rd out of 11 teams. Beating a weak midpoint heuristic proves nothing.

**Foresightful Dense Affordance** (ICCV 2023, arXiv:2303.11057) already produces per-point value maps for ropes and fabrics, property-blind. We do not claim per-point maps as architecture. **GenDOM** (ICRA 2024, arXiv:2309.09051) anchors explicit parameter estimation plus conditioning. **Kamaras and Ramamoorthy** (arXiv:2502.18615) use BayesSim posteriors to shape a domain randomization distribution for PPO, never conditioning the policy on the parameters. **Navarro-Alarcon and Liu** (T-RO 2018) are the classical ancestor of the interaction buffer. **WireCraft** (arXiv:2606.18097) reports that vision and VLA pipelines bottleneck at contact-rich DLO tasks where privileged-state RL succeeds, and makes imitation rows table stakes.

### Open ground

Across the benchmarks, surveys, and primary sources we checked, four cells are open. The quantitative property-to-grasp-boundary map is unpublished. The material-conditioning by per-point-critic cell is empty. A mechanics-derived boundary with a stated criterion has not replaced a fitted monotone scalar. And no work varies mechanics parameters under controlled intervention and reports how the grasp-quality landscape responds.

---

## 5. The paper we intend to write

**§1 Introduction, diagnosis first.** The contact decision depends on mechanics parameters that static appearance does not generally determine. Humans shift grasp with deformability (Mazzeo et al., 2024), while benchmarks hardcode the grasp point or treat its variation as a disturbance (DLO-Lab; Luo et al., T-RO 2024). State the task-axis versus material-axis split, then the one-sentence question from Section 2.

**§2 The mechanics of where to grasp.** The derivation program of Section 3, through to the explicit statement that optimality needs an objective.

**§3 Grasp landscapes, Figure 1.** Define the object of measurement formally:

    Q*_U(s; theta, G, x0) = max over u in U of J(s, u; theta, G, x0)

This is a **planner landscape relative to the motion family U**, not a property of the object. Three consequences are stated rather than buried: an insufficient template family misplaces the peaks, enlarging U can change the landscape, and the peaks depend on task geometry as well as on material. With that caveat carried openly, sweep candidate grasp points against controlled parameter settings and show whether the feasible band edge moves as §2 predicts, with existing selectors' picks plotted on the same landscape.

**§4 Method.** m_t and the per-point critic, with the factorization argument. Lineage stated honestly: per-point critic from HACMan, teacher-student recipe from RMA (Kumar et al., RSS 2021), interaction buffer from Navarro-Alarcon and Liu, task-agnostic identification module from Wiggle and Go!.

**§5 Experiments.** The two benchmarks and two ceilings of Section 8, the scaling-consistency figure, and the real-robot protocol.

**§6 Limitations and what we do not claim.** We are not discovering that grasp depends on material (Mazzeo; Li and Choi). We are not introducing per-point value maps (Foresightful Dense Affordance). We are not the first task-agnostic identification module (Wiggle and Go!). We do not claim a good DLO dynamics model: that ground is AdaptiGraph's and the sequel's.

---

## 6. Task design

### Primary analytic diagnostics

**Lift-and-clear** and **distal tip placement** are the primary tasks, because the derivation of Section 3 is directly measurable in them. A single grasp, a single primitive, a settled outcome, and a clearance or placement criterion that maps onto delta_tip without intermediate composition. The go/no-go of Section 9 runs here, on the cleanest available instrument.

### Gap placement, demoted to downstream demonstration

Two rigid supports with a gap, a failure plane below the support tops, the object starting on the table, and success requiring both ends supported with mid-span sag above the plane. This is a downstream demonstration rather than a primary diagnostic, and the reason is mechanical.

Gap placement composes transport, contact, friction, and settling, and critically **the grasp point does not persist in the settled equilibrium**. A flexible spanning rope settles into a tension and catenary dominated shape, a stiff one into a beam dominated shape, and in neither case does the final configuration remember where the gripper was. The grasp point acts *indirectly*, through the landing configuration it produces, the overhang length it leaves, and the friction-locked metastable states it makes reachable. That is a real effect and a poor instrument for a scaling law. If gap placement is ever used for a law claim, the regime (beam versus catenary) must be classified first.

Its lineage and grounding stay, with this honest role. The nearest parent is the Draping task of Wiggle and Go!, rope thrown over a wall with heavy material variation but rigidly attached to the tool. A secondary lineage treats a spanned cable as a structure: the ETH aerial rope bridge (Augugliaro et al., Flying Machine Arena), and the catenary robot (RA-L, arXiv:2102.12519), which makes span and sag explicit control degrees of freedom with both cable ends held by actively controlled aerial robots, so again there is no along-object grasp choice. The Berkeley cable routing work is the clean inversion of our framing. Industrially, transmission line design evaluates ground clearance at final sag under worst-case conditions (USDA RUS Bulletin 1724E-200), and cable tray standards treat support span as the design lever against sagging (NEMA VE 2). The task remains unclaimed: DLO-Lab, DaXBench, SoftGym, DEDO, MoDeSuite, and WireCraft were checked, and the IJRR 2026 survey contains zero occurrences of "sag", "drape", "clearance", or "catenary", with no gap-spanning family in its taxonomy.

### Staging protocol

Complexity is admitted in stages, and each stage is allowed to support only the claims it can carry.

1. **Single grasp, fixed primitive, single release.** Law validation happens only here.
2. **Multi-step without regrasp.** Adaptation and history use, still with a persistent contact.
3. **Regrasp-enabled full task.** System-level results only.

### Remaining supporting tasks

**Hook and peg draping** opens the friction axis with a wrap geometry where the capstan relation of Section 3 is the binding constraint, and where the wrap angle beta is directly controllable by the placement. **Tip insertion** is the real-world flagship, the direct contrast with Li and Choi, and the task where the buckling bound binds.

---

## 7. Method: m_t and the per-point critic

The interface is fixed now; the internals are decided by the loop.

### Interface

m_t consumes a sequence of tuples

    (x_t, a_t, x_{t+1}, c_t)

where c_t is grasp and contact metadata: initial shape, grasp index, contact state, and the settled trajectory. Action and settled deformation alone would be too thin. The grasp index in particular is part of the evidence, since the same motion applied at a different s produces a different response from the same object.

### What m_t can and cannot identify

Static self-weight deformation senses the **ratio** B / w, equivalently ell_gb, not B and w separately. Saying so is not a concession. The downstream decision depends on task-sufficient combinations, principally ell_gb and friction, so recovering the combination is recovering exactly what the decision needs. We therefore frame m_t as learning a **task-sufficient statistic** of the mechanics, not a disentangled parameter vector, and we design the property grid to test that framing directly (Section 8).

Friction is nearly unobservable without slip. The interaction repertoire therefore includes a **slip-inducing probe primitive**, so the history contains evidence about the interface property and not only about the bulk.

### Why a latent rather than a point estimate

RMA (Kumar et al., RSS 2021) and HORA (Qi et al.) both report explicit identification underperforming latent teacher-student adaptation. The reason we give is not that the latent preserves posterior information, which overstates what a deterministic encoder does. It is that the latent is a task-sufficient statistic that is **not forced into a premature point-estimate collapse**: the pipeline never has to commit to a single parameter vector at a stage where the evidence does not support one. Whether the encoder should additionally be stochastic or distributional is a loop-decided option, not an assumption. Kamaras and Ramamoorthy (arXiv:2502.18615) motivate the concern, keeping BayesSim posteriors to shape randomization rather than conditioning on an estimate. We carry explicit sysID as a baseline rather than assuming the published result reproduces here.

Training is two-phase teacher-student: the teacher receives true mechanics parameters, the student reproduces the conditioning latent from interaction history alone.

### Module-level validation

Three claims, each winnable or losable on its own. The latent is **probed against measured ground truth**, in simulation and on real ropes. It is **transferred frozen across tasks**, trained on one task's interactions and reused without retraining on another, which is the evidence that the material axis really factors out of the task axis. And its **interface is liftable**, specified tightly enough to drop into a larger system unchanged.

Real properties are measured with a Peirce-inspired cantilever test. We are careful about what that standard covers: ASTM D1388 (current edition D1388-23) is a fabric stiffness standard for woven and knit fabrics, and ropes and cords fall outside its scope by omission. So we describe our protocol as Peirce-inspired, validate its conversion against an independent bending-stiffness measurement, and report an effective stiffness B_eff that acknowledges tension dependence, internal strand sliding, and hysteresis.

### The per-point critic

The critic scores every candidate grasp point along the object, conditioned on m_t's latent. FiLM is the default conditioning, motivated by DyWA, and we treat that as a **motivated default from a rigid-object domain whose advantage is re-tested here, not assumed to transfer**. The output is a map, not an index.

The factorization is the architectural claim: the critic is per-task and carries G and U; m_t is shared and carries theta. Frozen cross-task reuse is what turns that from notation into a result.

---

## 8. Evaluation design

Two benchmarks, deliberately separated, because mixing them is how selector claims get contaminated by planner budget.

### Selector-only benchmark

Identical candidate set, identical fixed motion family, identical interaction history for every row. Only the grasp-index choice differs. Rows: random selection; tuned best-of geometric heuristics reported as one row; a VLM proposer; property-blind per-point critic (ours minus adaptation, essentially HACMan); scalar-flexibility monotone-map selector in the style of Li and Choi's Visual Baseline, labeled as their baseline and not their method; and ours.

The VLM proposer needs a caveat, since this document has already established that DLO-Lab ships no VLM code. There is no reference implementation to adopt, so the row is reimplemented from the proposal modes described in the DLO-Lab paper, Candidate and Coefficient. Fidelity of that reimplementation is our responsibility and is reported as such, including prompt, model, and the mapping from model output to grasp index.

### Full-system benchmark

Methods with different action spaces and budgets, compared with those differences reported rather than hidden. CMA-ES over grasp and motion, the strongest method overall in DLO-Lab, winning 7 of 8 tasks and losing clearly on one. AdaptiGraph-style material-conditioned dynamics with planning. An imitation row, since WireCraft made ACT, Diffusion Policy, and pi-0.5 standard. Explicit sysID pipelines, including a Wiggle-and-Go-style interaction sysID baseline. Every row reports **rollout counts, compute, and privileged access**.

### Two ceilings, clearly distinguished

- **Adaptation ceiling:** a policy given ground-truth parameters. It bounds what adaptation could recover, but it is not a true oracle, because learning can fail for reasons unrelated to property knowledge.
- **Task oracle:** the exhaustive best grasp under the same motion family, argmax over s of Q*_U(s). This is the real upper bound on the decision itself.

RAPiD reports no such bound, so carrying both is also a differentiator.

### Property grid design

Vary B and w **independently**, and deliberately include pairs with **fixed B / w**. That pairing is the shortcut control: if m_t is genuinely recovering ell_gb, points sharing B / w should collapse; if it is exploiting some incidental cue, they will not. This is the cheapest available test of the task-sufficient-statistic framing.

### Real-robot protocol sketch

A rope set with measured properties, using Wiggle and Go!'s published set as the template since it spans light twine through steel chain and therefore covers the stiffness-to-weight axis rather than clustering. Properties measured with the Peirce-inspired protocol so simulated and real objects are indexed by the same quantity. Evaluation is held-out-parameter zero-shot on a Franka. The reportable outcome of the scaling analysis is behavior consistent with the predicted mechanics scaling under held-out interventions.

---

## 9. The fastest validation loop

The immediate plan, free of targets by design.

**Step 1: minimal diagnostic build.** A lift-and-clear or distal-tip-placement scene on the existing simulator stack, using the parameter variation hooks already in place. Gap placement is not built here.

**Step 2: mini landscape.** A small candidate set crossed with a small parameter grid and a small motion-template family, in parallel environments. One question: **does the feasible band edge move with parameters in the predicted direction?**

The scope of a negative is stated honestly. If the band does not move for this task and this action family, the task is unsuitable as the primary diagnostic. Failing on the cleanest instrument we can build is strong evidence against the thesis, and it would send us back to rethink rather than to patch, but it does not falsify the thesis for all DLO tasks.

**Step 3, if the edge moves: identifiability.** Reuse the Step 2 rollouts, no new collection, plus the friction probe. Test whether the task-sufficient combinations, principally ell_gb and the friction term, are readable from short interaction histories better than from proprioception alone. This is m_t's first validation rather than a throwaway check.

**Step 4, if readable: miniature teacher-student.** On the single diagnostic task, conditioned against blind, with the three mechanism candidates side by side: history latent, explicit sysID, and the one-step deformation auxiliary. Directional outcomes only; this chooses the architecture.

**After the loop:** decide, then expand. Gap placement is built in post-loop expansion, alongside task expansion, the real-robot protocol, statistics, and ablations.

**Hardware.** Pilot runs on an RTX PRO 6000, with a 4090 also available. Two Franka Pandas and mountable cameras are available for the real phase. A larger multi-GPU machine exists but is not set up, and nothing here depends on it.

---

## 10. Relationship to DLO-JEPA

DLO-JEPA (owner-internal architecture document, 2026-08-11) is a VLA with explicit DLO state, a latent mechanics encoder E_H over interaction history, a frozen visual prior, and an action-conditioned predictor. It is the sequel system paper.

The two-axis argument is what makes the split between the papers principled rather than convenient. The **task axis needs integration, not proof**: that grasp choice depends on the task is uncontroversial, and the work there is building a system that handles many tasks coherently, which is precisely a system paper's job. The **material axis needs proof, not integration**: it is invisible, currently designed away, and provable only under controlled intervention against an external mechanics ground truth, which a system paper cannot deliver because every component's contribution is entangled with every other.

So this proposal builds and validates E_H under conditions that isolate it. m_t *is* E_H. The interface contract is fixed here and inherited there: a history of (x_t, a_t, x_{t+1}, c_t) tuples in, a conditioning latent out, with frozen reuse without retraining as a stated deliverable of paper one rather than an aspiration of paper two. A validated m_t gives the sequel a component it can hold fixed while arguing about everything else.

---

## 11. Deferred hardening and open items

Held until after the loop: gap-placement construction, task expansion, the full real-robot protocol, statistical design, and appendix ablations (one-step deformation auxiliary, FiLM against concatenation, history encoder form).

**Precision metrics reserved for hardening.** Chosen-point regret against the task oracle. Feasible-band IoU between predicted and measured admissible sets. Full-map rank correlation between the critic's map and Q*_U. Fitted scaling exponent with a confidence interval, compared against the criterion-specific prediction of Section 3. Dimensionless collapse of landscapes plotted against Gamma_g.

**Interventional splits reserved for hardening.** Independent variation of B and w. Fixed ell_gb with varied total length. Friction-only variation. Unseen-combination splits where a parameter pair is withheld entirely.

**Style discipline.** Three phrasings are settled once here and carried consistently through the paper and its figures. The claim about the critic is always behavior consistent with the predicted mechanics scaling under held-out interventions, never that the policy rediscovers the physics: the teacher injects the properties, so "rediscovery" is an open invitation to attack. Li and Choi's flexibility estimate follows a prescribed robot interaction, so no contrast of the form "they only look, we touch" appears anywhere. And DLO-Lab's grasp-mode gap is always described as evidence about grasp proposal, never as evidence of material-dependence.

**Title candidates.** Two are on the table and the choice is deferred:

1. "Where to Grasp Depends on Material: Mechanics-Grounded Contact Selection for Deformable Linear Objects"
2. "Mechanics-Grounded Material-Adaptive Grasp Selection for Deformable Linear Objects"

**Open items.** Huang and Au (RA-L 2023) is unread pending institutional access and is treated as a to-be-verified citation with nothing load-bearing attached to it. Hierarchical DLO Routing (arXiv:2510.19268) is cited for its reuse of a predicted grasping index only: any award or finalist status is unconfirmed and is claimed nowhere in this document or the eventual paper. The regime boundary between beam-dominated and catenary-dominated spanning needs an operational classifier before gap placement can support any law claim. The choice of optimality objective inside the feasible interval is open, and the loop should inform it. And the sufficiency of the motion-template family remains the main threat to Step 2's measurement, so escalation to CMA-ES over motions is held in reserve if the templates prove too coarse to locate the band edge.

The gate is unchanged in spirit and sharpened in wording: the feasible band edge moves with mechanics parameters in the predicted direction, or the thesis does not survive its cleanest test.
