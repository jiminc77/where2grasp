# GPT-5.6 Sol (Effort: Pro) — where2grasp inspection reply

- Conversation URL: https://chatgpt.com/c/6a826a50-131c-83e8-b0ac-5300fb999589
- Retrieved: 2026-08-17T03:04:08.349Z
- Model/effort verified: composer pill "Pro" pre-send; "Pro thinking" indicator post-send
- Extraction: backend-api markdown (4 part(s))
- Length: 32567 chars

---

I’ll audit the repository in the specified order, cross-check derivations, preregistration, code, results, and QA, then independently re-check 2024–2026 primary literature for novelty.

Early audit suggests strong self-correction and labeling discipline. Key risks are B_eff’s simulator-specific calibration, mechanics-gate non-independence, and teacher/student evidence remaining limited to map recovery.

A sharper teacher/student flaw emerged: its bootstrap CI effectively resamples identical deterministic landscapes, so “CI95 < 0” does not quantify uncertainty. Novelty search has found no direct counterexample yet.

## 검토 범위와 총평

검토 기준은 **2026년 8월 17일 public repository 상태**임. `hardening-B`는 현재 `plan/hardening_directive_B.md`만 공개되어 있고 code, data, frozen manifest, result는 아직 없음. 따라서 hardening-B 관련 평가는 **design/pre-registration audit**이며 결과 검증은 아님. citeturn124037view5

핵심 판정은 다음과 같음.

- Euler–Bernoulli 기반 \(\delta_{\text{tip}}=w\ell^4/(8B)\), \(\ell_{\max}=(8Bh/w)^{1/4}\) 유도 자체는 명시된 small-deflection cantilever 가정 아래 맞음.
- 현재 frozen Q1의 **formal GO는 해당 pre-registered rule대로 유효**함. 이를 뒤집을 오류는 찾지 못함.
- 다만 가장 인상적인 수치인 `0.5189 vs 0.5180`은 독립적 mechanics closure라기보다 **동일 sag law로 수행한 \(B_{\mathrm{eff}}\) calibration의 내부 closure** 성격이 강함.
- \(m_t\) 결과는 **task-functional map conditioning의 강한 신호**이나, 아직 sequential adaptation, physical property identification, training uncertainty까지 확립한 결과는 아님.
- 좁혀진 novelty sentence는 현재 검색 범위에서 유지 가능함. 단, 바로 다음 문장의 broad claim은 축소 필요.

---

# 1. Mechanics 및 experimental-design audit

### 1. **[BLOCKER] `plan/hardening_directive_B.md` §Distal-tip placement — interior optimum이 아직 mechanics로 보장되지 않음**

Directive는 reach lower bound와 sag upper bound가 함께 존재하므로 tip-error objective가 interior, material-dependent optimum을 가진다고 전제함. 그러나 **feasible interval의 존재만으로 objective의 unique interior optimum은 따라오지 않음**. Proposal 본문도 admissible set과 optimum을 구분하고 있음. citeturn124037view0turn124037view3

예를 들어 small-deflection 근사에서

\[
x_{\text{tip}}(\ell)\simeq \ell,\qquad
z_{\text{tip}}(\ell)\simeq-a\ell^4,\qquad
a=\frac{w}{8B}
\]

이고 target이 \((d,z^\star)\)라면,

\[
E(\ell)=(\ell-d)^2+(-a\ell^4-z^\star)^2
\]

임. \(z^\star=0\)이면 strict convexity와 unique minimizer를 비교적 쉽게 보일 수 있지만, 일반 \(z^\star\), workspace clipping, orientation penalty, binary tolerance가 들어가면 endpoint optimum, multiple stationary points, flat success plateau가 가능함.

**Freeze 전 필수사항**

1. continuous primary objective를 수식으로 고정  
2. admissible domain에서 existence/uniqueness 또는 \(\epsilon\)-optimal set 정의  
3. material intervention에 따른 predicted optimum shift가 최소 2 grid cells 이상인지 사전 계산  
4. grid tie, clipped optimum, censored optimum 처리 규칙 고정  

이를 충족하지 않으면 hardening-B가 “material-dependent optimum을 발견”한 것이 아니라 **그런 optimum이 나오도록 target/grid를 조정한 결과**로 공격받기 쉬움.

---

### 2. **[BLOCKER] `sim/analyze_gate.py::boundary()` vs `sim/manifests/hard_s34_manifest.json` critic metric — boundary 정의가 서로 다르며 hardening-B에서 충돌함**

현재 hard gate의 `boundary()`는 모든 \(\tau\)-crossing을 찾은 뒤 **maximum crossing**을 반환함. 반면 critic/map evaluation은 **first crossing** 계열 정의를 사용함. 현재 lift task의 monotone \(1\rightarrow0\) map에서는 둘이 같으므로 Q1 결과에는 영향 없음. citeturn657813view0turn372011view15

하지만 hardening-B가 의도한 success landscape는 reach 때문에 짧은 \(\ell\)에서 실패하고, 중간에서 성공하며, sag 때문에 긴 \(\ell\)에서 다시 실패하는 \(0\rightarrow1\rightarrow0\) 형태임.

- first crossing: reach-limited lower boundary
- maximum/last crossing: sag-limited upper boundary

따라서 “boundary recovery”를 scalar 하나로 유지하면 teacher와 student가 서로 다른 물리적 경계를 잘 맞추면서도 동일 metric으로 평가될 수 있음.

**권고:** hardening-B manifest에 아래를 별도 co-primary로 freeze해야 함.

\[
\ell_L=\text{lower crossing},\qquad
\ell_U=\text{upper crossing}
\]

그리고 interval IoU 또는 Hausdorff error, continuous objective argmin error, selection regret를 함께 사용. 기존 scalar boundary evaluator 재사용은 금지하는 편이 안전함.

---

### 3. **[MAJOR] `sim/analyze_gate.py` §Pi guard / `hard_gate_verdict.json` B1_w2 — \(\Pi_g\le0.5\) guard가 grid uncertainty에 대해 robust하지 않음**

Frozen rule은 interpolated point boundary에서만

\[
\Pi_g=\frac{w\ell^3}{B_{\mathrm{eff}}}
\]

를 계산함. `B1_w2`는 boundary \(0.195\)에서 \(\Pi_g=0.4193\)이므로 literal frozen rule을 통과함. citeturn657813view2turn557131view1

그러나 이 boundary는 grid bracket \([0.18,0.21]\)의 midpoint임. 같은 \(B_{\mathrm{eff}}=0.0161052\), \(w=0.910680\)에서 upper bracket endpoint를 사용하면

\[
\Pi_g(0.21)\simeq0.5236>0.5
\]

임. 즉 다음 두 문장은 구분되어야 함.

- **맞는 문장:** “15개 point-estimated boundaries가 모두 \(\Pi_g\le0.5\)”
- **아직 맞지 않는 문장:** “15개 boundary uncertainty intervals 전체가 small-deflection regime 안에 있음”

현재 formal GO는 manifest의 point-estimate rule 아래 유지됨. 다만 후속 gate에서는 `valid iff Pi_g(upper bracket)<=0.5`처럼 conservative guard를 사용하거나, bracket 전체를 in-regime로 요구해야 함.

---

### 4. **[MAJOR] `sim/calibrate_beff.py` §Physics/method 및 lift boundary test — \(B_{\mathrm{eff}}\) calibration과 validation이 동일 law를 공유함**

\(B_{\mathrm{eff}}\)는 self-weight cantilever sag를

\[
\delta=\frac{w\ell^4}{8B_{\mathrm{eff}}}
\]

에 맞춰 추정함. 이후 lift task에서도 같은 gravity cantilever, 같은 clamp geometry, 같은 solver/discretization을 사용하고, \(\delta=h\)를 대입해 boundary를 예측함. citeturn319879view3turn124037view6

이는 잘못된 calibration은 아니지만 evidence category가 중요함. `0.5189 vs 0.5180`은 주로 다음을 확인함.

- sag calibration과 boundary extraction pipeline의 내부 일관성
- \(B/w\) variation에 대한 scaling consistency
- grid와 label implementation의 closure

반면 다음을 독립적으로 확인하지는 못함.

- simulator의 bending knob가 실제 continuum \(EI\)를 구현하는지
- fourth-root prefactor가 calibration law와 독립적으로 맞는지
- 다른 loading mode에서도 동일 \(B_{\mathrm{eff}}\)가 유지되는지

가장 강한 보완은 **known tip force 또는 imposed moment/curvature로 \(B_{\mathrm{eff}}\)를 독립 calibration**한 뒤, gravity boundary를 no-refit prediction하는 것임.

---

### 5. **[MAJOR] `sim/material.py::apply_properties()` 및 `sim/calibrate_beff.py::_nv()` — per-vertex mass lumping이 \(\ell\)-dependent quadrature bias를 만듦**

Code는 free arm \(N\) segments에 대해 `n_vertices=N+2`를 사용하고 vertices 0, 1을 clamp함. 각 free vertex에는 동일한 full `segment_mass`가 들어감. 따라서 continuum uniform load는 사실상 각 segment의 right endpoint에 배치된 \(N\)개 point load로 적분됨. citeturn319879view0turn319879view3

Linear Euler–Bernoulli point-load analogue에서 이 discretization의 tip sag와 continuum UDL sag 비율은

\[
r_N
=\frac{\delta_{\text{lumped}}}{\delta_{\text{UDL}}}
=\frac{(N+1)(3N+1)}{3N^2}
=1+\frac{4}{3N}+\frac{1}{3N^2}.
\]

따라서 예상 bias는 다음과 같음.

- \(N=12\): \(+11.34\%\)
- \(N=18\): \(+7.51\%\)
- \(N=24\): \(+5.61\%\)
- \(N=60\): \(+2.23\%\)

더 중요한 점은 calibration 구간 \(N=18,20,22,24\)에서 이 quadrature factor만 포함해도 apparent log-log exponent가 약 **3.94**가 된다는 점임. 실제 보고값 3.83–3.93과 상당히 가까움. citeturn470580view0

이는 Genesis의 모든 nonlinear/detail effect를 설명한다는 뜻은 아니지만, 현재 “approximately fourth-order physics”의 일부가 **known mesh quadrature artifact와 구분되지 않은 상태**임.

권고안은 endpoint half-mass/trapezoidal mass lumping, element-consistent gravity integration, 또는 최소한 analytical \(r_N\) correction과 mesh study임.

---

### 6. **[MAJOR] `sim/calibrate_beff.py` lines 10–16 및 manifest `discretization_note` — fixed interval consistency를 convergence라고 부르면 안 됨**

Repository는 raw bending parameter가 interval-dependent임을 솔직히 명시하고 cross-interval \(B_{\mathrm{eff}}\) equality를 주장하지 않음. 다만 fixed interval에서 \(\ell^4\) fit과 load invariance가 좋다는 사실을 “meaningful convergence evidence”로 표현함. 이것은 엄밀히는 **consistency at one discretization**이지 numerical convergence가 아님. citeturn319879view3turn470580view0

Convergence claim에는 최소한 \(\Delta s=0.02,0.01,0.005\) 등의 mesh sequence가 필요함. Raw \(E\rightarrow B_{\mathrm{eff}}\) map은 mesh-dependent여도 괜찮지만, 다음 observables는 calibrated continuum limit로 수렴해야 함.

- normalized sag curve
- boundary location
- hardening-B optimum
- common-scale invariance error
- teacher/student measured target map

현재 상태에서는 “fixed-discretization validation”으로 명명하는 편이 정확함.

---

### 7. **[MAJOR] `sim/calibrate_beff.py` lines 123–137 — multi-mass acceptance가 light-mass fit만 regime-check하는 loophole 존재**

`accepted` condition은 `f = fits[m0]`로 지정된 light mass의 CV, residual, exponent, worst \(\Pi_g\)만 검사하고, second mass에는 \(B_{\mathrm{eff}}\) difference만 검사함. citeturn319879view3

그 결과 softest material의 heavy-mass points는

\[
\Pi_g=[0.326,\;0.447,\;0.596,\;0.773]
\]

로 마지막 두 점이 guard를 넘지만 material 전체가 `accepted=true`, 최종 `all_accepted=true`가 됨. citeturn470580view0

따라서 현재 확립된 것은 다음 정도임.

- light-load calibration은 declared regime에서 양호함
- heavy-load \(B_{\mathrm{eff}}\) estimate가 전체 fit 기준으로 크게 변하지 않음

하지만 “두 mass 모두 small-deflection regime에서 load invariance를 통과”한 것은 아님. Multi-mass invariance는 양 mass가 동시에 guard 안에 있는 shared lengths만 사용해 재계산하거나, softest material의 second mass를 낮춰야 함.

---

### 8. **[MAJOR] `calibration.json` vs `hard_sweep_manifest.json` — calibration domain보다 task domain이 훨씬 넓음**

Calibration은 \(\ell=0.18\)–\(0.24\), mass \(0.0002/0.0004\)에서 수행됨. Formal gate는 \(\ell=0.12\)–\(0.60\), mass 최대 \(0.0009283\)까지 사용함. citeturn470580view0turn657813view6turn557131view3

같은 interval을 사용했다는 것만으로 이 extrapolation이 안전해지지는 않음. 특히 finding 5의 mass quadrature bias는 \(N\)에 따라 달라지므로 calibration이 흡수한 bias가 \(N=12\) 또는 \(N=60\)에서 동일하지 않음.

후속 calibration은 다음 중 하나가 필요함.

- task-relevant \(\ell\) 전 범위에서 independent-load calibration
- 여러 \(\ell\) 구간을 held-out으로 두는 cross-validation
- mesh-corrected \(B_{\mathrm{eff}}(\Delta s)\)와 no-refit task prediction

---

### 9. **[MAJOR] `sim/analyze_gate.py::boundary()` — mechanics root와 statistical boundary estimand가 혼합되어 있음**

Mechanics prediction은 continuous margin

\[
J(\ell)=h-\delta_{\text{tip}}(\ell)
\]

의 \(J=0\) root임. 하지만 gate는 binary success를 평균한 뒤 success rate의 \(\tau=0.5\) crossing을 boundary로 사용함. citeturn657813view0turn657813view2

현재 settled task에서는 seed/template별 success가 사실상 sharp 0/1이고 bootstrap도 완전히 degenerate함. 따라서 \(\tau=0.5\)는 확률적 50% success boundary가 아니라, 단순히 마지막 success grid point와 첫 failure grid point의 midpoint임. citeturn557131view0turn124037view5

Half-grid-step bound \(\pm0.015\) 자체는 다음 조건 아래 올바른 deterministic bracket bound임.

- continuous \(J\)
- 정확한 binary classification
- 하나의 monotone crossing
- crossing이 grid 내부에 존재

그러나 statistical CI는 아니며, nonmonotone landscape나 noisy deployment success에는 자동으로 적용되지 않음. Mechanics validation에는 `mean_J` 또는 per-cell continuous \(J\)를 직접 interpolation해 \(J=0\) root를 쓰고, stochastic deployment metric이 필요할 때만 \(P(\text{success})=0.5\)를 별도로 정의하는 편이 좋음.

---

### 10. **[MAJOR] `STATUS.md` §Prefactor with uncertainty — “0.2% match”는 resolved precision보다 강한 표현임**

Point estimates는 실제로

\[
0.518850\quad\text{vs}\quad0.518004
\]

이므로 산술적 차이는 약 \(0.16\%\)임. 그러나 같은 verdict가 보고하는 grid-resolution bound는 \(\pm0.02633\), 즉 observed prefactor의 약 \(5.1\%\)임. Bootstrap CI가 zero-width인 것은 반복 precision이 높아서가 아니라 deterministic settled maps가 evaluation-seed resampling에 반응하지 않기 때문임. citeturn557131view0

따라서 가장 정확한 문구는 다음임.

> “Observed prefactor is consistent with the predicted value within the declared grid resolution; its point estimate differs by 0.16%.”

추가로 현재 prefactor \((8h)^{1/4}\)는 **dimensionful이고 \(h=0.009\)에 종속적**임. 비교를 dimensionless하게 쓰려면

\[
K=\frac{\ell_{\max}}{(Bh/w)^{1/4}},\qquad K_{\text{theory}}=8^{1/4}
\]

로 보고하는 편이 좋음. 또한 현재 실험은 \(h\)가 고정되어 있으므로 엄밀히는 **fixed-\(h\)에서 \(B/w\)의 fourth-root behavior**를 확인한 것임. \(h^{1/4}\) dependence까지 직접 측정한 것은 아님.

---

### 11. **[MINOR] `sim/calibrate_beff.py::fit_material()` — `max_slope=delta/ell`은 tip slope가 아님**

Code의 `max_slope`는 \(\delta_{\text{tip}}/\ell\), 즉 secant droop ratio임. Uniform-load Euler–Bernoulli cantilever의 실제 root-to-tip rotation은

\[
\theta_{\text{tip}}=\frac{w\ell^3}{6B}
=\frac{4}{3}\frac{\delta_{\text{tip}}}{\ell}.
\]

따라서 proxy가 tip angle을 약 25% 낮게 표현함. citeturn319879view3

다만 \(\Pi_g=0.5\)에서는

\[
\delta/\ell=\Pi_g/8=0.0625,\qquad
\theta_{\text{tip}}\simeq0.083\text{ rad}\simeq4.8^\circ
\]

이므로 guard의 수치적 크기 자체는 합리적인 conservative heuristic임. 명칭을 `max_droop_ratio`로 바꾸고 measured tangent rotation을 추가하면 충분함.

---

### 12. **[MINOR] `hard_sweep_manifest.json` §decision_rule — adjacent ties를 positive evidence에 포함하는 규칙이 다소 permissive함**

Frozen rule은 adjacent contrasts에서 `correct-signed-or-tied`가 과반이면 PASS 가능하도록 설계되어 있음. Tie는 prediction에 반하지는 않지만 intervention effect의 positive evidence도 아님. citeturn657813view0turn557131view4

현재 데이터는 B 9/9, w 8/8이 모두 strict correct direction이므로 formal verdict에는 영향 없음. 후속 gate에서는 tie를 `INCONCLUSIVE`로 처리하거나 minimum effect size를 별도 요구하는 편이 좋음.

---

### 13. **[MAJOR] `strategy/proposal_v2.md` §Shape-channel degeneracy — theorem scope가 실제 test scope보다 넓게 읽힐 수 있음**

Static equilibrium에서 constant \(B\), constant \(w\), 동일 geometry와 boundary condition, gravity-only load라면 \((B,w)\mapsto(cB,cw)\)가 total energy를 공통 scale하므로 shape가 변하지 않는다는 주장은 맞음. Calibration의 fixed-ratio pairs가 centerline offset 0을 보인 것도 좋은 implementation sanity check임. citeturn470580view0

다만 다음 조건에서는 자동으로 동일하지 않음.

- non-scaled external force 또는 payload
- contact penalty, friction, adhesion
- intrinsic curvature/plasticity
- heterogeneous \(B(s),w(s)\)
- fixed damping이 있는 finite-time dynamics
- force/torque saturation 또는 force-based success criterion

Code에는 fixed damping/angular damping과 fixed \(G\)가 있고, 현재 test는 contact-free planar self-weight equilibrium에 가까움. citeturn319879view1turn319879view3

따라서 claim은 다음처럼 제한하는 것이 안전함.

> “Common scaling is unobservable from quasi-static, gravity-only shape observations under fixed geometry and boundary conditions.”

또한 insertion이 degeneracy를 반드시 깨는 것은 아님. Pure displacement-controlled, shape-only insertion이면 여전히 ratio-only일 수 있음. Fixed external \(P_{\text{task}}\), force limit, absolute wrench 또는 contact reaction criterion이 명시되어야 함.

---

### 14. **[MINOR] `hard_identifiability_v2.json` §confound guard — equivalence margin가 observed numerical floor보다 지나치게 큼**

Paired common-scale shape relative RMS는 약 \(0.0012\)–\(0.0028\)이고 frozen tolerance는 \(0.05\)임. 즉 tolerance가 observed discrepancy보다 약 18–40배 큼. citeturn124037view5

현재 결과는 매우 명확한 PASS이므로 결론 자체는 문제없음. 다만 real-sensor claim으로 연결하려면 equivalence margin를 다음 중 하나에 묶어야 함.

- camera/keypoint noise floor
- calibration repeatability
- downstream critic sensitivity
- numerical mesh variation

그리고 \(c\) 값, support geometry, length, motion duration을 추가해 theorem scope를 넓히는 편이 좋음.

---

### 15. **[MAJOR] `hard_s34_manifest.json` §TEST split — TEST 6 settings는 6개의 독립 landscape가 아니라 3개 unique landscape + 3개 invariance controls임**

TEST는 정확히 다음 common-scale pairs로 구성됨.

\[
(B1\_w1,R0),\quad(B3\_w2,R1),\quad(B2\_w1,R2).
\]

현재 task map은 \(B/w\)에 의존하고 ratio pair boundaries가 정확히 중복되므로, response-function 관점의 effective TEST sample은 **3개 unique landscapes**임. 나머지 3개는 degeneracy/invariance validation controls임. citeturn124037view5turn657813view4

6개를 단순 평균하면 같은 세 landscape를 두 번 가중하는 셈임. 보고 시 다음처럼 분리 필요.

- generalization: 3 unique \(B/w\) groups
- invariance: 각 group 내 common-scale counterpart
- uncertainty unit: setting이 아니라 ratio cluster

이는 task-only student 결과를 무효화하지 않지만, generalization breadth를 크게 좁힘.

---

### 16. **[MAJOR] `hard_critic_results_v2.json` §bootstrap — uncertainty가 training/property-cohort uncertainty를 전혀 포함하지 않음**

현재 teacher, blind, student는 fixed architecture와 fixed training seed에 조건부인 point estimates임. Evaluation bootstrap은 5개 simulation draws를 resample하지만 sharp settled maps가 동일해 CI가 degenerate함. Repository가 이 한계를 이미 명시하고 있는 점은 좋음. citeturn124037view5

하지만 다음 uncertainty는 남아 있음.

- critic/student initialization
- TRAIN/VAL property split
- 3개 unique TEST ratio groups
- hyperparameter selection
- history trajectory subset
- sensor/noise realization

최소한 여러 model seeds와 여러 property-cluster splits를 사용하고, ratio group을 cluster unit으로 하는 hierarchical 또는 cluster bootstrap이 필요함. 현재 숫자는 “fixed trained models에 조건부인 point estimates”로만 써야 함.

---

### 17. **[MAJOR] history encoder implementation — 현재 student는 sequential \(m_t\) adaptation이 아니라 offline batch estimator임**

Proposal의 \(H_t\)는 observation, action, grasp index, contact metadata를 포함하고 performance-vs-interaction-count adaptation curve를 핵심으로 둠. 현재 implementation은 setting마다 4 grasps × 4 motion templates × 2 seeds, 총 32 prescribed histories를 만든 뒤 trajectory별 latent를 평균해 TEST representation을 구성함. citeturn680569view0turn124037view5

즉 현재 결과는 다음을 보여줌.

> “충분한 offline prescribed interaction set이 주어졌을 때 task map을 복원할 수 있음.”

반면 아직 보여주지 않은 것은 다음임.

> “새 object를 처음 만난 상태에서 \(m_t\)가 상호작용마다 update되어 grasp decision을 개선함.”

또한 current task feature builder는 proposal에 적힌 explicit grasp/free-length identifier와 complete action metadata를 그대로 포함하지 않음. Shape가 action과 length를 간접 암호화할 수 있으므로 누락이 무해하다고 볼 수 없음. Fresh test에서는 explicit \(g_i,\ell_i,a_i\), masks, timestamps를 넣고 \(k=0,1,2,4,8,\ldots\) prefix curve를 직접 평가해야 함.

---

### 18. **[MAJOR] teacher/student comparison — map matching은 유효하지만 latent identification 및 equal-budget evidence는 아님**

Teacher의 \(\phi(B,w)\)도 critic과 jointly learned된 4-D coordinate이며, mechanics-defined latent가 아님. Student는 이 arbitrary teacher coordinate를 distill함. 따라서 latent coordinate 자체의 일치나 linear property readout은 식별성 근거가 되기 어려움. Repository의 explicit property recovery가 INCONCLUSIVE인 것과 일관됨. citeturn124037view5

현재 가장 강한 근거는 latent coordinate가 아니라 **held-out map recovery**임. 이 부분은 올바른 선택임.

다만 비교 budget은 구분해야 함.

- teacher: privileged \(B,w\), zero deployment interaction
- student: 32 prescribed deployment interactions
- blind: neither property nor history

따라서 teacher는 fair-budget competitor보다 adaptation ceiling임. Training samples, model capacity, deployment information cost를 별도 표로 보고해야 함.

또한 boundary metric에서 censored prediction을 제외하면 all-success/all-failure map이 boundary penalty를 회피할 가능성이 있음. 현재 TEST rows에는 실질 영향이 없더라도 후속 metric에는 interval-censored penalty와 resolved-coverage를 함께 넣는 편이 안전함.

---

### 19. **[MAJOR] `sim/sweep.py` 및 `hard_sweep_manifest.json` — 4,335 rollouts는 4,335개의 독립 stochastic task trials가 아님**

모든 motion templates는 동일한 exact terminal clamp pose에서 끝나고 static equilibrium까지 settle함. Random draw는 clamp start, duration, arc amplitude뿐이며 최종 geometry는 동일함. 따라서 elastic contact-free system에서는 많은 rollout이 같은 static map의 numerical/path robustness replicate에 가까움. citeturn319879view1turn557131view4

실제 independent scientific units는 대략 property × grasp cells이며, template/seed는 auxiliary robustness unit임. “4,335 rollouts”를 sample size처럼 전면에 내세우면 pseudoreplication 공격을 받을 수 있음.

추가 문제는 convergence criterion임.

- chunk 단위 tip drift threshold가 약 1 mm
- task clearance \(h=9\) mm이므로 threshold가 \(h\)의 약 11%
- batch-global maximum으로 convergence를 판정하고 동일 boolean을 batch rows에 기록

현재 0% non-converged 결과는 좋지만 boundary-neighbor cells에서는 per-environment drift, force residual, energy change를 더 엄격히 확인해야 함. Initial shape, clamp terminal error, contact, sensor noise처럼 **final task outcome에 남는 perturbation**을 넣어야 stochastic success probability가 의미를 가짐.

---

# 2. Epistemic-labeling audit

### 20. **[MAJOR] `STATUS.md` §Q1 — formal gate와 scientific interpretation을 한 문장 더 분리할 필요 있음**

`Q1 formal frozen gate = GO`는 정확함. Pre-registration provenance, fixed decision function, all resolved point estimates, strict B/w directions, exact ratio-pair invariance도 확인됨. citeturn124037view5turn657813view3turn657813view4

다만 다음 표현은 과도함.

- “fourth-root boundary law confirmed”
- “prefactor matches to 0.2%”
- “all 15 settings in-regime”를 uncertainty qualification 없이 사용

권장 문구:

> “The frozen directional/boundary gate passed. At fixed \(h\), the extracted boundaries are consistent with the \(B/w\) fourth-root prediction at the declared grid resolution, conditional on same-simulator self-weight \(B_{\mathrm{eff}}\) calibration. All point-estimated boundaries pass \(\Pi_g\le0.5\); one boundary bracket overlaps the guard.”

Formal verdict는 유지하면서 scientific claim만 정확히 좁히는 수정임.

---

### 21. **[MINOR] `STATUS.md` §Q2 task-only student — 약간 under-claimed되어 있음**

현재 `SECONDARY/EXPLORATORY` label은 안전하고 pre-registered YES로 취급하지 않는 것도 맞음. Frozen success rule이 probe-enriched variant에 있었고 task-only variant에는 thresholded success rule이 없었기 때문임. citeturn124037view5

다만 task-only row와 map metric 자체가 완전히 사후 발명된 것은 아니므로, 가장 정밀한 label은 다음임.

> **“Pre-specified secondary evidence; non-confirmatory because no frozen task-only PASS rule existed.”**

단순 “exploratory”보다 provenance를 더 정확히 전달함. \(m_t\) functional이 established가 아니라는 결론은 그대로 유지해야 함.

---

### 22. **[MINOR] `STATUS.md` §full temporal schema — causal attribution이 아직 강함**

STATUS는 terminal-z student의 실패가 full-temporal \((y,z)\) schema에서 reversal된 점을 들어 temporal schema가 key change였다고 읽음. 같은 문단에서 same-split ablation이 없음을 인정하므로 심각한 overclaim은 아니지만, split, architecture, training target, data volume도 함께 바뀌었을 가능성이 있음. citeturn124037view5

“consistent with the full temporal schema being the key change”보다는 다음이 적절함.

> “The reversal is associated with the full-temporal pipeline; causal attribution to temporal information remains untested.”

또한 `hard_adaptation_curve.png`가 interaction prefix \(k\)에 따른 curve가 아니라 history-composition variants 비교라면 파일명을 `history_variant_comparison`으로 변경하는 편이 좋음. 진짜 adaptation curve는 \(k=0,1,2,\ldots\) 성능이어야 함.

---

### 23. **[MINOR] `STATUS.md` §Process deviation — STOP-branch deviation 처리는 epistemically 적절함**

Probe qualification이 INCONCLUSIVE였고 approved branch가 STOP이었음에도 C4/C5 histories를 생성한 것은 실제 deviation임. Repository는 이를 명시하고, owner ruling을 retroactive acceptance로 기록하며, probe-path 숫자를 pre-registered result로 계산하지 않고 Q2를 NOT ESTABLISHED로 유지함. 이 처리는 타당함. citeturn124037view5

중요한 원칙은 다음임.

- owner acceptance는 governance/provenance acceptance
- methodological preregistration을 소급 복원하지는 못함
- fresh manifest + new data만 confirmatory status를 부여할 수 있음

현재 문서가 이 구분을 대체로 지키고 있으므로 추가 penalty는 필요하지 않음.

---

### 24. **[MINOR] `README.md`와 QA 표현 — public entry point가 STATUS보다 뒤처져 있음**

README 상단의 “nothing has been run yet” 및 headline의 “interaction-identified mechanics latent”는 현재 STATUS와 맞지 않음. Q1은 실행 완료됐지만 \(m_t\)는 아직 pre-registered functional confirmation이나 property identification을 얻지 못했음. citeturn790549view0turn124037view5

권장 headline은 예를 들면 다음과 같음.

> “Mechanics-predicted grasp boundary and a candidate history-conditioned task latent”

또한 red-team `47/47`은 매우 유용하지만 범위는 data integrity, metric arithmetic, gate recomputation임. Student model training과 construct validity의 independent replication은 아님. STATUS가 이미 이를 caveat하므로 README/figure caption에서도 같은 한정을 유지하면 됨. citeturn124037view5

---

# 3. 2024–2026 novelty re-check

### 25. **[MAJOR] `strategy/proposal_v2.md` line 148 — 정확히 좁혀진 conjunctive novelty claim은 현재 검색에서 유지 가능함**

2024–2026 primary sources와 ICRA 2026 program을 다시 확인한 결과, 다음 네 요소를 **동시에** 수행한 prior DLO work는 찾지 못했음.

1. along-object grasp-quality landscape 측정  
2. controlled mechanics-property intervention  
3. landscape boundary 추출  
4. task-specific mechanics prediction과 boundary 비교  

따라서 다음 sentence는 유지 가능함.

> “In the reviewed corpus through 2026-08-17, we found no prior work satisfying all four elements simultaneously.”

Repository의 현재 문장도 이 네 요소를 결합하고 있음. citeturn707287view6

가까운 신규/재확인 neighbor는 다음과 같음.

- **CG-CNN, 2024:** simulated grasp attempts로 candidate grasp quality를 label하고 optimal grasp pose를 선택함. 그러나 핵심 축은 collision-aware grasping이며 controlled \(B,w\) intervention이나 mechanics boundary comparison은 없음. citeturn488429view0
- **In-Hand Following, IROS 2024:** tactile sensing으로 DLO를 따라가며 in-hand contact/grasp location을 연속적으로 조절하지만, property-intervention landscape나 predicted boundary는 없음. citeturn488429view2
- **CaRoBio, 2025:** node-slot distance와 torsion angle을 이용해 optimal grasp node를 선택하고 rigidity가 다른 cables를 실험함. 그러나 controlled mechanics landscape나 boundary law는 없음. citeturn707287view2
- **Saccani et al., ICRA 2026:** uncertainty로 candidates를 filter한 뒤 geometric ranking을 수행함. Mechanics intervention이나 boundary prediction은 없음. citeturn707287view3
- **Active Perception for DLO Stiffness Estimation, ICRA 2026:** boundary constraint와 grasp location이 stiffness identifiability에 미치는 영향을 분석하고 RL probing을 학습함. 이는 interaction-based identification novelty를 직접 위협하지만, grasp-quality landscape와 mechanics boundary claim은 충족하지 않음. citeturn707287view4
- **DLO-Lab, 2026:** 다양한 material parameters와 strategic grasp proposals를 제공하고 system identification도 수행하지만, controlled along-object landscape의 boundary를 별도 mechanics law와 비교하지 않음. citeturn318710view6turn488429view3

검색 기반 결론이므로 “all published work”가 아니라 **reviewed corpus와 검색 종료일을 명시한 claim**으로 유지하는 것이 안전함.

---

### 26. **[MAJOR] `strategy/proposal_v2.md` line 150 — “material-conditioning by per-point-critic cell is empty”는 더 이상 안전하지 않음**

Broad novelty는 다음 prior work들과 충돌할 가능성이 큼.

- Li and Choi는 DLO flexibility에 따라 grasp point를 parameterize하고 flexibility-conditioned insertion policy를 사용함. citeturn707287view5
- CG-CNN은 per-candidate grasp quality를 예측함. citeturn488429view0
- CaRoBio는 cable nodes 전반을 score하여 optimal node를 결정함. citeturn707287view2
- DLO-Lab은 grasp sensitivity를 명시적으로 다루고 strategic grasp points를 제안함. citeturn318710view6

따라서 다음은 주장하지 않는 편이 좋음.

- first material-aware DLO grasp selection
- first per-point or dense DLO grasp-quality predictor
- first interaction-based DLO mechanics identification
- first cross-task dynamics latent

안전한 delta는 끝까지 네 conjuncts를 유지하는 것임.

> “Prior work contains material-aware policies, candidate-quality predictors, and interaction-based identification, but we found no work combining controlled mechanics interventions, an along-object task-quality landscape, boundary extraction, and a task-specific mechanics prediction.”

---

### 27. **[MINOR] `strategy/proposal_v2.md` §Related-work matrix — 신규 행과 명시적 Boolean columns 추가 필요**

최종 paper의 related-work table에는 최소한 다음 행을 추가하는 편이 좋음.

- CG-CNN
- In-Hand Following
- CaRoBio
- Uncertainty-Aware Stereo Grasp Point Selection
- Active Perception for DLO Stiffness Estimation

Columns는 다음처럼 claim의 conjuncts에 직접 대응시키는 것이 유리함.

`along-object map / controlled B or w / boundary extracted / mechanics prediction / interaction history / task objective / real robot`

이렇게 하면 novelty가 수사적 “first”가 아니라 auditable set intersection으로 보임. citeturn488429view0turn488429view2turn707287view2turn707287view3turn707287view4

---

# 4. Skeptical CoRL/ICRA reviewer simulation

### 28. **[MAJOR] Reviewer attack 1 — “Mechanics validation is circular, mesh-specific, and simulator-specific.”**

Reviewer가 가장 먼저 지적할 가능성이 높은 내용:

> “You calibrate \(B_{\mathrm{eff}}\) using the same self-weight sag equation and the same discretized simulator that you later claim to validate through a self-weight sag boundary. The 0.2% prefactor agreement is largely calibration closure. There is no mesh convergence, second simulator, or hardware closure.”

이 공격은 findings 4–8 때문에 상당히 강함. citeturn319879view3turn470580view0turn557131view0

**Preempting evidence**

- known tip-force 또는 moment-based independent \(B_{\mathrm{eff}}\) calibration
- 최소 3개 interval에서 observable convergence
- held-out gravity lengths/masses에 no-refit prediction
- boundary exponent의 interval-censored confidence interval
- 소수라도 real rod의 sag/boundary/optimum paired test
- 가능하면 second solver 또는 analytical discrete-beam reference

---

### 29. **[MAJOR] Reviewer attack 2 — “The \(m_t\) result is tiny-\(N\), offline pseudoreplication, not adaptation.”**

예상 공격:

> “There are only three unique held-out response landscapes, one trained model per row, and 32 prescribed interactions are averaged offline. The bootstrap resamples deterministic simulation draws rather than training or object uncertainty. This does not demonstrate cold-start adaptation or a reusable mechanics latent.”

현재 STATUS가 이 한계를 상당 부분 인정하고 있으나, headline thesis와는 아직 거리가 있음. citeturn124037view5

**Preempting evidence**

- fresh preregistered property clusters
- multiple model-training seeds
- explicit action/grasp metadata
- frozen acquisition policy
- \(k=0,1,2,4,8,16,32\) sequential adaptation curve
- cluster-level uncertainty
- prescribed-probe, generic sysID, blind, teacher baselines
- simulated sensor noise와 real interaction histories
- frozen encoder의 genuinely unseen task transfer

---

### 30. **[MAJOR] Reviewer attack 3 — “The task is engineered to produce the predicted answer, with no demonstrated manipulation utility.”**

예상 공격:

> “The lift task has a trivial material-independent optimum, so only a boundary moves. The new tip-placement task is then designed specifically to create an interior material-dependent optimum. Even if the analytic toy task works, why does this improve real manipulation?”

현재 proposal도 feasible boundary와 optimum utility를 분리하고 있으며, STATUS는 selection regret가 null임을 명시함. citeturn124037view3turn124037view5

**Preempting evidence**

- target/objective 공개 후 predicted optimum curve 사전 도출
- optimum shift effect size를 grid 전에 고정
- blind-held-out target poses와 materials
- direct selection regret versus blind/material-agnostic/mechanics heuristic
- qualitatively different second task
- real robot에서 failure reduction 또는 task error 감소
- task를 바꾸어도 frozen encoder가 유효하다는 cross-task transfer

---

# 5. Hardening-B 이후 가장 가치가 높은 세 실험

### 31. **[MAJOR] Experiment 1 — independent mechanics closure + mesh convergence + 소규모 hardware closure**

**설계**

1. self-weight가 아닌 known tip force 또는 imposed moment로 \(B_{\mathrm{eff}}\) calibration  
2. \(\Delta s=0.02,0.01,0.005\) 등에서 각각 calibration  
3. held-out gravity lengths/masses의 sag, Q1 boundary, hardening-B optimum을 no-refit prediction  
4. analytical lumped-load correction과 결과 비교  
5. 3–5개 real DLO에 동일 protocol 적용  

**이유**

현재 프로그램의 가장 큰 단일 취약점인 calibration closure와 mesh bias를 동시에 해소함. 성공하면 “same-simulator scaling sanity check”를 “independently calibrated mechanics prediction”으로 격상할 수 있음. citeturn319879view0turn319879view3turn470580view0

**Primary outcomes**

- held-out normalized boundary/optimum error
- mesh refinement에 따른 convergence rate
- real/sim prediction interval coverage
- no-refit fourth-root exponent와 prefactor

---

### 32. **[MAJOR] Experiment 2 — fresh preregistered online cold-start \(m_t\) adaptation trial**

**설계**

- 기존 TEST와 disjoint한 새로운 property clusters
- ratio pairs는 동일 cluster로 묶되 unique ratios 수를 크게 확대
- 여러 training seeds
- explicit \(g_i,\ell_i,a_i\), trajectory timing, contact/wrench metadata
- acquisition policy를 manifest에서 freeze
- interaction prefixes \(k=0,1,2,4,8,16,32\)
- 각 \(k\)에서 map error, lower/upper boundary error, optimum error, selection regret 측정
- teacher, blind, prescribed one-shot probe, generic sysID, scratch encoder와 비교

**이유**

현재 evidence가 보여주는 offline set decoding과 thesis가 요구하는 sequential interaction identification 사이의 간극을 직접 닫는 실험임. Task-only student를 fresh success rule의 PRIMARY로 지정하는 STATUS의 upgrade path와도 일치함. citeturn124037view5

가장 중요한 plot은 architecture variants가 아니라

\[
\text{task performance or regret}\quad\text{vs}\quad k
\]

여야 함.

---

### 33. **[MAJOR] Experiment 3 — common-scale degeneracy를 의도적으로 깨는 real force/contact task**

**설계 후보**

- insertion/buckling with fixed external \(P_{\text{task}}\)
- payload-supported placement
- force-limited distal pushing
- contact-rich routing with measured reaction wrench

각 task에서 \((B,w)\)와 \((cB,cw)\) pairs를 사용하고 다음 variants를 비교함.

- shape-only
- shape + wrench
- explicit sysID
- hardening-B frozen history encoder + new head
- scratch encoder

**이유**

현재 lift와 shape channel은 본질적으로 \(B/w\) ratio에 지배됨. Absolute wrench나 non-scaled external load가 들어가야 \(B\)와 \(w\)의 common-scale direction이 task outcome에 실제로 드러남. 이 실험 하나로 다음 세 주장을 동시에 시험할 수 있음.

1. degeneracy derivation이 실제 sensor/task design을 예측하는가  
2. \(m_t\)가 단순 ratio decoder를 넘어서는가  
3. frozen encoder가 다른 task에서 reusable mechanics module인가  

Real objects를 포함하면 novelty보다 더 중요한 **실제 task utility와 causal mechanics validity**를 확보할 수 있음. 현재 repository가 friction/contact sensing, multi-task generality, real-robot transfer를 미확립으로 명시한 공백과 직접 대응함. citeturn124037view5

---

## 최종 판정

현재 프로그램의 strongest established result는 다음으로 보는 것이 정확함.

> **동일 simulator와 fixed discretization에서, self-weight-calibrated \(B_{\mathrm{eff}}\)를 사용한 lift-and-clear task의 feasible boundary가 \(B\)와 \(w\) interventions에 대해 예측 방향으로 이동하고 common-scale ratio pairs에서 불변이며, fixed-\(h\) fourth-root prediction과 grid resolution 내에서 일치함.**

아직 established로 보기 어려운 항목:

- prefactor의 독립적 external validation
- full boundary-law exponent 및 \(h^{1/4}\) dependence
- material-dependent optimum
- sequential cold-start \(m_t\)
- physical \(B,w\) identification
- more than three unique held-out task landscapes
- training/object-population uncertainty
- contact-rich/real-robot utility

따라서 현재 formal Q1 GO는 유지하되, paper-level claim은 한 단계 낮춰 쓰는 것이 적절함. Hardening-B freeze 전에는 findings **1과 2**가 실제 BLOCKER이며, 그 이후 최우선은 finding **31의 independent mechanics closure**와 finding **32의 fresh sequential preregistration**임.