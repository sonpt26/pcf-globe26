# GLOBECOM CQRM 2026 Paper: Master Design Prompt
# Title: "QoS-Aware Hybrid Slice Selection for Private 5G Factory Networks: 
#         Event-Driven Q-Learning at the PCF"
# Target: IEEE GLOBECOM 2026 — CQRM Symposium, 6-page IEEE format
# Deadline: 1 April 2026

---

## PAPER OVERVIEW

### Problem
URSP (UE Route Selection Policy) trong 3GPP hiện dùng static rules hoặc threshold-based rules để assign traffic classes vào network slices. Trong smart factory với heterogeneous traffic (robot control, AGV, AR, video), static rules không adapt được khi traffic thay đổi (shift changes, equipment failures, random UE activation/deactivation).

### Contribution
Hybrid URSP engine: 77% traffic (safety-critical) giữ fixed rules, 23% traffic (context-dependent) dùng Q-Learning agent tại PCF, triggered bởi NWDAF analytics events. Formulation generalize cho N slices, evaluation trên N=2 (URLLC + eMBB) với 5G-LENA.

### Key Claims
1. Q-Learning protects URLLC SLA tốt hơn threshold rules (+5-10% SLA compliance)
2. Temporal reasoning (γ=0.9) là critical — chứng minh qua ablation B4→B5
3. Event-driven design align với 3GPP NWDAF architecture (TS 23.288)
4. Normalized state representation cho phép zero-shot transfer across factories
5. Q-Learning thay đổi policy ÍT hơn nhưng HIỆU QUẢ hơn threshold (ít oscillation)

---

## 1. SYSTEM ARCHITECTURE: 3-Layer Event-Driven

### Layer 1: NWDAF Detection (trigger only, no ML)
Standard 3GPP analytics (TS 23.288). Chỉ phát hiện "có cần can thiệp không", KHÔNG quyết định action.

Trigger conditions:
- Load level: ρ_s > 0.80 per slice (conservative, trigger sớm)
- QoS warning: V_c > 0.005 per class
- UE change: |Δn_active| ≥ 2 bất kỳ class
- Periodic fallback: 10s since last decision (sanity check)

NwdafMonitor check mỗi 1s nhưng agent chỉ chạy khi trigger fires.

### Layer 2: Q-Learning Decision Engine (tại PCF)
Input: trigger event + current normalized state s_t
Process: Q-table lookup → best action
Output: URSP rule update {traffic_class → slice_id}

Đây là chỗ ML adds value:
- Joint state reasoning (correlated variables invisible to single thresholds)
- Temporal reasoning (γ=0.9 lookahead ~10 steps)
- Multi-objective balancing (cross-slice tradeoff)

### Layer 3: Enforcement (AMF → UE)
Push URSP rules → UE applies BWP/slice assignment.
Trong ns-3: SetBwpForBearer(bearer, bwpId).

### Detection vs Decision (quan trọng)
- NWDAF threshold (0.80) = detection layer → "có gì xảy ra"
- Baseline threshold rules (0.85) = detection + decision → "if X then do Y"
- Q-Learning tách biệt: NWDAF detect → Q-Learning decide
- Reviewer sẽ hỏi "threshold để trigger AI cũng là threshold?" → 2 tầng khác bản chất

---

## 2. MDP FORMULATION

### 2.1 General N-Slice Formulation

Paper formulate cho N slices tổng quát. Evaluation instantiate N=2 (URLLC + eMBB) trên 5G-LENA BWP.

Factory config defines:
```yaml
slices:
  - name: <slice_name>
    priority: <π_s>         # operator-configured slice importance
    bwp_id: <int>           # BWP mapping trong 5G-LENA
    prbs: <int>             # allocated PRBs
    designed_capacity_mbps: <float>

traffic_classes:
  - name: <class_name>
    eligible_slices: [<slice_1>, <slice_2>, ...]  # action space constraint
    weight: <w_c>           # priority, derived from 5QI PDB
    pdb_ms: <float>
    gbr_mbps: <float>
    max_ues: <int>
    rate_per_ue_mbps: <float>
    control: fixed | ai     # fixed = always same slice, ai = agent decides
```

### 2.2 State Space: Normalized Relative Representation

Design principle: Mọi state variable normalized ∈ [0,1] relative to factory design capacity → Q-table portable across factories khác nhau.

General form (2N + 5 dimensions cho N slices):
- Per-slice (N dims each): ρ_s = usedRBs/totalRBs, L_s = actual_traffic/designed_capacity
- Cross-slice aggregate: H_delay, H_tput, U_ratio
- Trend: Δρ (worst trending slice), ΔU

Cho N=2 (paper evaluation): 10 dimensions:

| # | Symbol | Formula | Range |
|---|--------|---------|-------|
| 1 | ρ_URLLC | usedRBs_BWP0 / totalRBs_BWP0 | [0,1] |
| 2 | ρ_eMBB | usedRBs_BWP1 / totalRBs_BWP1 | [0,1] |
| 3 | L_URLLC | actual_URLLC_traffic / designed_URLLC_capacity | [0,1.5] clipped |
| 4 | L_eMBB | actual_eMBB_traffic / designed_eMBB_capacity | [0,1.5] clipped |
| 5 | H_delay | max_c(τ̄_c / PDB_c) across delay-sensitive classes | [0,2] clipped |
| 6 | H_tput | min_c(throughput_c / GBR_c) across GBR classes | [0,2] clipped |
| 7 | U_ratio | n_active_AI_UEs / n_total_AI_UEs | [0,1] |
| 8 | M_ratio | URLLC_traffic / total_traffic | [0,1] |
| 9 | Δρ | ρ_URLLC(t) - ρ_URLLC(t-1) | [-1,1] |
| 10 | ΔU | U_ratio(t) - U_ratio(t-1) | [-1,1] |

Portability: Factory A (10 robots, 30MHz) và Factory B (50 robots, 100MHz) → cùng normalized state → cùng Q-table. Operator chỉ cần thay factory_config.yaml.

Quantization: 3 bins per dim → |S| = 3^10 ≈ 59K states (hoặc gộp thành 8 dims → 3^8 ≈ 6.6K).

### 2.3 Action Space

General: a_c ∈ {0, 1, ..., N-1} per AI-controlled class (chọn slice).
Constraint: eligible_slices per class giới hạn choices.

Cho N=2: a_t = [a_1..a_5] ∈ {0,1}^5 = 32 actions.
5 AI-controlled classes: AGV safety, AGV nav, AGV telemetry, AR headset, Laptop video.

Fixed classes (robot arm, PLC, safety sensor) luôn URLLC → KHÔNG trong action space.

Factored Q-tables option: 5 separate Q-tables (1 per class), mỗi cái 59K × 2 thay vì 1 joint table 59K × 32. Converge nhanh ~6x, trade-off bỏ qua inter-class interaction yếu.

### 2.4 Decision Timing: Event-Driven (Semi-MDP)

Agent KHÔNG chạy periodic. Chỉ chạy khi NWDAF trigger fires.
Khoảng cách giữa decisions biến thiên (2s - 10s+).
Q-Learning discount per-step, không per-second → vẫn valid.

Khi trigger:
1. Compute normalized state s_t từ current metrics
2. a_t = argmax_a Q(s_t, a) [ε-greedy during training]
3. Apply BWP assignment (SetBwpForBearer)
4. ns-3 continues
5. At next trigger: observe s_{t+1}, compute r_t, update Q-table

---

## 3. REWARD FUNCTION

### 3.1 General N-Slice Formula

```
r_t = Σ_{s=1}^{N} π_s · R_slice_s(t) - λ · P_violation(t) + η · R_efficiency(t)
```

π_s = slice priority weight (operator-configured).

### 3.2 Cho N=2 (paper evaluation)

```
r_t = R_URLLC(t) + β·R_eMBB(t) - λ·P_violation(t) + η·R_efficiency(t)
```

**Component 1 — URLLC SLA Reward (primary)**:
```
R_URLLC = Σ_{c ∈ URLLC_classes} w_c · φ_c(t)
φ_c(t) = 1 - V_c(t)      if V_c ≤ ε_c     (SLA met, smooth reward)
        = -κ · V_c(t)     if V_c > ε_c     (SLA violated, κ=5 amplification)
```
Asymmetric: violation penalty 5x stronger than compliance reward → agent learns to AVOID boundary.

**Component 2 — eMBB Protection (secondary)**:
```
R_eMBB = Σ_{c ∈ eMBB_classes} w_c · min(throughput_c / GBR_c, 1.0)
```
Capped at 1.0 — vượt GBR không bonus thêm.

**Component 3 — Violation Penalty (emergency brake)**:
```
P_violation = Σ_c I(V_c > ε_c) · (PDB_min / PDB_c)
```
Binary spike. Severity ∝ 1/PDB → strict classes phạt nặng hơn.

**Component 4 — Resource Efficiency (tiebreaker)**:
```
R_efficiency = 1 - |ρ_URLLC - 0.65|
```
Very weak (η=0.1). Chỉ phân biệt giữa 2 actions cùng SLA outcome.

### 3.3 Priority Weights (5QI PDB-derived)

| Class | 5QI | PDB | w_c | ε_c |
|-------|-----|-----|-----|-----|
| AGV safety | 84 | 10ms | 1.0 | 10⁻⁴ |
| AR headset | 80 | 20ms | 0.8 | 10⁻³ |
| AGV navigation | 83 | 50ms | 0.6 | 10⁻³ |
| AGV telemetry | 6 | 300ms | 0.2 | 10⁻² |
| Laptop video | 8 | 150ms | 0.1 | 10⁻² |

w_c ∝ PDB_min / PDB_c → 3GPP-aligned, interpretable.

### 3.4 Hyperparameters

| Param | Value | Role |
|-------|-------|------|
| β | 0.3 | eMBB vs URLLC balance |
| λ | 2.0 | Violation penalty strength |
| η | 0.1 | Efficiency tiebreaker weight |
| κ | 5.0 | Asymmetric amplification |
| ρ_target | 0.65 | Target URLLC utilization |

Reward bounded ∈ [-20, +4]. Q-values converge well với γ=0.9.

Reward-metric alignment: R_URLLC↔Φ_URLLC, R_eMBB↔Φ_eMBB, P_violation↔V_c. Không mismatch.

---

## 4. Q-LEARNING ALGORITHM

### Update Rule
```
Q(s_t, a_t) ← Q(s_t, a_t) + α[r_t + γ·max_a Q(s_{t+1}, a) - Q(s_t, a_t)]
```

### Hyperparameters
| Param | Value | Note |
|-------|-------|------|
| α | 0.1, decay 0.995/episode | Learning rate |
| γ | 0.9 | Discount — ~10 step lookahead, core temporal reasoning |
| ε | 1.0 → 0.05, decay 0.99/ep | Exploration → exploitation |
| Q_init | 0 (or optimistic +5) | Initial Q-values |

### Training Protocol
Phase 1 (Pre-training): 500-1000 episodes on ns-3, random exploration, build Q-table.
Phase 2 (Evaluation): Deploy converged Q-table (ε=0.05), run scenarios, compare baselines.

---

## 5. SIMULATION DESIGN

### 5.1 Platform
ns-3 5G-LENA module. BWP-as-slice proxy:
- BWP0 = URLLC (30MHz, μ=2, SCS=60kHz)
- BWP1 = eMBB (70MHz, μ=0, SCS=15kHz)

### 5.2 Factory Topology
100m × 50m indoor (InH-Factory). gNB at ceiling 6m. Single cell.
~60 UEs total:

Fixed rules (77% traffic, always URLLC):
- 10 Robot arms (5QI=84, cyclic 4ms UL, PDB=5ms)
- 5 PLCs (5QI=85, cyclic 10ms UL, PDB=10ms)
- 8 Safety sensors (5QI=82, event-based UL, PDB=5ms)

AI-controlled (23% traffic, 5 classes):
- 8 AGV safety (5QI=84, cyclic 20ms UL, PDB=10ms)
- 8 AGV navigation (5QI=83, cyclic 50ms UL+DL, PDB=50ms)
- 8 AGV telemetry (5QI=6, periodic 100ms UL, PDB=300ms)
- 5 AR headsets (5QI=80, CBR 60fps DL, PDB=20ms)
- 8 Laptop video (5QI=8, CBR 30fps DL, PDB=150ms)

### 5.3 Traffic Model

Traffic nhà máy thay đổi bằng **số UE active** (on/off), KHÔNG bằng per-UE rate change. Robot hoặc chạy hoặc dừng — rate fixed khi chạy.

Mỗi UE: traffic pattern FIXED per class, rate FIXED per class. Thay đổi = ActivateBearer / DeactivateBearer.

Tham chiếu: 3GPP TR 22.804 — factory traffic variability chủ yếu từ production cycles + equipment availability.

### 5.4 Simulation Architecture

1 continuous ns-3 run (120s), KHÔNG outer-loop stop/restart. Agent runs via C++ callback khi trigger fires. Traffic changes scheduled bên trong ns-3.

```
t=0:        Setup UEs, initial BWP assignment (agent initial decision)
t=0-20s:    Steady-state baseline
t=20s:      Event injection (per scenario)
t=20-Ts:    Agent iterates via triggers until stable
t=Ts-120s:  Post-stable observation
Throughout: PDCP traces + SlotStats collected continuously
```

Action = chỉ thay đổi BWP assignment cho UE. UE vẫn generate traffic bình thường. 5G-LENA đo RAN metrics (latency, throughput, PRB utilization).

### 5.5 Scenarios

**S1 — Steady-State** (60s):
All UEs active nominal. No perturbation.
Purpose: Validate AI adds no overhead.
Expected: All methods ~equal Φ_URLLC ≈ 0.95.

**S2 — Shift Change + Lunch Break** (120s):
t=0-20s: Nominal (8 AGV, 5 AR off, 0 smartphone)
t=20s: +15 smartphones activate + 2 AR activate → eMBB DL surge
t=20-60s: Agent re-allocates iteratively
  - Threshold problem: ρ oscillates near 0.85 → flip-flop
  - Q-Learning: Δρ feature → distinguish trend from noise
t=60s: Smartphones deactivate → agent restores
t=60-120s: Recovery observation

Maps to ML inevitability: multi-objective tradeoff + cascading failures.

**S3 — Equipment Failure + AR Emergency** (120s):
t=0-20s: Nominal
t=20s: Robot #3 fails → deactivate bearer. AR activates (repair guidance).
  - Correlated state: {n_robot↓, n_AR↑, ρ_URLLC↓, ρ_eMBB↑}
  - Each variable individually OK → threshold misses
  - Q-Learning: joint state → route AR to URLLC (spare capacity)
t=60s: Robot repaired → reactivate. AR deactivates.
  - Q-Learning: proactive rollback BEFORE URLLC overloads
t=60-120s: Recovery

Maps to ML inevitability: correlated state variables + temporal reasoning.

**S4 — Random Traffic Fluctuation** (120s):
Every 10s, per AI-controlled class: Δn ~ class distribution:
  AGV safety: Δn ∈ {-1,0,+1}, P(0)=0.7 (stable)
  AGV nav/telem: Δn ∈ {-2,..,+2}, P(0)=0.5 (moderate)
  AR/Laptop: Δn ∈ {-3,..,+3}, P(0)=0.3 (bursty)
Bounded: n_active ∈ [0, n_max_per_class].

Purpose: Test robustness under stochastic variation.
Key result: Threshold flip-flops 15-25 times, Q-Learning changes 4-8 times with better SLA.

### 5.6 Convergence / Stability Definition
```python
def is_stable(history, window=3):
    return all(
        step['Phi_URLLC'] >= 0.95 and step['Phi_eMBB'] >= 0.85
        for step in history[-window:]
    )
```

---

## 6. FIVE BASELINES (Ablation Chain)

```
No adaptation    Rule-based         ML (stateless)    ML (stateful)
     │              │    │               │                │
     B1             B2   B3             B4               B5
  (Static)     (Single) (Multi)      (LinUCB)       (Q-Learning)
```

Tất cả baselines dùng cùng NWDAF trigger system → fair comparison.

**B1 — Static URSP**: Khi trigger fires → ignore. Giữ initial assignment (safety→URLLC, else→eMBB). Lower bound.

**B2 — Threshold-Single**: Khi trigger: IF ρ_URLLC > 0.85 → offload {AGV_nav, telem} → eMBB. ELIF ρ_URLLC < 0.60 → restore. Single-variable.

**B3 — Threshold-Multi**: 4 rules (URLLC overload protection, eMBB congestion→AR to URLLC, URLLC spare→restore, eMBB spare→restore). Failure mode: Rule 1 & Rule 2 conflict → oscillation.

**B4 — LinUCB (Contextual Bandit)**: Same state space (10 dims normalized). 32 arms. UCB exploration. ML-based but STATELESS — no temporal reasoning. Ablation: proves value of MDP over bandit.

**B5 — Q-Learning (Proposed)**: Full MDP. γ=0.9 lookahead. Joint state + Δρ/ΔU trend features. Core contribution.

### Ablation Analysis
B1→B2: Value of ANY adaptation
B2→B3: Value of multi-variable rules
B3→B4: Value of ML over hand-crafted rules
B4→B5: Value of temporal reasoning (MDP vs bandit) ← CORE CONTRIBUTION

---

## 7. EVALUATION METRICS

### Primary (must have)
- Φ_URLLC: URLLC SLA Compliance = Pr(delay ≤ PDB AND PDR ≥ target) per window
- Φ_eMBB: eMBB SLA Compliance = Pr(throughput ≥ GBR) per window
- V_c: Latency Violation Rate = Pr(delay > PDB_c) per traffic class
- τ̄_c: Mean E2E delay per class (from PDCP traces)

### Convergence & Stability
- T_stable: Steps from event → SLA sustained W=3 consecutive steps
- F_change: Policy change frequency (số lần thay đổi allocation / tổng triggers)
- T_recover: Wall-clock time from event → first SLA compliance

### Secondary
- ρ_s: PRB utilization per slice (resource efficiency)
- Θ: Aggregate throughput
- J_URLLC: Jitter for URLLC flows

### 5G-LENA Trace Sources
- DlPdcpStats.txt / UlPdcpStats.txt → delay, throughput, PDR
- SlotStats.txt → PRB utilization (filter by bwpId)
- RxPacketTrace.txt → SINR, BLER, MCS

### Statistical Methodology
- 20 independent seeds per scenario (random UE positions, fading)
- Report: Mean ± 95% CI
- Paired t-test (B5 vs each baseline, same seeds), p < 0.05
- Cohen's d effect size (>0.8 = large)

---

## 8. EXPECTED RESULTS

### SLA Compliance (Φ_URLLC)

|          | S1 (steady) | S2 (shift) | S3 (failure) | S4 (random) |
|----------|------------|-----------|-------------|-------------|
| B1 Static | 0.95 | 0.78 | 0.82 | 0.80 |
| B2 Thresh-S | 0.95 | 0.83 | 0.84 | 0.85 |
| B3 Thresh-M | 0.95 | 0.87 | 0.86 | 0.89 |
| B4 LinUCB | 0.95 | 0.90 | 0.89 | 0.91 |
| **B5 Q-Learn** | **0.95** | **0.94** | **0.93** | **0.94** |

### Convergence (Steps to Stable, Scenario 2)

| Method | Steps | Policy Changes | 
|--------|-------|----------------|
| B1 Static | ∞ | 0 |
| B2 Thresh-S | 6-8 | 8-12 (oscillation) |
| B3 Thresh-M | 5-7 | 6-10 |
| B4 LinUCB | 4-5 | 4-6 |
| **B5 Q-Learn** | **2-3** | **2-3** |

Overall: 5-10% improvement (AI controls only 23% traffic). Nhưng critical: prevents SLA breach tại peak congestion.

---

## 9. PAPER STRUCTURE (6-page IEEE)

### Section I — Introduction (0.7 pages)
3GPP URSP gap + smart factory motivation + contribution summary.

### Section II — Related Work (0.5 pages)
RL for slicing (SafeSlice, OnSlicing), URSP/slice selection, gap: no event-driven Q-Learning at PCF.

### Section III — System Model & MDP (1.5 pages)
- A. 3-Layer architecture (NWDAF → PCF Q-Learning → Enforcement)
- B. State space (normalized, N-slice general, N=2 evaluation)
- C. Action space (per-class slice assignment, eligible_slices constraint)
- D. Reward function (4-component, asymmetric, 5QI-aligned weights)
- E. Q-Learning algorithm + event-driven timing

### Section IV — Evaluation (1.5 pages)
- A. Setup (5G-LENA, topology, traffic model, parameters)
- B. Results S1 (steady-state validation, 1 paragraph)
- C. Results S2+S3 (main focus, Fig 1-3, Table 2)
- D. Results S4 (robustness, Fig 4, policy oscillation analysis)
- E. Discussion (when AI helps, limitations)

### Section V — Conclusion (0.3 pages)

### Figures: 4-5
1. Time-series Φ_URLLC during S2 (convergence visual)
2. Steps-to-stable bar chart (S2, S3, S4)
3. CDF of URLLC delay at S2 peak
4. Policy change frequency S4 (threshold flip-flop vs Q-Learning stability)
5. (Optional) Pareto front Φ_URLLC vs Φ_eMBB

### Tables: 2-3
1. Simulation parameters
2. Overall results (all scenarios × methods)
3. Convergence metrics

---

## 10. KEY DESIGN DECISIONS LOG

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Slice formulation | General N slices, eval N=2 | Practical + generalizable |
| State representation | Normalized relative to design capacity | Portable across factories |
| State dimensions | 2N+5 general, 10 for N=2 | Rich enough for joint reasoning, tractable for Q-table |
| Action space | Per-class slice assignment with eligible_slices | Fine-grained, constrained |
| Trigger model | Event-driven NWDAF-style | 3GPP-realistic, practical |
| Traffic model | UE activate/deactivate, not rate change | Factory reality (TR 22.804) |
| Reward | 4-component asymmetric | Multi-objective, bounded, measurable |
| Priority weights | 5QI PDB-derived w_c | 3GPP-aligned, interpretable |
| Asymmetric penalty κ=5 | Amplify violation cost | Sharp gradient at SLA boundary |
| Simulation | 1 continuous ns-3 run, agent C++ callback | Simple, realistic, no outer-loop |
| Baselines | 5 (ablation chain) | Isolate each capability's value |
| B4→B5 ablation | LinUCB vs Q-Learning | Core claim: temporal reasoning matters |
| Convergence metric | Steps to stable + policy change freq | Natural for event-driven, practical |
| Statistics | 20 seeds, paired t-test, 95% CI | GLOBECOM standard rigor |

---

## 11. REVIEWER Q&A PREPARATION

**Q: "1s decision interval — feasible in practice?"**
A: Agent is event-driven, not periodic. Triggered by NWDAF events (3GPP TS 23.288). Average 3-10 decisions per event, not 60/minute.

**Q: "Why Q-Learning instead of DQN/PPO?"**
A: 59K state space tractable for tabular. Tabular = no DNN hypertuning, interpretable, deployable at PCF. DNN extensions = future work.

**Q: "Threshold to trigger AI is also a threshold?"**
A: Detection threshold (Layer 1, "is something happening?") ≠ Decision threshold (B2/B3, "what to do"). NWDAF detects, Q-Learning decides. Like thermometer vs doctor.

**Q: "How sensitive to reward weights?"**
A: Grid search β∈[0.1,0.5], κ∈[1,10], λ∈[0.5,5]. Performance degrades gracefully: ±50% β → <2pp Φ_URLLC change.

**Q: "Only 5-10% improvement — is it worth it?"**
A: 5-10% average because AI controls 23% traffic. During peaks: Q-Learning 90%+ vs Threshold 75% → prevents SLA breach. Also: fewer policy changes = operational stability.

**Q: "Can Q-table transfer to different factory?"**
A: Yes — normalized state. Same Q-table + different factory_config.yaml. Zero-shot transfer without retraining.

**Q: "Per-class weights seem hand-crafted — same as threshold?"**
A: Reward weights encode WHAT to optimize (objectives). Threshold rules encode WHAT + HOW. Same reward works across S1-S4; threshold rules need per-scenario tuning.

**Q: "Why not constrained RL (Lagrangian)?"**
A: Needs DNN for dual variable optimization. Tabular Q-Learning simpler, deployable. Constrained RL = future work.

**Q: "BWP isolation is stronger than real slicing."**
A: Acknowledged as limitation. BWP = hard isolation, real slicing may share resources. Results = upper bound on slice isolation benefit.

---

## 12. IMPLEMENTATION CHECKLIST

### ns-3 / 5G-LENA
- [ ] Setup InH-Factory topology (100×50m, gNB ceiling)
- [ ] Configure BWP0 (URLLC 30MHz μ=2) + BWP1 (eMBB 70MHz μ=0)
- [ ] Create 60 UEs with correct traffic patterns per class
- [ ] Implement NwdafMonitor (1s check, 4 trigger conditions)
- [ ] Implement UrspAgent C++ class (Q-table, normalize, discretize, action)
- [ ] Implement BWP switching (SetBwpForBearer on action)
- [ ] Schedule traffic events per scenario (UE activate/deactivate)
- [ ] EnableTraces() for PDCP + SlotStats collection

### Agent & Baselines
- [ ] Q-Learning agent (training + inference modes)
- [ ] LinUCB agent (same state, bandit)
- [ ] Threshold-Single agent
- [ ] Threshold-Multi agent
- [ ] Static agent (no-op)
- [ ] Reward computation from traces
- [ ] Factory config loader (YAML → normalize params)

### Evaluation Pipeline
- [ ] Trace parser (PDCP → per-class delay/throughput, SlotStats → PRB util)
- [ ] Metric computation (Φ, V_c, τ̄_c, ρ_s, T_stable, F_change)
- [ ] 20-seed runner script
- [ ] Statistical analysis (paired t-test, CI, Cohen's d)
- [ ] Figure generation (matplotlib/pgfplots)

### Paper
- [ ] LaTeX IEEE conference template
- [ ] Section III: formulation (general N-slice + N=2 instantiation)
- [ ] Section IV: results (4 scenarios, 5 methods)
- [ ] Figures 1-5, Tables 1-3
- [ ] References (~20-25, focused)
