# PCF-MORL: Multi-Objective QoS Optimization for 5G Network Slicing
# IEEE GLOBECOM 2026 - Track CQRM
# Specification v6

## 1. System Model

### 1.1 Network Topology
- 1 gNB, 3.5 GHz (n78), 40 MHz, μ=1 (SCS 30 kHz)
- Channel: 3GPP Indoor Factory Dense-High (InF-DH), TS 38.901 Table 7.2-4
- 2 BWP on 1 CC:
  - BWP 0 (URLLC): 15 MHz, 41 PRBs, PF scheduler
  - BWP 1 (eMBB): 25 MHz, 65 PRBs, PF scheduler
- 14 UEs: 6 URLLC (SST=2) + 8 eMBB (SST=1)

### 1.2 Traffic Models

**URLLC (6 UEs):**
- Periodic: 256 bytes/ms (2.048 Mbps) via UdpClient
- Poisson burst: λ=2/s, ~50 KB, peak ~7 Mbps via OnOffApplication

**eMBB (8 UEs):**
- 4 HD cameras: CBR ~25 Mbps via UdpClient (H.265 1080p60)
- 4 digital twin: FTP Model 2 via ThreeGppFtpM2, ~32 Mbps mean

### 1.3 Action Space (|A| = 76)

Per-UE rate control via Token Bucket (TbfQueueDisc) at PGW:
```
a = (rate_urllc, rate_embb)
rate_urllc ∈ {5, 10, 15, 20} Mbps       → 4 values
rate_embb  ∈ {10, 15, 20, ..., 100} Mbps → 19 values (step 5 Mbps)
Total: 4 × 19 = 76 (Discrete)
```

**CRITICAL:** 5G-LENA does NOT implement Session-AMBR enforcement.
Use ns-3 built-in `TbfQueueDisc` on EPC link PGW→gNB (downlink) for per-UE rate limiting.

Dynamic rate change each step:
```cpp
qdisc->SetAttribute("Rate", DataRateValue(DataRate(new_rate_bps)));
```

### 1.4 State Space (S ∈ ℝ¹²)

Per-slice (2 × 5 = 10 dims):
1. offered_load_per_ue (RLC buffer)
2. served_ratio (PDCP TX / offered)
3. delay_95th (PDCP delay)
4. buffer_backlog_per_ue (MAC buffer)
5. active_ue_ratio (scheduler)

System-level (2 dims):
6. total_prb_utilization
7. energy_consumption

All normalized [0, 1].

### 1.5 Vector Reward (ℝ³)

```
r₁ = mean_served_throughput_embb / 30 Mbps        ∈ [0, 1]   → Throughput QoS
r₂ = -frac(URLLC_packets_delay > 1ms)             ∈ [-1, 0]  → Delay QoS
r₃ = -(energy/served_bits) / E_ref                ∈ [-1, 0]  → Energy efficiency
```

E_ref: calibrated from 10 episodes with midpoint policy (rate_urllc=10, rate_embb=50).

### 1.6 Episode Design

- Step: 500ms simTime (~1000 slots)
- Episode: 100 steps = 50s simTime
- γ = 0.99

**Training profile** (1 stochastic production cycle per episode):
- 5 phases: Setup(0-10s), Assembly(10-25s), QC(25-35s), Transport(35-45s), Cooldown(45-50s)
- Stochastic: ±30% intensity, ±3s timing, Poisson bursts λ=1.5/episode

**Eval scenarios** (fixed, no randomness):
- E1: Constant 90% load, 50s → steady-state optimality
- E2: Steady → URLLC burst (5s) → steady → delay protection
- E3: Steady → eMBB surge (10s) → steady → cross-slice protection
- E4: Alternating bursts 5-10s → stability vs reactivity
- E5: Both slices >100% for 25s → damage minimization

---

## 2. Algorithm: GPI-PD

Using morl-baselines library.

```python
agent = GPIPD(
    env=PcfMorlEnv(),
    learning_rate=3e-4,
    gamma=0.99,
    batch_size=128,
    net_arch=[256, 256, 256, 256],
    buffer_size=1_000_000,
    target_net_update_freq=1000,
    tau=1.0,
    dyna_rollout_freq=250,
    initial_epsilon=0.01,
)
agent.train(total_timesteps=10_000 * 100)  # 10K episodes × 100 steps
```

---

## 3. Baselines

**A1 Conservative:** Default rate_embb=20, rate_urllc=15. Decrease if delay>0.7ms, increase if <0.3ms for 5 steps.

**A2 Aggressive:** Max rates (100/20). Decrease embb by 10 if delay>0.8ms.

**A3 Hysteresis:** Decrease embb by 10 if delay>0.8ms (cooldown 5 steps), increase by 5 if <0.4ms for 10 steps.

**Scalarized DQN:** 3 independent DQN, fixed ω ∈ {(0.2,0.7,0.1), (0.5,0.2,0.3), (0.1,0.3,0.6)}, 5000 eps each.

**Oracle DQN:** Per-ω dedicated DQN, 20 test ω × 5000 eps.

---

## 4. Parallel Orchestrator

3-layer: Coordinator → Workers (P) → ns-3 instances (P).
SF sharing every K=200 episodes.
Test P ∈ {1, 2, 4, 8, 16}.

---

## 5. Experiments

| Exp | What | Output |
|-----|------|--------|
| 1 | QoS Performance: PCF-MORL vs A1-A3 vs DQN on E1-E5, 3 ω | Tab III, Tab IV, Fig 2 |
| 2 | Zero-shot adaptation: GPI vs Oracle vs Retrain vs Fine-tune, 20 test ω | Fig 3, ZSR |
| 3 | 24h factory cycle: 8 phase transitions, cumulative J | Text paragraph |
| 4 | Parallel scaling: P={1,2,4,8,16}, wall-clock + HV | Fig 4 |
| 5 | K tuning: K={50,100,200,500,1000}, P=8 | Best K |
| 6 | SF sharing ablation: 4 strategies at P=8 | Ablation paragraph |

---

## 6. Paper Outputs

### Fig 2: Pareto Front (1-col, 3.5" × 2.5")
2D scatter: r₁ (throughput) vs r₂ (delay). Series: PCF-MORL, A1-A3, DQN, Sequential.

### Fig 3: Adaptation Curves (2-col, 7" × 2", 3 subplots)
X: episodes since ω switch. Y: scalarized return J(ω). 4 lines: Oracle, GPI, Fine-tune, Retrain.

### Fig 4: Speedup (1-col, 3.5" × 2.5")
X: P (workers). Y: Speedup. Measured + ideal + Amdahl fit.

### Tab III: QoS E2 (5 methods × 5 metrics)
VR↓, TTR↓, MVD↓, Throughput↑, Energy↓

### Tab IV: Pareto Quality (4 methods × 4 metrics)
HV↑, EU↑, MUL↓, |CCS|
