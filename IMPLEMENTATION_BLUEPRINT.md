# IMPLEMENTATION BLUEPRINT
# 5G Factory Slice Selection Simulator
# GLOBECOM CQRM 2026 — "QoS-Aware Hybrid Slice Selection for Private 5G Factory Networks"
# Dùng prompt này để implement từng module trong các session riêng.

---

## MỤC LỤC

1. [Tổng quan kiến trúc](#1-tổng-quan)
2. [Module 1: 5G-LENA ns-3](#2-module-1)
3. [Module 2: Python Agents + Training](#3-module-2)
4. [Module 3: React Dashboard](#4-module-3)
5. [Data Flow giữa các module](#5-data-flow)
6. [Phụ lục: Bảng tham số đầy đủ](#6-phụ-lục)

---

## 1. TỔNG QUAN KIẾN TRÚC

### 1.1 Ba module và vai trò

```
┌─────────────────────────────────────────────────────┐
│  Module 1: 5G-LENA (C++)                            │
│  Simulation engine thật, chạy trên ns-3             │
│  Output: PDCP traces, SlotStats, PRB utilization    │
├─────────────────────────────────────────────────────┤
│  Module 2: Python Agents + Environment              │
│  5 agents (B1-B5), reward, training loop            │
│  Có 2 mode:                                         │
│    (a) PythonSim: env giả lập nhanh bằng analytical │
│    (b) Ns3Bridge: đọc trace từ Module 1 qua IPC     │
├─────────────────────────────────────────────────────┤
│  Module 3: React Dashboard                          │
│  Giao diện trực quan: floor plan, biểu đồ, so sánh │
│  Đọc JSON results từ Module 2                       │
└─────────────────────────────────────────────────────┘
```

### 1.2 Thứ tự implement khuyến nghị

1. **Module 2 trước** — chạy được ngay bằng PythonSim, không cần ns-3
2. **Module 3** — dashboard đọc output JSON từ Module 2
3. **Module 1** — khi cần kết quả chính xác cho paper, cài ns-3 + 5G-LENA

### 1.3 Cây thư mục hoàn chỉnh

```
5g-factory-sim/
├── module1-5glena/
│   ├── configs/
│   │   └── factory_config.yaml          # Tham số nhà máy + mạng
│   ├── src/
│   │   ├── factory-sim-main.cc          # Chương trình ns-3 chính (120s scenario)
│   │   ├── nwdaf-monitor.h              # Layer 1: Event detection
│   │   ├── ursp-agent-bridge.h          # Layer 2: C++ ↔ Python IPC
│   │   └── traffic-manager.h            # UE activate/deactivate scheduler
│   └── tests/
│       ├── test-01-single-ue.cc         # Cơ bản: 1 gNB + 1 UE
│       ├── test-02-dual-bwp.cc          # 2 BWP (URLLC + eMBB)
│       ├── test-03-multi-ue.cc          # 31 UE, heterogeneous traffic
│       └── test-04-bwp-switch.cc        # Dynamic BWP switching
│
├── module2-agents/
│   ├── config.yaml                      # Tất cả hyperparameters
│   ├── environment.py                   # Gym-like env (PythonSim + Ns3Bridge)
│   ├── reward.py                        # 4-component reward function
│   ├── state.py                         # State normalization + discretization
│   ├── agents/
│   │   ├── base.py                      # Abstract interface
│   │   ├── b1_static.py
│   │   ├── b2_threshold_single.py
│   │   ├── b3_threshold_multi.py
│   │   ├── b4_linucb.py
│   │   └── b5_qlearning.py
│   ├── train.py                         # Training loop (ε-greedy, episodes)
│   ├── benchmark.py                     # Run all agents × all scenarios × 20 seeds
│   └── results/                         # Output JSON cho Module 3
│       └── benchmark_results.json
│
├── module3-dashboard/
│   └── App.jsx                          # Single-file React dashboard
│
└── README.md
```

---

## 2. MODULE 1: 5G-LENA ns-3

### 2.1 Mục đích
Chạy mô phỏng RAN chính xác: latency thật từ scheduler, fading, HARQ. Output PDCP traces cho evaluation trong paper.

### 2.2 Yêu cầu hệ thống
- ns-3.38+ với module 5G-LENA v2.x
- Ubuntu 20.04/22.04
- GCC 11+, CMake 3.16+

### 2.3 Test Programs (4 tests để làm quen)

#### Test 01: Single UE — Kiểm tra cài đặt
```
Mục đích: Verify ns-3 + 5G-LENA hoạt động.
Setup:
  - 1 gNB tại (0, 0, 6m)
  - 1 UE tại (10, 0, 1.5m)
  - 1 UDP DL flow: 250B mỗi 10ms
  - SCS = 15kHz (μ=0), 20MHz
  - Thời gian: 5s
Output: DlPdcpStats.txt → kiểm tra có delay + throughput
Kỳ vọng: Delay < 5ms, PDR > 99%
```

**Điểm cần chú ý khi code:**
- `NrHelper` → `SetEpcHelper` → `NrPointToPointEpcHelper`
- `CcBwpCreator::SimpleOperationBandConf` tạo band
- `nrHelper->InstallGnbDevice()` / `InstallUeDevice()`
- `nrHelper->EnableTraces()` bật PDCP + SlotStats output
- Dùng `UdpClientHelper` / `UdpServerHelper` cho traffic

#### Test 02: Dual BWP — Setup 2 slice
```
Mục đích: Tạo BWP0 (URLLC) + BWP1 (eMBB) trên cùng gNB.
Setup:
  - 1 gNB tại (50, 25, 6m) — tâm nhà máy
  - 2 UE: UE0 → BWP0, UE1 → BWP1
  - BWP0: 30MHz, μ=2 (SCS=60kHz) — freq 3.5GHz
  - BWP1: 70MHz, μ=0 (SCS=15kHz) — freq 3.535GHz
  - Channel: InH-FactoryDenseClutter (3GPP TR 38.901)
  - UE0: URLLC traffic (250B @ 4ms, UL)
  - UE1: eMBB traffic (1400B @ 33ms, DL)
  - Thời gian: 10s
Output: PDCP traces per BWP
Kỳ vọng:
  BWP0: delay thấp hơn (SCS cao → slot ngắn hơn)
  BWP1: throughput cao hơn (bandwidth rộng hơn)
```

**Điểm cần chú ý:**
- Tạo 2 `OperationBandInfo` riêng biệt, mỗi cái 1 BWP
- `nrHelper->InitializeOperationBand()` cho từng band
- `CcBwpCreator::GetAllBwps({band0, band1})` gộp lại
- BWP assignment qua `BwpManagerGnb::SetOutputLink(bearerType, bwpId)`

#### Test 03: Multi-UE Factory Traffic
```
Mục đích: Deploy 31 UE (subset) với traffic heterogeneous.
Setup:
  - 1 gNB ceiling center
  - 10 Robot Arm (URLLC, 250B @ 4ms UL)
  - 8 AGV Safety (URLLC, 150B @ 20ms UL)
  - 5 AR Headset (eMBB, 1500B @ 16.67ms DL)
  - 8 Laptop Video (eMBB, 1400B @ 33ms DL)
  - UE positions: zone-based random
    Robot:  x∈[10,40], y∈[10,20], z=1.0m
    AGV:    x∈[5,95],  y∈[20,30], z=0.5m
    AR:     x∈[60,90], y∈[35,45], z=1.7m
    Laptop: x∈[70,95], y∈[5,15],  z=1.0m
  - Thời gian: 15s
Output: Per-UE PDCP stats → group by class
Kỳ vọng: URLLC classes delay < PDB, eMBB classes throughput ≥ GBR
```

#### Test 04: BWP Switching — Dynamic slice reassignment
```
Mục đích: Demo SetBwpForBearer — hành động mà agent thực hiện.
Setup:
  - 3 UE: Robot (fixed URLLC), AGV (switches), Laptop (fixed eMBB)
  - Timeline:
    t=0-5s:  AGV trên BWP1 (eMBB)
    t=5s:    SWITCH AGV → BWP0 (URLLC)  ← simulated agent action
    t=5-10s: AGV trên BWP0
    t=10s:   SWITCH AGV → BWP1 (eMBB)
    t=10-15s: AGV trên BWP1 (restored)
  - Thời gian: 15s
Output: AGV delay across 3 phases
Kỳ vọng:
  Phase 1 (eMBB): AGV delay ~ 5-10ms
  Phase 2 (URLLC): AGV delay < 3ms (cải thiện rõ)
  Phase 3 (eMBB): AGV delay quay lại ~ 5-10ms
```

**Điểm cần chú ý:**
- `Simulator::Schedule(Seconds(5.0), &SwitchBwpCallback, ...)` để lên lịch switch
- Trong callback: `bwpManager->SetOutputLink(bearerType, newBwpId)`
- Đây chính xác là action mà RL agent sẽ gọi

### 2.4 Chương trình chính (factory-sim-main.cc)

```
Kiến trúc: 1 file .cc duy nhất, chạy 1 continuous run 120s.

Luồng chạy:
1. Load factory_config.yaml (dùng ns-3 ConfigStore hoặc hardcode)
2. Tạo topology: 1 gNB ceiling + 60 UE phân zone
3. Tạo 2 BWP (URLLC + eMBB)
4. Install UE devices, assign initial BWP per class
5. Install traffic generators per class (UdpClient/Server)
6. Khởi tạo NwdafMonitor (check mỗi 1s)
7. Khởi tạo UrspAgentBridge (IPC → Python agent)
8. Schedule scenario events (UE activate/deactivate per S1-S4)
9. Run simulation 120s
10. Output: PDCP traces, SlotStats, agent decision log

Components chính (mỗi cái 1 header file):

NwdafMonitor (nwdaf-monitor.h):
  - Timer callback mỗi 1s
  - Đọc SlotStats → tính ρ_s per BWP
  - Đọc PDCP delay → tính V_c per class
  - Track n_active per class → tính Δn
  - Track time since last decision → periodic fallback
  - Khi bất kỳ trigger fires → gọi agent callback
  - 4 trigger conditions:
    (1) ρ_s > 0.80 bất kỳ slice
    (2) V_c > 0.005 bất kỳ class
    (3) |Δn_active| ≥ 2 bất kỳ class
    (4) now - lastDecision ≥ 10s

UrspAgentBridge (ursp-agent-bridge.h):
  - Nhận trigger event từ NwdafMonitor
  - Compute normalized state s_t (10 dims)
  - Gửi state → Python agent qua named pipe / TCP socket / shared file
  - Nhận action a_t (5 ints, mỗi cái 0 hoặc 1)
  - Gọi SetBwpForBearer cho từng AI-controlled UE class
  - Log decision: {time, state, action, trigger_type}
  
  IPC options (chọn 1):
    (a) File-based: write state.json, read action.json (đơn giản nhất)
    (b) Named pipe: /tmp/ns3-agent-state, /tmp/ns3-agent-action
    (c) ZMQ socket: tcp://localhost:5555 (mạnh nhất)

TrafficManager (traffic-manager.h):
  - Schedule ActivateBearer / DeactivateBearer per scenario
  - S1: không event nào
  - S2: t=20s activate 15 smartphone + 2 AR, t=60s deactivate smartphone
  - S3: t=20s deactivate 3 robot + activate 3 AR, t=60s reverse
  - S4: mỗi 10s, random Δn per class (xem phân phối bên dưới)
```

### 2.5 factory_config.yaml (nội dung đầy đủ)

```yaml
factory:
  name: "InH-Factory-Reference"
  length_m: 100
  width_m: 50
  height_m: 10
  gnb_position: [50.0, 25.0, 6.0]
  gnb_tx_power_dbm: 30
  channel_model: "InH-FactoryDenseClutter"

slices:
  - name: URLLC
    bwp_id: 0
    bandwidth_mhz: 30
    numerology: 2            # SCS = 60kHz
    total_prbs: 24
    priority: 1.0
    designed_capacity_mbps: 50.0

  - name: eMBB
    bwp_id: 1
    bandwidth_mhz: 70
    numerology: 0            # SCS = 15kHz
    total_prbs: 375
    priority: 0.3
    designed_capacity_mbps: 200.0

traffic_classes:
  # [control, name, 5QI, PDB_ms, GBR_Mbps, rate/UE_Mbps, pkt_bytes, period_ms, dir, max_ues, weight, epsilon, default_slice]
  # === FIXED (always URLLC) ===
  - [fixed, "Robot Arm",     84,  5,   0.5, 0.5,  250,   4,    UL,   10, 1.0,  1e-4, URLLC]
  - [fixed, "PLC",           85,  10,  0.2, 0.2,  100,   10,   UL,   5,  0.8,  1e-4, URLLC]
  - [fixed, "Safety Sensor", 82,  5,   0.1, 0.1,  64,    1,    UL,   8,  1.0,  1e-4, URLLC]
  # === AI-CONTROLLED ===
  - [ai, "AGV Safety",       84,  10,  0.3, 0.3,  150,   20,   UL,   8,  1.0,  1e-4, URLLC]
  - [ai, "AGV Navigation",   83,  50,  2.0, 2.0,  500,   50,   BOTH, 8,  0.6,  1e-3, URLLC]
  - [ai, "AGV Telemetry",    6,   300, 1.0, 1.0,  1000,  100,  UL,   8,  0.2,  1e-2, eMBB]
  - [ai, "AR Headset",       80,  20,  15.0,15.0, 1500,  16.67,DL,   5,  0.8,  1e-3, eMBB]
  - [ai, "Laptop Video",     8,   150, 5.0, 5.0,  1400,  33.33,DL,   8,  0.1,  1e-2, eMBB]

nwdaf:
  check_interval_s: 1.0
  load_threshold: 0.80
  qos_violation_threshold: 0.005
  ue_change_threshold: 2
  periodic_fallback_s: 10.0

agent_hyperparams:
  alpha: 0.1
  alpha_decay: 0.995
  gamma: 0.9
  epsilon_start: 1.0
  epsilon_end: 0.05
  epsilon_decay: 0.99
  bins_per_dim: 3
  reward:
    beta: 0.3
    lambda: 2.0
    eta: 0.1
    kappa: 5.0
    rho_target: 0.65
```

---

## 3. MODULE 2: PYTHON AGENTS + TRAINING

### 3.1 Mục đích
- Implement 5 agents (B1-B5)
- Provide Gym-like environment (fast PythonSim + Ns3Bridge)
- Training loop cho Q-Learning + LinUCB
- Benchmark runner: all agents × all scenarios × 20 seeds → JSON

### 3.2 File: config.yaml
Dùng cùng file `factory_config.yaml` từ Module 1. Module 2 parse nó để lấy tham số.

### 3.3 File: state.py — State Normalization + Discretization

```python
"""
FactoryState: 10 dimensions, normalized.
Chịu trách nhiệm:
  1. Nhận raw metrics → normalize → state vector
  2. Discretize state vector → Q-table index

State dimensions (N=2):
  [0] ρ_URLLC     = usedRBs_BWP0 / totalRBs_BWP0           ∈ [0, 1]
  [1] ρ_eMBB      = usedRBs_BWP1 / totalRBs_BWP1           ∈ [0, 1]
  [2] L_URLLC     = actual_URLLC_traffic / designed_cap      ∈ [0, 1.5] clip
  [3] L_eMBB      = actual_eMBB_traffic / designed_cap       ∈ [0, 1.5] clip
  [4] H_delay     = max_c(mean_delay_c / PDB_c)             ∈ [0, 2] clip
  [5] H_tput      = min_c(throughput_c / GBR_c)             ∈ [0, 2] clip
  [6] U_ratio     = n_active_AI_UEs / n_total_AI_UEs        ∈ [0, 1]
  [7] M_ratio     = URLLC_traffic / total_traffic            ∈ [0, 1]
  [8] Δρ          = ρ_URLLC(t) - ρ_URLLC(t-1)               ∈ [-1, 1]
  [9] ΔU          = U_ratio(t) - U_ratio(t-1)               ∈ [-1, 1]

Discretization cho Q-table:
  - 3 bins per dim → flat index ∈ [0, 3^10 - 1] = [0, 59048]
  - Binning scheme per dim:
    dims 0-7 (range ~[0, 2]):  bin edges = [0, 0.5, 0.85, ∞]  → low/med/high
    dims 8-9 (range [-1, 1]):  bin edges = [-∞, -0.1, 0.1, ∞] → decreasing/stable/increasing
  - Flat index = Σ bin[i] * 3^i

Class/Methods:
  FactoryState:
    - __init__(vector: np.ndarray[10])
    - from_raw_metrics(slice_metrics, class_metrics, prev_state) → FactoryState
    - to_vector() → np.ndarray[10]
    - discretize(bins_per_dim=3) → int   # flat index cho Q-table
    - __repr__() → readable string
"""
```

### 3.4 File: reward.py — 4-Component Reward

```python
"""
r_t = R_URLLC + β·R_eMBB - λ·P_violation + η·R_efficiency

Inputs:
  - per_class_metrics: dict[class_name → {mean_delay_ms, throughput_mbps, violation_rate}]
  - rho_urllc: float (PRB utilization BWP0)
  - config: RewardConfig(β=0.3, λ=2.0, η=0.1, κ=5.0, ρ_target=0.65)

Implementation detail:

Component 1 — R_URLLC (asymmetric):
  for each URLLC class c (PDB ≤ 50ms):
    V_c = violation_rate
    if V_c ≤ ε_c:  φ_c = 1 - V_c          # SLA met, smooth
    else:           φ_c = -κ * V_c          # SLA violated, amplified
    R_URLLC += w_c * φ_c

Component 2 — R_eMBB (capped):
  for each eMBB class c (GBR ≥ 5 Mbps):
    ratio = min(throughput_c / GBR_c, 1.0)  # cap at 1.0
    R_eMBB += w_c * ratio

Component 3 — P_violation (binary spike):
  PDB_min = min(all PDB) = 5ms
  for each class c:
    if V_c > ε_c:
      P_violation += PDB_min / PDB_c        # strict classes penalized more

Component 4 — R_efficiency (tiebreaker):
  R_efficiency = 1 - |ρ_URLLC - 0.65|

Total clipped to [-20, +4].

Also export:
  compute_sla_compliance(metrics) → {Phi_URLLC, Phi_eMBB}
  is_stable(history, window=3) → bool
    all last W steps: Phi_URLLC ≥ 0.95 and Phi_eMBB ≥ 0.85
"""
```

### 3.5 File: environment.py — PythonSim (Analytical Approximation)

```python
"""
FactoryEnv: Gym-like interface.
  reset() → FactoryState
  step(action: list[5 ints]) → (next_state, reward, done, info)

PythonSim uses analytical models to approximate RAN behavior:

1. PRB Utilization (ρ_s):
   traffic_on_slice_s = Σ (rate_per_ue * n_active_ues) for all classes assigned to slice s
   ρ_s = traffic_on_slice_s / designed_capacity_s
   Capped at 1.0 (overload → packets queued/dropped)

2. Delay Model (per class):
   base_delay_ms = numerology-dependent:
     μ=0: 2.0ms, μ=1: 1.0ms, μ=2: 0.5ms
   load_factor = 1 / (1 - min(ρ_s, 0.95))     # M/D/1 approximation
   mean_delay = base_delay_ms * load_factor + propagation_delay(0.1ms)
   noise = np.random.normal(0, 0.1 * mean_delay)
   
   violation_rate = Pr(delay > PDB):
     if mean_delay < PDB * 0.5:   V_c ≈ 0.0001 (minimal)
     if mean_delay < PDB * 0.8:   V_c ≈ 0.001
     if mean_delay < PDB:         V_c ≈ 0.01
     if mean_delay ≥ PDB:         V_c ≈ 0.1 + 0.5*(mean_delay/PDB - 1)

3. Throughput Model:
   if ρ_s < 1.0:
     throughput_c = rate_per_ue * n_active  (full rate)
   else:
     throughput_c = rate_per_ue * n_active * (1.0 / ρ_s)  (shared proportionally)

4. Scenario Events:
   events = list of {time, action: activate/deactivate, class, count}
   Processed at each step when time matches.
   
   S4 random: every 10 steps, for each AI class:
     delta_n = random choice from range, with p_zero probability of 0
     n_active = clip(n_active + delta_n, 0, max_ues)

5. NWDAF Trigger Logic:
   Mỗi step (1s), check 4 conditions → fire trigger → agent decides.
   Nếu không trigger → agent không được gọi, giữ action cũ.
   
   QUAN TRỌNG: Environment phải track "trigger_fired" flag.
   Training loop chỉ gọi agent.select_action() khi trigger fires.
   Khi không trigger: action = previous action, env vẫn advance.

Interface:
  env = FactoryEnv(config_path, scenario="S2")
  state = env.reset()
  while not done:
    trigger_fired = env.check_triggers()
    if trigger_fired:
      action = agent.select_action(state)
      next_state, reward, done, info = env.step(action)
      agent.update(state, action, reward, next_state)
      state = next_state
    else:
      env.advance()  # time += dt, process events, no agent decision
"""
```

### 3.6 Agents (agents/ directory)

#### base.py — Interface

```python
"""
class BaseAgent(ABC):
    name: str
    total_decisions: int = 0
    policy_changes: int = 0
    _last_action: list[int] | None = None

    @abstractmethod
    select_action(state: FactoryState) → list[int]  # 5 ints, each 0 or 1

    @abstractmethod
    update(state, action, reward, next_state) → None

    on_trigger(state) → list[int]:
      action = self.select_action(state)
      self.total_decisions += 1
      if action != self._last_action: self.policy_changes += 1
      self._last_action = action
      return action

    policy_change_rate() → float: policy_changes / total_decisions
    reset() → None
    save(path) / load(path) → None
"""
```

#### b1_static.py

```python
"""
B1 — Static: Ignore trigger, return default assignment.
Default: [0, 0, 1, 1, 1]
  AGV safety → URLLC, AGV nav → URLLC, AGV telem → eMBB, AR → eMBB, Laptop → eMBB

select_action(state): return [0, 0, 1, 1, 1]
update(): pass
"""
```

#### b2_threshold_single.py

```python
"""
B2 — Threshold-Single: 1 variable, 2 rules.
State tracked: _offloaded (bool)

select_action(state):
  if ρ_URLLC > 0.85 and not offloaded:
    offloaded = True
    return [0, 1, 1, 1, 1]        # AGV nav → eMBB (offload)
  elif ρ_URLLC < 0.60 and offloaded:
    offloaded = False
    return [0, 0, 1, 1, 1]        # AGV nav → URLLC (restore)
  else:
    return current assignment

Failure mode: khi ρ oscillates near 0.85 → flip-flop mỗi step.
"""
```

#### b3_threshold_multi.py

```python
"""
B3 — Threshold-Multi: 4 rules, priority ordering.
Track: current_assignment = [0, 0, 1, 1, 1]

select_action(state):
  a = copy(current_assignment)
  
  # Rule 1 (highest): URLLC overload protection
  if ρ_URLLC > 0.85:
    a[1] = 1  # AGV nav → eMBB
    a[2] = 1  # AGV telem → eMBB (stay)
  
  # Rule 2: eMBB congestion + high delay → move AR to URLLC
  elif ρ_eMBB > 0.85 and H_delay > 0.7:
    a[3] = 0  # AR → URLLC
  
  # Rule 3: URLLC spare → restore
  if ρ_URLLC < 0.60:
    a[1] = 0  # AGV nav → URLLC
  
  # Rule 4: eMBB spare → restore
  if ρ_eMBB < 0.60:
    a[3] = 1  # AR → eMBB

  current_assignment = a
  return a

Failure mode: Rule 1 fires → offload → ρ_URLLC drops → Rule 3 fires → restore → 
ρ_URLLC rises → Rule 1 fires → oscillation. Rules 1&2 can also conflict.
"""
```

#### b4_linucb.py

```python
"""
B4 — LinUCB: Contextual bandit. ML-based but STATELESS (no γ).
Same 10-dim state. 32 arms (2^5 joint actions).

Per-arm parameters:
  A[a] = identity matrix (10×10)   # precision
  b[a] = zero vector (10,)         # reward-weighted features

select_action(state):
  x = state.to_vector()  # 10-dim
  for each action a in 0..31:
    θ_a = inv(A[a]) @ b[a]
    UCB_a = θ_a @ x + α_ucb * sqrt(x @ inv(A[a]) @ x)
  return argmax UCB_a  → decode to 5-bit assignment

update(state, action, reward, next_state):
  a = action_to_index(action)  # 0..31
  x = state.to_vector()
  A[a] += outer(x, x)
  b[a] += reward * x
  # NOTE: next_state IGNORED — this is a bandit, no temporal reasoning

α_ucb = 1.0 (exploration parameter)

Key ablation point: B4 vs B5 proves value of γ=0.9 temporal reasoning.
B4 sees same state, same actions, same reward — but cannot plan ahead.
"""
```

#### b5_qlearning.py

```python
"""
B5 — Q-Learning (PROPOSED METHOD): Full MDP, γ=0.9.

Q-table options:
  (a) Joint: 1 table, |S|×32  (59K × 32 = 1.9M entries)
  (b) Factored: 5 tables, each |S|×2  (59K × 2 × 5 = 590K entries)
  DEFAULT: Factored (converges ~6x faster)

Hyperparameters:
  α = 0.1, decay 0.995/episode
  γ = 0.9
  ε = 1.0 → 0.05, decay 0.99/episode
  Q_init = 0.0 (or optimistic +5)
  bins_per_dim = 3

select_action(state):
  s = state.discretize()  # flat index
  if random() < ε:  # explore
    return [randint(0,1) for _ in range(5)]
  else:  # exploit (factored)
    return [argmax(Q[c][s, :]) for c in range(5)]

update(state, action, reward, next_state):
  s = state.discretize()
  s' = next_state.discretize()
  for c in range(5):  # factored update
    a_c = action[c]
    Q[c][s, a_c] += α * (reward + γ * max(Q[c][s', :]) - Q[c][s, a_c])

end_episode():
  ε *= ε_decay (min ε_end)
  α *= α_decay (min 0.01)

Training protocol:
  Phase 1: 500 episodes on PythonSim, full exploration
  Phase 2: Deploy converged Q-table (ε=0.05), evaluate on all scenarios

Save/Load: JSON file with Q-tables + hyperparams
"""
```

### 3.7 File: train.py — Training Loop

```python
"""
Usage: python train.py --agent b5_qlearning --scenario S2 --episodes 500

Training loop:
  agent = create_agent(agent_name)
  env = FactoryEnv(config_path, scenario)
  
  for episode in range(n_episodes):
    state = env.reset()
    total_reward = 0
    
    while not done:
      trigger = env.check_triggers()
      if trigger:
        action = agent.on_trigger(state)
        next_state, reward, done, info = env.step(action)
        agent.update(state, action, reward, next_state)
        state = next_state
        total_reward += reward
      else:
        env.advance()
    
    agent.end_episode()  # decay ε, α
    log(episode, total_reward, agent.get_stats())
  
  agent.save(f"models/{agent_name}_{scenario}.json")

Output: trained model + training_log.csv
"""
```

### 3.8 File: benchmark.py — Full Evaluation

```python
"""
Usage: python benchmark.py --episodes 500 --eval-seeds 20

Workflow:
  1. Train B5 (Q-Learning) on scenario mix: 500 episodes
  2. For each scenario S1-S4:
     For each agent B1-B5:
       For each seed 1-20:
         Run 1 episode (ε=0.05 for B5, ε=0 for B4)
         Record: Phi_URLLC, Phi_eMBB, V_c per class, T_stable, F_change,
                 rho_per_slice, reward_total, action_history, state_history

  3. Compute statistics:
     Per (agent, scenario): mean ± 95% CI for each metric
     Paired t-test: B5 vs each baseline (same seeds)
     Cohen's d effect size

  4. Output: benchmark_results.json
     {
       "metadata": {date, config, n_seeds, ...},
       "results": {
         "B1_static": {
           "S1": {"Phi_URLLC": {mean, ci95, std}, ...},
           "S2": {...}, "S3": {...}, "S4": {...}
         },
         ...
       },
       "pairwise_tests": {
         "B5_vs_B1": {"S2": {t_stat, p_value, cohens_d}, ...},
         ...
       },
       "timeseries": {
         "S2": {
           "B1": [{time, Phi_URLLC, rho_urllc, action}, ...],  # per step
           ...
         }
       }
     }
"""
```

---

## 4. MODULE 3: REACT DASHBOARD

### 4.1 Mục đích
Single-file React app (.jsx) hiển thị:
- Sơ đồ sàn nhà máy (100×50m) với vị trí gNB, UE theo zone
- Traffic + QoS metrics real-time (hoặc replay từ benchmark data)
- So sánh 5 agents trên 4 scenarios
- Agent selection (dropdown: B1-B5)
- Slice allocation visualization

### 4.2 Layout (Single Page, 3 Panels)

```
┌──────────────────────────────────────────────────────────┐
│  HEADER: "5G Factory Slice Selection Simulator"          │
│  [Agent: ▼ B5 Q-Learning] [Scenario: ▼ S2] [▶ Play]    │
├────────────────────┬─────────────────────────────────────┤
│                    │  RIGHT PANEL (Charts)               │
│  LEFT PANEL        │  ┌─────────────────────────────┐    │
│  Factory Floor     │  │ Φ_URLLC Time Series         │    │
│  (SVG 100×50)      │  │ (5 lines, 1 per agent)      │    │
│                    │  └─────────────────────────────┘    │
│  ● gNB (center)   │  ┌──────────┬──────────────────┐    │
│  ▲ Robot (red)     │  │ PRB Util │ Policy Changes   │    │
│  ■ AGV (blue)      │  │ (stacked │ (bar chart)      │    │
│  ◆ AR (green)      │  │  bar)    │                  │    │
│  ○ Laptop (gray)   │  └──────────┴──────────────────┘    │
│                    │  ┌─────────────────────────────┐    │
│  UE colors by      │  │ SLA Compliance Table        │    │
│  current slice:    │  │ (agents × scenarios)        │    │
│  Blue=URLLC        │  └─────────────────────────────┘    │
│  Orange=eMBB       │                                     │
│                    │                                     │
├────────────────────┴─────────────────────────────────────┤
│  BOTTOM: Slice Allocation Strip                          │
│  [AGV Safe: URLLC] [AGV Nav: eMBB→URLLC] [AR: eMBB] ... │
│  Current time: t=23.0s  |  Trigger: LOAD_LEVEL (ρ=0.83) │
└──────────────────────────────────────────────────────────┘
```

### 4.3 Data Source

Dashboard có 2 mode:

**(a) Demo mode (mặc định):**
Dùng hardcoded simulated data gần với expected results trong paper.
Không cần backend. Chạy ngay trong browser.

Data structure (embed trong JSX):

```javascript
const DEMO_DATA = {
  factory: {
    width: 100, height: 50,
    gnb: { x: 50, y: 25, z: 6 },
    ue_zones: {
      "Robot Arm":     { xRange: [10,40], yRange: [10,20], count: 10, color: "#ef4444", icon: "▲" },
      "PLC":           { xRange: [15,35], yRange: [22,28], count: 5,  color: "#f97316", icon: "●" },
      "Safety Sensor": { xRange: [5,45],  yRange: [8,22],  count: 8,  color: "#eab308", icon: "◆" },
      "AGV Safety":    { xRange: [5,95],  yRange: [20,30], count: 8,  color: "#3b82f6", icon: "■" },
      "AGV Navigation": { xRange: [5,95], yRange: [22,28], count: 8,  color: "#6366f1", icon: "■" },
      "AGV Telemetry": { xRange: [5,95],  yRange: [24,30], count: 8,  color: "#8b5cf6", icon: "■" },
      "AR Headset":    { xRange: [60,90], yRange: [35,45], count: 5,  color: "#22c55e", icon: "◆" },
      "Laptop Video":  { xRange: [70,95], yRange: [5,15],  count: 8,  color: "#6b7280", icon: "○" },
    }
  },
  // Timeseries: pre-generated per (agent, scenario)
  timeseries: {
    S2: {
      B1_static:    generateStaticTimeseries(...),
      B2_threshold: generateThresholdTimeseries(...),
      ...
      B5_qlearning: generateQLearningTimeseries(...)
    }
  },
  // Summary table
  sla_table: {
    //         S1    S2    S3    S4
    B1_static: [0.95, 0.78, 0.82, 0.80],
    B2_thresh: [0.95, 0.83, 0.84, 0.85],
    B3_multi:  [0.95, 0.87, 0.86, 0.89],
    B4_linucb: [0.95, 0.90, 0.89, 0.91],
    B5_qlearn: [0.95, 0.94, 0.93, 0.94],
  }
};
```

**(b) File mode:**
Load `benchmark_results.json` từ Module 2 (drag & drop hoặc file input).

### 4.4 Components (React, Tailwind, Recharts)

```
App
├── Header (agent selector, scenario selector, play/pause)
├── FactoryFloorPlan (SVG)
│   ├── GridBackground (10m grid)
│   ├── GnbMarker (antenna icon at center)
│   ├── UeCluster (per traffic class, positioned by zone)
│   │   └── UeMarker (circle/square, color = current slice)
│   └── Legend (class → icon mapping)
├── MetricsPanel
│   ├── SlaTimeSeriesChart (Recharts LineChart)
│   │   Lines: Φ_URLLC over time, 1 per agent
│   │   Vertical dashed line at event time (t=20s, t=60s)
│   ├── PrbUtilChart (Recharts BarChart)
│   │   Stacked: URLLC used, eMBB used, unused
│   ├── PolicyChangeChart (Recharts BarChart)
│   │   Bars: F_change per agent
│   └── SlaComplianceTable (HTML table)
│       Rows: agents, Cols: scenarios, Cell: Φ_URLLC value
│       Color coding: green ≥ 0.93, yellow ≥ 0.85, red < 0.85
├── SliceAllocationStrip
│   Per AI class: card showing current_slice, with transition animation
│   Timestamp + trigger info
└── PlaybackController
    Slider: t ∈ [0, 120s]
    Play/Pause button
    Speed: 1x, 2x, 5x
```

### 4.5 Key Interactions

1. **Agent dropdown** → switch timeseries data, update floor plan colors
2. **Scenario dropdown** → reload timeseries, update event markers
3. **Play** → animate t from 0→120s, update all charts per step
4. **Hover UE** → tooltip: class, current slice, delay, throughput
5. **Click chart point** → jump to that time, update floor plan

### 4.6 Styling Direction
- Dark industrial theme (phù hợp factory context)
- Font: monospace cho số, sans-serif cho label
- Color palette:
  URLLC = blue (#3b82f6), eMBB = amber (#f59e0b)
  Compliant = green, Warning = yellow, Violation = red
  Background = slate-900, Panel = slate-800
- Grid lines mờ trên floor plan (mô phỏng factory grid)

### 4.7 Libraries (available in Claude artifacts)
- React (hooks: useState, useEffect, useMemo, useCallback, useRef)
- Recharts (LineChart, BarChart, AreaChart, ResponsiveContainer)
- Tailwind CSS (utility classes)
- lucide-react (icons: Play, Pause, Settings, Radio, Wifi, etc.)

---

## 5. DATA FLOW GIỮA CÁC MODULE

```
Module 1 (ns-3)                    Module 2 (Python)               Module 3 (React)
─────────────                      ─────────────────               ────────────────
                                   
[Chạy ns-3 sim]                    
  │                                
  ├─→ DlPdcpStats.txt ──────────→ [trace_parser.py]
  ├─→ UlPdcpStats.txt ──────────→   parses per-class delay,
  ├─→ SlotStats.txt ─────────────→   throughput, PRB util
  └─→ decision_log.json                    │
                                           ▼
                                   [environment.py]
                                     Ns3Bridge mode: đọc traces
                                     PythonSim mode: analytical
                                           │
                                           ▼
                                   [train.py]
                                     Train B5, evaluate B1-B5
                                           │
                                           ▼
                                   [benchmark.py]
                                     20 seeds × 4 scenarios × 5 agents
                                           │
                                           ▼
                                   benchmark_results.json ────────→ [App.jsx]
                                                                     Load JSON
                                                                     Visualize
```

### Workflow thực tế:

**Bước 1 (nhanh, không cần ns-3):**
- Implement Module 2 với PythonSim
- Chạy `python train.py` → train Q-Learning
- Chạy `python benchmark.py` → benchmark_results.json
- Implement Module 3 → dashboard đọc JSON

**Bước 2 (cho paper, cần ns-3):**
- Cài ns-3 + 5G-LENA
- Chạy 4 tests để làm quen
- Implement factory-sim-main.cc
- Chạy sim, agent quyết định qua IPC
- Parse traces → chạy lại benchmark với data thật

---

## 6. PHỤ LỤC: BẢNG THAM SỐ ĐẦY ĐỦ

### A. Traffic Classes (đầy đủ 8 class)

| # | Class | Control | 5QI | PDB (ms) | GBR (Mbps) | Rate/UE | Pkt (B) | Period (ms) | Dir | Max UE | w_c | ε_c | Default |
|---|-------|---------|-----|----------|------------|---------|---------|-------------|-----|--------|-----|-----|---------|
| 1 | Robot Arm | fixed | 84 | 5 | 0.5 | 0.5 | 250 | 4 | UL | 10 | 1.0 | 10⁻⁴ | URLLC |
| 2 | PLC | fixed | 85 | 10 | 0.2 | 0.2 | 100 | 10 | UL | 5 | 0.8 | 10⁻⁴ | URLLC |
| 3 | Safety Sensor | fixed | 82 | 5 | 0.1 | 0.1 | 64 | 1 | UL | 8 | 1.0 | 10⁻⁴ | URLLC |
| 4 | AGV Safety | **ai** | 84 | 10 | 0.3 | 0.3 | 150 | 20 | UL | 8 | 1.0 | 10⁻⁴ | URLLC |
| 5 | AGV Navigation | **ai** | 83 | 50 | 2.0 | 2.0 | 500 | 50 | BOTH | 8 | 0.6 | 10⁻³ | URLLC |
| 6 | AGV Telemetry | **ai** | 6 | 300 | 1.0 | 1.0 | 1000 | 100 | UL | 8 | 0.2 | 10⁻² | eMBB |
| 7 | AR Headset | **ai** | 80 | 20 | 15.0 | 15.0 | 1500 | 16.67 | DL | 5 | 0.8 | 10⁻³ | eMBB |
| 8 | Laptop Video | **ai** | 8 | 150 | 5.0 | 5.0 | 1400 | 33.33 | DL | 8 | 0.1 | 10⁻² | eMBB |

### B. Scenario Events

**S1 (60s):** No events.

**S2 (120s):**
| Time | Event |
|------|-------|
| t=20s | +15 smartphones activate (eMBB DL burst) + 2 AR activate |
| t=60s | 15 smartphones deactivate |

**S3 (120s):**
| Time | Event |
|------|-------|
| t=20s | 3 Robot Arms deactivate + 3 AR Headsets activate |
| t=60s | 3 Robot Arms reactivate + 3 AR Headsets deactivate |

**S4 (120s):** Every 10s, random Δn per AI class:
| Class | Δn range | P(0) |
|-------|----------|------|
| AGV Safety | {-1, 0, +1} | 0.7 |
| AGV Nav/Telem | {-2, ..., +2} | 0.5 |
| AR/Laptop | {-3, ..., +3} | 0.3 |
Bounded: n_active ∈ [0, max_ues]

### C. UE Positions (Factory Zones)

| Zone | Classes | x range | y range | z (height) |
|------|---------|---------|---------|------------|
| Production Line | Robot Arm, PLC | 10-40 | 10-20 | 1.0m |
| Safety Perimeter | Safety Sensor | 5-45 | 8-22 | 0.3m |
| Floor Corridor | AGV (all) | 5-95 | 20-30 | 0.5m |
| Maintenance Bay | AR Headset | 60-90 | 35-45 | 1.7m |
| Control Office | Laptop Video | 70-95 | 5-15 | 1.0m |

### D. Expected Results (target cho validation)

**Φ_URLLC (SLA Compliance):**
|          | S1 | S2 | S3 | S4 |
|----------|-----|-----|-----|-----|
| B1 Static | 0.95 | 0.78 | 0.82 | 0.80 |
| B2 Thresh-S | 0.95 | 0.83 | 0.84 | 0.85 |
| B3 Thresh-M | 0.95 | 0.87 | 0.86 | 0.89 |
| B4 LinUCB | 0.95 | 0.90 | 0.89 | 0.91 |
| B5 Q-Learn | 0.95 | 0.94 | 0.93 | 0.94 |

**Convergence (S2):**
| Agent | Steps to Stable | Policy Changes |
|-------|-----------------|----------------|
| B1 | ∞ | 0 |
| B2 | 6-8 | 8-12 |
| B3 | 5-7 | 6-10 |
| B4 | 4-5 | 4-6 |
| B5 | 2-3 | 2-3 |

### E. Q-Learning Hyperparameters

| Param | Value | Description |
|-------|-------|-------------|
| α (learning rate) | 0.1, decay ×0.995/ep | |
| γ (discount) | 0.9 | ~10 step lookahead |
| ε (exploration) | 1.0 → 0.05, decay ×0.99/ep | |
| Q_init | 0.0 | |
| bins_per_dim | 3 | |S| = 3¹⁰ ≈ 59K |
| Factored tables | Yes (5 tables × |S|×2) | |

### F. Reward Hyperparameters

| Param | Value | Role |
|-------|-------|------|
| β | 0.3 | eMBB importance relative to URLLC |
| λ | 2.0 | Violation penalty multiplier |
| η | 0.1 | Efficiency tiebreaker weight |
| κ | 5.0 | Asymmetric violation amplification |
| ρ_target | 0.65 | Target URLLC utilization for efficiency |

### G. NWDAF Trigger Thresholds

| Condition | Threshold | Check Interval |
|-----------|-----------|----------------|
| Load level (ρ_s) | > 0.80 | 1s |
| QoS violation (V_c) | > 0.005 | 1s |
| UE change (|Δn|) | ≥ 2 | 1s |
| Periodic fallback | 10s since last decision | 1s |

---

## CÁCH DÙNG TÀI LIỆU NÀY

### Khi implement Module 2 (Python):
Copy sections 3.3 → 3.8 + Phụ lục A-G vào prompt. Yêu cầu:
"Implement Module 2 theo thiết kế này. Tạo các file Python hoàn chỉnh, chạy được. 
Bắt đầu từ state.py → reward.py → environment.py → agents → train.py → benchmark.py"

### Khi implement Module 3 (React):
Copy section 4 + Phụ lục D-E vào prompt. Yêu cầu:
"Tạo single-file React dashboard (App.jsx) theo layout này. 
Dùng demo mode với hardcoded data. Libraries: React, Recharts, Tailwind, lucide-react."

### Khi implement Module 1 (ns-3):
Copy sections 2.3 → 2.5 + Phụ lục A-C vào prompt. Yêu cầu:
"Implement test-01-single-ue.cc cho ns-3 5G-LENA theo spec này."
Làm từng test một, verify trước khi tiếp test sau.
