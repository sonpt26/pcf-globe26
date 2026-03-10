# PCF-MORL Implementation Plan

## Module Dependencies
M1 → M2 → M3 → M4 → M5

---

## M1: ns-3 5G-LENA Environment

### M1.1: ns-3 C++ Scenario (`sim/pcf-morl-scenario.cc`)
1. Set up 1 gNB at (0,0,10), 14 UEs in factory layout
2. Configure 1 CC with 2 BWPs:
   - BWP 0 (URLLC): 15 MHz, 41 PRBs, PF scheduler, 3.5 GHz
   - BWP 1 (eMBB): 25 MHz, 65 PRBs, PF scheduler, 3.5 GHz
3. Channel: InF-DH (3GPP TS 38.901)
4. EPC setup with TbfQueueDisc per-UE on PGW→gNB link
5. Traffic generators:
   - URLLC: UdpClient (periodic 256B/ms) + OnOffApplication (burst)
   - eMBB: UdpClient (25 Mbps CBR) + ThreeGppFtpM2 (32 Mbps mean)
6. KPI collection: PDCP delay, throughput, PRB utilization, energy
7. ns3-gym ZMQ interface for Python communication

### M1.2: Python MO-Gymnasium Wrapper (`env/pcf_morl_env.py`)
1. Extend ns3-gym for vector reward (3-dim)
2. Implement action decoding: int → (rate_urllc, rate_embb)
3. State normalization [0,1]
4. Episode management: 100 steps × 500ms

### M1.3: Validation
- [ ] 1 episode runs 100 steps without crash
- [ ] State shape (12,), reward shape (3,), action discrete(76)
- [ ] Sweep rate_embb fixed, plot r₁ vs rate → monotonic up to capacity cap
- [ ] Sweep rate_urllc fixed, plot r₂ vs rate → knee at burst threshold
- [ ] Change rate mid-episode → KPIs respond within 2-3 steps

---

## M2: MORL Training

### M2.1: GPI-PD Agent (`training/gpi_pd_agent.py`)
- morl-baselines GPIPD wrapper
- Checkpoint every 1000 episodes
- JSON logging per episode

### M2.2: Baselines (`baselines/`)
- threshold_baselines.py: A1 Conservative, A2 Aggressive, A3 Hysteresis
- scalarized_dqn.py: 3 fixed-ω DQN via stable-baselines3
- oracle_dqn.py: Per-ω dedicated DQN

---

## M3: Parallel Orchestrator (`parallel/`)
- coordinator.py: CCS + GPI-LS weight selection
- worker.py: GPI-PD agent + MO-Gym wrapper
- ns-3 instance management with unique ZMQ ports

---

## M4: Experiments (`experiments/`)
- experiment_runner.py: Config-driven experiment execution
- metrics.py: VR, TTR, MVD, HV, EU, MUL, ZSR, TTM
- 5 seeds per config, 95% CI, Wilcoxon test

---

## M5: Paper Outputs (`paper/`)
- figures.py: Fig 2, 3, 4 generation
- tables.py: Tab III, IV LaTeX snippets
- main.tex: Paper draft with \input{} for tables

---

## Technology Stack

| Component | Library | Version |
|-----------|---------|---------|
| Simulation | ns-3 + 5G-LENA NR | ns-3.46 + NR v4.1.1 |
| RL bridge | ns3-gym (fork, vector reward) | custom |
| MORL | morl-baselines (GPI-PD) | latest |
| RL baselines | stable-baselines3 (DQN) | latest |
| MO metrics | pymoo (HV) | latest |
| Parallelism | multiprocessing + ZeroMQ | - |
| Figures | matplotlib + seaborn | - |
| Stats | scipy.stats | - |
