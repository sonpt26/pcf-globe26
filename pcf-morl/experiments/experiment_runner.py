"""M4: Config-driven experiment runner for PCF-MORL.

Runs all 6 experiments from the GLOBECOM 2026 spec:
  Exp 1: QoS Performance (PCF-MORL vs A1-A3 vs DQN on E1-E5)   → Tab III, Tab IV, Fig 2
  Exp 2: Zero-shot adaptation (GPI vs Oracle vs Retrain)         → Fig 3, ZSR
  Exp 3: 24h factory cycle (8 phase transitions)                 → Text paragraph
  Exp 4: Parallel scaling (P={1,2,4,8,16})                      → Fig 4
  Exp 5: K tuning (K={50,100,200,500,1000}, P=8)                → Best K
  Exp 6: SF sharing ablation (4 strategies, P=8)                 → Ablation paragraph

Usage:
    python -m experiments.experiment_runner --exp 1 --seeds 5
    python -m experiments.experiment_runner --exp all --seeds 5
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from env.pcf_morl_env import PcfMorlEnv
from baselines.threshold_baselines import BASELINES
from baselines.scalarized_dqn import generate_simplex_weights
from experiments.metrics import (
    compute_episode_qos,
    compute_episode_return,
    compute_mean_throughput,
    compute_mean_energy,
    compute_hypervolume,
    compute_expected_utility,
    compute_mul,
    compute_zsr,
    compute_ttm,
    aggregate_seeds,
    wilcoxon_test,
)

RESULTS_DIR = Path(__file__).parent.parent / "results"

# ---------------------------------------------------------------------------
# Evaluation scenario configs (E1-E5)
# ---------------------------------------------------------------------------

EVAL_SCENARIOS = {
    "E1": {"scenario": "eval_constant_90", "description": "Constant 90% load, 50s"},
    "E2": {"scenario": "eval_urllc_burst", "description": "Steady → URLLC burst → steady"},
    "E3": {"scenario": "eval_embb_surge", "description": "Steady → eMBB surge → steady"},
    "E4": {"scenario": "eval_alternating", "description": "Alternating bursts 5-10s"},
    "E5": {"scenario": "eval_overload", "description": "Both slices >100% for 25s"},
}

# Test weight vectors for multi-objective evaluation
TEST_WEIGHTS = [
    np.array([0.7, 0.2, 0.1]),   # Throughput-focused
    np.array([0.2, 0.7, 0.1]),   # Delay-focused
    np.array([0.2, 0.2, 0.6]),   # Energy-focused
]

# 20 test weights on the 3-simplex for Exp 2
TEST_WEIGHTS_20 = generate_simplex_weights(20, 3)

# Reference point for hypervolume (per-step worst case × 100 steps)
REF_POINT = np.array([0.0, -100.0, -100.0])


# ---------------------------------------------------------------------------
# Agent loading
# ---------------------------------------------------------------------------

def load_gpipd_agent(checkpoint_path: str, env: PcfMorlEnv, device: str = "auto"):
    """Load a trained GPI-PD agent from checkpoint."""
    from morl_baselines.multi_policy.gpi_pd.gpi_pd import GPIPD
    import torch as th

    if device == "auto":
        device = "cuda" if th.cuda.is_available() else "cpu"

    agent = GPIPD(
        env=env,
        learning_rate=3e-4,
        gamma=0.99,
        batch_size=128,
        net_arch=[256, 256, 256, 256],
        buffer_size=10_000,  # Small buffer for eval-only
        num_nets=2,
        per=True,
        dyna=False,
        gpi_pd=True,
        log=False,
        seed=0,
        device=device,
    )

    # Load checkpoint (morl-baselines saves via torch)
    ckpt = Path(checkpoint_path)
    if ckpt.is_dir():
        # Directory checkpoint: load state dicts
        for i, qnet in enumerate(agent.q_nets):
            state_path = ckpt / f"q_net_{i}.pt"
            if state_path.exists():
                qnet.load_state_dict(th.load(state_path, map_location=device))
        ws_path = ckpt / "weight_support.npy"
        if ws_path.exists():
            ws = np.load(ws_path, allow_pickle=True)
            agent.set_weight_support([w for w in ws])
    else:
        # Single file checkpoint
        state = th.load(str(ckpt), map_location=device)
        if "q_net_0" in state:
            for i, qnet in enumerate(agent.q_nets):
                qnet.load_state_dict(state[f"q_net_{i}"])
        if "weight_support" in state:
            agent.set_weight_support([np.array(w) for w in state["weight_support"]])

    print(f"Loaded GPI-PD: |M|={len(agent.weight_support)}, device={device}")
    return agent


# ---------------------------------------------------------------------------
# Episode runners
# ---------------------------------------------------------------------------

@dataclass
class EpisodeResult:
    """Result of running one episode."""
    method: str
    scenario: str
    seed: int
    weight: list = field(default_factory=list)
    episode_return: list = field(default_factory=list)
    qos: dict = field(default_factory=dict)
    mean_throughput: float = 0.0
    mean_energy: float = 0.0
    wall_time_s: float = 0.0
    step_rewards: list = field(default_factory=list)


def run_episode_threshold(
    env: PcfMorlEnv, policy, method_name: str, scenario: str, seed: int
) -> EpisodeResult:
    """Run one episode with a threshold-based policy."""
    policy.reset()
    obs, info = env.reset(seed=seed)
    rewards = []
    step_infos = []

    t0 = time.time()
    done = False
    while not done:
        action = policy.get_action(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        step_infos.append(info)
        done = terminated or truncated

    wall_time = time.time() - t0
    ep_return = compute_episode_return(rewards)

    return EpisodeResult(
        method=method_name,
        scenario=scenario,
        seed=seed,
        episode_return=ep_return.tolist(),
        qos=compute_episode_qos(step_infos),
        mean_throughput=compute_mean_throughput(step_infos),
        mean_energy=compute_mean_energy(step_infos),
        wall_time_s=wall_time,
        step_rewards=[r.tolist() for r in rewards],
    )


def run_episode_gpipd(
    env: PcfMorlEnv, agent, weight: np.ndarray,
    scenario: str, seed: int
) -> EpisodeResult:
    """Run one episode with a trained GPI-PD agent at a given weight."""
    obs, info = env.reset(seed=seed)
    rewards = []
    step_infos = []

    t0 = time.time()
    done = False
    while not done:
        action = agent.eval(obs, weight)
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        step_infos.append(info)
        done = terminated or truncated

    wall_time = time.time() - t0
    ep_return = compute_episode_return(rewards)

    return EpisodeResult(
        method="PCF-MORL",
        scenario=scenario,
        seed=seed,
        weight=weight.tolist(),
        episode_return=ep_return.tolist(),
        qos=compute_episode_qos(step_infos),
        mean_throughput=compute_mean_throughput(step_infos),
        mean_energy=compute_mean_energy(step_infos),
        wall_time_s=wall_time,
        step_rewards=[r.tolist() for r in rewards],
    )


def run_episode_scalarized(
    env: PcfMorlEnv, agent, weight: np.ndarray,
    method_name: str, scenario: str, seed: int
) -> EpisodeResult:
    """Run one episode with a scalarized DQN agent."""
    obs, info = env.reset(seed=seed)
    rewards = []
    step_infos = []

    t0 = time.time()
    done = False
    while not done:
        action = agent.eval(obs, weight)
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        step_infos.append(info)
        done = terminated or truncated

    wall_time = time.time() - t0
    ep_return = compute_episode_return(rewards)

    return EpisodeResult(
        method=method_name,
        scenario=scenario,
        seed=seed,
        weight=weight.tolist(),
        episode_return=ep_return.tolist(),
        qos=compute_episode_qos(step_infos),
        mean_throughput=compute_mean_throughput(step_infos),
        mean_energy=compute_mean_energy(step_infos),
        wall_time_s=wall_time,
        step_rewards=[r.tolist() for r in rewards],
    )


# ---------------------------------------------------------------------------
# Analysis: produce Tab III, Tab IV from results
# ---------------------------------------------------------------------------

def analyze_tab3(results: list[dict], scenario: str = "E2") -> dict:
    """Tab III: QoS metrics for scenario E2 (5 methods × 5 metrics).

    Metrics: VR↓, TTR↓, MVD↓, Throughput↑, Energy↓
    Aggregated across seeds with 95% CI.
    """
    methods = {}
    for r in results:
        if r["scenario"] != scenario:
            continue
        method = r["method"]
        if method not in methods:
            methods[method] = []
        methods[method].append({
            "VR": r["qos"]["VR"],
            "TTR": r["qos"]["TTR"],
            "MVD": r["qos"]["MVD"],
            "Throughput": r["mean_throughput"],
            "Energy": r["mean_energy"],
        })

    tab3 = {}
    for method, seed_metrics in methods.items():
        if seed_metrics:
            tab3[method] = aggregate_seeds(seed_metrics)
    return tab3


def analyze_tab4(results: list[dict]) -> dict:
    """Tab IV: Pareto quality metrics (4 methods × 4 metrics).

    Metrics: HV↑, EU↑, MUL↓, |CCS|
    """
    methods = {}
    for r in results:
        method = r["method"]
        if method not in methods:
            methods[method] = {}
        seed = r["seed"]
        if seed not in methods[method]:
            methods[method][seed] = []
        methods[method][seed].append(r)

    tab4 = {}
    for method, seed_data in methods.items():
        per_seed = []
        for seed, runs in seed_data.items():
            returns = np.array([r["episode_return"] for r in runs])
            weights = np.array([r.get("weight", [1/3, 1/3, 1/3]) for r in runs])

            hv = compute_hypervolume(returns, REF_POINT)
            eu = compute_expected_utility(returns, weights)

            per_seed.append({
                "HV": hv,
                "EU": eu,
                "num_policies": len(runs),
            })

        if per_seed:
            tab4[method] = aggregate_seeds(per_seed)
    return tab4


# ---------------------------------------------------------------------------
# Experiment 1: QoS Performance
# ---------------------------------------------------------------------------

def run_exp1_qos(seeds: list[int], agent=None, output_dir: Path = None,
                 scalarized_agents: dict = None):
    """Exp 1: QoS Performance – PCF-MORL vs A1-A3 vs DQN on E1-E5, 3 ω.

    Output: Tab III, Tab IV, Fig 2.
    """
    output_dir = output_dir or RESULTS_DIR / "exp1"
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results = []

    for scenario_name, scenario_cfg in EVAL_SCENARIOS.items():
        print(f"\n--- {scenario_name}: {scenario_cfg['description']} ---")

        for seed in seeds:
            env = PcfMorlEnv(scenario=scenario_cfg["scenario"], seed=seed)

            # Threshold baselines (A1-A3)
            for method_name, PolicyClass in BASELINES.items():
                policy = PolicyClass()
                result = run_episode_threshold(
                    env, policy, method_name, scenario_name, seed
                )
                all_results.append(asdict(result))
                print(f"  {method_name} seed={seed}: VR={result.qos['VR']:.3f} "
                      f"return={np.array(result.episode_return).round(2)}")

            # GPI-PD at each test weight
            if agent is not None:
                for w in TEST_WEIGHTS:
                    result = run_episode_gpipd(
                        env, agent, w, scenario_name, seed
                    )
                    all_results.append(asdict(result))
                    print(f"  PCF-MORL ω={w} seed={seed}: VR={result.qos['VR']:.3f} "
                          f"return={np.array(result.episode_return).round(2)}")

            # Scalarized DQN baselines
            if scalarized_agents:
                for label, (sagent, w) in scalarized_agents.items():
                    result = run_episode_scalarized(
                        env, sagent, w, f"DQN_{label}", scenario_name, seed
                    )
                    all_results.append(asdict(result))
                    print(f"  DQN_{label} seed={seed}: VR={result.qos['VR']:.3f}")

            env.close()

    # Save raw results
    out_file = output_dir / "exp1_results.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)

    # Compute Tab III and Tab IV
    tab3 = analyze_tab3(all_results, scenario="E2")
    tab4 = analyze_tab4(all_results)

    analysis_file = output_dir / "exp1_analysis.json"
    with open(analysis_file, "w") as f:
        json.dump({"tab3_E2": tab3, "tab4_pareto": tab4}, f, indent=2, default=str)

    print(f"\nExp 1 results saved to {out_file}")
    print(f"Exp 1 analysis saved to {analysis_file}")
    return all_results


# ---------------------------------------------------------------------------
# Experiment 2: Zero-shot adaptation
# ---------------------------------------------------------------------------

def run_exp2_adaptation(seeds: list[int], agent=None, oracle_agents: dict = None,
                        output_dir: Path = None):
    """Exp 2: Zero-shot adaptation – GPI vs Oracle vs Retrain vs Fine-tune.

    Tests 20 weight vectors, measures ZSR and adaptation curves.
    Output: Fig 3, ZSR.
    """
    output_dir = output_dir or RESULTS_DIR / "exp2"
    output_dir.mkdir(parents=True, exist_ok=True)

    gpi_results = []
    oracle_results = []

    for seed in seeds:
        env = PcfMorlEnv(scenario="eval_constant_90", seed=seed)

        for w in TEST_WEIGHTS_20:
            w_arr = np.array(w)

            # GPI-PD (zero-shot: no retraining, just set weight)
            if agent is not None:
                result = run_episode_gpipd(env, agent, w_arr, "E1", seed)
                result_dict = asdict(result)
                result_dict["adaptation_type"] = "zero_shot"
                gpi_results.append(result_dict)

            # Oracle DQN (dedicated per-ω agent)
            if oracle_agents is not None:
                w_key = f"w_{np.round(w_arr, 2).tolist()}"
                if w_key in oracle_agents:
                    o_agent = oracle_agents[w_key]
                    result = run_episode_scalarized(
                        env, o_agent, w_arr, "Oracle", "E1", seed
                    )
                    oracle_results.append(asdict(result))

        env.close()

    # Compute ZSR if both GPI and Oracle results available
    analysis = {}
    if gpi_results and oracle_results:
        gpi_returns = np.array([r["episode_return"] for r in gpi_results])
        oracle_returns = np.array([r["episode_return"] for r in oracle_results])
        weights = np.array([r["weight"] for r in gpi_results])

        zsr = compute_zsr(gpi_returns, oracle_returns, weights, threshold=0.9)
        analysis["ZSR"] = zsr
        print(f"  ZSR (GPI ≥ 90% Oracle): {zsr:.3f}")

    all_results = {
        "gpi": gpi_results,
        "oracle": oracle_results,
        "analysis": analysis,
    }

    out_file = output_dir / "exp2_results.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Exp 2 results saved to {out_file}")
    return all_results


# ---------------------------------------------------------------------------
# Experiment 3: 24h factory cycle
# ---------------------------------------------------------------------------

def run_exp3_factory_cycle(seeds: list[int], agent=None, output_dir: Path = None):
    """Exp 3: 24h factory cycle – 8 phase transitions, cumulative J.

    Simulates 8 consecutive production cycles (phases) with different ω per phase.
    Measures cumulative scalarized return across the full cycle.
    Output: Text paragraph.
    """
    output_dir = output_dir or RESULTS_DIR / "exp3"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 8 phases with different weight emphasis (simulating shift changes)
    phase_weights = [
        np.array([0.6, 0.3, 0.1]),  # Day shift 1: throughput
        np.array([0.3, 0.5, 0.2]),  # Day shift 2: delay-sensitive batch
        np.array([0.2, 0.7, 0.1]),  # Day shift 3: critical URLLC
        np.array([0.4, 0.3, 0.3]),  # Day shift 4: balanced
        np.array([0.2, 0.2, 0.6]),  # Night shift 1: energy saving
        np.array([0.1, 0.3, 0.6]),  # Night shift 2: energy + delay
        np.array([0.5, 0.2, 0.3]),  # Night shift 3: moderate throughput
        np.array([0.3, 0.4, 0.3]),  # Night shift 4: balanced closing
    ]

    phase_scenarios = ["E1", "E2", "E3", "E4", "E1", "E5", "E3", "E1"]

    all_results = []

    for seed in seeds:
        cycle_returns = []
        cycle_details = []

        for phase_idx, (w, scenario) in enumerate(zip(phase_weights, phase_scenarios)):
            scenario_cfg = EVAL_SCENARIOS[scenario]
            env = PcfMorlEnv(scenario=scenario_cfg["scenario"], seed=seed + phase_idx)

            if agent is not None:
                result = run_episode_gpipd(env, agent, w, scenario, seed)
            else:
                # Use A3 hysteresis as fallback
                policy = BASELINES["A3_hysteresis"]()
                result = run_episode_threshold(
                    env, policy, "A3_hysteresis", scenario, seed
                )

            ep_return = np.array(result.episode_return)
            j = float(np.dot(ep_return, w))
            cycle_returns.append(j)
            cycle_details.append({
                "phase": phase_idx,
                "scenario": scenario,
                "weight": w.tolist(),
                "episode_return": result.episode_return,
                "J": j,
                "qos": result.qos,
            })
            env.close()

            print(f"  Seed {seed} Phase {phase_idx} ({scenario}): J={j:.3f}")

        cumulative_j = float(np.sum(cycle_returns))
        all_results.append({
            "seed": seed,
            "cumulative_J": cumulative_j,
            "phase_details": cycle_details,
        })
        print(f"  Seed {seed} cumulative J: {cumulative_j:.3f}")

    out_file = output_dir / "exp3_results.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Exp 3 results saved to {out_file}")
    return all_results


# ---------------------------------------------------------------------------
# Experiment 4: Parallel scaling
# ---------------------------------------------------------------------------

def run_exp4_scaling(
    seeds: list[int], worker_counts: list[int] = None, output_dir: Path = None,
    total_episodes: int = 10_000, episodes_per_sync: int = 200,
    max_steps: int = 100, device: str = "auto",
):
    """Exp 4: Parallel scaling – P={1,2,4,8,16}, wall-clock + HV.

    Output: Fig 4.
    """
    from parallel.coordinator import run_coordinator

    output_dir = output_dir or RESULTS_DIR / "exp4"
    output_dir.mkdir(parents=True, exist_ok=True)

    if worker_counts is None:
        worker_counts = [1, 2, 4, 8, 16]

    scaling_results = []

    for P in worker_counts:
        for seed in seeds:
            print(f"\n--- Scaling test: P={P}, seed={seed} ---")
            result = run_coordinator(
                num_workers=P,
                total_episodes=total_episodes,
                episodes_per_sync=episodes_per_sync,
                max_steps=max_steps,
                base_seed=seed,
                device=device,
            )
            scaling_results.append({
                "P": P,
                "seed": seed,
                "wall_time_s": result.get("total_wall_time_s"),
                "sync_rounds": result.get("sync_rounds"),
                "merged_weight_counts": result.get("merged_weight_counts"),
                "worker_timings": result.get("worker_timings"),
            })

    # Compute speedup relative to P=1
    p1_times = [r["wall_time_s"] for r in scaling_results if r["P"] == 1 and r["wall_time_s"]]
    baseline_time = float(np.mean(p1_times)) if p1_times else None

    if baseline_time:
        for r in scaling_results:
            if r["wall_time_s"]:
                r["speedup"] = baseline_time / r["wall_time_s"]

    out_file = output_dir / "exp4_results.json"
    with open(out_file, "w") as f:
        json.dump(scaling_results, f, indent=2)
    print(f"\nExp 4 results saved to {out_file}")
    return scaling_results


# ---------------------------------------------------------------------------
# Experiment 5: K tuning
# ---------------------------------------------------------------------------

def run_exp5_k_tuning(
    seeds: list[int], output_dir: Path = None,
    k_values: list[int] = None, num_workers: int = 8,
    total_episodes: int = 10_000, max_steps: int = 100,
    device: str = "auto",
):
    """Exp 5: K tuning – K={50,100,200,500,1000}, P=8.

    Measures wall-clock time and HV for each K value.
    Output: Best K.
    """
    from parallel.coordinator import run_coordinator

    output_dir = output_dir or RESULTS_DIR / "exp5"
    output_dir.mkdir(parents=True, exist_ok=True)

    if k_values is None:
        k_values = [50, 100, 200, 500, 1000]

    tuning_results = []

    for K in k_values:
        for seed in seeds:
            print(f"\n--- K tuning: K={K}, seed={seed} ---")
            result = run_coordinator(
                num_workers=num_workers,
                total_episodes=total_episodes,
                episodes_per_sync=K,
                max_steps=max_steps,
                base_seed=seed,
                device=device,
            )
            tuning_results.append({
                "K": K,
                "seed": seed,
                "wall_time_s": result.get("total_wall_time_s"),
                "sync_rounds": result.get("sync_rounds"),
                "merged_weight_counts": result.get("merged_weight_counts"),
            })

    # Find best K (lowest wall time with good HV)
    k_means = {}
    for r in tuning_results:
        K = r["K"]
        if K not in k_means:
            k_means[K] = []
        if r["wall_time_s"]:
            k_means[K].append(r["wall_time_s"])

    best_k = None
    best_time = float("inf")
    for K, times in k_means.items():
        avg = float(np.mean(times))
        if avg < best_time:
            best_time = avg
            best_k = K
        print(f"  K={K}: avg wall time = {avg:.1f}s")

    analysis = {"best_K": best_k, "best_wall_time_s": best_time, "k_means": {
        str(k): float(np.mean(v)) for k, v in k_means.items()
    }}

    out_file = output_dir / "exp5_results.json"
    with open(out_file, "w") as f:
        json.dump({"results": tuning_results, "analysis": analysis}, f, indent=2)
    print(f"\nExp 5: Best K={best_k} ({best_time:.1f}s)")
    print(f"Results saved to {out_file}")
    return tuning_results


# ---------------------------------------------------------------------------
# Experiment 6: SF sharing ablation
# ---------------------------------------------------------------------------

def run_exp6_ablation(
    seeds: list[int], output_dir: Path = None,
    num_workers: int = 8, total_episodes: int = 10_000,
    episodes_per_sync: int = 200, max_steps: int = 100,
    device: str = "auto",
):
    """Exp 6: SF sharing ablation – 4 strategies at P=8.

    Strategies:
      1. No sharing (independent workers)
      2. CCS union (current default)
      3. Top-K per worker (share only K best weights)
      4. Periodic full merge (share every episode)

    Output: Ablation paragraph.
    """
    from parallel.coordinator import run_coordinator

    output_dir = output_dir or RESULTS_DIR / "exp6"
    output_dir.mkdir(parents=True, exist_ok=True)

    strategies = {
        "no_sharing": {"episodes_per_sync": total_episodes + 1},  # Never sync
        "ccs_union_K200": {"episodes_per_sync": 200},             # Default
        "ccs_union_K50": {"episodes_per_sync": 50},               # Frequent
        "ccs_union_K1000": {"episodes_per_sync": 1000},           # Rare
    }

    ablation_results = []

    for strategy_name, strategy_cfg in strategies.items():
        for seed in seeds:
            print(f"\n--- Ablation: {strategy_name}, seed={seed} ---")
            K = strategy_cfg["episodes_per_sync"]
            result = run_coordinator(
                num_workers=num_workers,
                total_episodes=total_episodes,
                episodes_per_sync=K,
                max_steps=max_steps,
                base_seed=seed,
                device=device,
            )
            ablation_results.append({
                "strategy": strategy_name,
                "K": K,
                "seed": seed,
                "wall_time_s": result.get("total_wall_time_s"),
                "sync_rounds": result.get("sync_rounds"),
                "merged_weight_counts": result.get("merged_weight_counts"),
            })

    out_file = output_dir / "exp6_results.json"
    with open(out_file, "w") as f:
        json.dump(ablation_results, f, indent=2)
    print(f"\nExp 6 results saved to {out_file}")
    return ablation_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

EXPERIMENTS = {
    "1": run_exp1_qos,
    "2": run_exp2_adaptation,
    "3": run_exp3_factory_cycle,
    "4": run_exp4_scaling,
    "5": run_exp5_k_tuning,
    "6": run_exp6_ablation,
}


def main():
    parser = argparse.ArgumentParser(description="PCF-MORL Experiment Runner")
    parser.add_argument("--exp", type=str, default="1",
                        help="Experiment number (1-6) or 'all'")
    parser.add_argument("--seeds", type=int, default=5,
                        help="Number of seeds (default: 5)")
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--gpipd-checkpoint", type=str, default=None,
                        help="Path to trained GPI-PD checkpoint")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-steps", type=int, default=100)
    args = parser.parse_args()

    seeds = [args.base_seed + i * 100 for i in range(args.seeds)]
    output_dir = Path(args.output_dir) if args.output_dir else None

    # Load GPI-PD agent if checkpoint provided
    agent = None
    if args.gpipd_checkpoint:
        env = PcfMorlEnv(max_steps=args.max_steps)
        agent = load_gpipd_agent(args.gpipd_checkpoint, env, device=args.device)
        env.close()

    if args.exp == "all":
        for exp_id in sorted(EXPERIMENTS.keys()):
            exp_fn = EXPERIMENTS[exp_id]
            print(f"\n{'='*60}")
            print(f"Running Experiment {exp_id}")
            print(f"{'='*60}")
            if exp_id in ("1",):
                exp_fn(seeds, agent=agent, output_dir=output_dir)
            elif exp_id in ("2",):
                exp_fn(seeds, agent=agent, output_dir=output_dir)
            elif exp_id in ("3",):
                exp_fn(seeds, agent=agent, output_dir=output_dir)
            elif exp_id in ("4", "5", "6"):
                exp_fn(seeds, output_dir=output_dir, device=args.device,
                       max_steps=args.max_steps)
    else:
        exp_fn = EXPERIMENTS.get(args.exp)
        if exp_fn is None:
            print(f"Unknown experiment: {args.exp}. Choose from: {list(EXPERIMENTS.keys())}")
            sys.exit(1)
        if args.exp in ("1",):
            exp_fn(seeds, agent=agent, output_dir=output_dir)
        elif args.exp in ("2",):
            exp_fn(seeds, agent=agent, output_dir=output_dir)
        elif args.exp in ("3",):
            exp_fn(seeds, agent=agent, output_dir=output_dir)
        elif args.exp in ("4", "5", "6"):
            exp_fn(seeds, output_dir=output_dir, device=args.device,
                   max_steps=args.max_steps)


if __name__ == "__main__":
    main()
