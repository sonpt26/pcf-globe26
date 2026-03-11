"""M4: Config-driven experiment runner for PCF-MORL.

Runs experiments E1-E5 across all methods (GPI-PD, A1-A3, Scalarized DQN)
with multiple seeds, collects metrics, and saves results to JSON.

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
from experiments.metrics import (
    compute_episode_qos,
    compute_episode_return,
    compute_mean_throughput,
    compute_mean_energy,
    compute_hypervolume,
    compute_expected_utility,
    compute_mul,
    compute_zsr,
    aggregate_seeds,
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

# Reference point for hypervolume (per-step worst case × 100 steps)
REF_POINT = np.array([0.0, -100.0, -100.0])


# ---------------------------------------------------------------------------
# Episode runner
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
    )


# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------

def run_exp1_qos(seeds: list[int], agent=None, output_dir: Path = None):
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

            # Threshold baselines
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

            env.close()

    # Save results
    out_file = output_dir / "exp1_results.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nExp 1 results saved to {out_file}")
    return all_results


def run_exp2_adaptation(seeds: list[int], agent=None, output_dir: Path = None):
    """Exp 2: Zero-shot adaptation – GPI vs Oracle vs Retrain vs Fine-tune.

    Tests 20 weight vectors, measures ZSR and adaptation curves.
    Output: Fig 3, ZSR.
    """
    from baselines.scalarized_dqn import generate_simplex_weights

    output_dir = output_dir or RESULTS_DIR / "exp2"
    output_dir.mkdir(parents=True, exist_ok=True)

    test_weights = generate_simplex_weights(20, 3)
    all_results = []

    for seed in seeds:
        env = PcfMorlEnv(scenario="eval_constant_90", seed=seed)

        for w in test_weights:
            if agent is not None:
                result = run_episode_gpipd(env, agent, np.array(w), "E1", seed)
                all_results.append(asdict(result))

        env.close()

    out_file = output_dir / "exp2_results.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Exp 2 results saved to {out_file}")
    return all_results


def run_exp4_scaling(
    seeds: list[int], worker_counts: list[int] = None, output_dir: Path = None
):
    """Exp 4: Parallel scaling – P={1,2,4,8,16}, wall-clock + HV.

    Output: Fig 4.
    """
    output_dir = output_dir or RESULTS_DIR / "exp4"
    output_dir.mkdir(parents=True, exist_ok=True)

    if worker_counts is None:
        worker_counts = [1, 2, 4, 8, 16]

    # This experiment requires the parallel orchestrator
    # Results will be collected by coordinator.py
    scaling_results = []
    for P in worker_counts:
        print(f"\n--- Scaling test: P={P} workers ---")
        for seed in seeds:
            # Placeholder: actual timing comes from parallel/coordinator.py
            scaling_results.append({
                "P": P,
                "seed": seed,
                "wall_time_s": None,  # Filled by coordinator
                "hv": None,
                "status": "pending",
            })

    out_file = output_dir / "exp4_config.json"
    with open(out_file, "w") as f:
        json.dump(scaling_results, f, indent=2)
    print(f"Exp 4 config saved to {out_file}")
    return scaling_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

EXPERIMENTS = {
    "1": run_exp1_qos,
    "2": run_exp2_adaptation,
    "4": run_exp4_scaling,
}


def main():
    parser = argparse.ArgumentParser(description="PCF-MORL Experiment Runner")
    parser.add_argument("--exp", type=str, default="1",
                        help="Experiment number (1,2,4) or 'all'")
    parser.add_argument("--seeds", type=int, default=5,
                        help="Number of seeds (default: 5)")
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--gpipd-checkpoint", type=str, default=None,
                        help="Path to trained GPI-PD checkpoint")
    args = parser.parse_args()

    seeds = [args.base_seed + i * 100 for i in range(args.seeds)]
    output_dir = Path(args.output_dir) if args.output_dir else None

    # Load GPI-PD agent if checkpoint provided
    agent = None
    if args.gpipd_checkpoint:
        print(f"Loading GPI-PD from {args.gpipd_checkpoint}")
        # Agent loading deferred to when morl-baselines provides load API
        # agent = GPIPD.load(args.gpipd_checkpoint)

    if args.exp == "all":
        for exp_id, exp_fn in EXPERIMENTS.items():
            print(f"\n{'='*60}")
            print(f"Running Experiment {exp_id}")
            print(f"{'='*60}")
            exp_fn(seeds, agent=agent, output_dir=output_dir)
    else:
        exp_fn = EXPERIMENTS.get(args.exp)
        if exp_fn is None:
            print(f"Unknown experiment: {args.exp}. Choose from: {list(EXPERIMENTS.keys())}")
            sys.exit(1)
        exp_fn(seeds, agent=agent, output_dir=output_dir)


if __name__ == "__main__":
    main()
