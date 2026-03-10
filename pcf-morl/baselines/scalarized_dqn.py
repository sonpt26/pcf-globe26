"""Scalarized DQN baseline: train separate DQN for each fixed weight vector.

Uses morl-baselines MOQLearning with fixed scalarization weights.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from morl_baselines.single_policy.ser.mo_q_learning import MOQLearning
from env.pcf_morl_env import PcfMorlEnv

# Fixed weight vectors for scalarized DQN baseline
SCALARIZED_WEIGHTS = [
    np.array([0.2, 0.7, 0.1]),  # Delay-focused
    np.array([0.5, 0.2, 0.3]),  # Balanced
    np.array([0.1, 0.3, 0.6]),  # Energy-focused
]

# Oracle DQN: 20 evenly-spaced test weights on the 3-simplex
def generate_simplex_weights(n=20, dim=3):
    """Generate n approximately uniformly distributed weights on the simplex."""
    weights = []
    # Use a grid on the 2-simplex
    step = int(np.ceil(n ** (1.0 / (dim - 1))))
    for i in range(step + 1):
        for j in range(step + 1 - i):
            k = step - i - j
            w = np.array([i, j, k], dtype=np.float64) / step
            if np.all(w >= 0.05):  # Ensure minimum weight per objective
                weights.append(w)
    # Trim or pad to exactly n
    if len(weights) > n:
        indices = np.linspace(0, len(weights) - 1, n, dtype=int)
        weights = [weights[i] for i in indices]
    return weights


ORACLE_WEIGHTS = generate_simplex_weights(20, 3)


def train_scalarized(weights_list, total_timesteps=500_000, seed=42, max_steps=100,
                     label="scalarized"):
    """Train one DQN per weight vector."""
    results = {}
    for i, w in enumerate(weights_list):
        print(f"\n{'='*60}")
        print(f"Training {label} DQN [{i+1}/{len(weights_list)}] with ω={w}")
        print(f"{'='*60}")

        env = PcfMorlEnv(seed=seed + i, max_steps=max_steps)
        eval_env = PcfMorlEnv(seed=seed + 1000 + i, max_steps=max_steps)

        agent = MOQLearning(
            env=env,
            scalarization=lambda r, w=w: np.dot(r, w),
            weights=w,
            learning_rate=3e-4,
            gamma=0.99,
            initial_epsilon=1.0,
            final_epsilon=0.05,
            epsilon_decay_steps=int(total_timesteps * 0.5),
            log=False,
            seed=seed + i,
        )

        agent.train(
            total_timesteps=total_timesteps,
            eval_env=eval_env,
            ref_point=np.array([0.0, -100.0, -100.0]),
            eval_freq=total_timesteps // 10,
        )

        results[f"w_{i}"] = {
            "weight": w.tolist(),
            "agent": agent,
        }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["scalarized", "oracle"], default="scalarized")
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=100)
    args = parser.parse_args()

    if args.mode == "scalarized":
        train_scalarized(SCALARIZED_WEIGHTS, args.total_timesteps, args.seed,
                         args.max_steps, "scalarized")
    else:
        train_scalarized(ORACLE_WEIGHTS, args.total_timesteps, args.seed,
                         args.max_steps, "oracle")


if __name__ == "__main__":
    main()
