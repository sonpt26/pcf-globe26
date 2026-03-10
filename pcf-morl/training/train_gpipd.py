"""M2: GPI-PD training for PCF-MORL.

Usage:
    python -m training.train_gpipd [--total-timesteps 500000] [--seed 42] [--wandb]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from morl_baselines.multi_policy.gpi_pd.gpi_pd import GPIPD
from env.pcf_morl_env import PcfMorlEnv


def make_env(seed=0, max_steps=100):
    return PcfMorlEnv(seed=seed, max_steps=max_steps)


def main():
    parser = argparse.ArgumentParser(description="Train GPI-PD on PCF-MORL")
    parser.add_argument("--total-timesteps", type=int, default=500_000,
                        help="Total training timesteps (default: 500K = 5K episodes)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=100,
                        help="Steps per episode")
    parser.add_argument("--wandb", action="store_true",
                        help="Enable W&B logging")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--project-name", type=str, default="PCF-MORL")
    parser.add_argument("--experiment-name", type=str, default="GPI-PD")
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    # Reference point for hypervolume (worst possible reward per objective)
    # r1 ∈ [0,1], r2 ∈ [-1,0], r3 ∈ [-1,0]
    # Episodic return = sum over 100 steps, so ref point is 100 × worst per-step
    ref_point = np.array([0.0, -100.0, -100.0])

    env = make_env(seed=args.seed, max_steps=args.max_steps)
    eval_env = make_env(seed=args.seed + 1000, max_steps=args.max_steps)

    agent = GPIPD(
        env=env,
        learning_rate=3e-4,
        gamma=0.99,
        batch_size=128,
        net_arch=[256, 256, 256, 256],
        buffer_size=500_000,
        target_net_update_freq=1000,
        tau=1.0,
        initial_epsilon=0.01,
        num_nets=2,
        gradient_updates=20,
        per=True,
        dyna=True,
        gpi_pd=True,
        dynamics_rollout_freq=250,
        dynamics_rollout_starts=5000,
        dynamics_rollout_batch_size=25000,
        dynamics_buffer_size=100000,
        project_name=args.project_name,
        experiment_name=args.experiment_name,
        wandb_entity=args.wandb_entity,
        log=args.wandb,
        seed=args.seed,
        device=args.device,
    )

    print(f"Starting GPI-PD training:")
    print(f"  total_timesteps={args.total_timesteps}")
    print(f"  episodes≈{args.total_timesteps // args.max_steps}")
    print(f"  eval_freq={args.eval_freq}")
    print(f"  ref_point={ref_point}")
    print(f"  device={args.device}")
    print(f"  seed={args.seed}")

    agent.train(
        total_timesteps=args.total_timesteps,
        eval_env=eval_env,
        ref_point=ref_point,
        eval_freq=args.eval_freq,
        eval_mo_freq=args.eval_freq * 5,
        timesteps_per_iter=args.eval_freq,
        num_eval_weights_for_front=50,
        num_eval_episodes_for_front=3,
        checkpoints=True,
    )

    # Save final model
    output_dir = Path(__file__).parent.parent / "results" / "gpipd"
    output_dir.mkdir(parents=True, exist_ok=True)
    # morl-baselines saves checkpoints automatically via wandb/local

    print(f"Training complete. Checkpoints in wandb or local dir.")


if __name__ == "__main__":
    main()
