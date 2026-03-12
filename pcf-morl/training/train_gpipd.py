"""M2: GPI-PD training for PCF-MORL.

Usage:
    python -m training.train_gpipd [--total-timesteps 500000] [--seed 42] [--wandb]

Logs:
    results/logs/<run_name>/episodes.csv    — per-episode returns, QoS, KPIs
    results/logs/<run_name>/steps.csv       — per-step obs, reward, action (if --log-steps)
    results/logs/<run_name>/morl_evals.csv  — per-iteration HV, EU, |CCS|
    results/logs/<run_name>/pareto_snapshots.json — Pareto front at each eval
    results/logs/<run_name>/config.json     — training config
    results/logs/<run_name>/dashboard.png   — training dashboard plot

Monitor during training:
    tail -f results/logs/<run_name>/episodes.csv
    python3 -m training.dashboard --log-dir results/logs/<run_name>
    python3 -m training.dashboard --log-dir results/logs/<run_name> --live 60
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from morl_baselines.multi_policy.gpi_pd.gpi_pd import GPIPD
from env.pcf_morl_env import PcfMorlEnv
from env.logged_env import LoggedPcfMorlEnv
from training.logger import TrainingLogger
from training.morl_eval_logger import MorlEvalLogger


def make_env(seed=0, max_steps=100, logger=None):
    if logger:
        return LoggedPcfMorlEnv(logger=logger, seed=seed, max_steps=max_steps)
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
    parser.add_argument("--log-steps", action="store_true",
                        help="Log per-step data (large files)")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Name for log directory (default: auto-generated)")
    args = parser.parse_args()

    # Reference point for hypervolume (worst possible reward per objective)
    # r1 ∈ [0,1], r2 ∈ [-1,0], r3 ∈ [-1,0]
    # Episodic return = sum over 100 steps, so ref point is 100 × worst per-step
    ref_point = np.array([0.0, -100.0, -100.0])

    # Setup logger
    run_name = args.run_name or f"gpipd_s{args.seed}_{time.strftime('%Y%m%d_%H%M%S')}"
    log_dir = Path(__file__).parent.parent / "results" / "logs" / run_name
    logger = TrainingLogger(str(log_dir), step_level=args.log_steps)
    logger.save_config({
        "total_timesteps": args.total_timesteps,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "device": args.device,
        "log_steps": args.log_steps,
    })
    print(f"Logging to: {log_dir}")
    print(f"  Monitor: tail -f {log_dir}/episodes.csv")
    print(f"  Plot:    python3 -m training.logger --plot {log_dir}")
    print()

    env = make_env(seed=args.seed, max_steps=args.max_steps, logger=logger)
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

    # Setup MORL eval logger
    eval_mo_freq = args.eval_freq * 5  # MORL eval every 5 iterations
    morl_logger = MorlEvalLogger(
        log_dir=str(log_dir),
        ref_point=ref_point,
        num_eval_weights=20,
        num_eval_episodes=1,
    )

    print(f"Starting GPI-PD training:")
    print(f"  total_timesteps={args.total_timesteps}")
    print(f"  episodes≈{args.total_timesteps // args.max_steps}")
    print(f"  eval_freq={args.eval_freq}")
    print(f"  eval_mo_freq={eval_mo_freq}")
    print(f"  ref_point={ref_point}")
    print(f"  device={args.device}")
    print(f"  seed={args.seed}")
    print()
    print(f"Logging to: {log_dir}")
    print(f"  Episodes:  tail -f {log_dir}/episodes.csv")
    print(f"  MORL eval: tail -f {log_dir}/morl_evals.csv")
    print(f"  Dashboard: python3 -m training.dashboard --log-dir {log_dir}")
    print(f"  Live:      python3 -m training.dashboard --log-dir {log_dir} --live 60")
    print()

    # Use manual iteration loop to inject MORL eval logging
    # (morl-baselines only logs to W&B, we need local CSV)
    from morl_baselines.common.evaluation import policy_evaluation_mo
    from morl_baselines.common.weights import equally_spaced_weights
    from morl_baselines.multi_policy.linear_support.linear_support import LinearSupport

    max_iter = args.total_timesteps // args.eval_freq
    linear_support = LinearSupport(num_objectives=agent.reward_dim, epsilon=None)
    eval_weights = equally_spaced_weights(agent.reward_dim, n=50)

    for iteration in range(1, max_iter + 1):
        # Weight selection via GPI-LS
        agent.set_weight_support(linear_support.get_weight_support())
        use_gpi = agent.use_gpi
        agent.use_gpi = True
        w = linear_support.next_weight(
            algo="gpi-ls", gpi_agent=agent, env=eval_env, rep_eval=3
        )
        agent.use_gpi = use_gpi

        if w is None:
            print(f"  No more weights to explore, stopping at iter {iteration}")
            break

        print(f"\n--- Iteration {iteration}/{max_iter} | weight={np.round(w, 3)} ---")

        M = linear_support.get_weight_support() + linear_support.get_corner_weights(top_k=4) + [w]

        # Train for one iteration
        agent.train_iteration(
            total_timesteps=args.eval_freq,
            weight=w,
            weight_support=M,
            change_w_every_episode=True,
            eval_env=eval_env,
            eval_freq=args.eval_freq,
            reset_num_timesteps=False,
            reset_learning_starts=False,
        )

        # Update linear support
        for wcw in M:
            n_value = policy_evaluation_mo(agent, eval_env, wcw, rep=3)[3]
            linear_support.add_solution(n_value, wcw)

        # MORL eval logging (HV, EU, |CCS|)
        if iteration % 5 == 0 or iteration == 1:
            morl_logger.log_iteration(agent, eval_env, iteration, agent.global_step)

        # Checkpoint
        agent.save(filename=f"GPI-PD gpi-ls iter={iteration}", save_replay_buffer=False)

    # Final MORL eval
    morl_logger.log_iteration(agent, eval_env, iteration, agent.global_step)

    # Generate dashboard
    try:
        from training.dashboard import generate_dashboard
        generate_dashboard(str(log_dir))
    except Exception as e:
        print(f"Dashboard generation failed: {e}")

    # Close loggers
    env.close()
    eval_env.close()
    logger.close()
    morl_logger.close()

    print(f"\nTraining complete.")
    print(f"Logs:      {log_dir}")
    print(f"Dashboard: {log_dir}/dashboard.png")
    print(f"Replot:    python3 -m training.dashboard --log-dir {log_dir}")


if __name__ == "__main__":
    main()
