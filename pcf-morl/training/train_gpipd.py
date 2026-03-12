"""M2: GPI-PD training for PCF-MORL.

Usage:
    # Start new training
    python -m training.train_gpipd --seed 42 --device cuda

    # Resume from checkpoint
    python -m training.train_gpipd --resume results/logs/gpipd_s42_20260312

Logs:
    results/logs/<run_name>/episodes.csv    — per-episode returns, QoS, KPIs
    results/logs/<run_name>/steps.csv       — per-step obs, reward, action (if --log-steps)
    results/logs/<run_name>/morl_evals.csv  — per-iteration HV, EU, |CCS|
    results/logs/<run_name>/pareto_snapshots.json — Pareto front at each eval
    results/logs/<run_name>/config.json     — training config
    results/logs/<run_name>/dashboard.png   — training dashboard plot
    results/logs/<run_name>/checkpoints/    — per-iteration model + loop state

Monitor during training:
    tail -f results/logs/<run_name>/episodes.csv
    python3 -m training.dashboard --log-dir results/logs/<run_name>
    python3 -m training.dashboard --log-dir results/logs/<run_name> --live 60
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch as th

sys.path.insert(0, str(Path(__file__).parent.parent))

from morl_baselines.multi_policy.gpi_pd.gpi_pd import GPIPD
from morl_baselines.common.evaluation import policy_evaluation_mo
from morl_baselines.common.weights import equally_spaced_weights
from morl_baselines.multi_policy.linear_support.linear_support import LinearSupport

from env.pcf_morl_env import PcfMorlEnv
from env.logged_env import LoggedPcfMorlEnv
from training.logger import TrainingLogger
from training.morl_eval_logger import MorlEvalLogger


def make_env(seed=0, max_steps=100, logger=None):
    if logger:
        return LoggedPcfMorlEnv(logger=logger, seed=seed, max_steps=max_steps)
    return PcfMorlEnv(seed=seed, max_steps=max_steps)


# ---------------------------------------------------------------------------
# Checkpoint: save/load full training state (agent + loop state)
# ---------------------------------------------------------------------------

def _cleanup_old_checkpoints(ckpt_dir: Path, keep: int):
    """Remove old per-iteration checkpoints, keeping only the most recent `keep`."""
    if keep <= 0:
        return
    iter_files = sorted(ckpt_dir.glob("agent_iter*.tar"))
    if len(iter_files) > keep:
        for old in iter_files[:-keep]:
            old.unlink()


def save_checkpoint(agent, linear_support, iteration: int, log_dir: Path,
                    keep_checkpoints: int = 3):
    """Save full training state for resume."""
    ckpt_dir = log_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Agent weights (Q-nets, optimizer, weight support, dynamics)
    agent.save(
        save_dir=str(ckpt_dir),
        filename=f"agent_iter{iteration:04d}",
        save_replay_buffer=False,
    )

    # Also save as "latest" for easy resume
    agent.save(
        save_dir=str(ckpt_dir),
        filename="agent_latest",
        save_replay_buffer=True,
    )

    # Prune old per-iteration checkpoints
    _cleanup_old_checkpoints(ckpt_dir, keep_checkpoints)

    # Loop state (iteration, linear support, global_step)
    loop_state = {
        "iteration": iteration,
        "global_step": agent.global_step,
        "linear_support_weights": [w.tolist() if hasattr(w, 'tolist') else w
                                    for w in linear_support.get_weight_support()],
        "linear_support_corners": [],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Save linear support solutions
    try:
        solutions = []
        for w, v in zip(linear_support.weight_support, linear_support.ccs):
            solutions.append({
                "weight": w.tolist() if hasattr(w, 'tolist') else list(w),
                "value": v.tolist() if hasattr(v, 'tolist') else list(v),
            })
        loop_state["linear_support_solutions"] = solutions
    except Exception:
        pass

    with open(ckpt_dir / "loop_state.json", "w") as f:
        json.dump(loop_state, f, indent=2)

    print(f"  Checkpoint saved: iter={iteration}, step={agent.global_step}")


def load_checkpoint(agent, log_dir: Path):
    """Load training state for resume. Returns (linear_support, start_iteration)."""
    ckpt_dir = log_dir / "checkpoints"

    # Load agent weights
    agent_path = ckpt_dir / "agent_latest.tar"
    if not agent_path.exists():
        raise FileNotFoundError(f"No checkpoint at {agent_path}")

    agent.load(str(agent_path), load_replay_buffer=True)
    print(f"Loaded agent: |M|={len(agent.weight_support)}")

    # Load loop state
    loop_state_path = ckpt_dir / "loop_state.json"
    if not loop_state_path.exists():
        raise FileNotFoundError(f"No loop state at {loop_state_path}")

    with open(loop_state_path) as f:
        loop_state = json.load(f)

    iteration = loop_state["iteration"]
    agent.global_step = loop_state["global_step"]

    # Rebuild linear support
    linear_support = LinearSupport(num_objectives=agent.reward_dim, epsilon=None)

    if "linear_support_solutions" in loop_state:
        for sol in loop_state["linear_support_solutions"]:
            w = np.array(sol["weight"])
            v = np.array(sol["value"])
            linear_support.add_solution(v, w)

    print(f"Resumed from: iter={iteration}, step={agent.global_step}, "
          f"|CCS|={len(linear_support.ccs)}")

    return linear_support, iteration


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to log directory to resume from")
    parser.add_argument("--keep-checkpoints", type=int, default=3,
                        help="Number of recent per-iteration checkpoints to keep (0=keep all)")
    args = parser.parse_args()

    ref_point = np.array([0.0, -100.0, -100.0])
    resuming = args.resume is not None

    # ── Determine log directory ──
    if resuming:
        log_dir = Path(args.resume)
        if not log_dir.exists():
            print(f"ERROR: Resume directory not found: {log_dir}")
            sys.exit(1)
        # Load config from previous run
        config_path = log_dir / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                prev_config = json.load(f)
            # Inherit settings unless overridden on CLI
            if args.seed == 42 and "seed" in prev_config:
                args.seed = prev_config["seed"]
            if args.max_steps == 100 and "max_steps" in prev_config:
                args.max_steps = prev_config["max_steps"]
            if args.device == "auto" and "device" in prev_config:
                args.device = prev_config["device"]
        run_name = log_dir.name
    else:
        run_name = args.run_name or f"gpipd_s{args.seed}_{time.strftime('%Y%m%d_%H%M%S')}"
        log_dir = Path(__file__).parent.parent / "results" / "logs" / run_name

    # ── Setup loggers (append mode if resuming) ──
    logger = TrainingLogger(str(log_dir), step_level=args.log_steps,
                            flush_every=1)
    if not resuming:
        logger.save_config({
            "total_timesteps": args.total_timesteps,
            "seed": args.seed,
            "max_steps": args.max_steps,
            "device": args.device,
            "eval_freq": args.eval_freq,
            "log_steps": args.log_steps,
        })

    morl_logger = MorlEvalLogger(
        log_dir=str(log_dir),
        ref_point=ref_point,
        num_eval_weights=20,
        num_eval_episodes=1,
    )

    # ── Create envs ──
    env = make_env(seed=args.seed, max_steps=args.max_steps, logger=logger)
    eval_env = make_env(seed=args.seed + 1000, max_steps=args.max_steps)

    # ── Create agent ──
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

    # ── Resume or fresh start ──
    max_iter = args.total_timesteps // args.eval_freq
    start_iter = 1

    if resuming:
        linear_support, last_iter = load_checkpoint(agent, log_dir)
        start_iter = last_iter + 1
        print(f"\nResuming from iteration {start_iter}/{max_iter}")
    else:
        linear_support = LinearSupport(num_objectives=agent.reward_dim, epsilon=None)

    eval_weights = equally_spaced_weights(agent.reward_dim, n=50)

    # ── Print info ──
    print(f"\n{'='*60}")
    print(f"GPI-PD Training {'(RESUMED)' if resuming else '(NEW)'}")
    print(f"{'='*60}")
    print(f"  total_timesteps = {args.total_timesteps}")
    print(f"  iterations      = {start_iter}-{max_iter} ({max_iter - start_iter + 1} remaining)")
    print(f"  episodes/iter   ≈ {args.eval_freq // args.max_steps}")
    print(f"  eval_freq       = {args.eval_freq}")
    print(f"  ref_point       = {ref_point}")
    print(f"  device          = {args.device}")
    print(f"  seed            = {args.seed}")
    print(f"  log_dir         = {log_dir}")
    print()
    print(f"Monitor:")
    print(f"  tail -f {log_dir}/episodes.csv")
    print(f"  tail -f {log_dir}/morl_evals.csv")
    print(f"  python3 -m training.dashboard --log-dir {log_dir}")
    print(f"  python3 -m training.dashboard --log-dir {log_dir} --live 60")
    print()
    print(f"Resume after cancel:")
    print(f"  python -m training.train_gpipd --resume {log_dir}")
    print(f"{'='*60}\n")

    # ── Training loop ──
    iteration = start_iter
    try:
        for iteration in range(start_iter, max_iter + 1):
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
            if iteration % 5 == 0 or iteration == start_iter:
                morl_logger.log_iteration(agent, eval_env, iteration, agent.global_step)

            # Checkpoint every iteration (for resume)
            save_checkpoint(agent, linear_support, iteration, log_dir,
                           keep_checkpoints=args.keep_checkpoints)

    except KeyboardInterrupt:
        print(f"\n\nTraining interrupted at iteration {iteration}.")
        print("Saving checkpoint...")
        save_checkpoint(agent, linear_support, iteration, log_dir,
                       keep_checkpoints=args.keep_checkpoints)
        print(f"\nResume with:")
        print(f"  python -m training.train_gpipd --resume {log_dir}")

    else:
        # Training completed normally
        # Final MORL eval
        morl_logger.log_iteration(agent, eval_env, iteration, agent.global_step)

    # ── Generate dashboard ──
    try:
        from training.dashboard import generate_dashboard
        generate_dashboard(str(log_dir))
    except Exception as e:
        print(f"Dashboard generation failed: {e}")

    # ── Cleanup ──
    env.close()
    eval_env.close()
    logger.close()
    morl_logger.close()

    print(f"\nTraining {'complete' if iteration >= max_iter else 'paused'}.")
    print(f"Logs:      {log_dir}")
    print(f"Dashboard: {log_dir}/dashboard.png")
    if iteration < max_iter:
        print(f"Resume:    python -m training.train_gpipd --resume {log_dir}")
    else:
        print(f"Replot:    python3 -m training.dashboard --log-dir {log_dir}")


if __name__ == "__main__":
    main()
