"""MORL evaluation logger: captures HV, EU, |CCS| at each iteration.

Hooks into GPIPD's train() by wrapping it to compute and log
multi-objective metrics locally (without W&B).

Usage:
    from training.morl_eval_logger import MorlEvalLogger

    morl_logger = MorlEvalLogger(log_dir="results/logs/run_001", ref_point=ref_point)
    # After each GPIPD iteration:
    morl_logger.log_iteration(agent, eval_env, iteration, global_step)
    morl_logger.close()
"""

import csv
import json
import time
from pathlib import Path

import numpy as np
from morl_baselines.common.evaluation import policy_evaluation_mo
from morl_baselines.common.weights import equally_spaced_weights


class MorlEvalLogger:
    """Logs MORL multi-policy metrics (HV, EU, |CCS|) to CSV."""

    def __init__(self, log_dir: str, ref_point: np.ndarray,
                 num_eval_weights: int = 20, num_eval_episodes: int = 1):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.ref_point = ref_point
        self.num_eval_weights = num_eval_weights
        self.num_eval_episodes = num_eval_episodes

        self.eval_weights = equally_spaced_weights(len(ref_point), n=num_eval_weights)

        # CSV for iteration-level MORL metrics (append if exists)
        eval_path = self.log_dir / "morl_evals.csv"
        append = eval_path.exists() and eval_path.stat().st_size > 0
        self._eval_file = open(eval_path, "a" if append else "w", newline="")
        self._eval_writer = csv.writer(self._eval_file)
        if not append:
            self._eval_writer.writerow([
                "iteration", "global_step", "wall_time_s",
                "HV", "EU", "CCS_size",
                "mean_return_r1", "mean_return_r2", "mean_return_r3",
            ])

        # JSON for Pareto front snapshots (load existing if resuming)
        snap_path = self.log_dir / "pareto_snapshots.json"
        if snap_path.exists():
            with open(snap_path) as f:
                self._snapshots = json.load(f)
        else:
            self._snapshots = []
        self._start_time = time.time()

    def log_iteration(self, agent, eval_env, iteration: int, global_step: int):
        """Evaluate agent and log MORL metrics for one iteration."""
        from pymoo.indicators.hv import HV as HyperVolume

        # Evaluate on all test weights
        returns = []
        for w in self.eval_weights:
            _, _, _, disc_vec_return = policy_evaluation_mo(
                agent, eval_env, w, rep=self.num_eval_episodes
            )
            returns.append(disc_vec_return)

        returns_arr = np.array(returns)

        # Hypervolume
        try:
            hv_ind = HyperVolume(ref_point=-self.ref_point)
            hv = float(hv_ind(-returns_arr))
        except Exception:
            hv = 0.0

        # Expected Utility
        eu = float(np.mean([np.dot(w, r) for w, r in zip(self.eval_weights, returns_arr)]))

        # |CCS| (weight support size)
        ccs_size = len(agent.weight_support) if hasattr(agent, "weight_support") else 0

        # Mean returns
        mean_r = returns_arr.mean(axis=0)

        elapsed = time.time() - self._start_time

        self._eval_writer.writerow([
            iteration, global_step, round(elapsed, 1),
            round(hv, 4), round(eu, 4), ccs_size,
            round(float(mean_r[0]), 4),
            round(float(mean_r[1]), 4),
            round(float(mean_r[2]), 4),
        ])
        self._eval_file.flush()

        # Save Pareto front snapshot
        self._snapshots.append({
            "iteration": iteration,
            "global_step": global_step,
            "front": returns_arr.tolist(),
            "HV": hv,
            "EU": eu,
            "CCS_size": ccs_size,
        })

        print(f"  [MORL Eval] iter={iteration} step={global_step} "
              f"HV={hv:.2f} EU={eu:.4f} |CCS|={ccs_size}")

    def save_snapshots(self):
        """Save all Pareto front snapshots to JSON."""
        with open(self.log_dir / "pareto_snapshots.json", "w") as f:
            json.dump(self._snapshots, f, indent=2)

    def close(self):
        self._eval_file.close()
        self.save_snapshots()
