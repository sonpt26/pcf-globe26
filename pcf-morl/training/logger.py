"""Training logger: records per-episode and per-step data to CSV.

Usage:
    from training.logger import TrainingLogger

    logger = TrainingLogger("results/logs/run_001", step_level=True)
    # ... in training loop:
    logger.log_step(episode=1, step=5, obs=obs, action=3, reward=reward, info=info)
    logger.log_episode(episode=1, episode_return=ret, qos=qos, wall_time=12.3)
    logger.close()

Output files:
    <log_dir>/episodes.csv   — 1 row per episode (always written)
    <log_dir>/steps.csv      — 1 row per step (if step_level=True)
    <log_dir>/config.json    — training config snapshot

View during training:
    tail -f results/logs/run_001/episodes.csv
    python3 -m training.logger --plot results/logs/run_001
"""

import csv
import json
import time
from pathlib import Path

import numpy as np


class TrainingLogger:
    """Lightweight CSV logger for training monitoring."""

    EPISODE_HEADER = [
        "episode", "wall_time_s", "steps",
        "r1_throughput", "r2_delay", "r3_energy",
        "return_r1", "return_r2", "return_r3",
        "VR", "TTR", "MVD",
        "mean_embb_thr_mbps", "mean_urllc_delay_ms", "mean_energy_per_bit",
    ]

    STEP_HEADER = [
        "episode", "step", "action", "rate_urllc", "rate_embb",
        "r1", "r2", "r3",
        "obs_0", "obs_1", "obs_2", "obs_3", "obs_4", "obs_5",
        "obs_6", "obs_7", "obs_8", "obs_9", "obs_10", "obs_11",
        "embb_thr_mbps", "urllc_delay_95_ms", "urllc_vr", "energy_per_bit",
    ]

    def __init__(self, log_dir: str, step_level: bool = False, flush_every: int = 1):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.step_level = step_level
        self.flush_every = flush_every

        # Episode-level CSV (append if exists, else create with header)
        ep_path = self.log_dir / "episodes.csv"
        append = ep_path.exists() and ep_path.stat().st_size > 0
        self._ep_file = open(ep_path, "a" if append else "w", newline="")
        self._ep_writer = csv.writer(self._ep_file)
        if not append:
            self._ep_writer.writerow(self.EPISODE_HEADER)
        self._ep_count = 0

        # Step-level CSV (optional)
        self._step_file = None
        self._step_writer = None
        if step_level:
            step_path = self.log_dir / "steps.csv"
            step_append = step_path.exists() and step_path.stat().st_size > 0
            self._step_file = open(step_path, "a" if step_append else "w", newline="")
            self._step_writer = csv.writer(self._step_file)
            if not step_append:
                self._step_writer.writerow(self.STEP_HEADER)

        self._start_time = time.time()

    def log_step(self, episode: int, step: int, obs, action: int,
                 reward, info: dict, rate_urllc: float = 0, rate_embb: float = 0):
        """Log a single step (only if step_level=True)."""
        if not self.step_level or self._step_writer is None:
            return

        kpis = info.get("kpis", {})
        row = [
            episode, step, action, rate_urllc, rate_embb,
            float(reward[0]), float(reward[1]), float(reward[2]),
        ]
        # 12 obs values
        for i in range(12):
            row.append(float(obs[i]) if i < len(obs) else 0.0)
        row.extend([
            kpis.get("embb_mean_throughput_mbps", 0),
            kpis.get("urllc_delay_95th_ms", 0),
            kpis.get("urllc_delay_violation_frac", 0),
            kpis.get("energy_per_bit", 0),
        ])
        self._step_writer.writerow(row)

    def log_episode(self, episode: int, rewards: list = None,
                    step_infos: list = None, wall_time: float = 0):
        """Log episode summary."""
        rewards = rewards or []
        step_infos = step_infos or []

        # Last-step reward
        r1 = float(rewards[-1][0]) if rewards else 0
        r2 = float(rewards[-1][1]) if rewards else 0
        r3 = float(rewards[-1][2]) if rewards else 0

        # Discounted return
        ret = np.zeros(3)
        for t, r in enumerate(rewards):
            ret += (0.99 ** t) * np.asarray(r, dtype=np.float64)

        # QoS from step infos
        vr = 0.0
        ttr = 0.0
        mvd = 0
        if step_infos:
            violations = sum(1 for info in step_infos if info.get("VR", 0) > 0)
            vr = violations / len(step_infos)

            streak = 0
            streaks = []
            max_s = 0
            for info in step_infos:
                if info.get("VR", 0) > 0:
                    streak += 1
                    max_s = max(max_s, streak)
                else:
                    if streak > 0:
                        streaks.append(streak)
                    streak = 0
            if streak > 0:
                streaks.append(streak)
            mvd = max_s
            ttr = float(np.mean(streaks)) if streaks else 0.0

        # Mean KPIs
        mean_thr = 0.0
        mean_delay = 0.0
        mean_epb = 0.0
        if step_infos:
            kpi_vals = [info.get("kpis", {}) for info in step_infos]
            mean_thr = float(np.mean([k.get("embb_mean_throughput_mbps", 0) for k in kpi_vals]))
            mean_delay = float(np.mean([k.get("urllc_delay_95th_ms", 0) for k in kpi_vals]))
            mean_epb = float(np.mean([k.get("energy_per_bit", 0) for k in kpi_vals]))

        elapsed = time.time() - self._start_time

        self._ep_writer.writerow([
            episode, round(elapsed, 1), len(rewards),
            round(r1, 6), round(r2, 6), round(r3, 6),
            round(ret[0], 4), round(ret[1], 4), round(ret[2], 4),
            round(vr, 4), round(ttr, 2), mvd,
            round(mean_thr, 3), round(mean_delay, 3), round(mean_epb, 9),
        ])

        self._ep_count += 1
        if self._ep_count % self.flush_every == 0:
            self._ep_file.flush()
            if self._step_file:
                self._step_file.flush()

    def save_config(self, config: dict):
        """Save training config as JSON."""
        with open(self.log_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2, default=str)

    def close(self):
        self._ep_file.close()
        if self._step_file:
            self._step_file.close()


# ---------------------------------------------------------------------------
# Quick plot utility
# ---------------------------------------------------------------------------

def plot_training_log(log_dir: str):
    """Plot episode-level training curves from CSV."""
    import matplotlib.pyplot as plt
    import pandas as pd

    log_dir = Path(log_dir)
    df = pd.read_csv(log_dir / "episodes.csv")

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle(f"Training Log: {log_dir.name}", fontsize=14)

    # Returns
    for i, (col, label) in enumerate([
        ("return_r1", "Return r₁ (throughput)"),
        ("return_r2", "Return r₂ (delay)"),
        ("return_r3", "Return r₃ (energy)"),
    ]):
        ax = axes[0, i]
        ax.plot(df["episode"], df[col], alpha=0.3, linewidth=0.5)
        # Rolling average
        window = max(1, len(df) // 50)
        ax.plot(df["episode"], df[col].rolling(window).mean(), linewidth=1.5)
        ax.set_xlabel("Episode")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)

    # QoS
    axes[1, 0].plot(df["episode"], df["VR"], alpha=0.3, linewidth=0.5)
    axes[1, 0].plot(df["episode"], df["VR"].rolling(max(1, len(df)//50)).mean(), linewidth=1.5)
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("Violation Ratio")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(df["episode"], df["mean_embb_thr_mbps"], alpha=0.3, linewidth=0.5)
    axes[1, 1].plot(df["episode"], df["mean_embb_thr_mbps"].rolling(max(1, len(df)//50)).mean(), linewidth=1.5)
    axes[1, 1].set_xlabel("Episode")
    axes[1, 1].set_ylabel("eMBB Throughput (Mbps)")
    axes[1, 1].grid(True, alpha=0.3)

    # Wall time
    axes[1, 2].plot(df["episode"], df["wall_time_s"] / 3600)
    axes[1, 2].set_xlabel("Episode")
    axes[1, 2].set_ylabel("Wall time (hours)")
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = log_dir / "training_curves.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved to {out_path}")
    plt.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", type=str, help="Path to log directory to plot")
    args = parser.parse_args()

    if args.plot:
        plot_training_log(args.plot)
