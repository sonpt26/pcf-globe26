"""Training dashboard: comprehensive visualization of PCF-MORL training progress.

Generates multi-panel plots of:
  - Per-episode rewards (r1 throughput, r2 delay, r3 energy)
  - Discounted returns
  - QoS metrics (VR, TTR, MVD)
  - MORL metrics over iterations (HV, EU, |CCS|)
  - Pareto front snapshots
  - KPIs (throughput, delay, energy)
  - Training speed (episodes/hour)

Usage:
    # Plot from episode logs
    python3 -m training.dashboard --log-dir results/logs/run_001

    # Plot from episode logs + MORL eval logs
    python3 -m training.dashboard --log-dir results/logs/run_001

    # Live monitoring (re-plot every N seconds)
    python3 -m training.dashboard --log-dir results/logs/run_001 --live 30
"""

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# IEEE conference style
plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def load_episodes(log_dir: Path) -> pd.DataFrame:
    """Load episodes.csv."""
    ep_file = log_dir / "episodes.csv"
    if not ep_file.exists():
        raise FileNotFoundError(f"No episodes.csv in {log_dir}")
    return pd.read_csv(ep_file)


def load_morl_evals(log_dir: Path) -> pd.DataFrame:
    """Load morl_evals.csv (MORL-specific metrics per iteration)."""
    eval_file = log_dir / "morl_evals.csv"
    if eval_file.exists():
        return pd.read_csv(eval_file)
    return None


def load_pareto_snapshots(log_dir: Path) -> list:
    """Load pareto front snapshots."""
    snap_file = log_dir / "pareto_snapshots.json"
    if snap_file.exists():
        with open(snap_file) as f:
            return json.load(f)
    return []


def plot_episode_rewards(df: pd.DataFrame, ax_row: list):
    """Plot per-episode reward components."""
    window = max(1, len(df) // 50)
    titles = ["r1 (Throughput QoS)", "r2 (Delay Violation)", "r3 (Energy Efficiency)"]
    cols = ["r1_throughput", "r2_delay", "r3_energy"]
    colors = ["#2196F3", "#F44336", "#4CAF50"]

    for ax, col, title, color in zip(ax_row, cols, titles, colors):
        if col not in df.columns:
            continue
        ax.plot(df["episode"], df[col], alpha=0.15, linewidth=0.5, color=color)
        ax.plot(df["episode"], df[col].rolling(window).mean(), linewidth=1.5, color=color)
        ax.set_title(title)
        ax.set_xlabel("Episode")


def plot_returns(df: pd.DataFrame, ax_row: list):
    """Plot discounted returns."""
    window = max(1, len(df) // 50)
    cols = ["return_r1", "return_r2", "return_r3"]
    titles = ["Return R1", "Return R2", "Return R3"]
    colors = ["#2196F3", "#F44336", "#4CAF50"]

    for ax, col, title, color in zip(ax_row, cols, titles, colors):
        if col not in df.columns:
            continue
        ax.plot(df["episode"], df[col], alpha=0.15, linewidth=0.5, color=color)
        ax.plot(df["episode"], df[col].rolling(window).mean(), linewidth=1.5, color=color)
        ax.set_title(title)
        ax.set_xlabel("Episode")


def plot_qos(df: pd.DataFrame, ax_row: list):
    """Plot QoS metrics."""
    window = max(1, len(df) // 50)

    if "VR" in df.columns:
        ax_row[0].plot(df["episode"], df["VR"], alpha=0.15, linewidth=0.5, color="#FF9800")
        ax_row[0].plot(df["episode"], df["VR"].rolling(window).mean(), linewidth=1.5, color="#FF9800")
        ax_row[0].set_title("Violation Ratio (VR)")
        ax_row[0].set_xlabel("Episode")

    if "mean_embb_thr_mbps" in df.columns:
        ax_row[1].plot(df["episode"], df["mean_embb_thr_mbps"], alpha=0.15, linewidth=0.5, color="#2196F3")
        ax_row[1].plot(df["episode"], df["mean_embb_thr_mbps"].rolling(window).mean(), linewidth=1.5, color="#2196F3")
        ax_row[1].set_title("eMBB Throughput (Mbps)")
        ax_row[1].set_xlabel("Episode")

    if "mean_energy_per_bit" in df.columns:
        ax_row[2].plot(df["episode"], df["mean_energy_per_bit"], alpha=0.15, linewidth=0.5, color="#4CAF50")
        ax_row[2].plot(df["episode"], df["mean_energy_per_bit"].rolling(window).mean(), linewidth=1.5, color="#4CAF50")
        ax_row[2].set_title("Energy per Bit")
        ax_row[2].set_xlabel("Episode")


def plot_morl_metrics(morl_df: pd.DataFrame, ax_row: list):
    """Plot MORL evaluation metrics (HV, EU, |CCS|)."""
    if morl_df is None or morl_df.empty:
        for ax in ax_row:
            ax.text(0.5, 0.5, "No MORL eval data", ha="center", va="center", transform=ax.transAxes)
        return

    if "HV" in morl_df.columns:
        ax_row[0].plot(morl_df["iteration"], morl_df["HV"], "o-", markersize=3, color="#9C27B0")
        ax_row[0].set_title("Hypervolume (HV)")
        ax_row[0].set_xlabel("Iteration")

    if "EU" in morl_df.columns:
        ax_row[1].plot(morl_df["iteration"], morl_df["EU"], "s-", markersize=3, color="#009688")
        ax_row[1].set_title("Expected Utility (EU)")
        ax_row[1].set_xlabel("Iteration")

    if "CCS_size" in morl_df.columns:
        ax_row[2].plot(morl_df["iteration"], morl_df["CCS_size"], "^-", markersize=3, color="#795548")
        ax_row[2].set_title("|CCS| (Weight Support Size)")
        ax_row[2].set_xlabel("Iteration")


def plot_speed(df: pd.DataFrame, ax):
    """Plot training speed."""
    if "wall_time_s" not in df.columns:
        return
    hours = df["wall_time_s"] / 3600
    ax.plot(df["episode"], hours, color="#607D8B")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Wall time (hours)")
    ax.set_title("Training Progress")

    # Add episodes/hour annotation
    if len(df) > 10:
        total_h = hours.iloc[-1]
        eps_per_h = len(df) / total_h if total_h > 0 else 0
        ax.text(0.02, 0.95, f"{eps_per_h:.1f} eps/hr", transform=ax.transAxes,
                fontsize=8, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))


def plot_pareto_front(snapshots: list, ax):
    """Plot latest Pareto front snapshot (r1 vs r2)."""
    if not snapshots:
        ax.text(0.5, 0.5, "No Pareto snapshots", ha="center", va="center", transform=ax.transAxes)
        return

    latest = snapshots[-1]
    points = np.array(latest.get("front", []))
    if len(points) == 0:
        return

    ax.scatter(points[:, 0], points[:, 1], s=20, c="#9C27B0", zorder=5, label=f"Iter {latest.get('iteration', '?')}")
    ax.set_xlabel("r1 (Throughput)")
    ax.set_ylabel("r2 (Delay)")
    ax.set_title("Pareto Front (latest)")
    ax.legend()


def generate_dashboard(log_dir: str, output: str = None):
    """Generate full training dashboard."""
    log_dir = Path(log_dir)

    df = load_episodes(log_dir)
    morl_df = load_morl_evals(log_dir)
    snapshots = load_pareto_snapshots(log_dir)

    has_morl = morl_df is not None and not morl_df.empty
    n_rows = 5 if has_morl else 4

    fig, axes = plt.subplots(n_rows, 3, figsize=(14, 3.2 * n_rows))
    fig.suptitle(f"PCF-MORL Training Dashboard: {log_dir.name}", fontsize=13, fontweight="bold")

    # Row 0: Per-step rewards
    plot_episode_rewards(df, axes[0])

    # Row 1: Discounted returns
    plot_returns(df, axes[1])

    # Row 2: QoS metrics
    plot_qos(df, axes[2])

    # Row 3: MORL metrics (if available)
    if has_morl:
        plot_morl_metrics(morl_df, axes[3])
        row_extra = 4
    else:
        row_extra = 3

    # Last row: speed + pareto front
    plot_speed(df, axes[row_extra, 0])
    plot_pareto_front(snapshots, axes[row_extra, 1])

    # Summary stats in last panel
    ax_summary = axes[row_extra, 2]
    ax_summary.axis("off")
    n_eps = len(df)
    total_h = df["wall_time_s"].iloc[-1] / 3600 if "wall_time_s" in df.columns and n_eps > 0 else 0

    summary_lines = [
        f"Episodes: {n_eps}",
        f"Wall time: {total_h:.1f}h",
        f"Speed: {n_eps/total_h:.1f} eps/hr" if total_h > 0 else "",
    ]

    # Latest metrics
    window = min(50, max(1, n_eps // 10))
    if n_eps > 0:
        for col, label in [("return_r1", "R1"), ("return_r2", "R2"), ("return_r3", "R3")]:
            if col in df.columns:
                recent = df[col].tail(window).mean()
                summary_lines.append(f"Recent {label}: {recent:.3f}")

        if "VR" in df.columns:
            recent_vr = df["VR"].tail(window).mean()
            summary_lines.append(f"Recent VR: {recent_vr:.3f}")

    if has_morl:
        latest_hv = morl_df["HV"].iloc[-1] if "HV" in morl_df.columns else "N/A"
        latest_eu = morl_df["EU"].iloc[-1] if "EU" in morl_df.columns else "N/A"
        latest_ccs = morl_df["CCS_size"].iloc[-1] if "CCS_size" in morl_df.columns else "N/A"
        summary_lines.extend([f"Latest HV: {latest_hv}", f"Latest EU: {latest_eu}", f"|CCS|: {latest_ccs}"])

    summary_text = "\n".join([l for l in summary_lines if l])
    ax_summary.text(0.1, 0.9, summary_text, transform=ax_summary.transAxes,
                    fontsize=9, verticalalignment="top", fontfamily="monospace",
                    bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
    ax_summary.set_title("Summary")

    plt.tight_layout()

    out_path = Path(output) if output else log_dir / "dashboard.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Dashboard saved to {out_path}")
    return out_path


def live_monitor(log_dir: str, interval: int = 30):
    """Re-generate dashboard every `interval` seconds."""
    print(f"Live monitoring {log_dir} (refresh every {interval}s, Ctrl+C to stop)")
    while True:
        try:
            generate_dashboard(log_dir)
            print(f"  Updated at {time.strftime('%H:%M:%S')}")
        except Exception as e:
            print(f"  Error: {e}")
        time.sleep(interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PCF-MORL Training Dashboard")
    parser.add_argument("--log-dir", type=str, required=True,
                        help="Path to training log directory")
    parser.add_argument("--output", type=str, default=None,
                        help="Output image path (default: <log-dir>/dashboard.png)")
    parser.add_argument("--live", type=int, default=0,
                        help="Live refresh interval in seconds (0=one-shot)")
    args = parser.parse_args()

    if args.live > 0:
        live_monitor(args.log_dir, args.live)
    else:
        generate_dashboard(args.log_dir, args.output)
