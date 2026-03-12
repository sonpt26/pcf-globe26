"""Compare pipe vs ZMQ environment implementations.

Measures:
  1. Reset overhead (process restart vs in-process rebuild)
  2. Per-step latency
  3. Multi-episode throughput (episodes/sec)
  4. Observation/reward consistency (both envs should produce similar distributions)

Usage:
    python3 tests/test_env_comparison.py [--episodes 5] [--steps 10]
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / "results" / "env_comparison"


def benchmark_env(EnvClass, label: str, num_episodes: int, steps_per_episode: int,
                  **env_kwargs) -> dict:
    """Benchmark an environment over multiple episodes."""
    print(f"\n{'='*50}")
    print(f"Benchmarking: {label}")
    print(f"  Episodes: {num_episodes}, Steps/ep: {steps_per_episode}")

    env = EnvClass(max_steps=steps_per_episode, **env_kwargs)

    reset_times = []
    step_times = []
    episode_times = []
    all_rewards = []
    all_obs = []

    for ep in range(num_episodes):
        # Reset
        t0 = time.perf_counter()
        obs, info = env.reset(seed=42 + ep)
        t_reset = time.perf_counter() - t0
        reset_times.append(t_reset)

        ep_rewards = []
        ep_step_times = []

        for s in range(steps_per_episode):
            action = np.random.randint(0, env.action_space.n)

            t0 = time.perf_counter()
            obs, reward, terminated, truncated, info = env.step(action)
            t_step = time.perf_counter() - t0
            ep_step_times.append(t_step)

            ep_rewards.append(reward.copy())
            all_obs.append(obs.copy())

            if terminated or truncated:
                break

        step_times.extend(ep_step_times)
        all_rewards.extend(ep_rewards)
        episode_times.append(sum(ep_step_times) + t_reset)

        print(f"  Episode {ep+1}/{num_episodes}: "
              f"reset={t_reset:.3f}s, "
              f"avg_step={np.mean(ep_step_times):.4f}s, "
              f"total={episode_times[-1]:.3f}s")

    env.close()

    total_time = sum(episode_times)
    results = {
        "label": label,
        "num_episodes": num_episodes,
        "steps_per_episode": steps_per_episode,
        "total_time_s": total_time,
        "episodes_per_sec": num_episodes / total_time if total_time > 0 else 0,
        "reset_times": {
            "mean": float(np.mean(reset_times)),
            "std": float(np.std(reset_times)),
            "min": float(np.min(reset_times)),
            "max": float(np.max(reset_times)),
            "values": [float(x) for x in reset_times],
        },
        "step_times": {
            "mean": float(np.mean(step_times)),
            "std": float(np.std(step_times)),
            "p50": float(np.percentile(step_times, 50)),
            "p95": float(np.percentile(step_times, 95)),
            "p99": float(np.percentile(step_times, 99)),
        },
        "episode_times": {
            "mean": float(np.mean(episode_times)),
            "std": float(np.std(episode_times)),
        },
        "reward_stats": {
            "mean": np.mean(all_rewards, axis=0).tolist() if all_rewards else [],
            "std": np.std(all_rewards, axis=0).tolist() if all_rewards else [],
        },
        "obs_stats": {
            "mean": np.mean(all_obs, axis=0).tolist() if all_obs else [],
            "std": np.std(all_obs, axis=0).tolist() if all_obs else [],
        },
    }

    print(f"\n  Summary:")
    print(f"    Reset: {results['reset_times']['mean']:.3f}s ± {results['reset_times']['std']:.3f}s")
    print(f"    Step:  {results['step_times']['mean']:.4f}s (p50={results['step_times']['p50']:.4f}s, p95={results['step_times']['p95']:.4f}s)")
    print(f"    Total: {total_time:.1f}s ({results['episodes_per_sec']:.2f} eps/s)")

    return results


def compare_distributions(pipe_results: dict, zmq_results: dict) -> list:
    """Compare reward/obs distributions between two envs."""
    checks = []

    # Reward means should be within tolerance
    if pipe_results["reward_stats"]["mean"] and zmq_results["reward_stats"]["mean"]:
        pipe_r = np.array(pipe_results["reward_stats"]["mean"])
        zmq_r = np.array(zmq_results["reward_stats"]["mean"])
        # Same seed + same actions → rewards should be similar (not exact due to RNG)
        # Allow 50% relative tolerance since episodes use different seeds
        diff = np.abs(pipe_r - zmq_r)
        max_diff = np.max(diff)
        checks.append((f"Reward means similar (max_diff={max_diff:.4f})", max_diff < 0.5))

    # Obs means should be within tolerance
    if pipe_results["obs_stats"]["mean"] and zmq_results["obs_stats"]["mean"]:
        pipe_o = np.array(pipe_results["obs_stats"]["mean"])
        zmq_o = np.array(zmq_results["obs_stats"]["mean"])
        diff = np.abs(pipe_o - zmq_o)
        max_diff = np.max(diff)
        checks.append((f"Obs means similar (max_diff={max_diff:.4f})", max_diff < 0.5))

    return checks


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Compare pipe vs ZMQ env")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--pipe-only", action="store_true", help="Only run pipe env")
    parser.add_argument("--zmq-only", action="store_true", help="Only run ZMQ env")
    args = parser.parse_args()

    print("PCF-MORL Environment Comparison: Pipe vs ZMQ")
    print("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    pipe_results = None
    zmq_results = None

    # ── Pipe environment ──
    if not args.zmq_only:
        from env.pcf_morl_env import PcfMorlEnv
        pipe_results = benchmark_env(
            PcfMorlEnv, "Pipe (stdin/stdout)",
            num_episodes=args.episodes,
            steps_per_episode=args.steps,
        )

    # ── ZMQ environment ──
    if not args.pipe_only:
        try:
            from env.pcf_morl_env_zmq import PcfMorlEnvZmq
            zmq_results = benchmark_env(
                PcfMorlEnvZmq, "ZMQ (tcp)",
                num_episodes=args.episodes,
                steps_per_episode=args.steps,
            )
        except FileNotFoundError as e:
            print(f"\n  SKIP ZMQ: {e}")
            print("  Build the ZMQ binary first (see sim/pcf-morl-scenario-zmq.cc)")

    # ── Comparison ──
    print(f"\n{'='*60}")
    print("COMPARISON")
    print(f"{'='*60}")

    checks = []

    if pipe_results:
        checks.append(("Pipe env runs", True))
        print(f"\n  Pipe:")
        print(f"    Reset: {pipe_results['reset_times']['mean']:.3f}s")
        print(f"    Step:  {pipe_results['step_times']['mean']:.4f}s")
        print(f"    Rate:  {pipe_results['episodes_per_sec']:.2f} eps/s")

    if zmq_results:
        checks.append(("ZMQ env runs", True))
        print(f"\n  ZMQ:")
        print(f"    Reset: {zmq_results['reset_times']['mean']:.3f}s")
        print(f"    Step:  {zmq_results['step_times']['mean']:.4f}s")
        print(f"    Rate:  {zmq_results['episodes_per_sec']:.2f} eps/s")

    if pipe_results and zmq_results:
        # Speedup comparison
        pipe_reset = pipe_results["reset_times"]["mean"]
        zmq_reset = zmq_results["reset_times"]["mean"]
        reset_speedup = pipe_reset / zmq_reset if zmq_reset > 0 else float("inf")

        pipe_step = pipe_results["step_times"]["mean"]
        zmq_step = zmq_results["step_times"]["mean"]
        step_ratio = zmq_step / pipe_step if pipe_step > 0 else float("inf")

        pipe_eps = pipe_results["episodes_per_sec"]
        zmq_eps = zmq_results["episodes_per_sec"]
        throughput_speedup = zmq_eps / pipe_eps if pipe_eps > 0 else float("inf")

        print(f"\n  Speedup (ZMQ vs Pipe):")
        print(f"    Reset: {reset_speedup:.2f}x faster")
        print(f"    Step:  {step_ratio:.2f}x (ratio, <1 = ZMQ faster)")
        print(f"    Throughput: {throughput_speedup:.2f}x faster")

        checks.append((f"Reset speedup > 1x ({reset_speedup:.2f}x)", reset_speedup > 1.0))
        checks.append((f"Step latency comparable ({step_ratio:.2f}x)", step_ratio < 2.0))

        # Distribution checks
        dist_checks = compare_distributions(pipe_results, zmq_results)
        checks.extend(dist_checks)

    # ── Validation ──
    print(f"\n{'='*60}")
    print("Validation:")
    all_pass = True
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        if not ok:
            all_pass = False

    # ── Save results ──
    combined = {
        "pipe": pipe_results,
        "zmq": zmq_results,
        "checks": [(n, o) for n, o in checks],
    }
    out_file = RESULTS_DIR / "comparison.json"
    with open(out_file, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\nResults saved to {out_file}")

    # ── Recommendation ──
    print(f"\n{'='*60}")
    if pipe_results and zmq_results:
        if reset_speedup > 1.5 and step_ratio < 1.5:
            print("RECOMMENDATION: Use ZMQ env for training (faster reset, similar step latency)")
        elif reset_speedup < 0.8:
            print("RECOMMENDATION: Use Pipe env (ZMQ overhead not justified)")
        else:
            print("RECOMMENDATION: Both comparable. Use Pipe for simplicity, ZMQ for many episodes.")
    elif pipe_results:
        print("RECOMMENDATION: Only Pipe tested. Build ZMQ binary to compare.")
    elif zmq_results:
        print("RECOMMENDATION: Only ZMQ tested.")

    status = "PASSED" if all_pass else "FAILED"
    print(f"\nENV COMPARISON {status}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
