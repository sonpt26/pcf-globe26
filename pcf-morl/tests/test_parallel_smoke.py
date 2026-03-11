"""M3 smoke test: Run 2 parallel workers for 1 sync round each.

Verifies:
  1. Two workers launch and train independently against separate ns-3 instances
  2. Workers send weight support to coordinator
  3. Coordinator merges and broadcasts back
  4. Both workers complete without error
  5. Results JSON is written with timing data

Uses tiny config: 2 workers × 1 episode × 5 steps = 10 total timesteps.

Usage:
    python3 tests/test_parallel_smoke.py
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from parallel.coordinator import run_coordinator

RESULTS_DIR = ROOT / "results" / "parallel"


def main():
    print("M3 Parallel Smoke Test")
    print("=" * 50)
    print("Config: 2 workers, 1 episode/sync, 5 steps/episode")
    print()

    t0 = time.time()
    result = run_coordinator(
        num_workers=2,
        total_episodes=1,          # 1 episode per worker (= 1 sync round)
        episodes_per_sync=1,       # K=1: sync every episode
        max_steps=5,               # 5 steps per episode (short)
        base_seed=42,
        device="cuda",
    )
    elapsed = time.time() - t0

    # ── Validation ──
    print(f"\n{'='*50}")
    print("Validation:")
    checks = []

    checks.append(("Result returned", result is not None))
    checks.append(("2 workers configured", result.get("num_workers") == 2))
    checks.append((f"Wall time > 0 ({result.get('total_wall_time_s', 0):.1f}s)",
                    result.get("total_wall_time_s", 0) > 0))

    timings = result.get("worker_timings", [])
    checks.append((f"Both workers reported ({len(timings)}/2)",
                    len(timings) == 2))

    if len(timings) == 2:
        checks.append(("Worker 0 completed",
                        timings[0].get("episodes", 0) >= 1))
        checks.append(("Worker 1 completed",
                        timings[1].get("episodes", 0) >= 1))

    sync_rounds = result.get("sync_rounds", 0)
    checks.append((f"≥1 sync round ({sync_rounds})", sync_rounds >= 1))

    merged_counts = result.get("merged_weight_counts", [])
    if merged_counts:
        checks.append((f"Merged weights > 0 ({merged_counts[-1]})",
                        merged_counts[-1] > 0))

    # Check results file
    out_file = RESULTS_DIR / "scaling_P2.json"
    checks.append((f"Results file exists", out_file.exists()))

    if out_file.exists():
        with open(out_file) as f:
            saved = json.load(f)
        checks.append(("Saved JSON has worker_timings",
                        "worker_timings" in saved))

    all_pass = True
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        if not ok:
            all_pass = False

    status = "M3 PARALLEL SMOKE PASSED" if all_pass else "M3 PARALLEL SMOKE FAILED"
    print(f"\n{status} (total {elapsed:.1f}s)")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
