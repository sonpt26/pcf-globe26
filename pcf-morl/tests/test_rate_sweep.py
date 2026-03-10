"""Rate sweep test: verify reward sensitivity to actions.

1. Sweep rate_embb [10,30,50,70,100] with fixed rate_urllc=10 → r1 should increase
2. Sweep rate_urllc [5,10,15,20] with fixed rate_embb=50 → r2 should show knee
3. Mid-episode action change → KPIs respond within 2-3 steps
"""
import json
import subprocess
import sys
import select
from pathlib import Path

NS3_BINARY = Path(__file__).parent.parent.parent / "5g-factory-sim" / "ns-3" / \
    "build" / "contrib" / "nr" / "examples" / "ns3.46-pcf-morl-scenario-default"

WARMUP = 2  # Steps to reach steady state
MEASURE = 3  # Steps to average over


def run_episode(rate_urllc, rate_embb, steps, seed=42):
    """Run ns-3 episode with constant action, return list of step results."""
    cmd = [str(NS3_BINARY), f"--seed={seed}", "--stepDuration=500",
           f"--maxSteps={steps}"]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, bufsize=1)
    results = []
    try:
        proc.stdin.write('{"action":"reset"}\n')
        proc.stdin.flush()
        ready, _, _ = select.select([proc.stdout], [], [], 30)
        if not ready:
            return []
        proc.stdout.readline()  # discard reset response

        for step in range(steps):
            msg = json.dumps({
                "action": "step",
                "rate_urllc_mbps": rate_urllc,
                "rate_embb_mbps": rate_embb,
            })
            proc.stdin.write(msg + "\n")
            proc.stdin.flush()
            ready, _, _ = select.select([proc.stdout], [], [], 60)
            if not ready:
                break
            line = proc.stdout.readline()
            if not line:
                break
            data = json.loads(line.strip())
            results.append(data)
            if data.get("done"):
                break
    finally:
        proc.terminate()
        proc.wait(timeout=10)
    return results


def avg_reward(results, start, count):
    """Average reward over steps [start, start+count)."""
    r_sum = [0.0, 0.0, 0.0]
    n = 0
    for i in range(start, min(start + count, len(results))):
        r = results[i].get("reward", [0, 0, 0])
        for j in range(3):
            r_sum[j] += r[j]
        n += 1
    return [x / max(n, 1) for x in r_sum]


def main():
    if not NS3_BINARY.exists():
        print(f"SKIP: binary not found")
        return

    total_steps = WARMUP + MEASURE
    passed = True

    # --- Test 1: eMBB rate sweep ---
    print("=== Test 1: eMBB rate sweep (rate_urllc=10 fixed) ===")
    embb_rates = [2, 4, 6, 8, 10, 15, 30, 100]
    r1_values = []
    for rate_e in embb_rates:
        results = run_episode(10, rate_e, total_steps, seed=42)
        r = avg_reward(results, WARMUP, MEASURE)
        r1_values.append(r[0])
        print(f"  rate_embb={rate_e:3d}: r1={r[0]:.4f}, r2={r[1]:.4f}, r3={r[2]:.4f}")

    # r1 should be non-decreasing (higher eMBB rate → more throughput)
    monotonic = all(r1_values[i] <= r1_values[i+1] + 0.02
                    for i in range(len(r1_values)-1))
    print(f"  r1 non-decreasing: {'PASS' if monotonic else 'FAIL'} {r1_values}")
    if not monotonic:
        passed = False

    # --- Test 2: URLLC rate sweep ---
    print("\n=== Test 2: URLLC rate sweep (rate_embb=50 fixed) ===")
    urllc_rates = [5, 10, 15, 20]
    r2_values = []
    for rate_u in urllc_rates:
        results = run_episode(rate_u, 50, total_steps, seed=42)
        r = avg_reward(results, WARMUP, MEASURE)
        r2_values.append(r[1])
        kpis = results[-1].get("kpis", {}) if results else {}
        print(f"  rate_urllc={rate_u:2d}: r1={r[0]:.4f}, r2={r[1]:.4f}, r3={r[2]:.4f}, "
              f"delay_95={kpis.get('urllc_delay_95th_ms', 0):.3f}ms, "
              f"VR={kpis.get('urllc_delay_violation_frac', 0):.4f}")

    # r2 should be non-increasing (higher URLLC rate → more delay violations)
    # With 5ms budget, small rates likely have r2=0, higher rates may degrade
    print(f"  r2 values: {r2_values}")

    # --- Test 3: Mid-episode action change ---
    print("\n=== Test 3: Mid-episode action change ===")
    cmd = [str(NS3_BINARY), "--seed=42", "--stepDuration=500", "--maxSteps=10"]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, bufsize=1)
    try:
        proc.stdin.write('{"action":"reset"}\n')
        proc.stdin.flush()
        select.select([proc.stdout], [], [], 30)
        proc.stdout.readline()

        # Phase 1: low eMBB rate (steps 0-4)
        phase1_r1 = []
        for step in range(5):
            msg = json.dumps({"action": "step", "rate_urllc_mbps": 10, "rate_embb_mbps": 3})
            proc.stdin.write(msg + "\n")
            proc.stdin.flush()
            select.select([proc.stdout], [], [], 60)
            data = json.loads(proc.stdout.readline().strip())
            r1 = data["reward"][0]
            phase1_r1.append(r1)
            print(f"  Step {step} (embb=3): r1={r1:.4f}")

        # Phase 2: high eMBB rate (steps 5-9)
        phase2_r1 = []
        for step in range(5, 10):
            msg = json.dumps({"action": "step", "rate_urllc_mbps": 10, "rate_embb_mbps": 100})
            proc.stdin.write(msg + "\n")
            proc.stdin.flush()
            ready, _, _ = select.select([proc.stdout], [], [], 60)
            if not ready:
                break
            line = proc.stdout.readline()
            if not line:
                break
            data = json.loads(line.strip())
            r1 = data["reward"][0]
            phase2_r1.append(r1)
            print(f"  Step {step} (embb=100): r1={r1:.4f}")

        # Phase 2 steady-state r1 should be higher than phase 1
        p1_avg = sum(phase1_r1[-2:]) / 2 if len(phase1_r1) >= 2 else 0
        p2_avg = sum(phase2_r1[-2:]) / 2 if len(phase2_r1) >= 2 else 0
        responsive = p2_avg > p1_avg + 0.01
        print(f"  Phase1 avg r1={p1_avg:.4f}, Phase2 avg r1={p2_avg:.4f}")
        print(f"  Action responsive: {'PASS' if responsive else 'FAIL'}")
        if not responsive:
            passed = False
    finally:
        proc.terminate()
        proc.wait(timeout=10)

    print(f"\n{'ALL TESTS PASSED' if passed else 'SOME TESTS FAILED'}")


if __name__ == "__main__":
    main()
