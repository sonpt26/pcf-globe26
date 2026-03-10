"""Test running a short ns-3 episode with random actions."""

import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from env.action_space import decode_action, N_ACTIONS

NS3_BINARY = Path(__file__).parent.parent.parent / "5g-factory-sim" / "ns-3" / \
    "build" / "contrib" / "nr" / "examples" / "ns3.46-pcf-morl-scenario-default"

MAX_STEPS = 3  # Short test (ns-3 is slow on 2 cores)


def test_ns3_episode():
    """Run a short episode communicating with ns-3 via pipes."""
    if not NS3_BINARY.exists():
        print(f"SKIP: ns-3 binary not found at {NS3_BINARY}")
        return

    cmd = [
        str(NS3_BINARY),
        "--seed=42",
        "--scenario=training",
        "--stepDuration=500",
        f"--maxSteps={MAX_STEPS}",
        "--eRef=1.0",
    ]

    print(f"Launching: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    try:
        # Send reset
        proc.stdin.write(json.dumps({"action": "reset"}) + "\n")
        proc.stdin.flush()

        # Read initial observation
        resp_line = proc.stdout.readline()
        if not resp_line:
            stderr = proc.stderr.read()
            print(f"FAIL: ns-3 terminated on reset. stderr:\n{stderr}")
            return
        data = json.loads(resp_line.strip())
        print(f"Reset response: obs_shape={len(data['observation'])}, step={data['step']}")
        assert len(data["observation"]) == 12, "Expected 12-dim observation"

        # Run steps with random actions
        for step in range(MAX_STEPS):
            action_id = np.random.randint(N_ACTIONS)
            rate_u, rate_e = decode_action(action_id)

            msg = json.dumps({
                "action": "step",
                "rate_urllc_mbps": rate_u,
                "rate_embb_mbps": rate_e,
            })
            proc.stdin.write(msg + "\n")
            proc.stdin.flush()

            resp_line = proc.stdout.readline()
            if not resp_line:
                stderr = proc.stderr.read()
                print(f"FAIL: ns-3 terminated at step {step}. stderr:\n{stderr}")
                return

            data = json.loads(resp_line.strip())
            obs = data["observation"]
            reward = data.get("reward", [0, 0, 0])
            done = data.get("done", False)
            kpis = data.get("kpis", {})

            print(f"Step {step}: action=({rate_u},{rate_e}) "
                  f"obs={[f'{x:.3f}' for x in obs]} "
                  f"reward={[f'{x:.4f}' for x in reward]} "
                  f"done={done}")
            print(f"  KPIs: embb_thr={kpis.get('embb_mean_throughput_mbps', 0):.2f} Mbps, "
                  f"urllc_delay_95={kpis.get('urllc_delay_95th_ms', 0):.3f} ms, "
                  f"VR={kpis.get('urllc_delay_violation_frac', 0):.4f}")

            assert len(obs) == 12, f"Expected 12-dim obs, got {len(obs)}"
            assert len(reward) == 3, f"Expected 3-dim reward, got {len(reward)}"

            if done:
                print(f"Episode done at step {step}")
                break

        print("\nns-3 episode test: PASS")

    finally:
        proc.terminate()
        proc.wait(timeout=10)


if __name__ == "__main__":
    test_ns3_episode()
