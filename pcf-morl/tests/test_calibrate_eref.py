"""Calibrate E_ref by running midpoint policy and measuring energy_per_bit."""
import json
import subprocess
import sys
import select
from pathlib import Path

NS3_BINARY = Path(__file__).parent.parent.parent / "5g-factory-sim" / "ns-3" / \
    "build" / "contrib" / "nr" / "examples" / "ns3.46-pcf-morl-scenario-default"

STEPS = 10
# Midpoint policy
RATE_URLLC = 10
RATE_EMBB = 50


def main():
    if not NS3_BINARY.exists():
        print(f"SKIP: binary not found at {NS3_BINARY}")
        return

    cmd = [str(NS3_BINARY), "--seed=42", "--stepDuration=500",
           f"--maxSteps={STEPS}", "--eRef=1.0"]
    print(f"CMD: {' '.join(cmd)}")

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, bufsize=1)
    try:
        # Reset
        proc.stdin.write('{"action":"reset"}\n')
        proc.stdin.flush()
        ready, _, _ = select.select([proc.stdout], [], [], 30)
        if not ready:
            print("TIMEOUT on reset")
            return
        line = proc.stdout.readline()
        data = json.loads(line.strip())
        print(f"Reset: obs has {len(data['observation'])} dims")

        epb_values = []
        for step in range(STEPS):
            msg = json.dumps({
                "action": "step",
                "rate_urllc_mbps": RATE_URLLC,
                "rate_embb_mbps": RATE_EMBB,
            })
            proc.stdin.write(msg + "\n")
            proc.stdin.flush()

            ready, _, _ = select.select([proc.stdout], [], [], 60)
            if not ready:
                print(f"TIMEOUT at step {step}")
                break
            line = proc.stdout.readline()
            if not line:
                break
            data = json.loads(line.strip())
            kpis = data.get("kpis", {})
            epb = kpis.get("energy_per_bit", 0)
            r = data.get("reward", [0, 0, 0])
            embb_thr = kpis.get("embb_mean_throughput_mbps", 0)
            urllc_delay = kpis.get("urllc_delay_95th_ms", 0)

            print(f"Step {step}: epb={epb:.3e}, r={[f'{x:.4f}' for x in r]}, "
                  f"embb_thr={embb_thr:.2f}, urllc_delay={urllc_delay:.3f}ms")
            if epb > 0:
                epb_values.append(epb)

            if data.get("done", False):
                break

        if epb_values:
            mean_epb = sum(epb_values) / len(epb_values)
            # Set eRef so that midpoint gives r3 ≈ -0.5
            recommended_eref = mean_epb / 0.5
            print(f"\n--- Calibration Results ---")
            print(f"Mean energy_per_bit: {mean_epb:.3e} J/bit")
            print(f"Min epb: {min(epb_values):.3e}, Max epb: {max(epb_values):.3e}")
            print(f"Recommended eRef (midpoint→r3≈-0.5): {recommended_eref:.3e}")
            print(f"Recommended eRef (midpoint→r3≈-0.3): {mean_epb / 0.3:.3e}")
        else:
            print("No valid epb values collected!")

    finally:
        proc.terminate()
        proc.wait(timeout=10)


if __name__ == "__main__":
    main()
