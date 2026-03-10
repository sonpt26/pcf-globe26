"""Full 100-step episode validation with random actions."""
import json
import subprocess
import sys
import select
import random
from pathlib import Path

NS3_BINARY = Path(__file__).parent.parent.parent / "5g-factory-sim" / "ns-3" / \
    "build" / "contrib" / "nr" / "examples" / "ns3.46-pcf-morl-scenario-default"

sys.path.insert(0, str(Path(__file__).parent.parent))
from env.action_space import decode_action, N_ACTIONS

STEPS = 100
EREF = 1.8e-7


def main():
    if not NS3_BINARY.exists():
        print(f"SKIP: binary not found")
        return

    cmd = [str(NS3_BINARY), f"--seed=42", "--stepDuration=500",
           f"--maxSteps={STEPS}", f"--eRef={EREF}"]
    print(f"Running {STEPS}-step episode with random actions (eRef={EREF})")

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, bufsize=1)
    random.seed(42)

    try:
        proc.stdin.write('{"action":"reset"}\n')
        proc.stdin.flush()
        ready, _, _ = select.select([proc.stdout], [], [], 30)
        if not ready:
            print("TIMEOUT on reset")
            return
        line = proc.stdout.readline()
        data = json.loads(line.strip())
        print(f"Reset OK: obs[{len(data['observation'])}]")

        rewards = []
        for step in range(STEPS):
            action_id = random.randint(0, N_ACTIONS - 1)
            rate_u, rate_e = decode_action(action_id)

            msg = json.dumps({
                "action": "step",
                "rate_urllc_mbps": rate_u,
                "rate_embb_mbps": rate_e,
            })
            proc.stdin.write(msg + "\n")
            proc.stdin.flush()

            ready, _, _ = select.select([proc.stdout], [], [], 120)
            if not ready:
                print(f"TIMEOUT at step {step}")
                break
            line = proc.stdout.readline()
            if not line:
                print(f"EOF at step {step}")
                break
            data = json.loads(line.strip())
            r = data.get("reward", [0, 0, 0])
            rewards.append(r)

            if step % 20 == 0 or step == STEPS - 1:
                kpis = data.get("kpis", {})
                print(f"  Step {step:3d}: a={action_id:2d}(u={rate_u:.0f},e={rate_e:.0f}) "
                      f"r=[{r[0]:.3f},{r[1]:.3f},{r[2]:.3f}] "
                      f"embb_thr={kpis.get('embb_mean_throughput_mbps', 0):.2f} "
                      f"delay={kpis.get('urllc_delay_95th_ms', 0):.2f}ms "
                      f"VR={kpis.get('urllc_delay_violation_frac', 0):.3f}")

            if data.get("done"):
                print(f"Done at step {step}")
                break

        if rewards:
            import numpy as np
            r_arr = np.array(rewards)
            print(f"\n--- Episode Summary ({len(rewards)} steps) ---")
            print(f"r1 (eMBB thr):  mean={r_arr[:,0].mean():.4f}, min={r_arr[:,0].min():.4f}, max={r_arr[:,0].max():.4f}")
            print(f"r2 (URLLC del): mean={r_arr[:,1].mean():.4f}, min={r_arr[:,1].min():.4f}, max={r_arr[:,1].max():.4f}")
            print(f"r3 (energy):    mean={r_arr[:,2].mean():.4f}, min={r_arr[:,2].min():.4f}, max={r_arr[:,2].max():.4f}")

            # Validation checks
            checks = []
            checks.append(("100 steps completed", len(rewards) == STEPS))
            checks.append(("r1 in [0,1]", r_arr[:,0].min() >= 0 and r_arr[:,0].max() <= 1))
            checks.append(("r2 in [-1,0]", r_arr[:,1].min() >= -1 and r_arr[:,1].max() <= 0))
            checks.append(("r3 in [-1,0]", r_arr[:,2].min() >= -1 and r_arr[:,2].max() <= 0))
            checks.append(("r1 has variance", r_arr[:,0].std() > 0.01))
            checks.append(("r3 has variance", r_arr[:,2].std() > 0.01))

            print("\nValidation:")
            all_pass = True
            for name, ok in checks:
                print(f"  {name}: {'PASS' if ok else 'FAIL'}")
                if not ok:
                    all_pass = False
            print(f"\n{'M1 VALIDATION PASSED' if all_pass else 'M1 VALIDATION FAILED'}")

    finally:
        proc.terminate()
        proc.wait(timeout=10)


if __name__ == "__main__":
    main()
