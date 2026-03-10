"""Quick test - just launch ns-3 and check if it accepts reset+step."""
import json
import subprocess
import sys
import time
from pathlib import Path

NS3_BINARY = Path(__file__).parent.parent.parent / "5g-factory-sim" / "ns-3" / \
    "build" / "contrib" / "nr" / "examples" / "ns3.46-pcf-morl-scenario-default"

cmd = [str(NS3_BINARY), "--seed=1", "--stepDuration=500", "--maxSteps=3", "--eRef=1.0"]
print(f"CMD: {' '.join(cmd)}")

proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE, text=True, bufsize=1)

try:
    # Reset
    print("Sending reset...")
    proc.stdin.write('{"action":"reset"}\n')
    proc.stdin.flush()

    print("Waiting for response (10s timeout)...")
    import select
    ready, _, _ = select.select([proc.stdout], [], [], 30)
    if ready:
        line = proc.stdout.readline()
        print(f"Got: {line[:200]}")
    else:
        print("TIMEOUT waiting for reset response")
        stderr = proc.stderr.read(1000) if proc.stderr else ""
        print(f"stderr: {stderr[:500]}")
        proc.kill()
        sys.exit(1)

    # First action
    print("Sending first step...")
    proc.stdin.write('{"action":"step","rate_urllc_mbps":10,"rate_embb_mbps":50}\n')
    proc.stdin.flush()

    ready, _, _ = select.select([proc.stdout], [], [], 60)
    if ready:
        line = proc.stdout.readline()
        print(f"Step 1: {line[:200]}")
    else:
        print("TIMEOUT at step 1")
        stderr = proc.stderr.read(1000) if proc.stderr else ""
        print(f"stderr: {stderr[:500]}")
        proc.kill()
        sys.exit(1)

    # Second action
    proc.stdin.write('{"action":"step","rate_urllc_mbps":15,"rate_embb_mbps":80}\n')
    proc.stdin.flush()

    ready, _, _ = select.select([proc.stdout], [], [], 60)
    if ready:
        line = proc.stdout.readline()
        print(f"Step 2: {line[:200]}")
    else:
        print("TIMEOUT at step 2")

    print("SUCCESS")

finally:
    proc.terminate()
    proc.wait(timeout=5)
