"""PCF-MORL MO-Gymnasium Environment.

Wraps ns-3 5G-LENA simulation via subprocess pipe (stdin/stdout JSON).
Observation: R^12 (per-slice + system metrics, normalized [0,1])
Action: Discrete(76) -> (rate_urllc, rate_embb)
Reward: R^3 (throughput QoS, delay QoS, energy efficiency)
"""

import json
import subprocess
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from .action_space import ACTION_TABLE, N_ACTIONS, decode_action

# Default ns-3 binary path
DEFAULT_NS3_PATH = Path(__file__).parent.parent.parent / "5g-factory-sim" / "ns-3"
DEFAULT_BINARY = "build/contrib/nr/examples/ns3.46-pcf-morl-scenario-default"


class PcfMorlEnv(gym.Env):
    """Multi-objective 5G network slicing environment."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        ns3_path: str | None = None,
        step_duration_ms: float = 500.0,
        max_steps: int = 100,
        scenario: str = "training",
        seed: int = 0,
        e_ref: float = 1.8e-7,
    ):
        super().__init__()

        self.ns3_path = Path(ns3_path) if ns3_path else DEFAULT_NS3_PATH
        self.binary = self.ns3_path / DEFAULT_BINARY
        self.step_duration_ms = step_duration_ms
        self.max_steps = max_steps
        self.scenario = scenario
        self.seed_val = seed
        self.e_ref = e_ref

        # Spaces
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(12,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(N_ACTIONS)
        self.reward_space = gym.spaces.Box(
            low=np.array([0.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        )

        self._step_count = 0
        self._process: subprocess.Popen | None = None

    def _start_ns3(self):
        """Launch ns-3 simulation as a subprocess."""
        if not self.binary.exists():
            raise FileNotFoundError(
                f"ns-3 binary not found at {self.binary}. "
                "Build it with: cmake --build cmake-cache --target pcf-morl-scenario"
            )

        cmd = [
            str(self.binary),
            f"--seed={self.seed_val}",
            f"--scenario={self.scenario}",
            f"--stepDuration={self.step_duration_ms}",
            f"--maxSteps={self.max_steps}",
            f"--eRef={self.e_ref}",
        ]
        self._process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

    def _send(self, msg: dict) -> dict:
        """Send JSON message to ns-3 and receive response."""
        line = json.dumps(msg)
        self._process.stdin.write(line + "\n")
        self._process.stdin.flush()

        resp_line = self._process.stdout.readline()
        if not resp_line:
            stderr = self._process.stderr.read()
            raise RuntimeError(f"ns-3 process terminated unexpectedly. stderr: {stderr}")

        return json.loads(resp_line.strip())

    def _send_no_response(self, msg: dict):
        """Send without waiting for response."""
        line = json.dumps(msg)
        self._process.stdin.write(line + "\n")
        self._process.stdin.flush()

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        if seed is not None:
            self.seed_val = seed

        # Kill previous process if any
        if self._process is not None:
            try:
                self._process.terminate()
                self._process.wait(timeout=5)
            except Exception:
                self._process.kill()
            self._process = None

        # Start fresh ns-3 instance
        self._start_ns3()

        # Send reset command
        data = self._send({"action": "reset"})

        obs = np.array(data["observation"], dtype=np.float32)
        obs = np.clip(obs, 0.0, 1.0)
        self._step_count = 0

        info = {
            "kpis": data.get("kpis", {}),
            "sim_time_s": data.get("sim_time_s", 0.0),
        }
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, np.ndarray, bool, bool, dict]:
        rate_urllc, rate_embb = decode_action(action)

        # Send action, ns-3 simulates one step, then sends back obs
        data = self._send({
            "action": "step",
            "rate_urllc_mbps": rate_urllc,
            "rate_embb_mbps": rate_embb,
        })

        obs = np.array(data["observation"], dtype=np.float32)
        obs = np.clip(obs, 0.0, 1.0)

        reward_vec = np.array(data.get("reward", [0, 0, 0]), dtype=np.float32)

        self._step_count += 1
        terminated = False
        truncated = data.get("done", False) or self._step_count >= self.max_steps

        kpis = data.get("kpis", {})
        info = {
            "kpis": kpis,
            "sim_time_s": data.get("sim_time_s", 0.0),
            "VR": kpis.get("urllc_delay_violation_frac", 0.0),
            "TTR": kpis.get("time_to_resolve", 0.0),
            "MVD": kpis.get("max_violation_duration", 0),
            "raw_kpis": kpis,
        }
        return obs, reward_vec, terminated, truncated, info

    def close(self):
        if self._process is not None:
            try:
                self._send_no_response({"action": "close"})
                self._process.terminate()
                self._process.wait(timeout=5)
            except Exception:
                if self._process is not None:
                    self._process.kill()
            self._process = None

    def render(self):
        pass
