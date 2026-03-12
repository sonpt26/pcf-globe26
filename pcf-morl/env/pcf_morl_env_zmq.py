"""PCF-MORL MO-Gymnasium Environment (ZMQ variant).

Same interface as PcfMorlEnv but uses ZMQ REQ/REP sockets instead of
subprocess stdin/stdout pipes.

Key advantage: ns-3 process stays alive across episodes (no restart overhead).
The C++ side rebuilds the simulation topology on each reset but skips
process spawn, library loading, and link setup.

Protocol:
    Python (REQ) → C++ (REP)
    {"action":"reset"}  → initial obs
    {"action":"step", "rate_urllc_mbps":X, "rate_embb_mbps":Y} → step obs
    {"action":"close"} → {"status":"closed"}
"""

import json
import subprocess
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import zmq

from .action_space import ACTION_TABLE, N_ACTIONS, decode_action

DEFAULT_NS3_PATH = Path(__file__).parent.parent.parent / "5g-factory-sim" / "ns-3"
DEFAULT_ZMQ_BINARY = "build/contrib/nr/examples/ns3.46-pcf-morl-scenario-zmq-default"


def _find_free_port() -> int:
    """Find a free TCP port."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class PcfMorlEnvZmq(gym.Env):
    """Multi-objective 5G network slicing environment (ZMQ variant)."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        ns3_path: str | None = None,
        step_duration_ms: float = 500.0,
        max_steps: int = 100,
        scenario: str = "training",
        seed: int = 0,
        e_ref: float = 1.8e-7,
        zmq_port: int | None = None,
    ):
        super().__init__()

        self.ns3_path = Path(ns3_path) if ns3_path else DEFAULT_NS3_PATH
        self.binary = self.ns3_path / DEFAULT_ZMQ_BINARY
        self.step_duration_ms = step_duration_ms
        self.max_steps = max_steps
        self.scenario = scenario
        self.seed_val = seed
        self.e_ref = e_ref
        self.zmq_port = zmq_port or _find_free_port()

        # Spaces (same as pipe version)
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
        self._ctx: zmq.Context | None = None
        self._socket: zmq.Socket | None = None
        self._episode_count = 0

    def _start_ns3(self):
        """Launch ns-3 ZMQ server process and connect."""
        if not self.binary.exists():
            raise FileNotFoundError(
                f"ns-3 ZMQ binary not found at {self.binary}. "
                "Build with: cmake --build cmake-cache --target pcf-morl-scenario-zmq"
            )

        cmd = [
            str(self.binary),
            f"--seed={self.seed_val}",
            f"--scenario={self.scenario}",
            f"--stepDuration={self.step_duration_ms}",
            f"--maxSteps={self.max_steps}",
            f"--eRef={self.e_ref}",
            f"--zmqPort={self.zmq_port}",
        ]
        self._process = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Setup ZMQ REQ socket
        self._ctx = zmq.Context()
        self._socket = self._ctx.socket(zmq.REQ)
        self._socket.setsockopt(zmq.RCVTIMEO, 60000)  # 60s timeout
        self._socket.setsockopt(zmq.SNDTIMEO, 10000)   # 10s send timeout
        self._socket.setsockopt(zmq.LINGER, 0)

        endpoint = f"tcp://localhost:{self.zmq_port}"

        # Give ns-3 time to bind the port
        time.sleep(0.5)
        self._socket.connect(endpoint)

    def _send(self, msg: dict) -> dict:
        """Send JSON via ZMQ REQ and receive reply."""
        self._socket.send_string(json.dumps(msg))
        resp = self._socket.recv_string()
        return json.loads(resp)

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        if seed is not None:
            self.seed_val = seed

        # First call: start ns-3 process
        if self._process is None:
            self._start_ns3()

        # Check if process is still alive
        if self._process.poll() is not None:
            # Process died, restart it
            self._cleanup_zmq()
            self.zmq_port = _find_free_port()
            self._start_ns3()

        # Send reset
        reset_msg = {"action": "reset", "seed": self.seed_val}
        data = self._send(reset_msg)

        obs = np.array(data["observation"], dtype=np.float32)
        obs = np.clip(obs, 0.0, 1.0)
        self._step_count = 0
        self._episode_count += 1

        info = {
            "kpis": data.get("kpis", {}),
            "sim_time_s": data.get("sim_time_s", 0.0),
        }
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, np.ndarray, bool, bool, dict]:
        rate_urllc, rate_embb = decode_action(action)

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

    def _cleanup_zmq(self):
        """Clean up ZMQ socket and context."""
        if self._socket is not None:
            try:
                self._socket.close()
            except Exception:
                pass
            self._socket = None
        if self._ctx is not None:
            try:
                self._ctx.term()
            except Exception:
                pass
            self._ctx = None

    def close(self):
        if self._socket is not None:
            try:
                self._send({"action": "close"})
            except Exception:
                pass

        self._cleanup_zmq()

        if self._process is not None:
            try:
                self._process.terminate()
                self._process.wait(timeout=5)
            except Exception:
                if self._process is not None:
                    self._process.kill()
            self._process = None

    def render(self):
        pass
