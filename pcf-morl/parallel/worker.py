"""M3: Parallel worker – runs GPI-PD agent with its own ns-3 instance.

Each worker:
  1. Creates a PcfMorlEnv (spawns its own ns-3 subprocess)
  2. Trains GPI-PD for K episodes
  3. Sends successor features (SF) back to coordinator
  4. Receives merged SF from coordinator
  5. Repeats until total episodes reached

Communication: multiprocessing.Queue (coordinator ↔ worker).
"""

import time
import traceback
from dataclasses import dataclass, field
from multiprocessing import Process, Queue
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class WorkerConfig:
    """Configuration for a single worker."""
    worker_id: int
    seed: int
    max_steps: int = 100
    episodes_per_sync: int = 200  # K
    total_episodes: int = 10_000
    ns3_path: str | None = None
    scenario: str = "training"
    e_ref: float = 1.8e-7
    # GPI-PD hyperparameters
    learning_rate: float = 3e-4
    gamma: float = 0.99
    batch_size: int = 128
    buffer_size: int = 500_000
    net_arch: list = field(default_factory=lambda: [256, 256, 256, 256])


@dataclass
class SyncMessage:
    """Message exchanged between worker and coordinator."""
    worker_id: int
    msg_type: str  # "sf_update" | "sf_merge" | "done" | "error"
    episode_count: int = 0
    wall_time_s: float = 0.0
    hv: float = 0.0
    payload: Any = None  # Serialised successor features


def worker_process(config: WorkerConfig, to_coord: Queue, from_coord: Queue):
    """Main worker loop. Runs in a separate process."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from env.pcf_morl_env import PcfMorlEnv

    try:
        from morl_baselines.multi_policy.gpi_pd.gpi_pd import GPIPD
    except ImportError:
        to_coord.put(SyncMessage(
            worker_id=config.worker_id,
            msg_type="error",
            payload="morl-baselines not installed",
        ))
        return

    try:
        env = PcfMorlEnv(
            ns3_path=config.ns3_path,
            seed=config.seed,
            max_steps=config.max_steps,
            scenario=config.scenario,
            e_ref=config.e_ref,
        )

        agent = GPIPD(
            env=env,
            learning_rate=config.learning_rate,
            gamma=config.gamma,
            batch_size=config.batch_size,
            net_arch=config.net_arch,
            buffer_size=config.buffer_size,
            target_net_update_freq=1000,
            tau=1.0,
            initial_epsilon=0.01,
            num_nets=2,
            gradient_updates=20,
            per=True,
            dyna=True,
            gpi_pd=True,
            dynamics_rollout_freq=250,
            dynamics_rollout_starts=5000,
            dynamics_rollout_batch_size=25000,
            dynamics_buffer_size=100000,
            log=False,
            seed=config.seed,
        )

        ref_point = np.array([0.0, -100.0, -100.0])
        total_steps_done = 0
        episodes_done = 0
        K = config.episodes_per_sync
        steps_per_sync = K * config.max_steps

        t0 = time.time()

        while episodes_done < config.total_episodes:
            # Train for K episodes
            agent.train(
                total_timesteps=steps_per_sync,
                eval_env=None,
                ref_point=ref_point,
                eval_freq=steps_per_sync + 1,  # No eval during sync interval
            )
            total_steps_done += steps_per_sync
            episodes_done += K

            # Extract successor features for sharing
            sf_data = None
            if hasattr(agent, "successor_features"):
                sf_data = [sf.cpu().numpy().tolist() for sf in agent.successor_features]

            # Send SF update to coordinator
            to_coord.put(SyncMessage(
                worker_id=config.worker_id,
                msg_type="sf_update",
                episode_count=episodes_done,
                wall_time_s=time.time() - t0,
                payload=sf_data,
            ))

            # Wait for merged SF from coordinator
            merge_msg: SyncMessage = from_coord.get(timeout=300)
            if merge_msg.msg_type == "sf_merge" and merge_msg.payload is not None:
                # Apply merged SF back to agent
                if hasattr(agent, "set_successor_features"):
                    agent.set_successor_features(merge_msg.payload)

            print(f"[Worker {config.worker_id}] episodes={episodes_done}/{config.total_episodes} "
                  f"time={time.time() - t0:.1f}s")

        # Signal completion
        to_coord.put(SyncMessage(
            worker_id=config.worker_id,
            msg_type="done",
            episode_count=episodes_done,
            wall_time_s=time.time() - t0,
        ))

        env.close()

    except Exception as e:
        to_coord.put(SyncMessage(
            worker_id=config.worker_id,
            msg_type="error",
            payload=f"{e}\n{traceback.format_exc()}",
        ))
