"""M3: Parallel worker – runs GPI-PD agent with its own ns-3 instance.

Each worker:
  1. Creates a PcfMorlEnv (spawns its own ns-3 subprocess)
  2. Trains GPI-PD for K episodes via train_iteration()
  3. Sends weight support (M) + Q-net state dicts to coordinator
  4. Receives merged weight support from coordinator
  5. Repeats until total episodes reached

Communication: multiprocessing.Queue (coordinator ↔ worker).
"""

import io
import time
import traceback
from dataclasses import dataclass, field
from multiprocessing import Process, Queue
from pathlib import Path
from typing import Any

import numpy as np
import torch as th


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
    e_ref: float = 1.5e-7
    device: str = "auto"
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
    payload: Any = None  # weight_support list or Q-net state bytes


def _serialize_state(agent) -> dict:
    """Extract shareable state from GPI-PD agent.

    Returns dict with:
      - weight_support: list of numpy weight vectors
      - q_net_state: serialised Q-net state_dict bytes
    """
    ws = []
    if hasattr(agent, "weight_support"):
        ws = [w.cpu().numpy().tolist() for w in agent.weight_support]

    # Serialise first Q-net state dict to bytes
    buf = io.BytesIO()
    th.save(agent.q_nets[0].state_dict(), buf)
    q_bytes = buf.getvalue()

    return {
        "weight_support": ws,
        "q_net_bytes": q_bytes,
    }


def _apply_merged_weights(agent, merged_weights: list):
    """Apply merged weight support to the agent."""
    if merged_weights:
        agent.set_weight_support([np.array(w) for w in merged_weights])


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

        # Resolve device
        if config.device == "auto":
            device = "cuda" if th.cuda.is_available() else "cpu"
        else:
            device = config.device

        eval_env = PcfMorlEnv(
            ns3_path=config.ns3_path,
            seed=config.seed + 500,
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
            dyna=False,  # Disable dyna in parallel (memory-heavy)
            gpi_pd=True,
            log=False,
            seed=config.seed,
            device=device,
        )

        ref_point = np.array([0.0, -float(config.max_steps), -float(config.max_steps)])
        episodes_done = 0
        K = config.episodes_per_sync
        steps_per_sync = K * config.max_steps

        t0 = time.time()

        while episodes_done < config.total_episodes:
            # Train for K episodes
            agent.train(
                total_timesteps=steps_per_sync,
                eval_env=eval_env,
                ref_point=ref_point,
                eval_freq=steps_per_sync + 1,  # No mid-sync eval
                eval_mo_freq=steps_per_sync + 1,
                timesteps_per_iter=steps_per_sync,
                num_eval_weights_for_front=3,
                num_eval_episodes_for_front=1,
                checkpoints=False,
            )
            episodes_done += K

            # Extract state for sharing
            state = _serialize_state(agent)

            # Send update to coordinator
            to_coord.put(SyncMessage(
                worker_id=config.worker_id,
                msg_type="sf_update",
                episode_count=episodes_done,
                wall_time_s=time.time() - t0,
                payload=state,
            ))

            # Wait for merged weights from coordinator
            merge_msg: SyncMessage = from_coord.get(timeout=600)
            if merge_msg.msg_type == "sf_merge" and merge_msg.payload is not None:
                _apply_merged_weights(agent, merge_msg.payload)

            print(f"[Worker {config.worker_id}] episodes={episodes_done}/{config.total_episodes} "
                  f"time={time.time() - t0:.1f}s "
                  f"|M|={len(agent.weight_support)}")

        # Signal completion
        to_coord.put(SyncMessage(
            worker_id=config.worker_id,
            msg_type="done",
            episode_count=episodes_done,
            wall_time_s=time.time() - t0,
        ))

        env.close()
        eval_env.close()

    except Exception as e:
        to_coord.put(SyncMessage(
            worker_id=config.worker_id,
            msg_type="error",
            payload=f"{e}\n{traceback.format_exc()}",
        ))
