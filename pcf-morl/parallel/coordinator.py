"""M3: Parallel coordinator – manages P workers for distributed GPI-PD training.

Architecture:
  Coordinator → P Workers → P ns-3 instances.
  SF sharing every K episodes.

Usage:
    python -m parallel.coordinator --workers 4 --episodes-per-sync 200 --total-episodes 10000
"""

import argparse
import json
import sys
import time
from multiprocessing import Queue
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from parallel.worker import WorkerConfig, SyncMessage, worker_process
from multiprocessing import Process

RESULTS_DIR = Path(__file__).parent.parent / "results" / "parallel"


def merge_successor_features(sf_list: list[Any]) -> Any:
    """Merge successor features from multiple workers.

    Strategy: union of all SF vectors (CCS-based merging).
    Each worker contributes its learned SF; the coordinator takes
    the union and broadcasts back to all workers.
    """
    if not sf_list or all(sf is None for sf in sf_list):
        return None

    valid = [sf for sf in sf_list if sf is not None]
    if not valid:
        return None

    # Union merge: concatenate all SF and deduplicate
    all_sf = []
    for sf in valid:
        if isinstance(sf, list):
            all_sf.extend(sf)

    if not all_sf:
        return None

    # Deduplicate by converting to numpy and using unique
    arr = np.array(all_sf)
    if arr.ndim == 1:
        return all_sf

    # Keep unique rows (approximate: round to 4 decimals)
    rounded = np.round(arr, decimals=4)
    _, unique_idx = np.unique(rounded, axis=0, return_index=True)
    merged = arr[sorted(unique_idx)].tolist()

    return merged


def run_coordinator(
    num_workers: int,
    total_episodes: int = 10_000,
    episodes_per_sync: int = 200,
    max_steps: int = 100,
    base_seed: int = 42,
    ns3_path: str | None = None,
    scenario: str = "training",
):
    """Launch workers and coordinate SF sharing."""

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Create communication queues
    to_coord_queues = Queue()  # Workers → Coordinator (shared)
    from_coord_queues = [Queue() for _ in range(num_workers)]  # Coordinator → Worker_i

    # Create worker configs
    configs = []
    for i in range(num_workers):
        cfg = WorkerConfig(
            worker_id=i,
            seed=base_seed + i * 1000,
            max_steps=max_steps,
            episodes_per_sync=episodes_per_sync,
            total_episodes=total_episodes,
            ns3_path=ns3_path,
            scenario=scenario,
        )
        configs.append(cfg)

    # Launch workers
    workers = []
    for i, cfg in enumerate(configs):
        p = Process(
            target=worker_process,
            args=(cfg, to_coord_queues, from_coord_queues[i]),
            name=f"worker-{i}",
        )
        p.start()
        workers.append(p)

    print(f"Launched {num_workers} workers (K={episodes_per_sync}, "
          f"total={total_episodes} episodes)")

    # Coordination loop
    t0 = time.time()
    active_workers = set(range(num_workers))
    sync_round = 0
    pending_sf: dict[int, Any] = {}
    timing_log = []

    while active_workers:
        # Collect messages from workers
        msg: SyncMessage = to_coord_queues.get(timeout=600)

        if msg.msg_type == "error":
            print(f"[Coordinator] Worker {msg.worker_id} ERROR: {msg.payload}")
            active_workers.discard(msg.worker_id)
            continue

        if msg.msg_type == "done":
            print(f"[Coordinator] Worker {msg.worker_id} DONE: "
                  f"{msg.episode_count} episodes in {msg.wall_time_s:.1f}s")
            timing_log.append({
                "worker_id": msg.worker_id,
                "episodes": msg.episode_count,
                "wall_time_s": msg.wall_time_s,
            })
            active_workers.discard(msg.worker_id)
            continue

        if msg.msg_type == "sf_update":
            pending_sf[msg.worker_id] = msg.payload
            print(f"[Coordinator] Worker {msg.worker_id}: {msg.episode_count} episodes "
                  f"({msg.wall_time_s:.1f}s)")

            # When all active workers have submitted, merge and broadcast
            if set(pending_sf.keys()) >= active_workers:
                sync_round += 1
                merged = merge_successor_features(list(pending_sf.values()))

                for wid in active_workers:
                    from_coord_queues[wid].put(SyncMessage(
                        worker_id=-1,  # From coordinator
                        msg_type="sf_merge",
                        payload=merged,
                    ))

                print(f"[Coordinator] Sync round {sync_round}: "
                      f"merged SF from {len(pending_sf)} workers")
                pending_sf.clear()
            else:
                # Send empty merge to unblock worker (async mode)
                from_coord_queues[msg.worker_id].put(SyncMessage(
                    worker_id=-1,
                    msg_type="sf_merge",
                    payload=None,
                ))

    total_time = time.time() - t0

    # Wait for workers to finish
    for p in workers:
        p.join(timeout=30)

    # Save results
    result = {
        "num_workers": num_workers,
        "total_episodes": total_episodes,
        "episodes_per_sync": episodes_per_sync,
        "total_wall_time_s": total_time,
        "sync_rounds": sync_round,
        "worker_timings": timing_log,
    }

    out_file = RESULTS_DIR / f"scaling_P{num_workers}.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {out_file}")
    print(f"Total wall time: {total_time:.1f}s ({num_workers} workers)")

    return result


def main():
    parser = argparse.ArgumentParser(description="PCF-MORL Parallel Coordinator")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel workers P")
    parser.add_argument("--total-episodes", type=int, default=10_000)
    parser.add_argument("--episodes-per-sync", type=int, default=200,
                        help="K: episodes between SF sharing rounds")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--ns3-path", type=str, default=None)
    parser.add_argument("--scenario", type=str, default="training")
    args = parser.parse_args()

    run_coordinator(
        num_workers=args.workers,
        total_episodes=args.total_episodes,
        episodes_per_sync=args.episodes_per_sync,
        max_steps=args.max_steps,
        base_seed=args.base_seed,
        ns3_path=args.ns3_path,
        scenario=args.scenario,
    )


if __name__ == "__main__":
    main()
