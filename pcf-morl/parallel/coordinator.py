"""M3: Parallel coordinator – manages P workers for distributed GPI-PD training.

Architecture:
  Coordinator → P Workers → P ns-3 instances.
  SF sharing every K episodes via weight support union.

Usage:
    python -m parallel.coordinator --workers 4 --episodes-per-sync 200 --total-episodes 10000
"""

import argparse
import json
import sys
import time
from multiprocessing import Process, Queue
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from parallel.worker import WorkerConfig, SyncMessage, worker_process

RESULTS_DIR = Path(__file__).parent.parent / "results" / "parallel"


def merge_weight_supports(worker_states: list[dict]) -> list:
    """Merge weight support sets from multiple workers.

    Strategy: CCS union – take the union of all weight vectors,
    deduplicate by rounding tolerance.
    """
    all_weights = []
    for state in worker_states:
        if state is None:
            continue
        ws = state.get("weight_support", [])
        all_weights.extend(ws)

    if not all_weights:
        return []

    # Deduplicate: round to 4 decimals, keep unique
    arr = np.array(all_weights)
    if arr.ndim == 1:
        return all_weights

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
    device: str = "auto",
):
    """Launch workers and coordinate weight support sharing."""

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Shared queue: Workers → Coordinator
    to_coord = Queue()
    # Per-worker queues: Coordinator → Worker_i
    from_coord = [Queue() for _ in range(num_workers)]

    # Create worker configs with distinct seeds
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
            device=device,
        )
        configs.append(cfg)

    # Launch worker processes
    workers = []
    for i, cfg in enumerate(configs):
        p = Process(
            target=worker_process,
            args=(cfg, to_coord, from_coord[i]),
            name=f"worker-{i}",
        )
        p.start()
        workers.append(p)

    print(f"[Coordinator] Launched {num_workers} workers "
          f"(K={episodes_per_sync}, total={total_episodes} eps, "
          f"max_steps={max_steps})")

    # ── Coordination loop ──
    t0 = time.time()
    active_workers = set(range(num_workers))
    sync_round = 0
    pending_states: dict[int, dict] = {}
    timing_log = []
    merged_weight_count_log = []

    while active_workers:
        try:
            msg: SyncMessage = to_coord.get(timeout=1800)
        except Exception:
            print(f"[Coordinator] Timeout waiting for workers. Active: {active_workers}")
            break

        if msg.msg_type == "error":
            print(f"[Coordinator] Worker {msg.worker_id} ERROR:\n{msg.payload}")
            active_workers.discard(msg.worker_id)
            continue

        if msg.msg_type == "done":
            elapsed = msg.wall_time_s
            print(f"[Coordinator] Worker {msg.worker_id} DONE: "
                  f"{msg.episode_count} eps in {elapsed:.1f}s")
            timing_log.append({
                "worker_id": msg.worker_id,
                "episodes": msg.episode_count,
                "wall_time_s": elapsed,
            })
            active_workers.discard(msg.worker_id)
            continue

        if msg.msg_type == "sf_update":
            pending_states[msg.worker_id] = msg.payload
            print(f"[Coordinator] Worker {msg.worker_id}: {msg.episode_count} eps "
                  f"({msg.wall_time_s:.1f}s) "
                  f"|M|={len(msg.payload.get('weight_support', [])) if msg.payload else 0}")

            # Barrier sync: wait for all active workers before merging
            if set(pending_states.keys()) >= active_workers:
                sync_round += 1
                merged = merge_weight_supports(list(pending_states.values()))

                for wid in active_workers:
                    from_coord[wid].put(SyncMessage(
                        worker_id=-1,
                        msg_type="sf_merge",
                        payload=merged,
                    ))

                merged_weight_count_log.append(len(merged))
                print(f"[Coordinator] Sync round {sync_round}: "
                      f"merged |M|={len(merged)} from {len(pending_states)} workers")
                pending_states.clear()
            else:
                # Async mode: unblock worker immediately with None (no merge yet)
                from_coord[msg.worker_id].put(SyncMessage(
                    worker_id=-1,
                    msg_type="sf_merge",
                    payload=None,
                ))

    total_time = time.time() - t0

    # Wait for processes
    for p in workers:
        p.join(timeout=30)
        if p.is_alive():
            p.terminate()

    # Save results
    result = {
        "num_workers": num_workers,
        "total_episodes": total_episodes,
        "episodes_per_sync": episodes_per_sync,
        "max_steps": max_steps,
        "total_wall_time_s": total_time,
        "sync_rounds": sync_round,
        "merged_weight_counts": merged_weight_count_log,
        "worker_timings": timing_log,
    }

    out_file = RESULTS_DIR / f"scaling_P{num_workers}.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Results saved to {out_file}")
    print(f"Total wall time: {total_time:.1f}s ({num_workers} workers)")
    if timing_log:
        avg_worker = np.mean([t["wall_time_s"] for t in timing_log])
        print(f"Avg worker time: {avg_worker:.1f}s")
    print(f"Sync rounds: {sync_round}")

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
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    run_coordinator(
        num_workers=args.workers,
        total_episodes=args.total_episodes,
        episodes_per_sync=args.episodes_per_sync,
        max_steps=args.max_steps,
        base_seed=args.base_seed,
        ns3_path=args.ns3_path,
        scenario=args.scenario,
        device=args.device,
    )


if __name__ == "__main__":
    main()
