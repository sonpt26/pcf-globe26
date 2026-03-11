"""M2 smoke test: Run GPI-PD for a few episodes against live ns-3.

Verifies:
  1. GPIPD agent initialises on GPU
  2. Agent can interact with PcfMorlEnv (ns-3 subprocess)
  3. Training loop completes without crash
  4. Eval with a weight vector returns valid actions & rewards
  5. GPU memory is used

Usage:
    python3 tests/test_training_smoke.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from morl_baselines.multi_policy.gpi_pd.gpi_pd import GPIPD
from env.pcf_morl_env import PcfMorlEnv

# Short training: 3 episodes × 10 steps = 30 timesteps
TRAIN_TIMESTEPS = 30
MAX_STEPS = 10
SEED = 77


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── 1. Create envs ──
    print("\n[1] Creating PcfMorlEnv (ns-3 backed) ...")
    env = PcfMorlEnv(seed=SEED, max_steps=MAX_STEPS)
    eval_env = PcfMorlEnv(seed=SEED + 1000, max_steps=MAX_STEPS)
    print(f"  obs={env.observation_space.shape}, act={env.action_space.n}, "
          f"rew={env.reward_space.shape}")

    # ── 2. Create agent ──
    print("\n[2] Creating GPI-PD agent ...")
    ref_point = np.array([0.0, -float(MAX_STEPS), -float(MAX_STEPS)])

    agent = GPIPD(
        env=env,
        learning_rate=3e-4,
        gamma=0.99,
        batch_size=32,
        net_arch=[64, 64],         # Small net for smoke test
        buffer_size=10_000,
        target_net_update_freq=50,
        tau=1.0,
        initial_epsilon=0.5,       # High exploration
        num_nets=2,
        gradient_updates=2,
        per=False,                 # Skip PER for speed
        dyna=False,                # Skip dynamics model for speed
        gpi_pd=False,              # Skip PD for speed (test core GPI)
        learning_starts=10,
        log=False,
        seed=SEED,
        device=device,
    )
    print("  Agent created OK")

    # Verify network device
    param = next(agent.q_nets[0].parameters())
    actual_device = str(param.device)
    print(f"  Q-network on: {actual_device}")

    # ── 3. Train ──
    print(f"\n[3] Training: {TRAIN_TIMESTEPS} steps "
          f"(~{TRAIN_TIMESTEPS // MAX_STEPS} episodes × {MAX_STEPS} steps) ...")
    t0 = time.time()

    agent.train(
        total_timesteps=TRAIN_TIMESTEPS,
        eval_env=eval_env,
        ref_point=ref_point,
        eval_freq=TRAIN_TIMESTEPS + 1,     # No mid-train eval
        eval_mo_freq=TRAIN_TIMESTEPS + 1,
        timesteps_per_iter=TRAIN_TIMESTEPS,
        num_eval_weights_for_front=3,
        num_eval_episodes_for_front=1,
        checkpoints=False,
    )

    elapsed = time.time() - t0
    print(f"  Training done in {elapsed:.1f}s")

    # ── 4. Eval with weight vector ──
    print("\n[4] Eval episode with ω=[0.5, 0.3, 0.2] ...")
    test_w = np.array([0.5, 0.3, 0.2])
    obs, info = eval_env.reset(seed=SEED + 2000)
    print(f"  Reset OK, obs={obs.shape}")

    actions = []
    rewards = []
    for step in range(min(5, MAX_STEPS)):
        action = agent.eval(obs, test_w)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        actions.append(int(action))
        rewards.append(reward.copy())

    eval_env.close()
    env.close()

    from env.action_space import decode_action
    print(f"  Actions: {actions}")
    for i, (a, r) in enumerate(zip(actions, rewards)):
        u, e = decode_action(a)
        print(f"    step {i}: a={a} (u={u:.0f}, e={e:.0f})  "
              f"r=[{r[0]:.3f}, {r[1]:.3f}, {r[2]:.3f}]")

    # ── 5. GPU stats ──
    if device == "cuda":
        mem_peak = torch.cuda.max_memory_allocated() / 1e6
        print(f"\n  GPU peak memory: {mem_peak:.1f} MB")

    # ── Validation ──
    print(f"\n{'='*50}")
    print("Validation:")
    checks = [
        ("Agent initialised",
         agent is not None),
        ("On correct device",
         ("cuda" in actual_device) if device == "cuda" else ("cpu" in actual_device)),
        ("Training completed",
         elapsed > 0),
        ("Valid actions (0..75)",
         all(0 <= a < 76 for a in actions)),
        ("Valid r1 [0,1]",
         all(0 - 0.01 <= r[0] <= 1.01 for r in rewards)),
        ("Valid r2 [-1,0]",
         all(-1.01 <= r[1] <= 0.01 for r in rewards)),
        ("Valid r3 [-1,0]",
         all(-1.01 <= r[2] <= 0.01 for r in rewards)),
        ("Eval produced 5 steps",
         len(actions) == 5),
    ]

    if device == "cuda":
        checks.append(("GPU memory used (>0 MB)", mem_peak > 0))

    all_pass = True
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        if not ok:
            all_pass = False

    result = "M2 TRAINING SMOKE PASSED" if all_pass else "M2 TRAINING SMOKE FAILED"
    print(f"\n{result}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
