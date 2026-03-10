"""Smoke test: verify GPI-PD training pipeline runs for a few steps."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from morl_baselines.multi_policy.gpi_pd.gpi_pd import GPIPD
from env.pcf_morl_env import PcfMorlEnv


def main():
    print("Creating environments...")
    env = PcfMorlEnv(seed=42, max_steps=5)  # Short episodes for smoke test
    eval_env = PcfMorlEnv(seed=43, max_steps=5)

    ref_point = np.array([0.0, -5.0, -5.0])  # Scaled for 5-step episodes

    print("Creating GPI-PD agent...")
    agent = GPIPD(
        env=env,
        learning_rate=3e-4,
        gamma=0.99,
        batch_size=32,
        net_arch=[64, 64],  # Smaller network for smoke test
        buffer_size=1000,
        target_net_update_freq=50,
        tau=1.0,
        initial_epsilon=0.5,
        num_nets=2,
        gradient_updates=1,
        per=True,
        dyna=False,  # Disable dynamics model for speed
        gpi_pd=False,  # Disable PD for speed
        learning_starts=10,
        project_name="PCF-MORL-smoke",
        experiment_name="GPI-PD-smoke",
        log=False,
        seed=42,
        device="cpu",
    )

    print("Starting training (100 timesteps = 20 episodes × 5 steps)...")
    agent.train(
        total_timesteps=100,
        eval_env=eval_env,
        ref_point=ref_point,
        eval_freq=50,
        eval_mo_freq=100,
        timesteps_per_iter=50,
        num_eval_weights_for_front=3,
        num_eval_episodes_for_front=1,
        checkpoints=False,
    )

    print("\nSmoke test PASSED - GPI-PD training pipeline works!")


if __name__ == "__main__":
    main()
