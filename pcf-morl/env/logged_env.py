"""Logging wrapper for PcfMorlEnv.

Intercepts step() and reset() to record all data to TrainingLogger.
Use this as a drop-in replacement for PcfMorlEnv during training.
"""

import numpy as np
from .pcf_morl_env import PcfMorlEnv
from .action_space import decode_action


class LoggedPcfMorlEnv(PcfMorlEnv):
    """PcfMorlEnv with automatic CSV logging of every step and episode."""

    def __init__(self, logger, **kwargs):
        super().__init__(**kwargs)
        self.logger = logger
        self._episode_num = 0
        self._ep_rewards = []
        self._ep_infos = []

    def reset(self, *, seed=None, options=None):
        # Log previous episode if it had any steps
        if self._ep_rewards:
            self.logger.log_episode(
                episode=self._episode_num,
                rewards=self._ep_rewards,
                step_infos=self._ep_infos,
            )
            self._episode_num += 1

        self._ep_rewards = []
        self._ep_infos = []
        return super().reset(seed=seed, options=options)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        rate_urllc, rate_embb = decode_action(action)
        self.logger.log_step(
            episode=self._episode_num,
            step=self._step_count,
            obs=obs,
            action=action,
            reward=reward,
            info=info,
            rate_urllc=rate_urllc,
            rate_embb=rate_embb,
        )

        self._ep_rewards.append(reward.copy())
        self._ep_infos.append(info)

        return obs, reward, terminated, truncated, info

    def close(self):
        # Log final episode
        if self._ep_rewards:
            self.logger.log_episode(
                episode=self._episode_num,
                rewards=self._ep_rewards,
                step_infos=self._ep_infos,
            )
        super().close()
