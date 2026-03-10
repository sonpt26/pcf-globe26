"""Threshold-based baselines A1 (Conservative), A2 (Aggressive), A3 (Hysteresis).

These are rule-based policies that don't require training.
They serve as non-learning baselines for comparison.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from env.action_space import encode_action, RATE_EMBB_VALUES, RATE_URLLC_VALUES


class BaseThresholdPolicy:
    """Base class for threshold-based policies."""

    def __init__(self):
        self.rate_embb_idx = self._find_idx(RATE_EMBB_VALUES, self.default_embb)
        self.rate_urllc_idx = self._find_idx(RATE_URLLC_VALUES, self.default_urllc)

    @staticmethod
    def _find_idx(arr, val):
        return int(np.argmin(np.abs(arr - val)))

    @property
    def default_embb(self):
        raise NotImplementedError

    @property
    def default_urllc(self):
        raise NotImplementedError

    def get_action(self, obs, info=None):
        """Return discrete action id."""
        return encode_action(
            RATE_URLLC_VALUES[self.rate_urllc_idx],
            RATE_EMBB_VALUES[self.rate_embb_idx],
        )

    def reset(self):
        self.rate_embb_idx = self._find_idx(RATE_EMBB_VALUES, self.default_embb)
        self.rate_urllc_idx = self._find_idx(RATE_URLLC_VALUES, self.default_urllc)


class ConservativePolicy(BaseThresholdPolicy):
    """A1 Conservative: Default rate_embb=20, rate_urllc=15.
    Decrease if delay > 0.7ms for any step, increase if < 0.3ms for 5 consecutive steps.
    """

    default_embb = 20
    default_urllc = 15

    def __init__(self):
        super().__init__()
        self.low_delay_streak = 0

    def reset(self):
        super().reset()
        self.low_delay_streak = 0

    def get_action(self, obs, info=None):
        # obs[2] = urllc_delay_95 / 10ms (normalized), so delay_ms ≈ obs[2] * 10
        delay_ms = obs[2] * 10.0 if obs is not None else 0

        if delay_ms > 0.7:
            # Decrease eMBB rate
            self.rate_embb_idx = max(0, self.rate_embb_idx - 1)
            self.low_delay_streak = 0
        elif delay_ms < 0.3:
            self.low_delay_streak += 1
            if self.low_delay_streak >= 5:
                # Increase eMBB rate
                self.rate_embb_idx = min(len(RATE_EMBB_VALUES) - 1, self.rate_embb_idx + 1)
                self.low_delay_streak = 0
        else:
            self.low_delay_streak = 0

        return super().get_action(obs, info)


class AggressivePolicy(BaseThresholdPolicy):
    """A2 Aggressive: Max rates (embb=100, urllc=20).
    Decrease embb by ~10 Mbps if delay > 0.8ms.
    """

    default_embb = 100
    default_urllc = 20

    def get_action(self, obs, info=None):
        delay_ms = obs[2] * 10.0 if obs is not None else 0

        if delay_ms > 0.8:
            # Decrease embb rate (move ~2 steps down in the array)
            self.rate_embb_idx = max(0, self.rate_embb_idx - 2)

        return super().get_action(obs, info)


class HysteresisPolicy(BaseThresholdPolicy):
    """A3 Hysteresis: Decrease embb by ~10 Mbps if delay > 0.8ms (cooldown 5 steps),
    increase by ~5 Mbps if < 0.4ms for 10 consecutive steps.
    """

    default_embb = 50
    default_urllc = 10

    def __init__(self):
        super().__init__()
        self.cooldown = 0
        self.low_delay_streak = 0

    def reset(self):
        super().reset()
        self.cooldown = 0
        self.low_delay_streak = 0

    def get_action(self, obs, info=None):
        delay_ms = obs[2] * 10.0 if obs is not None else 0

        if self.cooldown > 0:
            self.cooldown -= 1

        if delay_ms > 0.8 and self.cooldown == 0:
            self.rate_embb_idx = max(0, self.rate_embb_idx - 2)
            self.cooldown = 5
            self.low_delay_streak = 0
        elif delay_ms < 0.4:
            self.low_delay_streak += 1
            if self.low_delay_streak >= 10:
                self.rate_embb_idx = min(len(RATE_EMBB_VALUES) - 1, self.rate_embb_idx + 1)
                self.low_delay_streak = 0
        else:
            self.low_delay_streak = 0

        return super().get_action(obs, info)


BASELINES = {
    "A1_conservative": ConservativePolicy,
    "A2_aggressive": AggressivePolicy,
    "A3_hysteresis": HysteresisPolicy,
}
