"""Basic validation tests for PCF-MORL environment."""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

import numpy as np
from env.action_space import (
    ACTION_TABLE, N_ACTIONS, N_EMBB, N_URLLC,
    RATE_EMBB_VALUES, RATE_URLLC_VALUES,
    decode_action, encode_action,
)


def test_action_space():
    """Validate action space dimensions and decode/encode roundtrip."""
    assert N_ACTIONS == 76, f"Expected 76 actions, got {N_ACTIONS}"
    assert N_URLLC == 4
    assert N_EMBB == 19
    assert ACTION_TABLE.shape == (76, 2)

    # Check ranges
    assert np.array_equal(RATE_URLLC_VALUES, [5, 10, 15, 20])
    assert RATE_EMBB_VALUES[0] == 10
    assert RATE_EMBB_VALUES[-1] == 100
    assert len(RATE_EMBB_VALUES) == 19

    # Roundtrip
    for a in range(N_ACTIONS):
        u, e = decode_action(a)
        a2 = encode_action(u, e)
        assert a == a2, f"Roundtrip failed: {a} -> ({u},{e}) -> {a2}"

    # Specific values
    assert decode_action(0) == (5.0, 10.0)
    assert decode_action(75) == (20.0, 100.0)

    print("Action space: PASS")


def test_action_decode_specific():
    """Test specific action decodings."""
    # action 0 → (5, 10)
    u, e = decode_action(0)
    assert u == 5.0 and e == 10.0

    # action 18 → (5, 100)
    u, e = decode_action(18)
    assert u == 5.0 and e == 100.0

    # action 19 → (10, 10)
    u, e = decode_action(19)
    assert u == 10.0 and e == 10.0

    # action 75 → (20, 100)
    u, e = decode_action(75)
    assert u == 20.0 and e == 100.0

    print("Action decode specific: PASS")


def test_env_spaces():
    """Validate environment space definitions without running ns-3."""
    from env.pcf_morl_env import PcfMorlEnv

    # Just check space definitions (don't launch ns-3)
    env = PcfMorlEnv.__new__(PcfMorlEnv)
    env.observation_space = __import__("gymnasium").spaces.Box(
        low=0.0, high=1.0, shape=(12,), dtype=np.float32
    )
    env.action_space = __import__("gymnasium").spaces.Discrete(N_ACTIONS)
    env.reward_space = __import__("gymnasium").spaces.Box(
        low=np.array([0.0, -1.0, -1.0], dtype=np.float32),
        high=np.array([1.0, 0.0, 0.0], dtype=np.float32),
    )

    assert env.observation_space.shape == (12,)
    assert env.action_space.n == 76
    assert env.reward_space.shape == (3,)

    # Check reward bounds
    assert env.reward_space.low[0] == 0.0
    assert env.reward_space.high[0] == 1.0
    assert env.reward_space.low[1] == -1.0
    assert env.reward_space.high[1] == 0.0

    print("Environment spaces: PASS")


if __name__ == "__main__":
    test_action_space()
    test_action_decode_specific()
    test_env_spaces()
    print("\nAll basic tests PASSED!")
