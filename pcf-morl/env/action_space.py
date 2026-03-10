"""Action space utilities for PCF-MORL environment.

Action space: Discrete(76)
  a = (rate_urllc, rate_embb)
  rate_urllc ∈ {5, 10, 15, 20} Mbps       → 4 values
  rate_embb  ∈ {2,3,...,10,12,...,100} Mbps → 19 values (spans sub- to over-capacity)
  Total: 4 × 19 = 76
"""

import numpy as np

RATE_URLLC_VALUES = np.array([5, 10, 15, 20], dtype=np.float32)  # Mbps
# Range spans sub-capacity (2-8 Mbps/UE × 8 UEs < 80 Mbps BWP capacity)
# to over-capacity (15+ Mbps/UE × 8 UEs > 80 Mbps)
RATE_EMBB_VALUES = np.array(
    [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30, 40, 50, 60, 80, 100],
    dtype=np.float32,
)  # 19 values

N_URLLC = len(RATE_URLLC_VALUES)  # 4
N_EMBB = len(RATE_EMBB_VALUES)    # 19
N_ACTIONS = N_URLLC * N_EMBB      # 76

# Precompute lookup table: action_id → (rate_urllc_mbps, rate_embb_mbps)
ACTION_TABLE = np.array(
    [(u, e) for u in RATE_URLLC_VALUES for e in RATE_EMBB_VALUES],
    dtype=np.float32,
)
assert ACTION_TABLE.shape == (N_ACTIONS, 2)


def decode_action(action_id: int) -> tuple[float, float]:
    """Convert discrete action to (rate_urllc_mbps, rate_embb_mbps)."""
    row = ACTION_TABLE[action_id]
    return float(row[0]), float(row[1])


def encode_action(rate_urllc_mbps: float, rate_embb_mbps: float) -> int:
    """Convert (rate_urllc_mbps, rate_embb_mbps) to discrete action id."""
    u_idx = int(np.searchsorted(RATE_URLLC_VALUES, rate_urllc_mbps))
    e_idx = int(np.searchsorted(RATE_EMBB_VALUES, rate_embb_mbps))
    return u_idx * N_EMBB + e_idx
