"""M4: Multi-objective and QoS metrics for PCF-MORL experiments.

Metrics computed:
  - VR:  Violation Ratio (fraction of steps with URLLC delay > 1ms)
  - TTR: Time To Resolve (steps from violation start to resolution)
  - MVD: Max Violation Duration (longest consecutive violation streak)
  - HV:  Hypervolume indicator (Pareto front quality)
  - EU:  Expected Utility (mean scalarized return over test weights)
  - MUL: Maximum Utility Loss (worst-case gap vs oracle)
  - ZSR: Zero-Shot Ratio (fraction of test weights where GPI ≥ 90% of oracle)
  - TTM: Time To Match (episodes until GPI matches oracle within 5%)
"""

import numpy as np
from pymoo.indicators.hv import HV as HyperVolume


# ---------------------------------------------------------------------------
# Per-episode QoS metrics (computed from step-level info dicts)
# ---------------------------------------------------------------------------

def compute_vr(step_infos: list[dict]) -> float:
    """Violation Ratio: fraction of steps where URLLC delay exceeds 1 ms."""
    if not step_infos:
        return 0.0
    violations = sum(1 for info in step_infos if info.get("VR", 0.0) > 0.0)
    return violations / len(step_infos)


def compute_ttr(step_infos: list[dict]) -> float:
    """Time To Resolve: mean steps from violation onset to resolution."""
    streaks = []
    current = 0
    for info in step_infos:
        if info.get("VR", 0.0) > 0.0:
            current += 1
        elif current > 0:
            streaks.append(current)
            current = 0
    if current > 0:
        streaks.append(current)
    return float(np.mean(streaks)) if streaks else 0.0


def compute_mvd(step_infos: list[dict]) -> int:
    """Max Violation Duration: longest consecutive violation streak."""
    max_streak = 0
    current = 0
    for info in step_infos:
        if info.get("VR", 0.0) > 0.0:
            current += 1
            max_streak = max(max_streak, current)
        else:
            current = 0
    return max_streak


def compute_episode_qos(step_infos: list[dict]) -> dict:
    """Compute all per-episode QoS metrics."""
    return {
        "VR": compute_vr(step_infos),
        "TTR": compute_ttr(step_infos),
        "MVD": compute_mvd(step_infos),
    }


# ---------------------------------------------------------------------------
# Per-episode reward aggregation
# ---------------------------------------------------------------------------

def compute_episode_return(rewards: list[np.ndarray], gamma: float = 0.99) -> np.ndarray:
    """Discounted vector return for one episode."""
    ret = np.zeros(3, dtype=np.float64)
    for t, r in enumerate(rewards):
        ret += (gamma ** t) * np.asarray(r, dtype=np.float64)
    return ret


def compute_mean_throughput(step_infos: list[dict]) -> float:
    """Mean served eMBB throughput across steps (from KPIs)."""
    vals = [info.get("kpis", {}).get("embb_throughput_mbps", 0.0) for info in step_infos]
    return float(np.mean(vals)) if vals else 0.0


def compute_mean_energy(step_infos: list[dict]) -> float:
    """Mean energy per bit across steps."""
    vals = [info.get("kpis", {}).get("energy_per_bit", 0.0) for info in step_infos]
    return float(np.mean(vals)) if vals else 0.0


# ---------------------------------------------------------------------------
# Pareto front metrics (computed over sets of policies / weight evaluations)
# ---------------------------------------------------------------------------

def compute_hypervolume(points: np.ndarray, ref_point: np.ndarray) -> float:
    """Hypervolume indicator for a set of objective vectors.

    Args:
        points: (N, d) array of objective vectors (higher is better).
        ref_point: (d,) reference point (worst-case per objective).

    Returns:
        Hypervolume scalar.
    """
    if len(points) == 0:
        return 0.0
    # pymoo HV minimises, so negate for maximisation objectives
    hv = HyperVolume(ref_point=-ref_point)
    return float(hv(-points))


def compute_expected_utility(
    returns: np.ndarray, weights: np.ndarray
) -> float:
    """Expected Utility: mean scalarized return over test weights.

    Args:
        returns: (N, d) episodic returns, one per test weight.
        weights:  (N, d) test weight vectors.
    """
    if len(returns) == 0:
        return 0.0
    scalarized = np.array([np.dot(r, w) for r, w in zip(returns, weights)])
    return float(np.mean(scalarized))


def compute_mul(
    returns: np.ndarray, oracle_returns: np.ndarray, weights: np.ndarray
) -> float:
    """Maximum Utility Loss: worst-case gap vs oracle over test weights.

    MUL = max_w [ J_oracle(w) - J_agent(w) ]
    """
    gaps = []
    for r, o, w in zip(returns, oracle_returns, weights):
        gap = np.dot(o, w) - np.dot(r, w)
        gaps.append(gap)
    return float(np.max(gaps)) if gaps else 0.0


def compute_zsr(
    returns: np.ndarray, oracle_returns: np.ndarray, weights: np.ndarray,
    threshold: float = 0.9,
) -> float:
    """Zero-Shot Ratio: fraction of test weights where agent >= threshold * oracle quality."""
    if len(weights) == 0:
        return 0.0
    count = 0
    for r, o, w in zip(returns, oracle_returns, weights):
        j_agent = np.dot(r, w)
        j_oracle = np.dot(o, w)
        if j_oracle > 0:
            passed = j_agent >= threshold * j_oracle
        elif j_oracle < 0:
            # For negative utility: agent loss must not exceed oracle loss / threshold
            passed = j_agent >= j_oracle / threshold
        else:
            passed = j_agent >= 0
        if passed:
            count += 1
    return count / len(weights)


def compute_ttm(
    episode_returns: list[np.ndarray],
    oracle_return: np.ndarray,
    weight: np.ndarray,
    threshold: float = 0.95,
) -> int:
    """Time To Match: episodes until agent matches oracle within threshold.

    Returns number of episodes, or -1 if never matched.
    """
    j_oracle = np.dot(oracle_return, weight)
    for i, ret in enumerate(episode_returns):
        j = np.dot(ret, weight)
        if j_oracle <= 0:
            if j >= j_oracle * threshold:
                return i + 1
        elif j >= threshold * j_oracle:
            return i + 1
    return -1


# ---------------------------------------------------------------------------
# Aggregate across seeds
# ---------------------------------------------------------------------------

def aggregate_seeds(
    per_seed_metrics: list[dict], confidence: float = 0.95
) -> dict:
    """Aggregate metrics across seeds with mean + CI.

    Args:
        per_seed_metrics: list of dicts, each with same keys and scalar values.
        confidence: confidence level for CI (default 0.95 → 95%).
    """
    from scipy import stats

    keys = per_seed_metrics[0].keys()
    result = {}
    n = len(per_seed_metrics)

    for k in keys:
        vals = np.array([m[k] for m in per_seed_metrics], dtype=np.float64)
        mean = float(np.mean(vals))
        if n > 1:
            se = float(stats.sem(vals))
            t_crit = float(stats.t.ppf((1 + confidence) / 2, n - 1))
            ci = t_crit * se
        else:
            ci = 0.0
        result[k] = {"mean": mean, "ci": ci, "values": vals.tolist()}
    return result


def wilcoxon_test(vals_a: list[float], vals_b: list[float]) -> float:
    """Wilcoxon signed-rank test p-value (paired comparison)."""
    from scipy.stats import wilcoxon
    if len(vals_a) < 5:
        # Too few samples for reliable test
        return float("nan")
    _, p = wilcoxon(vals_a, vals_b)
    return float(p)
