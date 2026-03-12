"""Generate realistic pseudo experiment results for paper pipeline testing.

Produces synthetic data for all 6 experiments in the expected JSON format,
so M5 (figures.py, tables.py) can generate paper-ready outputs.

Usage:
    python -m experiments.generate_pseudo_results
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

RESULTS_DIR = Path(__file__).parent.parent / "results"
np.random.seed(2026)

# Reward ranges from real env observations:
#   r1 (throughput): 0.48 – 0.82
#   r2 (delay):     -0.51 – 0.0
#   r3 (energy):    -0.58 – -0.49

# ---------------------------------------------------------------------------
# Method profiles: (r1_mean, r1_std, r2_mean, r2_std, r3_mean, r3_std,
#                   VR_mean, TTR_mean, MVD_mean, thr_mbps, energy)
# PCF-MORL is best overall; baselines each have tradeoffs.
# ---------------------------------------------------------------------------

METHOD_PROFILES = {
    "PCF-MORL": {
        "returns": (0.72, 0.06, -0.08, 0.04, -0.50, 0.02),
        "qos": (0.05, 1.2, 3),
        "kpis": (9.2, 0.35),
    },
    "A1_conservative": {
        "returns": (0.55, 0.08, -0.15, 0.06, -0.52, 0.03),
        "qos": (0.12, 2.5, 7),
        "kpis": (7.5, 0.42),
    },
    "A2_aggressive": {
        "returns": (0.78, 0.05, -0.35, 0.10, -0.55, 0.03),
        "qos": (0.25, 3.8, 12),
        "kpis": (9.5, 0.50),
    },
    "A3_hysteresis": {
        "returns": (0.62, 0.07, -0.12, 0.05, -0.53, 0.02),
        "qos": (0.10, 2.0, 5),
        "kpis": (8.1, 0.40),
    },
    "Scalarized_DQN": {
        "returns": (0.65, 0.08, -0.18, 0.07, -0.52, 0.03),
        "qos": (0.14, 2.2, 6),
        "kpis": (8.5, 0.41),
    },
}

# Scenario difficulty modifiers (applied to returns and VR)
SCENARIO_MODS = {
    "E1": {"r_scale": 1.0, "vr_mult": 0.8},    # Constant 90% — easier
    "E2": {"r_scale": 0.95, "vr_mult": 1.2},    # URLLC burst — harder on delay
    "E3": {"r_scale": 0.90, "vr_mult": 1.0},    # eMBB surge — harder on throughput
    "E4": {"r_scale": 0.88, "vr_mult": 1.3},    # Alternating — variable
    "E5": {"r_scale": 0.80, "vr_mult": 1.8},    # Overload — hardest
}

TEST_WEIGHTS = [
    [0.7, 0.2, 0.1],
    [0.2, 0.7, 0.1],
    [0.2, 0.2, 0.6],
]


def gen_episode(method: str, scenario: str, seed: int, weight=None):
    """Generate one synthetic episode result."""
    rng = np.random.RandomState(hash((method, scenario, seed)) % 2**31)
    p = METHOD_PROFILES[method]
    mod = SCENARIO_MODS.get(scenario, SCENARIO_MODS["E1"])

    r1_m, r1_s, r2_m, r2_s, r3_m, r3_s = p["returns"]
    scale = mod["r_scale"]

    # Per-step rewards (100 steps)
    step_rewards = []
    for _ in range(100):
        r1 = np.clip(rng.normal(r1_m * scale, r1_s), 0, 1)
        r2 = np.clip(rng.normal(r2_m * scale, r2_s), -1, 0)
        r3 = np.clip(rng.normal(r3_m * scale, r3_s), -1, 0)
        step_rewards.append([float(r1), float(r2), float(r3)])

    # Discounted return (gamma=0.99)
    ret = np.zeros(3)
    for t, r in enumerate(step_rewards):
        ret += (0.99 ** t) * np.array(r)

    # QoS
    vr_base, ttr_base, mvd_base = p["qos"]
    vr = np.clip(rng.normal(vr_base * mod["vr_mult"], vr_base * 0.3), 0, 1)
    ttr = max(0, rng.normal(ttr_base, ttr_base * 0.3))
    mvd = max(1, int(rng.normal(mvd_base, mvd_base * 0.3)))

    thr, epb = p["kpis"]
    thr_val = max(0, rng.normal(thr, thr * 0.05))
    epb_val = max(0, rng.normal(epb, epb * 0.1))

    return {
        "method": method,
        "scenario": scenario,
        "seed": seed,
        "weight": weight if weight else [1/3, 1/3, 1/3],
        "episode_return": ret.tolist(),
        "qos": {"VR": float(vr), "TTR": float(ttr), "MVD": int(mvd)},
        "mean_throughput": float(thr_val),
        "mean_energy": float(epb_val),
        "wall_time_s": float(rng.uniform(1800, 2400)),
        "step_rewards": step_rewards,
    }


def generate_simplex_weights(n, dim=3):
    """Generate n weights on the simplex."""
    rng = np.random.RandomState(42)
    weights = []
    for _ in range(n):
        w = rng.dirichlet(np.ones(dim))
        w = np.maximum(w, 0.05)
        w /= w.sum()
        weights.append(w.tolist())
    return weights


# ===========================================================================
# Experiment 1: QoS Performance
# ===========================================================================

def gen_exp1(seeds):
    results = []
    for scenario in ["E1", "E2", "E3", "E4", "E5"]:
        for seed in seeds:
            # Threshold baselines
            for method in ["A1_conservative", "A2_aggressive", "A3_hysteresis"]:
                results.append(gen_episode(method, scenario, seed))

            # GPI-PD at 3 test weights
            for w in TEST_WEIGHTS:
                results.append(gen_episode("PCF-MORL", scenario, seed, weight=w))

            # Scalarized DQN
            results.append(gen_episode("Scalarized_DQN", scenario, seed,
                                       weight=[1/3, 1/3, 1/3]))

    return results


# ===========================================================================
# Experiment 2: Zero-shot adaptation
# ===========================================================================

def gen_exp2(seeds):
    weights_20 = generate_simplex_weights(20)
    gpi_results = []
    oracle_results = []

    for seed in seeds:
        for w in weights_20:
            # GPI zero-shot
            r = gen_episode("PCF-MORL", "E1", seed, weight=w)
            r["adaptation_type"] = "zero_shot"
            gpi_results.append(r)

            # Oracle (slightly better on each individual weight, but no generalization)
            rng = np.random.RandomState(hash(("oracle", str(w), seed)) % 2**31)
            oracle_ret = np.array(r["episode_return"]) * rng.uniform(1.0, 1.08, 3)
            oracle_r = dict(r)
            oracle_r["method"] = "Oracle"
            oracle_r["episode_return"] = oracle_ret.tolist()
            oracle_results.append(oracle_r)

    # Compute ZSR
    gpi_returns = np.array([r["episode_return"] for r in gpi_results])
    oracle_returns = np.array([r["episode_return"] for r in oracle_results])
    weights_arr = np.array([r["weight"] for r in gpi_results])

    count = 0
    for gr, oret, w in zip(gpi_returns, oracle_returns, weights_arr):
        j_gpi = np.dot(gr, w)
        j_oracle = np.dot(oret, w)
        if j_oracle <= 0 or j_gpi >= 0.9 * j_oracle:
            count += 1
    zsr = count / len(weights_arr)

    return {
        "gpi": gpi_results,
        "oracle": oracle_results,
        "analysis": {"ZSR": float(zsr)},
    }


# ===========================================================================
# Experiment 3: 24h factory cycle
# ===========================================================================

def gen_exp3(seeds):
    phase_weights = [
        [0.6, 0.3, 0.1], [0.3, 0.5, 0.2], [0.2, 0.7, 0.1], [0.4, 0.3, 0.3],
        [0.2, 0.2, 0.6], [0.1, 0.3, 0.6], [0.5, 0.2, 0.3], [0.3, 0.4, 0.3],
    ]
    phase_scenarios = ["E1", "E2", "E3", "E4", "E1", "E5", "E3", "E1"]

    all_results = []
    for seed in seeds:
        cycle_details = []
        cycle_returns = []
        for phase_idx, (w, scenario) in enumerate(zip(phase_weights, phase_scenarios)):
            r = gen_episode("PCF-MORL", scenario, seed + phase_idx, weight=w)
            j = float(np.dot(r["episode_return"], w))
            cycle_returns.append(j)
            cycle_details.append({
                "phase": phase_idx,
                "scenario": scenario,
                "weight": w,
                "episode_return": r["episode_return"],
                "J": j,
                "qos": r["qos"],
            })
        all_results.append({
            "seed": seed,
            "cumulative_J": float(sum(cycle_returns)),
            "phase_details": cycle_details,
        })

    return all_results


# ===========================================================================
# Experiment 4: Parallel scaling
# ===========================================================================

def gen_exp4(seeds):
    # Realistic scaling: sub-linear due to sync overhead
    # Base: P=1 takes ~40h for 10K episodes
    base_time_s = 40 * 3600  # 144000s
    f_parallel = 0.88  # 88% parallelizable

    results = []
    for P in [1, 2, 4, 8, 16]:
        for seed in seeds:
            rng = np.random.RandomState(hash((P, seed)) % 2**31)
            # Amdahl's law with noise
            ideal_speedup = 1 / ((1 - f_parallel) + f_parallel / P)
            actual_speedup = ideal_speedup * rng.uniform(0.90, 0.98)
            wall_time = base_time_s / actual_speedup

            results.append({
                "P": P,
                "seed": seed,
                "wall_time_s": float(wall_time),
                "speedup": float(actual_speedup),
                "sync_rounds": max(1, int(10000 / (200 * P))),
                "merged_weight_counts": [int(rng.randint(5, 25)) for _ in range(max(1, int(10000 / (200 * P))))],
            })

    return results


# ===========================================================================
# Experiment 5: K tuning
# ===========================================================================

def gen_exp5(seeds):
    # K=200 is sweet spot (our default)
    k_profiles = {
        50:   {"time_mult": 1.15, "hv_mult": 0.95},   # Too frequent sync = overhead
        100:  {"time_mult": 1.05, "hv_mult": 0.98},
        200:  {"time_mult": 1.00, "hv_mult": 1.00},   # Best
        500:  {"time_mult": 0.98, "hv_mult": 0.96},   # Less sharing
        1000: {"time_mult": 0.97, "hv_mult": 0.90},   # Almost no sharing benefit
    }

    base_time = 40 * 3600 / 6.0  # P=8 baseline

    results = []
    for K, profile in k_profiles.items():
        for seed in seeds:
            rng = np.random.RandomState(hash((K, seed)) % 2**31)
            wall = base_time * profile["time_mult"] * rng.uniform(0.95, 1.05)
            results.append({
                "K": K,
                "seed": seed,
                "wall_time_s": float(wall),
                "sync_rounds": max(1, 10000 // K),
            })

    analysis = {
        "best_K": 200,
        "best_wall_time_s": float(base_time),
        "k_means": {str(k): float(base_time * p["time_mult"]) for k, p in k_profiles.items()},
    }

    return {"results": results, "analysis": analysis}


# ===========================================================================
# Experiment 6: SF sharing ablation
# ===========================================================================

def gen_exp6(seeds):
    # CCS union K=200 is best
    strategy_profiles = {
        "no_sharing":      {"time_mult": 0.95, "hv_mult": 0.75},
        "ccs_union_K200":  {"time_mult": 1.00, "hv_mult": 1.00},
        "ccs_union_K50":   {"time_mult": 1.12, "hv_mult": 0.97},
        "ccs_union_K1000": {"time_mult": 0.97, "hv_mult": 0.88},
    }

    base_time = 40 * 3600 / 6.0

    results = []
    for strategy, profile in strategy_profiles.items():
        for seed in seeds:
            rng = np.random.RandomState(hash((strategy, seed)) % 2**31)
            wall = base_time * profile["time_mult"] * rng.uniform(0.95, 1.05)
            results.append({
                "strategy": strategy,
                "seed": seed,
                "wall_time_s": float(wall),
                "hv_relative": float(profile["hv_mult"] * rng.uniform(0.97, 1.03)),
            })

    return results


# ===========================================================================
# Main
# ===========================================================================

def main():
    seeds = [42, 142, 242, 342, 442]  # 5 seeds
    print("Generating pseudo experiment results...")

    # Exp 1
    exp1_dir = RESULTS_DIR / "exp1"
    exp1_dir.mkdir(parents=True, exist_ok=True)
    exp1 = gen_exp1(seeds)
    with open(exp1_dir / "exp1_results.json", "w") as f:
        json.dump(exp1, f, indent=2)
    print(f"  Exp 1: {len(exp1)} results → {exp1_dir}")

    # Exp 2
    exp2_dir = RESULTS_DIR / "exp2"
    exp2_dir.mkdir(parents=True, exist_ok=True)
    exp2 = gen_exp2(seeds)
    with open(exp2_dir / "exp2_results.json", "w") as f:
        json.dump(exp2, f, indent=2)
    print(f"  Exp 2: ZSR={exp2['analysis']['ZSR']:.3f} → {exp2_dir}")

    # Exp 3
    exp3_dir = RESULTS_DIR / "exp3"
    exp3_dir.mkdir(parents=True, exist_ok=True)
    exp3 = gen_exp3(seeds)
    with open(exp3_dir / "exp3_results.json", "w") as f:
        json.dump(exp3, f, indent=2)
    cum_j = [r["cumulative_J"] for r in exp3]
    print(f"  Exp 3: mean cumJ={np.mean(cum_j):.2f} → {exp3_dir}")

    # Exp 4
    exp4_dir = RESULTS_DIR / "exp4"
    exp4_dir.mkdir(parents=True, exist_ok=True)
    exp4 = gen_exp4(seeds)
    with open(exp4_dir / "exp4_results.json", "w") as f:
        json.dump(exp4, f, indent=2)
    print(f"  Exp 4: {len(exp4)} results → {exp4_dir}")

    # Also save per-P files for figures.py (expects parallel/scaling_P*.json)
    par_dir = RESULTS_DIR / "parallel"
    par_dir.mkdir(parents=True, exist_ok=True)
    by_p = {}
    for r in exp4:
        P = r["P"]
        if P not in by_p:
            by_p[P] = []
        by_p[P].append(r)
    for P, runs in by_p.items():
        avg_wall = float(np.mean([r["wall_time_s"] for r in runs]))
        avg_speedup = float(np.mean([r.get("speedup", 1.0) for r in runs]))
        with open(par_dir / f"scaling_P{P}.json", "w") as f:
            json.dump({
                "num_workers": P,
                "total_wall_time_s": avg_wall,
                "speedup": avg_speedup,
                "runs": runs,
            }, f, indent=2)

    # Exp 5
    exp5_dir = RESULTS_DIR / "exp5"
    exp5_dir.mkdir(parents=True, exist_ok=True)
    exp5 = gen_exp5(seeds)
    with open(exp5_dir / "exp5_results.json", "w") as f:
        json.dump(exp5, f, indent=2)
    print(f"  Exp 5: best K={exp5['analysis']['best_K']} → {exp5_dir}")

    # Exp 6
    exp6_dir = RESULTS_DIR / "exp6"
    exp6_dir.mkdir(parents=True, exist_ok=True)
    exp6 = gen_exp6(seeds)
    with open(exp6_dir / "exp6_results.json", "w") as f:
        json.dump(exp6, f, indent=2)
    print(f"  Exp 6: {len(exp6)} results → {exp6_dir}")

    print("\nAll pseudo results generated. Run M5:")
    print("  python -m paper.figures --results-dir results/")
    print("  python -m paper.tables --results-dir results/")


if __name__ == "__main__":
    main()
