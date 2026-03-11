"""M5: LaTeX table generation for GLOBECOM 2026 paper.

Generates:
  - Tab III: QoS E2 (5 methods × 5 metrics)
  - Tab IV: Pareto Quality (4 methods × 4 metrics)

Usage:
    python -m paper.tables --results-dir results/ --output-dir paper/tabs/
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.metrics import aggregate_seeds


def format_val(mean: float, ci: float, fmt: str = ".3f", bold: bool = False) -> str:
    """Format value ± CI for LaTeX, optionally bold."""
    s = f"{mean:{fmt}} $\\pm$ {ci:{fmt}}"
    if bold:
        s = f"\\textbf{{{s}}}"
    return s


def generate_tab3(results: list[dict], output_dir: Path):
    """Tab III: QoS Performance on E2 (URLLC Burst Scenario).

    5 methods × 5 metrics: VR↓, TTR↓, MVD↓, Throughput↑, Energy↓
    """
    # Filter E2 results
    e2_results = [r for r in results if r.get("scenario") == "E2"]
    if not e2_results:
        print("Warning: No E2 results found, generating template table")
        _write_tab3_template(output_dir)
        return

    # Group by method
    methods_data: dict[str, list[dict]] = {}
    for r in e2_results:
        m = r["method"]
        if m not in methods_data:
            methods_data[m] = []
        methods_data[m].append({
            "VR": r["qos"]["VR"],
            "TTR": r["qos"]["TTR"],
            "MVD": r["qos"]["MVD"],
            "Throughput": r["mean_throughput"],
            "Energy": r["mean_energy"],
        })

    # Aggregate across seeds
    method_stats = {}
    for m, data in methods_data.items():
        method_stats[m] = aggregate_seeds(data)

    # Find best per metric
    metric_dirs = {"VR": "min", "TTR": "min", "MVD": "min",
                   "Throughput": "max", "Energy": "min"}
    best = {}
    for metric, direction in metric_dirs.items():
        vals = [(m, s[metric]["mean"]) for m, s in method_stats.items()]
        if direction == "min":
            best[metric] = min(vals, key=lambda x: x[1])[0]
        else:
            best[metric] = max(vals, key=lambda x: x[1])[0]

    # Generate LaTeX
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{QoS Performance on E2 (URLLC Burst Scenario)}",
        r"\label{tab:qos_e2}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Method & VR$\downarrow$ & TTR$\downarrow$ & MVD$\downarrow$ & Thr.$\uparrow$ & Energy$\downarrow$ \\",
        r"\midrule",
    ]

    method_order = ["PCF-MORL", "A1_conservative", "A2_aggressive",
                    "A3_hysteresis", "Scalarized_DQN"]
    method_labels = {
        "PCF-MORL": "PCF-MORL",
        "A1_conservative": "A1 Conserv.",
        "A2_aggressive": "A2 Aggress.",
        "A3_hysteresis": "A3 Hyster.",
        "Scalarized_DQN": "Scalar. DQN",
    }

    for m in method_order:
        if m not in method_stats:
            continue
        s = method_stats[m]
        cells = []
        for metric in ["VR", "TTR", "MVD", "Throughput", "Energy"]:
            is_best = (best.get(metric) == m)
            cells.append(format_val(s[metric]["mean"], s[metric]["ci"],
                                    fmt=".3f", bold=is_best))

        label = method_labels.get(m, m)
        lines.append(f"{label} & {' & '.join(cells)} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    tex = "\n".join(lines)
    out = output_dir / "tab3_qos_e2.tex"
    out.write_text(tex)
    print(f"Saved {out}")


def _write_tab3_template(output_dir: Path):
    """Write template Tab III when no results available."""
    tex = r"""\begin{table}[t]
\centering
\caption{QoS Performance on E2 (URLLC Burst Scenario)}
\label{tab:qos_e2}
\begin{tabular}{lccccc}
\toprule
Method & VR$\downarrow$ & TTR$\downarrow$ & MVD$\downarrow$ & Thr.$\uparrow$ & Energy$\downarrow$ \\
\midrule
PCF-MORL     & --- & --- & --- & --- & --- \\
A1 Conserv.  & --- & --- & --- & --- & --- \\
A2 Aggress.  & --- & --- & --- & --- & --- \\
A3 Hyster.   & --- & --- & --- & --- & --- \\
Scalar. DQN  & --- & --- & --- & --- & --- \\
\bottomrule
\end{tabular}
\end{table}"""
    out = output_dir / "tab3_qos_e2.tex"
    out.write_text(tex)
    print(f"Saved template {out}")


def generate_tab4(results: list[dict], output_dir: Path):
    """Tab IV: Pareto Quality (4 methods × 4 metrics).

    HV↑, EU↑, MUL↓, |CCS|
    """
    if not results:
        _write_tab4_template(output_dir)
        return

    # Group by method, compute Pareto metrics
    methods_data: dict[str, list[list]] = {}
    for r in results:
        m = r["method"]
        if m not in methods_data:
            methods_data[m] = []
        methods_data[m].append(r["episode_return"])

    from experiments.metrics import compute_hypervolume

    ref_point = np.array([0.0, -100.0, -100.0])
    method_stats = {}

    for m, returns in methods_data.items():
        points = np.array(returns)
        hv = compute_hypervolume(points, ref_point)
        method_stats[m] = {
            "HV": hv,
            "EU": float(np.mean(np.sum(points, axis=1))),
            "MUL": 0.0,  # Requires oracle comparison
            "CCS": len(points),
        }

    # Generate LaTeX
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Pareto Front Quality}",
        r"\label{tab:pareto}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Method & HV$\uparrow$ & EU$\uparrow$ & MUL$\downarrow$ & $|$CCS$|$ \\",
        r"\midrule",
    ]

    for m, s in method_stats.items():
        label = m.replace("_", " ")
        lines.append(
            f"{label} & {s['HV']:.2f} & {s['EU']:.2f} & "
            f"{s['MUL']:.3f} & {s['CCS']} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    tex = "\n".join(lines)
    out = output_dir / "tab4_pareto.tex"
    out.write_text(tex)
    print(f"Saved {out}")


def _write_tab4_template(output_dir: Path):
    """Write template Tab IV when no results available."""
    tex = r"""\begin{table}[t]
\centering
\caption{Pareto Front Quality}
\label{tab:pareto}
\begin{tabular}{lcccc}
\toprule
Method & HV$\uparrow$ & EU$\uparrow$ & MUL$\downarrow$ & $|$CCS$|$ \\
\midrule
PCF-MORL     & --- & --- & --- & --- \\
Scalar. DQN  & --- & --- & --- & --- \\
Oracle DQN   & --- & --- & --- & --- \\
Sequential   & --- & --- & --- & --- \\
\bottomrule
\end{tabular}
\end{table}"""
    out = output_dir / "tab4_pareto.tex"
    out.write_text(tex)
    print(f"Saved template {out}")


def main():
    parser = argparse.ArgumentParser(description="Generate paper tables")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--output-dir", type=str, default="paper/tabs")
    parser.add_argument("--table", type=str, default="all",
                        help="Which table (3, 4, or 'all')")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    exp1_path = results_dir / "exp1" / "exp1_results.json"
    exp1 = []
    if exp1_path.exists():
        with open(exp1_path) as f:
            exp1 = json.load(f)

    if args.table in ("all", "3"):
        generate_tab3(exp1, output_dir)

    if args.table in ("all", "4"):
        generate_tab4(exp1, output_dir)


if __name__ == "__main__":
    main()
