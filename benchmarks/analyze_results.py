"""
Post-hoc analysis of benchmark results.

Loads results_summary.csv, computes aggregate statistics,
and prints formatted tables for the paper.

Usage:
    python analyze_results.py
    python analyze_results.py --summary-path path/to/results_summary.csv
"""

import argparse
import csv
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmarks.config import OUTPUT_DIR, SUMMARY_CSV, ALGORITHMS, ENVIRONMENTS


def load_summary(path: str) -> list:
    """Load results_summary.csv into a list of dicts."""
    results = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            for key in ["seed", "timesteps"]:
                row[key] = int(row[key])
            for key in ["wall_time_s", "mean_reward", "std_reward",
                         "total_energy_kwh", "total_co2_grams",
                         "avg_power_watts", "energy_per_1M_steps_kwh"]:
                try:
                    row[key] = float(row[key])
                except (ValueError, KeyError):
                    row[key] = 0.0
            results.append(row)
    return results


def compute_stats(values: list) -> dict:
    """Compute mean and std of a list of numbers."""
    n = len(values)
    if n == 0:
        return {"mean": 0.0, "std": 0.0, "n": 0}
    mean = sum(values) / n
    if n > 1:
        variance = sum((x - mean) ** 2 for x in values) / (n - 1)
        std = variance ** 0.5
    else:
        std = 0.0
    return {"mean": mean, "std": std, "n": n}


def group_results(results: list) -> dict:
    """
    Group results by (algo, env) and compute statistics.

    Returns:
        dict[(algo, env)] -> dict of stats for each metric
    """
    groups = defaultdict(list)
    for r in results:
        if r.get("status") != "success":
            continue
        key = (r["algo"], r["env"])
        groups[key].append(r)

    stats = {}
    for key, runs in groups.items():
        stats[key] = {
            "wall_time_s": compute_stats([r["wall_time_s"] for r in runs]),
            "mean_reward": compute_stats([r["mean_reward"] for r in runs]),
            "total_energy_kwh": compute_stats([r["total_energy_kwh"] for r in runs]),
            "total_co2_grams": compute_stats([r["total_co2_grams"] for r in runs]),
            "avg_power_watts": compute_stats([r["avg_power_watts"] for r in runs]),
            "energy_per_1M_steps_kwh": compute_stats([r["energy_per_1M_steps_kwh"] for r in runs]),
            "timesteps": runs[0]["timesteps"],
            "n_seeds": len(runs),
        }
    return stats


def print_main_table(stats: dict):
    """Print the main results table (Table 1 in the paper)."""
    print("\n" + "=" * 120)
    print("TABLE 1: Benchmark Results (mean +/- std across seeds)")
    print("=" * 120)
    header = f"{'Algorithm':>6} | {'Environment':>20} | {'Steps':>8} | {'Reward':>16} | " \
             f"{'Time (s)':>12} | {'Energy (kWh)':>16} | {'CO2 (g)':>14} | {'kWh/1M steps':>14}"
    print(header)
    print("-" * 120)

    for env_name in ENVIRONMENTS:
        for algo in ALGORITHMS:
            key = (algo, env_name)
            if key not in stats:
                continue
            s = stats[key]
            reward = s["mean_reward"]
            wall = s["wall_time_s"]
            energy = s["total_energy_kwh"]
            co2 = s["total_co2_grams"]
            eff = s["energy_per_1M_steps_kwh"]

            print(f"{algo:>6} | {env_name:>20} | {s['timesteps']:>8,} | "
                  f"{reward['mean']:>7.1f} +/- {reward['std']:>5.1f} | "
                  f"{wall['mean']:>5.1f} +/- {wall['std']:>4.1f} | "
                  f"{energy['mean']:>7.6f} +/- {energy['std']:.6f} | "
                  f"{co2['mean']:>6.4f} +/- {co2['std']:.4f} | "
                  f"{eff['mean']:>7.6f}")
        print("-" * 120)


def print_algo_summary(stats: dict):
    """Print per-algorithm aggregated summary."""
    print("\n" + "=" * 80)
    print("TABLE 2: Per-Algorithm Energy Summary (aggregated across environments)")
    print("=" * 80)

    algo_data = defaultdict(lambda: {"energy": [], "co2": [], "power": [], "time": []})

    for (algo, env), s in stats.items():
        algo_data[algo]["energy"].append(s["total_energy_kwh"]["mean"])
        algo_data[algo]["co2"].append(s["total_co2_grams"]["mean"])
        algo_data[algo]["power"].append(s["avg_power_watts"]["mean"])
        algo_data[algo]["time"].append(s["wall_time_s"]["mean"])

    print(f"{'Algorithm':>6} | {'Avg Energy (kWh)':>16} | {'Avg CO2 (g)':>14} | "
          f"{'Avg Power (W)':>14} | {'Avg Time (s)':>12} | {'Envs':>5}")
    print("-" * 80)

    for algo in ALGORITHMS:
        if algo not in algo_data:
            continue
        d = algo_data[algo]
        n = len(d["energy"])
        print(f"{algo:>6} | "
              f"{sum(d['energy'])/n:>16.6f} | "
              f"{sum(d['co2'])/n:>14.4f} | "
              f"{sum(d['power'])/n:>14.1f} | "
              f"{sum(d['time'])/n:>12.1f} | "
              f"{n:>5}")


def print_env_summary(stats: dict):
    """Print per-environment aggregated summary."""
    print("\n" + "=" * 80)
    print("TABLE 3: Per-Environment Energy Summary (aggregated across algorithms)")
    print("=" * 80)

    env_data = defaultdict(lambda: {"energy": [], "co2": [], "power": [], "time": []})

    for (algo, env), s in stats.items():
        env_data[env]["energy"].append(s["total_energy_kwh"]["mean"])
        env_data[env]["co2"].append(s["total_co2_grams"]["mean"])
        env_data[env]["power"].append(s["avg_power_watts"]["mean"])
        env_data[env]["time"].append(s["wall_time_s"]["mean"])

    print(f"{'Environment':>20} | {'Avg Energy (kWh)':>16} | {'Avg CO2 (g)':>14} | "
          f"{'Avg Power (W)':>14} | {'Avg Time (s)':>12} | {'Algos':>5}")
    print("-" * 80)

    for env_name in ENVIRONMENTS:
        if env_name not in env_data:
            continue
        d = env_data[env_name]
        n = len(d["energy"])
        print(f"{env_name:>20} | "
              f"{sum(d['energy'])/n:>16.6f} | "
              f"{sum(d['co2'])/n:>14.4f} | "
              f"{sum(d['power'])/n:>14.1f} | "
              f"{sum(d['time'])/n:>12.1f} | "
              f"{n:>5}")


def print_on_vs_off_policy(stats: dict):
    """Compare on-policy vs off-policy energy profiles."""
    print("\n" + "=" * 60)
    print("TABLE 4: On-Policy vs Off-Policy Energy Comparison")
    print("=" * 60)

    on_policy = {"energy": [], "co2": [], "eff": []}
    off_policy = {"energy": [], "co2": [], "eff": []}

    on_policy_algos = {"PPO", "A2C"}
    off_policy_algos = {"DQN", "SAC", "TD3"}

    for (algo, env), s in stats.items():
        target = on_policy if algo in on_policy_algos else off_policy
        target["energy"].append(s["total_energy_kwh"]["mean"])
        target["co2"].append(s["total_co2_grams"]["mean"])
        target["eff"].append(s["energy_per_1M_steps_kwh"]["mean"])

    for label, data in [("On-Policy (PPO, A2C)", on_policy),
                        ("Off-Policy (DQN, SAC, TD3)", off_policy)]:
        n = len(data["energy"])
        if n == 0:
            continue
        print(f"\n  {label}:")
        print(f"    Avg Energy:       {sum(data['energy'])/n:.6f} kWh")
        print(f"    Avg CO2:          {sum(data['co2'])/n:.4f} g")
        print(f"    Avg kWh/1M steps: {sum(data['eff'])/n:.6f}")


def export_latex_table(stats: dict, output_path: str):
    """Export results as a LaTeX table for the paper."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{Energy consumption of RL algorithms across Gymnasium environments.}\n")
        f.write("\\label{tab:results}\n")
        f.write("\\begin{tabular}{llrrrrr}\n")
        f.write("\\toprule\n")
        f.write("Algorithm & Environment & Steps & Reward & Time (s) & Energy (kWh) & CO\\textsubscript{2} (g) \\\\\n")
        f.write("\\midrule\n")

        for env_name in ENVIRONMENTS:
            for algo in ALGORITHMS:
                key = (algo, env_name)
                if key not in stats:
                    continue
                s = stats[key]
                r = s["mean_reward"]
                w = s["wall_time_s"]
                e = s["total_energy_kwh"]
                c = s["total_co2_grams"]

                f.write(f"{algo} & {env_name.replace('_', '\\_')} & "
                        f"{s['timesteps']:,} & "
                        f"${r['mean']:.1f} \\pm {r['std']:.1f}$ & "
                        f"${w['mean']:.1f}$ & "
                        f"${e['mean']:.6f}$ & "
                        f"${c['mean']:.4f}$ \\\\\n")
            f.write("\\midrule\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    print(f"\n  LaTeX table saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze benchmark results")
    parser.add_argument("--summary-path", type=str, default=SUMMARY_CSV,
                        help="Path to results_summary.csv")
    parser.add_argument("--latex", action="store_true",
                        help="Also export LaTeX table")
    args = parser.parse_args()

    if not os.path.exists(args.summary_path):
        print(f"Error: {args.summary_path} not found. Run run_benchmark.py first.")
        return

    results = load_summary(args.summary_path)
    print(f"Loaded {len(results)} results from {args.summary_path}")

    successful = [r for r in results if r.get("status") == "success"]
    failed = [r for r in results if r.get("status") != "success"]
    print(f"  Successful: {len(successful)}, Failed: {len(failed)}")

    if failed:
        print("  Failed runs:")
        for r in failed:
            print(f"    {r['algo']} x {r['env']} (seed={r['seed']}): {r['status']}")

    stats = group_results(results)

    print_main_table(stats)
    print_algo_summary(stats)
    print_env_summary(stats)
    print_on_vs_off_policy(stats)

    if args.latex:
        latex_path = os.path.join(OUTPUT_DIR, "table_results.tex")
        export_latex_table(stats, latex_path)


if __name__ == "__main__":
    main()
