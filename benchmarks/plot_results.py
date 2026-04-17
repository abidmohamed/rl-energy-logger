"""
Generate publication-quality figures from benchmark results.

Produces bar charts, time-series plots, and a radar chart
for the paper.

Usage:
    python plot_results.py
    python plot_results.py --summary-path path/to/results_summary.csv
"""

import argparse
import csv
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmarks.config import OUTPUT_DIR, SUMMARY_CSV, FIGURES_DIR, ALGORITHMS, ENVIRONMENTS

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend for Kaggle/servers
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib not installed. Install with: pip install matplotlib")

try:
    import numpy as np
    HAS_NP = True
except ImportError:
    HAS_NP = False
    print("WARNING: numpy not installed. Install with: pip install numpy")

# Color palette for algorithms
ALGO_COLORS = {
    "PPO": "#3498db",   # Blue
    "A2C": "#2ecc71",   # Green
    "DQN": "#e74c3c",   # Red
    "SAC": "#9b59b6",   # Purple
    "TD3": "#f39c12",   # Orange
}

# Style settings
STYLE = {
    "figure.figsize": (10, 6),
    "font.size": 12,
    "font.family": "serif",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
}


def load_and_group(summary_path: str) -> dict:
    """Load results and group by (algo, env) with mean/std."""
    raw = []
    with open(summary_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") != "success":
                continue
            for key in ["wall_time_s", "mean_reward", "total_energy_kwh",
                         "total_co2_grams", "avg_power_watts", "energy_per_1M_steps_kwh"]:
                try:
                    row[key] = float(row[key])
                except (ValueError, KeyError):
                    row[key] = 0.0
            raw.append(row)

    groups = defaultdict(list)
    for r in raw:
        groups[(r["algo"], r["env"])].append(r)

    stats = {}
    for key, runs in groups.items():
        def _stats(field):
            vals = [r[field] for r in runs]
            mean = sum(vals) / len(vals)
            if len(vals) > 1:
                std = (sum((v - mean)**2 for v in vals) / (len(vals) - 1)) ** 0.5
            else:
                std = 0.0
            return mean, std
        stats[key] = {
            "energy_mean": _stats("total_energy_kwh")[0],
            "energy_std": _stats("total_energy_kwh")[1],
            "co2_mean": _stats("total_co2_grams")[0],
            "co2_std": _stats("total_co2_grams")[1],
            "reward_mean": _stats("mean_reward")[0],
            "reward_std": _stats("mean_reward")[1],
            "time_mean": _stats("wall_time_s")[0],
            "time_std": _stats("wall_time_s")[1],
            "power_mean": _stats("avg_power_watts")[0],
            "power_std": _stats("avg_power_watts")[1],
            "eff_mean": _stats("energy_per_1M_steps_kwh")[0],
            "eff_std": _stats("energy_per_1M_steps_kwh")[1],
        }
    return stats


def plot_energy_by_env(stats: dict, output_dir: str):
    """Fig 1: Bar chart of total energy (kWh) per algorithm, grouped by environment."""
    with plt.rc_context(STYLE):
        envs = [e for e in ENVIRONMENTS if any((a, e) in stats for a in ALGORITHMS)]
        algos = [a for a in ALGORITHMS if any((a, e) in stats for e in envs)]

        fig, axes = plt.subplots(1, len(envs), figsize=(4 * len(envs), 5), sharey=False)
        if len(envs) == 1:
            axes = [axes]

        for ax, env in zip(axes, envs):
            env_algos = [a for a in algos if (a, env) in stats]
            means = [stats[(a, env)]["energy_mean"] * 1000 for a in env_algos]  # mWh
            stds = [stats[(a, env)]["energy_std"] * 1000 for a in env_algos]
            colors = [ALGO_COLORS.get(a, "#95a5a6") for a in env_algos]

            bars = ax.bar(env_algos, means, yerr=stds, capsize=4,
                          color=colors, edgecolor="white", linewidth=0.5, alpha=0.85)
            ax.set_title(env.replace("-", " "), fontweight="bold", fontsize=11)
            ax.set_ylabel("Energy (mWh)" if ax == axes[0] else "")
            ax.tick_params(axis="x", rotation=30)

            # Add value labels on bars
            for bar, mean in zip(bars, means):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{mean:.1f}", ha="center", va="bottom", fontsize=8)

        fig.suptitle("Energy Consumption per Algorithm by Environment",
                     fontweight="bold", fontsize=14, y=1.02)
        plt.tight_layout()
        path = os.path.join(output_dir, "fig1_energy_by_env.pdf")
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.savefig(path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {path}")


def plot_energy_efficiency(stats: dict, output_dir: str):
    """Fig 2: Energy efficiency — kWh per unit of reward achieved."""
    with plt.rc_context(STYLE):
        envs = [e for e in ENVIRONMENTS if any((a, e) in stats for a in ALGORITHMS)]

        fig, axes = plt.subplots(1, len(envs), figsize=(4 * len(envs), 5), sharey=False)
        if len(envs) == 1:
            axes = [axes]

        for ax, env in zip(axes, envs):
            env_algos = [a for a in ALGORITHMS if (a, env) in stats]
            # kWh per unit reward (handle negative rewards like Pendulum)
            effs = []
            for a in env_algos:
                s = stats[(a, env)]
                reward = abs(s["reward_mean"]) if s["reward_mean"] != 0 else 1e-10
                effs.append((s["energy_mean"] / reward) * 1e6)  # μWh per reward unit

            colors = [ALGO_COLORS.get(a, "#95a5a6") for a in env_algos]
            ax.bar(env_algos, effs, color=colors, edgecolor="white",
                   linewidth=0.5, alpha=0.85)
            ax.set_title(env.replace("-", " "), fontweight="bold", fontsize=11)
            ax.set_ylabel("μWh / |reward|" if ax == axes[0] else "")
            ax.tick_params(axis="x", rotation=30)

        fig.suptitle("Energy Efficiency: Energy per Unit of Reward",
                     fontweight="bold", fontsize=14, y=1.02)
        plt.tight_layout()
        path = os.path.join(output_dir, "fig2_energy_efficiency.pdf")
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.savefig(path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {path}")


def plot_on_vs_off_policy(stats: dict, output_dir: str):
    """Fig 3: Grouped bar comparing on-policy vs off-policy energy."""
    with plt.rc_context(STYLE):
        on_policy_algos = ["PPO", "A2C"]
        off_policy_algos = ["DQN", "SAC", "TD3"]

        # Aggregate per category
        on_data = {"energy": [], "eff": [], "time": []}
        off_data = {"energy": [], "eff": [], "time": []}

        for (algo, env), s in stats.items():
            target = on_data if algo in on_policy_algos else off_data
            target["energy"].append(s["energy_mean"] * 1000)  # mWh
            target["eff"].append(s["eff_mean"])
            target["time"].append(s["time_mean"])

        if not on_data["energy"] or not off_data["energy"]:
            print("  Skipping on/off-policy plot (insufficient data)")
            return

        categories = ["Avg Energy\n(mWh)", "Avg kWh/1M steps\n(x1000)", "Avg Time\n(s)"]
        on_vals = [
            sum(on_data["energy"]) / len(on_data["energy"]),
            sum(on_data["eff"]) / len(on_data["eff"]) * 1000,
            sum(on_data["time"]) / len(on_data["time"]),
        ]
        off_vals = [
            sum(off_data["energy"]) / len(off_data["energy"]),
            sum(off_data["eff"]) / len(off_data["eff"]) * 1000,
            sum(off_data["time"]) / len(off_data["time"]),
        ]

        x = range(len(categories))
        width = 0.35

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar([i - width/2 for i in x], on_vals, width, label="On-Policy (PPO, A2C)",
               color="#3498db", alpha=0.85, edgecolor="white")
        ax.bar([i + width/2 for i in x], off_vals, width, label="Off-Policy (DQN, SAC, TD3)",
               color="#e74c3c", alpha=0.85, edgecolor="white")

        ax.set_ylabel("Value")
        ax.set_title("On-Policy vs Off-Policy Energy Profiles", fontweight="bold", fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()

        plt.tight_layout()
        path = os.path.join(output_dir, "fig3_on_vs_off_policy.pdf")
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.savefig(path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {path}")


def plot_overhead(output_dir: str):
    """Fig 4: Tool overhead comparison bar chart."""
    overhead_csv = os.path.join(OUTPUT_DIR, "overhead_results.csv")
    if not os.path.exists(overhead_csv):
        print("  Skipping overhead plot (overhead_results.csv not found)")
        return

    rows = []
    with open(overhead_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    rle_overheads = [float(r["rle_overhead_pct"]) for r in rows]
    cc_overheads = [float(r["codecarbon_overhead_pct"]) for r in rows
                    if r.get("codecarbon_overhead_pct", "N/A") != "N/A"]

    with plt.rc_context(STYLE):
        tools = ["rl-energy-logger"]
        means = [sum(rle_overheads) / len(rle_overheads)]
        stds_vals = [
            (sum((x - means[0])**2 for x in rle_overheads) / max(len(rle_overheads)-1, 1)) ** 0.5
        ]
        colors = ["#3498db"]

        if cc_overheads:
            cc_mean = sum(cc_overheads) / len(cc_overheads)
            cc_std = (sum((x - cc_mean)**2 for x in cc_overheads) / max(len(cc_overheads)-1, 1)) ** 0.5
            tools.append("CodeCarbon")
            means.append(cc_mean)
            stds_vals.append(cc_std)
            colors.append("#e74c3c")

        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(tools, means, yerr=stds_vals, capsize=5,
                      color=colors, edgecolor="white", alpha=0.85)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.set_ylabel("Wall-Clock Overhead (%)")
        ax.set_title("Monitoring Tool Overhead Comparison", fontweight="bold", fontsize=14)

        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{mean:.2f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")

        plt.tight_layout()
        path = os.path.join(output_dir, "fig4_overhead.pdf")
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.savefig(path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {path}")


def main():
    if not HAS_MPL:
        print("Cannot generate plots without matplotlib. Exiting.")
        return

    parser = argparse.ArgumentParser(description="Generate benchmark figures")
    parser.add_argument("--summary-path", type=str, default=SUMMARY_CSV)
    args = parser.parse_args()

    os.makedirs(FIGURES_DIR, exist_ok=True)

    if not os.path.exists(args.summary_path):
        print(f"Error: {args.summary_path} not found. Run run_benchmark.py first.")
        return

    print(f"Loading results from {args.summary_path}")
    stats = load_and_group(args.summary_path)
    print(f"  {len(stats)} (algo, env) groups loaded")

    print("\nGenerating figures:")
    plot_energy_by_env(stats, FIGURES_DIR)
    plot_energy_efficiency(stats, FIGURES_DIR)
    plot_on_vs_off_policy(stats, FIGURES_DIR)
    plot_overhead(FIGURES_DIR)

    print(f"\nAll figures saved to: {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
