"""
Tool overhead benchmark: rl-energy-logger vs CodeCarbon vs bare training.

Measures the wall-clock overhead of each monitoring tool on the same
training workload (PPO on CartPole, 100K steps).

Usage:
    python run_overhead.py
    python run_overhead.py --dry-run
"""

import argparse
import csv
import os
import sys
import time
import warnings

import gymnasium as gym
from stable_baselines3 import PPO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_energy_logger import EnergyLogCallback
from benchmarks.config import OUTPUT_DIR, DEVICE

# Overhead benchmark settings
OVERHEAD_ENV = "CartPole-v1"
OVERHEAD_ALGO = "PPO"
OVERHEAD_TIMESTEPS = 100_000
OVERHEAD_SEEDS = [42, 123, 456, 789, 1024]
OVERHEAD_CSV = f"{OUTPUT_DIR}/overhead_results.csv"


def run_bare(seed: int, timesteps: int) -> float:
    """Train with no monitoring. Returns wall-clock seconds."""
    env = gym.make(OVERHEAD_ENV)
    model = PPO("MlpPolicy", env, seed=seed, verbose=0, device=DEVICE)
    start = time.time()
    model.learn(total_timesteps=timesteps)
    elapsed = time.time() - start
    env.close()
    return elapsed


def run_with_rl_energy_logger(seed: int, timesteps: int) -> float:
    """Train with rl-energy-logger callback. Returns wall-clock seconds."""
    log_path = os.path.join(OUTPUT_DIR, f"overhead_rle_{seed}.csv")
    env = gym.make(OVERHEAD_ENV)
    model = PPO("MlpPolicy", env, seed=seed, verbose=0, device=DEVICE)
    callback = EnergyLogCallback(
        log_path=log_path,
        sampling_interval_s=1.0,
        track_gpu=True,
        log_rl_metrics=False,
        verbose=0,
    )
    start = time.time()
    model.learn(total_timesteps=timesteps, callback=callback)
    elapsed = time.time() - start
    env.close()
    # Cleanup log file
    try:
        os.remove(log_path)
    except OSError:
        pass
    return elapsed


def run_with_codecarbon(seed: int, timesteps: int) -> float:
    """Train with CodeCarbon tracker. Returns wall-clock seconds."""
    try:
        from codecarbon import EmissionsTracker
    except ImportError:
        warnings.warn("CodeCarbon not installed. Skipping. Install with: pip install codecarbon")
        return -1.0

    env = gym.make(OVERHEAD_ENV)
    model = PPO("MlpPolicy", env, seed=seed, verbose=0, device=DEVICE)
    tracker = EmissionsTracker(
        project_name="overhead_benchmark",
        output_dir=OUTPUT_DIR,
        log_level="error",   # Suppress verbose output
        save_to_file=False,
    )
    tracker.start()
    start = time.time()
    model.learn(total_timesteps=timesteps)
    elapsed = time.time() - start
    tracker.stop()
    env.close()
    return elapsed


def main():
    parser = argparse.ArgumentParser(description="Tool Overhead Benchmark")
    parser.add_argument("--dry-run", action="store_true",
                        help="Quick test: 1000 steps")
    args = parser.parse_args()

    timesteps = 1000 if args.dry_run else OVERHEAD_TIMESTEPS
    seeds = OVERHEAD_SEEDS

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Tool Overhead Benchmark")
    print(f"  {OVERHEAD_ALGO} x {OVERHEAD_ENV} x {timesteps:,} steps x {len(seeds)} seeds")
    print(f"  Comparing: bare | rl-energy-logger | CodeCarbon")
    print()

    results = []

    for seed in seeds:
        print(f"  Seed {seed}:")

        # Bare
        t_bare = run_bare(seed, timesteps)
        print(f"    Bare:              {t_bare:.2f}s")

        # rl-energy-logger
        t_rle = run_with_rl_energy_logger(seed, timesteps)
        overhead_rle = ((t_rle - t_bare) / t_bare) * 100 if t_bare > 0 else 0
        print(f"    rl-energy-logger:  {t_rle:.2f}s  ({overhead_rle:+.2f}%)")

        # CodeCarbon
        t_cc = run_with_codecarbon(seed, timesteps)
        if t_cc >= 0:
            overhead_cc = ((t_cc - t_bare) / t_bare) * 100 if t_bare > 0 else 0
            print(f"    CodeCarbon:        {t_cc:.2f}s  ({overhead_cc:+.2f}%)")
        else:
            overhead_cc = None
            print(f"    CodeCarbon:        SKIPPED (not installed)")

        results.append({
            "seed": seed,
            "timesteps": timesteps,
            "bare_s": round(t_bare, 3),
            "rle_s": round(t_rle, 3),
            "rle_overhead_pct": round(overhead_rle, 3),
            "codecarbon_s": round(t_cc, 3) if t_cc >= 0 else "N/A",
            "codecarbon_overhead_pct": round(overhead_cc, 3) if overhead_cc is not None else "N/A",
        })

    # Save results
    with open(OVERHEAD_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    # Print summary
    bare_times = [r["bare_s"] for r in results]
    rle_overheads = [r["rle_overhead_pct"] for r in results]
    cc_overheads = [r["codecarbon_overhead_pct"] for r in results
                    if r["codecarbon_overhead_pct"] != "N/A"]

    print(f"\n{'='*50}")
    print(f"  OVERHEAD SUMMARY")
    print(f"  Bare mean:             {sum(bare_times)/len(bare_times):.2f}s")
    print(f"  rl-energy-logger:      {sum(rle_overheads)/len(rle_overheads):+.2f}% overhead")
    if cc_overheads:
        print(f"  CodeCarbon:            {sum(cc_overheads)/len(cc_overheads):+.2f}% overhead")
    print(f"  Results saved to: {OVERHEAD_CSV}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
