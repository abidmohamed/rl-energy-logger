"""
Main benchmarking script for the RL Energy Benchmark study.

Trains each (algorithm, environment, seed) combination using SB3,
logs hardware metrics via EnergyLogCallback, evaluates the final policy,
and saves results to CSV.

Usage:
    # Full benchmark
    python run_benchmark.py

    # Dry run (1000 steps per config, for testing)
    python run_benchmark.py --dry-run

    # Single run
    python run_benchmark.py --algo PPO --env CartPole-v1 --seed 42
"""

import argparse
import csv
import os
import sys
import time
import warnings

import gymnasium as gym
from stable_baselines3 import PPO, A2C, DQN, SAC, TD3
from stable_baselines3.common.evaluation import evaluate_policy

# Add parent directory to path so we can import the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_energy_logger import EnergyLogCallback, EnergyEstimator
from benchmarks.config import (
    ALGORITHMS,
    ENVIRONMENTS,
    SEEDS,
    SAMPLING_INTERVAL_S,
    CO2_REGION,
    DEVICE,
    OUTPUT_DIR,
    SUMMARY_CSV,
    ENERGY_LOGS_DIR,
    get_valid_combinations,
    get_full_matrix,
)

# Map algorithm names to SB3 classes
ALGO_CLASSES = {
    "PPO": PPO,
    "A2C": A2C,
    "DQN": DQN,
    "SAC": SAC,
    "TD3": TD3,
}

# Summary CSV header
SUMMARY_HEADER = [
    "algo", "env", "seed", "timesteps",
    "wall_time_s", "mean_reward", "std_reward",
    "total_energy_kwh", "total_co2_grams",
    "avg_power_watts", "energy_per_1M_steps_kwh",
    "gpu_name", "status",
]


def compute_energy_from_log(log_path: str, region: str = "world"):
    """
    Read an energy log CSV and compute total kWh and CO2 using EnergyEstimator.

    Returns:
        dict with total_energy_kwh, total_co2_grams, avg_power_watts, sample_count
    """
    estimator = EnergyEstimator(region=region)

    try:
        with open(log_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            prev_ts = None
            for row in reader:
                ts = float(row.get("timestamp", 0))
                power_str = row.get("gpu_power_watts", "")

                if power_str and power_str.lower() not in ("", "none"):
                    power = float(power_str)
                    if prev_ts is not None:
                        dt = ts - prev_ts
                        if 0 < dt < 30:  # Ignore gaps > 30s (likely pauses)
                            estimator.update(power_watts=power, duration_seconds=dt)
                prev_ts = ts
    except FileNotFoundError:
        warnings.warn(f"Energy log not found: {log_path}")
    except Exception as e:
        warnings.warn(f"Error reading energy log {log_path}: {e}")

    return estimator.summary()


def run_single(algo_name: str, env_name: str, seed: int,
               timesteps: int, eval_episodes: int = 10) -> dict:
    """
    Train one (algo, env, seed) configuration and return results.
    """
    run_id = f"{algo_name}_{env_name}_{seed}"
    log_path = os.path.join(ENERGY_LOGS_DIR, f"{run_id}.csv")

    print(f"\n{'='*60}")
    print(f"  {algo_name} x {env_name} (seed={seed}, steps={timesteps:,})")
    print(f"  Log: {log_path}")
    print(f"{'='*60}")

    # Create environment
    env = gym.make(env_name)

    # Auto-detect GPU availability
    track_gpu = False
    try:
        import pynvml
        pynvml.nvmlInit()
        pynvml.nvmlShutdown()
        track_gpu = True
    except Exception:
        pass  # No GPU available — CPU-only monitoring

    callback = EnergyLogCallback(
        log_path=log_path,
        sampling_interval_s=SAMPLING_INTERVAL_S,
        track_gpu=track_gpu,
        log_rl_metrics=True,
        verbose=0,
    )

    # Create model
    algo_class = ALGO_CLASSES[algo_name]
    model = algo_class(
        "MlpPolicy",
        env,
        seed=seed,
        verbose=0,
        device=DEVICE,
    )

    # Train
    start_time = time.time()
    try:
        model.learn(total_timesteps=timesteps, callback=callback)
        status = "success"
    except Exception as e:
        warnings.warn(f"Training failed for {run_id}: {e}")
        status = f"error: {e}"
    wall_time = time.time() - start_time

    # Evaluate
    mean_reward, std_reward = 0.0, 0.0
    if status == "success":
        try:
            mean_reward, std_reward = evaluate_policy(
                model, env, n_eval_episodes=eval_episodes, deterministic=True
            )
        except Exception as e:
            warnings.warn(f"Evaluation failed for {run_id}: {e}")

    env.close()

    # Compute energy from log
    energy = compute_energy_from_log(log_path, region=CO2_REGION)

    # Compute energy per 1M steps
    energy_per_1M = 0.0
    if energy["sample_count"] > 0 and timesteps > 0:
        energy_per_1M = (energy["total_energy_kwh"] / timesteps) * 1_000_000

    # GPU name (try to get from pynvml)
    gpu_name = "N/A"
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu_name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(gpu_name, bytes):
            gpu_name = gpu_name.decode("utf-8")
        pynvml.nvmlShutdown()
    except Exception:
        pass

    result = {
        "algo": algo_name,
        "env": env_name,
        "seed": seed,
        "timesteps": timesteps,
        "wall_time_s": round(wall_time, 2),
        "mean_reward": round(mean_reward, 2),
        "std_reward": round(std_reward, 2),
        "total_energy_kwh": energy["total_energy_kwh"],
        "total_co2_grams": energy["total_co2_grams"],
        "avg_power_watts": energy["average_power_watts"],
        "energy_per_1M_steps_kwh": round(energy_per_1M, 6),
        "gpu_name": gpu_name,
        "status": status,
    }

    print(f"  OK Done in {wall_time:.1f}s | Reward: {mean_reward:.1f}+/-{std_reward:.1f} "
          f"| Energy: {energy['total_energy_kwh']:.6f} kWh | CO2: {energy['total_co2_grams']:.4f} g")

    return result


def append_to_summary(result: dict, summary_path: str):
    """Append a single result row to the summary CSV."""
    file_exists = os.path.exists(summary_path) and os.path.getsize(summary_path) > 0
    with open(summary_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_HEADER)
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)


def main():
    parser = argparse.ArgumentParser(description="RL Energy Benchmark")
    parser.add_argument("--dry-run", action="store_true",
                        help="Quick test: 1000 steps per config")
    parser.add_argument("--algo", type=str, default=None,
                        help="Run only this algorithm (e.g., PPO)")
    parser.add_argument("--env", type=str, default=None,
                        help="Run only this environment (e.g., CartPole-v1)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Run only this seed")
    args = parser.parse_args()

    # Create output directories
    os.makedirs(ENERGY_LOGS_DIR, exist_ok=True)

    # Build experiment matrix
    if args.algo and args.env and args.seed is not None:
        matrix = [(args.algo, args.env, args.seed)]
    elif args.algo:
        matrix = [(a, e, s) for a, e, s in get_full_matrix() if a == args.algo]
    elif args.env:
        matrix = [(a, e, s) for a, e, s in get_full_matrix() if e == args.env]
    else:
        matrix = get_full_matrix()

    if not matrix:
        print("No valid configurations found. Check --algo/--env arguments.")
        return

    print(f"RL Energy Benchmark")
    print(f"  Configurations: {len(matrix)}")
    print(f"  Dry run: {args.dry_run}")
    print(f"  Output: {OUTPUT_DIR}/")

    total_start = time.time()
    completed = 0

    for algo, env_name, seed in matrix:
        env_cfg = ENVIRONMENTS[env_name]
        timesteps = 1000 if args.dry_run else env_cfg["timesteps"]

        result = run_single(
            algo_name=algo,
            env_name=env_name,
            seed=seed,
            timesteps=timesteps,
            eval_episodes=env_cfg["eval_episodes"],
        )
        append_to_summary(result, SUMMARY_CSV)
        completed += 1

        elapsed = time.time() - total_start
        remaining = len(matrix) - completed
        est_per_run = elapsed / completed
        print(f"\n  Progress: {completed}/{len(matrix)} "
              f"({elapsed:.0f}s elapsed, ~{est_per_run * remaining:.0f}s remaining)")

    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  BENCHMARK COMPLETE: {completed} runs in {total_elapsed:.1f}s")
    print(f"  Results: {SUMMARY_CSV}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
