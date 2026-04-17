"""
Example: Using EnergyLogCallback with Stable-Baselines3
========================================================

This example demonstrates how to use the ``EnergyLogCallback``
to automatically log hardware metrics during SB3 training.

Requirements::

    pip install rl-energy-logger[sb3]
    pip install gymnasium            # For the CartPole environment

Usage::

    python example_sb3_callback.py
"""

import os

try:
    import gymnasium as gym
except ImportError:
    import gym  # Fallback for older gym versions

from stable_baselines3 import PPO
from rl_energy_logger import EnergyLogCallback, EnergyEstimator

LOG_PATH = "sb3_energy_log.csv"
TOTAL_TIMESTEPS = 10_000


def main():
    # --- 1. Create the environment ---
    env = gym.make("CartPole-v1")

    # --- 2. Create the energy-logging callback ---
    energy_callback = EnergyLogCallback(
        log_path=LOG_PATH,
        sampling_interval_s=1.0,    # Log hardware metrics every 1 second
        track_gpu=False,            # Set True if training on GPU
        log_rl_metrics=True,        # Also capture SB3's internal metrics
        verbose=1,
    )

    # --- 3. Create and train the model ---
    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        device="cpu",  # Use "cuda" for GPU training
    )

    print(f"Training PPO on CartPole-v1 for {TOTAL_TIMESTEPS} timesteps...")
    print(f"Hardware metrics will be logged to '{LOG_PATH}'")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=energy_callback,
    )
    print("Training complete!")

    # --- 4. Post-hoc energy summary ---
    print("\n--- Energy Summary ---")

    # Read the CSV log and compute energy
    import csv

    estimator = EnergyEstimator(region="world")
    prev_timestamp = None

    with open(LOG_PATH, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = float(row["timestamp"])
            power_str = row.get("gpu_power_watts", "")

            if power_str and power_str != "None":
                power = float(power_str)
                if prev_timestamp is not None:
                    dt = ts - prev_timestamp
                    estimator.update(power_watts=power, duration_seconds=dt)

            prev_timestamp = ts

    summary = estimator.summary()
    if summary["sample_count"] == 0:
        print("  No GPU power samples (GPU tracking was disabled).")
        print("  Enable track_gpu=True on a machine with an NVIDIA GPU to see energy estimates.")
    else:
        print(f"  Total Energy : {summary['total_energy_kwh']:.6f} kWh")
        print(f"  Total CO₂    : {summary['total_co2_grams']:.2f} gCO₂eq ({summary['region']})")
        print(f"  Avg Power    : {summary['average_power_watts']:.1f} W")

    print(f"\nFull log available at: {os.path.abspath(LOG_PATH)}")

    env.close()


if __name__ == "__main__":
    main()
