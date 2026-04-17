"""
Example: Using EnergyLoggerContext (generic context manager)
============================================================

This example demonstrates how to use the ``EnergyLoggerContext``
to log hardware metrics during any Python training loop — no
specific framework integration required.

It also shows how to use the ``EnergyEstimator`` to compute
cumulative energy (kWh) and CO₂ emissions from the logged data.

Usage::

    pip install rl-energy-logger[gpu]   # or just: pip install rl-energy-logger
    python example_context_manager.py
"""

import time
import json
from rl_energy_logger import EnergyLoggerContext, EnergyEstimator

LOG_PATH = "training_log.jsonl"
NUM_EPOCHS = 5
STEPS_PER_EPOCH = 20


def fake_training_step():
    """Simulate a training step with a short sleep."""
    time.sleep(0.05)  # ~50ms per step


def main():
    # --- 1. Use the context manager to log hardware metrics ---
    print(f"Starting training — logging hardware metrics to '{LOG_PATH}'")

    with EnergyLoggerContext(
        log_path=LOG_PATH,
        sampling_interval_s=0.5,   # Sample hardware every 0.5 seconds
        track_gpu=False,           # Set to True if you have an NVIDIA GPU
    ) as ctx:
        for epoch in range(1, NUM_EPOCHS + 1):
            epoch_start = time.time()

            for step in range(STEPS_PER_EPOCH):
                fake_training_step()

            epoch_duration = time.time() - epoch_start

            # Manually log epoch-level metrics alongside a hardware snapshot
            ctx.log(
                event="epoch_end",
                epoch=epoch,
                epoch_duration_s=round(epoch_duration, 3),
                fake_loss=1.0 / epoch,      # Decreasing fake loss
                fake_accuracy=0.5 + 0.1 * epoch,
            )
            print(f"  Epoch {epoch}/{NUM_EPOCHS} — loss={1.0/epoch:.3f}")

    print(f"Training complete. Log saved to '{LOG_PATH}'")

    # --- 2. Post-hoc energy estimation from the log ---
    print("\n--- Energy Summary ---")
    estimator = EnergyEstimator(region="world")

    with open(LOG_PATH, "r", encoding="utf-8") as f:
        prev_timestamp = None
        for line in f:
            record = json.loads(line)
            ts = record.get("timestamp")
            power = record.get("gpu_power_watts")  # Will be None without GPU

            if power is not None and prev_timestamp is not None:
                dt = ts - prev_timestamp
                estimator.update(power_watts=power, duration_seconds=dt)

            prev_timestamp = ts

    summary = estimator.summary()
    if summary["sample_count"] == 0:
        print("  No GPU power samples found (GPU tracking was disabled).")
        print("  To see energy estimates, re-run with track_gpu=True on a machine with an NVIDIA GPU.")
    else:
        print(f"  Total Energy : {summary['total_energy_kwh']:.6f} kWh")
        print(f"  Total CO₂    : {summary['total_co2_grams']:.2f} gCO₂eq")
        print(f"  Avg Power    : {summary['average_power_watts']:.1f} W")
        print(f"  Duration     : {summary['total_duration_hours']:.4f} hours")
        print(f"  Region       : {summary['region']}")


if __name__ == "__main__":
    main()
