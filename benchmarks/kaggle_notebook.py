"""
Kaggle Notebook Script — RL Energy Benchmark
=============================================

Copy-paste each cell block into a Kaggle notebook.
Enable GPU T4 accelerator before running.

Prerequisites:
- Upload the rl-energy-logger repo as a Kaggle dataset
  (the dataset will appear at /kaggle/input/rl-energy-logger/)
"""

# ===========================================================================
# Cell 1: Setup — install dependencies and copy repo
# ===========================================================================

# Install swig first (needed to build box2d-py for BipedalWalker)
# Then install SB3 with box2d support, and other deps
# !apt-get install -y swig > /dev/null 2>&1
# !pip install -q "stable-baselines3[extra]>=2.0.0" "gymnasium[box2d]" codecarbon matplotlib

# Copy repo from read-only input to working directory (editable install needs write access)
# !cp -r /kaggle/input/rl-energy-logger /kaggle/working/rl-energy-logger-src
# !pip install -q -e /kaggle/working/rl-energy-logger-src

# ===========================================================================
# Cell 2: Verify GPU and imports
# ===========================================================================

import subprocess
print("=== GPU Check ===")
try:
    result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                             "--format=csv,noheader"],
                            capture_output=True, text=True)
    print(result.stdout.strip())
except Exception:
    print("No NVIDIA GPU found!")

import gymnasium as gym
from stable_baselines3 import PPO, A2C, DQN, SAC, TD3
from rl_energy_logger import EnergyLogCallback, EnergyEstimator
print("\nAll imports successful!")

# ===========================================================================
# Cell 3: Run the full benchmark
# ===========================================================================

import os
import sys

# Add the working copy to path
sys.path.insert(0, "/kaggle/working/rl-energy-logger-src")

from benchmarks.run_benchmark import main as run_benchmark_main

# Run full benchmark (or uncomment --dry-run for quick test)
sys.argv = ["run_benchmark.py"]
# sys.argv = ["run_benchmark.py", "--dry-run"]  # Uncomment for quick test
run_benchmark_main()

# ===========================================================================
# Cell 4: Run overhead benchmark
# ===========================================================================

from benchmarks.run_overhead import main as run_overhead_main

sys.argv = ["run_overhead.py"]
# sys.argv = ["run_overhead.py", "--dry-run"]  # Uncomment for quick test
run_overhead_main()

# ===========================================================================
# Cell 5: Analyze results
# ===========================================================================

from benchmarks.analyze_results import main as analyze_main

sys.argv = ["analyze_results.py", "--latex"]
analyze_main()

# ===========================================================================
# Cell 6: Generate figures
# ===========================================================================

from benchmarks.plot_results import main as plot_main

sys.argv = ["plot_results.py"]
plot_main()

print("\nAll done! Download benchmark_results/ for the data and figures.")

# ===========================================================================
# Cell 7: Display figures inline
# ===========================================================================

from IPython.display import display, Image
import glob

figures = sorted(glob.glob("benchmark_results/figures/*.png"))
for fig_path in figures:
    print(f"\n--- {os.path.basename(fig_path)} ---")
    display(Image(filename=fig_path))

# ===========================================================================
# Cell 8: Package results for download
# ===========================================================================

# !zip -r /kaggle/working/benchmark_results.zip benchmark_results/
# print("Download benchmark_results.zip from the Output tab")
