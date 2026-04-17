"""
Kaggle Notebook Script — RL Energy Benchmark
=============================================

Copy-paste this into a Kaggle notebook cell (with GPU T4 accelerator enabled)
to run the full benchmark.

Steps:
1. Upload the rl-energy-logger repo as a Kaggle dataset
2. Create a new notebook, enable GPU T4 accelerator
3. Paste this script into a cell and run
4. Download benchmark_results/ when complete
"""

# ===========================================================================
# Cell 1: Setup — install dependencies
# ===========================================================================

# !pip install -q stable-baselines3[extra] gymnasium[box2d] codecarbon matplotlib

# If uploaded as dataset (adjust path as needed):
# import sys
# sys.path.insert(0, "/kaggle/input/rl-energy-logger")
# !pip install -e /kaggle/input/rl-energy-logger

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

# Make sure we can import benchmarks module
sys.path.insert(0, os.path.dirname(os.path.abspath(".")))

from benchmarks.run_benchmark import main as run_benchmark_main
from benchmarks.config import get_full_matrix

matrix = get_full_matrix()
print(f"Running {len(matrix)} experiments...")

# Run with sys.argv override
sys.argv = ["run_benchmark.py"]  # Full run
# sys.argv = ["run_benchmark.py", "--dry-run"]  # Uncomment for quick test
run_benchmark_main()

# ===========================================================================
# Cell 4: Run overhead benchmark
# ===========================================================================

from benchmarks.run_overhead import main as run_overhead_main

sys.argv = ["run_overhead.py"]
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

print("\n✅ All done! Download benchmark_results/ for the data and figures.")

# ===========================================================================
# Cell 7: Display figures inline (optional)
# ===========================================================================

from IPython.display import display, Image
import glob

figures = sorted(glob.glob("benchmark_results/figures/*.png"))
for fig_path in figures:
    print(f"\n--- {os.path.basename(fig_path)} ---")
    display(Image(filename=fig_path))
