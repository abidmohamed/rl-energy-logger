"""
Kaggle Notebook Script — RL Energy Benchmark
=============================================

Copy-paste each cell block into separate Kaggle notebook cells.
Enable GPU T4 accelerator before running.
"""

# ===========================================================================
# Cell 1: Setup — install dependencies and clone repo
# ===========================================================================

# !apt-get install -y swig > /dev/null 2>&1
# !pip install -q "stable-baselines3[extra]>=2.0.0" "gymnasium[box2d]" codecarbon matplotlib
# !git clone https://github.com/abidmohamed/rl-energy-logger.git /kaggle/working/rl-energy-logger
# !pip install -q -e /kaggle/working/rl-energy-logger

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

sys.path.insert(0, "/kaggle/working/rl-energy-logger")

from benchmarks.run_benchmark import main as run_benchmark_main

# Full run (or uncomment --dry-run for quick test)
sys.argv = ["run_benchmark.py"]
# sys.argv = ["run_benchmark.py", "--dry-run"]  # Uncomment for quick test
run_benchmark_main()

# ===========================================================================
# Cell 4: Run overhead benchmark
# ===========================================================================

from benchmarks.run_overhead import main as run_overhead_main

sys.argv = ["run_overhead.py"]
# sys.argv = ["run_overhead.py", "--dry-run"]
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
