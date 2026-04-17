# RL Energy Logger

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

`rl-energy-logger` is a lightweight Python library for monitoring energy consumption and hardware resource utilization (CPU, RAM, NVIDIA GPU) during deep reinforcement learning training loops. It aims to be plug-and-play with minimal overhead, addressing the gap in energy-aware tooling for RL research.

## Key Features

*   **Resource Tracking:** Monitors wall-clock time, CPU %, RAM %, NVIDIA GPU utilisation %, GPU memory %, GPU power (Watts), GPU temperature.
*   **Energy & CO₂ Estimation:** Built-in `EnergyEstimator` converts raw GPU power samples to cumulative **kWh** and **gCO₂eq** using regional carbon intensity data (20+ regions and cloud zones included).
*   **Low Overhead:** Designed for minimal performance impact with background-threaded sampling. Target <1% overhead.
*   **Integrations:**
    *   Native `Callback` for [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3).
    *   Generic Python `Context Manager` for use with PyTorch or any other framework.
*   **Flexible Output:** Logs data to CSV or JSON Lines (`.jsonl`) format.
*   **Easy Installation:** Install via pip with optional dependencies for GPU or specific framework support.

## Installation

**Basic (CPU-only monitoring):**
```bash
pip install rl-energy-logger
```

**With GPU support:**
```bash
pip install rl-energy-logger[gpu]
```

**With Stable-Baselines3 integration:**
```bash
pip install rl-energy-logger[sb3]
```

**All optional dependencies:**
```bash
pip install rl-energy-logger[gpu,sb3,torch]
```

**Development install (from source):**
```bash
git clone https://github.com/abidmohamed/rl-energy-logger.git
cd rl-energy-logger
pip install -e ".[dev]"
```

## Quick Start

### Context Manager (any training loop)

```python
from rl_energy_logger import EnergyLoggerContext

with EnergyLoggerContext(
    log_path="training_log.jsonl",
    sampling_interval_s=1.0,
    track_gpu=True,
) as ctx:
    for epoch in range(10):
        train_one_epoch()
        ctx.log(event="epoch_end", epoch=epoch, loss=compute_loss())
```

### Stable-Baselines3 Callback

```python
from stable_baselines3 import PPO
from rl_energy_logger import EnergyLogCallback

callback = EnergyLogCallback(
    log_path="sb3_energy.csv",
    sampling_interval_s=2.0,
    track_gpu=True,
    log_rl_metrics=True,
)

model = PPO("MlpPolicy", "CartPole-v1")
model.learn(total_timesteps=100_000, callback=callback)
```

### Post-hoc Energy & CO₂ Estimation

```python
from rl_energy_logger import EnergyEstimator

estimator = EnergyEstimator(region="france")  # Low-carbon grid
estimator.update(power_watts=150.0, duration_seconds=3600)  # 150W for 1h

print(estimator.summary())
# {'total_energy_kwh': 0.15, 'total_co2_grams': 8.25, ...}
```

## Available Regions

Use `rl_energy_logger.list_regions()` to see all supported region keys. Examples: `"world"`, `"us"`, `"france"`, `"china"`, `"gcp-us-central1"`, `"aws-eu-west-1"`.

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## Project Structure

```
rl-energy-logger/
├── rl_energy_logger/
│   ├── __init__.py          # Public API exports
│   ├── collector.py         # CPU/RAM/GPU metrics via psutil + pynvml
│   ├── energy.py            # kWh & CO₂eq estimation
│   ├── exceptions.py        # Custom exception hierarchy
│   ├── sb3_callback.py      # Stable-Baselines3 callback
│   ├── torch_wrapper.py     # Generic context manager
│   └── writers.py           # CSV and JSONL output writers
├── tests/                   # pytest test suite
├── examples/                # Usage examples
├── pyproject.toml           # Build config & dependencies
├── LICENSE                  # MIT License
├── CITATION.cff             # Citation metadata
└── README.md
```

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{abid2025rlenergy,
  author    = {Abid, Mohamed Nadhir},
  title     = {RL Energy Logger: Lightweight Energy and Resource Logging for Deep Reinforcement Learning},
  year      = {2025},
  url       = {https://github.com/abidmohamed/rl-energy-logger},
  license   = {MIT}
}
```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.