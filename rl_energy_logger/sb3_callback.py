import logging
import time
import warnings
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Check if stable-baselines3 is available
try:
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import Logger as SB3Logger # To access SB3 metrics
    _SB3_FOUND = True
except ImportError:
    _SB3_FOUND = False
    # Define dummy classes if SB3 is not installed, so the file can be imported
    # but the callback cannot be used.
    class BaseCallback:
        def __init__(self, verbose: int = 0):
            self.verbose = verbose
            self.logger = None # SB3 logger instance, set by parent
            self.num_timesteps = 0

        def _on_step(self) -> bool:
             return True # Default behavior
        def _on_training_start(self) -> None:
             pass
        def _on_training_end(self) -> None:
             pass

    class SB3Logger:
         def __init__(self):
             self.name_to_value = {}
             self.name_to_count = {}
             self.name_to_excluded = {}

from .collector import MetricsCollector
from .writers import BaseWriter, get_writer

# Check if SB3 is actually installed before defining the real callback
if _SB3_FOUND:
    class EnergyLogCallback(BaseCallback):
        """
        Stable-Baselines3 callback to log hardware metrics and optionally RL metrics.

        Logs are written periodically based on wall-clock time.

        :param log_path: Path to the output log file (.csv or .jsonl).
        :param sampling_interval_s: How often (in seconds) to sample hardware metrics.
        :param gpu_id: Index of the GPU to monitor.
        :param track_gpu: Whether to attempt GPU monitoring.
        :param log_rl_metrics: If True, attempts to log metrics recorded by SB3's logger
                               (e.g., rollout/ep_rew_mean, train/loss).
        :param verbose: Verbosity level (0 or 1).
        """
        def __init__(
            self,
            log_path: str,
            sampling_interval_s: float = 2.0,
            gpu_id: int = 0,
            track_gpu: bool = True,
            log_rl_metrics: bool = True,
            verbose: int = 0,
        ):
            super().__init__(verbose)
            if sampling_interval_s <= 0:
                raise ValueError("sampling_interval_s must be positive.")

            self.log_path = log_path
            self.sampling_interval_s = sampling_interval_s
            self.log_rl_metrics = log_rl_metrics

            self.collector = MetricsCollector(gpu_id=gpu_id, track_gpu=track_gpu)
            self.writer: Optional[BaseWriter] = None # Initialized in _on_training_start
            self._last_log_time: float = 0.0

            if verbose > 0:
                logger.info("EnergyLogCallback initialized. Logging to '%s' every ~%.1fs.", log_path, sampling_interval_s)

        def _init_callback(self) -> None:
            """Initializes the writer."""
            # This is called before _on_training_start by SB3 BaseCallback
            try:
                self.writer = get_writer(self.log_path)
                if self.verbose > 0:
                     logger.info("EnergyLogCallback writer initialized for %s", self.log_path)
            except Exception as e:
                warnings.warn(f"[EnergyLogCallback] Failed to initialize writer: {e}. Callback disabled.", RuntimeWarning)
                self.writer = None # Disable writing if init fails

        def _on_training_start(self) -> None:
            """Called once before the training loop starts."""
            self._last_log_time = time.time()
            # Log initial state? Maybe not necessary, wait for first interval.

        def _on_step(self) -> bool:
            """
            Called after each environment step.
            Checks if it's time to log based on the interval.
            """
            if self.writer is None:
                return True # Do nothing if writer failed to initialize

            current_time = time.time()
            if current_time - self._last_log_time >= self.sampling_interval_s:
                self._last_log_time = current_time

                # Collect hardware metrics
                hw_metrics = self.collector.sample()

                # Collect RL metrics if enabled and logger is available
                rl_metrics = {}
                if self.log_rl_metrics and self.logger is not None:
                    # Access the internal key-value store of the SB3 logger
                    # We typically want the latest mean/median values
                    for key, (value, _) in self.logger.name_to_value.items():
                         # Exclude SB3 internal time metrics if desired, as we have our own
                         if key.startswith("time/"):
                             continue
                         # Add prefix to avoid potential name collisions
                         rl_metrics[f"rl_{key.replace('/', '_')}"] = value

                # Combine metrics
                log_entry = {
                    **hw_metrics,
                    **rl_metrics,
                    "sb3_num_timesteps": self.num_timesteps # Add SB3 step count
                }

                # Write the combined record
                try:
                    self.writer.write(log_entry)
                except Exception as e:
                     warnings.warn(f"[EnergyLogCallback] Error writing log entry: {e}", RuntimeWarning)
                     # Optionally disable writer on repeated errors? For now, just warn.

            return True # Continue training

        def _on_training_end(self) -> None:
            """Called once after the training loop ends."""
            if self.verbose > 0:
                  logger.info("EnergyLogCallback: Training ended.")
            # Ensure final data is flushed and resources are released
            if self.writer:
                self.writer.close()
            self.collector.close() # Shuts down NVML if needed
else:
    # If SB3 is not installed, provide a placeholder class that raises an error on instantiation.
    class EnergyLogCallback:
         def __init__(self, *args, **kwargs):
              raise ImportError(
                  "Stable-Baselines3 is not installed. Cannot use EnergyLogCallback. "
                  "Install with: pip install stable-baselines3 rl-energy-logger[sb3]"
              )