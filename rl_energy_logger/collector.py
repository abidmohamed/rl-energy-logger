import logging
import os
import time
import warnings

import psutil

logger = logging.getLogger(__name__)
from typing import Optional, Dict, Any

from .exceptions import NVMLDriverError, CollectorError

try:
    import pynvml  # provided by nvidia-ml-py (preferred) or legacy pynvml
    _NVML_FOUND = True
except ImportError:
    _NVML_FOUND = False
    pynvml = None  # Define pynvml as None if not found

class MetricsCollector:
    """
    Collects hardware metrics like CPU, RAM, and optionally NVIDIA GPU stats.

    Manages the lifecycle of the NVML library if used.
    """

    def __init__(self, gpu_id: int = 0, track_gpu: bool = True):
        """
        Initializes the MetricsCollector.

        Args:
            gpu_id (int): The index of the NVIDIA GPU to monitor (default: 0).
            track_gpu (bool): Whether to attempt GPU monitoring. Set to False
                              to disable even if pynvml is installed.
        """
        self._process = psutil.Process(os.getpid())
        self.gpu_id = gpu_id
        self._track_gpu = track_gpu
        self._nvml_initialized = False
        self._gpu_handle = None
        self._gpu_name = None

        if self._track_gpu:
            if not _NVML_FOUND:
                warnings.warn(
                    "pynvml library not found. GPU metrics will not be collected. "
                    "Install with: pip install rl-energy-logger[gpu]",
                    ImportWarning
                )
                self._track_gpu = False # Disable GPU tracking if lib missing
            else:
                self._initialize_nvml()

    def _initialize_nvml(self):
        """Initializes the NVML library and gets the GPU handle."""
        if not pynvml: # Should not happen if _NVML_FOUND is True, but defensive check
             self._track_gpu = False
             return
        try:
            pynvml.nvmlInit()
            self._nvml_initialized = True
            num_devices = pynvml.nvmlDeviceGetCount()
            if self.gpu_id >= num_devices:
                warnings.warn(
                    f"GPU ID {self.gpu_id} is invalid. Found {num_devices} devices. "
                    f"Disabling GPU tracking for this collector.",
                    RuntimeWarning
                )
                self._track_gpu = False
                self.shutdown_nvml() # Shutdown if we can't get the handle
                return

            self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_id)
            self._gpu_name = pynvml.nvmlDeviceGetName(self._gpu_handle).decode('utf-8')
            logger.info("Initialized NVML. Monitoring GPU %d: %s", self.gpu_id, self._gpu_name)

        except pynvml.NVMLError as e:
            self._nvml_initialized = False
            self._track_gpu = False
            # Try to shutdown cleanly even if init failed partially
            if pynvml and hasattr(pynvml, 'nvmlShutdown'):
                 try:
                      pynvml.nvmlShutdown()
                 except pynvml.NVMLError:
                      pass # Ignore shutdown error if init failed badly
            raise NVMLDriverError(
                f"Could not initialize NVML or get handle for GPU {self.gpu_id}. "
                f"Is the NVIDIA driver installed and running? Original error: {e}"
            ) from e

    def sample(self) -> Dict[str, Any]:
        """
        Samples current hardware metrics.

        Returns:
            Dict[str, Any]: A dictionary containing metrics like:
                'timestamp': float (time.time())
                'wall_time_s': float (time since process start)
                'cpu_percent': float (% CPU usage of the current process)
                'ram_percent': float (% RAM usage of the system)
                'ram_rss_mb': float (Resident Set Size memory in MB for the process)
                'gpu_id': int (if tracked)
                'gpu_util_percent': float (% GPU core utilization, if tracked)
                'gpu_mem_util_percent': float (% GPU memory controller utilization, if tracked)
                'gpu_mem_used_mb': float (GPU memory used in MB, if tracked)
                'gpu_power_watts': float (GPU power draw in Watts, if tracked)
                'gpu_temp_c': int (GPU temperature in Celsius, if tracked)
        """
        metrics = {}
        current_time = time.time()
        metrics['timestamp'] = current_time

        # --- CPU/RAM Metrics (using process info) ---
        try:
            with self._process.oneshot(): # Efficiently get multiple stats
                metrics['wall_time_s'] = current_time - self._process.create_time()
                metrics['cpu_percent'] = self._process.cpu_percent()
                # Use system-wide RAM % as process % can be misleading without context
                metrics['ram_percent'] = psutil.virtual_memory().percent
                metrics['ram_rss_mb'] = self._process.memory_info().rss / (1024 * 1024)
        except psutil.Error as e:
            warnings.warn(f"Could not read CPU/RAM metrics: {e}", RuntimeWarning)
            metrics.update({'cpu_percent': None, 'ram_percent': None, 'ram_rss_mb': None})

        # --- GPU Metrics ---
        if self._track_gpu and self._nvml_initialized and self._gpu_handle and pynvml:
            metrics['gpu_id'] = self.gpu_id
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self._gpu_handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
                power_milliwatts = pynvml.nvmlDeviceGetPowerUsage(self._gpu_handle)
                temp = pynvml.nvmlDeviceGetTemperature(self._gpu_handle, pynvml.NVML_TEMPERATURE_GPU)

                metrics['gpu_util_percent'] = float(util.gpu)
                metrics['gpu_mem_util_percent'] = float(util.memory) # Memory controller util
                metrics['gpu_mem_used_mb'] = mem.used / (1024 * 1024)
                metrics['gpu_power_watts'] = power_milliwatts / 1000.0
                metrics['gpu_temp_c'] = int(temp)
            except pynvml.NVMLError as e:
                # Handle specific error if device polling is lost mid-run
                if hasattr(pynvml, 'NVML_ERROR_GPU_IS_LOST') and e.value == pynvml.NVML_ERROR_GPU_IS_LOST:
                     warnings.warn(f"GPU {self.gpu_id} is lost. Stopping GPU monitoring. Error: {e}", RuntimeWarning)
                     self._track_gpu = False # Stop trying to track this GPU
                     self.shutdown_nvml() # Attempt clean shutdown
                else:
                     warnings.warn(f"Could not read GPU metrics for GPU {self.gpu_id}: {e}", RuntimeWarning)
                # Set GPU keys to None if an error occurred
                metrics.update({
                    'gpu_util_percent': None, 'gpu_mem_util_percent': None,
                    'gpu_mem_used_mb': None, 'gpu_power_watts': None, 'gpu_temp_c': None
                })
            except Exception as e: # Catch other potential errors
                 warnings.warn(f"Unexpected error reading GPU metrics for GPU {self.gpu_id}: {e}", RuntimeWarning)
                 metrics.update({
                    'gpu_util_percent': None, 'gpu_mem_util_percent': None,
                    'gpu_mem_used_mb': None, 'gpu_power_watts': None, 'gpu_temp_c': None
                 })
        elif self._track_gpu:
            # If tracking was intended but failed init or handle is missing
            metrics['gpu_id'] = self.gpu_id
            metrics.update({
                'gpu_util_percent': None, 'gpu_mem_util_percent': None,
                'gpu_mem_used_mb': None, 'gpu_power_watts': None, 'gpu_temp_c': None
            })


        return metrics

    def shutdown_nvml(self):
        """Shuts down the NVML library if it was initialized."""
        if self._nvml_initialized and pynvml:
            try:
                pynvml.nvmlShutdown()
                self._nvml_initialized = False
                self._gpu_handle = None
                # print(f"[rl-energy-logger] NVML shut down for GPU {self.gpu_id}.") # Optional verbose log
            except pynvml.NVMLError as e:
                # Log warning but don't raise - shutdown failure is less critical
                warnings.warn(f"Error shutting down NVML: {e}", RuntimeWarning)

    def close(self):
        """Alias for shutdown_nvml for consistent interface."""
        self.shutdown_nvml()

    def __del__(self):
        """Ensure NVML is shut down when the collector object is garbage collected."""
        self.shutdown_nvml()