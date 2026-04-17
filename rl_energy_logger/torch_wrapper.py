import logging
import threading
import time
import warnings
from collections import deque

logger = logging.getLogger(__name__)
from typing import Dict, Any, Optional, Deque

from .collector import MetricsCollector
from .writers import BaseWriter, get_writer

class EnergyLoggerContext:
    """
    A context manager for logging hardware metrics during generic Python loops.

    Uses a background thread for periodic hardware sampling. Custom metrics
    can be logged manually via the `log()` method.

    :param log_path: Path to the output log file (.csv or .jsonl).
    :param sampling_interval_s: How often (in seconds) the background thread
                                samples hardware metrics.
    :param gpu_id: Index of the GPU to monitor.
    :param track_gpu: Whether to attempt GPU monitoring.
    :param queue_flush_interval_s: How often (in seconds) the background thread
                                   attempts to write buffered hardware samples
                                   to disk. Set to 0 or None to only flush on exit.
                                   Defaults to 10 seconds.
    :param buffer_size: Max number of hardware samples to buffer in memory before
                        forcing a flush (helps manage memory if disk I/O is slow).
                        Defaults to 500.
    """
    def __init__(
        self,
        log_path: str,
        sampling_interval_s: float = 1.0,
        gpu_id: int = 0,
        track_gpu: bool = True,
        queue_flush_interval_s: Optional[float] = 10.0,
        buffer_size: int = 500,
    ):
        if sampling_interval_s <= 0:
            raise ValueError("sampling_interval_s must be positive.")
        if buffer_size <= 0:
             raise ValueError("buffer_size must be positive.")

        self.log_path = log_path
        self.sampling_interval_s = sampling_interval_s
        self.queue_flush_interval_s = queue_flush_interval_s if queue_flush_interval_s and queue_flush_interval_s > 0 else None
        self.buffer_size = buffer_size

        self.collector = MetricsCollector(gpu_id=gpu_id, track_gpu=track_gpu)
        self.writer: Optional[BaseWriter] = None # Initialized in __enter__

        self._metrics_queue: Deque[Dict[str, Any]] = deque()
        self._queue_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._background_thread: Optional[threading.Thread] = None
        self._last_flush_time: float = 0.0

        logger.info("EnergyLoggerContext initialized. Logging to '%s'. Hardware poll interval: %.1fs.", log_path, sampling_interval_s)

    def _background_sampler(self):
        """Target function for the background sampling thread."""
        self._last_flush_time = time.time()

        while not self._stop_event.wait(self.sampling_interval_s):
            try:
                hw_metrics = self.collector.sample()
                # Add an identifier for background samples vs manual logs
                hw_metrics["log_type"] = "background_hw_sample"

                with self._queue_lock:
                    self._metrics_queue.append(hw_metrics)
                    queue_size = len(self._metrics_queue)

                # --- Periodic Flush Logic ---
                now = time.time()
                should_flush = False
                # Flush if buffer size exceeded
                if queue_size >= self.buffer_size:
                    should_flush = True
                    if self.queue_flush_interval_s: # Prevent spamming flushes if buffer fills fast
                        time.sleep(0.1) # Small sleep if flushing due to size

                # Flush if flush interval passed (and interval is set)
                elif self.queue_flush_interval_s and (now - self._last_flush_time >= self.queue_flush_interval_s):
                    should_flush = True

                if should_flush and queue_size > 0:
                    self._flush_queue() # Flush the current buffer
                    self._last_flush_time = time.time() # Reset timer after flush

            except Exception as e:
                # Catch errors within the thread to prevent it from crashing silently
                warnings.warn(f"[EnergyLoggerContext Background Thread] Error during sampling or flushing: {e}", RuntimeWarning)
                # Optional: Add logic to stop thread after repeated failures?
                time.sleep(self.sampling_interval_s * 2) # Wait longer after an error

        # --- Final Flush After Stop Signal ---
        # print("[EnergyLoggerContext Background Thread] Stop signal received. Flushing final data.")
        self._flush_queue()

    def _flush_queue(self):
        """Flushes metrics from the internal queue to the writer."""
        if self.writer is None:
             warnings.warn("[EnergyLoggerContext] Cannot flush queue, writer is not initialized.", RuntimeWarning)
             return

        entries_to_write = []
        with self._queue_lock:
            while self._metrics_queue:
                entries_to_write.append(self._metrics_queue.popleft())

        if entries_to_write:
            # print(f"[EnergyLoggerContext] Flushing {len(entries_to_write)} entries from queue.") # Verbose log
            try:
                # Assuming writer handles writing multiple entries efficiently if needed,
                # but the base writer expects one dict at a time. Iterate.
                for entry in entries_to_write:
                    self.writer.write(entry)
            except Exception as e:
                warnings.warn(f"[EnergyLoggerContext] Error during queue flush write: {e}", RuntimeWarning)
                # Re-add failed entries back to queue? Could cause loops. Best effort for now.


    def log(self, event: str = "manual_log", **kwargs):
        """
        Manually log custom metrics along with the latest hardware snapshot.

        This bypasses the queue for immediate writing (or near-immediate).

        Args:
            event (str): A string identifier for the type of event being logged
                         (e.g., 'epoch_end', 'validation_score').
            **kwargs: Custom key-value pairs to include in the log record.
        """
        if self.writer is None:
            warnings.warn("[EnergyLoggerContext] Cannot log event, writer is not initialized.", RuntimeWarning)
            return

        try:
            # Get a fresh hardware sample at the time of the manual log event
            hw_metrics = self.collector.sample()

            # Combine with user metrics and event type
            log_entry = {
                **hw_metrics,
                "log_type": event, # Use event name as log type
                **kwargs
            }

            # Write immediately (bypasses background queue)
            self.writer.write(log_entry)

        except Exception as e:
             warnings.warn(f"[EnergyLoggerContext] Error during manual log: {e}", RuntimeWarning)


    def __enter__(self):
        """Starts the background sampler thread and opens the writer."""
        try:
            self.writer = get_writer(self.log_path)
        except Exception as e:
             warnings.warn(f"[EnergyLoggerContext] Failed to initialize writer: {e}. Context disabled.", RuntimeWarning)
             self.writer = None # Disable context if writer fails
             return self # Still return self, but log/flush won't work

        self._stop_event.clear()
        self._background_thread = threading.Thread(target=self._background_sampler, daemon=True)
        self._background_thread.start()
        logger.info("EnergyLoggerContext: Background sampling thread started.")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stops the background thread, flushes remaining data, and closes resources."""
        logger.info("EnergyLoggerContext: Exiting context...")
        if self._background_thread and self._background_thread.is_alive():
            self._stop_event.set()
            logger.info("EnergyLoggerContext: Waiting for background thread to finish...")
            self._background_thread.join(timeout=self.sampling_interval_s * 3) # Wait reasonable time
            if self._background_thread.is_alive():
                 warnings.warn("[EnergyLoggerContext] Background thread did not exit cleanly.", RuntimeWarning)

        # Final flush is handled by the thread itself upon seeing stop_event.
        # Ensure queue is empty after thread join (add final check/flush maybe?)
        # print("[EnergyLoggerContext] Final check/flush of queue...")
        # self._flush_queue() # One last check

        if self.writer:
            self.writer.close()
        self.collector.close() # Shutdown NVML etc.
        logger.info("EnergyLoggerContext: Resources closed.")