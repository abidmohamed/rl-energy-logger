"""
Custom exceptions for the rl-energy-logger library.
"""


class RLEnergyLoggerError(Exception):
    """Base exception for all rl-energy-logger errors."""
    pass


class NVMLDriverError(RLEnergyLoggerError):
    """
    Raised when the NVIDIA Management Library (NVML) cannot be initialized
    or a GPU handle cannot be obtained.

    This typically indicates that the NVIDIA driver is not installed,
    the driver version is incompatible, or the specified GPU ID is invalid.
    """
    pass


class CollectorError(RLEnergyLoggerError):
    """
    Raised when the MetricsCollector encounters an unrecoverable error
    during hardware metric sampling.
    """
    pass


class WriterError(RLEnergyLoggerError):
    """
    Raised when a log writer (CSV, JSONL) fails to open, write to,
    or close its output file.
    """
    pass
