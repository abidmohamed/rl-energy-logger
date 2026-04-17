__version__ = "0.1.0"

import warnings
from .exceptions import RLEnergyLoggerError, NVMLDriverError, CollectorError, WriterError
from .collector import MetricsCollector
from .writers import BaseWriter, CSVWriter, JSONLWriter, get_writer
from .energy import (
    EnergyEstimator,
    watts_to_kwh,
    kwh_to_co2,
    list_regions,
    CARBON_INTENSITY_GRAMS_PER_KWH,
)

# Conditionally import framework-specific parts
try:
    from .sb3_callback import EnergyLogCallback
    _SB3_INSTALLED = True
except ImportError:
    EnergyLogCallback = None  # type: ignore
    _SB3_INSTALLED = False

try:
    from .torch_wrapper import EnergyLoggerContext
    _TORCH_WRAPPER_AVAILABLE = True
except ImportError:
    EnergyLoggerContext = None  # type: ignore
    _TORCH_WRAPPER_AVAILABLE = False


# Public API definition
__all__ = [
    # Core
    "MetricsCollector",
    "BaseWriter",
    "CSVWriter",
    "JSONLWriter",
    "get_writer",
    # Energy estimation
    "EnergyEstimator",
    "watts_to_kwh",
    "kwh_to_co2",
    "list_regions",
    "CARBON_INTENSITY_GRAMS_PER_KWH",
    # Exceptions
    "RLEnergyLoggerError",
    "NVMLDriverError",
    "CollectorError",
    "WriterError",
    # Conditionally expose based on imports
    "EnergyLogCallback" if _SB3_INSTALLED else None,  # type: ignore
    "EnergyLoggerContext" if _TORCH_WRAPPER_AVAILABLE else None,  # type: ignore
]

# Filter out None values from __all__ that result from conditional imports
__all__ = [name for name in __all__ if name is not None]