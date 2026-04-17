"""
Energy and carbon emission estimation utilities.

Provides functions to convert raw power measurements (Watts) into
cumulative energy (kWh) and approximate CO₂-equivalent emissions (gCO₂eq).

Carbon intensity data is based on publicly available grid averages per region.
Users can supply a custom carbon intensity value for more accurate estimates.
"""

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default regional carbon intensity factors (gCO₂/kWh)
# Sources:
#   - IEA (2023) – World Energy Outlook
#   - Electricity Maps (2024) – https://app.electricitymaps.com
#   - US EPA eGRID (2022)
# These are *approximate annual averages*. Real-time values may differ
# significantly depending on time-of-day and renewable mix.
# ---------------------------------------------------------------------------
CARBON_INTENSITY_GRAMS_PER_KWH: Dict[str, float] = {
    # --- Global / Default ---
    "world":            475.0,

    # --- North America ---
    "us":               390.0,
    "us-west":          250.0,    # WECC (California, Oregon, Washington)
    "us-east":          350.0,    # SERC / RFC average
    "us-midwest":       450.0,    # MROW
    "canada":           120.0,    # Heavy hydro

    # --- Europe ---
    "eu":               230.0,    # EU-27 average
    "france":            55.0,    # Nuclear-dominant
    "germany":          350.0,
    "uk":               200.0,
    "sweden":            30.0,    # Hydro + nuclear
    "norway":            20.0,
    "poland":           650.0,    # Coal-heavy
    "algeria":          480.0,    # Fossil-dominant North Africa

    # --- Asia ---
    "china":            530.0,
    "india":            700.0,
    "japan":            450.0,
    "south-korea":      420.0,

    # --- Oceania ---
    "australia":        550.0,

    # --- Cloud regions (approximate) ---
    "gcp-us-central1":  450.0,
    "gcp-europe-west1": 120.0,
    "aws-us-east-1":    350.0,
    "aws-eu-west-1":    300.0,
    "azure-eastus":     350.0,
}


def watts_to_kwh(power_watts: float, duration_seconds: float) -> float:
    """
    Convert power draw over a duration to energy in kilowatt-hours.

    Args:
        power_watts: Average power draw in Watts.
        duration_seconds: Duration of the measurement window in seconds.

    Returns:
        Energy consumed in kWh.
    """
    if power_watts < 0 or duration_seconds < 0:
        return 0.0
    hours = duration_seconds / 3600.0
    return (power_watts / 1000.0) * hours


def kwh_to_co2(
    energy_kwh: float,
    region: str = "world",
    carbon_intensity_grams_per_kwh: Optional[float] = None,
) -> float:
    """
    Estimate CO₂-equivalent emissions from energy consumption.

    Args:
        energy_kwh: Energy consumed in kWh.
        region: A region key from ``CARBON_INTENSITY_GRAMS_PER_KWH``.
                Ignored if ``carbon_intensity_grams_per_kwh`` is provided.
        carbon_intensity_grams_per_kwh: Custom carbon intensity override
                                         (gCO₂/kWh). Takes precedence over
                                         ``region``.

    Returns:
        Estimated emissions in grams of CO₂-equivalent.
    """
    if carbon_intensity_grams_per_kwh is not None:
        ci = carbon_intensity_grams_per_kwh
    else:
        region_lower = region.lower()
        if region_lower not in CARBON_INTENSITY_GRAMS_PER_KWH:
            logger.warning(
                "Unknown region '%s'. Falling back to world average (%.0f gCO₂/kWh).",
                region,
                CARBON_INTENSITY_GRAMS_PER_KWH["world"],
            )
            ci = CARBON_INTENSITY_GRAMS_PER_KWH["world"]
        else:
            ci = CARBON_INTENSITY_GRAMS_PER_KWH[region_lower]

    return energy_kwh * ci


def list_regions() -> List[str]:
    """Return a sorted list of all available region keys."""
    return sorted(CARBON_INTENSITY_GRAMS_PER_KWH.keys())


class EnergyEstimator:
    """
    Accumulates power samples over time and provides running totals
    of energy (kWh) and CO₂ (gCO₂eq).

    Designed to be fed with periodic ``(power_watts, duration_seconds)``
    measurements — typically derived from consecutive ``MetricsCollector``
    samples.

    Example::

        estimator = EnergyEstimator(region="france")
        for sample in samples:
            estimator.update(sample["gpu_power_watts"], interval_s)
        print(estimator.summary())
    """

    def __init__(
        self,
        region: str = "world",
        carbon_intensity_grams_per_kwh: Optional[float] = None,
    ):
        self.region = region
        self._custom_ci = carbon_intensity_grams_per_kwh
        self._total_energy_kwh: float = 0.0
        self._total_duration_s: float = 0.0
        self._sample_count: int = 0

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def update(self, power_watts: float, duration_seconds: float) -> None:
        """
        Record a power sample.

        Args:
            power_watts: Average power draw (Watts) over the interval.
                         ``None`` values are silently skipped.
            duration_seconds: Length of the measurement interval in seconds.
        """
        if power_watts is None or duration_seconds is None:
            return
        if power_watts < 0 or duration_seconds <= 0:
            return
        self._total_energy_kwh += watts_to_kwh(power_watts, duration_seconds)
        self._total_duration_s += duration_seconds
        self._sample_count += 1

    @property
    def total_energy_kwh(self) -> float:
        """Cumulative energy consumed (kWh)."""
        return self._total_energy_kwh

    @property
    def total_co2_grams(self) -> float:
        """Cumulative estimated CO₂eq emissions (grams)."""
        return kwh_to_co2(
            self._total_energy_kwh,
            region=self.region,
            carbon_intensity_grams_per_kwh=self._custom_ci,
        )

    @property
    def total_duration_hours(self) -> float:
        """Total tracked duration (hours)."""
        return self._total_duration_s / 3600.0

    @property
    def average_power_watts(self) -> float:
        """Average power draw over all samples (Watts)."""
        if self._total_duration_s == 0:
            return 0.0
        return (self._total_energy_kwh * 1000.0) / self.total_duration_hours

    def summary(self) -> Dict[str, float]:
        """
        Return a dict summarising the accumulated energy stats.

        Returns:
            Dictionary with keys ``total_energy_kwh``, ``total_co2_grams``,
            ``total_duration_hours``, ``average_power_watts``,
            ``sample_count``.
        """
        return {
            "total_energy_kwh": round(self.total_energy_kwh, 6),
            "total_co2_grams": round(self.total_co2_grams, 4),
            "total_duration_hours": round(self.total_duration_hours, 4),
            "average_power_watts": round(self.average_power_watts, 2),
            "sample_count": self._sample_count,
            "region": self.region,
        }

    def reset(self) -> None:
        """Reset all accumulated counters."""
        self._total_energy_kwh = 0.0
        self._total_duration_s = 0.0
        self._sample_count = 0

    def __repr__(self) -> str:
        return (
            f"EnergyEstimator(energy={self.total_energy_kwh:.6f} kWh, "
            f"co2={self.total_co2_grams:.2f} gCO₂eq, "
            f"duration={self.total_duration_hours:.4f} h, "
            f"region='{self.region}')"
        )
