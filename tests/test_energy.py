"""Tests for rl_energy_logger.energy (EnergyEstimator, watts_to_kwh, kwh_to_co2)."""

import pytest

from rl_energy_logger.energy import (
    EnergyEstimator,
    watts_to_kwh,
    kwh_to_co2,
    list_regions,
    CARBON_INTENSITY_GRAMS_PER_KWH,
)


# ---------------------------------------------------------------------------
# watts_to_kwh
# ---------------------------------------------------------------------------

class TestWattsToKwh:
    def test_one_kilowatt_one_hour(self):
        assert watts_to_kwh(1000.0, 3600.0) == pytest.approx(1.0)

    def test_100w_for_one_hour(self):
        assert watts_to_kwh(100.0, 3600.0) == pytest.approx(0.1)

    def test_250w_for_30_minutes(self):
        assert watts_to_kwh(250.0, 1800.0) == pytest.approx(0.125)

    def test_zero_power(self):
        assert watts_to_kwh(0.0, 3600.0) == pytest.approx(0.0)

    def test_zero_duration(self):
        assert watts_to_kwh(150.0, 0.0) == pytest.approx(0.0)

    def test_negative_power_returns_zero(self):
        assert watts_to_kwh(-50.0, 3600.0) == 0.0

    def test_negative_duration_returns_zero(self):
        assert watts_to_kwh(100.0, -10.0) == 0.0


# ---------------------------------------------------------------------------
# kwh_to_co2
# ---------------------------------------------------------------------------

class TestKwhToCo2:
    def test_world_default(self):
        co2 = kwh_to_co2(1.0, region="world")
        assert co2 == pytest.approx(475.0)

    def test_france_low_carbon(self):
        co2 = kwh_to_co2(1.0, region="france")
        assert co2 == pytest.approx(55.0)

    def test_custom_intensity_overrides_region(self):
        co2 = kwh_to_co2(1.0, region="france", carbon_intensity_grams_per_kwh=999.0)
        assert co2 == pytest.approx(999.0)

    def test_unknown_region_falls_back_to_world(self):
        co2 = kwh_to_co2(1.0, region="narnia")
        assert co2 == pytest.approx(CARBON_INTENSITY_GRAMS_PER_KWH["world"])

    def test_case_insensitive_region(self):
        co2 = kwh_to_co2(1.0, region="FRANCE")
        assert co2 == pytest.approx(55.0)

    def test_zero_energy(self):
        assert kwh_to_co2(0.0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# list_regions
# ---------------------------------------------------------------------------

class TestListRegions:
    def test_returns_sorted_list(self):
        regions = list_regions()
        assert isinstance(regions, list)
        assert regions == sorted(regions)

    def test_contains_world(self):
        assert "world" in list_regions()

    def test_minimum_region_count(self):
        assert len(list_regions()) >= 15


# ---------------------------------------------------------------------------
# EnergyEstimator
# ---------------------------------------------------------------------------

class TestEnergyEstimator:
    def test_initial_state_is_zero(self):
        est = EnergyEstimator()
        assert est.total_energy_kwh == 0.0
        assert est.total_co2_grams == 0.0
        assert est.total_duration_hours == 0.0
        assert est._sample_count == 0

    def test_single_update(self):
        est = EnergyEstimator(region="world")
        est.update(power_watts=200.0, duration_seconds=3600.0)  # 0.2 kWh
        assert est.total_energy_kwh == pytest.approx(0.2)
        assert est.total_co2_grams == pytest.approx(0.2 * 475.0)
        assert est._sample_count == 1

    def test_multiple_updates_accumulate(self):
        est = EnergyEstimator()
        est.update(100.0, 3600.0)  # 0.1 kWh
        est.update(100.0, 3600.0)  # 0.1 kWh
        assert est.total_energy_kwh == pytest.approx(0.2)
        assert est._sample_count == 2

    def test_none_power_is_skipped(self):
        est = EnergyEstimator()
        est.update(None, 1.0)
        assert est._sample_count == 0

    def test_none_duration_is_skipped(self):
        est = EnergyEstimator()
        est.update(100.0, None)
        assert est._sample_count == 0

    def test_negative_power_is_skipped(self):
        est = EnergyEstimator()
        est.update(-50.0, 10.0)
        assert est._sample_count == 0

    def test_zero_duration_is_skipped(self):
        est = EnergyEstimator()
        est.update(100.0, 0.0)
        assert est._sample_count == 0

    def test_average_power(self):
        est = EnergyEstimator()
        # 200W for 1 hour = 0.2 kWh
        est.update(200.0, 3600.0)
        assert est.average_power_watts == pytest.approx(200.0)

    def test_average_power_zero_when_no_samples(self):
        est = EnergyEstimator()
        assert est.average_power_watts == 0.0

    def test_summary_returns_dict(self):
        est = EnergyEstimator(region="france")
        est.update(150.0, 1800.0)
        summary = est.summary()
        assert "total_energy_kwh" in summary
        assert "total_co2_grams" in summary
        assert "total_duration_hours" in summary
        assert "average_power_watts" in summary
        assert "sample_count" in summary
        assert summary["region"] == "france"
        assert summary["sample_count"] == 1

    def test_reset_clears_state(self):
        est = EnergyEstimator()
        est.update(100.0, 3600.0)
        est.reset()
        assert est.total_energy_kwh == 0.0
        assert est._sample_count == 0

    def test_custom_carbon_intensity(self):
        est = EnergyEstimator(carbon_intensity_grams_per_kwh=100.0)
        est.update(1000.0, 3600.0)  # 1 kWh
        assert est.total_co2_grams == pytest.approx(100.0)

    def test_repr_string(self):
        est = EnergyEstimator(region="uk")
        est.update(100.0, 3600.0)
        r = repr(est)
        assert "EnergyEstimator" in r
        assert "kWh" in r
        assert "uk" in r
