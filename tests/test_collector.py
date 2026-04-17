"""Tests for rl_energy_logger.collector.MetricsCollector."""

import time
from unittest.mock import patch, MagicMock

import pytest

from rl_energy_logger.collector import MetricsCollector


class TestMetricsCollectorCPUOnly:
    """Tests for the collector with GPU tracking disabled."""

    def test_sample_returns_dict(self):
        collector = MetricsCollector(track_gpu=False)
        try:
            sample = collector.sample()
            assert isinstance(sample, dict)
        finally:
            collector.close()

    def test_sample_contains_cpu_keys(self):
        collector = MetricsCollector(track_gpu=False)
        try:
            sample = collector.sample()
            assert "timestamp" in sample
            assert "wall_time_s" in sample
            assert "cpu_percent" in sample
            assert "ram_percent" in sample
            assert "ram_rss_mb" in sample
        finally:
            collector.close()

    def test_sample_does_not_contain_gpu_keys_when_disabled(self):
        collector = MetricsCollector(track_gpu=False)
        try:
            sample = collector.sample()
            assert "gpu_util_percent" not in sample
            assert "gpu_power_watts" not in sample
        finally:
            collector.close()

    def test_timestamp_is_recent(self):
        collector = MetricsCollector(track_gpu=False)
        try:
            before = time.time()
            sample = collector.sample()
            after = time.time()
            assert before <= sample["timestamp"] <= after
        finally:
            collector.close()

    def test_ram_percent_in_range(self):
        collector = MetricsCollector(track_gpu=False)
        try:
            sample = collector.sample()
            assert 0.0 <= sample["ram_percent"] <= 100.0
        finally:
            collector.close()

    def test_ram_rss_mb_positive(self):
        collector = MetricsCollector(track_gpu=False)
        try:
            sample = collector.sample()
            assert sample["ram_rss_mb"] > 0.0
        finally:
            collector.close()

    def test_wall_time_positive(self):
        collector = MetricsCollector(track_gpu=False)
        try:
            sample = collector.sample()
            assert sample["wall_time_s"] >= 0.0
        finally:
            collector.close()

    def test_multiple_samples_have_increasing_timestamps(self):
        collector = MetricsCollector(track_gpu=False)
        try:
            s1 = collector.sample()
            time.sleep(0.01)
            s2 = collector.sample()
            assert s2["timestamp"] >= s1["timestamp"]
        finally:
            collector.close()

    def test_close_is_idempotent(self):
        collector = MetricsCollector(track_gpu=False)
        collector.close()
        collector.close()  # Should not raise


class TestMetricsCollectorGPUWarning:
    """Tests for GPU-related warnings when pynvml is unavailable."""

    def test_warns_when_pynvml_not_installed(self):
        with patch("rl_energy_logger.collector._NVML_FOUND", False):
            with pytest.warns(ImportWarning, match="pynvml library not found"):
                collector = MetricsCollector(track_gpu=True)
                collector.close()
