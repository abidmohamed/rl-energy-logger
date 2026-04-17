"""Tests for rl_energy_logger.sb3_callback.EnergyLogCallback."""

import csv
import os
import time
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

# The callback module conditionally imports SB3.  We mock the import so
# tests can run without SB3 installed.
# First, let's check if we can import the real callback.
try:
    from stable_baselines3.common.callbacks import BaseCallback as _RealBase
    _HAS_SB3 = True
except ImportError:
    _HAS_SB3 = False


@pytest.fixture
def tmp_csv(tmp_path):
    return str(tmp_path / "sb3_log.csv")


@pytest.fixture
def tmp_jsonl(tmp_path):
    return str(tmp_path / "sb3_log.jsonl")


# ---------------------------------------------------------------------------
# Tests that work regardless of SB3 installation
# ---------------------------------------------------------------------------

class TestEnergyLogCallbackWithoutSB3:
    """When SB3 is NOT installed, the placeholder class should raise ImportError."""

    @pytest.mark.skipif(_HAS_SB3, reason="SB3 is installed — skip no-SB3 test")
    def test_raises_import_error_without_sb3(self, tmp_csv):
        from rl_energy_logger.sb3_callback import EnergyLogCallback
        with pytest.raises(ImportError, match="Stable-Baselines3 is not installed"):
            EnergyLogCallback(log_path=tmp_csv)


class TestEnergyLogCallbackWithSB3:
    """Tests for the real callback — requires SB3 installed."""

    @pytest.mark.skipif(not _HAS_SB3, reason="SB3 not installed")
    def test_instantiation(self, tmp_csv):
        from rl_energy_logger.sb3_callback import EnergyLogCallback
        cb = EnergyLogCallback(log_path=tmp_csv, track_gpu=False)
        assert cb.log_path == tmp_csv
        assert cb.sampling_interval_s == 2.0
        cb.collector.close()

    @pytest.mark.skipif(not _HAS_SB3, reason="SB3 not installed")
    def test_sampling_interval_must_be_positive(self, tmp_csv):
        from rl_energy_logger.sb3_callback import EnergyLogCallback
        with pytest.raises(ValueError, match="positive"):
            EnergyLogCallback(log_path=tmp_csv, sampling_interval_s=0)

    @pytest.mark.skipif(not _HAS_SB3, reason="SB3 not installed")
    def test_init_callback_creates_writer(self, tmp_csv):
        from rl_energy_logger.sb3_callback import EnergyLogCallback
        cb = EnergyLogCallback(log_path=tmp_csv, track_gpu=False)
        cb._init_callback()
        assert cb.writer is not None
        cb.writer.close()
        cb.collector.close()

    @pytest.mark.skipif(not _HAS_SB3, reason="SB3 not installed")
    def test_on_step_writes_after_interval(self, tmp_csv):
        from rl_energy_logger.sb3_callback import EnergyLogCallback
        cb = EnergyLogCallback(
            log_path=tmp_csv,
            sampling_interval_s=0.01,  # Very short interval for testing
            track_gpu=False,
            log_rl_metrics=False,
        )
        cb._init_callback()
        cb._on_training_start()
        cb.num_timesteps = 100

        time.sleep(0.02)  # Wait past the interval
        result = cb._on_step()

        assert result is True
        cb._on_training_end()
        assert os.path.exists(tmp_csv)

    @pytest.mark.skipif(not _HAS_SB3, reason="SB3 not installed")
    def test_on_step_returns_true(self, tmp_csv):
        """_on_step must always return True to continue training."""
        from rl_energy_logger.sb3_callback import EnergyLogCallback
        cb = EnergyLogCallback(log_path=tmp_csv, track_gpu=False)
        cb._init_callback()
        cb._on_training_start()
        assert cb._on_step() is True
        cb._on_training_end()

    @pytest.mark.skipif(not _HAS_SB3, reason="SB3 not installed")
    def test_on_step_with_no_writer_is_safe(self, tmp_csv):
        """If writer init failed, _on_step should not crash."""
        from rl_energy_logger.sb3_callback import EnergyLogCallback
        cb = EnergyLogCallback(log_path=tmp_csv, track_gpu=False)
        cb.writer = None  # Simulate failed init
        assert cb._on_step() is True
        cb.collector.close()

    @pytest.mark.skipif(not _HAS_SB3, reason="SB3 not installed")
    def test_jsonl_output_format(self, tmp_jsonl):
        import json
        from rl_energy_logger.sb3_callback import EnergyLogCallback
        cb = EnergyLogCallback(
            log_path=tmp_jsonl,
            sampling_interval_s=0.01,
            track_gpu=False,
            log_rl_metrics=False,
        )
        cb._init_callback()
        cb._on_training_start()
        cb.num_timesteps = 50
        time.sleep(0.02)
        cb._on_step()
        cb._on_training_end()

        with open(tmp_jsonl, "r", encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) >= 1
        record = json.loads(lines[0])
        assert "timestamp" in record
        assert "sb3_num_timesteps" in record
