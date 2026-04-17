"""Tests for rl_energy_logger.torch_wrapper.EnergyLoggerContext."""

import json
import os
import time

import pytest

from rl_energy_logger.torch_wrapper import EnergyLoggerContext


@pytest.fixture
def tmp_csv(tmp_path):
    return str(tmp_path / "ctx_log.csv")


@pytest.fixture
def tmp_jsonl(tmp_path):
    return str(tmp_path / "ctx_log.jsonl")


class TestEnergyLoggerContextInit:
    """Tests for constructor validation."""

    def test_sampling_interval_must_be_positive(self, tmp_csv):
        with pytest.raises(ValueError, match="sampling_interval_s must be positive"):
            EnergyLoggerContext(log_path=tmp_csv, sampling_interval_s=0)

    def test_buffer_size_must_be_positive(self, tmp_csv):
        with pytest.raises(ValueError, match="buffer_size must be positive"):
            EnergyLoggerContext(log_path=tmp_csv, buffer_size=0)


class TestEnergyLoggerContextLifecycle:
    """Tests for the context manager lifecycle (enter/exit)."""

    def test_context_creates_file(self, tmp_csv):
        with EnergyLoggerContext(
            log_path=tmp_csv,
            sampling_interval_s=0.1,
            track_gpu=False,
        ) as ctx:
            time.sleep(0.3)  # Allow a few samples
        assert os.path.exists(tmp_csv)

    def test_context_creates_jsonl_file(self, tmp_jsonl):
        with EnergyLoggerContext(
            log_path=tmp_jsonl,
            sampling_interval_s=0.1,
            track_gpu=False,
        ) as ctx:
            time.sleep(0.3)
        assert os.path.exists(tmp_jsonl)

    def test_background_thread_stops_after_exit(self, tmp_csv):
        with EnergyLoggerContext(
            log_path=tmp_csv,
            sampling_interval_s=0.1,
            track_gpu=False,
        ) as ctx:
            assert ctx._background_thread is not None
            assert ctx._background_thread.is_alive()
        # After exit, thread should have stopped
        assert not ctx._background_thread.is_alive()

    def test_writer_closed_after_exit(self, tmp_csv):
        with EnergyLoggerContext(
            log_path=tmp_csv,
            sampling_interval_s=0.1,
            track_gpu=False,
        ) as ctx:
            pass
        assert ctx.writer._is_closed


class TestEnergyLoggerContextSampling:
    """Tests for background hardware sampling."""

    def test_csv_contains_data_rows(self, tmp_csv):
        import csv as csv_mod
        with EnergyLoggerContext(
            log_path=tmp_csv,
            sampling_interval_s=0.05,
            track_gpu=False,
            queue_flush_interval_s=0.1,
        ) as ctx:
            time.sleep(0.4)  # Should produce ~8 samples

        with open(tmp_csv, "r", newline="", encoding="utf-8") as f:
            reader = csv_mod.DictReader(f)
            rows = list(reader)
        # At least a few samples should have been written
        assert len(rows) >= 2

    def test_jsonl_records_are_valid_json(self, tmp_jsonl):
        with EnergyLoggerContext(
            log_path=tmp_jsonl,
            sampling_interval_s=0.05,
            track_gpu=False,
            queue_flush_interval_s=0.1,
        ) as ctx:
            time.sleep(0.3)

        with open(tmp_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line.strip())
                assert "timestamp" in record
                assert "log_type" in record

    def test_background_samples_have_correct_log_type(self, tmp_jsonl):
        with EnergyLoggerContext(
            log_path=tmp_jsonl,
            sampling_interval_s=0.05,
            track_gpu=False,
            queue_flush_interval_s=0.1,
        ) as ctx:
            time.sleep(0.2)

        with open(tmp_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line.strip())
                if record.get("log_type") == "background_hw_sample":
                    return  # Found at least one — pass
        # If we get here, no background samples were found
        pytest.fail("No background_hw_sample records found in output")


class TestEnergyLoggerContextManualLog:
    """Tests for the manual log() method."""

    def test_manual_log_event(self, tmp_jsonl):
        with EnergyLoggerContext(
            log_path=tmp_jsonl,
            sampling_interval_s=10.0,  # Long interval — so only manual logs
            track_gpu=False,
        ) as ctx:
            ctx.log(event="epoch_end", epoch=1, loss=0.42)

        with open(tmp_jsonl, "r", encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) >= 1
        record = json.loads(lines[0])
        assert record["log_type"] == "epoch_end"
        assert record["epoch"] == 1
        assert record["loss"] == pytest.approx(0.42)

    def test_manual_log_includes_hardware_metrics(self, tmp_jsonl):
        with EnergyLoggerContext(
            log_path=tmp_jsonl,
            sampling_interval_s=10.0,
            track_gpu=False,
        ) as ctx:
            ctx.log(event="checkpoint")

        with open(tmp_jsonl, "r", encoding="utf-8") as f:
            record = json.loads(f.readline())
        assert "timestamp" in record
        assert "cpu_percent" in record
        assert "ram_percent" in record

    def test_manual_log_without_writer_warns(self, tmp_csv):
        ctx = EnergyLoggerContext(log_path=tmp_csv, track_gpu=False)
        # Don't enter context — writer is None
        with pytest.warns(RuntimeWarning, match="writer is not initialized"):
            ctx.log(event="test")
        ctx.collector.close()
