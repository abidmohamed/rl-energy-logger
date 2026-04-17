"""Tests for rl_energy_logger.writers (CSVWriter, JSONLWriter, get_writer)."""

import csv
import json
import os
import tempfile

import pytest

from rl_energy_logger.writers import CSVWriter, JSONLWriter, get_writer, BaseWriter
from rl_energy_logger.exceptions import WriterError


# ---------------------------------------------------------------------------
# Helper fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_csv(tmp_path):
    """Return a path to a temporary CSV file."""
    return str(tmp_path / "test_output.csv")


@pytest.fixture
def tmp_jsonl(tmp_path):
    """Return a path to a temporary JSONL file."""
    return str(tmp_path / "test_output.jsonl")


SAMPLE_RECORD = {
    "timestamp": 1700000000.0,
    "cpu_percent": 42.5,
    "ram_percent": 60.1,
    "ram_rss_mb": 512.3,
}

SAMPLE_RECORD_2 = {
    "timestamp": 1700000002.0,
    "cpu_percent": 55.0,
    "ram_percent": 61.2,
    "ram_rss_mb": 520.0,
}


# ---------------------------------------------------------------------------
# CSVWriter tests
# ---------------------------------------------------------------------------

class TestCSVWriter:
    def test_creates_file(self, tmp_csv):
        writer = CSVWriter(tmp_csv)
        writer.write(SAMPLE_RECORD)
        writer.close()
        assert os.path.exists(tmp_csv)

    def test_writes_header_and_row(self, tmp_csv):
        writer = CSVWriter(tmp_csv)
        writer.write(SAMPLE_RECORD)
        writer.close()

        with open(tmp_csv, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 1
        assert float(rows[0]["cpu_percent"]) == pytest.approx(42.5)

    def test_writes_multiple_rows(self, tmp_csv):
        writer = CSVWriter(tmp_csv)
        writer.write(SAMPLE_RECORD)
        writer.write(SAMPLE_RECORD_2)
        writer.close()

        with open(tmp_csv, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 2

    def test_creates_parent_directories(self, tmp_path):
        nested = str(tmp_path / "sub" / "dir" / "log.csv")
        writer = CSVWriter(nested)
        writer.write(SAMPLE_RECORD)
        writer.close()
        assert os.path.exists(nested)

    def test_warns_on_new_keys(self, tmp_csv):
        writer = CSVWriter(tmp_csv)
        writer.write(SAMPLE_RECORD)
        record_with_extra = {**SAMPLE_RECORD_2, "new_field": 999}
        with pytest.warns(RuntimeWarning, match="New keys"):
            writer.write(record_with_extra)
        writer.close()

    def test_close_is_idempotent(self, tmp_csv):
        writer = CSVWriter(tmp_csv)
        writer.write(SAMPLE_RECORD)
        writer.close()
        writer.close()  # Should not raise

    def test_write_after_close_warns(self, tmp_csv):
        writer = CSVWriter(tmp_csv)
        writer.write(SAMPLE_RECORD)
        writer.close()
        with pytest.warns(RuntimeWarning, match="closed"):
            writer.write(SAMPLE_RECORD_2)

    def test_context_manager(self, tmp_csv):
        with CSVWriter(tmp_csv) as writer:
            writer.write(SAMPLE_RECORD)
        # File should be closed after exiting context
        assert writer._is_closed


# ---------------------------------------------------------------------------
# JSONLWriter tests
# ---------------------------------------------------------------------------

class TestJSONLWriter:
    def test_creates_file(self, tmp_jsonl):
        writer = JSONLWriter(tmp_jsonl)
        writer.write(SAMPLE_RECORD)
        writer.close()
        assert os.path.exists(tmp_jsonl)

    def test_writes_valid_json_lines(self, tmp_jsonl):
        writer = JSONLWriter(tmp_jsonl)
        writer.write(SAMPLE_RECORD)
        writer.write(SAMPLE_RECORD_2)
        writer.close()

        with open(tmp_jsonl, "r", encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) == 2
        parsed = json.loads(lines[0])
        assert parsed["cpu_percent"] == pytest.approx(42.5)

    def test_handles_extra_keys_gracefully(self, tmp_jsonl):
        """JSONL writer should handle records with varying keys without error."""
        writer = JSONLWriter(tmp_jsonl)
        writer.write(SAMPLE_RECORD)
        writer.write({**SAMPLE_RECORD_2, "gpu_power_watts": 150.0})
        writer.close()

        with open(tmp_jsonl, "r", encoding="utf-8") as f:
            lines = f.readlines()
        row2 = json.loads(lines[1])
        assert row2["gpu_power_watts"] == 150.0

    def test_close_is_idempotent(self, tmp_jsonl):
        writer = JSONLWriter(tmp_jsonl)
        writer.write(SAMPLE_RECORD)
        writer.close()
        writer.close()

    def test_write_after_close_warns(self, tmp_jsonl):
        writer = JSONLWriter(tmp_jsonl)
        writer.close()
        with pytest.warns(RuntimeWarning, match="closed"):
            writer.write(SAMPLE_RECORD)

    def test_context_manager(self, tmp_jsonl):
        with JSONLWriter(tmp_jsonl) as writer:
            writer.write(SAMPLE_RECORD)
        assert writer._is_closed


# ---------------------------------------------------------------------------
# get_writer factory tests
# ---------------------------------------------------------------------------

class TestGetWriter:
    def test_returns_csv_writer(self, tmp_csv):
        writer = get_writer(tmp_csv)
        assert isinstance(writer, CSVWriter)
        writer.close()

    def test_returns_jsonl_writer(self, tmp_jsonl):
        writer = get_writer(tmp_jsonl)
        assert isinstance(writer, JSONLWriter)
        writer.close()

    def test_raises_on_unsupported_extension(self, tmp_path):
        bad_path = str(tmp_path / "log.txt")
        with pytest.raises(ValueError, match="Unsupported log file extension"):
            get_writer(bad_path)

    def test_case_insensitive_extension(self, tmp_path):
        csv_upper = str(tmp_path / "log.CSV")
        writer = get_writer(csv_upper)
        assert isinstance(writer, CSVWriter)
        writer.close()
