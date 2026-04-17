"""Tests for rl_energy_logger.exceptions."""

import pytest

from rl_energy_logger.exceptions import (
    RLEnergyLoggerError,
    NVMLDriverError,
    CollectorError,
    WriterError,
)


class TestExceptionHierarchy:
    """Verify the exception class hierarchy."""

    def test_base_exception_is_exception(self):
        assert issubclass(RLEnergyLoggerError, Exception)

    def test_nvml_driver_error_inherits_base(self):
        assert issubclass(NVMLDriverError, RLEnergyLoggerError)

    def test_collector_error_inherits_base(self):
        assert issubclass(CollectorError, RLEnergyLoggerError)

    def test_writer_error_inherits_base(self):
        assert issubclass(WriterError, RLEnergyLoggerError)

    def test_catch_all_via_base(self):
        """All custom exceptions should be catchable via the base class."""
        for exc_class in (NVMLDriverError, CollectorError, WriterError):
            with pytest.raises(RLEnergyLoggerError):
                raise exc_class("test message")

    def test_exception_message_preserved(self):
        msg = "GPU 0 not found"
        exc = NVMLDriverError(msg)
        assert str(exc) == msg
