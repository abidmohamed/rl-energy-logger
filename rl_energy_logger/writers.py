import csv
import json
import pathlib
import warnings
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

from .exceptions import WriterError

class BaseWriter(ABC):
    """Abstract base class for log writers."""
    def __init__(self, path: str):
        self.path = pathlib.Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
        self._is_closed = False

    @abstractmethod
    def write(self, record: Dict[str, Any]):
        """Writes a single data record."""
        pass

    @abstractmethod
    def close(self):
        """Closes the writer and any open file handles."""
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        # Ensure close is called if the object is garbage collected
        # without explicit closing or context manager use.
        self.close()


class CSVWriter(BaseWriter):
    """Writes log data to a CSV file."""
    def __init__(self, path: str):
        super().__init__(path)
        self._file_handle = None
        self._writer = None
        self._fieldnames: Optional[List[str]] = None
        try:
            # Check if file exists and is not empty to determine if header needs writing
            write_header = not self.path.exists() or self.path.stat().st_size == 0
            # Open in append mode with newline='' to prevent extra blank rows
            self._file_handle = open(self.path, "a", newline="", encoding="utf-8")
            # No writer/fieldnames yet, will be initialized on first write
            if write_header:
                 self._requires_header = True
            else:
                 self._requires_header = False
                 # TODO: Optionally read existing header to ensure consistency? For simplicity, we don't now.

        except IOError as e:
            raise WriterError(f"Failed to open or prepare CSV file at {self.path}: {e}") from e

    def write(self, record: Dict[str, Any]):
        """Writes a record to the CSV file. Initializes header on first write if needed."""
        if self._is_closed:
            warnings.warn(f"Attempted to write to closed CSVWriter ({self.path})", RuntimeWarning)
            return
        if not self._file_handle:
             raise WriterError(f"CSVWriter file handle is not open for {self.path}.")

        if self._writer is None:
            # First write operation, setup the DictWriter
            self._fieldnames = list(record.keys())
            self._writer = csv.DictWriter(self._file_handle, fieldnames=self._fieldnames, extrasaction='ignore')
            if self._requires_header:
                try:
                    self._writer.writeheader()
                    self._requires_header = False # Header has been written
                except IOError as e:
                    raise WriterError(f"Failed to write header to CSV file {self.path}: {e}") from e

        # Check if new fields have appeared compared to the initial header
        current_keys = list(record.keys())
        if self._fieldnames is not None and set(current_keys) != set(self._fieldnames):
             # This simple implementation ignores new fields based on `extrasaction='ignore'`
             # A more robust implementation might:
             # 1. Re-open the file, read all data, add new columns, rewrite (expensive).
             # 2. Log a warning about ignored fields.
             # 3. Use a different library (like pandas) that handles this better.
             new_keys = set(current_keys) - set(self._fieldnames)
             if new_keys:
                 warnings.warn(f"New keys {new_keys} found in record will be ignored as they are not in the initial CSV header: {self._fieldnames}", RuntimeWarning)
             # We could update self._fieldnames and recreate the writer, but this can corrupt CSV structure easily. Sticking to ignore.

        try:
            self._writer.writerow(record)
            self._file_handle.flush() # Ensure data is written promptly
        except IOError as e:
            warnings.warn(f"Failed to write record to CSV file {self.path}: {e}", RuntimeWarning)
        except Exception as e: # Catch potential DictWriter errors (e.g., dict contains non-primitive types)
             warnings.warn(f"Unexpected error writing record {record} to CSV {self.path}: {e}", RuntimeWarning)


    def close(self):
        """Closes the CSV file handle."""
        if not self._is_closed and self._file_handle:
            try:
                self._file_handle.flush()
                self._file_handle.close()
                self._is_closed = True
                self._file_handle = None
                self._writer = None
                # print(f"[rl-energy-logger] Closed writer for {self.path}") # Optional verbose log
            except IOError as e:
                warnings.warn(f"Error closing CSV file {self.path}: {e}", RuntimeWarning)


class JSONLWriter(BaseWriter):
    """Writes log data to a JSON Lines (.jsonl) file."""
    def __init__(self, path: str):
        super().__init__(path)
        self._file_handle = None
        try:
            # Open in append mode
            self._file_handle = open(self.path, "a", encoding="utf-8")
        except IOError as e:
            raise WriterError(f"Failed to open or prepare JSONL file at {self.path}: {e}") from e

    def write(self, record: Dict[str, Any]):
        """Writes a record as a JSON string on a new line."""
        if self._is_closed:
            warnings.warn(f"Attempted to write to closed JSONLWriter ({self.path})", RuntimeWarning)
            return
        if not self._file_handle:
            raise WriterError(f"JSONLWriter file handle is not open for {self.path}.")

        try:
            json_str = json.dumps(record, separators=(',', ':')) # Compact format
            self._file_handle.write(json_str + "\n")
            self._file_handle.flush() # Ensure data is written promptly
        except TypeError as e:
             warnings.warn(f"Failed to serialize record to JSON for {self.path}. Record contained non-serializable data? Error: {e}\nRecord: {record}", RuntimeWarning)
        except IOError as e:
            warnings.warn(f"Failed to write record to JSONL file {self.path}: {e}", RuntimeWarning)

    def close(self):
        """Closes the JSONL file handle."""
        if not self._is_closed and self._file_handle:
            try:
                self._file_handle.flush()
                self._file_handle.close()
                self._is_closed = True
                self._file_handle = None
                # print(f"[rl-energy-logger] Closed writer for {self.path}") # Optional verbose log
            except IOError as e:
                warnings.warn(f"Error closing JSONL file {self.path}: {e}", RuntimeWarning)

# Factory function to get the correct writer based on file extension
def get_writer(log_path: str) -> BaseWriter:
    """
    Creates and returns a suitable BaseWriter instance based on the log file extension.

    Args:
        log_path (str): The path to the log file.

    Returns:
        BaseWriter: An instance of CSVWriter or JSONLWriter.

    Raises:
        ValueError: If the file extension is not supported (.csv or .jsonl).
    """
    path = pathlib.Path(log_path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return CSVWriter(log_path)
    elif suffix == ".jsonl":
        return JSONLWriter(log_path)
    else:
        raise ValueError(f"Unsupported log file extension: '{suffix}'. Use '.csv' or '.jsonl'.")