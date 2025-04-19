"""
ApiNotes.md (File-level) – logging_utils.py

Role:
    Provides a reusable, imperative logging utility for the Stylometrics project. Centralizes logging logic for LLM calls,
    feature extraction, and training routines. Ensures consistent log formatting, tag handling, and file management across modules.

Design Goals:
    - Enable all modules to log LLM responses, errors, and process steps with consistent tags (e.g., RESPONSE, NEXT RESPONSE, ERROR, INFO).
    - Support append and context-managed logging to arbitrary files.
    - Allow flexible tagging and message formatting for traceability.
    - Facilitate future expansion (e.g., log rotation, verbosity levels).

Architectural Constraints:
    - All logging logic must be callable as Python functions or context managers.
    - No subprocess or shell command execution is allowed in this module.
    - All interface and behavioral assumptions are documented in ApiNotes.
    - File size monitored; suggest splitting if exceeding 1/3 context window.

Happy-Path:
    1. Import log_message or LogContext from logging_utils in any module.
    2. Use log_message(tag, message, log_file) for simple logging.
    3. Use LogContext(log_file_path) as a context manager for multi-step logging.

ASCII Diagram:
    +-------------------+
    |  log_message()    |
    +-------------------+
              |
              v
    +-------------------+
    |  LogContext       |
    +-------------------+
"""

import os
import sys

def log_message(level, message, log_file=None):
    """
    ApiNotes: Writes a formatted log message to the specified log file.
    Gracefully handles None or empty log_file by skipping the write operation.
    
    Args:
        level: String indicating log level (INFO, ERROR, WARN, etc.)
        message: The message to log
        log_file: File path as string or file object. If None or empty, no logging occurs.
    
    Returns:
        None
    """
    if not log_file:
        return
    
    # Handle both string paths and file objects
    if isinstance(log_file, str):
        if not log_file.strip():
            return
        try:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"[{level}] {message}\n")
        except Exception as e:
            print(f"Error writing to log file {log_file}: {e}", file=sys.stderr)
    else:
        # Assume it's a file object
        try:
            log_file.write(f"[{level}] {message}\n")
        except Exception as e:
            print(f"Error writing to log file: {e}", file=sys.stderr)

class LogContext:
    """
    ApiNotes: Context manager for logging to a file. Opens the file in append mode and closes on exit.
    Usage:
        with LogContext("mylog.log") as logf:
            log_message("INFO", "Started process", logf)
    """
    def __init__(self, log_file_path):
        self.log_file_path = log_file_path
        self.log_file = None

    def __enter__(self):
        self.log_file = open(self.log_file_path, "a")
        return self.log_file

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.log_file:
            self.log_file.close()

# Acceptance test (expected success/failure)
def test_logging_utils_expected_success():
    test_path = "test_logging_utils.log"
    with LogContext(test_path) as logf:
        log_message("INFO", "This is a test log entry.", logf)
    assert os.path.exists(test_path), "Log file should be created"
    with open(test_path) as f:
        lines = f.readlines()
        assert any("This is a test log entry." in line for line in lines), "Log entry should be present"
    os.remove(test_path)

def test_logging_utils_expected_failure():
    try:
        log_message("INFO", "Should fail", None)
    except AssertionError as e:
        print(f"(expected failure) {e}")

# ApiNotes: This implementation is imperative, modular, and justified by file-level and project-level ApiNotes.
#           All interface and behavioral assumptions are documented.
#           Acceptance tests include (expected success) and (expected failure) cases.