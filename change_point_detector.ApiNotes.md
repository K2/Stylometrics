# Change Point Detector GenAIScript Wrapper - API Notes

## Overview

This script (`change_point_detector.genai.mts`) provides a GenAIScript interface to the Python-based change point detection logic in `change_point_detection.py`. It accepts a timeline of texts (as a single string with newline separators or an array), prepares the input for the Python script, executes it via `host.exec`, and returns the detected change point information as JSON.

## Design Goals

*   **Integration:** Bridge GenAIScript with the Python `change_point_detection.py` module.
*   **Input Handling:** Accept timeline data (string or array) and manage temporary file creation.
*   **Execution:** Execute the Python script using `host.exec`.
*   **Output Parsing:** Parse the JSON output `[detected: boolean, index: number]` from the Python script.
*   **Error Handling:** Report errors from the Python execution.

## Key Components

*   **`script({...})`:** Defines the script and parameters (`timelineData`, `agreementThreshold`).
*   **`workspace.writeText()`:** Creates a temporary file with the timeline text (one entry per line).
*   **`host.exec()`:** Executes `python3 change_point_detection.py <temp_file_path> [--threshold <value>]`.
*   **`JSON.parse()`:** Parses the `stdout` from the Python script.

## Core Flow

1.  **Input:** Receives `timelineData` (string or array) and optional `agreementThreshold`.
2.  **Format Input:** Converts `timelineData` into a newline-separated string if it's an array.
3.  **Temp File:** Writes the formatted timeline to a temporary file.
4.  **Execution:** Calls `host.exec` to run `python3 change_point_detection.py` with the temp file path and threshold argument.
5.  **Output Handling:** Parses the JSON array `[detected, index]` from `stdout` or reports errors.
6.  **Return:** Returns an object `{ detected: boolean, changePoint: number }` or an error structure.

## Constraints & Assumptions

*   **Python Environment:** Assumes `python3` is available in the host's PATH.
*   **Dependencies:** Assumes the Python environment has necessary libraries installed (`numpy`, `pandas`, `ruptures` - as required by `change_point_detection.py`).
*   **`change_point_detection.py` Interface:** Assumes the script accepts a file path argument, an optional `--threshold` argument, and prints a JSON array `[boolean, number]` to `stdout`.
*   **Visualization:** This wrapper does *not* handle the visualization part (`analyze_timeline`) of the Python script. It focuses on returning the detection result data.

## Usage Example (Conceptual)

```genaiscript
const timeline = [
    "First text by author A.",
    "Second text by author A.",
    "Third text, maybe by B?",
    "Fourth text, definitely B."
];
const result = await runPrompt("change_point_detector", {
    vars: { timelineData: timeline, agreementThreshold: 0.2 }
});
if (!result.error) {
    console.log("Change Detection Result:", result.json);
    // result.json should be like { detected: true, changePoint: 1 }
} else {
    console.error("Error:", result.error);
}
```

## Debugging

*   Check `stderr` from `host.exec` for Python errors (e.g., `ruptures` not found).
*   Verify the temporary timeline file format (one text per line).
*   Ensure `change_point_detection.py` handles its command-line arguments (`--threshold`) correctly and outputs valid JSON `[bool, int]`.
*   Confirm Python dependencies are installed (`pip install numpy pandas ruptures`).