# stylometric_console.ApiNotes.md

## Overview

`stylometric_console.py` provides a unified, extensible rich console interface for monitoring all phases of stylometric/phonetic model workflows:
- **Training** (including fine-tuning)
- **Testing/Evaluation**
- **Custom/Experimental** phases

It is designed to:
- Replace and generalize the functionality of `stylometric_rich_debugger_console.py` and related scripts.
- Allow easy extension for new metrics, panels, and data streams.
- Support dataset-driven, model-driven, or custom monitoring workflows.

## Key Features

- **Phase-agnostic rendering**: Table, histogram, and log panels are reused across phases.
- **Phase-specific panels**: E.g., loss curve for training/fine-tuning.
- **Pluggable data streams**: Accepts iterables of dicts with phase-appropriate keys (e.g., similarities, labels, loss, log).
- **Extensible API**: Add new panels or metrics by extending the `run_console_monitor` function.
- **Example usage**: Demonstrates both training and testing phases with simulated data.

## Usage

```python
from stylometric_console import run_console_monitor

def my_data_stream():
    # yield dicts with keys: similarities, labels, loss (optional), log (optional)
    ...
run_console_monitor('train', my_data_stream())
```

## Extension Points
- Add new panels by extending the `panels` list in `run_console_monitor`.
- Add new data fields to the data stream dicts as needed.
- For new phases, add phase-specific logic as appropriate.

## Migration Notes
- The original `stylometric_rich_debugger_console.py` logic is now phase-agnostic and lives here.
- For dataset-driven training/finetuning, pass the appropriate data stream (e.g., from CAPT/concept datasets).
- For new model implementations, simply yield the relevant metrics/logs to the monitor.

## Design Rationale
- This design enables rapid prototyping and monitoring of new workflows, datasets, and models without duplicating UI logic.
- All rendering is handled via [rich](https://rich.readthedocs.io/), and the API is kept simple for LLM/agent integration.

## See Also
- Main project `ApiNotes.md` for integration and orchestration notes.
- Example usage in `stylometric_console.py`.
