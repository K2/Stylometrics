"""
ApiNotes:
  File-Level: pytest suite for tsfm_adapter
  - Verifies correct parsing of OpenSMILE config (frameSize/frameStep)
  - Checks ARFF→DataFrame conversion, time‐axis insertion, and missing‐data handling
  - Ensures full pipeline (`preprocess_opensmile_series`) returns a non‐empty DataFrame
  Dependencies: tsfm_adapter, pandas, pytest
  Paradigm: imperative step‑by‑step assertions, explicit expected success/failure annotations
"""

import os
import pandas as pd
import pytest
from tsfm_adapter import (
    _read_frame_params,
    arff_csv_to_timeseries,
    preprocess_opensmile_series
)

CONF = "conf/opensmile/emo_large.conf"
SAMPLE_CSV = "warmup_data/wav_files/phonetic_pairs_ollama_9_paraphrased.wav.smile.csv"

def test_read_frame_params_success():
    """(expected success) frameSize and frameStep must parse as positive floats."""
    fs, st = _read_frame_params(CONF)
    assert isinstance(fs, float) and fs > 0, "frameSize must be a positive float"
    assert isinstance(st, float) and st > 0, "frameStep must be a positive float"

def test_read_frame_params_missing(tmp_path):
    """(expected failure) missing config should raise AssertionError."""
    fake = tmp_path / "nope.conf"
    with pytest.raises(AssertionError):
        _read_frame_params(str(fake))

def test_arff_to_ts_shape_and_columns():
    """(expected success) DataFrame has 'time' column, correct header count, uniform step."""
    df = arff_csv_to_timeseries(SAMPLE_CSV, CONF)
    assert "time" in df.columns, "time column must be present"
    assert len(df.columns) > 5, "should have multiple feature columns"
    diffs = df.time.diff().dropna().unique()
    expected_step = _read_frame_params(CONF)[1]
    assert pytest.approx(diffs[0], rel=1e-6) == expected_step, \
        f"time step must equal frameStep ({expected_step})"

def test_arff_to_ts_no_data(tmp_path):
    """(expected failure) CSR with no data rows must assert."""
    bad = tmp_path / "empty.arff.csv"
    bad.write_text("@relation foo\n@attribute a numeric\n@data\n")
    with pytest.raises(AssertionError):
        arff_csv_to_timeseries(str(bad), CONF)

def test_full_pipeline_returns_df():
    """(expected success) preprocess_opensmile_series returns non-zero‐row DataFrame."""
    out = preprocess_opensmile_series(SAMPLE_CSV, CONF, window_size=3, normalize=False)
    assert isinstance(out, pd.DataFrame), "output must be a pandas DataFrame"
    assert out.shape[0] > 0, "pipeline must not collapse to zero rows"