"""
ApiNotes:  
  File-Level: Adapter for OpenSMILE ARFF/CSV -> IBM Granite TimeSeriesPreprocessor  
  - Parses emo_large.conf for frameSize/frameStep              
  - Reads all @data rows into a pandas.DataFrame              
  - Injects a 'time' column = frame_index * frameStep          
  - Calls TimeSeriesPreprocessor.fit_transform(df)            
  Design: imperative, observable, inline asserts on assumptions.
"""
import os
import re
import pandas as pd
from tsfm_public.toolkit.time_series_preprocessor import TimeSeriesPreprocessor

def _read_frame_params(config_path: str) -> tuple[float, float]:
    """Extract frameSize and frameStep (in seconds) from an OpenSMILE config."""
    assert os.path.isfile(config_path), f"Config not found: {config_path}"
    fs = None
    st = None
    with open(config_path, 'r', encoding='utf-8') as f:
        for line in f:
            l = line.split(';',1)[0].strip()
            if l.startswith('frameSize'):
                fs = float(l.split('=',1)[1])
            elif l.startswith('frameStep'):
                st = float(l.split('=',1)[1])
            if fs is not None and st is not None:
                break
    assert fs is not None and st is not None, "frameSize/frameStep not found in config"
    return fs, st  # now a tuple

def arff_csv_to_timeseries(
    csv_path: str,
    config_path: str
) -> pd.DataFrame:
    """
    Reads an OpenSMILE ARFF‑CSV and returns a DataFrame with:
      ['time', feature1, feature2, …]
    where time = frame_index * frameStep.
    """
    assert os.path.isfile(csv_path), f"CSV not found: {csv_path}"
    # load all lines
    lines = open(csv_path, encoding='utf-8').read().splitlines()
    header = []
    data_idx = None
    for i,l in enumerate(lines):
        ll = l.strip().lower()
        if ll == '@data':
            data_idx = i+1
            break
        if l.startswith('@attribute'):
            parts = l.strip().split()
            assert len(parts)>=3, f"Malformed @attribute line: {l}"
            header.append(parts[1])
    assert data_idx is not None, "@data section missing"
    # parse all data rows
    raw = []
    for row in lines[data_idx:]:
        row = row.strip()
        if not row or row.startswith('%'):  # skip blanks/comments
            continue
        parts = [v.strip() for v in row.split(',')]
        # allow rows shorter than header (pad with nan)
        if len(parts) < len(header):
            parts += [float('nan')]*(len(header)-len(parts))
        raw.append(parts)
    assert raw, "No data rows found"
    df = pd.DataFrame(raw, columns=header)
    # build time axis
    _, frameStep = _read_frame_params(config_path)  # frameSize unused
    df.insert(0, 'time', [i * frameStep for i in range(len(df))])
    return df

def preprocess_opensmile_series(
     csv_path: str,
     config_path: str = "conf/opensmile/emo_large.conf",
     **tsfm_kwargs
 ) -> pd.DataFrame:
   """
   ...ApiNotes: use TimeSeriesPreprocessor.train + preprocess for DataFrame I/O...
   """
   ts = arff_csv_to_timeseries(csv_path, config_path)
   tsp = TimeSeriesPreprocessor(**tsfm_kwargs)
   # train the preprocessor
   tsp.train(ts)
   # apply preprocessing
   return tsp.preprocess(ts)

# Example usage:
# df = preprocess_opensmile_series(
#     "warmup_data/wav_files/phonetic_pairs_ollama_9_paraphrased.wav.smile.csv",
#     config_path="conf/opensmile/emo_large.conf",
#     window_size=5,
#     normalize=True
# )