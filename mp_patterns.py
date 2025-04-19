"""
ApiNotes.md (File-level) - mp_patterns.py

Role:
    Implementation of multiprocessing-enabled functions for scanning and analyzing OpenSMILE CSV files.
    Provides performance-optimized extraction and analysis of phonetic patterns from audio features.

Design Goals:
    - Parallelize CSV parsing for significant performance improvement on large datasets
    - Maintain consistent interface with the main stylometric_phonetic_encoder_ollama.py file
    - Provide clear error handling and logging
    - Enable scalable processing of acoustic features across many files

Architectural Constraints:
    - Must be compatible with the existing project structure and dependencies
    - Parallel processing must be safe for file I/O operations
    - All functions must include comprehensive documentation
    - Error handling should be robust and informative

Happy-Path:
    1. User calls scan_csvs_for_phonetic_patterns with directory path and label map
    2. Function distributes file parsing across multiple processes
    3. Results are aggregated and processed
    4. Each feature is analyzed via LLM with appropriate context
    5. A comprehensive summary is returned to the caller
"""

import os
import json
import concurrent.futures
from typing import Dict, List, Tuple, Any, Optional
from rich.progress import track
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Required to access these functions from the main module
from stylometric_phonetic_encoder_ollama import call_llm, load_ollama_model
from logging_utils import log_message

def _parse_opensmile_csv_for_mp(args: Tuple[str, Dict[str, Any], List[str]]) -> Optional[Tuple[str, Dict[str, float], Any]]:
    """
    Helper function for multiprocessing that parses a single OpenSMILE CSV file.
    
    Args:
        args: Tuple of (csv_path, label_map, feature_keys)
        
    Returns:
        Tuple of (filename, features_dict, label) or None if parsing fails
    
    This function is designed to be called by concurrent.futures.ProcessPoolExecutor
    for parallel CSV processing, improving performance on multi-core systems.
    """
    csv_path, label_map, feature_keys = args
    fname = os.path.basename(csv_path)
    
    # Determine label from filename marker
    label = None
    for marker, lab in label_map.items():
        if marker in fname:
            label = lab
            break
    
    if label is None:
        return None  # Skip if no label can be determined
    
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        # Parse ARFF header and find @data section
        data_start = None
        header_keys = []
        for idx, line in enumerate(lines):
            if line.strip().lower() == "@data":
                data_start = idx + 1
                break
            if line.startswith("@attribute"):
                parts = line.strip().split()
                if len(parts) >= 3:
                    header_keys.append(parts[1])
        
        if data_start is None or data_start >= len(lines):
            return None  # Invalid CSV format
        
        # Parse data values
        data_line = lines[data_start].strip()
        data_values = [v.strip() for v in data_line.split(",")]
        
        # Extract feature values
        feature_dict = {}
        use_keys = feature_keys if feature_keys else header_keys
        key_to_idx = {k: i for i, k in enumerate(header_keys)}
        
        for k in use_keys:
            idx = key_to_idx.get(k)
            if idx is not None and idx < len(data_values):
                try:
                    feature_dict[k] = float(data_values[idx])
                except ValueError:
                    feature_dict[k] = float('nan')
            else:
                feature_dict[k] = None
        
        return (fname, feature_dict, label)
    
    except Exception:
        return None  # Return None on any parsing error


def scan_csvs_for_phonetic_patterns(
    audio_dir: str,
    label_map: dict,
    log_file: str = "",
    max_files: int = 100,
    feature_keys: list = []
) -> Dict[str, Any]:
    """
    Scans all OpenSMILE CSVs in audio_dir using multiprocessing for performance,
    aggregates values for each feature across files, and queries the LLM for pattern analysis.
    
    Args:
        audio_dir: Directory containing OpenSMILE CSV files
        label_map: Dictionary mapping filename patterns to labels
        log_file: Path to log file for detailed logging
        max_files: Maximum number of files to process (0 for unlimited)
        feature_keys: Specific feature keys to extract (empty list for all features)
        
    Returns:
        Dictionary containing analysis results and summary statistics
    
    Performance optimization:
        Uses ProcessPoolExecutor to parse CSV files in parallel,
        significantly improving performance on large datasets and multi-core systems.
    """
    # Initialize the LLM model with large context window
    load_ollama_model("phi4", num_ctx_val=32768)
    
    # Data structures for feature collection
    feature_matrix = {}  # {feature_name: {filename: value, ...}}
    label_dict = {}      # {filename: label}
    processed = 0
    skipped = 0
    all_header_keys = set()
    
    # Gather all CSV files to process
    csv_files = [os.path.join(audio_dir, f) for f in os.listdir(audio_dir) 
                if f.endswith(".wav.smile.csv")]
    
    if max_files > 0:
        csv_files = csv_files[:max_files]
    
    log_message("INFO", f"[START] Processing {len(csv_files)} CSV files with multiprocessing", log_file)
    
    # Prepare arguments for multiprocessing
    mp_args = [(csv_path, label_map, feature_keys) for csv_path in csv_files]
    
    # Process CSV files in parallel
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for result in track(executor.map(_parse_opensmile_csv_for_mp, mp_args), 
                         total=len(mp_args),
                         description="Parsing CSV files"):
            if result is not None:
                fname, feature_dict, label = result
                label_dict[fname] = label
                
                # Organize features by feature name
                for feature_name, value in feature_dict.items():
                    if feature_name not in feature_matrix:
                        feature_matrix[feature_name] = {}
                    feature_matrix[feature_name][fname] = value
                
                all_header_keys.update(feature_dict.keys())
                processed += 1
            else:
                skipped += 1
    
    log_message("INFO", f"[PROGRESS] Processed {processed} files, skipped {skipped}", log_file)
    
    # Analyze features with LLM
    feature_results = {}
    for feature_name in track(list(feature_matrix.keys()), 
                           description="Analyzing features with LLM"):
        values = feature_matrix[feature_name]
        
        prompt = (
            f"You are given the values of the OpenSMILE acoustic feature '{feature_name}' "
            f"across multiple speech samples. Each sample is identified by filename. "
            f"You are also given a label map indicating whether each file is harmonic (1) or dissonant (0).\n"
            f"Analyze the values for '{feature_name}' and reply with:\n"
            "- The variance of the feature across all files\n"
            "- The direction of change (does it tend to be higher for harmonic or dissonant?)\n"
            "- Any additional qualifiers or patterns you observe (e.g., outliers, bimodality, etc)\n"
            "Respond with a JSON object of the form:\n"
            "{\n"
            "  \"variance\": float,\n"
            "  \"direction\": \"higher for harmonic\" | \"higher for dissonant\" | \"no clear direction\",\n"
            "  \"qualifiers\": \"...\"\n"
            "}\n\n"
            f"Feature Values (JSON):\n{json.dumps(values, indent=2)}\n"
            f"Label Map (JSON):\n{json.dumps(label_dict, indent=2)}"
        )
        
        response = call_llm(prompt, max_tokens=4096, stream=False)
        feature_results[feature_name] = response
        
        if log_file:
            log_message("INFO", f"[LLM_ANALYSIS] {feature_name}: {response}", log_file)
    
    summary = {
        "total_processed": processed,
        "total_skipped": skipped,
        "feature_keys": list(all_header_keys),
        "results": feature_results
    }
    
    print(f"[INFO] Multiprocessed LLM feature scan complete. Processed: {processed}, Skipped: {skipped}")
    if log_file:
        log_message("INFO", f"[SUMMARY] Multiprocessed scan complete. Processed: {processed}, Skipped: {skipped}", log_file)
    
    return summary


if __name__ == "__main__":
    # Example usage when script is run directly
    import argparse
    
    parser = argparse.ArgumentParser(description="Scan OpenSMILE CSVs for phonetic patterns using multiprocessing")
    parser.add_argument('--audio-dir', required=True, help="Directory containing OpenSMILE CSV files")
    parser.add_argument('--log-file', default="", help="Path to log file")
    parser.add_argument('--max-files', type=int, default=100, help="Maximum number of files to process (0 for unlimited)")
    args = parser.parse_args()
    
    # Example label map
    label_map = {
        "harmonic": 1,
        "dissonant": 0
    }
    
    results = scan_csvs_for_phonetic_patterns(
        audio_dir=args.audio_dir,
        label_map=label_map,
        log_file=args.log_file,
        max_files=args.max_files
    )
    
    print(f"Processed {results['total_processed']} files")
    print(f"Found {len(results['feature_keys'])} unique features")
    print(f"LLM analysis completed for {len(results['results'])} features")
