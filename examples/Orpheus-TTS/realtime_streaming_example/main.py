"""
ApiNotes.md (File-level) – main.py

Role:
    Flask server for Orpheus-TTS: exposes /tts endpoint for text-to-speech synthesis (GET/POST), streaming WAV audio.
    Loads OrpheusModel and manages all inference logic for local or remote TTS clients.

Design Goals:
    - Provide a robust, imperative, and modular TTS HTTP API for local and remote use.
    - Support both GET (debug) and POST (production) requests with flexible parameters.
    - Ensure multiprocessing/vllm initialization is safe for Python 3.8+ and CUDA (see multiprocessing notes).
    - Reference nearest ApiNotes.md for project and directory context.
    - All interface and behavioral assumptions must be documented in ApiNotes.
    - File must be importable and runnable as a script.
    - All changes must be reflected in project-level ApiNotes.md.

Architectural Constraints:
    - All multiprocessing/vllm/OrpheusModel code must be inside if __name__ == "__main__" to avoid spawn errors.
    - No subprocess or shell command execution outside Flask endpoints.
    - All configuration is loaded from canonical config files or passed as arguments.
    - All interface and behavioral assumptions are documented in ApiNotes.
    - File size monitored; suggest splitting if exceeding 1/3 context window.

Happy-Path:
    1. User runs this script directly (python main.py).
    2. OrpheusModel and Flask app are initialized inside main block.
    3. /tts endpoint accepts POST (JSON) or GET (query param) and streams WAV audio.
    4. Client receives audio and can play or save it.

ASCII Diagram:
    +-------------------+
    |   Flask app      |
    +-------------------+
              |
              v
    +-------------------+
    |  /tts endpoint    |
    +-------------------+
              |
              v
    +-------------------+
    | OrpheusModel/gen  |
    +-------------------+
              |
              v
    +-------------------+
    |  WAV stream resp  |
    +-------------------+
"""

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # Set before any CUDA/torch/model import

from multiprocessing import freeze_support
import struct
from flask import Flask, Response, request
import torch  # Added for dtype specification

from orpheus_tts import OrpheusModel  # Adjust import path if model is local or in another module

# Reference: file-level ApiNotes.md, imperative paradigm

def create_wav_header(sample_rate=24000, bits_per_sample=16, channels=1):
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    data_size = 0
    header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF',
        36 + data_size,
        b'WAVE',
        b'fmt ',
        16,
        1,
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b'data',
        data_size
    )
    return header

#@app.route('/tts', methods=['GET', 'POST'])
def tts(engine):
    """
    ApiNotes: Handles both GET and POST requests for TTS synthesis. POST expects JSON with 'text', 'voice', 'max_tokens', 'sample_rate'.
    GET uses 'prompt' query param for quick testing. Returns WAV audio stream.
    """
    if request.method == 'POST':
        # Parse JSON payload
        data = request.get_json(force=True)
        assert data is not None, 'POST /tts requires a JSON body'
        prompt = data.get('text')
        assert prompt and isinstance(prompt, str) and prompt.strip(), 'POST /tts: "text" must be a non-empty string'
        voice = data.get('voice', 'tara')
        max_tokens = data.get('max_tokens', 2000)
        sample_rate = data.get('sample_rate', 24000)
    else:
        # GET fallback for browser/debug use
        prompt = request.args.get('prompt', 'Hey there, looks like you forgot to provide a prompt!')
        voice = 'tara'
        max_tokens = 2000
        sample_rate = 24000

    # Generate audio using OrpheusModel
    audio = engine.generate(prompt, voice=voice, max_tokens=max_tokens, sample_rate=sample_rate)
    # Assumption: engine.generate returns raw PCM bytes or numpy array; adjust as needed

    # Prepare WAV header + audio
    wav_header = create_wav_header(sample_rate=sample_rate)
    def generate_wav():
        yield wav_header
        yield audio  # If audio is a generator, use: yield from audio

    return Response(generate_wav(), mimetype="audio/wav")

def create_app():
    app = Flask(__name__)
    engine = OrpheusModel(
        model_name="canopylabs/orpheus-tts-0.1-finetune-prod",
        dtype=torch.float16  # Explicitly set dtype for M60 compatibility; see file-level ApiNotes.md
    )

    @app.route('/tts', methods=['GET', 'POST'])
    def tts_route():
        return tts(engine)

    return app

if __name__ == "__main__":
    freeze_support()
    app = create_app()
    app.run(host='0.0.0.0', port=8181, threaded=True)

