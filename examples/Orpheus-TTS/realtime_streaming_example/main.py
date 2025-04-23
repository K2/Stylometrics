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

import struct
from flask import Flask, Response, request

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

# All multiprocessing/vllm/OrpheusModel code must be inside main block

def create_app(engine):
    app = Flask(__name__)

    @app.route('/tts', methods=['GET', 'POST'])
    def tts():
        """
        ApiNotes: Handles both GET and POST requests for TTS synthesis. POST expects JSON with 'text', 'voice', 'max_tokens', 'sample_rate'.
        GET uses 'prompt' query param for quick testing. Returns WAV audio stream.
        """
        # Reference: file-level ApiNotes.md, imperative paradigm
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

        def generate_audio_stream():
            yield create_wav_header(sample_rate=sample_rate)
            syn_tokens = engine.generate_speech(
                prompt=prompt,
                voice=voice,
                repetition_penalty=1.1,
                stop_token_ids=[128258],
                max_tokens=max_tokens,
                temperature=0.4,
                top_p=0.9
            )
            for chunk in syn_tokens:
                yield chunk

        return Response(generate_audio_stream(), mimetype='audio/wav')

    return app

if __name__ == "__main__":
    # All multiprocessing/vllm/OrpheusModel code must be inside this block (see ApiNotes)
    from orpheus_tts import OrpheusModel
    engine = OrpheusModel(model_name="canopylabs/orpheus-tts-0.1-finetune-prod")
    app = create_app(engine)
    app.run(host='0.0.0.0', port=8080, threaded=True)
