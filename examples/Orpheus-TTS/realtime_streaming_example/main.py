from flask import Flask, Response, request
import struct
from orpheus_tts import OrpheusModel

app = Flask(__name__)
engine = OrpheusModel(model_name="canopylabs/orpheus-tts-0.1-finetune-prod")

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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, threaded=True)
