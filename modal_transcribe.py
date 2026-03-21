import modal
import subprocess
import os
from pathlib import Path

# Look up deployed class
Transcriber = modal.Cls.from_name("whisperx_transcribe", "Transcriber")
transcriber = Transcriber()


def _preprocess_audio(file_path: str) -> str:
    """Convert audio to mono 16kHz PCM WAV — mirrors TranscriptionPipeline._preprocess_audio"""
    out_path = str(Path(file_path).with_suffix(".converted.wav"))
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", file_path,
            "-ar", "16000",
            "-ac", "1",
            "-c:a", "pcm_s16le",
            out_path,
        ],
        check=True,
        capture_output=True,
    )
    return out_path


def transcribe(file_path: str) -> list:
    converted_path = _preprocess_audio(file_path)
    try:
        # Read as bytes — no torch/whisperx needed locally
        audio_bytes = Path(converted_path).read_bytes()
        result = transcriber.transcribe.remote(audio_bytes)
    finally:
        os.remove(converted_path)
        os.remove(file_path)
    return result