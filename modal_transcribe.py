import modal
import torchaudio
import numpy as np
from pipeline import TranscriptionPipeline
import os
# Look up deployed class
Transcriber = modal.Cls.from_name("whisperx_transcribe", "Transcriber")
transcriber = Transcriber()


def transcribe(file_path):
    audio,converted_path = TranscriptionPipeline.load_audio("a.wav")
    # Call remotely — blocks until result
    result = transcriber.transcribe.remote(audio)
    os.remove(converted_path)
    return result