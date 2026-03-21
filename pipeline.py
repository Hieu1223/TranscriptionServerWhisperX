import torch
import whisperx
import gc
import os
import subprocess
from pathlib import Path


class TranscriptionPipeline:
    def __init__(
        self,
        model_size: str = "large-v3-turbo",
        device: str = "cuda",
        compute_type: str = "float16",  # ✅ int8 causes CPU fallback on some CT2 builds
    ) -> None:
        self.device = device
        self.compute_type = compute_type

        print(f"Initializing WhisperX {model_size}...")
        self.model = whisperx.load_model(
            model_size,
            device,
            compute_type=self.compute_type,
            language="ja",
        )

        self.model_a = None
        self.metadata = None
        self.current_align_lang = None
        print(f"Initializing WhisperX {model_size}... Done")

    @staticmethod  # ✅ was missing self, must be staticmethod
    def _preprocess_audio(file_path: str) -> str:
        """Convert audio to mono 16kHz PCM WAV to prevent resampling drift."""
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

    @staticmethod  # ✅ was missing self, must be staticmethod
    def load_audio(file_path: str):
        converted_path = TranscriptionPipeline._preprocess_audio(file_path)
        audio = whisperx.load_audio(converted_path)
        return audio, converted_path

    def transcribe_with_tensor(self, audio, batch_size: int = 16) -> list[list[dict]]:  # ✅ increased batch_size
        language = "ja"
        print("Transcribing starts")
        result = self.model.transcribe(audio, batch_size=batch_size, chunk_size=10)

        print("Aligning")
        if self.model_a is None or self.current_align_lang != language:
            self.model_a, self.metadata = whisperx.load_align_model(
                language_code=language, device=self.device
            )
            self.current_align_lang = language

        aligned_result = whisperx.align(
            result["segments"],
            self.model_a,
            self.metadata,
            audio,
            self.device,
            return_char_alignments=False,
        )

        print("Align Done")

        all_lines = []
        for segment in aligned_result["segments"]:
            line_data = []
            words = segment.get("words", [])

            for w in words:
                start = w.get("start")
                end = w.get("end")
                if start is None or end is None:
                    continue

                line_data.append({
                    "token": str(w.get("word", "")),
                    "start": float(start),
                    "end": float(end),
                })

            if line_data:
                all_lines.append(line_data)

        print("Transcribing ends")

        del audio
        del aligned_result
        del result
        gc.collect()

        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        return all_lines

    def transcribe(self, file_path: str, batch_size: int = 32) -> list[list[dict]]:
        audio, converted_path = TranscriptionPipeline.load_audio(file_path)  # ✅ fixed order (was reversed)
        try:
            result = self.transcribe_with_tensor(audio, batch_size)
        finally:
            os.remove(converted_path)  # ✅ always cleanup even if transcribe fails
        return result