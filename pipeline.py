import torch
import whisperx
import gc
import os
import subprocess
from pathlib import Path

# Prevent the 'libiomp5md.dll' kernel crash
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class TranscriptionPipeline:
    def __init__(
        self,
        model_size: str = "large-v3-turbo",
        device: str = "cuda",
        compute_type: str = "int8",
    ) -> None:
        cuda_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin"
        if os.path.exists(cuda_bin):
            try:
                os.add_dll_directory(cuda_bin)
            except Exception:
                pass

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

    def _preprocess_audio(self, file_path: str) -> str:
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

    def transcribe(self, file_path: str, batch_size: int = 16) -> list[list[dict]]:
        # Pre-convert to clean mono 16kHz WAV — eliminates timestamp drift
        converted_path = self._preprocess_audio(file_path)
        audio = whisperx.load_audio(converted_path)
        language = "ja"

        # chunk_size=10 reduces VAD boundary errors on Japanese (no word spaces)
        result = self.model.transcribe(audio, batch_size=batch_size, chunk_size=10)

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

        all_lines = []
        for segment in aligned_result["segments"]:
            line_data = []
            words = segment.get("words", [])

            for w in words:
                start = w.get("start")
                end = w.get("end")

                # Skip words with missing timestamps — they bleed into silence
                if start is None or end is None:
                    continue

                line_data.append({
                    "token": str(w.get("word", "")),
                    "start": float(start),
                    "end": float(end),
                })

            if line_data:
                all_lines.append(line_data)

        return all_lines