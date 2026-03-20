import torch
import whisperx
import gc
import os
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
            language='ja'
        )
        
        self.model_a = None
        self.metadata = None
        self.current_align_lang = None

    def transcribe(self, file_path: str, batch_size: int = 4) -> list[list[dict]]:
        audio = whisperx.load_audio(file_path)
        language = 'ja'
        result = self.model.transcribe(audio, batch_size=batch_size)
        
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
            return_char_alignments=False
        )

        all_lines = []
        for segment in aligned_result["segments"]:
            line_data = []
            
            # WhisperX returns 'words' list after alignment
            words = segment.get("words", [])
            
            for w in words:
                # CRITICAL: Mapping 'word' -> 'token' to match your BaseModel
                # Also casting np.float64 to float for Pydantic validation
                line_data.append({
                    "token": str(w.get("word", "")),
                    "start": w.get("start") if w.get("start") is not None else None,
                    "end": w.get("end") if w.get("end") is not None else None
                })
            
            if line_data:
                all_lines.append(line_data)

        return all_lines