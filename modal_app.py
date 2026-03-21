import modal
from pipeline import TranscriptionPipeline


app = modal.App("whisperx_transcribe")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04", add_python="3.10"
    )
    .apt_install("ffmpeg", "git")
    .pip_install(
        "whisperx==3.8.2",          # install everything whisperx needs
        "pyannote.audio",
    )
    .pip_install(
        # ✅ reinstall torch last to override whatever whisperx pulled in
        "torch==2.8.0+cu126",
        "torchaudio==2.8.0+cu126",
        extra_index_url="https://download.pytorch.org/whl/cu126",
    )
    .add_local_file("pipeline.py", "/root/pipeline.py")
)

@app.cls(gpu='a10g', image=image)
class Transcriber:
    @modal.enter()
    def load_pipeline(self):
        from pipeline import TranscriptionPipeline
        import torch
        import ctranslate2
        self.pipeline = TranscriptionPipeline()
        # Torch check
        assert torch.cuda.is_available(), "CUDA not available!"
        print(f"Torch: {torch.__version__}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

        # CTranslate2 check
        print(f"CT2 CUDA types: {ctranslate2.get_supported_compute_types('cuda')}")
    @modal.method()
    def transcribe(self, audio_bytes: bytes) -> list:
        import tempfile
        import os

        # Write bytes to temp file, let pipeline handle the rest
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(audio_bytes)
            tmp_path = f.name
        try:
            result = self.pipeline.transcribe(tmp_path)  # ✅ uses full pipeline including ffmpeg preprocess
        finally:
            os.remove(tmp_path)

        return result