"""
modal_app.py  –  Single-container, multi-threaded Qwen3 pipeline.
                  Uses QwenPipeline class name and qwen_transcribe app name.
"""

import gc
import os
import re
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path

import modal



MODEL_NAME = "Qwen/Qwen3-ForcedAligner-0.6B"

def download_model_weights():
    from qwen_asr import Qwen3ForcedAligner
    aligner = Qwen3ForcedAligner.from_pretrained(
                MODEL_NAME, device_map="auto", torch_dtype="auto"
            )

# ──────────────────────────────────────────────
# Single Image containing both CPU & GPU tools
# ──────────────────────────────────────────────

pipeline_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "yt-dlp",
        "torch",
        "torchaudio",
        "transformers",
        "accelerate",
        "qwen-asr",
    )
    .run_function(download_model_weights)
)

app = modal.App("qwen_transcribe")

MODEL_NAME = "Qwen/Qwen3-ForcedAligner-0.6B"
LANGUAGE = "Japanese"
SUBTITLE_LANG = os.environ.get("SUBTITLE_LANG", "ja")
SENTENCES_PER_SEG = 1
PADDING_SECS = 1.5

# ──────────────────────────────────────────────
# Core Processing Helpers
# ──────────────────────────────────────────────

def _download(url: str, work_dir: str) -> tuple[str, str]:
    import yt_dlp

    proxy = (
        os.getenv("HTTPS_PROXY")
        or os.getenv("https_proxy")
        or os.getenv("HTTP_PROXY")
        or os.getenv("http_proxy")
        or os.getenv("ALL_PROXY")
    )

    ydl_opts = {
        "writesubtitles": False,
        "writeautomaticsub": True,
        "subtitleslangs": [SUBTITLE_LANG],
        "skip_download": False,
        "format": "bestaudio/best",
        "outtmpl": os.path.join(work_dir, "%(title)s.%(ext)s"),
        "quiet": True,
    }

    if proxy:
        ydl_opts["proxy"] = proxy

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    subtitle_files = list(Path(work_dir).glob("*.vtt"))
    if not subtitle_files:
        raise RuntimeError("No subtitle file found")

    audio_files = []
    for ext in ["webm", "m4a", "mp3", "opus"]:
        audio_files.extend(Path(work_dir).glob(f"*.{ext}"))

    if not audio_files:
        raise RuntimeError("No audio file found")

    return str(subtitle_files[0]), str(audio_files[0])


def _convert_audio(input_path: str, work_dir: str) -> str:
    out = os.path.join(work_dir, "audio.wav")
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", input_path,
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", out
        ],
        check=True, capture_output=True
    )
    return out


def _parse_time_string(ts: str) -> float:
    ts = ts.replace(",", ".")
    hms, ms = ts.split(".")
    h, m, s = map(int, hms.split(":"))
    return h * 3600 + m * 60 + s + int(ms) / 1000


def _parse_vtt(vtt_path: str) -> list[dict]:
    with open(vtt_path, encoding="utf-8") as f:
        lines = [l.rstrip() for l in f]

    raw_entries = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        if not line or line.startswith(("WEBVTT", "Kind:", "Language:", "NOTE")):
            i += 1
            continue

        if "-->" in line:
            timing_part = line.split()[0:3] 
            timing_line = " ".join(timing_part)
            
            parts = timing_line.split(" --> ")
            start = _parse_time_string(parts[0].strip())
            end = _parse_time_string(parts[1].strip())
            i += 1

            text_lines = []
            while i < len(lines) and not ("-->" in lines[i] or lines[i].strip().startswith(("WEBVTT", "Kind:", "Language:", "NOTE"))):
                text = re.sub(r"<[^>]+>", "", lines[i].strip())
                text = re.sub(r"\s+", " ", text).strip()
                if text and text != "\u200b": 
                    text_lines.append(text)
                i += 1

            final = " ".join(text_lines).strip()
            if final:
                raw_entries.append({"start": start, "end": end, "text": final})
        else:
            i += 1
            
    # ─── DE-DUPLICATION ENGINE FOR PROGRESSIVE CAPTIONS ───
    cleaned_entries = []
    for entry in raw_entries:
        if not cleaned_entries:
            cleaned_entries.append(entry)
            continue
            
        prev = cleaned_entries[-1]
        
        # Scenario A: Exact exact clone text or rolling subtitle block duplication
        if entry["text"] == prev["text"]:
            # Stretch the duration of the original block to cover the timeline step
            prev["end"] = max(prev["end"], entry["end"])
            continue
            
        # Scenario B: Progressive block build-up (e.g., 'はい、どうぞ。' -> 'はい、どうぞ。このスーパー今日は...')
        if prev["text"] in entry["text"] and (entry["start"] - prev["end"]) <= 0.5:
            # Update the old entry with the complete phrase and expand its end timestamp
            prev["text"] = entry["text"]
            prev["end"] = max(prev["end"], entry["end"])
            continue

        # Scenario C: Clean, new distinct phrase break
        cleaned_entries.append(entry)
        
    return cleaned_entries


def _group_entries(entries: list[dict], n: int) -> list[dict]:
    groups = []
    for i in range(0, len(entries), n):
        chunk = entries[i:i + n]
        groups.append({
            "start": chunk[0]["start"],
            "end": chunk[-1]["end"],
            "text": "".join([e["text"] for e in chunk]),
            "entries": chunk,
        })
    return groups


def _cut_chunk(wav: str, out: str, start: float, duration: float) -> None:
    os.makedirs(os.path.dirname(out), exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-ss", str(start), "-t", str(duration), "-i", wav, out],
        check=True, capture_output=True
    )


def _allocate_words_sequentially(words: list[dict], entries: list[dict]) -> list[list[dict]]:
    """
    Allocates tokens to entry buckets by comparing clean, punctuation-free character lengths.
    """
    result = []
    word_idx = 0
    num_words = len(words)
    
    # Simple regex to strip out Japanese and English structural punctuation marks
    punctuation_cleaner = re.compile(r"[、。！？・\s\.,!\?_\-~～]")

    for entry in entries:
        bucket = []
        # Calculate exactly how many actual spoken characters are in this text segment
        clean_entry_text = punctuation_cleaner.sub("", entry["text"])
        target_length = len(clean_entry_text)
        
        accumulated_clean_len = 0
        
        while word_idx < num_words:
            w = words[word_idx]
            bucket.append(w)
            
            # Count only the clean characters found inside the token
            clean_token = punctuation_cleaner.sub("", w["token"])
            accumulated_clean_len += len(clean_token)
            
            word_idx += 1
            
            # Break out of the loop once we satisfy the spoken character length target
            if accumulated_clean_len >= target_length:
                break
                
        result.append(bucket)
        
    # Safety Valve: Sweep any lingering orphan words cleanly into the last entry segment
    while word_idx < num_words:
        if result:
            result[-1].append(words[word_idx])
        word_idx += 1
        
    return result

# ──────────────────────────────────────────────
# Single Persistent Container Pipeline Class
# ──────────────────────────────────────────────
@app.cls(
        image=pipeline_image,
        gpu="T4",
        secrets=[modal.Secret.from_name("proxy")],
)
class QwenPipeline:
    @modal.enter()
    def start_model_initialization(self):
        self.aligner = None
        self.model_error = None
        
        self.init_thread = threading.Thread(target=self._bg_load_model, daemon=True)
        self.init_thread.start()

    def _bg_load_model(self):
        try:
            print("[Thread 1/2] Loading Qwen3 Forced Aligner into GPU memory...")
            from qwen_asr import Qwen3ForcedAligner
            self.aligner = Qwen3ForcedAligner.from_pretrained(
                MODEL_NAME, device_map="auto", torch_dtype="auto"
            )
            print("[Thread 1/2] Model initialization complete.")
        except Exception as e:
            self.model_error = e

    @modal.method()
    def transcribe_url(self, url: str) -> dict:
        import torch
        print(f"[Thread 2/2] Fetching media via yt-dlp from: {url}")
        work_dir = tempfile.mkdtemp()
        temp_chunk_dir = None
        
        try:
            subtitle_path, audio_path = _download(url, work_dir)
            wav_path = _convert_audio(audio_path, work_dir)
            
            self.init_thread.join()
            if self.model_error or self.aligner is None:
                raise RuntimeError("Model initialization failed.")

            entries = _parse_vtt(subtitle_path)
            temp_chunk_dir = tempfile.mkdtemp()
            
            # --- PREPARE BATCH TASKS ---
            tasks = []
            i = 0
            num_entries = len(entries)
            
            while i < num_entries:
                batch_entries = [entries[i]]
                start_time = entries[i]["start"]
                end_time = entries[i]["end"]
                
                idx = i + 1
                while idx < num_entries:
                    next_entry = entries[idx]
                    if (next_entry["end"] - start_time) <= 12.0:
                        batch_entries.append(next_entry)
                        end_time = next_entry["end"]
                        idx += 1
                    else:
                        break
                i = idx 

                batch_text = "".join([e["text"] for e in batch_entries])
                clip_start = max(0.0, start_time - PADDING_SECS)
                clip_end = end_time + PADDING_SECS
                
                tasks.append({
                    "batch_entries": batch_entries,
                    "batch_text": batch_text,
                    "clip_start": clip_start,
                    "duration": clip_end - clip_start
                })

            all_segments = [None] * len(tasks)
            lock = threading.Lock()

            # --- PARALLEL WORKER ENGINE ---
            def worker(task_idx, task):
                if task["duration"] <= 0 or not task["batch_text"].strip():
                    return

                chunk_path = os.path.join(temp_chunk_dir, f"parallel_{task_idx}.wav")
                _cut_chunk(wav_path, chunk_path, task["clip_start"], task["duration"])

                if not os.path.exists(chunk_path) or os.path.getsize(chunk_path) == 0:
                    return

                # Multiple threads call this at the exact same time, filling VRAM
                with torch.inference_mode():
                    result = self.aligner.align(
                        audio=chunk_path, text=task["batch_text"], language=LANGUAGE
                    )

                if os.path.exists(chunk_path):
                    os.remove(chunk_path)

                flat_words = []
                if result and len(result) > 0:
                    items = result[0].items
                    for item in items:
                        flat_words.append({
                            "token": item.text,
                            "start": round(float(item.start_time) + task["clip_start"], 3),
                            "end": round(float(item.end_time) + task["clip_start"], 3),
                        })

                entry_segments = _allocate_words_sequentially(flat_words, task["batch_entries"])
                
                segment_outputs = []
                for words, original_entry in zip(entry_segments, task["batch_entries"]):
                    segment_outputs.append({
                        "text": original_entry["text"],
                        "words": words,
                    })

                with lock:
                    all_segments[task_idx] = segment_outputs

            from concurrent.futures import ThreadPoolExecutor
            
            # 8 parallel threads is optimal for soaking up a 16GB T4 without OOM (Out of Memory) crash
            with ThreadPoolExecutor(max_workers=16) as executor:
                executor.map(lambda arg: worker(*arg), enumerate(tasks))

            # Flatten compiled sequence list
            final_output = []
            for out in all_segments:
                if out:
                    final_output.extend(out)

            return {"segments": final_output}

        finally:
            shutil.rmtree(work_dir, ignore_errors=True)
            if temp_chunk_dir:
                shutil.rmtree(temp_chunk_dir, ignore_errors=True)
            gc.collect()
            torch.cuda.empty_cache()