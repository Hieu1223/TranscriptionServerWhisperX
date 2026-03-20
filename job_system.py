from threading import Thread, Lock
from concurrent.futures import Future
from queue import Queue  # thread-safe queue, not asyncio.Queue
from utils import download_from_url, get_video_id
from caching import get_existing_transcript, save_transcript, check_exist_and_has_content, create_entry,update_transcript
from pipeline import TranscriptionPipeline
from sqlmodel import Session
import os

temp_folder = "temp"

download_queue: Queue = Queue()
transcription_queue: Queue = Queue()

# Each entry is (video_id: str, future: Future)
# Guards all reads and writes to this list
currently_processing_futures: list[tuple[str, Future]] = []
futures_lock = Lock()


def download_from_youtube(session: Session):
    """
    Blocking download worker.
    - Pulls (url, future) from download_queue.
    - If a transcript already exists, resolves the future immediately.
    - If the entry is missing entirely, creates it and enqueues transcription.
    - If the entry exists but has no content yet (another request is already
      transcribing it), registers the future so the transcription worker
      resolves it when done.
    """
    while True:
        url, future = download_queue.get()  # blocks until work arrives
        try:
            video_id = get_video_id(url)
            file_name_no_extension = f"{temp_folder}/{video_id}"
            file_name = f"{temp_folder}/{video_id}.wav"
            exist, has_content = check_exist_and_has_content(session, video_id)

            if has_content:
                # Transcript already ready — resolve immediately, no transcription needed.
                id, res = get_existing_transcript(session, video_id)
                future.set_result(res)
            elif not exist:
                # Brand-new video: create DB entry, queue transcription, register future.
                create_entry(session, video_id)
                download_from_url(file_name_no_extension, url)
                with futures_lock:
                    currently_processing_futures.append((video_id, future))
                transcription_queue.put((video_id, file_name))

            else:
                # Entry exists but content is still being transcribed by another request.
                # Register future so the transcription worker resolves it when done.
                with futures_lock:
                    currently_processing_futures.append((video_id, future))

        except Exception as exc:
            # Propagate exceptions to the caller instead of silently dropping them.
            future.set_exception(exc)


def transcription(session: Session):
    """
    Blocking transcription worker.
    - Before processing the next item, scans registered futures and resolves
      any whose content has already been saved (e.g. by a concurrent request).
    - Transcribes the audio file, persists the result, then resolves every
      future waiting on that specific video_id.
    """
    pipeline = TranscriptionPipeline()

    while True:
        video_id, file_name = transcription_queue.get()  # blocks until work arrives
        try:
            # ----------------------------------------------------------------
            # Phase 1: resolve any futures that are already satisfied
            # (covers the case where two identical URLs were queued back-to-back)
            # ----------------------------------------------------------------
            with futures_lock:
                still_pending = []
                for fid, future in currently_processing_futures:
                    _, has_content = check_exist_and_has_content(session, fid)
                    if has_content:
                        future.set_result(get_existing_transcript(session, fid))
                    else:
                        still_pending.append((fid, future))
                currently_processing_futures[:] = still_pending
            print(currently_processing_futures)
            # ----------------------------------------------------------------
            # Phase 2: transcribe the current item
            # ----------------------------------------------------------------
            result = pipeline.transcribe(file_name)
            update_transcript(session, video_id, result)

            # ----------------------------------------------------------------
            # Phase 3: resolve every future waiting on this video_id
            # ----------------------------------------------------------------
            with futures_lock:
                remaining = []
                for fid, future in currently_processing_futures:
                    if fid == video_id:
                        future.set_result(result)
                    else:
                        remaining.append((fid, future))
                currently_processing_futures[:] = remaining

        except Exception as exc:
            # If transcription fails, surface the error to all waiters for
            # this video so they are not blocked forever.
            with futures_lock:
                remaining = []
                for fid, future in currently_processing_futures:
                    if fid == video_id:
                        future.set_exception(exc)
                    else:
                        remaining.append((fid, future))
                currently_processing_futures[:] = remaining
        os.remove(file_name)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def start_workers(session: Session):
    """Initialise and start both background worker threads."""
    transcription_thread = Thread(
        target=transcription,
        args=(session,),
        daemon=True,
        name="transcription-worker",
    )
    download_thread = Thread(
        target=download_from_youtube,
        args=(session,),
        daemon=True,
        name="download-worker",
    )
    transcription_thread.start()
    download_thread.start()
    return download_thread, transcription_thread


def transcribe(url: str) -> dict:
    """
    Submit a URL for transcription and block until the result is ready.
    Thread-safe: can be called from any thread concurrently.
    """
    future: Future = Future()
    download_queue.put((url, future))
    return future.result()  # blocks the calling thread until resolved or raises