from threading import Thread, Lock
from concurrent.futures import Future
from queue import Queue  
from utils import download_from_url, get_video_id
from caching import get_existing_transcript, save_transcript, check_exist_and_has_content, create_entry,update_transcript,get_all_incomplete_entries
import modal_transcribe
from sqlmodel import Session
import os
import gc

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
        print(f"Processing {url}")
        try:
            video_id = get_video_id(url)
            file_name_no_extension = f"{temp_folder}/{video_id}"
            file_name = f"{temp_folder}/{video_id}.wav"
            exist, has_content = check_exist_and_has_content(session, video_id)
            print(f"Processing {url} exists {exist} has content {has_content}")
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
                print("push to transcipt thread")
            else:
                # Entry exists but content is still being transcribed by another request.
                # Register future so the transcription worker resolves it when done.
                with futures_lock:
                    currently_processing_futures.append((video_id, future))
                    print("Push to currently processing")
        except Exception as exc:
            # Propagate exceptions to the caller instead of silently dropping them.
            future.set_exception(exc)


def transcription(session: Session):

    while True:
        video_id, file_name = transcription_queue.get()
        try:
            # ── Phase 1: resolve already-satisfied futures ──
            print("Start Transcripting")
            with futures_lock:
                still_pending = []
                for fid, future in currently_processing_futures:
                    exist, has_content = check_exist_and_has_content(session, fid)
                    print("In the damn loop")
                    if has_content:
                        id, res = get_existing_transcript(session, fid)
                        future.set_result(res)
                    else:
                        still_pending.append((fid, future))
                currently_processing_futures[:] = still_pending
            print(currently_processing_futures)

            # ── Phase 2: transcribe ──
            result = modal_transcribe.transcribe(file_name)
            update_transcript(session, video_id, result)

            # ── Phase 3: resolve waiters ──
            with futures_lock:
                remaining = []
                for fid, future in currently_processing_futures:
                    if fid == video_id:
                        future.set_result(result)
                    else:
                        remaining.append((fid, future))
                currently_processing_futures[:] = remaining

        except Exception as exc:
            print(exc)
            with futures_lock:
                remaining = []
                for fid, future in currently_processing_futures:
                    if fid == video_id:
                        future.set_exception(exc)
                    else:
                        remaining.append((fid, future))
                currently_processing_futures[:] = remaining

        finally:
            try:
                os.remove(file_name)
                print(f"[cleanup] Deleted {file_name}")
            except FileNotFoundError:
                pass

            session.expire_all()
            print(f"[cleanup] Done for {video_id}")
# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def recover_orphaned_entries(session: Session):
    """
    Called once at startup. Finds DB entries that exist but have no content
    (server was killed mid-transcription) and re-queues them for transcription.
    """
    
    orphaned = get_all_incomplete_entries(session)
    for video_id in orphaned:
        file_name = f"{temp_folder}/{video_id}.wav"
        if os.path.exists(file_name):
            # Audio file survived the crash — re-queue directly
            print(f"[recovery] Re-queuing orphaned transcription: {video_id}")
            transcription_queue.put((video_id, file_name))
        else:
            # Audio file was lost — we need to re-download
            # We can't recover the original URL from video_id alone,
            # so reconstruct the YouTube URL from the video_id
            url = f"https://www.youtube.com/watch?v={video_id}"
            file_name_no_extension = f"{temp_folder}/{video_id}"
            print(f"[recovery] Re-downloading orphaned entry: {video_id}")
            try:
                download_from_url(file_name_no_extension, url)
                transcription_queue.put((video_id, file_name))
            except Exception as e:
                print(f"[recovery] Failed to re-download {video_id}: {e}")


def start_workers(session: Session):
    """Initialise and start both background worker threads."""
    recover_orphaned_entries(session)  # <-- heal state before workers start
    
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