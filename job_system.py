from threading import Thread
from queue import Queue
import logging
import os
from utils import download_from_url, get_video_id
from caching import get_existing_transcript, save_transcript
from pipeline import TranscriptionPipeline
from uuid import uuid4

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TEMP_FOLDER = "temp"
os.makedirs(TEMP_FOLDER, exist_ok=True)

# Thread-safe queues (unlike plain lists)
download_queue: Queue = Queue()
transcription_queue: Queue = Queue()


def put_job(url: str):
    """Enqueue a YouTube URL for processing."""
    download_queue.put(url)
    logger.info(f"Queued: {url}")


def download_worker():
    """Continuously pulls URLs from download_queue, downloads, and enqueues for transcription."""
    while True:
        url = download_queue.get()          # blocks until an item is available
        if url is None:                     # poison pill — signals shutdown
            download_queue.task_done()
            break

        uid = uuid4()
        video_id = get_video_id(url)
        file_name = f"{TEMP_FOLDER}/{uid}"

        try:
            logger.info(f"Downloading {url}")
            download_from_url(file_name, url)
            transcription_queue.put((uid, video_id, file_name))
        except Exception as e:
            logger.error(f"Download failed for {url}: {e}")
        finally:
            download_queue.task_done()


def transcription_worker(session):
    """Pulls downloaded files from transcription_queue, checks cache, then transcribes."""
    pipeline = TranscriptionPipeline()

    while True:
        item = transcription_queue.get()    # blocks until an item is available
        if item is None:                    # poison pill — signals shutdown
            transcription_queue.task_done()
            break

        uid, video_id, file_name = item

        try:
            # Check whether we already have a transcript for this video
            existing = get_existing_transcript(session, video_id)
            if existing:
                logger.info(f"Cache hit for video {video_id}, skipping transcription")
                continue

            logger.info(f"Transcribing {file_name}")
            transcript = pipeline.run(file_name)
            save_transcript(session, video_id, transcript)
            logger.info(f"Saved transcript for video {video_id}")

        except Exception as e:
            logger.error(f"Transcription failed for {file_name}: {e}")
        finally:
            # Always clean up the temp file, even on error
            if os.path.exists(file_name):
                os.remove(file_name)
            transcription_queue.task_done()


def start_workers(session):
    """Spin up the download and transcription daemon threads."""
    download_thread = Thread(
        target=download_worker,
        daemon=True,
        name="download-worker",
    )
    transcription_thread = Thread(
        target=transcription_worker,
        args=(session,),
        daemon=True,
        name="transcription-worker",
    )

    download_thread.start()
    transcription_thread.start()

    return download_thread, transcription_thread


def stop_workers(download_thread, transcription_thread):
    """Gracefully drain the queues and stop both workers."""
    download_queue.join()           # wait for all downloads to complete
    download_queue.put(None)        # poison pill

    transcription_queue.join()      # wait for all transcriptions to complete
    transcription_queue.put(None)   # poison pill

    download_thread.join()
    transcription_thread.join()
    logger.info("All workers stopped")