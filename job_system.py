from threading import Thread,Lock
from concurrent.futures import Future
from queue import Queue
from utils import download_from_url,get_video_id
from caching import get_existing_transcript,save_transcript,check_exist_and_has_content,create_entry
from pipeline import TranscriptionPipeline
from dataclasses import dataclass
from uuid import uuid4
from sqlmodel import Session

temp_folder = "temp"

download_queue = Queue()
transcription_queue = Queue()

currently_processing_futures = []

futures_lock = Lock()  # ← guards the shared list

def download_from_youtube(session:Session):
    while True:
        url,future = download_queue.get()
        id = get_video_id(url)
        file_name = f"{temp_folder}/{id}"
        download_from_url(file_name,url)
        exist, has_content =  check_exist_and_has_content(session, id)
        future_desc = (id,future)
        if not exist:
            create_entry(session,id)
            transcription_queue.put((id,file_name))
            currently_processing_futures.append(future_desc)
        elif not has_content:
            currently_processing_futures.append(future_desc)
        else:
            future.set_result(get_existing_transcript(session,id))




def transcription(session: Session):
    pipeline = TranscriptionPipeline()
    while True:
        id,file_name = transcription_queue.get()     
        for i, (id, future) in  enumerate(currently_processing_futures):
            exist, has_content =  check_exist_and_has_content(session, id)
            if has_content:
                future.set_result(get_existing_transcript(id))
                currently_processing_futures.pop(i)
        id,file_name = transcription_queue.get()
        result = pipeline.transcribe(file_name)
        save_transcript(session,id, result)
        for id, future in currently_processing_futures:
            future.set_result(result)
        currently_processing_futures.clear()




transcription_thread= Thread(target=transcription, daemon=True)
download_thread = Thread(target=download_from_youtube,daemon=True)


def transcribe(url):
    future = Future()
    download_queue.put((url,future))
    res = future.result()
    return res


