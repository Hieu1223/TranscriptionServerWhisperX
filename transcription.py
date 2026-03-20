from fastapi.responses import FileResponse
from utils import download_from_url
from pipeline import TranscriptionPipeline
from caching import create_db_and_tables,SessionDep,save_transcript,get_existing_transcript
from uuid import uuid4
import json
import gc
import torch

pipeline = TranscriptionPipeline()
temp_folder = "temp"

def transcribe_from_youtube(url: str,session:SessionDep) -> dict:
    id, existing = get_existing_transcript(session,url)
    if existing:
        return {
            'transcript' : existing,
        }
    id = uuid4()
    download_from_url(f"{temp_folder}/{id}", url)
    result = pipeline.transcribe(f'temp/{id}')
    save_transcript(session,id,url,result)
    gc.collect()
    torch.cuda.empty_cache()
    return {
        'transcript': result,
    }

