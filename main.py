

from dotenv import load_dotenv
load_dotenv()


from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from schema import YoutubeTranscriptionResponse
from caching import create_db_and_tables, SessionDep,engine
from multithread_transcribe import *
import asyncio

import os
os.makedirs("temp", exist_ok=True)
# ---------------------------------------------------------------------------
# Lifespan: replaces the deprecated @app.on_event("startup")
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs once on startup (before the `yield`) and once on shutdown (after).
    Creates DB tables and starts the background download/transcription workers.
    """
    create_db_and_tables()
    with Session(engine) as session:
        start_workers(session)   # spins up daemon threads; see pipeline_worker.py
    yield
    # add any shutdown/cleanup logic here if needed


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(lifespan=lifespan)

app.mount("/static", StaticFiles(directory="temp"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/ping")
async def ping():
    return {"status": "ok"}




@app.get("/transcribe_from_youtube")
async def get_transcribe(
    url: str,
    session: SessionDep,
) -> YoutubeTranscriptionResponse:
    """
    Async route that delegates to the pipeline worker (download queue →
    transcription queue) introduced in pipeline_worker.py.

    `pipeline_worker.transcribe` calls `future.result()` which blocks the
    calling thread, so we offload it to the thread-pool executor with
    `asyncio.to_thread` to avoid blocking the event loop.
    """

    try:
        result = await asyncio.to_thread(transcribe, url)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return YoutubeTranscriptionResponse(transcript= result)