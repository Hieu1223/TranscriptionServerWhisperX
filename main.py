from typing import Annotated
from fastapi import FastAPI,Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transcription import transcribe_from_youtube
from schema import YoutubeTranscriptionResponse,YouTubeTranscriptionRequest
from caching import create_db_and_tables,SessionDep
from fastapi.staticfiles import StaticFiles
app = FastAPI()

app.mount("/static", StaticFiles(directory="temp"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def on_startup():
    create_db_and_tables()
    



@app.get("/transcribe_from_youtube")
def get_transcribe_from_youtube(url: str,session: SessionDep) -> YoutubeTranscriptionResponse:
    result = transcribe_from_youtube(url,session)
    return result

@app.get("/ping")
async def ping():
    return {"status": "ok"}