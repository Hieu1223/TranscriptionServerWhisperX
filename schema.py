from pydantic import BaseModel
from uuid import UUID

class TranscriptionToken(BaseModel):
    start: float | None
    end: float | None
    token: str


class YouTubeTranscriptionRequest(BaseModel):
    url: str

class YoutubeTranscriptionResponse(BaseModel):
    transcript: list[list[TranscriptionToken]]
    
class TranscriptResponseV2(BaseModel):
    transcript: list[list[TranscriptionToken]] | None
    done: bool
    detail: str