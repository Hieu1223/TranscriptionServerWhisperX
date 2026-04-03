from pydantic import BaseModel
from uuid import UUID

class TranscriptionToken(BaseModel):
    start: float | None
    end: float | None
    token: str



class TokenTimestamp(BaseModel):
    start: float | None
    end: float | None
    token: str

class TranscriptSegment(BaseModel):
    text : str
    words: list[TokenTimestamp]

class TranscriptResult(BaseModel):
    segments : list[TranscriptSegment]


class YouTubeTranscriptionRequest(BaseModel):
    url: str

class YoutubeTranscriptionResponse(BaseModel):
    transcript: TranscriptResult