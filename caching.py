from typing import Annotated,Dict,Any
from sqlmodel import Field, Session, SQLModel, create_engine, select
from datetime import datetime,timezone
from fastapi import Depends, FastAPI, HTTPException, Query
import json
from schema import YoutubeTranscriptionResponse
from uuid import UUID,uuid4

class Transcript(SQLModel, table=True):
    __table_args__ = {"extend_existing": True}
    id: str = Field(primary_key=True)
    date_created: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    data: str = Field(default="")

DATABASE_URL = "postgresql+psycopg2://postgres:1@localhost:5432/transcript_server"

engine = create_engine(DATABASE_URL)

def create_db_and_tables():
    SQLModel.metadata.create_all(engine)



def get_session():
    with Session(engine) as session:
        yield session


SessionDep = Annotated[Session, Depends(get_session)]

def check_exist_and_has_content(session:Session,id:str):
    statement = select(Transcript).where(Transcript.id == id)
    transcript = session.exec(statement).first()

    if transcript is None:
        return False,False

    return True , transcript.data != ""

def create_entry(session:Session ,id:str):
    transcript = Transcript(id=id)
    session.add(transcript)
    session.commit()


def get_existing_transcript(session: Session, id: str) -> dict:
    """
    Queries the database for a Transcript with a matching URL.
    Returns the Transcript object if found, otherwise returns None.
    """
    statement = select(Transcript).where(Transcript.id == id)
    results = session.exec(statement)
    res =  results.first()
    if res:
        return res.id, json.loads(res.data)
    return None,None

def save_transcript(session: Session,id :str, data: dict):
    new_record = Transcript(id = id, data=json.dumps(data))
    session.add(new_record)
    session.commit()
    
def update_transcript(session: Session, id: str, data: dict):
    record = session.get(Transcript, id)
    if record is None:
        raise ValueError(f"Transcript '{id}' not found")
    record.data = json.dumps(data)
    session.add(record)
    session.commit()

def get_all_incomplete_entries(session: Session) -> list[str]:
    """Return video_ids that exist in the DB but have no transcript content."""
    results = session.exec(
        select(Transcript.id).where(Transcript.data == "")
    ).all()
    return results