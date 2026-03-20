from typing import Annotated,Dict,Any
from sqlmodel import Field, Session, SQLModel, create_engine, select
from datetime import datetime,timezone
from fastapi import Depends, FastAPI, HTTPException, Query
import json
from schema import YoutubeTranscriptionResponse
from uuid import UUID,uuid4

class Transcript(SQLModel, table=True):
    __table_args__ = {"extend_existing": True}
    id:  UUID | None = Field(default=None, primary_key=True)
    url: str = Field(index=True, unique=True)
    date_created: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    data: str = Field(default="")

sqlite_file_name = "database.db"
sqlite_url = f"sqlite:///{sqlite_file_name}"

connect_args = {"check_same_thread": False}
engine = create_engine(sqlite_url, connect_args=connect_args)

def create_db_and_tables():
    SQLModel.metadata.create_all(engine)



def get_session():
    with Session(engine) as session:
        yield session


SessionDep = Annotated[Session, Depends(get_session)]

def get_existing_transcript(session: Session, url: str) -> dict:
    """
    Queries the database for a Transcript with a matching URL.
    Returns the Transcript object if found, otherwise returns None.
    """
    statement = select(Transcript).where(Transcript.url == url)
    results = session.exec(statement)
    res =  results.first()
    if res:
        return res.id, json.loads(res.data)
    return None,None

def save_transcript(session: Session,id : UUID, url: str, data: dict):
    new_record = Transcript(id = id,url=url, data=json.dumps(data))
    session.add(new_record)
    session.commit()
    