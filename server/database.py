from typing import Optional

from config import CV_Config
from sqlmodel import SQLModel, create_engine, Field

settings = CV_Config()
engine = create_engine("sqlite:///" + settings.users_database)


class Users(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    username: str
    first_name: str
    last_name: str
    is_enabled: bool = Field(default=True)


class Photos(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    faiss_id: int
    user_id: int
    photo_path: str
    uploaded_at: int


SQLModel.metadata.create_all(engine)
