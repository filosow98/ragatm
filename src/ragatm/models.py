from datetime import datetime
import uuid
from pydantic import BaseModel, Field


# TODO: Check out pydantic's validate_assignment


class MinimalSource(BaseModel):
    """A single source of information."""

    file_path: str
    first_character_index: int
    last_character_index: int


class Source(MinimalSource):
    """A single source with additional computed data."""

    modification_timestamp: datetime
    wordcount: int
    word_occurance: dict[str, int]


class Sources(BaseModel):
    """All sources with additional computed data."""

    sources: list[Source]
    number_of_sources_with_word: dict[str, int]
    average_document_length: float


class UnansweredQuestion(BaseModel):
    """A single unanswered question."""

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """A single answered question."""

    sources: list[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """Dataset of RAG questions."""

    rag_questions: list[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """Search result of a single question."""

    question_id: str
    question: str
    retrieved_sources: list[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Answer of a single question."""

    answer: str


class StudentSearchResults(BaseModel):
    """List of the search results."""

    search_results: list[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """List of the search results with an answer."""

    search_results: list[MinimalAnswer]
    k: int
