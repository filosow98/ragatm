from typing_extensions import Self
from pathlib import Path
from datetime import datetime
import uuid
from pydantic import (
    BaseModel,
    Field,
    model_validator,
)


# TODO: Check out pydantic's validate_assignment


class MinimalSource(BaseModel):
    """A single source of information."""

    file_path: str
    first_character_index: int
    last_character_index: int

    @model_validator(mode="after")
    def validate_path_is_a_file(self) -> Self:
        """Check if path is a file."""
        path = Path(self.file_path)
        if not path.is_file():
            raise ValueError(
                "'MinimalSource' must use a valid path to a file."
                + f" Got '{self.file_path}'."
            )
        self.file_path = str(path)
        return self

    @model_validator(mode="after")
    def validate_source_length_is_less_than_2000(self) -> Self:
        """Length of a single source is capped at 2000 characters."""
        if self.last_character_index - self.first_character_index > 2000:
            raise ValueError(
                "MinimalSource contains more than 2000"
                + f" characters: '{self}'."
            )
        return self


class Source(MinimalSource):
    """A single source with additional computed data."""

    wordcount: int
    word_occurance: dict[str, int]


class SourceFile(BaseModel):
    """Path to a file with indexed sources."""

    file_path: str
    modification_timestamp: datetime

    def __hash__(self) -> int:
        return hash((self.file_path, self.modification_timestamp))

    @model_validator(mode="after")
    def validate_path_is_a_directory_and_make_it_absolute(self) -> Self:
        """Check if a path is a file."""
        if not Path(self.file_path).is_file():
            raise ValueError(
                "'SourceFile' model must be a file_path to a file."
                + f" Got '{self.file_path}'."
            )
        return self


class Sources(BaseModel):
    """All sources with additional computed data."""

    sources: dict[SourceFile, list[Source]]
    number_of_sources_with_word: dict[str, int]
    average_document_wordcount: float


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
