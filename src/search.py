from __future__ import annotations
import heapq
from itertools import chain
from ragatm.retrieval import get_proper_words, get_score_bm25
from pydantic import ValidationError
from pathlib import Path

from dataclasses import dataclass

from ragatm import Source, Sources


@dataclass
class ScoredSource:
    """Source with score."""

    score: float
    source: Source

    def __lt__(self, other: ScoredSource) -> bool:
        return self.score < other.score

    def __le__(self, other: ScoredSource) -> bool:
        return self.score <= other.score

    def __gt__(self, other: ScoredSource) -> bool:
        return self.score > other.score

    def __ge__(self, other: ScoredSource) -> bool:
        return self.score >= other.score

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ScoredSource):
            return self.score == other.score
        return False


def _search(query: str, k: int, index: Path) -> None:
    """Search for `k` most relevant sources.

    query : str
        Query question.
    k : int
        Number of sources to find.
    """

    try:
        with index.open("r") as f:
            content = f.read()
        sources = Sources.model_validate_json(content)
    except OSError as e:
        print(f"Error: Could not load index: {e}.")
        return
    except ValidationError as e:
        print(f"Error: Could not load index: {e}.")
        return

    best_matches = []

    best_matches = heapq.nlargest(
        k,
        chain.from_iterable([src.sources for src in sources.sources]),
        key=lambda q: get_score_bm25(
            query=query,
            word_occurance=q.word_occurance,
            document_wordcount=q.wordcount,
            average_document_wordcount=sources.average_document_wordcount,
            number_of_documents=sources.number_of_sources,
            number_of_documents_with_word=sources.number_of_sources_with_word,
        ),
    )

    for match in best_matches:
        print(
            f"{match.file_path} [{match.first_character_index}:"
            + f"{match.last_character_index}]"
        )
