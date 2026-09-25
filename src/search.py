from __future__ import annotations

import heapq
from itertools import chain
from pathlib import Path

from pydantic import ValidationError

from ragatm import Sources, get_score_bm25


def _search(query: str, k: int, index: str) -> None:
    """Search for `k` most relevant sources.

    Parameters
    ----------
    query : str
        Query question.
    k : int
        Number of sources to find.
    index : str
        Path to a sources index.
    """

    if not isinstance(index, str):
        raise TypeError(
            "'index' must be a valid path." + f" Got index={index}."
        )
    if not isinstance(query, str):
        raise TypeError(
            "'query' must be a valid string." + f" Got query={query}."
        )
    if not isinstance(k, int) or k < 1:
        raise ValueError(
            "'k' must be an integer greater than 0." + f" Got k={k}."
        )
    index: Path = Path(index)
    try:
        with index.open("r") as f:
            content = f.read()
        sources = Sources.model_validate_json(content)
    except OSError as e:
        raise OSError(f"Could not load index: {e}.")
    except ValidationError as _:
        raise ValueError(f"Could not parse {index!s} file.")

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
