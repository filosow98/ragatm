from __future__ import annotations

import heapq
from itertools import chain
from pathlib import Path

from pydantic import ValidationError

from ragatm import Source, Sources, get_score_bm25

sources: None | Sources = None


def _search(query: str, k: int, index: str) -> list[Source]:
    """Search for `k` most relevant sources.

    Parameters
    ----------
    query : str
        Query question.
    k : int
        Number of sources to find.
    index : str
        Path to a sources index.

    Returns
    -------
    list[Source]
        List of k sources relevant to query.
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
    global sources
    if not sources:
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

    return best_matches
