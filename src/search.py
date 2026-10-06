from __future__ import annotations

import heapq
from functools import cache
from itertools import chain
from pathlib import Path
from typing import cast

from pydantic import ValidationError

from rag import Source, Sources, get_score_bm25

sources: None | Sources = None


def search_inner(
    query: str, k: int, index: str, extensions: str
) -> list[Source]:
    """Search for `k` most relevant sources.

    Parameters
    ----------
    query : str
        Query question.
    k : int
        Number of sources to find.
    index : str
        Path to a sources index.
    extensions : str
        File extensions separated by space to filter files for indexing.

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
    if not isinstance(extensions, str):
        raise TypeError(
            "'extensions' must be a string of file extensions separated by "
            + "commas."
            + f" Got {extensions}."
        )
    extensions_list: list[str] = []
    for ext in extensions.split():
        if not ext.startswith("."):
            ext = "." + ext
        extensions_list.append(ext)

    global sources
    if sources is None:
        index_path: Path = Path(index)
        try:
            with index_path.open("r") as f:
                content = f.read()
            sources = Sources.model_validate_json(content)
        except OSError as e:
            raise OSError(f"Could not load index: {e}.")
        except ValidationError:
            raise ValueError(f"Could not parse {index!s} file.")

    srcs = cast(Sources, sources)  # fuck mypy

    @cache
    def score(s: Source) -> float:
        """Get score from source."""

        return get_score_bm25(
            query=query,
            word_occurance=s.word_occurance,
            document_wordcount=s.wordcount,
            average_document_wordcount=srcs.average_document_wordcount,
            number_of_documents=srcs.number_of_sources,
            number_of_documents_with_word=srcs.number_of_sources_with_word,
        )

    best_matches = heapq.nlargest(
        k,
        chain.from_iterable(
            [
                src.sources
                for src in srcs.sources
                if Path(src.file_path).suffix in extensions_list
            ]
        ),
        key=score,
    )

    return best_matches
