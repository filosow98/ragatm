import heapq
import os
from functools import cache, partial
from itertools import chain
from multiprocessing import Pool
from pathlib import Path
from typing import Literal

import tqdm
from pydantic import ValidationError

from rag import (
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    Source,
    Sources,
    StudentSearchResults,
    UnansweredQuestion,
    get_score_bm25,
)


def _search_sources_for_question(
    question: UnansweredQuestion,
    sources: Sources,
    k: int,
    extensions: list[str],
) -> MinimalSearchResults:
    """Search k sources for the question.

    Parameters
    ----------
    question : UnansweredQuestion
        Model with a question.
    sources : Sources
        Model with indexed sources.
    k : int
        Number of sources requested.
    extensions : list[str]
        List of file extensions to filter sources.
    """

    @cache
    def score(s: Source) -> float:
        """Get score from source."""

        return get_score_bm25(
            query=question.question,
            word_occurance=s.word_occurance,
            document_wordcount=s.wordcount,
            average_document_wordcount=sources.average_document_wordcount,
            number_of_documents=sources.number_of_sources,
            number_of_documents_with_word=sources.number_of_sources_with_word,
        )

    best_matches = heapq.nlargest(
        k,
        chain.from_iterable(
            [
                src.sources
                for src in sources.sources
                if Path(src.file_path).suffix in extensions
            ]
        ),
        key=score,
    )
    best_matches.sort(
        key=score,
        reverse=True,
    )

    minimal_sources = [
        MinimalSource(
            file_path=source.file_path,
            first_character_index=source.first_character_index,
            last_character_index=source.last_character_index,
        )
        for source in best_matches
    ]

    return MinimalSearchResults(
        question_id=question.question_id,
        question=question.question,
        retrieved_sources=minimal_sources,
    )


def search_dataset_inner(
    dataset_path: str,
    k: int,
    save_directory: str,
    index: str,
    processes: Literal["max", "none"] | int,
    process_chunk_size: int,
    extensions: str,
) -> None:
    """Run search on a dataset for `k` most relevant sources.

    Parameters
    ----------
    dataset_path : str
        Path to a json with a question dataset.
    k : int
        Number of sources to find.
    save_directory : str
        Directory where the answer json will be saved.
    index : str
        Path to a sources index.
    processes: int | Literal["max", "none"], default='none'
        Number of processes to use for serching. `none` if no processes
        should be used.
    process_chunk_size: int, default=50
        Chunksize of the process pool.
    extensions : str
        File extensions separated by space to filter files for searching.
    """

    if not isinstance(index, str):
        raise TypeError(
            "'index' must be a valid path." + f" Got index={index}."
        )
    if not isinstance(dataset_path, str):
        raise TypeError(
            "'dataset_path' must be a valid path."
            + f" Got dataset_path={dataset_path}."
        )
    dataset_path_path: Path = Path(dataset_path)
    if not dataset_path_path.is_file():
        raise ValueError(
            f"'dataset_path' must be a file. Got dataset_path={dataset_path}"
        )
    if not isinstance(save_directory, str):
        raise TypeError(
            "'save_directory' must be a valid path."
            + f" Got save_directory={save_directory}."
        )

    save_directory_path: Path = Path(save_directory)
    if save_directory_path.is_file():
        raise ValueError(
            "'save_directory' must be a directory."
            + f" Got save_directory={save_directory}"
        )
    if not isinstance(k, int) or k < 1:
        raise ValueError(
            "'k' must be an integer greater than 0." + f" Got k={k}."
        )
    if not isinstance(processes, int):
        if processes not in ["none", "max"]:
            raise ValueError(
                f"'processes' does not support '{processes}' value."
            )
    elif processes < 1:
        raise ValueError(
            f"'processes' cannot be set to less than 1. Got {processes}."
        )
    if not isinstance(process_chunk_size, int) or process_chunk_size < 1:
        raise ValueError(
            "'process_chunk_size' must be an integer greater than 0."
            + f" Got {process_chunk_size}."
        )

    if processes == "max":
        processes = os.cpu_count() or 1

    if processes == "none":
        processes = 0

    index_path: Path = Path(index)
    try:
        with index_path.open("r") as f:
            index_content = f.read()
        sources = Sources.model_validate_json(index_content)
    except OSError as e:
        raise OSError(f"Could not load index: {e}.")
    except ValidationError:
        raise ValueError(f"Could not parse {index!s} file.")

    try:
        with dataset_path_path.open("r") as f:
            dataset_content = f.read()
        dataset = RagDataset.model_validate_json(dataset_content)
    except OSError as e:
        raise OSError(f"Could not load dataset: {e}.")
    except ValidationError:
        raise ValueError(f"Could not parse {dataset_path!s} file.")

    if not isinstance(extensions, str):
        raise TypeError(
            "'extensions' must be a string of file extensions separated by "
            + "spaces."
            + f" Got {extensions}."
        )

    extensions_list: list[str] = []
    for ext in extensions.split():
        if not ext.startswith("."):
            ext = "." + ext
        extensions_list.append(ext)

    func = partial(
        _search_sources_for_question,
        sources=sources,
        k=k,
        extensions=extensions_list,
    )
    q_count = len(dataset.rag_questions)
    if processes > 0:
        with Pool(processes) as p:
            search_results = [
                q
                for q in tqdm.tqdm(
                    p.imap(
                        func,
                        dataset.rag_questions,
                        chunksize=process_chunk_size,
                    ),
                    total=q_count,
                    desc="Searching sources",
                )
            ]
    else:
        search_results = [
            func(q)
            for q in tqdm.tqdm(
                dataset.rag_questions,
                total=q_count,
                desc="Searching sources",
            )
        ]

    try:
        student_search_results = StudentSearchResults(
            search_results=search_results, k=k
        )
        result_json = student_search_results.model_dump_json(indent=4)
        os.makedirs(save_directory, exist_ok=True)
        save_path = save_directory_path / dataset_path_path.name
        with save_path.open("w") as f:
            f.write(result_json)
        print(f"Search results saved to: {save_path}")
    except OSError as e:
        raise OSError(f"Could not save json to file: {e}.")
