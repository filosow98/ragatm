import heapq
import os
from functools import partial
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
    Sources,
    StudentSearchResults,
    UnansweredQuestion,
    get_score_bm25,
)


def _search_sources_for_question(
    question: UnansweredQuestion, sources: Sources, k: int
):
    """"""

    best_matches = heapq.nlargest(
        k,
        chain.from_iterable([src.sources for src in sources.sources]),
        key=lambda q: get_score_bm25(
            query=question.question,
            word_occurance=q.word_occurance,
            document_wordcount=q.wordcount,
            average_document_wordcount=sources.average_document_wordcount,
            number_of_documents=sources.number_of_sources,
            number_of_documents_with_word=sources.number_of_sources_with_word,
        ),
    )
    best_matches.sort(
        key=lambda q: get_score_bm25(
            query=question.question,
            word_occurance=q.word_occurance,
            document_wordcount=q.wordcount,
            average_document_wordcount=sources.average_document_wordcount,
            number_of_documents=sources.number_of_sources,
            number_of_documents_with_word=sources.number_of_sources_with_word,
        ),
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
    dataset_path: Path = Path(dataset_path)
    if not dataset_path.is_file():
        raise ValueError(
            f"'dataset_path' must be a file. Got dataset_path={dataset_path}"
        )
    if not isinstance(save_directory, str):
        raise TypeError(
            "'save_directory' must be a valid path."
            + f" Got save_directory={save_directory}."
        )

    save_directory: Path = Path(save_directory)
    if save_directory.is_file():
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

    index: Path = Path(index)
    try:
        with index.open("r") as f:
            index_content = f.read()
        sources = Sources.model_validate_json(index_content)
    except OSError as e:
        raise OSError(f"Could not load index: {e}.")
    except ValidationError as _:
        raise ValueError(f"Could not parse {index!s} file.")

    try:
        with dataset_path.open("r") as f:
            dataset_content = f.read()
        dataset = RagDataset.model_validate_json(dataset_content)
    except OSError as e:
        raise OSError(f"Could not load dataset: {e}.")
    except ValidationError as _:
        raise ValueError(f"Could not parse {dataset_path!s} file.")

    f = partial(_search_sources_for_question, sources=sources, k=k)
    q_count = len(dataset.rag_questions)
    if processes > 0:
        with Pool(processes) as p:
            search_results = [
                q
                for q in tqdm.tqdm(
                    p.imap(
                        f, dataset.rag_questions, chunksize=process_chunk_size
                    ),
                    total=q_count,
                )
            ]
    else:
        search_results = [
            f(q) for q in tqdm.tqdm(dataset.rag_questions, total=q_count)
        ]

    student_search_results = StudentSearchResults(
        search_results=search_results, k=k
    )
    result_json = student_search_results.model_dump_json(indent=4)
    os.makedirs(save_directory, exist_ok=True)
    save_path = save_directory / dataset_path.name
    try:
        with save_path.open("w") as f:
            f.write(result_json)
    except OSError as e:
        raise OSError(f"Could not save json to file: {e}.")
