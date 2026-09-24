import os
from collections.abc import Callable
from datetime import datetime
from itertools import chain
from multiprocessing import Pool
from pathlib import Path

import tqdm

from ragatm.models import Source, SourceFile, Sources


def index_files(
    files: list[Path],
    chunker: Callable[[Path], SourceFile],
) -> set[SourceFile]:
    """Split files into Sources.

    Parameters
    ----------
    files : list[Path]
        List of paths to files to be indexed.
    chunker: Callable[[str], list[Span]]
        Chunker function to use.

    Returns
    -------
    dict[SourceFile, list[Source]]
        Dict mapping SourceFile to created Sources.
    """
    files_count = len(files)
    return {k for k in tqdm.tqdm(map(chunker, files), total=files_count)}


def index_files_multiprocess(
    files: list[Path],
    chunker: Callable[[Path], tuple[SourceFile, list[Source]]],
    processes: int,
    chunksize: int,
) -> set[SourceFile]:
    """"""

    files_count = len(files)
    # index_files() function should run in this case.
    if processes < 1:
        raise ValueError("Cannot use less than 1 process.")
    with Pool(processes) as p:
        sources = {
            k
            for k in tqdm.tqdm(
                p.imap(chunker, files, chunksize=chunksize), total=files_count
            )
        }

    return sources


def update_index(
    path: Path,
    sources: Sources,
    indexer: Callable[[list[Path]], dict[SourceFile, list[Source]]],
) -> Sources:
    """Create an index of files inside a directory. Will skip files that
    didn't change since last indexing.

    Parameters
    ----------
    path : Path
        Path to a directory with files to index.
    sources : Sources
        File index to update.
    use_multiprocessing : int | None, default=None
        Use multiple processes to index files. None if no additional processes
        should be spawned, 0 if os.cpu_count() should be used, >0 to
        set how many processes should be used.
    multiprocess_chunksize : int, default=1
        Chunksize used in multiprocessing. Fallbacks to 1 for values less
        than 1.

    Returns
    -------
    Sources
        Index of files.
    """

    current_tree: set[SourceFile] = set()

    for root, dirs, files in path.walk():
        for file in files:
            file_path = root / file
            m_timestamp = os.path.getmtime(file_path)
            current_tree.add(
                SourceFile(
                    file_path=str(file_path),
                    modification_timestamp=datetime.fromtimestamp(m_timestamp),
                    sources=[],
                )
            )

    old_tree = set(sources.sources)
    updated = current_tree - old_tree
    removed = old_tree - current_tree
    sources.sources -= removed

    files = [Path(file.file_path) for file in updated]
    updated_sources = indexer(files)
    sources.sources.update(updated_sources)
    wordcount = 0
    sourcecount = 0
    word_occurance: dict[str, int] = {}
    for src_file in sources.sources:
        for src in src_file.sources:
            wordcount += src.wordcount
            sourcecount += 1
            for word in src.word_occurance:
                if word_occurance.get(word):
                    word_occurance[word] += 1
                else:
                    word_occurance[word] = 1
    sources.average_document_wordcount = wordcount / sourcecount
    sources.number_of_sources_with_word = word_occurance
    sources.number_of_sources = sourcecount

    return sources


def create_index(
    path: Path, indexer: Callable[[list[Path]], set[SourceFile]]
) -> Sources:
    """Create an index of files inside a directory. Will skip files that
    didn't change since last indexing. If 'sources' is None, a new index will
    be created.

    Parameters
    ----------
    path : Path
        Path to a directory with files to index.
    sources : Sources | None, default=None
        File index to update. New index will be created if it is set to None.
    use_multiprocessing : int | None, default=None
        Use multiple processes to index files. None if no additional processes
        should be spawned, 0 if os.cpu_count() should be used, >0 to
        set how many processes should be used.
    multiprocess_chunksize : int, default=1
        Chunksize used in multiprocessing. Fallbacks to 1 for values less
        than 1.

    Returns
    -------
    Sources
        Index of files.
    """

    sources_tree: set[SourceFile] = set()

    for root, dirs, files in path.walk():
        for file in files:
            file_path = root / file
            m_timestamp = os.path.getmtime(file_path)
            sources_tree.add(
                SourceFile(
                    file_path=str(file_path),
                    modification_timestamp=datetime.fromtimestamp(m_timestamp),
                    sources=[],
                )
            )

    files = [Path(file.file_path) for file in sources_tree]
    indexed_sources = indexer(files)
    wordcount = 0
    sourcecount = 0
    word_occurance: dict[str, int] = {}
    for src_file in indexed_sources:
        for src in src_file.sources:
            wordcount += src.wordcount
            sourcecount += 1
            for word in src.word_occurance:
                if word_occurance.get(word):
                    word_occurance[word] += 1
                else:
                    word_occurance[word] = 1
    return Sources(
        sources=indexed_sources,
        average_document_wordcount=wordcount / max(sourcecount, 1),
        number_of_sources_with_word=word_occurance,
        number_of_sources=sourcecount,
    )
