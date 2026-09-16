import multiprocessing
from ragatm.retrieval import get_document_wordcount, get_word_occurance
from functools import partial
import tqdm
from multiprocessing import Pool
from collections.abc import Callable
from ragatm.chunker import Span
from typing import SupportsIndex
from datetime import datetime
import os
from pathlib import Path
from ragatm.models import Sources, SourceFile, Source


def index_file(
    file: Path,
    chunker: Callable[[str], list[Span]],
) -> tuple[SourceFile, list[Source]]:
    """Split file into Source list.

    Parameters
    ----------
    file : Path
        Path to file.
    chunker: Callable[[str], list[Span]]
        Chunker function to use.

    Returns
    -------
    tuple[SourceFile, list[Source]]
        SourceFile and belonginig to this file Sources.
    """

    with open(file, "r") as f:
        content = f.read()

    chunks = chunker(content)
    sources = []
    for chunk in chunks:
        slice = content[chunk.start : chunk.end]
        wordcount = get_document_wordcount(slice)
        word_occurance = get_word_occurance(slice)
        sources.append(
            Source(
                file_path=str(file),
                first_character_index=chunk.start,
                last_character_index=chunk.end,
                wordcount=wordcount,
                word_occurance=word_occurance,
            )
        )
    m_timestamp = os.path.getmtime(file)
    source_file = SourceFile(
        path=file,
        modification_timestamp=datetime.fromtimestamp(m_timestamp),
    )
    return (source_file, sources)


def index_files(
    files: list[Path],
    chunker: Callable[[str], list[Span]],
) -> dict[SourceFile, list[Source]]:
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
    f = partial(index_file, chunker=chunker)
    return {k: v for k, v in tqdm.tqdm(map(f, files), total=files_count)}


def index_files_multiprocess(
    files: list[Path],
    chunker: Callable[[str], list[Span]],
    processes: int = 0,
    chunksize: int = 1,
) -> dict[SourceFile, list[Source]]:
    """"""

    files_count = len(files)
    f = partial(index_file, chunker=chunker)
    if processes < 1:
        processes = os.cpu_count() or 1
    with Pool(processes) as p:
        sources = {
            k: v
            for k, v in tqdm.tqdm(
                p.imap(f, files, chunksize=chunksize), total=files_count
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

    current_tree = set()

    for root, dirs, files in path.walk():
        for file in files:
            file_path = (root / file).absolute()
            m_timestamp = os.path.getmtime(file_path)
            current_tree.add(
                SourceFile(
                    path=file_path,
                    modification_timestamp=datetime.fromtimestamp(m_timestamp),
                )
            )

    old_tree = set(sources.sources)
    updated = current_tree - old_tree
    removed = old_tree - current_tree
    for key in removed:
        sources.sources.pop(key)

    files = [file.file_path for file in updated]
    if use_multiprocessing is not None:
        new_sources = _index_files_multiprocess(files, chunker)


def create_index(
    path: Path,
    indexer: Callable[[list[Path]], dict[SourceFile, list[Source]]],
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

    if sources:
        return update_index(path, sources)
