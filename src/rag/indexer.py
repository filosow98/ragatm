import os
from collections.abc import Callable
from datetime import UTC, datetime
from multiprocessing import Pool
from pathlib import Path

import tqdm

from rag.models import SourceFile, Sources


def index_files(
    files: list[Path],
    chunker: Callable[[Path], SourceFile],
) -> set[SourceFile]:
    """Split files into Sources.

    Parameters
    ----------
    files : list[Path]
        List of paths to files to be indexed.
    chunker: Callable[[Path], list[Span]]
        Chunker function to use.

    Returns
    -------
    set[SourceFile]
        Sources grouped by files.
    """
    files_count = len(files)
    return {k for k in tqdm.tqdm(map(chunker, files), total=files_count)}


def index_files_multiprocess(
    files: list[Path],
    chunker: Callable[[Path], SourceFile],
    processes: int,
    chunksize: int,
) -> set[SourceFile]:
    """Split files into Sources.

    Parameters
    ----------
    files : list[Path]
        List of paths to files to be indexed.
    chunker: Callable[[Path], list[Span]]
        Chunker function to use.
    processes : int
        Number of processes to use. Should be higher than 0.
    chunksize : int
        Chunksize used by the Pool.

    Returns
    -------
    set[SourceFile]
        Sources grouped by files.
    """

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
    indexer: Callable[[list[Path]], set[SourceFile]],
) -> Sources:
    """Create an index of files inside a directory. Will skip files that
    didn't change since last indexing.

    Parameters
    ----------
    path : Path
        Path to a directory with files to index.
    sources : Sources
        File index to update.
    indexer : Callable[[list[Path]], set[SourceFile]]
        Indexer to use.

    Returns
    -------
    Sources
        Index of files.
    """

    current_tree: dict[Path, datetime] = {}

    for root, _, files in path.walk():
        for file in files:
            file_path = root / file
            suf = file_path.suffix
            if suf not in [".py", ".md", ".txt"]:
                continue
            m_timestamp = os.path.getmtime(file_path)
            current_tree[file_path] = datetime.fromtimestamp(m_timestamp, UTC)

    old_tree = set(sources.sources)

    updated = []
    removed = set()
    for file in old_tree:
        current_mtime = current_tree.get(Path(file.file_path))
        if not current_mtime:
            removed.add(file)
            continue
        if current_mtime > file.modification_timestamp:
            updated.append(Path(file.file_path))
    if removed:
        print(f"Removing {len(removed)} missing sources.")
        sources.sources -= removed
        # for r in removed:
        #     print(f"Removed: {r.file_path}")
    if not updated:
        print("Nothing to do.")
        return sources

    sources.sources -= removed
    updated_sources = indexer(updated)
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
    sources.average_document_wordcount = wordcount / max(sourcecount, 1)
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
    indexer : Callable[[list[Path]], set[SourceFile]]
        Indexer to use.

    Returns
    -------
    Sources
        Index of files.
    """

    paths_tree: list[Path] = []

    for root, _, files in path.walk():
        for file in files:
            file_path = root / file
            suf = file_path.suffix
            if suf not in [".py", ".md", ".txt"]:
                continue
            paths_tree.append(file_path)

    indexed_sources = indexer(paths_tree)
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
