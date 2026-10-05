import os
from functools import partial
from pathlib import Path
from typing import Literal

from rag import (
    Sources,
    create_index,
    index_files,
    index_files_multiprocess,
    reqursive_chunker,
    update_index,
)


def index_inner(
    input: str,
    chunk_size: int,
    max_chunk_size: int,
    update: bool,
    processes: int | Literal["max", "none"],
    process_chunk_size: int,
    output: str,
    extensions: str,
) -> None:
    """
    Index files.

    Parameters
    ----------
    input : str
        Path to the directory containing raw files.
    chunk_size : int
        Target chunk size.
    max_chunk_size : int
        Max chunk size.
    update : bool
        True if index should be updated instead of creating a new one.
    processes : int | Literal["max", "none"]
        Number of processes to use during indexing. `none` if no processes
        should be used.
    process_chunk_size : int
        Chunksize of the process pool.
    output : str
        Output file path.
    extensions : str
        File extensions separated by space to filter files for indexing.
    """

    if not isinstance(input, str):
        raise TypeError(f"'input' must be a valid path. Got input={input}")
    if not isinstance(output, str):
        raise TypeError(f"'output' must be a valid path. Got output={output}")
    input_path: Path = Path(input)
    output_path: Path = Path(output)
    if not input_path.is_dir():
        raise ValueError(f"'input' path is not a directory. Got '{input}'.")
    if not isinstance(chunk_size, int):
        raise TypeError(f"'chunk_size' must be an integer. Got '{chunk_size}'")
    if not isinstance(max_chunk_size, int):
        raise TypeError(
            f"'max_chunk_size' must be an integer. Got '{max_chunk_size}'"
        )
    if chunk_size < 1:
        raise ValueError(
            f"'chunk_size' must be greater than 0. Got chunk_size={chunk_size}"
        )
    if max_chunk_size < 1:
        raise ValueError(
            "'max_chunk_size' must be greater than 0."
            + f" Got max_chunk_size={max_chunk_size}"
        )
    if max_chunk_size < chunk_size:
        raise ValueError(
            "'max_chunk_size' must be greater than 'chunk_size'."
            + f" Got max_chunk_size={max_chunk_size}, chunk_size={chunk_size}"
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

    chunker = partial(
        reqursive_chunker, chunk_size=chunk_size, max_chunk_size=max_chunk_size
    )
    if processes == "max":
        processes = os.cpu_count() or 1

    if processes == "none":
        processes = 0

    if processes > 0:
        indexer = partial(
            index_files_multiprocess,
            chunker=chunker,
            processes=processes,
            chunksize=process_chunk_size,
        )
    else:
        indexer = partial(
            index_files,
            chunker=chunker,
        )

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if update:
            with output_path.open() as f:
                content = f.read()
            sources = Sources.model_validate_json(content)
            indexed_sources = update_index(
                input_path,
                sources,
                indexer=indexer,
                extensions=extensions_list,
            )
        else:
            indexed_sources = create_index(
                input_path, indexer=indexer, extensions=extensions_list
            )

        sources_dump = indexed_sources.model_dump_json(indent=4)
        with output_path.open("w") as f:
            f.write(sources_dump)
        print(f"Index saved to: {output!s}")
    except OSError as e:
        raise OSError(f"Failed to index sources: {e}")
