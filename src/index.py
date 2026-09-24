import os
from functools import partial
from pathlib import Path
from typing import Literal

from ragatm import (
    Sources,
    create_index,
    index_files,
    index_files_multiprocess,
    reqursive_chunker,
    update_index,
)


def _index(
    input: Path,
    chunk_size: int,
    max_chunk_size: int,
    update: bool,
    processes: int | Literal["max", "none"],
    process_chunk_size: int,
    output: Path,
):
    """"""

    chunker = partial(
        reqursive_chunker, chunk_size=chunk_size, max_chunk_size=max_chunk_size
    )
    if processes == "max":
        processes = os.cpu_count() or 1

    if processes == "none" or processes < 1:
        processes = 0

    if processes > 0:
        indexer = partial(
            index_files_multiprocess,
            chunker=chunker,
            processes=processes,
            chunksize=10,
        )
    else:
        indexer = partial(
            index_files,
            chunker=chunker,
        )

    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        if update:
            with output.open() as f:
                content = f.read()
            sources = Sources.model_validate_json(content)
            indexed_sources = update_index(input, sources, indexer=indexer)
        else:
            indexed_sources = create_index(input, indexer=indexer)

        sources_dump = indexed_sources.model_dump_json(indent=4)
        with output.open("w") as f:
            f.write(sources_dump)
    except OSError as e:
        print(f"Failed to index sources: {e}")
