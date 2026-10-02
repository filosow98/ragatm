from __future__ import annotations

import os
import re
from abc import ABC
from collections.abc import Iterator
from dataclasses import astuple, dataclass
from datetime import UTC, datetime
from functools import reduce
from math import ceil
from pathlib import Path

from rag.models import Source, SourceFile
from rag.retrieval import get_document_wordcount, get_word_occurance


@dataclass
class SplitPosition(ABC):
    """Specify if the text should be split before or after the separator."""

    separator: str


@dataclass
class Before(SplitPosition):
    """The text should be split before the separator."""


@dataclass
class After(SplitPosition):
    """The text should be split after the separator."""


@dataclass
class Span:
    """A fragment of text as a starting and ending index."""

    start: int
    end: int

    def __post_init__(self):
        if self.start > self.end:
            raise ValueError(
                "Start index cannot be bigger than the end"
                + f" index in a Span. Got Span({self.start}, {self.end})."
            )

    def __iter__(self) -> Iterator[int]:
        """Iterate over the attributes of this object."""
        return iter(astuple(self))

    def __contains__(self, other: int | Span) -> bool:
        """Check if a number or Span is in this Span. End index is not
        included."""
        if isinstance(other, int):
            if other < self.start or other >= self.end:
                return False
        if isinstance(other, Span):
            if other.end <= self.start or other.start >= self.end:
                return False

        return True

    def __add__(self, other: Span) -> Span:
        """Add two spans."""
        # Without this two spans will 'fill in' the gap between ranges.
        # boundary_test = Span(self.start - 1, self.end + 1)
        # if other not in boundary_test:
        #     raise ValueError("Cannot merge two spans that do not touch.")
        end = self.end
        end = max(end, other.end)
        start = self.start
        start = min(start, other.start)
        return Span(start, end)

    def __len__(self) -> int:
        """Length of this Span."""
        return self.end - self.start

    def calc_iou(self, other: Span) -> float:
        """Calculate IoU of two spans."""
        if self not in other:
            return 0.0
        union = self + other
        start = max(self.start, other.start)
        end = min(self.end, other.end)
        return (end - start) / len(union)


def get_ranges(
    document: str,
    separator: SplitPosition,
    range: Span,
) -> list[Span]:
    """Get the document ranges delimited by the separators."""
    doc_start, doc_end = range
    doc = document[doc_start:doc_end]
    start = 0
    end = len(doc)
    ranges: list[Span] = []
    for mtch in re.finditer(separator.separator, doc, flags=re.MULTILINE):
        if mtch.start() == -1:
            continue
        if mtch.start() == mtch.end():
            continue
        if isinstance(separator, Before):
            if mtch.start() == 0:
                continue
            end = mtch.start()
        elif isinstance(separator, After):
            if mtch.end() == len(doc):
                continue
            end = mtch.end()
        else:
            raise NotImplementedError(
                f"'{type(separator)}' not supported as a separator."
            )
        ranges.append(Span(doc_start + start, doc_start + end))

        if isinstance(separator, Before):
            start = mtch.start()
        elif isinstance(separator, After):
            start = mtch.end()
        else:
            raise NotImplementedError(
                f"'{type(separator)}' not supported as a separator."
            )
    if len(doc) >= 1:
        ranges.append(Span(doc_start + start, doc_end))

    return ranges


def _get_sum_of_chunks_range(start: int, end: int, ranges: list[Span]) -> Span:
    """Get sum of Spans from `start` to `end` index."""

    span = reduce(
        lambda acc, it: acc + it,
        ranges[start:end],
    )
    return span


def _merge_chunks_by_size(
    chunk_size: int,
    max_chunk_size: int,
    ranges: list[Span],
) -> list[Span]:
    """Merge ranges that are too small."""
    if len(ranges) <= 1:
        return ranges[:]
    chunk_indices: list[int] = [0]
    i = 0
    acc_chunk = ranges[0]
    while i < len(ranges) - 1:
        i += 1
        next_chunk = ranges[i]
        if len(acc_chunk) > max_chunk_size:
            chunk_indices.append(i)
            acc_chunk = ranges[i]
            continue
        if len(acc_chunk + next_chunk) > max_chunk_size:
            if i < len(ranges):
                chunk_indices.append(i)
            acc_chunk = ranges[i]
            continue
        if len(acc_chunk) > chunk_size:
            chunk_indices.append(i)
            acc_chunk = ranges[i]

    chunk_indices.append(len(ranges))
    # Move indices left and right to minimize the size difference of final
    # chunks.
    max_iterations = 10
    while max_iterations:
        max_iterations -= 1
        i = 0
        while i < len(chunk_indices) - 2:
            i += 1
            prev_idx = chunk_indices[i - 1]
            next_idx = chunk_indices[i + 1]
            this_idx = chunk_indices[i]
            len_prev = len(
                _get_sum_of_chunks_range(prev_idx, this_idx, ranges)
            )
            len_this = len(
                _get_sum_of_chunks_range(this_idx, next_idx, ranges)
            )
            new_idx = this_idx
            curr_diff = abs(len_prev - len_this)
            move_left_idx = this_idx - 1
            if move_left_idx > prev_idx:
                len_prev = len(
                    _get_sum_of_chunks_range(prev_idx, move_left_idx, ranges)
                )
                len_this = len(
                    _get_sum_of_chunks_range(move_left_idx, next_idx, ranges)
                )
                if len_prev <= max_chunk_size and len_this <= max_chunk_size:
                    move_left_diff = abs(len_prev - len_this)
                    if move_left_diff < curr_diff:
                        new_idx = move_left_idx
            move_right_idx = this_idx + 1
            if move_right_idx < next_idx:
                len_prev = len(
                    _get_sum_of_chunks_range(prev_idx, move_right_idx, ranges)
                )
                len_this = len(
                    _get_sum_of_chunks_range(move_right_idx, next_idx, ranges)
                )
                if len_prev <= max_chunk_size and len_this <= max_chunk_size:
                    move_right_diff = abs(len_prev - len_this)
                    if move_right_diff < curr_diff:
                        new_idx = move_right_idx
            chunk_indices[i] = new_idx
    result = []
    i = 0
    while i < len(chunk_indices) - 1:
        i += 1
        prev_idx = chunk_indices[i - 1]
        this_idx = chunk_indices[i]
        span = _get_sum_of_chunks_range(prev_idx, this_idx, ranges)
        result.append(span)

    return result


def into_chunks(
    document: str,
    chunk_size: int,
    max_chunk_size: int,
    separators: list[SplitPosition],
) -> list[Span]:
    """Split the document into chunks."""

    if len(document.strip()) == 0:
        return []

    chunked: list[Span] = []
    stack: list[tuple[Span, int]] = []
    stack.append((Span(0, len(document)), -1))
    while len(stack):
        (start, end), sep_idx = stack.pop()
        sep_idx += 1
        if end - start < max_chunk_size:
            chunked.append(Span(start, end))
            continue
        # Fallback for when there is no separator to chunk the text with.
        if sep_idx >= len(separators) or not separators[sep_idx]:
            length = end - start
            number_of_chunks = (length // max_chunk_size) + 1
            chunk_length = ceil(length / number_of_chunks)
            chunk_length = max(chunk_length, 1)
            chunk_length = min(chunk_length, 2000)
            for _ in range(number_of_chunks - 1):
                chunked.append(Span(start, start + chunk_length))
                start += chunk_length
            if end - start > max_chunk_size:
                middle = (start + end) // 2
                chunked.append(Span(start, middle))
                chunked.append(Span(middle, end))
            else:
                chunked.append(Span(start, end))
            continue
        sep = separators[sep_idx]
        ranges = get_ranges(document, sep, Span(start, end))

        average_size = sum([len(c) for c in ranges]) / len(ranges)
        if average_size < chunk_size:
            ranges = _merge_chunks_by_size(chunk_size, max_chunk_size, ranges)

        ranges_with_separator = [(r, sep_idx) for r in ranges]
        stack += ranges_with_separator

    chunked.reverse()
    return chunked


def reqursive_chunker(
    file: Path,
    chunk_size: int,
    max_chunk_size: int,
) -> SourceFile:
    """Split file into Source list.

    Parameters
    ----------
    file : Path
        Path to file.

    Returns
    -------
    tuple[SourceFile, list[Source]]
        SourceFile and belonginig to this file Sources. If read fails, the
        Source list will be empty.
    """

    m_timestamp = os.path.getmtime(file)
    source_file = SourceFile(
        file_path=str(file),
        modification_timestamp=datetime.fromtimestamp(m_timestamp, UTC),
        sources=[],
    )
    # Read fails with png, jpeg, cpy, etc. Using EAFP to avoid checking for
    # all possible edgecases.
    try:
        with file.open("r") as f:
            content = f.read()
    except OSError as _:
        return source_file
    except UnicodeError as _:
        return source_file

    match file.suffix:
        case ".py":
            sep = [
                Before(r"^def\s"),
                Before(r"^class\s"),
                Before(r" +def\s"),
                Before(r" +class\s"),
                Before(r"\n\n"),
                Before(r"\n"),
                Before(r" "),
                Before(r""),
            ]
        case ".md":
            sep = [
                Before(r"#\s+"),
                Before(r"\n[^\n]*\S+[^\n]*\n {0,3}=+[ \t]+\n"),
                Before(r"##\s+"),
                Before(r"\n[^\n]*\S+[^\n]*\n {0,3}-+[ \t]+\n"),
                Before(r"###\s+"),
                Before(r"####\s+"),
                Before(r"#####\s+"),
                Before(r"######\s+"),
                Before(r"\n\n"),
                Before(r"\n"),
                Before(r" "),
                Before(r""),
            ]
        case ".txt":
            sep = [
                Before(r"\n\n"),
                Before(r"\n"),
                Before(r" "),
                Before(r""),
            ]
        case _:
            sep = [
                Before(r"\n\n"),
                Before(r"\n"),
                Before(r" "),
                Before(r""),
            ]

    chunks = into_chunks(content, chunk_size, max_chunk_size, sep)
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
    source_file.sources = sources
    return source_file
