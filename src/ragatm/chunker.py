from __future__ import annotations

import re
from abc import ABC
from collections.abc import Iterator
from dataclasses import astuple, dataclass
from functools import reduce


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
    if len(ranges) < 2:
        return ranges[:]
    chunk_indices: list[int] = [i for i in range(len(ranges) + 1)]
    max_iterations = 100  # Can merge 2^100 chunks into 1 so it's enough.
    while max_iterations:
        max_iterations -= 1
        new_indices = [0]
        i = 0
        while i < len(chunk_indices) - 2:
            i += 1
            prev_idx = chunk_indices[i - 1]
            this_idx = chunk_indices[i]
            next_idx = chunk_indices[i + 1]
            len_prev = len(
                _get_sum_of_chunks_range(prev_idx, this_idx, ranges)
            )
            len_this = len(
                _get_sum_of_chunks_range(this_idx, next_idx, ranges)
            )
            if len_prev + len_this > max_chunk_size:
                new_indices.append(this_idx)
                # skip two
                i += 1
                continue
            if len_prev < chunk_size or len_this < chunk_size:
                # skip two
                i += 1
                continue
            new_indices.append(this_idx)
        new_indices.append(len(ranges))
        if chunk_indices == new_indices:
            break
        chunk_indices = new_indices
    # Move indices left and right to minimize the size difference of final
    # chunks.
    max_iterations = 100
    while max_iterations:
        max_iterations -= 1
        new_indices = [0]
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
):
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
            length = end - start + 1
            number_of_chunks = (length // max_chunk_size) + 1
            chunk_length = round(length / number_of_chunks)
            if chunk_length < 1:
                chunk_length = 1
            for i in range(number_of_chunks - 1):
                chunked.append(Span(start, start + chunk_length - 1))
                start += chunk_length
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


if __name__ == "__main__":
    test = """

import re
from abc import ABC


class SplitPosition(ABC):
    \"\"\"Specify if the text should be split before or after the separator.\"\"\"

    def __init__(self, separator: str) -> None:
        self.separator = separator


class Before(SplitPosition):
    \"\"\"The text should be split before the separator.\"\"\"

    def __init__(self, separator: str) -> None:
        self.separator = separator


class After(SplitPosition):
    \"\"\"The text should be split after the separator.\"\"\"

    def __init__(self, separator: str) -> None:
        self.separator = separator


def get_ranges(
    document: str,
    separator: SplitPosition,
    range: tuple[int, int] | None = None,
) -> list[tuple[int, int]]:
    \"\"\"Get the document ranges delimited by the separators.\"\"\"
    doc_start = 0
    doc_end = len(document) - 1
    if range is not None:
        doc_start, doc_end = range
        document = document[doc_start : doc_end + 1]
    start = 0
    end = len(document)
    ranges: list[tuple[int, int]] = []
    for mtch in re.finditer(separator.separator, document):
        if mtch.start() == -1:
            continue
        if mtch.start() == mtch.end():
            continue
        if isinstance(separator, Before):
            if mtch.indexstart() == 0:
                continue
            end = mtch.start() - 1
        elif isinstance(separator, After):
            if mtch.end() == len(document):
                continue
            end = mtch.end() - 1
        else:
            raise TypeError(
                f"'{type(separator)}' not supported as a separator."
            )
        ranges.append((doc_start + start, doc_start + end))

        if isinstance(separator, Before):
            start = mtch.start()
        elif isinstance(separator, After):
            start = mtch.end()
        else:
            raise TypeError(
                f"'{type(separator)}' not supported as a separator."
            )
    if len(document) >= 1:
        ranges.append((doc_start + start, doc_end))

    return ranges


def into_chunks(
    document: str,
    chunk_size: int,
    max_chunk_size: int,
    overlap: int,
    separators: list[SplitPosition],
):
    \"\"\"Split the document into chunks.\"\"\"

    if len(document) == 0:
        return []

    chunked: list[tuple[int, int]] = []
    stack: list[tuple[tuple[int, int], int]] = []
    stack.append(((0, len(document) - 1), -1))
    while len(stack):
        (start, end), sep_idx = stack[-1]
        sep_idx += 1
        if end - start + 1 < max_chunk_size:
            chunked.append((start, end))
            continue
        if sep_idx >= len(separators) or not separators[sep_idx]:
            length = end - start + 1
            number_of_chunks = (length // max_chunk_size) + 1
            chunk_length = round(length / number_of_chunks)
            if chunk_length < 1:
                chunk_length = 1
            for i in range(number_of_chunks - 1):
                chunked.append((start, start + chunk_length - 1))
                start += chunk_length
            chunked.append((start, end))
            continue
        sep = separators[sep_idx]
        ranges = get_ranges(document, sep, (start, end))
        ranges_with_separator = [(r, sep_idx) for r in ranges]
        stack += ranges_with_separator

"""
    chunks = into_chunks(
        test,
        50,
        2000,
        [
            Before(r"^def\s"),
            Before(r"^class\s"),
            Before(r" +def\s"),
            Before(r" +class\s"),
            Before(r"\n\n"),
            Before(r"\n"),
            Before(r" "),
            Before(r""),
        ],
    )
    for chunk in chunks:
        print()
        print(f"==========({chunk.start}-{chunk.end})============")
        print(test[chunk.start : chunk.end])
