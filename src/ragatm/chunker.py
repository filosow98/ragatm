from __future__ import annotations
from math import inf
from functools import reduce
from collections.abc import Iterator
from dataclasses import dataclass, astuple
import re
from abc import ABC


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
        """Add two adjecent spans."""
        boundary_test = Span(self.start - 1, self.end + 1)
        if other not in boundary_test:
            raise ValueError("Cannot merge two spans that do not touch.")
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
    document = document[doc_start:doc_end]
    start = 0
    end = len(document)
    ranges: list[Span] = []
    for mtch in re.finditer(separator.separator, document):
        if mtch.start() == -1:
            continue
        if mtch.start() == mtch.end():
            continue
        if isinstance(separator, Before):
            if mtch.start() == 0:
                continue
            end = mtch.start()
        elif isinstance(separator, After):
            if mtch.end() == len(document):
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
    if len(document) >= 1:
        ranges.append(Span(doc_start + start, doc_end))

    return ranges


def merge_ranges(
    document: str,
    chunk_size: int,
    max_chunk_size: int,
    ranges: list[Span],
) -> list[Span]:
    """Merge ranges that are too small."""
    if len(ranges) < 2:
        return ranges[:]
    ranges.sort(key=lambda x: x.start)
    min_idx = reduce(lambda acc, item: min(acc, item.start), ranges, inf)
    max_idx = reduce(lambda acc, item: max(acc, item.end), ranges, -inf)
    length = max_idx - min_idx
    target_count = length // chunk_size
    if target_count == 0:
        return [reduce(lambda acc, item: acc + item, ranges)]
    target_size = length // target_count
    if target_size > max_chunk_size:
        target_size = max_chunk_size
    acc_range = None
    result = []
    range_stop = None
    for range in ranges:
        if range_stop is None:
            range_stop = range.start
        if acc_range is None:
            acc_range = range
            range_stop += target_size
        if len(acc_range + range) > max_chunk_size:
            print("Too big")
            result.append(acc_range)
            acc_range = range
            while range_stop <= range.end:
                range_stop += target_size
            continue
        if range.end >= range_stop:
            print("Just right")
            result.append(acc_range + range)
            acc_range = None
            continue
        print("Runnig")
        acc_range += range
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
        ranges = merge_ranges(document, chunk_size, max_chunk_size, ranges)
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
        500,
        2000,
        [
            Before("class"),
            Before("def"),
            Before("\n\n"),
            Before("\n"),
            Before(" "),
            Before(""),
        ],
    )
    for chunk in chunks:
        print()
        print(f"==========({chunk.start}-{chunk.end})============")
        print(test[chunk.start : chunk.end])
