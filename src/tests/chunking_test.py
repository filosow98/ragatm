import pytest

from rag import After, Before, Span, SplitPosition, get_ranges


@pytest.mark.parametrize(
    "test_string,separator,expected,range",
    [
        (
            "12345before12345",
            Before("before"),
            [Span(0, 5), Span(5, 16)],
            None,
        ),
        ("before12345", Before("before"), [Span(0, 11)], None),
        ("12345before", Before("before"), [Span(0, 5), Span(5, 11)], None),
        ("12345", Before("before"), [Span(0, 5)], None),
        ("", Before("before"), [], None),
        (
            "12345before12345",
            Before("before"),
            [Span(3, 5), Span(5, 11)],
            Span(3, 11),
        ),
        ("12345before12345", Before("before"), [Span(3, 10)], Span(3, 10)),
        ("12345after12345", After("after"), [Span(0, 10), Span(10, 15)], None),
        ("after12345", After("after"), [Span(0, 5), Span(5, 10)], None),
        ("12345after", After("after"), [Span(0, 10)], None),
        ("12345", After("after"), [Span(0, 5)], None),
        ("", After("after"), [], None),
        ("12345after12345", After("after"), [Span(3, 10)], Span(3, 10)),
        (
            "12345after12345",
            After("after"),
            [Span(3, 10), Span(10, 11)],
            Span(3, 11),
        ),
        ("12345after12345", After("after"), [Span(3, 9)], Span(3, 9)),
    ],
)
def test_get_ranges(
    test_string: str,
    separator: SplitPosition,
    expected: list[Span],
    range: Span | None,
) -> None:
    """Test if ranges have correct bounds."""

    if range is None:
        range = Span(0, len(test_string))
    ranges = get_ranges(test_string, separator, range)
    assert ranges == expected
