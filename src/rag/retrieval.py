import re
from functools import lru_cache
from math import log

# Source: https://en.wikipedia.org/wiki/Okapi_BM25


def get_idf(
    *,
    word: str,
    number_of_documents: int,
    number_of_documents_with_word: dict[str, int],
) -> float:
    """Get IDF (inverse document frequency) weight for a word."""
    documents_with_word = number_of_documents_with_word.get(word, 0)
    divident = number_of_documents - documents_with_word + 0.5
    divisor = documents_with_word + 0.5
    return log(divident / divisor + 1)


def get_word_frequency(
    *,
    word: str,
    word_occurance: dict[str, int],
    document_wordcount: int,
    average_document_wordcount: float,
    k: float = 1.6,
    b: float = 0.75,
) -> float:
    """Get document word frequency adjusted for document length."""
    occurance = word_occurance.get(word, 0)
    if occurance == 0:
        return 0
    frequency = (
        occurance
        * (k + 1)
        / (
            occurance
            + k * (1 - b + b * document_wordcount / average_document_wordcount)
        )
    )
    return frequency


word_patt = re.compile(r"[A-Za-z][a-z]+|([A-Z][A-Z0-9]*[A-Z])(?=[A-Z]|$|\s|_)")
banned_word_patt = re.compile(r"0x[0-9abcdefABCDEF]+")


@lru_cache(maxsize=128)
def get_proper_words(text: str) -> list[str]:
    """Get words that are separated by whitespace, underscores, and
    capitalization. Makes words lowercase. Ignore strings that are numbers."""
    whole_words = [
        m.string[m.start() : m.end()] for m in re.finditer(r"\w+", text)
    ]
    sub_words: list[str] = []
    for w in whole_words:
        sub = [
            m.string[m.start() : m.end()] for m in re.finditer(word_patt, w)
        ]

        if len(sub) == 1:
            continue
        sub_words = sub_words + sub

    words: list[str] = whole_words + sub_words
    words = [w.lower() for w in words if len(w) > 2]
    return words


# @lru_cache(maxsize=128)
def get_word_occurance(text: str) -> dict[str, int]:
    """Get how many times each word appears in the text."""
    words: dict[str, int] = {}

    for word in get_proper_words(text):
        if not word.strip():
            continue
        count = words.setdefault(word, 0)
        words[word] = count + 1

    return words


# @lru_cache(maxsize=128)
def get_document_wordcount(text: str) -> int:
    """Get document wordcount."""
    return len(get_proper_words(text))


def get_score_bm25(
    *,
    query: str,
    word_occurance: dict[str, int],
    document_wordcount: int,
    average_document_wordcount: float,
    number_of_documents: int,
    number_of_documents_with_word: dict[str, int],
    k: float = 1.6,
    b: float = 0.75,
    delta: float = 0.0,  # If you want to use BM25+, set to 1.0
) -> float:
    """Get BM25 score of the query."""

    words = get_word_occurance(query)

    score = 0.0
    for word in words:
        idf = get_idf(
            word=word,
            number_of_documents=number_of_documents,
            number_of_documents_with_word=number_of_documents_with_word,
        )

        score += idf * (
            get_word_frequency(
                word=word,
                word_occurance=word_occurance,
                document_wordcount=document_wordcount,
                average_document_wordcount=average_document_wordcount,
                k=k,
                b=b,
            )
            + delta
        )
    return score
