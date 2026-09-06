import re
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
    document_length: int,
    average_document_length: float,
    k: float = 1.6,
    b: float = 0.75,
):
    """Get document frequency adjusted for document length."""
    occurance = word_occurance.get(word, 0)
    if occurance == 0:
        return 0
    frequency = (
        occurance
        * (k + 1)
        / (
            occurance
            + k * (1 - b + b * document_length / average_document_length)
        )
    )
    return frequency


def get_score_bm25(
    *,
    query: str,
    word_occurance: dict[str, int],
    document_length: int,
    average_document_length: float,
    number_of_documents: int,
    number_of_documents_with_word: dict[str, int],
    k: float = 1.6,
    b: float = 0.75,
) -> float:
    """Get BM25 score of the query."""
    words = set()

    sep = re.compile("\W+")
    for word in re.split(sep, query):
        if not word.strip():
            continue
        words.add(word)

    score = 0
    for word in words:
        idf = get_idf(
            word=word,
            number_of_documents=number_of_documents,
            number_of_documents_with_word=number_of_documents_with_word,
        )

        score += idf * get_word_frequency(
            word=word,
            word_occurance=word_occurance,
            document_length=document_length,
            average_document_length=average_document_length,
            k=k,
            b=b,
        )
    return score
