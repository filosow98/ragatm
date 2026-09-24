from index import _index
from typing import Literal
from pathlib import Path


class App:
    """Tool for generating answers with RAG."""

    def index(
        self,
        input: str = "data/raw",
        chunk_size: int = 1000,
        max_chunk_size: int = 2000,
        update: bool = False,
        processes: int | Literal["max"] | Literal["none"] = "none",
        process_chunk_size: int = 50,
        output: str = "data/processed/index.json",
    ):
        """Process raw data into a search index."""
        _index(
            Path(input),
            chunk_size,
            max_chunk_size,
            update,
            processes,
            process_chunk_size,
            Path(output),
        )

    def search(self, query: str, k: int = 1):
        """Search sources relevant to the query."""

    def search_dataset(
        self,
        dataset_path: str,
        k: int,
        save_directory: Path = Path(
            "data/output/search_results/UnansweredQuestions"
        ),
    ):
        """"""

    def answer(self, query: str, k: int):
        """Answer the query with the retrieved context."""

    def answer_dataset(
        self,
        student_search_results_path: Path = Path("data/datasets"),
        save_directory: Path = Path("data/output/"),
    ):
        """"""

    def evaluate(self, student_search_results_path: Path, dataset_path: Path):
        """"""
