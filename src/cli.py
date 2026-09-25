from pathlib import Path
from typing import Literal

from index import _index
from search import _search


class App:
    """Tool for generating answers with RAG."""

    def index(
        self,
        input: str = "data/raw",
        chunk_size: int = 200,
        max_chunk_size: int = 2000,
        update: bool = False,
        processes: Literal["max", "none"] | int = "none",
        process_chunk_size: int = 50,
        output: str = "data/processed/index.json",
    ):
        """Process raw data into a search index.

        Parameters
        ----------
        input: str, default="data/raw"
            Path to the direcory containing raw files.
        chunk_size: int, default=200
            Target chunk size.
        max_chunk_size: int, default=2000
            Max chunk size.
        update: bool, default=False
            True if index should be updated instead of creating a new one.
        processes: int | Literal["max", "none"], default='none'
            Number of processes to use during indexing. `none` if no processes
            should be used.
        process_chunk_size: int, default=50
            Chunksize of the process pool.
        output: str, default="data/processed/index.json"
            Output file.
        """
        try:
            _index(
                input,
                chunk_size,
                max_chunk_size,
                update,
                processes,
                process_chunk_size,
                output,
            )
        except Exception as e:
            print(f"Error: {e}")

    def search(
        self, query: str, k: int = 1, index: str = "data/processed/index.json"
    ):
        """Search for `k` most relevant sources.

        Parameters
        ----------
        query : str
            Query question.
        k : int, default=1
            Number of sources to find.
        index : str, default="data/processed/index.json"
            Path to a sources index.
        """
        try:
            _search(query, k, index)
        except Exception as e:
            print(f"Error: {e}")

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
