from answer import _answer
from pathlib import Path
from typing import Literal

from index import _index
from search import _search
from search_dataset import _search_dataset


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
            Number of processes to use for indexing. `none` if no processes
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
            best_matches = _search(query, k, index)
            for match in best_matches:
                print(
                    f"{match.file_path} [{match.first_character_index}:"
                    + f"{match.last_character_index}]"
                )
        except Exception as e:
            print(f"Error: {e}")

    def search_dataset(
        self,
        dataset_path: str = "data/datasets/UnansweredQuestions/"
        + "dataset_docs_public.json",
        k: int = 1,
        save_directory: str = "data/output/search_results/UnansweredQuestions",
        index: str = "data/processed/index.json",
        processes: Literal["max", "none"] | int = "none",
        process_chunk_size: int = 5,
    ):
        """Run search on a dataset for `k` most relevant sources.

        Parameters
        ----------
        dataset_path : str
            Path to a json with a question dataset.
        k : int
            Number of sources to find.
        save_directory : str
            Directory where the answer json will be saved.
        index : str
            Path to a sources index.
        processes: int | Literal["max", "none"], default='none'
            Number of processes to use for serching. `none` if no processes
            should be used.
        process_chunk_size: int, default=10
            Chunksize of the process pool.
        """
        try:
            _search_dataset(
                dataset_path,
                k,
                save_directory,
                index,
                processes,
                process_chunk_size,
            )
        except Exception as e:
            print(f"Error: {e}")

    def answer(
        self, query: str, k: int = 1, index: str = "data/processed/index.json"
    ):
        """Answer the query with the retrieved context.

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
            print(
                _answer(
                    query,
                    k,
                    index,
                )
            )
        except Exception as e:
            print(f"Error: {e}")

    def answer_dataset(
        self,
        student_search_results_path: str = "data/datasets",
        save_directory: str = "data/output/",
        index: str = "data/processed/index.json",
    ):
        """"""

    def evaluate(self, student_search_results_path: Path, dataset_path: Path):
        """"""
