from typing import Literal

from answer import answer_inner
from answer_dataset import answer_dataset_inner
from evaluate import evaluate_inner
from index import index_inner
from search import search_inner
from search_dataset import search_dataset_inner


class App:
    """Tool for generating answers with RAG."""

    def index(
        self,
        input: str = "data/raw",
        chunk_size: int = 1500,
        max_chunk_size: int = 2000,
        update: bool = False,
        processes: Literal["max", "none"] | int = "none",
        process_chunk_size: int = 50,
        output: str = "data/processed/index.json",
        extensions: str = ".py .md .txt",
    ) -> None:
        """Process raw data into a search index.

        Parameters
        ----------
        input : str, default="data/raw"
            Path to the direcory containing raw files.
        chunk_size : int, default=200
            Target chunk size.
        max_chunk_size : int, default=2000
            Max chunk size.
        update : bool, default=False
            True if index should be updated instead of creating a new one.
        processes : int | Literal["max", "none"], default='none'
            Number of processes to use for indexing. `none` if no processes
            should be used.
        process_chunk_size : int, default=50
            Chunksize of the process pool.
        output : str, default="data/processed/index.json"
            Output file.
        extensions : str, default=".py .md .txt"
            File types to index.
        """
        index_inner(
            input,
            chunk_size,
            max_chunk_size,
            update,
            processes,
            process_chunk_size,
            output,
            extensions,
        )

    def search(
        self,
        query: str,
        k: int = 1,
        index: str = "data/processed/index.json",
        extensions: str = ".py .md .txt",
    ) -> None:
        """Search for `k` most relevant sources.

        Parameters
        ----------
        query : str
            Query question.
        k : int, default=1
            Number of sources to find.
        index : str, default="data/processed/index.json"
            Path to a sources index.
        extensions : str, default=".py .md .txt"
            File types to index.
        """
        best_matches = search_inner(query, k, index, extensions)
        for match in best_matches:
            print(
                f"{match.file_path} [{match.first_character_index}:"
                + f"{match.last_character_index}]"
            )

    def search_dataset(
        self,
        dataset_path: str = "data/datasets/UnansweredQuestions/"
        + "dataset_docs_public.json",
        k: int = 1,
        save_directory: str = "data/output/search_results/UnansweredQuestions",
        index: str = "data/processed/index.json",
        processes: Literal["max", "none"] | int = "none",
        process_chunk_size: int = 5,
        extensions: str = ".py .md .txt",
    ) -> None:
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
        process_chunk_size: int, default=5
            Chunksize of the process pool.
        extensions : str, default=".py .md .txt"
            File types to index.
        """
        search_dataset_inner(
            dataset_path,
            k,
            save_directory,
            index,
            processes,
            process_chunk_size,
            extensions,
        )

    def answer(
        self,
        query: str,
        k: int = 1,
        index: str = "data/processed/index.json",
        extensions: str = ".py .md .txt",
    ) -> None:
        """Answer the query with the retrieved context.

        Parameters
        ----------
        query : str
            Query question.
        k : int, default=1
            Number of sources to find.
        index : str, default="data/processed/index.json"
            Path to a sources index.
        extensions : str, default=".py .md .txt"
            File types to use to generate answer.
        """
        print(answer_inner(query, k, index, extensions))

    def answer_dataset(
        self,
        student_search_results_path: str = "data/output/search_results/"
        + "UnansweredQuestions/dataset_docs_public.json",
        save_directory: str = "data/output/search_results_and_answer/"
        + "UnansweredQuestions",
    ) -> None:
        """Generate answers to the search results.

        Parameters
        ----------
        student_search_results_path : str
            Path to the search result.
        save_directory : str
            Path to the directory where the answers should be saved.
        """
        answer_dataset_inner(
            student_search_results_path,
            save_directory,
        )

    def evaluate(
        self,
        student_search_results_path: str = "data/output/search_results/"
        + "UnansweredQuestions/dataset_docs_public.json",
        dataset_path: str = "data/datasets/AnsweredQuestions/"
        + "dataset_docs_public.json",
    ) -> None:
        """
        Evaluate student search result against a ground-truth dataset.
        Warning: Dataset provided for this project is not a ground-truth
        dataset and contains mistakes.

        Parameters
        ----------
        student_search_results_path : str, default="data/output/search_results/
        UnansweredQuestions/dataset_docs_public.json"
            Path to a json file containing student search results.
        dataset_path : str, default="data/datasets/AnsweredQuestions/
        dataset_docs_public.json"
            Path to a ground-truth dataset with correct answers.
        """

        recalls = evaluate_inner(
            student_search_results_path,
            dataset_path,
        )
        for i, recall in enumerate(recalls):
            print(f"recall@{i + 1}: {recall:.3f} ({100 * recall:.1f}%)")
