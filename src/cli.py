from pathlib import Path


class App:
    """Tool for generating answers with RAG."""

    def index(
        self,
        input: Path = Path("data/raw"),
        max_chunk_size: int = 2000,
        output: Path = Path("data/processed"),
    ):
        """Process raw data into a search index."""

    def search(self, query: str, k: int = 1):
        """Search sources relevant to the query."""

    def search_dataset(
        self,
        dataset_path: Path,
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
