import os
from contextlib import redirect_stderr, redirect_stdout
from functools import partial
from pathlib import Path

import tqdm
from optimum.intel import OVModelForCausalLM
from pydantic import ValidationError
from transformers import GenerationConfig, PreTrainedTokenizer

from answer import format_prompt
from llm import load_model
from rag import (
    MinimalAnswer,
    MinimalSearchResults,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)


def _generate_answer(
    search_result: MinimalSearchResults,
    model: OVModelForCausalLM,
    tokenizer: PreTrainedTokenizer,
    generation_config: GenerationConfig,
) -> MinimalAnswer:
    """"""
    messages = format_prompt(
        search_result.question, search_result.retrieved_sources
    )
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    input_ids = tokenizer([prompt], return_tensors="pt")
    generated_ids = model.generate(
        **input_ids, generation_config=generation_config
    )
    output_ids = generated_ids[0][len(input_ids.input_ids[0]) :].tolist()

    answer = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")

    return MinimalAnswer(
        question_id=search_result.question_id,
        question=search_result.question,
        retrieved_sources=search_result.retrieved_sources,
        answer=answer,
    )


def answer_dataset_inner(
    student_search_results_path: str,
    save_directory: str,
):
    """Generate answers to the search results.

    Parameters
    ----------
    student_search_results_path : str
        Path to the search result.
    save_directory : str
        Path to the directory where the answers should be saved.
    """
    if not isinstance(student_search_results_path, str):
        raise TypeError(
            "'student_search_results_path' must be a valid path."
            + f" Got student_search_results_path={student_search_results_path}."
        )
    student_search_results_path: Path = Path(student_search_results_path)
    if not student_search_results_path.is_file():
        raise ValueError(
            "'student_search_results_path' must be a file."
            + f" Got student_search_results_path={student_search_results_path}"
        )
    if not isinstance(save_directory, str):
        raise TypeError(
            "'save_directory' must be a valid path."
            + f" Got save_directory={save_directory}."
        )

    save_directory: Path = Path(save_directory)
    if save_directory.is_file():
        raise ValueError(
            "'save_directory' must be a directory."
            + f" Got save_directory={save_directory}"
        )

    try:
        with student_search_results_path.open("r") as f:
            search_content = f.read()
        student_search_results = StudentSearchResults.model_validate_json(
            search_content
        )
    except OSError as e:
        raise OSError(f"Could not load search results: {e}.")
    except ValidationError as _:
        raise ValueError(
            f"Could not parse {student_search_results_path!s} file."
        )

    (model, tokenizer, generation_config) = load_model()

    g = partial(
        _generate_answer,
        model=model,
        tokenizer=tokenizer,
        generation_config=generation_config,
    )
    q_count = len(student_search_results.search_results)
    answer_results = [
        g(q)
        for q in tqdm.tqdm(
            student_search_results.search_results, total=q_count
        )
    ]
    student_search_results_and_answer = StudentSearchResultsAndAnswer(
        search_results=answer_results, k=student_search_results.k
    )
    result_json = student_search_results_and_answer.model_dump_json(indent=4)
    os.makedirs(save_directory, exist_ok=True)
    save_path = save_directory / student_search_results_path.name
    try:
        with save_path.open("w") as f:
            f.write(result_json)
    except OSError as e:
        raise OSError(f"Could not save json to file: {e}.")
