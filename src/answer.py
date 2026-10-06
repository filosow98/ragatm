from llm import load_model
from rag import MinimalSource
from search import search_inner

READER_MODEL_NAME = "Qwen/Qwen3-0.6B"


def format_prompt(
    query: str, sources: list[MinimalSource]
) -> list[dict[str, str]]:
    """Format start of the prompt with query string, search results and start
    of the answer."""

    prompt = ""
    for s in sources:
        with open(s.file_path, "r") as f:
            f.seek(s.first_character_index)
            context = f.read(s.last_character_index - s.first_character_index)
        prompt += f"{context}\n"
    prompt += (
        "\nYou are a helpful assistant that answers the user's query"
        + " using the text above."
    )
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": query},
    ]
    return messages


# This code is based on the quickstart example from
# https://huggingface.co/Qwen/Qwen3-0.6B


def answer_inner(
    query: str,
    k: int,
    index: str,
    extensions: str,
) -> str:
    """Answer the query with the retrieved context.

    Parameters
    ----------
    query : str
        Query question.
    k : int
        Number of sources to find.
    index : str
        Path to a sources index.
    extensions : str
        File types to use to generate answer.

    Returns
    -------
    str
        Answer to the query.
    """

    (model, tokenizer, generation_config) = load_model()
    sources: list[MinimalSource] = search_inner(query, k, index, extensions)
    messages = format_prompt(query, sources)
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
    start = len(input_ids.input_ids[0])
    output_ids = generated_ids[0][start:].tolist()

    result = tokenizer.decode(output_ids, skip_special_tokens=True)
    if isinstance(result, str):
        return result.strip("\n")
    else:
        raise TypeError("Tokenizer in 'answer_inner' must return string.")
