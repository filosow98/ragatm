import json

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextGenerationPipeline,
    pipeline,
)

from ragatm import Source
from search import _search

READER_MODEL_NAME = "Qwen/Qwen3-0.6B"


def _format_prompt(query: str, sources: list[Source]) -> str:
    """Format start of the prompt with query, search results and start of the
    answer until the starting quote mark."""

    prompt = '{"context":['
    sep = ""
    for s in sources:
        prompt += sep
        sep = ","
        with open(s.file_path, "r") as f:
            f.seek(s.first_character_index)
            context = f.read(s.last_character_index - s.first_character_index)
        prompt += (
            r"{"
            + f'"file_path":{json.dumps(s.file_path)},"content":{json.dumps(context)}'
            + r"}"
        )
    prompt += f'],"query": {json.dumps(query)}, "answer":"'
    return prompt


model = AutoModelForCausalLM.from_pretrained(READER_MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(READER_MODEL_NAME)
generator: None | TextGenerationPipeline = None


def _answer(
    query: str, k: int = 1, index: str = "data/processed/index.json"
) -> str:
    """Answer the query with the retrieved context.

    Parameters
    ----------
    query : str
        Query question.
    k : int, default=1
        Number of sources to find.
    index : str, default="data/processed/index.json"
        Path to a sources index.

    Returns
    -------
    str
        Answer to the query.
    """
    sources = _search(query, k, index)
    prompt = _format_prompt(query, sources)
    global generator
    if not generator:
        generator = pipeline(
            model=model,
            tokenizer=tokenizer,
            task="text-generation",
            max_new_tokens=512,
            return_full_text=True,
        )
    output = generator(prompt)

    return str(output)
