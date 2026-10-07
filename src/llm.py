import warnings
from functools import cache
from typing import cast

from optimum.intel import OVModelForCausalLM
from transformers import (
    AutoTokenizer,
    GenerationConfig,
    SentencePieceBackend,
    TokenizersBackend,
)


@cache
def load_model(
    model_id: str = "Qwen/Qwen3-0.6B",
) -> tuple[
    OVModelForCausalLM,
    TokenizersBackend | SentencePieceBackend,
    GenerationConfig,
]:
    """Load LLM and cache it. Used to lazy-load models.

    Parameters
    ----------
    model_id : str
        Model ID.

    Returns
    -------
    tuple[OVModelForCausalLM, TokenizersBackend | SentencePieceBackend,
    GenerationConfig,]
        Model, tokenizer, and config.
    """

    warnings.filterwarnings("ignore")

    model = OVModelForCausalLM.from_pretrained(
        model_id,
        dtype="auto",
        device_map="auto",
        export=True,
        ov_config={
            "KV_CACHE_PRECISION": "u8",
            "DYNAMIC_QUANTIZATION_GROUP_SIZE": "32",
            "PERFORMANCE_HINT": "LATENCY",
        },
    ).to("cpu")

    tokenizer = cast(
        TokenizersBackend | SentencePieceBackend,
        AutoTokenizer.from_pretrained(model_id),
    )

    generation_config = GenerationConfig(
        max_new_tokens=32768,
        # do_sample=True,
        # num_beams=5,
        eos_token_id=model.config.eos_token_id,
        # repetition_penalty=1.5,
        # temperature=0.2,
        # exponential_decay_length_penalty=(50, 0.01),
        max_time=60.0,
    )

    return (model, tokenizer, generation_config)
