_This project has been created as part of the 42 curriculum by fsowinsk_

# Description

RagAgainstTheMachines is a project where the goal is to create a CLI tool that
can perform the following operations:

- Indexing files from a third-party project (vllm). The files are split
into smaller chunks with the algorithm of our choosing and then stored in an index
containing a filepath, starting character index, and ending character index for
each source.

- Search a query or dataset. The command will take a question or a dataset of
questions and search the index for `k` sources with the best match scores.
Here, I'm using the `BM25` algorithm to calculate the sources' scores.

- Answer the questions with LLM. This will generate answers to a question
or a dataset of questions using the retrieved sources.

- Evaluate the retrieved sources against the ground-truth sources. This
calculates the `recall@k` score for all sources in a dataset. 

### Examples

Searching sources:

```zsh
$ uv run -m src search "What hardware platforms does vLLM support?" -k 5
data/raw/vllm-0.10.1/vllm/attention/ops/triton_flash_attention.py [0:973]
data/raw/vllm-0.10.1/tests/quantization/test_ptpc_fp8.py [919:2449]
data/raw/vllm-0.10.1/tests/quantization/test_fp8.py [4819:6673]
data/raw/vllm-0.10.1/vllm/config/__init__.py [168143:168698]
data/raw/vllm-0.10.1/vllm/executor/ray_utils.py [1398:2589]
```

Generating answer:

```zsh
$ uv run -m src answer "What hardware platforms does vLLM support?"
VLLM supports a variety of hardware platforms, including:

- **CUDA**: Supports all supported CUDA platforms (e.g., ROCm, NVIDIA, AMD).
- **ROCm**: Supports GPUs with hardware support for FP8 calculations (e.g., MI300X and above).
- **NVIDIA**: Supports all supported NVIDIA GPUs.
- **AMD**: Supports GPUs with hardware support for FP8 calculations.

The platform is designed to work seamlessly across these environments.
```


## System architecture

The RAG pipeline consists of three steps:

1. Indexing documents.
2. Ranking and sources retrieval.
3. Answer generation with attached sources. 

In the indexing stage, the program will scan the folder with raw sources,
read all files with the right extensions, and split their content into smaller
chunks. Those chunks are saved to a JSON file as a filepath,
starting character index, and ending character index. Word count and word
occurrence are computed and saved with the source to improve the performance of
the retrieval algorithm. For all sources, there is also a number of documents
containing a word, average word count, and total number of sources, also for
the retrieval algorithm.

Sources are grouped by their filepath with an attached
modification timestamp. This allows updating the index by only looking for
files with a modification timestamp that is newer than the one in the index.

When searching, the index is loaded from a JSON file to find sources that best
match the query. All sources are pushed on a binary heap using their `bm25`
score as a key, and only a selected number of sources are kept. Because all
necessary data for the retrieval algorithm is already attached to the sources,
no files are read except for the index and query dataset. Search results and
original query are saved to a JSON file.

To generate answers, a JSON file with queries and retrieved sources is loaded.
Each query and its sources are formatted into a prompt that is used to infer an
answer with an LLM. Those answers are saved to a JSON file with sources and the
original query. Because for this project I have to use a CPU, the inference is
slow and can take even 40 minutes for the whole dataset. To speed it up, before
inference starts, I convert the model into an OpenVINO IR format to make it
more performant on Intel CPUs.

## Chunking algorithm

In the chunker implementation, I had to split documents into chunks of size up
to the max character count. I decided to use a recursive algorithm because it's
simple to implement, respects semantic boundaries, and can be easily adapted to
chunk different types of documents. It works by gradually splitting text using a
hierarchy of delimiters. First, the text is split with the first delimiter.
Then, for the chunks that are still too big, the algorithm will move to the
next delimiter and repeat the process.

After being split, some of the chunks will have a size that is too small to be
effectively retrieved. To avoid this, after splitting a chunk, I check the
average size of the chunks created and merge them into bigger chunks if they
are below the threshold.

I had to implement a chunker for Python and markdown files. For Python files I'm
splitting them first by free function and class declarations and later by
method declarations and inner classes. For markdown files, I'm using headers
from the biggest to lowest size. All files have a fallback that will split them
by empty lines (start of a paragraph), newline and finally by each character.
  
## Performance analysis

Recall@k scores are as follows:

 k   | markdown | python 
:---:|:--------:|:------:
 1   |  0.630   |  0.465  
 2   |  0.780   |  0.606  
 3   |  0.820   |  0.646  
 5   |  0.870   |  0.677  
 10  |  0.890   |  0.768  

Scores are calculated for search results limited only to the specified formats.
Search results are counted as accurate if their IoU is higher than 0.05 with
a reference source.

A markdown search performs better than a Python search. A possible reason might
be the difference in number of sources. There are 853 markdown and text file
sources after indexing, but for Python there are 16018 sources. It makes it
less likely that a source retrieved from a Python file will be overlapping with
the expected source from the reference.

Indexing takes less than 5 seconds on 1969 files with an average word count of
1294. Searching takes less than 10 seconds on the index of 16871 sources with
an average word count of 151 words. If the `--processes max` flag is selected,
which enables using multiple processes, then searching takes less than 5
seconds.


# Instructions

Install with:

```zsh
uv sync
```

Index files:

```zsh
uv run -m src index
```

Search a single query (5 best sources):

```zsh
uv run -m src search "What hardware platforms does vLLM support?" -k 5
```

Search index for a whole dataset:

```zsh
uv run -m src search_dataset\
 --dataset-path "data/datasets/UnansweredQuestions/dataset_docs_public.json"\
 --save-directory "data/output/search_results/UnansweredQuestions" 
```

Answer a single query:

```zsh
uv run -m src answer "What hardware platforms does vLLM support?"
```

Answer questions from a dataset:

```zsh
uv run -m src answer_dataset\
 --student-search-results-path "data/output/search_results/UnansweredQuestions/dataset_docs_public.json"\
 --save-directory "data/output/search_results_and_answer/UnansweredQuestions"
```

Evaluate retrieved sources against the reference:

```zsh
uv run -m src evaluate\
 --student-search-results-path "data/output/search_results/UnansweredQuestions/dataset_docs_public.json"\
 --dataset-path "data/datasets/AnsweredQuestions/dataset_docs_public.json"
```

For more options you can add `-h` flag after the subcommand name, e.g.:

```zsh
uv run -m src index -h
```

# Resources

  - [en.wikipedia.org](https://en.wikipedia.org/wiki/Okapi_BM25) - `Okapi bm25` algorithm.
  - [myengineeringpath.dev](https://myengineeringpath.dev/genai-engineer/rag-chunking/) - Reqursive chunker implementation. 
  - [huggingface.co/Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) quickstart example.
  - [docs.openvino.ai](https://docs.openvino.ai/2026/openvino-workflow-generative/inference-with-optimum-intel.html) - inference with Optimum Intel.

  _No AI was used in the making of this project._
  
