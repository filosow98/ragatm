_This project has been created as part of the 42 curriculum by fsowinsk_

# Description

RagAgainstTheMachines is a project where the goal is to create a CLI tool that
can perform following operations:

- Indexing files from a third party project (here vllm). The files are split
into smaller chunks with algorithm of our choosing and then stored in an index
containing a filepath, starting character index and ending character index for
each source.

- Search query or a dataset. The command will take a question or a dataset of
questions and search the index for `k` sources with with the best match scores.
Here, I'm using `bm25` algorithm to calculate the sources' scores.

- Answer the questions with LLM. This will generate an answers to a question
or a dataset of questions using the retrieved sources.

- Evaluate the retrieved sources against the ground-truth sources. This
calculates `recall@k` score for all sources in a dataset. 

# Instructions

Instll with:

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

  - [Okapi BM25]{https://en.wikipedia.org/wiki/Okapi_BM25} algorithm for
  sources retrieval.
  - [Reqursive chunking]{https://myengineeringpath.dev/genai-engineer/rag-chunking/} for chunker implementation. 

# System architecture

# Chunking strategy

In the chunker implementation, I had to split documents into chunks of size up
to the max character count. I decided to use reqursive algorithm because it's
simple to implement, respects semantic boundaries, and can be easily adapted
to chunk different types of documents. It works by gradualy spliting text
using hierarchy of delimiters. First, text will be split with a delimiter that
producess chunks that are the least related to one another, i.e., the most
of the semantic information in those chunks is self-contained. Then, for the
chunks that are still too big, the algorithm will move to the next delimiter
and repeat the process.

After being split, some of the chunks will have size that is too small to be
effectively retrived. To avoid this, after splitting a chunk, I check the
average size of the chunks created and merge them into bigger chunks if they
are bellow the treshold.

We had to implement chunker for python markdown files. For python files I'm
splitting them first by free function and class declarations and later by
method declarations and inner classes. For markdown files, I'm using headers
from the biggest to lowest size. All files have a fallback that will split them
by empty lines (start of a paragraph), a newline and finaly by characters.

  
  
# Retrieval method

# Performance analysis

# Design decisions

# Challenges faced

# Example usage
