# AI Engineer 15-Day Assignment

Applied Generative AI, RAG & Agentic Workflows.

**Author:** Abhinav Talluri

## Setup

1. Create and activate a virtual environment: `python3 -m venv .venv` and `source .venv/bin/activate`.
2. Install dependencies: `pip install -e ".[dev]"`.
3. Copy `.env.example` to `.env` and configure your cloud API key if needed.
4. Never commit `.env`.

## Project Progress

- [x] Day 1: Environment, configuration, secrets, linting and async smoke test
- [x] Day 2: Unified async LLM client and local-vs-cloud benchmarking
- [x] Day 3: Structured extraction, Pydantic validation and provider fallback
- [x] Day 4: Document loaders, preprocessing, resilient batch pipeline, log parsing and exports
- [x] Day 5: Vector embeddings, chunking strategies and Qdrant ingestion
- [x] Day 6: BM25, hybrid retrieval, cross-encoder reranking and retrieval evaluation
- [x] Day 7: Grounded RAG, structured generation, source citations and refusal validation

## Day 4: Document Processing Pipeline

Features include PDF, CSV, TSV, TXT and LOG loading; text preprocessing; deterministic table extraction; schema-driven LLM extraction with Pydantic validation; retry and fallback handling; per-document error isolation; regex-first log parsing; Pandas CSV exports; and SQLite storage.

### Run the demo

- Local: `python -m llmkit.pipeline_demo --providers ollama`
- Cloud: `python -m llmkit.pipeline_demo --providers cloud`
- Both: `python -m llmkit.pipeline_demo --providers ollama cloud`

The demo reads fixtures from `samples/day4/` and writes generated files to `output/day4/`. Generated outputs are excluded from Git.

See `docs/day4_demo_report.md` for observed results.



## Day 5: Vector Embeddings and Ingestion

Day 5 introduces dense text embeddings, three chunking strategies, and vector storage with Qdrant.

### Architecture

`load_document -> chunking -> OllamaEmbedder -> QdrantVectorStore -> similarity search`

- **Fixed-size chunking:** splits text into character windows with configurable overlap.
- **Recursive chunking:** tries progressively smaller separators to preserve text structure before falling back to character windows.
- **Semantic chunking:** groups neighboring sentences using embedding similarity and a configurable threshold.
- **Embeddings:** uses Ollama's `nomic-embed-text` model to convert text into 768-dimensional dense vectors.
- **Vector database:** stores chunk text, embeddings, and source metadata in a local persistent Qdrant collection.
- **Search:** retrieves the most similar chunks using cosine similarity.

Dense embeddings represent text as numerical vectors. Like engineered features in traditional machine learning, these vectors provide numerical representations for downstream comparison. Unlike manually selected columns, embedding dimensions are learned representations that capture aspects of semantic meaning.

### Setup

1. Install the project dependencies:

   `pip install -e ".[dev]"`

2. Install and start [Ollama](https://ollama.com/) if it is not already running.

3. Download the embedding model:

   `ollama pull nomic-embed-text`

4. Confirm Ollama is available at `http://localhost:11434`.

The embedding model can be configured with `OLLAMA_EMBEDDING_MODEL` in `.env`; the default is `nomic-embed-text`.

### Run the ingestion demo

Run all chunking strategies:

`python -m llmkit.ingestion_demo --strategies fixed recursive semantic`

Run one strategy:

`python -m llmkit.ingestion_demo --strategies recursive`

The demo loads sample documents from `samples/day4/`, chunks and embeds them, stores vectors in local Qdrant collections, and displays similarity-search results.

Generated vector database files are written under `output/day5/` and excluded from Git.

### Current scope and evaluation

Day 5 provides the document ingestion and dense-vector retrieval foundation. Day 6 extends it with lexical retrieval, hybrid ranking, reranking and retrieval-quality evaluation.

## Day 6: Hybrid Search and Evaluation

### Architecture

`query -> dense retrieval + BM25 retrieval -> rank fusion -> optional cross-encoder reranking`

- **BM25:** implements lexical retrieval using term frequency and inverse document frequency, useful for exact terms, identifiers and keyword matching.
- **Dense retrieval:** uses the Day 5 Ollama embeddings and Qdrant cosine similarity search.
- **Rank fusion:** combines dense and lexical rankings using Reciprocal Rank Fusion (RRF) or weighted score fusion.
- **Hybrid retriever:** supports `dense`, `bm25`, `hybrid` and `hybrid+rerank` retrieval modes.
- **Cross-encoder reranking:** optionally reranks retrieved candidates using `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- **Index refresh:** rebuilds the lexical index when the underlying vector-store documents change.

### Evaluation

The evaluation module implements:

- Precision@k
- Recall@k
- Reciprocal Rank (RR)
- Mean Reciprocal Rank (MRR)

Run the evaluation against the existing recursive-chunking Qdrant collection:

`python -m llmkit.evaluation_demo`

Include the cross-encoder reranker:

`python -m llmkit.evaluation_demo --rerank`

The current smoke benchmark uses five manually labelled queries and four stored chunks from the Day 4 sample documents.

| Retrieval strategy | Precision@1 | Recall@1 | Precision@3 | Recall@3 | MRR |
|---|---:|---:|---:|---:|---:|
| Dense | 0.600 | 0.600 | 0.333 | 1.000 | 0.767 |
| BM25 | 0.800 | 0.800 | 0.333 | 1.000 | 0.900 |
| Hybrid RRF | 0.800 | 0.800 | 0.333 | 1.000 | 0.867 |
| Hybrid + reranker | 0.800 | 0.800 | 0.333 | 1.000 | 0.867 |

### Findings and limitations

- Hybrid RRF improves Precision@1 and Recall@1 from 0.600 to 0.800 compared with dense retrieval.
- Hybrid RRF improves MRR from 0.767 to 0.867 compared with dense retrieval.
- BM25 achieves the highest MRR (0.900) on this benchmark.
- Cross-encoder reranking does not improve MRR over hybrid RRF on this dataset.
- All strategies retrieve every labelled relevant chunk within the top three results, so Recall@3 is 1.000 for each strategy.

These results are an initial functional benchmark, not a statistically robust performance claim. The corpus contains only four chunks and five queries, with one relevant chunk per query. Precision@3 and Recall@3 therefore have limited ability to distinguish the strategies. A larger labelled evaluation set is needed to establish general retrieval-quality improvements.


## Day 7: Production RAG with Citations

Day 7 connects the Day 6 hybrid retriever to structured LLM generation to produce grounded answers with validated source references.

### Architecture

`question -> HybridRetriever -> context preparation -> structured LLM generation -> citation validation -> grounded response`

- **Hybrid retrieval:** retrieves relevant document chunks using dense and BM25 search with rank fusion.
- **Context preparation:** formats retrieved chunks with source metadata and stable per-response labels such as `[S1]`, `[S2]` and `[S3]`.
- **Structured generation:** uses a Pydantic `GeneratedAnswer` schema to require an answer, citation list and abstention flag.
- **Inline citations:** instructs the model to cite the specific chunk supporting each factual claim, including when facts come from different chunks.
- **Citation validation:** rejects missing, unknown, duplicated or inconsistent citation labels and malformed structured responses.
- **Deterministic refusal:** returns a fixed refusal without calling the LLM when retrieval returns no chunks.
- **Source metadata:** returns the cited document, chunk index, document kind and chunking strategy.

Plain-text `.txt` files are classified as `text`; `.log` files retain the `log` classification.

### Run the Day 7 ingestion demo

Build the Day 7 recursive-chunking collection:

`python -m llmkit.ingestion_demo --strategies recursive --db-dir output/day7/qdrant --collection-prefix day7`

This stores the sample document chunks in the local `day7_recursive` Qdrant collection.

### Run the grounded RAG demo

`python -m llmkit.rag_demo`

The default demo reads from `output/day7/qdrant` and `day7_recursive`. It asks an invoice question and prints the grounded answer, source citations and execution summary.

### Citation validation and limitations

The application validates that inline citation labels refer to retrieved chunks and that the inline labels match the structured citations list. It also rejects malformed responses and invalid abstention/citation combinations.

Citation-label validation does not independently prove semantic entailment: it does not guarantee that every generated claim is factually supported by its cited chunk. The prompt encourages claim-level citations, but semantic verification remains a limitation.

The current demonstration uses a small local sample corpus. Larger document collections and a broader evaluation set are needed for production-quality reliability measurements.


## Tests

- Run tests: `pytest -q`
- Run lint: `ruff check .`
