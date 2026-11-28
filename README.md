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
- [x] Day 8: Conversational memory, follow-up query rewriting and SSE streaming
- [x] Day 9: Validated tool schemas, bounded tool-calling loop and guarded execution
- [x] Day 10: Hybrid compute and retrieval agent, deterministic SQL tools and query routing

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


## Day 8: Conversational RAG and Streaming

Day 8 adds multi-turn memory, follow-up query rewriting and Server-Sent Events (SSE) streaming on top of the Day 7 grounded RAG service. Day 7 behavior (citation validation, deterministic refusal) is unchanged.

### Architecture

`question -> ConversationMemory -> QueryRewriter -> HybridRetriever -> context preparation -> streaming structured generation -> citation/refusal validation -> SSE frames`

| Module | Responsibility |
| --- | --- |
| `memory.py` | `ConversationMemory`: sliding window of the last `max_turns` question/answer turns, one instance per session, no global state. |
| `rewriter.py` | `QueryRewriter`: turns a follow-up into a standalone question using history. Skips the LLM call when history is empty and falls back to the original question on LLM errors or invalid, empty or overlong output (`fallback_reason` records why). Never calls the retriever. |
| `json_stream.py` | `AnswerFieldExtractor`: incrementally decodes the `answer` string from streamed JSON, including escapes and surrogate pairs split across chunks. |
| `rag.py` | `RAGService.stream_ask()` streams answer tokens, then runs the same `_build_response()` validation as `ask()`. |
| `conversation.py` | `ConversationalRAG`: composes rewriter, `RAGService` and memory per session. Retrieval and generation receive the rewritten standalone question. |
| `sse.py` | Framework-agnostic SSE formatting and headers. No web framework is required; a FastAPI `StreamingResponse` can wrap it in Day 13. |

Design decisions:

- Memory stores the standalone question, not the raw follow-up, so entities such as invoice numbers stay in the window.
- Stored answers have `[S#]` markers removed, because labels are only valid within one answer.
- Memory is updated only after a successful answer. Abstentions are stored; errors are not.
- A per-session `asyncio.Lock` serializes turns so history cannot interleave.
- Rewriting is separate from retrieval: the rewriter only returns a query string.

### SSE event contract

Responses use `Content-Type: text/event-stream; charset=utf-8`, `Cache-Control: no-cache`, `Connection: keep-alive` and `X-Accel-Buffering: no`. Each frame is `event: <name>` followed by single-line JSON `data:`.

| Event | Payload |
| --- | --- |
| `rewrite` | `{"query": "<standalone question used for retrieval>"}` |
| `token` | `{"text": "<provisional answer text>"}` |
| `final` | Validated answer, citations, `abstained`, `retrieved_count`, `provider`, `latency_s`, `cost_usd` |
| `error` | `{"message": "<reason>"}` |
| `done` | `{}` (always the last frame) |

Unexpected exceptions become a generic `error` frame that contains only the exception type. If the consumer stops reading, the upstream LLM stream is closed.

### Run the Day 8 demo

Requires Ollama (`llama3.2:3b`, `nomic-embed-text`) and the Day 7 collection:

`python -m llmkit.conversation_demo`

Options: `--raw` prints literal SSE frames, `--question` (repeatable) sets custom turns, `--max-turns` sets the memory window. The default conversation asks about invoice `NS-2026-204`, follows up with "What is its total amount?" (rewritten to a standalone question) and then asks an independent question (not rewritten).

### Streaming trade-off

Citations can only be validated once the full JSON object exists. `token` events are therefore provisional: the `final` event is the authoritative result. If an `error` event arrives, clients must discard the streamed text.

### Known limitations

- Citation validation still checks labels, not semantic support. Observed example (before the prompt-example change; see the next item): asked for the vendor of `NS-2026-204`, the 3B model answered "Northstar Supplies" and cited `[S2]`, a chunk that says "NORTHSTAR OFFICE SOLUTIONS". "Northstar Supplies" appears only in a different invoice in `invoices.csv`, which was in the same context. The validation accepted the answer because `[S2]` was a valid label. Claim-level faithfulness measurement is planned for Day 11.
- After the prompt examples were made neutral, the same vendor question again returned "Northstar Supplies", this time citing `[S1]` from `invoices.csv`. That chunk contains the name, but for a different invoice (`INV-2026-101`), so the citation matches the text while the entity is wrong. The prompt-example change did not fix this; retrieved context that mixes two invoices is the more likely cause. In the same run, the follow-up "Who issued it?" was rewritten to a standalone query and correctly answered "NORTHSTAR OFFICE SOLUTIONS", so memory held two contradictory vendor answers. Both observations are single runs on a 3B model, not measured rates.
- Memory stores whatever answer was validated, so an unsupported answer can persist into later turns.
- The query rewriter uses the same small local model as generation and is only tested on a small number of follow-ups.
- Memory is in-process only; there is no HTTP endpoint or persistence yet (Day 13).
- Evaluated on a four-chunk sample corpus and unit tests with fake clients. This is not a production-readiness claim.

## Day 9: Tool-Calling Agent

Day 9 introduces a provider-independent agent loop that selects from registered Python tools, validates their arguments, executes them through a guarded boundary and uses the observations to produce a final answer.

### Architecture

`task -> ToolAgent -> structured action -> ToolRegistry validation -> guarded execution -> tool observation -> next action or final answer`

| Module | Responsibility |
| --- | --- |
| `tools.py` | Defines the `Tool` contract and `ToolRegistry`. Tool names are constrained, duplicate registrations are rejected, and Pydantic argument models reject unexpected fields. |
| `agent.py` | Implements the bounded asynchronous agent loop, structured action parsing, format-error retries, repeated-call detection, observation history and execution trace. |
| `sandbox.py` | Wraps tool execution with timeouts, output truncation and normalized error outcomes. Tool failures are returned as observations instead of escaping the agent loop. |
| `builtin_tools.py` | Provides a restricted AST-based calculator and a file-reading tool confined to a configured root directory. |
| `agent_demo.py` | Runs the live Ollama demonstration and prints the agent trace. `--debug-raw` also displays raw model responses for troubleshooting. |

### Design and safety decisions

- The action format uses the existing LLM client's JSON-schema generation interface rather than provider-specific native tool calling.
- Tool arguments are validated before a handler is invoked. Unknown tools and invalid arguments become controlled execution outcomes.
- The agent has a maximum step budget and a bounded format-error retry budget.
- An identical tool call is not executed again. The previous observation is supplied to the model so it can continue using the result.
- Tool output is treated as untrusted data; instructions inside tool results must not override the agent's rules.
- The calculator evaluates a restricted arithmetic AST. It does not use Python `eval`.
- The file reader resolves paths under its configured root and rejects traversal and symlink escapes.
- File access errors are surfaced without returning file contents. The demo's attempt to read `../../.env` is rejected.

The execution guard is an **in-process application-level boundary**, not an operating-system security sandbox. It does not isolate arbitrary Python code or guarantee interruption of CPU-bound handlers. Only trusted, deliberately registered handlers should be exposed.

### Run the Day 9 demo

Requires Ollama and the configured local chat model:

`python -m llmkit.agent_demo`

Print raw model responses alongside the execution trace:

`python -m llmkit.agent_demo --debug-raw`

The demo uses the calculator and a file reader restricted to `samples/day4/`. Its four default tasks cover arithmetic, invoice-number extraction, checking subtotal plus tax against the total, and refusing a path traversal attempt targeting `.env`.

### Observed live demo results

In the latest observed run:

- `(1234 * 5) + 678` returned `6848`.
- Invoice extraction returned `NS-2026-204`.
- The invoice arithmetic check reported `220.00 + 22.00 = 242.00`.
- The repeated calculator call was skipped, and the agent used the previous observation.
- Reading `../../.env` was rejected; no file contents were exposed.

These are observations from one run with the local 3B model, not reliability measurements. Unit tests use controlled fake clients; live model behavior can vary.

### Known limitations

- The tool boundary is designed for a small allowlist of trusted handlers, not arbitrary untrusted code execution.
- The local model may produce malformed or redundant actions; retries and repeated-call detection limit, but do not eliminate, those behaviors.
- A successful demo run is not a substitute for broader agent evaluation. Task success, tool selection, argument validity and final-answer completeness need a larger test set in later evaluation work.

## Day 10: Hybrid Compute and Retrieval Agent

Day 10 combines deterministic structured-data computation with document retrieval. A query router selects the compute path, retrieval path, or both.

### Architecture

`question -> QueryRouter -> compute | retrieve | both -> HybridAgent -> response`

| Module | Responsibility |
| --- | --- |
| `router.py` | Uses structured LLM output to classify a question as `compute`, `retrieve`, or `both`. Invalid output or LLM failures fall back to retrieval. |
| `sql_tool.py` | Loads the Day 10 invoice CSV into SQLite and exposes fixed, parameterized invoice queries through the existing tool interface. |
| `hybrid_agent.py` | Dispatches to the selected path. The `both` route runs computation and retrieval concurrently and preserves document citations. |
| `hybrid_demo.py` | Connects the router, SQL tool, and Day 7 RAG components for a live demonstration. |

The SQL interface supports invoice details, status summaries, and vendor summaries. Monetary values are stored as integer cents, and totals are grouped by currency to avoid combining unlike currencies. The model does not generate or execute arbitrary SQL.

The demo dataset in `samples/day10/invoices.csv` is synthetic and is separate from the Day 4 document samples.

### Run the Day 10 demo

Requires Ollama (`llama3.2:3b`, `nomic-embed-text`) and the Day 7 Qdrant collection:

`python -m llmkit.hybrid_demo`

Ask a custom question:

`python -m llmkit.hybrid_demo --query "What is the total amount of unpaid invoices in the structured dataset?"`

The default retrieval mode is hybrid search. To enable the optional cross-encoder reranker, install the extra:

`pip install -e ".[rerank]"`

Then run:

`python -m llmkit.hybrid_demo --mode hybrid --rerank --top-k 5`

The `--rerank` option is valid only with `--mode hybrid`. It uses the optional `sentence-transformers` dependency and loads the cross-encoder lazily.

### Observed live results

- A structured unpaid-invoice query selected `compute` and returned **USD 85.80 across two unpaid invoices** from the synthetic SQL dataset.
- A payment-gateway incident question selected `retrieve`. With hybrid retrieval and cross-encoder reranking, the observed answer identified the timeout and retry completion at 09:17:45, citing the application log.
- In a separate retrieval-ranking check, the cross-encoder ranked both chunks for invoice `NS-2026-204` ahead of unrelated chunks for a query asking about its line items and totals.

These are individual observations using the local 3B model, not reliability measurements. Cross-encoder scores are ranking scores, not probabilities.

### Known limitations

- Routing still depends on a small local model and a limited set of deterministic operational-event cues. The fallback to retrieval is intentional when structured routing fails.
- RAG citation validation checks source labels, not whether each claim is semantically supported by the cited passage. Retrieved chunks can contain facts about different invoices, so a valid citation does not guarantee correct entity attribution.
- Prompt instructions to focus on document-relevant facts do not guarantee that the model will exclude unrelated facts from retrieved context.
- The dataset and document collection are small demonstrations, not production financial data. Broader routing, calculation, retrieval, and faithfulness evaluation remains necessary.


## Day 11: RAG Evaluation Harness

Day 11 introduces an evaluation harness for measuring retrieval quality and generated-answer quality across a small, labelled RAG evaluation dataset.

### Architecture

`evaluation dataset -> RAGService -> retrieved context -> generated answer -> metrics and judge -> JSON report`

| Module | Responsibility |
| --- | --- |
| `eval_cases.py` | Defines and validates evaluation cases loaded from JSONL. |
| `eval_context.py` | Calculates deterministic Context Recall and Context Precision using evidence snippets. |
| `eval_judge.py` | Uses an LLM judge to evaluate Faithfulness and Answer Relevance. |
| `eval_harness.py` | Evaluates cases, records errors, aggregates metrics, and writes JSON reports. |
| `eval_harness_demo.py` | Connects the evaluation dataset, Day 7 Qdrant collection, retriever, generator, and optional judge. |

### Evaluation metrics

- **Context Recall:** Fraction of distinct expected evidence snippets found in at least one retrieved chunk.
- **Context Precision:** Fraction of retrieved chunks containing at least one expected evidence snippet. It is `n/a` when no chunks are retrieved.
- **Faithfulness:** Fraction of answer sentences judged as supported by the context supplied to the generator.
- **Answer Relevance:** LLM-judged rating of how directly the answer addresses the question. `full`, `partial`, and `none` map to `1.0`, `0.5`, and `0.0`.
- **False Refusal Rate:** Fraction of answerable cases for which the RAG response has `abstained=true`.
- **Correct Refusal Rate:** Fraction of unanswerable cases for which the RAG response has `abstained=true`.

Before generation, RAG also performs a deterministic identifier preflight. If a question contains a record-like identifier, every requested identifier must appear as a complete identifier token in the retrieved context. A missing or mismatched identifier causes an immediate refusal without calling the generator. This reduces the risk of transferring facts from a different record when the requested identifier is absent from the retrieved context. It does not prove that a generated fact belongs to the requested record when multiple records are present; it is a targeted safeguard, not a general guarantee of semantic correctness.

The harness also reports mean generation latency, generation cost, judge cost, generation errors, and per-case results.

Metric summaries include the number of scored cases and excluded cases. Missing scores and judge failures are not converted into zero scores.

### Dataset format

The evaluation dataset is `samples/day11/eval_cases.jsonl`.

Each non-empty line contains one independent JSON object:

```json
{
  "id": "txt-total-tax",
  "question": "What is the total amount and tax for invoice NS-2026-204?",
  "answerable": true,
  "category": "factual",
  "evidence": [
    "TOTAL: USD 242.00",
    "Tax: USD 22.00"
  ]
}
```

Supported categories are `factual`, `multi_chunk`, `adversarial`, and `out_of_scope`.

Answerable cases require at least one non-empty evidence snippet. Unanswerable cases must have an empty evidence list. Case IDs must be unique.

Evidence snippets are matched against individual indexed chunks after whitespace normalization and case folding.

### Run the Day 11 evaluation

Requires the Day 7 Qdrant collection, Ollama, and the configured local models:

`python -m llmkit.eval_harness_demo`

The default configuration uses:
- Dataset: `samples/day11/eval_cases.jsonl`
- Collection: `day7_recursive`
- Retrieval mode: `hybrid`
- Top-k: `5`
- Generator: configured Ollama model
- Judge: configured Ollama model

Run without an LLM judge to calculate deterministic metrics only:

`python -m llmkit.eval_harness_demo --no-judge`

Choose a retrieval mode:

`python -m llmkit.eval_harness_demo --mode dense`

`python -m llmkit.eval_harness_demo --mode bm25`

`python -m llmkit.eval_harness_demo --mode hybrid`

Use a different top-k value:

`python -m llmkit.eval_harness_demo --mode hybrid --top-k 5`

Configure the judge:

`python -m llmkit.eval_harness_demo --judge-provider ollama --judge-model llama3.2:3b`

A cloud judge can also be selected with `--judge-provider cloud`, provided the corresponding API credentials and configuration are available.

The default judge uses the same configured local model as the generator. The demo prints a warning in this situation because a model grading its own answers may produce overly generous scores.

### Evidence preflight

Before evaluation, the demo checks whether each answerable case's evidence snippets occur in at least one indexed chunk.

If evidence is missing, evaluation stops by default to avoid silently reporting misleading recall scores.

Use `--allow-missing-evidence` to continue deliberately:

`python -m llmkit.eval_harness_demo --allow-missing-evidence`

### Reports

The demo prints progress for individual cases and an aggregated evaluation report.

The default per-case JSON report is written to:

`output/day11/report.json`

Use `--output` to choose another report path.

Generation errors are recorded separately and excluded from metrics that cannot be calculated. Judge failures produce unavailable scores rather than artificial zero scores.

### Known limitations

- Context Recall and Context Precision use exact normalized substring matching, not semantic relevance judgments.
- A retrieved chunk containing one evidence snippet is counted as relevant even if it contains unrelated information.
- Faithfulness and Answer Relevance depend on an LLM judge and can vary between runs.
- Faithfulness uses a simple sentence splitter that may incorrectly split abbreviations.
- The default judge and generator may be the same model, creating self-evaluation bias.
- Refusal rates use the structured `abstained` flag. They do not detect natural-language or soft refusals when the model returns `abstained=false`.
- A judge failure excludes the affected score; it does not establish that the generated answer is correct or incorrect.
- The dataset is small and synthetic. A single run is an observation, not a statistically reliable measurement of production performance.

## Tests

- Run tests: `pytest -q`
- Run lint: `ruff check .`
