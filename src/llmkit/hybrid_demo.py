"""Day 10 demo: route questions between invoice SQL and grounded RAG."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from llmkit.agent import ToolAgent
from llmkit.client import make_client
from llmkit.embeddings import OllamaEmbedder
from llmkit.hybrid import HybridRetriever
from llmkit.hybrid_agent import HybridAgent
from llmkit.rag import RAGService
from llmkit.reranker import CrossEncoderReranker
from llmkit.router import QueryRouter
from llmkit.sql_tool import InvoiceDatabase, make_invoice_query_tool
from llmkit.tools import ToolRegistry
from llmkit.vector_store import QdrantVectorStore

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = ROOT / "output" / "day7" / "qdrant"
DEFAULT_COLLECTION = "day7_recursive"
DEFAULT_INVOICES = ROOT / "samples" / "day10" / "invoices.csv"


def print_result(result) -> None:
    """Print the route, answer, citations, and execution summary."""
    print("\n" + "=" * 68)
    print("ROUTING DECISION")
    print("=" * 68)
    print(f"Route:       {result.routing.route}")
    print(f"Confidence:  {result.routing.confidence:.2f}")
    print(f"Reason:      {result.routing.reason}")
    if result.routing.fallback_reason:
        print(f"Fallback:    {result.routing.fallback_reason}")

    print("\n" + "=" * 68)
    print("HYBRID ANSWER")
    print("=" * 68)
    print(result.answer)

    print("\n" + "=" * 68)
    print("SOURCE CITATIONS")
    print("=" * 68)
    if result.citations:
        for citation in result.citations:
            print(
                f"{citation.label} {Path(citation.document).name} "
                f"| chunk={citation.chunk_index} "
                f"| kind={citation.kind} "
                f"| strategy={citation.strategy}"
            )
    else:
        print("No document citations.")

    if result.compute_result is not None:
        print("\n" + "=" * 68)
        print("COMPUTE EXECUTION TRACE")
        print("=" * 68)
        for step in result.compute_result.steps:
            if step.action is not None:
                if step.action.action == "call_tool":
                    outcome = step.outcome
                    status = (
                        "skipped"
                        if outcome is None
                        else ("ok" if outcome.ok else outcome.error_kind)
                    )
                    print(
                        f"Step {step.index}: {step.action.tool}"
                        f"({step.action.arguments}) -> {status}"
                    )
                else:
                    print(f"Step {step.index}: final answer")
            if step.note:
                print(f"  Note: {step.note}")
        print(f"Agent status: {result.compute_result.status}")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--query",
        default="What is the total amount of unpaid invoices?",
        help="Question to route between invoice computation and document retrieval.",
    )
    parser.add_argument(
        "--invoice-csv",
        type=Path,
        default=DEFAULT_INVOICES,
        help="Structured invoice CSV dataset.",
    )
    parser.add_argument("--db-dir", type=Path, default=DEFAULT_DB)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument(
        "--mode",
        choices=["dense", "bm25", "hybrid"],
        default="hybrid",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Rerank hybrid candidates with the optional cross-encoder.",
    )
    parser.add_argument("--max-steps", type=int, default=6)
    args = parser.parse_args()

    if args.top_k <= 0 or args.max_steps <= 0:
        parser.error("--top-k and --max-steps must be positive")
    if args.rerank and args.mode != "hybrid":
        parser.error("--rerank can only be used with --mode hybrid")

    if not args.invoice_csv.is_file():
        parser.error(f"Invoice CSV does not exist: {args.invoice_csv}")

    print("\n=== Day 10: Hybrid Compute + Retrieval Agent ===")
    print(f"Question:   {args.query}")
    print(f"Invoices:   {args.invoice_csv}")
    print(f"Qdrant DB:  {args.db_dir}")
    print(f"Collection: {args.collection}")
    retrieval_mode = "hybrid+rerank" if args.rerank else args.mode
    print(f"Retrieval:  {retrieval_mode}")

    with InvoiceDatabase(args.invoice_csv) as database:
        registry = ToolRegistry([make_invoice_query_tool(database)])

        with QdrantVectorStore(
            args.db_dir,
            collection_name=args.collection,
            vector_size=768,
        ) as store:
            documents = store.scroll_all()
            if not documents:
                raise RuntimeError(
                    f"Collection '{args.collection}' is empty. Run the Day 5 ingestion demo first."
                )

            print(f"Indexed document chunks: {len(documents)}")

            async with OllamaEmbedder() as embedder:
                reranker = CrossEncoderReranker() if args.rerank else None
                retriever = HybridRetriever(
                    store,
                    embedder,
                    reranker=reranker,
                )

                async with make_client("ollama") as client:
                    router = QueryRouter(client)
                    tool_agent = ToolAgent(
                        client,
                        registry,
                        max_steps=args.max_steps,
                    )
                    rag_service = RAGService(
                        retriever,
                        client,
                        retrieval_mode=retrieval_mode,
                        top_k=args.top_k,
                    )
                    agent = HybridAgent(router, tool_agent, rag_service)

                    result = await agent.ask(args.query)

    print_result(result)


if __name__ == "__main__":
    asyncio.run(main())
