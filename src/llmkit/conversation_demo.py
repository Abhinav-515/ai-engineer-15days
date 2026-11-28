"""Day 8 demo: multi-turn grounded RAG with query rewriting and SSE streaming."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from llmkit.client import make_client
from llmkit.conversation import ConversationalRAG
from llmkit.embeddings import OllamaEmbedder
from llmkit.hybrid import HybridRetriever
from llmkit.memory import ConversationMemory
from llmkit.rag import RAGService
from llmkit.rewriter import QueryRewriter
from llmkit.sse import sse_stream
from llmkit.vector_store import QdrantVectorStore

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = ROOT / "output" / "day7" / "qdrant"
DEFAULT_COLLECTION = "day7_recursive"
DEFAULT_QUESTIONS = [
    "What is the payment status of invoice NS-2026-204?",
    "What is its total amount?",
    "Who is the vendor of invoice NS-2026-204?",
]


def parse_frame(frame: str) -> tuple[str, dict]:
    """Split one SSE frame into its event name and decoded JSON payload."""
    event_line, data_line = frame.strip().split("\n")
    return event_line[len("event: ") :], json.loads(data_line[len("data: ") :])


async def run_turn(conversation: ConversationalRAG, question: str, raw: bool) -> None:
    print(f"\nUser: {question}")
    async for frame in sse_stream(conversation.stream_ask(question)):
        if raw:
            print(frame, end="", flush=True)
            continue

        name, data = parse_frame(frame)
        if name == "rewrite":
            changed = data["query"] != question
            print(f"[search query{' (rewritten)' if changed else ''}] {data['query']}")
            print("Assistant: ", end="", flush=True)
        elif name == "token":
            print(data["text"], end="", flush=True)
        elif name == "final":
            print()
            for citation in data["citations"]:
                print(
                    f"  {citation['label']} {Path(citation['document']).name} "
                    f"| chunk={citation['chunk_index']}"
                )
            if data["abstained"]:
                print("  (abstained: no citations)")
            print(f"  generation: {data['latency_s']:.2f}s | retrieved={data['retrieved_count']}")
        elif name == "error":
            print(f"\n  ERROR: {data['message']}")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--question",
        action="append",
        help="Question for one turn; repeat for multiple turns (default: invoice scenario).",
    )
    parser.add_argument("--raw", action="store_true", help="Print raw SSE frames.")
    parser.add_argument("--max-turns", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--db-dir", type=Path, default=DEFAULT_DB)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    args = parser.parse_args()

    if args.max_turns <= 0 or args.top_k <= 0:
        parser.error("--max-turns and --top-k must be positive")

    questions = args.question or DEFAULT_QUESTIONS

    print("\n=== Day 8: Conversational RAG Demo ===")
    print(f"Database:   {args.db_dir}")
    print(f"Collection: {args.collection}")

    with QdrantVectorStore(
        args.db_dir,
        collection_name=args.collection,
        vector_size=768,
    ) as store:
        if not store.scroll_all():
            raise RuntimeError(
                f"Collection '{args.collection}' is empty. Run the Day 7 ingestion first."
            )

        async with OllamaEmbedder() as embedder, make_client("ollama") as client:
            retriever = HybridRetriever(store, embedder)
            conversation = ConversationalRAG(
                RAGService(retriever, client, top_k=args.top_k),
                QueryRewriter(client),
                ConversationMemory(max_turns=args.max_turns),
            )

            for question in questions:
                await run_turn(conversation, question, args.raw)

            print("\n" + "=" * 68)
            print(f"MEMORY ({len(conversation.memory)}/{args.max_turns} turns retained)")
            print("=" * 68)
            print(conversation.memory.format_for_prompt())


if __name__ == "__main__":
    asyncio.run(main())
