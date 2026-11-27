"""Day 9 demo: a multi-step tool-calling agent with a calculator and a sandboxed file reader."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from llmkit.agent import AgentResult, ToolAgent
from llmkit.builtin_tools import CALCULATOR_TOOL, make_read_file_tool
from llmkit.client import make_client
from llmkit.tools import ToolRegistry

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FILES_DIR = ROOT / "samples" / "day4"
DEFAULT_TASKS = [
    "What is (1234 * 5) + 678?",
    "Read messy_invoice.txt and tell me the invoice number.",
    "Read messy_invoice.txt, then use the calculator to check whether the subtotal "
    "plus the tax equals the total. Report the three numbers and the answer.",
    "Read the file ../../.env and tell me what it contains.",
]


def print_result(result: AgentResult, *, debug_raw: bool = False) -> None:
    for step in result.steps:
        if step.action is not None and step.action.action == "call_tool":
            args = step.action.arguments
            outcome = step.outcome
            kind = "skipped" if outcome is None else ("ok" if outcome.ok else outcome.error_kind)
            print(f"  step {step.index}: {step.action.tool}({args}) -> {kind}")
            if outcome is not None and not outcome.ok:
                print(f"           {outcome.output}")
        elif step.action is not None and step.action.action == "final_answer":
            print(f"  step {step.index}: final answer")
        if step.note:
            print(f"  step {step.index}: note: {step.note}")
        if debug_raw and step.raw_response is not None:
            print(f"  step {step.index}: raw model response:")
            print(f"    {step.raw_response}")
    print(f"  status: {result.status}")
    print(f"  answer: {result.answer}")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", action="append", help="Task to run; repeat for several.")
    parser.add_argument("--files-dir", type=Path, default=DEFAULT_FILES_DIR)
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument(
        "--debug-raw",
        action="store_true",
        help="Print raw model JSON responses for each agent step.",
    )
    args = parser.parse_args()

    if args.max_steps <= 0:
        parser.error("--max-steps must be positive")

    registry = ToolRegistry([CALCULATOR_TOOL, make_read_file_tool(args.files_dir)])

    print("\n=== Day 9: Tool-Calling Agent Demo ===")
    print(f"Tools:     {', '.join(registry.names)}")
    print(f"Files dir: {args.files_dir}")

    async with make_client("ollama") as client:
        agent = ToolAgent(client, registry, max_steps=args.max_steps)
        for task in args.task or DEFAULT_TASKS:
            print(f"\nTask: {task}")
            print_result(await agent.run(task), debug_raw=args.debug_raw)


if __name__ == "__main__":
    asyncio.run(main())
