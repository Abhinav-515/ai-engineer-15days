"""Tests for the Day 10 hybrid demo CLI validation."""

import sys

import pytest

from llmkit.hybrid_demo import main


@pytest.mark.asyncio
async def test_rerank_rejects_non_hybrid_mode(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["hybrid_demo", "--mode", "bm25", "--rerank"],
    )

    with pytest.raises(SystemExit) as exc_info:
        await main()

    assert exc_info.value.code == 2
    assert "--rerank can only be used with --mode hybrid" in capsys.readouterr().err
