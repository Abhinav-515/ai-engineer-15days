"""Checks that the seed evaluation dataset matches the real sample corpus."""

from pathlib import Path

from llmkit.eval_cases import load_eval_cases
from llmkit.eval_context import normalize_for_match
from llmkit.loaders import DocumentError, load_document

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "samples" / "day11" / "eval_cases.jsonl"
CORPUS_DIR = ROOT / "samples" / "day4"


async def load_corpus_text() -> str:
    parts = []
    for path in sorted(CORPUS_DIR.iterdir()):
        try:
            document = await load_document(path)
        except DocumentError:
            continue  # empty or unsupported files contribute nothing
        parts.append(normalize_for_match(document.text))
    return "\n".join(parts)


def test_seed_dataset_loads_and_validates():
    cases = load_eval_cases(DATASET)

    assert len(cases) == 8
    assert len({case.id for case in cases}) == len(cases)


def test_seed_dataset_covers_answerable_and_unanswerable_cases():
    cases = load_eval_cases(DATASET)

    assert any(case.answerable for case in cases)
    assert any(not case.answerable and case.category == "out_of_scope" for case in cases)
    assert any(not case.answerable and case.category == "adversarial" for case in cases)
    assert any(case.category == "multi_chunk" for case in cases)


async def test_every_evidence_snippet_occurs_in_the_sample_corpus():
    corpus = await load_corpus_text()
    missing = [
        (case.id, snippet)
        for case in load_eval_cases(DATASET)
        for snippet in case.evidence
        if normalize_for_match(snippet) not in corpus
    ]

    assert missing == [], f"evidence snippets not found in corpus: {missing}"


async def test_corpus_is_not_empty():
    assert len(await load_corpus_text()) > 100
