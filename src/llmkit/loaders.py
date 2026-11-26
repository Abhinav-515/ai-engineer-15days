"""Document loaders: PDF, table and log files -> RawDocument."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import pdfplumber


class DocumentError(ValueError):
    """A document could not be loaded or contained nothing usable."""


@dataclass
class RawDocument:
    source: str
    kind: str
    text: str
    tables: list[list[list[str]]] = field(default_factory=list)
    page_count: int | None = None
    warnings: list[str] = field(default_factory=list)


def load_pdf(path: str | Path) -> RawDocument:
    path = Path(path)
    try:
        with pdfplumber.open(path) as pdf:
            page_texts: list[str] = []
            tables: list[list[list[str]]] = []
            for page in pdf.pages:
                page_texts.append(page.extract_text() or "")
                for table in page.extract_tables():
                    tables.append([[(cell or "").strip() for cell in row] for row in table])
            page_count = len(pdf.pages)
    except Exception as error:
        raise DocumentError(f"cannot read PDF {path.name}: {error}") from error

    text = "\n\n".join(t for t in page_texts if t.strip())
    if not text.strip():
        raise DocumentError(f"{path.name}: no extractable text (scanned or empty PDF?)")
    return RawDocument(
        source=str(path), kind="pdf", text=text, tables=tables, page_count=page_count
    )


def load_table(path: str | Path) -> RawDocument:
    path = Path(path)
    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    try:
        df = pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False, on_bad_lines="skip")
    except Exception as error:
        raise DocumentError(f"cannot read table {path.name}: {error}") from error

    if df.empty:
        raise DocumentError(f"{path.name}: table has no data rows")
    df.columns = [str(c).strip() for c in df.columns]
    df = df.apply(lambda col: col.str.strip())
    rows = [list(df.columns), *df.values.tolist()]
    return RawDocument(
        source=str(path),
        kind="table",
        text=df.to_csv(index=False, lineterminator="\n"),
        tables=[rows],
    )


def load_log(path: str | Path) -> RawDocument:
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as error:
        raise DocumentError(f"cannot read log {path.name}: {error}") from error
    if not text.strip():
        raise DocumentError(f"{path.name}: log is empty")
    return RawDocument(source=str(path), kind="log", text=text)


def load_text(path: str | Path) -> RawDocument:
    """Load a plain-text document."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as error:
        raise DocumentError(f"cannot read text file {path.name}: {error}") from error
    if not text.strip():
        raise DocumentError(f"{path.name}: text file is empty")
    return RawDocument(source=str(path), kind="text", text=text)


_LOADERS = {
    ".pdf": load_pdf,
    ".csv": load_table,
    ".tsv": load_table,
    ".log": load_log,
    ".txt": load_text,
}


async def load_document(path: str | Path) -> RawDocument:
    """Pick a loader by file extension and run it off the event loop."""
    path = Path(path)
    loader = _LOADERS.get(path.suffix.lower())
    if loader is None:
        raise DocumentError(f"{path.name}: unsupported file type '{path.suffix}'")
    return await asyncio.to_thread(loader, path)
