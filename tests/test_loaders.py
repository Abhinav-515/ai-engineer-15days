import pytest
from fpdf import FPDF

from llmkit.loaders import (
    DocumentError,
    load_document,
    load_log,
    load_pdf,
    load_table,
)
from llmkit.preprocess import TRUNCATION_MARKER, clean_text, truncate_text


def make_pdf(path, lines):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    for line in lines:
        pdf.cell(text=line, new_x="LMARGIN", new_y="NEXT")
    pdf.output(str(path))


def test_load_pdf_extracts_text(tmp_path):
    path = tmp_path / "inv.pdf"
    make_pdf(path, ["Invoice A-1", "Total: USD 11.00"])
    doc = load_pdf(path)
    assert doc.kind == "pdf"
    assert doc.page_count == 1
    assert "Invoice A-1" in doc.text
    assert "Total: USD 11.00" in doc.text


def test_load_pdf_extracts_tables(tmp_path):
    path = tmp_path / "table.pdf"
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    with pdf.table() as table:
        for row in [["Item", "Qty"], ["Widget", "2"]]:
            table_row = table.row()
            for cell in row:
                table_row.cell(cell)
    pdf.output(str(path))
    doc = load_pdf(path)
    assert doc.tables
    assert doc.tables[0][0] == ["Item", "Qty"]
    assert doc.tables[0][1] == ["Widget", "2"]


def test_blank_pdf_is_rejected(tmp_path):
    path = tmp_path / "blank.pdf"
    pdf = FPDF()
    pdf.add_page()
    pdf.output(str(path))
    with pytest.raises(DocumentError, match="no extractable text"):
        load_pdf(path)


def test_corrupt_pdf_is_rejected(tmp_path):
    path = tmp_path / "bad.pdf"
    path.write_bytes(b"this is not a pdf")
    with pytest.raises(DocumentError, match="cannot read PDF"):
        load_pdf(path)


def test_missing_file_is_rejected(tmp_path):
    with pytest.raises(DocumentError):
        load_pdf(tmp_path / "nope.pdf")


def test_load_table_csv(tmp_path):
    path = tmp_path / "items.csv"
    path.write_text("description, qty ,amount\nWidget, 2 ,10.00\nGadget,1,5.00\n")
    doc = load_table(path)
    assert doc.kind == "table"
    assert doc.tables[0][0] == ["description", "qty", "amount"]
    assert doc.tables[0][1] == ["Widget", "2", "10.00"]
    assert "Gadget,1,5.00" in doc.text


def test_load_table_tsv(tmp_path):
    path = tmp_path / "items.tsv"
    path.write_text("description\tqty\nWidget\t2\n")
    assert load_table(path).tables[0][1] == ["Widget", "2"]


def test_empty_table_is_rejected(tmp_path):
    empty = tmp_path / "empty.csv"
    empty.write_text("")
    header_only = tmp_path / "header.csv"
    header_only.write_text("a,b\n")
    with pytest.raises(DocumentError):
        load_table(empty)
    with pytest.raises(DocumentError, match="no data rows"):
        load_table(header_only)


def test_load_log(tmp_path):
    path = tmp_path / "app.log"
    path.write_text("2026-03-14 10:00:01 ERROR payment failed\n")
    doc = load_log(path)
    assert doc.kind == "log"
    assert "payment failed" in doc.text


def test_empty_log_is_rejected(tmp_path):
    path = tmp_path / "app.log"
    path.write_text("  \n\n")
    with pytest.raises(DocumentError, match="empty"):
        load_log(path)


def test_log_with_invalid_utf8_still_loads(tmp_path):
    path = tmp_path / "app.log"
    path.write_bytes(b"ok line\n\xff\xfe broken bytes\n")
    assert "ok line" in load_log(path).text


async def test_load_document_dispatches_by_extension(tmp_path):
    path = tmp_path / "app.log"
    path.write_text("hello\n")
    assert (await load_document(path)).kind == "log"


async def test_load_document_rejects_unknown_extension(tmp_path):
    path = tmp_path / "data.xyz"
    path.write_text("x")
    with pytest.raises(DocumentError, match="unsupported"):
        await load_document(path)


def test_clean_text_normalizes_whitespace():
    assert clean_text("a  \r\n\n\n\nb\x00 ") == "a\n\nb"


def test_clean_text_keeps_unicode():
    assert clean_text("Müller Büromaterial EUR 5") == "Müller Büromaterial EUR 5"


def test_truncate_short_text_is_unchanged():
    assert truncate_text("short", max_chars=100) == ("short", False)


def test_truncate_keeps_head_and_tail():
    text = "H" * 60 + "M" * 900 + "T" * 30
    out, truncated = truncate_text(text, max_chars=90)
    assert truncated
    assert out.startswith("H" * 60)
    assert out.endswith("T" * 30)
    assert TRUNCATION_MARKER in out
    assert "M" * 100 not in out


def test_truncate_rejects_bad_limit():
    with pytest.raises(ValueError):
        truncate_text("x", max_chars=0)


async def test_load_text(tmp_path):
    path = tmp_path / "invoice.txt"
    path.write_text("Invoice total: USD 242.00")
    doc = await load_document(path)
    assert doc.kind == "text"
    assert "USD 242.00" in doc.text


async def test_empty_text_is_rejected(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("  " + chr(10))
    with pytest.raises(DocumentError, match="text file is empty"):
        await load_document(path)
