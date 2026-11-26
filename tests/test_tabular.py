import csv
import json
from types import SimpleNamespace

import pytest

from llmkit.pipeline import process_document
from llmkit.tabular import TableParseError, parse_invoice_table

HEADER = [
    "invoice_number",
    "vendor_name",
    "invoice_date",
    "due_date",
    "currency",
    "status",
    "tax",
    "description",
    "quantity",
    "unit_price",
    "amount",
]

GOOD = {
    "invoice_number": "A-1",
    "vendor_name": "Acme",
    "invoice_date": "2026-03-14",
    "due_date": None,
    "currency": "USD",
    "status": "paid",
    "line_items": [{"description": "Widget", "quantity": 2, "unit_price": 5.0, "amount": 10.0}],
    "subtotal": 10.0,
    "tax": 1.0,
    "total": 11.0,
}


def row(desc="Widget", qty="2", unit="5.00", amount="10.00", **overrides):
    base = {
        "invoice_number": "A-1",
        "vendor_name": "Acme",
        "invoice_date": "2026-03-14",
        "due_date": "2026-04-13",
        "currency": "USD",
        "status": "paid",
        "tax": "1.00",
        "description": desc,
        "quantity": qty,
        "unit_price": unit,
        "amount": amount,
    }
    base.update(overrides)
    return [base[column] for column in HEADER]


def table(*rows, header=HEADER):
    return [header, *rows]


def test_parses_clean_table():
    invoice = parse_invoice_table(table(row(), row("Gadget", "1", "5.00", "5.00")))
    assert invoice.invoice_number == "A-1"
    assert invoice.currency.value == "USD"
    assert len(invoice.line_items) == 2
    assert invoice.subtotal == 15.0
    assert invoice.tax == 1.0
    assert invoice.total == 16.0


def test_thousands_separator_is_accepted():
    invoice = parse_invoice_table(table(row("Server", "1", "1,200.00", "1,200.00")))
    assert invoice.subtotal == 1200.0


def test_empty_due_date_becomes_none():
    invoice = parse_invoice_table(table(row(due_date="")))
    assert invoice.due_date is None


def test_header_names_are_normalized():
    header = [name.upper().replace("_", " ") for name in HEADER]
    assert parse_invoice_table(table(row(), header=header)).invoice_number == "A-1"


def test_missing_column_is_rejected():
    drop = HEADER.index("tax")
    rows = [[cell for i, cell in enumerate(r) if i != drop] for r in table(row())]
    with pytest.raises(TableParseError, match="missing columns: tax"):
        parse_invoice_table(rows)


def test_inconsistent_header_field_is_rejected():
    with pytest.raises(TableParseError, match="inconsistent values in column 'vendor_name'"):
        parse_invoice_table(table(row(), row(vendor_name="Globex")))


def test_non_numeric_value_is_rejected():
    with pytest.raises(TableParseError, match="not a number"):
        parse_invoice_table(table(row(qty="two")))


def test_nan_is_rejected():
    with pytest.raises(TableParseError, match="not a number"):
        parse_invoice_table(table(row(amount="nan")))


def test_amount_must_match_quantity_times_price():
    with pytest.raises(TableParseError, match="row 1"):
        parse_invoice_table(table(row(amount="11.00")))


def test_due_date_before_invoice_date_is_rejected():
    with pytest.raises(TableParseError, match="validation failed"):
        parse_invoice_table(table(row(due_date="2026-01-01")))


def test_unknown_currency_is_rejected():
    with pytest.raises(TableParseError, match="validation failed"):
        parse_invoice_table(table(row(currency="XYZ")))


def test_no_data_rows_is_rejected():
    with pytest.raises(TableParseError, match="no data rows"):
        parse_invoice_table(table())


def test_ragged_row_is_rejected():
    with pytest.raises(TableParseError, match="cells"):
        parse_invoice_table([HEADER, ["only", "two"]])


class FakeClient:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = 0

    async def generate(self, messages, *, max_tokens=1024, temperature=0.0, json_schema=None):
        self.calls += 1
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return SimpleNamespace(text=response, total_s=0.1, cost_usd=0.5)


async def no_sleep(delay):
    return None


def write_csv(tmp_path, rows, name="inv.csv"):
    path = tmp_path / name
    with path.open("w", newline="") as handle:
        csv.writer(handle).writerows(rows)
    return path


async def test_pipeline_uses_deterministic_path_without_llm(tmp_path):
    path = write_csv(tmp_path, table(row(), row("Gadget", "1", "5.00", "5.00")))
    primary = FakeClient()
    outcome = await process_document(path, primary, FakeClient(), sleep=no_sleep)
    assert outcome.status == "ok"
    assert outcome.method == "table"
    assert outcome.provider == "deterministic"
    assert outcome.invoice.total == 16.0
    assert outcome.cost_usd == 0.0
    assert outcome.attempts == []
    assert primary.calls == 0


async def test_pipeline_falls_back_to_llm_for_messy_table(tmp_path):
    path = write_csv(tmp_path, table(row(), row(vendor_name="Globex")))
    primary = FakeClient(json.dumps(GOOD))
    outcome = await process_document(path, primary, FakeClient(), sleep=no_sleep)
    assert outcome.status == "ok"
    assert outcome.method == "llm"
    assert "inconsistent" in outcome.note
    assert primary.calls == 1
