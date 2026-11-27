"""Tests for the validated, parameterized SQLite invoice tool."""

import json
import sqlite3
from pathlib import Path

import pytest

from llmkit.sql_tool import (
    InvoiceDatabase,
    InvoiceQueryArgs,
    make_invoice_query_tool,
)
from llmkit.tools import ToolArgumentError, ToolRegistry

DATASET = Path(__file__).parents[1] / "samples/day10/invoices.csv"


@pytest.fixture
def database():
    db = InvoiceDatabase(DATASET)
    yield db
    db.close()


def test_loads_all_five_invoices(database):
    result = json.loads(database.query(InvoiceQueryArgs(operation="status_summary", status="paid")))
    assert result == {
        "status": "paid",
        "invoice_count": 2,
        "totals_by_currency": [{"currency": "USD", "invoice_count": 2, "total": "214.50"}],
    }


def test_invoice_query_returns_header_and_line_items(database):
    result = json.loads(
        database.query(
            InvoiceQueryArgs(
                operation="invoice",
                invoice_number="INV-2026-201",
            )
        )
    )

    assert result["found"] is True
    assert result["vendor_name"] == "Apex Office"
    assert result["subtotal"] == "120.00"
    assert result["tax"] == "12.00"
    assert result["total"] == "132.00"
    assert len(result["line_items"]) == 2


def test_missing_invoice_returns_not_found(database):
    result = json.loads(
        database.query(InvoiceQueryArgs(operation="invoice", invoice_number="UNKNOWN"))
    )
    assert result == {"found": False, "invoice_number": "UNKNOWN"}


def test_status_summary_does_not_double_count_line_items(database):
    result = json.loads(
        database.query(InvoiceQueryArgs(operation="status_summary", status="unpaid"))
    )
    assert result["invoice_count"] == 2
    assert result["totals_by_currency"] == [
        {"currency": "USD", "invoice_count": 2, "total": "85.80"}
    ]


def test_vendor_summary_groups_invoice_totals_by_status(database):
    result = json.loads(
        database.query(InvoiceQueryArgs(operation="vendor_summary", vendor_name="Apex Office"))
    )
    assert result == {
        "vendor_name": "Apex Office",
        "statuses": [
            {"status": "paid", "invoice_count": 1, "total": "132.00"},
            {"status": "unpaid", "invoice_count": 1, "total": "33.00"},
        ],
    }


def test_vendor_summary_for_unknown_vendor_is_empty(database):
    result = json.loads(
        database.query(InvoiceQueryArgs(operation="vendor_summary", vendor_name="Unknown"))
    )
    assert result["statuses"] == []


def test_sql_injection_string_is_treated_as_a_value(database):
    result = json.loads(
        database.query(
            InvoiceQueryArgs(
                operation="invoice",
                invoice_number="' OR 1=1 --",
            )
        )
    )
    assert result["found"] is False

    paid = json.loads(database.query(InvoiceQueryArgs(operation="status_summary", status="paid")))
    assert paid["invoice_count"] == 2


@pytest.mark.parametrize(
    "payload",
    [
        {"operation": "invoice"},
        {"operation": "status_summary"},
        {"operation": "vendor_summary"},
        {
            "operation": "invoice",
            "invoice_number": "INV-1",
            "status": "paid",
        },
        {"operation": "raw_sql", "query": "DROP TABLE invoices"},
        {"operation": "status_summary", "status": "all"},
    ],
)
def test_invalid_operation_arguments_are_rejected(payload):
    with pytest.raises(ValueError):
        InvoiceQueryArgs.model_validate(payload)


def test_tool_integrates_with_day9_registry(database):
    tool = make_invoice_query_tool(database)
    registry = ToolRegistry([tool])

    args = registry.parse_arguments(
        "invoice_query",
        {"operation": "status_summary", "status": "overdue"},
    )
    assert args.status == "overdue"


def test_tool_description_prioritizes_aggregate_operation(database):
    description = make_invoice_query_tool(database).description

    assert "ALWAYS use operation 'status_summary'" in description
    assert "combined total" in description
    assert "Do not use 'invoice' just because an invoice number appears" in description
    assert "it is not a structured-dataset lookup" in description


def test_registry_rejects_extra_tool_arguments(database):
    registry = ToolRegistry([make_invoice_query_tool(database)])

    with pytest.raises(ToolArgumentError):
        registry.parse_arguments(
            "invoice_query",
            {
                "operation": "status_summary",
                "status": "paid",
                "sql": "DROP TABLE invoices",
            },
        )


def test_status_summary_keeps_different_currencies_separate(tmp_path):
    path = tmp_path / "mixed_currency.csv"
    path.write_text(
        "invoice_number,vendor_name,invoice_date,due_date,currency,status,tax,"
        "description,quantity,unit_price,amount\n"
        "USD-1,Vendor A,2026-03-01,2026-03-31,USD,unpaid,10.00,"
        "Item A,1,100.00,100.00\n"
        "EUR-1,Vendor B,2026-03-01,2026-03-31,EUR,unpaid,5.00,"
        "Item B,1,50.00,50.00\n",
        encoding="utf-8",
    )

    with InvoiceDatabase(path) as database:
        result = json.loads(
            database.query(InvoiceQueryArgs(operation="status_summary", status="unpaid"))
        )

    assert result == {
        "status": "unpaid",
        "invoice_count": 2,
        "totals_by_currency": [
            {"currency": "EUR", "invoice_count": 1, "total": "55.00"},
            {"currency": "USD", "invoice_count": 1, "total": "110.00"},
        ],
    }


def test_missing_dataset_is_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        InvoiceDatabase(tmp_path / "missing.csv")


def test_connection_is_closed_when_initialization_fails(tmp_path, monkeypatch):
    original_connect = sqlite3.connect
    connections = []

    def tracking_connect(*args, **kwargs):
        connection = original_connect(*args, **kwargs)
        connections.append(connection)
        return connection

    monkeypatch.setattr("llmkit.sql_tool.sqlite3.connect", tracking_connect)

    with pytest.raises(FileNotFoundError):
        InvoiceDatabase(tmp_path / "missing.csv")

    assert len(connections) == 1
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        connections[0].execute("SELECT 1")


def test_empty_dataset_is_rejected(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text(
        "invoice_number,vendor_name,invoice_date,due_date,currency,status,tax,"
        "description,quantity,unit_price,amount\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="no invoice records"):
        InvoiceDatabase(path)
