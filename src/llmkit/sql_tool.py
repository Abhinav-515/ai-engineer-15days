"""Validated invoice data loaded into an in-memory SQLite database."""

from __future__ import annotations

import csv
import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from llmkit.schemas import Invoice
from llmkit.tabular import parse_invoice_table
from llmkit.tools import Tool


class InvoiceQueryArgs(BaseModel):
    """Arguments for the fixed invoice-query operations."""

    model_config = ConfigDict(extra="forbid")

    operation: Literal["invoice", "status_summary", "vendor_summary"]
    invoice_number: str | None = Field(default=None, max_length=80)
    status: Literal["paid", "unpaid", "overdue"] | None = None
    vendor_name: str | None = Field(default=None, max_length=120)

    @model_validator(mode="after")
    def validate_operation_arguments(self) -> Self:
        required = {
            "invoice": "invoice_number",
            "status_summary": "status",
            "vendor_summary": "vendor_name",
        }[self.operation]
        supplied = {
            "invoice_number": self.invoice_number,
            "status": self.status,
            "vendor_name": self.vendor_name,
        }
        if not supplied[required] or not str(supplied[required]).strip():
            raise ValueError(f"{required} is required for {self.operation}")
        if any(value is not None for key, value in supplied.items() if key != required):
            raise ValueError(f"only {required} is allowed for {self.operation}")
        return self


class InvoiceDatabase:
    """Loads validated CSV invoices into a private in-memory SQLite database."""

    def __init__(self, csv_path: str | Path) -> None:
        self._connection = sqlite3.connect(":memory:")
        try:
            self._connection.row_factory = sqlite3.Row
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._create_schema()
            self._load_csv(Path(csv_path))
        except Exception:
            self._connection.close()
            raise

    def _create_schema(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE invoices (
                invoice_number TEXT PRIMARY KEY,
                vendor_name TEXT NOT NULL,
                invoice_date TEXT NOT NULL,
                due_date TEXT,
                currency TEXT NOT NULL,
                status TEXT NOT NULL,
                subtotal_cents INTEGER NOT NULL,
                tax_cents INTEGER NOT NULL,
                total_cents INTEGER NOT NULL
            );

            CREATE TABLE line_items (
                id INTEGER PRIMARY KEY,
                invoice_number TEXT NOT NULL REFERENCES invoices(invoice_number),
                description TEXT NOT NULL,
                quantity REAL NOT NULL,
                unit_price_cents INTEGER NOT NULL,
                amount_cents INTEGER NOT NULL
            );
            """
        )

    def _load_csv(self, path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"invoice dataset not found: {path}")

        grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            if not reader.fieldnames:
                raise ValueError("invoice dataset is missing its header")
            for row in reader:
                invoice_number = (row.get("invoice_number") or "").strip()
                if not invoice_number:
                    raise ValueError("invoice dataset contains a row without an invoice number")
                grouped[invoice_number].append(row)

        if not grouped:
            raise ValueError("invoice dataset contains no invoice records")

        for rows in grouped.values():
            headers = list(rows[0])
            invoice = parse_invoice_table(
                [headers, *[[row.get(column, "") or "" for column in headers] for row in rows]]
            )
            self._insert_invoice(invoice)

        self._connection.commit()

    def _insert_invoice(self, invoice: Invoice) -> None:
        self._connection.execute(
            """
            INSERT INTO invoices VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                invoice.invoice_number,
                invoice.vendor_name,
                invoice.invoice_date.isoformat(),
                invoice.due_date.isoformat() if invoice.due_date else None,
                invoice.currency.value,
                invoice.status.value,
                round(invoice.subtotal * 100),
                round(invoice.tax * 100),
                round(invoice.total * 100),
            ),
        )
        self._connection.executemany(
            """
            INSERT INTO line_items (
                invoice_number, description, quantity, unit_price_cents, amount_cents
            ) VALUES (?, ?, ?, ?, ?)
            """,
            [
                (
                    invoice.invoice_number,
                    item.description,
                    item.quantity,
                    round(item.unit_price * 100),
                    round(item.amount * 100),
                )
                for item in invoice.line_items
            ],
        )

    @staticmethod
    def _money(cents: int) -> str:
        return f"{cents / 100:.2f}"

    def query(self, args: InvoiceQueryArgs) -> str:
        """Execute one of the three predefined, parameterized query operations."""
        if args.operation == "invoice":
            row = self._connection.execute(
                "SELECT * FROM invoices WHERE invoice_number = ?",
                (args.invoice_number,),
            ).fetchone()
            if row is None:
                result = {"found": False, "invoice_number": args.invoice_number}
            else:
                items = self._connection.execute(
                    """
                    SELECT description, quantity, unit_price_cents, amount_cents
                    FROM line_items WHERE invoice_number = ? ORDER BY id
                    """,
                    (args.invoice_number,),
                ).fetchall()
                result = {
                    "found": True,
                    "invoice_number": row["invoice_number"],
                    "vendor_name": row["vendor_name"],
                    "invoice_date": row["invoice_date"],
                    "due_date": row["due_date"],
                    "currency": row["currency"],
                    "status": row["status"],
                    "subtotal": self._money(row["subtotal_cents"]),
                    "tax": self._money(row["tax_cents"]),
                    "total": self._money(row["total_cents"]),
                    "line_items": [
                        {
                            "description": item["description"],
                            "quantity": item["quantity"],
                            "unit_price": self._money(item["unit_price_cents"]),
                            "amount": self._money(item["amount_cents"]),
                        }
                        for item in items
                    ],
                }

        elif args.operation == "status_summary":
            rows = self._connection.execute(
                """
                SELECT currency, COUNT(*) AS invoice_count,
                       COALESCE(SUM(total_cents), 0) AS total_cents
                FROM invoices WHERE status = ?
                GROUP BY currency ORDER BY currency
                """,
                (args.status,),
            ).fetchall()
            result = {
                "status": args.status,
                "invoice_count": sum(row["invoice_count"] for row in rows),
                "totals_by_currency": [
                    {
                        "currency": row["currency"],
                        "invoice_count": row["invoice_count"],
                        "total": self._money(row["total_cents"]),
                    }
                    for row in rows
                ],
            }

        else:
            rows = self._connection.execute(
                """
                SELECT status, COUNT(*) AS invoice_count,
                       COALESCE(SUM(total_cents), 0) AS total_cents
                FROM invoices WHERE vendor_name = ?
                GROUP BY status ORDER BY status
                """,
                (args.vendor_name,),
            ).fetchall()
            result = {
                "vendor_name": args.vendor_name,
                "statuses": [
                    {
                        "status": row["status"],
                        "invoice_count": row["invoice_count"],
                        "total": self._money(row["total_cents"]),
                    }
                    for row in rows
                ],
            }

        return json.dumps(result, indent=2)

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> InvoiceDatabase:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def make_invoice_query_tool(database: InvoiceDatabase) -> Tool:
    """Create a Day 9-compatible tool backed by the validated invoice database."""

    async def handle(args: InvoiceQueryArgs) -> str:
        return database.query(args)

    return Tool(
        name="invoice_query",
        description=(
            "Query only the structured invoice dataset. OPERATION SELECTION: "
            "For requests asking for a combined total, count, or aggregate by "
            "payment status (paid, unpaid, or overdue), ALWAYS use operation "
            "'status_summary' with the requested status. For a lookup of one "
            "specific invoice that belongs to this dataset, use operation "
            "'invoice' with invoice_number. Do not use 'invoice' just because "
            "an invoice number appears in a mixed-source question; if the user "
            "says that invoice is in documents, it is not a structured-dataset "
            "lookup. For a vendor's totals grouped by status, use "
            "'vendor_summary' with vendor_name. Never write SQL."
        ),
        args_model=InvoiceQueryArgs,
        handler=handle,
    )
